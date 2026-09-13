# DR-0012 — The tick phase offset is irreducible under a shared tick grid, and the only lever is sub-tick staggering

**Date:** 2026-09-13
**Status:** accepted

## What was proposed and why it does not work

An audit of the remaining two-day measurement window put this first, as the largest latency change
available and one needing no GPU:

> Align the session's chunk boundary to the tick grid at open — hold back under one period once —
> instead of paying `(tick phase − connect) mod period` on every chunk for the session's life.

The defect it names is real and is measured. `probes/tick_phase_sweep.py` establishes

    latency = chunk_ms + ((tick phase − connect time) mod tick period)

drawn once at connect and held for the session's life, because the client's chunk cadence and the
server's tick period are both 160 ms and stay in lockstep. Re-measured on an A6000 after the
publish-before-sleep fix (`reviews/tick_phase_sweep_after_publish_fix.json`), stepping the connect
delay across one period in 10 ms steps:

| connect delay | median latency | | connect delay | median latency |
|---|---|---|---|---|
| 0 ms | 222.3 ms | | 80 ms | 143.1 ms |
| 20 ms | 189.2 ms | | 100 ms | 124.4 ms |
| 40 ms | 183.2 ms | | 110 ms | 112.9 ms |
| 60 ms | 163.0 ms | | 120 ms | 262.9 ms (the wrap) |

Slope minus one, one wrap, **range 112.9 to 262.9 ms — a spread of 150.0 ms against a 310 ms
budget.** Roughly half the entire latency budget is decided by what moment a client happened to
connect, and an unlucky session pays it for its whole life with no recovery.

**The proposed fix cannot work, and the reason is worth writing down so it is not proposed again.**

A session's audio arrives at the client's cadence into a ring. Each tick the loop pops exactly one
chunk. In steady state the ring's excess over one chunk is constant — call it `e` samples — and the
added latency *is* `e / 16000` seconds: the newest sample in the frame is that old when the step
runs. Reducing the latency means reducing `e`, and `e` can only be reduced by consuming audio faster
than it arrives for one tick. There are exactly three ways to do that and all three are refused:

- **Drop `e` samples at the start.** Discards a caller's audio. `docs/SCOPE.md` forbids it and
  `RingBuffer` is built so that a full ring is back-pressure, never dropped audio.
- **Emit a larger first frame.** Breaks the fixed shape, which is the property this server exists
  for.
- **Delay the first cut.** Leaves `e` unchanged: the ring refills at exactly the rate it drains, so
  waiting a tick moves the whole schedule and changes nothing.

Holding the session back at admission adds latency; it does not remove any. **You cannot process a
chunk before the client has finished sending it, and the wait to the next tick boundary is exactly
that gap.** The offset is not a scheduler defect. It is the cost of a shared tick grid, and every
batched server with a fixed tick pays it.

## The decision

Record the phase offset as a structural property of the design, quantified, in the published
latency model:

    added latency = U[0, tick period), drawn once per session at connect

Mean 80 ms, worst case 160 ms at the 160 ms chunk mode. No code changes. `S = 38` on the B300 and
`S = 22` on the A6000 remain what they are; they are not corrected upward, and no criterion moves.

## The one real lever, and why it is not taken today

**Stagger sessions across sub-ticks.** Run the tick grid `k` times per chunk period and assign each
session to one sub-tick. A session is still stepped exactly once per chunk period, so the total
number of stepped rows per second is unchanged, but the longest wait becomes `period / k` instead of
`period`. At `k = 2` the spread above would fall from 150 ms to about 75 ms.

It is not free, and the cost is measured. The batch at each sub-tick holds `N / k` sessions rather
than `N`, and smaller fixed-shape batches are materially less efficient on this hardware — from
`rows/exploratory/nemo-ceiling-b300-bf16-2026-09-13.json`, upstream's own file-driven ceiling is
**149.48 RTFx at batch 128 and 92.46 at batch 32**, so quartering the batch costs about 38 % of the
ceiling. Halving the latency spread for something in that neighbourhood of throughput is a real
trade with numbers on both sides, and it is a design decision for the author rather than one to be
taken inside a measurement window on a borrowed machine.

Recorded in `docs/BACKLOG.md` with the numbers, so whoever takes it has both halves.

## Rejected alternatives

**Jitter the tick grid so each session's offset re-draws over time.** Converts a fixed per-session
penalty into one that sweeps the full range, which improves the across-session p95 at any instant
and makes every individual session's latency *less* predictable. Predictable latency is worth more
than a better percentile to the buyer this server is for, and moving a number the criterion reads
without improving what a caller experiences is the kind of change `docs/decisions/0001` exists to
prevent.

**Re-define the latency criterion to exclude the phase term.** The criterion measures what a client
experiences, and a client experiences this. DR-0001 froze the methodology precisely so that a number
that turns out to be hard is not redefined afterwards.

**Shorten the chunk mode instead.** An 80 ms chunk halves the offset for the same reason sub-ticking
does, but it also changes the model's own lookahead and the accuracy it delivers, so it is a
different server rather than a fix to this one. The 560 ms mode moves the opposite way and has never
been run anywhere; that measurement is in the backlog on its own merits.
