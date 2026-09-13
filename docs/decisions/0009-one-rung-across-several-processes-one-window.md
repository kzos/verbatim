# DR-0009 — One rung across several processes, and one window

**Date:** 2026-09-13
**Status:** accepted

## What happened

Two measurements taken on this box, both already recorded, decided this between them.

The target: NVIDIA's own file-driven script, at the release we run, with our checkpoint and corpus on
an RTX A6000, reaches **139.71 times real time**. The MVP bar is eight tenths of that, so about
**112 concurrent streams sustained**.

The instrument: nine 180-second null-floor windows at three concurrencies on 2026-09-11 measured the
generator's own pacing slip at the 99th percentile at **2.5 ms at 16 streams**, **3.9 to 5.4 ms at
32** with one window in three over the frozen 5.0 ms tolerance, and **33 to 38 ms at 128**. Nothing
failed and host pressure never left the floor. One asyncio process simply cannot hold a 20 ms send
schedule for that many streams.

So the instrument gives out at roughly a third of what the bar requires. A ladder run on one process
would go invalid for `pacing_slip` before any server limit appeared, and **a server that cleared the
bar could not be shown to have cleared it**. That supersedes the earlier decision to defer this work
until a ladder reached the pacing gate: the deferral rested on the necessity being a prediction, and
it is now a measurement of the target rather than a forecast of the instrument.

## The decision

`verbatim-bench ladder --processes N` spreads one rung's streams across `N` processes, each running
the same event loop out of `pace.py`, and combines what they measured into one rung. The processes
are the easy half. These four are the decision.

### The window opens at one instant, named once

Every process opens its window at the same instant on the same clock and closes it `window_s` later.
The coordinator names one instant, broadcasts it, and each process records **that instant verbatim**
rather than the clock it read after waiting for it, so the two timestamps a rung is combined from are
equal to the bit. `combine_shards` checks that and raises rather than pooling anything else: a
percentile taken over windows that did not overlap is a number no load ever produced.

The instant is on `CLOCK_MONOTONIC`, which Linux keeps per boot and not per process — measured here,
a child's reading falling between two of the parent's — and `require_shared_monotonic` checks the
clock's implementation at run time rather than assuming it.

**Rejected:** each process opening its own window and the rung taking the intersection. A rung would
then measure `window_s` minus the spread of four independent convergence decisions, and would report
`window_s` as the length it ran.

### The warm-up is the rung's, so it is decided in the coordinator

Convergence is a property of the rung. Each process reports the **raw latencies** of each reading
interval and the coordinator takes the rung's reading as the 95th percentile of the pooled sample,
two consecutive readings agreeing within the frozen tolerance opening the window. Raw, because a
percentile of percentiles is not a percentile. The reading grid runs from the instant the rung's
**last** process came live, which the coordinator hands back to every process when it reports its own
slots live, so the intervals the coordinator pools are the same intervals.

**Rejected:** per-process convergence with the rung opening at the last process to settle. Four
processes each declaring themselves settled on their own quarter, at four different moments, would
open the window at a moment no pooled reading was ever taken at, and the statistic the warm-up tested
would not be the statistic the window reports.

### Slots are named by their index in the rung, and dealt round-robin

Everything a slot seeds — its place in the ramp, which utterance it replays, its frame jitter, its
session id — is keyed off the slot's index in the **rung**, never off its position in one process's
own share. `plan_start_offsets` plans the whole rung in rung slot order and each process takes its own
entries out of that plan, so the draws are the same draws in the same order and nothing is re-seeded
per process. The same rung at the same seed therefore puts the same session in the same slot whether
one process drives 128 slots or eight drive sixteen each, which is the only thing that keeps the
ladder repeatable across a change in `--processes`.

Slots are dealt round-robin rather than in contiguous blocks. A process holding the first sixteen
slots of a sixty-second ramp would go live a minute before one holding the last sixteen, and the
reading intervals the coordinator pools would cover moments a minute apart.

**Rejected:** contiguous blocks, which are the obvious split and the one that quietly destroys the
reading grid.

### The host record is the box's, taken once

The box is one box. The record is sampled in the coordinator, over the shared window, and not once per
process: `N` records of the same box over the same window are not `N` observations, and picking one of
them attributes the box to whichever process happened to be asked.

The client's own CPU is the exception, and it is a real one. It now lives in the children, so
`HostSampler` takes every load process's pid and sums their CPU, and `client_processes` counts them.
Left alone, the client-CPU validity gate would have read a parent that forked and waited: a threshold
that could not fail, in a file whose whole subject is thresholds that can.

## What is not decided here

`--processes` defaults to **1**, which is the same event loop, the same local gate and the same rung
as before this existed. Nothing chooses `N` automatically. Which `N` a canonical row is measured at is
a measurement to be made on the box, from the pacing slip at the concurrency the rung runs, and it
belongs to the run that makes it rather than to this record.
