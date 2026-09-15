# DR-0017: the bucket is the capacity knob, and 128 is already at its optimum

**Status:** accepted, 2026-09-15.

## Context

DR-0016 fixed the ladder and the B300's fixed arm read **S = 124**, ending on
`integrity:refused` — sessions turned away because admission caps at the configured
bucket, which was 128. Every published statement of that number therefore carried a
caveat: *at least 124, bounded by the bucket rather than by the GPU, so the true ceiling
is higher by an unknown amount.*

That caveat was the obvious reading and it was wrong. The obvious next run — raise the
bucket, find the real ceiling — was run, and it went the other way.

## What was measured

Same B300, same checkpoint, same corpus, same 160 ms chunk, bfloat16, eager, three seeds,
fixed-shape padding. Only the bucket differs. Highest passing rung per seed:

| bucket | seed 14 | seed 15 | seed 16 | reported S | ending criterion |
|---|---|---|---|---|---|
| 128 | 124 | 126 | 124 | **124** | `integrity:refused` |
| 256 | 46 | 46 | 19 | **19** | `latency` |

`rows/exploratory/ladder-b300-fixed-dr0016-2026-09-15.json` and
`rows/exploratory/ladder-b300-fixed-bucket256-2026-09-15.json`.

**Doubling the bucket did not raise the ceiling. It more than halved it**, and the
criterion changed from "admission refused the stream" to "the latency budget was blown".
At bucket 256 the p95 is 289 ms at 46 streams and 319.6 ms at 47, against a 310 ms
budget; at bucket 128 it was 256–285 ms at *124*.

## Why, and it is the design working as described

Fixed-shape padding pads the steady batch to the bucket **at every occupancy**. That is
the whole mechanism behind batch invariance: the encoder sees one shape whether one
session is connected or a hundred, so a transcript cannot depend on who else is on the
GPU. The cost is that the bucket is paid in full on every tick regardless of load. A
bucket of 256 does 256 rows of work to serve 20 streams, and the tick budget goes to
padding instead of to sessions.

So capacity is not monotone in the bucket. Admission caps live sessions at the bucket, so
capacity can never exceed it; and the padded step's cost rises with it, so beyond some
point capacity falls. **The maximum is the fixed point where capacity equals the
bucket.** At 128 the server sustains 124 — within three per cent of its own bucket. At
256 it sustains 46, nowhere near.

**128 is therefore at or within a few streams of the optimum, and 124 is a ceiling
rather than a lower bound.** The caveat that the true number is "higher by an unknown
amount" is withdrawn.

## What was rejected

**Raising the bucket to find a higher ceiling.** Measured, and it costs 78 streams.

**Quoting 19 as the bucket-256 capacity without its spread.** Two seeds reached 46 and
the third 19, and `S` is the minimum across seeds. The honest reading of that arm is
"46, 46 and 19"; it is reported that way here because a single number would hide that the
spread reopened at this bucket. Why seed 16 is so much slower is not established — it ran
last, about two and a half hours in, which suggests drift, but no rung recorded a throttle
event and every rung was valid, so nothing in the record explains it. It is not claimed.

**Tuning between 128 and 256.** A bucket of 160 or 192 might sustain slightly more than
124, and nothing here rules it out. It is not worth the GPU time: the fixed point cannot
move far, since capacity is bounded above by the bucket and the 256 arm shows the cost
curve rising steeply. The remaining headroom is small and the measurement is not.

## What this changes

The MVP ratio stands on firmer ground and stops improving. Against NVIDIA's own graphed
file-driven ceiling on this card, 143.75 times real time, the bar is 115 sustained streams
and the server sustains 124 — **0.863, and that is the number, not a floor.** It will not
improve by tuning the bucket, and any future gain has to come from the step itself.

It also makes the bucket-128 arm the most reproducible capacity measurement this project
has: **124 / 126 / 124 across three seeds.** Every earlier ladder spread by a factor of
two or three. That spread was DR-0016's artefact, and with it gone the underlying
measurement turns out to be tight.

## The ragged arm is not a capacity measurement and must not be quoted as one

The ragged control was re-run at bucket 128 under the same procedure. It reported
**S = 77 on 2026-09-14 and S = 0 on 2026-09-15**, with per-seed highest passing rungs of
67, 68 and 101 in the repeat, and ten of thirty rungs still ending as `unstable` after the
DR-0016 repeat. That is consistent with what the arm is: a server whose batch shape
follows occupancy, whose latency therefore varies with its neighbours, and whose warm-up
consequently struggles to settle. It does its job — showing the invariance gate can fail —
and it has no business in a capacity claim.
