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

## What the per-row cost actually is, and it is not the encoder

Derived from `rows/exploratory/nemo-ceiling-b300-bf16-2026-09-13.json` rather than measured
on the server, so treat it as scaling and not as a profile. One row consumes one 160 ms
chunk per step, so a step's wall clock is `B x 0.16 / RTFx`:

| batch | eager | graphed |
|---|---|---|
| 32 | 55.4 ms | 42.9 ms |
| 128 | 137.0 ms | 142.5 ms |

Fitting a line through the two eager points: **`cost(B) = 28.2 ms + 0.850 ms per row`**. That
single number predicts everything this record is about. 99 rows fit the 70 % budget, 155 fit
the whole tick period, and the server's measured fixed point sits at 124, between them.

**The marginal cost is not GPU compute.** Four times the rows cost 2.47 times the time, so
the step is not launch-bound — and the encoder's own arithmetic is nowhere near 0.85 ms a
row: with `att_context [70, 1]` a chunk is two post-subsampling frames, and re-projecting 72
cached frames across 17 layers is on the order of a GFLOP per row, sub-millisecond for the
whole batch on this die. What scales with `B` is the per-row host-side work NeMo's
`BasePipeline.transcribe_step` does around the encoder: a `get_state` and a
`cleanup_after_response` per row, a context-manager dictionary walk per row, a greedy decode
per row, a tokenizer `ids_to_text` per row per tick, and the label-looping decoder's
`while active_mask.any()` host synchronisations, whose count is the maximum over rows.
**Pad rows pay all of it.**

The graphed column is the same story from the other side: capturing the encoder step removes
about 18 ms of *fixed* cost and leaves the per-row slope alone (1.037 ms). That is exactly
why the graph path measured x1.29 at batch 32, where the fixed term dominates, and nothing at
batch 128, where the per-row term does. Two independent measurements, one mechanism.

The consequence for what to do next is large enough to state here: **capacity on this server
is bounded by per-row host work, not by the GPU.** A faster encoder buys little; removing
per-row Python and host synchronisations buys a lot. The first thing to try is the decoder
graph path, which `pipeline_config` currently pins off (`use_cuda_graph_decoder: False`),
because that flag is what forces the label-looping decoder onto its synchronising branch.

None of this is profiled. `step_ms` in `cache_aware.py` times NeMo's whole `transcribe_step`
as one block and nothing in the repo splits it, so the attribution above is arithmetic and
reading, not a measurement. Splitting it is the one-day job that decides the next year of
optimisation work, and it should happen before anything is rewritten.

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
