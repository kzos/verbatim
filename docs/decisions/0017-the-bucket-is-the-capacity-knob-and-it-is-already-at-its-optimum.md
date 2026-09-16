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

## Where the step's time actually goes — profiled, 2026-09-16

An earlier version of this section derived a per-row cost of **0.850 ms** from the
file-driven ceiling row by treating `B x 0.16 / RTFx` as a step time, and observed that it
predicted the fixed point at 124. **That figure is withdrawn: it is wrong by a factor of six,
and the prediction was a coincidence.** RTFx there measures a different pipeline — NeMo's own
file-driven script, with its own I/O and batching — and converting it to "milliseconds per
step" assumed this server's step structure.

`probes/step_phase_split.py` measures it directly instead, wrapping
`CacheAwareRNNTInferenceWrapper.encoder_step` and `rnnt_decoder_predictions_tensor` with
device synchronisation on either side and attributing the remainder of NeMo's
`transcribe_step` to "the rest". Real corpus audio, B300, bfloat16, eager encoder
(`rows/exploratory/step-phase-speech-b300-2026-09-16.json`):

| batch | step | encoder | decoder | the rest |
|---|---|---|---|---|
| 32 | 26.2 ms | 14.5 ms | 7.0 ms | 4.7 ms |
| 128 | 40.2 ms | 15.1 ms | 12.3 ms | 12.8 ms |
| 256 | 56.2 ms | 14.6 ms | 18.0 ms | 23.6 ms |

| marginal | per row |
|---|---|
| step | 0.134 ms |
| **encoder** | **0.001 ms** |
| decoder | 0.049 ms |
| the rest | 0.084 ms |

**The encoder is flat.** 14.5, 15.1, 14.6 milliseconds for 32, 128 and 256 rows — eight times
the work for one per cent more time. It is entirely fixed cost, which is consistent with the
arithmetic: at `att_context [70, 1]` a chunk is two post-subsampling frames and re-projecting
72 cached frames through 17 layers is about a GFLOP per row, nothing for this die. The whole
marginal cost is host-side, split roughly one third decoder and two thirds the per-row Python
around it — `get_state` and `cleanup_after_response` per row, a context-manager dictionary
walk per row, a greedy decode per row, a tokenizer `ids_to_text` per row per tick. **Pad rows
pay all of it.**

Two consequences. **A faster or disaggregated encoder buys nothing**, because the encoder does
not scale with occupancy in the first place; the thing to attack is per-row host work, and the
first lever is the decoder graph path that `pipeline_config` pins off. And **the step cost
does not explain the 124 cap**: at bucket 128 a step is 40 ms against a 112 ms budget, so that
arm was never near a compute limit — it ended on `integrity:refused`, which is admission
capping at the bucket, exactly as the criterion said.

**What is now open, and was not visible before.** At bucket 256 a step costs 56 ms, still well
inside the budget, yet that arm failed on *latency* at 46 to 47 streams with p95 319 ms. The
step alone does not account for that. The phase offset contributes up to one tick period
(DR-0012) and the edge batches contribute more, but neither is measured against this. That gap
between step cost and observed latency is the next thing to profile, and nothing here should
be read as explaining it.

The probe measures a synthetic drive of `transcribe_step`, not a live server: no admission, no
edge batches, no session churn, no biasing. Uniform noise was tried first and understated the
marginal by a fifth (0.108 against 0.134 ms per row), because noise decodes to almost nothing
and the greedy loop, the endpointer and `ids_to_text` all do their cheapest possible work. Both
records are kept.

### CUDA graphs buy fixed cost, in the decoder exactly as in the encoder

`--decoder-graphs` lets NeMo's label-looping decoder capture graphs instead of running the
torch branch, whose `while active_mask.any()` loop synchronises the host every iteration. The
same probe, same audio, both ways
(`rows/exploratory/step-phase-speech-decgraph-b300-2026-09-16.json`):

| batch | decoder off | decoder on | removed |
|---|---|---|---|
| 32 | 7.0 ms | 3.7 ms | **48 %** |
| 128 | 12.3 ms | 9.5 ms | 22 % |
| 256 | 18.0 ms | 16.4 ms | 9 % |

The saving is a roughly constant 3 ms that does not grow with the batch — the marginal per-row
decoder cost is if anything slightly worse (0.057 against 0.049 ms). **Graphs remove launch
overhead, not per-row work**, which is the same shape as the encoder result of 2026-09-13:
x1.29 at batch 32, nothing at batch 128. Two independent measurements of two different graphs,
one mechanism.

For this server that is about 3 ms off a 40 ms step at bucket 128, or 7 %. Real, worth having,
and not a capacity lever — capacity at that bucket is bound by admission, not by the step.

**Invariance survives it**, which is the part that had to be checked before the flag could
ship: one digest across concurrency 1 / 32 / 32 / 42 under churn, zero divergences
(`rows/exploratory/invariance-decgraph-b300-2026-09-16.json`). The digest is `44bea7df3ec7…`
where the same server without decoder graphs gives `f68cbccfa089…` — a different decode path
gives different transcripts, exactly as the flag's own help says, which is why it is a separate
arm.

**The ladder could not answer this question and should not have been asked it.** The arm was
run at bucket 256 on the reasoning that latency binds there, and it returned S = 22 against a
baseline of 19, with per-seed highest passing rungs of 22 / 68 / 22 against 46 / 46 / 19 — and
a rung that failed on latency at n=46 with p95 419 ms while n=68 passed at 309 ms. A
non-monotonicity that large is not a capacity ceiling. Bucket 256 sits far past the fixed point
this record is about, and its variance swamps a 7 % effect. A 3 ms difference wants a profile,
not a capacity search.

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
