# DR-0011 — Warm-up asks the runtime whether it captured, and steps one more time than upstream

**Date:** 2026-09-13
**Status:** accepted

## What was wrong

DR-0010 gave the scheduler a warm-up: step each bucket shape `GRAPH_WARMUP_STEPS` times over pad
rows at construction, so the capture happens before the boundary grid opens rather than inside a
tick with a deadline. `GRAPH_WARMUP_STEPS` was 3, because upstream's
`CudaGraphsStreamingEncoderStep.__init__` takes `warmup_steps: int = 3`. After the warm-up, the
controller added the key to `self._captured` and returned it as captured.

Both halves were wrong, and the first one hid the second.

Upstream does not capture on the `warmup_steps`-th call. Its dispatch reads:

```python
self._key_counts[key] = self._key_counts.get(key, 0) + 1
if self._key_counts[key] > self.warmup_steps and len(self._graphs) < self.max_graphs:
```

`>`, not `>=`. Three steps leave the count at three, three is not greater than three, and nothing
is captured. And `self._captured.add(key)` recorded the capture regardless, because nothing in that
path ever asked the runtime what it was holding — the tenth guard in this project that cannot fail.

## What was measured

On the B300 (NVIDIA B300 SXM6, NeMo 3.1.0+abb8254da carrying PR #15863, torch 2.11.0+cu128),
checkpoint `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`, 160 ms, bucket 32, bfloat16,
reading `CudaGraphsStreamingEncoderStep._graphs` and `._key_counts` directly:

| stage | graphs held | upstream's call count for the shape |
|---|---|---|
| after the pipeline is built | 0 | — |
| after Verbatim's warm-up (3 steps) | **0** | 3 |
| after one more step | **1** | 4 |

Verbatim reported `GraphCapability(requested=True, available=True)` and claimed
`['160ms x 32 (steady)']` captured at every one of those stages. `serve` printed
`graphs   graph path (CUDA graphs, NeMo PR #15863)` and was telling the truth about the
capability and a falsehood about the capture.

The probe is `probes/graph_capture_truth.py`; its output is
`rows/exploratory/graph-capture-truth-b300-2026-09-13.json`.

Two things this also settled, both of which had been open:

- **bfloat16 does not disqualify the graph path.** Upstream runs eager under CUDA autocast, and its
  docstring notes cache-aware streaming models run in float32 anyway, so this was a live risk.
  Verbatim's NeMo config sets `asr.use_amp: false`, autocast is off, and the captured key's dtype
  reads `torch.bfloat16`. The graph path and the bfloat16 default are compatible.
- **Upstream's key is the shape plus the streaming parameters**, read off the captured key:
  `((32, 80, 25), torch.bfloat16, device 0, keep_all_outputs=False, drop_extra_pre_encoded=2,
  att_context (70, 1), last_channel_cache 70, valid_out_len 2)`. The bucket is the batch dimension,
  which is what `CapturePlan` assumed and is now confirmed rather than assumed.

## The decision

1. `GRAPH_WARMUP_STEPS = UPSTREAM_WARMUP_STEPS + 1`, with upstream's own count named as the constant
   it is derived from, so the relationship is in the code and not in a comment that can drift.
2. `PipelineAdapter.retained_graphs()` reports how many graphs the runtime is holding, or `None`
   for an adapter that cannot tell. `NeMoBoundary` binds a reader for it where it already reaches
   into NeMo. It reads a private `_graphs` dict because upstream exposes no count.
3. `CaptureController.warmup` reads that count before and after each shape and raises
   `CaptureNotRetained` if it did not rise. `None` is "no answer" and is never read as zero.
4. The CPU fake captures on the call whose count *exceeds* its `graph_warmup_steps`, which is
   upstream's rule, instead of on first sight.

## Rejected alternatives

**Set `GRAPH_WARMUP_STEPS = 4` and leave the recording alone.** This is the one-line fix and it
would have been enough for this defect and for no other. Upstream falls back to eager in silence on
any capture failure and on any step its own `_can_use_graphs` disqualifies — autocast, a training
flag, an absent cache, a full `max_graphs`. Every one of those produces a server that announces the
graph path, runs every tick eager, and publishes rows saying graphed. The count had to become
observable, not just correct.

**Force upstream's mode so a failed capture raises** (`force_cuda_graphs_mode`, which sets
`cuda_graphs_allow_fallback = False`). Tempting, and still open: it would turn a silent fallback
into an exception at the source rather than a count that did not move. Not taken here because it is
documented as testing-only and disables fallback for the life of the process, including for shapes
Verbatim deliberately runs eager. Recorded in `docs/BACKLOG.md` instead.

**Have the fake keep capturing on first sight, and test the threshold only against the real
runtime.** This is what let the defect through: a double easier to satisfy than the thing it stands
for. The whole CPU suite passed on an adapter that captured on step one, so no arrangement of
`GRAPH_WARMUP_STEPS` could have failed a test. The fake now implements the rule it is standing in
for, and `GRAPH_WARMUP_STEPS = 3` turns ten scheduler tests red.

## Settled afterwards, on the same box

Three things this record left open or a cold read raised, all measured the same day
(`probes/graph_key_reality.py`, `rows/exploratory/graph-key-reality-b300-2026-09-13.json`):

- **A live steady batch presents the key warm-up captured.** This record's open question. A real
  first frame batched with 31 pad rows left exactly one key at count 4, as did the steady frame
  after it, so `drop_extra_pre_encoded` is the same for both and `CapturePlan`'s one-key-per-bucket
  assumption holds. The graph budget is not understated.
- **A graph captured on the constructing thread replays on another thread.** Warm-up runs where the
  `TickLoop` is built and live steps run on the tick thread; torch's current stream is thread-local,
  so this was not obvious. A step from a second thread did not raise, left the count at one graph,
  and did not raise upstream's call count, which is replay rather than a second warm-up.
- **The crossed-wire guard discriminates.** The built `CacheAwareRNNTPipeline` declares
  `greedy_rnnt_decoder` and not `greedy_ctc_decoder`, so the adapters' mutual refusal rests on a
  real difference rather than accepting both in silence. It has to be read on the outer pipeline:
  the inner `CacheAwareRNNTInferenceWrapper` declares neither, and that is the object a future
  refactor could accidentally bind.

## What is still not proven here

That a row's `graphed` label is derived from the capture rather than from the request. `serve`'s
banner and `obs/counters.py` both compute the execution mode from the `--eager` flag, not from
`CaptureController.mode`. With warm-up now refusing to start when nothing was captured, a server
that says graphed did capture -- but the label and the fact still travel by different routes, and
only one of them is checked.
