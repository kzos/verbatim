# Batch invariance in NVIDIA NeMo's cache-aware streaming pipelines

**What this is.** Nine measurements of whether the same audio produces the same transcript when the
batch around it changes, taken between 2026-09-08 and 2026-09-12 on stock NeMo. **Every figure here
describes NeMo, not Verbatim**, which has published no benchmark row. The probes are in
[`probes/`](../probes/) and run against a released wheel with no patches.

The short version: **batch composition is a small effect and reduced precision is a large one, and the
large one is the shipped default.**

## The headline

Same audio, byte for byte. Same weights, same decode path. The only variable is what else shares the
batch, on the code path a server runs.

| precision | A6000, of 2,939 utterances | B300, of 2,939 |
|---|---|---|
| float32 | **6** | **1** |
| bfloat16 | **287** | **264** |

`probes/streaming_divergence_pipeline.py`, `probes/b300_precision_divergence.py`.

Roughly one utterance in ten changes at bfloat16, on two architectures three generations apart, and the
gap is **wider on the newer silicon**, not narrower.

**Equalising the batch shapes does not help; it makes it slightly worse.** Padding every row to one
length moves the A6000 count from 287 to 328. So this is not a shape effect that fixed-shape batching
removes. It is arithmetic: the batch changes which reduction order the library picks, and at bfloat16
that changes the answer often enough to reach the transcript.

## Why this matters to anyone serving speech recognition

bfloat16 is what NeMo's own examples select, so it is what a server built on them serves by default.
A caller who sends the same audio twice, to the same model, on the same card, can get two different
transcripts depending on who else was talking at the time. Nothing warns them.

For most uses that is a curiosity. For a transcript that is evidence, a compliance record, or the input
to an evaluation, it is not: an A/B test whose control is not reproducible measures its own noise.

## What is not claimed

- **This is not a defect report against NeMo.** Reduced precision trading exactness for speed is the
  point of reduced precision. What is new here is the size of the effect at the transcript level and
  that it survives fixed-shape batching.
- **These are not Verbatim measurements.** Verbatim wraps this pipeline; it has not been benchmarked,
  and the invariance gate on Verbatim itself does not exist yet.
- **Two effects were separated and one earlier claim was withdrawn.** An earlier count of 38 in 2,939,
  all at the utterance tail, was measured through NeMo's *example* streaming buffer rather than the
  serving pipeline. On the serving path the count is 6 with ragged lengths and 6 with them equalised,
  and none at the tail. The tail effect was an artefact of the example buffer. The withdrawn figure is
  left in the record rather than deleted.

## The controls, which are the reason to believe the rest

| control | result |
|---|---|
| Same batch, repeated, nothing varied | zero differences |
| Row position changed within a batch | zero differences |
| Batch shape pinned, contents varied | 0 text and 0 timing differences in 1,024 |
| Second corpus, FLEURS `en_us` | 1 in 647, same direction |
| Unpadded repeat at float32, same audio twice | identical, 3 of 3 |

The last one matters most: at float32 this pipeline is reproducible enough that any difference under a
change is attributable to the change. That is what makes the bfloat16 counts readable.

## What a server should do about it

Verbatim's answer today is to say so at startup rather than to choose for the operator: `serve` prints
the counts above and names the flag, and
[DR-0003](decisions/0003-serve-defaults-to-the-precision-that-loses-invariance.md) records that the
default it inherits is the precision that loses the property the project is named for.

**The cost of choosing float32 is not yet known.** Measured at six streams on one A6000, latency is
indistinguishable between the two: median 173 to 174 ms either way, and the 95th percentile within
2 ms. Whether the throughput cost is real is unmeasured, and the number will appear in a row under
[`rows/`](../rows/) or not at all.

## Reproducing it

```bash
pip install torch nemo_toolkit[asr] datasets soundfile
python probes/streaming_divergence_pipeline.py
```

Each probe prints its own counts and writes its raw comparisons. They need a GPU and the checkpoint,
and nothing from this repository beyond the probe file.
