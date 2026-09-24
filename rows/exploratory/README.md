# `rows/exploratory/` — runs that are NOT rows

A row under `rows/` is a measurement the day-21, day-45, day-60 and day-90 gates are read against. It
must be canonical, comparable and reproducible. Everything in **this** directory failed at least one of
those and is kept because the failure is informative.

Nothing here may be cited as a benchmark result, quoted in a comparison, or used to argue a gate either
way. Each file says in its own record why it is not a row.

> **Both files below carry `canonical_window: true` and neither run was canonical.** The ladder's rung
> executor built its load spec without the frozen window and with the ramp forced to zero, so a rung
> was a single pass of one utterance per slot: fifteen seconds of wall clock and 106 latency samples at
> six streams, against a frozen window of 180 s and a warm-up of 60 s that was applied nowhere. The flag
> compared command-line arguments with constants and attested only that the operator typed no override.
> Four repeats of one identical rung give p95 values of 289.8, 320.3, 302.6 and 301.8 ms against a
> 310 ms threshold: three passes and a failure from the same inputs. Evidence section 25.
> The executor was fixed on 2026-09-11 by
> [DR-0005](../../docs/decisions/0005-the-ladder-runs-the-window-it-reports.md). These two files were
> produced before it and stay withdrawn: a fixed harness does not make an old run canonical.

| file | why it is not a row |
|---|---|
| `ladder-a6000-bf16-eager-2026-09-13-all-criteria.json` | **The first search in which every rung evaluated all four criteria**, including thermal and word error rate against a batch-1 reference. Two seeds of three put the boundary at 22 streams; the third is non-monotonic across it, failing at 22 with a p95 of 312.5 ms and passing at 23 with 305.0, seven milliseconds either side of the 310 ms line. A ladder search assumes monotonicity, so it reports `s: 0` rather than 22. The number is real and the certification is not. Evidence section 40. |
| `ladder-b300-bf16-graphed-2026-09-11.json` | The graph path, reporting `s: 0`, which describes the search and not the server: one seed was non-monotonic, failing at 6 streams and passing at 7, and a ladder search assumes monotonicity. The two clusters read from its p95 values in evidence section 24 were repeat noise; a single rung repeated on one machine covers the whole 30 ms gap that section attributed to the graph path. Sections 24 and 25. |
| `ladder-a6000-bf16-eager-2026-09-11.json` | The three seeds located the boundary at N=4, 6 and 11. The server sits on its 310 ms latency budget at every concurrency tested, so pass/fail is decided by noise rather than by load. The reported `s: 4` is the low end of a spread, not a capacity. Sections 23 and 25. |

## The stock-pipeline probes and the words that changed

These are not server runs. Each takes the 2,939 recordings of LibriSpeech test-other through NeMo's
`CacheAwareRNNTPipeline` twice, alone and in slot 0 of a batch of 32, and keeps every recording whose
transcript differs. Model `stt_en_fastconformer_hybrid_large_streaming_multi`, att_context `[70,13]`
(1,120 ms chunks) — not the server's `[70,1]` at 160 ms, and not its bucket of 128, so they say how
often the stock pipeline's words depend on the batch, not what the padded server freezes. Kept
byte-for-byte as the probes wrote them; they predate the rule that a record states why it is not a
row, so the reason lives here.

| file | what it is |
|---|---|
| `stock-divergence-a6000-bf16-2026-09-10.json` | `probes/streaming_divergence_pipeline.py` on an RTX A6000, bfloat16, matmul precision `high`, NeMo's own pipeline builder. Varying arm (rows keep their lengths): 287 recordings changed. Equal-length arm: 328. NeMo version not stamped; dtype and matmul precision were recorded after the run, as the record says. |
| `stock-divergence-a6000-fp32-2026-09-10.json` | The same at float32 with matmul precision `highest` (true FP32, where NeMo's configs default to `high`): 6 and 6. |
| `stock-divergence-b300-2026-09-11.json` | `probes/b300_precision_divergence.py` on a B300 through the server's own builder, matmul precision `highest`, NeMo main `3.1.0+abb8254da` (a development version, not a release). bfloat16: 264 varying, 295 equal-length. float32: 1 and 1. |
| `unstable-words-2026-09-24.json` | Derived, not measured: `scripts/unstable_words.py` over the three files above. At each place where the two transcripts differ, which version matched the reference, word error rate either way, and how often reference words there are missing from a standard word list. The script's docstring states every scoring rule; the record carries the SHA-256 of its inputs and of the word list. |
