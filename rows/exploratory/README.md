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

## Step 1 on Nemotron, one RTX A6000 (2026-09-24 to 2026-09-26)

`step1-a6000-2026-09-26/` holds the raw records of step 1: is `nvidia/nemotron-speech-streaming-en-0.6b`
(revision `ebe59e5a`) batch-dependent in the stock pipeline, does the server's fixed padding freeze its
answers, and do the places where answers flip find errors better than the model's own word
confidence? Every record was taken on one NVIDIA RTX A6000 with NeMo `3.1.0+cf724ac33` and torch
`2.11.0+cu128`, on LibriSpeech test-other (2,939 recordings). None is a row.

Each file is the record its tool wrote, with one change, gzipped whole (`gzip -9 -n`). The tools
stamp absolute paths of the machine that ran them (its checkout, virtual environment, home
directory with the Hugging Face cache, and output directory), and a path is not committed:
`scripts/scrub_record_paths.py` replaced each by a placeholder (`<checkout>`, `<venv>`, `<home>`,
`<step1-out>`) and nothing else, and rewrote each places record's sha256 links to its captures to
the scrubbed captures' sha256. The four stock records held no path and are byte for byte as
written. `MANIFEST.json` gives, for each, the file name the tool wrote, the sha256 and size of the
raw bytes and of the gz file, the commit the record stamps (and the field it stamps it in), and
what it is; for each scrubbed record, also `sha256_as_written` and `bytes_as_written` (the record
as its tool wrote it) and `scrub` (each placeholder put in, by count, and the links rewritten).
Rerunning the scrub with the same paths on the records as written gives these bytes.

`step1-summary-2026-09-26.json` is `scripts/step1_summary.py`'s output over these records and
nothing else: the stock arms' counts (text, word-level, punctuation-only, timing-only) and word
timing shifts, word error rate per side, the replay's verdict, the server's FROZEN verdicts and
digests, fixed against ragged and ragged against ragged, the flips against the model's confidence
at the same number of words with both confidence modes (with the range over every way of breaking
a tie at the confidence cutoff), the 160 ms scores with their nulls, and each capture's tick
statistics. It refuses records whose sha256 or commit is not the manifest's, or that hold a
machine path, and computes every figure with the repo's own scorers and comparators
(`scripts/step1_places.py`, `scripts/compare_captures.py`, `scripts/compare_stock_replay.py`, the
probe's `words()`); its docstring states the rules. `tests/test_step1_summary.py` rebuilds it from
the committed records and requires the same bytes. It derives no out-of-dictionary figure (that
needs a word list, which is not among these records), and the tick budget is not stamped in the
captures.

| setting | stock, 1,120 ms | stock, 160 ms | server captures |
|---|---|---|---|
| path | NeMo's `CacheAwareRNNTPipeline`, built by `build_pipeline` | the same | a running `verbatim serve` |
| chunk, att_context | 1,120 ms, `[70,13]` (requested) | 160 ms, `[70,1]` | 160 ms, `[70,1]` (read off the built encoder by `/readyz`) |
| rows | alone, and slot 0 of a batch of 32 | the same | bucket 128, padding `fixed` or `ragged` (seen in the server's command line) |
| dtype, matmul | bfloat16 (and float32), `highest` | bfloat16, `high` | bfloat16, `high` (declared; the server does not report it) |
| encoder step, decoder graphs | not stamped at `6583a83`; no encoder CUDA graphs requested at `854d4b9`; decoder graphs not stamped | no encoder CUDA graphs requested; decoder graphs off (observed) | eager, as the server reports it; decoder graphs off |
| word confidence | off where stamped | off (observed on the built decoder) | off, or `paper-best` / `nemo-shipped` |
| recordings | 2,939 | the first 1,024 | 2,939, the client admitting 32 or 8 at once |

| file | commit | what it is |
|---|---|---|
| `stock-nemotron-bf16-1120.json.gz` | `6583a83` | `probes/stock_divergence.py`, when it was one top-level script: each recording alone and in slot 0 of a batch of 32, arms ragged, equalised and fixed, on the final text and the word timings. Keeps every recording's two transcripts, and both texts and word timings of those that differ; timings are (word, start_s, end_s). |
| `stock-nemotron-fp32-1120.json.gz` | `854d4b9` | The same at float32. |
| `stock-hybrid-bf16-1120.json.gz` | `854d4b9` | The old model, `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` revision `ae981433`, at the bfloat16 settings above. |
| `stock-nemotron-bf16-160-high-n1024.json.gz` | `daae587` | 160 ms chunks, matmul `high`, the first 1,024 recordings; stores every recording's two word lists (`every_recording`). |
| `replay-22e8406-nemotron-bf16-1120-n64.json.gz` | `22e8406` | The probe as restructured, replaying the first 64 targets of `stock-nemotron-bf16-1120`; `scripts/compare_stock_replay.py` compares the two. |
| `server-22e8406-fixed-c32.json.gz`, `-fixed-c8`, `-ragged-c32` | `22e8406` | `probes/server_frozen_answers.py`: every recording's final text and word timings from the running server, padding fixed with the client at concurrency 32 and at 8, and ragged at 32. |
| `places-22e8406.json.gz` | `22e8406` | `scripts/compare_captures.py --places-out` over those three: where the served (fixed) and the ragged answers differ, the second fixed capture as the repeat. It names each capture by sha256. |
| `server-a207de6-fixed-c32.json.gz`, `-fixed-c8`, `-ragged-c32`, `places-a207de6.json.gz` | `a207de6` | The same three captures and their places again, at `a207de6`. |
| `server-a207de6-fixed-paperbest-c32.json.gz`, `-fixed-nemoshipped-c32` | `a207de6` | Fixed padding at concurrency 32 with word confidence `paper-best` and `nemo-shipped`: every word carries the model's confidence. `scripts/step1_places.py` scores the `a207de6` places with them. |
| `server-a207de6-ragged-c32-repeat.json.gz` | `a207de6` | A second ragged capture at the same commit and setting: ragged against ragged. |

The commits are the development commits that wrote each record; this branch squashes them. Between
`a207de6` and this branch's `26eaebb`, `git diff --stat a207de6 26eaebb -- src probes scripts bench`
lists only `bench/src/verbatim_bench/calibrate.py`, `scripts/gen_protos.sh` and the protobuf generator
pin: the server and the probe that took the `a207de6` captures, and the scripts that score them, are
this branch's. The captures stamp `probe_sha256` `e9d8c34d…` and `bench_client_sha256` `7fc02e90…`, the
sha256 of `probes/server_frozen_answers.py` and `bench/src/verbatim_bench/client.py` at `26eaebb`,
and the replay stamps `b7b69a9f…`, that of `probes/stock_divergence.py`. `22e8406` predates the
word-confidence change in `src/verbatim/pipelines/nemo_runtime.py`, `observed.py` and
`protocols/health.py` that `26eaebb` carries.

Not here: the 160 ms run with `paper-best` word confidence, whose record holds 272 of its 1,024
targets and no finish stamp because NeMo raised mid-run in the decoder word-confidence pass that
`26eaebb` turns off; the smoke capture, the profiles and the invariance runbook's runs.
