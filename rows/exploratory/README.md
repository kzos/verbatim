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

## Step 1 on Nemotron, one B300, and the A6000's unpadded server twice at concurrency 8 (2026-09-26)

`step1-b300-2026-09-26/` holds the records of step 1 on one NVIDIA B300 SXM6 AC (one card: every
record here that names a card, 15 of the 36, names the same one), NeMo `3.1.0` and torch `2.11.0+cu128`, the same model and
revision as above, bucket 128, 160 ms, bfloat16, decoder graphs off, and of a retest at
concurrency 64 on the same card the next day (2026-09-27, UTC). None is a row.

| file | commit | what it is |
|---|---|---|
| `server-26eaebb-fixed-eager-c32.json.gz`, `-fixed-eager-c8` | `26eaebb` | `probes/server_frozen_answers.py`, all 2,939 recordings, padding fixed, eager encoder step, the client at concurrency 32 and 8. |
| `server-26eaebb-fixed-graphs-c32.json.gz`, `-fixed-graphs-c8` | `26eaebb` | The same with the encoder step on the CUDA-graph path (the server reports `graph path`). |
| `server-26eaebb-ragged-eager-c32.json.gz`, `-ragged-eager-c32-repeat` | `26eaebb` | Padding ragged, eager, concurrency 32, twice on one server: ragged against ragged. |
| `places-26eaebb-eager.json.gz` | `26eaebb` | `scripts/compare_captures.py --places-out`: fixed eager c32 (served) against ragged, fixed eager c8 as the repeat; names each capture by sha256. |
| `server-ba20c49-ragged-c128-overload-run1.json.gz` | `ba20c49` | An overload test: padding ragged, eager, the client at concurrency 128. Complete. |
| `server-ba20c49-ragged-c128-overload-run2.FAILED.json.gz` | `ba20c49` | **Not a capture.** The second run of the overload test, right after the first on the same server process: its admission control refused most sessions, so the probe wrote this record of the failure instead of a capture. Kept only as evidence of the refusal. |
| `server-ba20c49-ragged-c64-first-attempt.FAILED.json.gz` | `ba20c49` | **Not a capture.** The retest's first attempt at the ragged run at concurrency 64, on a server process started for it: most recordings got no final and the server refused most sessions, so the probe wrote this record of the failure instead of a capture. Its cause was not found. Kept only as evidence that the attempt failed; nothing is compared with it. |
| `server-ba20c49-ragged-c64-run1.json.gz`, `-ragged-c64-run2` | `ba20c49` | The retest: padding ragged, eager, the client at concurrency 64, twice, one right after the other on another server process started for it. Both complete: ragged against ragged at concurrency 64. |
| `server-ba20c49-fixed-c64.FAILED.json.gz` | `ba20c49` | **Not a capture.** Padding fixed at concurrency 64, on a third server process started for it after the two ragged runs: its admission control refused most sessions, so the probe wrote this record of the failure. Kept only as evidence of the refusal. |
| `gate-ba20c49-summary.json.gz` | `ba20c49` | `scripts/step1_gate_runbook.sh` (`RUNBOOK_GPU_INDEX` 0, bucket 128, max level 42, 256 LibriSpeech test-other streams, eager): the runbook's summary of its four arms and its exit code. |
| `gate-ba20c49-invariance-<arm>.json.gz`, `-finals-<arm>`, `-record-summary-<arm>`, `-smoke-check-<arm>`, `-serve-spec-<arm>` | `ba20c49` | For each arm (`fixed-churn`, `fixed-const`, `ragged-churn`, `ragged-const`): the invariance record, every level's finals, the runbook's check of the record (which names the record and the finals by sha256), its smoke check (tick budget and p95 during the smoke), and what `verbatim serve` builds from the arm's command line. These stamp no commit; the gate summary stamps `ba20c49` for every arm. |
| `gate-ba20c49-card.json.gz`, `gate-ba20c49-model.json.gz` | `ba20c49` | The runbook's preflight: the card, and the model revision and `.nemo` sha256. |

`step1-a6000-c8-2026-09-26/` holds two ragged captures at concurrency 8 on the RTX A6000 of the
section above (card index 3), at `ba20c49`, one right after the other on one server process (NeMo
`3.1.0+cf724ac33`: a different NeMo build from the B300's). They are kept out of
`step1-a6000-2026-09-26/` so that directory's manifest, and the summary that names it by sha256,
stay byte for byte as merged.

`step1-a6000-gate-smoke-2026-09-26/` holds what is left on record of the invariance runbook on that
A6000 (card index 3) at `5eda079`: two attempts, at bucket 64 and at bucket 32, each stopped by the
smoke check before its first arm's gate (`scripts/step1_gate_runbook.sh` stops when the smoke check
lists a problem). For each attempt: the smoke check (`gate-5eda079-b<bucket>-smoke-check-fixed-churn`:
the 112 ms budget, the highest p95 tick, 156.75 ms at bucket 64 and 123.96 ms at bucket 32, and the
problem that stopped the runbook), the serve spec (`-serve-spec-`: what `verbatim serve` builds from
the arm's command line, bucket included) and the server's `/readyz` before the smoke (`-readyz-`: the
device name, `NVIDIA RTX A6000`). None of them stamps a commit; the manifest gives the run's preflight
git head, which nothing checks. Not copied: the attempts' process-environment files, argv files,
admission and smoke readings, logs and the rest of the preflight.

As above, each file is the record its tool wrote, gzipped whole (`gzip -9 -n`), with the absolute
paths its tool stamped replaced by `scripts/scrub_record_paths.py` (`<checkout>`, `<venv>`, `<home>`,
`<step1-out>`, and on the B300 `<workdir>`, the directory that held the checkout, the environment,
the model cache, the corpus and the outputs), links rewritten to the scrubbed sha256, and each
manifest keeping `sha256_as_written`. Not copied: each gate arm's
`arm.json` and `server-environ.json`, which hold the server process's whole environment (host
names, network addresses, account names, paths outside any placeholder); the facts the summary
reads from an arm are in its record summary, smoke check and serve spec, which are here. Also not
copied: `compare_captures.py`'s own reports (the summary derives them again) and the text logs.

`step1-b300-summary-2026-09-26.json` is `scripts/step1_b300_summary.py`'s output over these three
directories and `step1-a6000-2026-09-26/`, and nothing else; it reads the last only for the cards
its records name and for the server processes of its unpadded pair at concurrency 32. It refuses a record whose sha256 or commit is not its manifest's, a
places record or record summary that does not name the records here by sha256, a record holding a
machine path, and a capture whose name says a padding, execution or concurrency it did not record.
Every figure comes through the repo's own tools: `compare_captures.py` (FROZEN, identity, the
comparator's refusals), `step1_places.derive` (word error rate, timing shifts), the stock probe's
`words()` (word-level), and `verbatim_bench.invariance.assess`, rerun on each gate arm's finals and
required to give the verdict, digests and divergences the invariance record, the finals and both
runbook summaries give. A card is counted by its GPU UUID wherever a record names one, under any
key whose name holds `uuid` (`card_uuids`), not by its device name: on the B300 one card, named by
15 of its 36 records; on the A6000 one card, named by 18 of the 24 records of its three
directories. Each unpadded pair's server processes are counted by the process ids its captures stamp:
the A6000's pair at concurrency 32 ran on two processes, each other pair on one.
`tests/test_step1_b300_summary.py` rebuilds it from the committed records and requires the same
bytes. What it gives:

| | B300 | A6000 |
|---|---|---|
| fixed, eager: c32 against c8, ragged as control | FROZEN, 2,939 of 2,939 identical, digest `5a672de4fe0cb924` | (section above: `45e1ddba…`) |
| fixed, graph path: c32 against c8 | identical on 2,939 of 2,939, same digest; FROZEN not claimed (no ragged graph-path control) | not measured |
| fixed eager against fixed graphs, c32 and c8 | identical on 2,939 of 2,939 each | not measured |
| ragged against fixed, c32 | 266 text (128 word-level), 815 timing-only | (section above) |
| ragged against ragged | c32: 1 text (0 word-level), 2 timing-only; c64: 4 text (0 word-level), 4 timing-only | c8: 204 text (104 word-level), 691 timing-only |
| ticks over budget during the ragged runs | 4 of 3,953 and 1 of 3,932 (c32); 285 of 2,016 and 248 of 2,001 (c64) | 985 of 15,613 and 1,003 of 15,607 (c8) |
| p95 tick after the ragged runs | 73.5 ms and 73.5 ms (c32); 124.8 ms and 109.1 ms (c64) | 118.2 ms and 116.1 ms |

The gate on the B300: `fixed-churn` and `fixed-const` invariant at 1 / 32a / 32b / max (one digest,
and the same digest in both arms at every level); `ragged-churn` divergent, 78 / 78 / 65 streams
against level 1 (84 distinct), `ragged-const` divergent, 78 / 78 / 36 (96 distinct), 0 between 32a
and 32b in both; runbook exit 0. The smoke checks' highest p95 tick was 109.9 ms (`fixed-churn`)
and 88.3 ms (`fixed-const`) against the 112 ms budget their serve spec derived; during the gate
itself the `fixed-churn` arm's monitor reached 112.53 ms, just over it, and admission control never
degraded. The captures do not stamp their server's tick budget.

The overload test is incomplete and does not say whether overload breaks run-to-run repeatability:
the first run at concurrency 128 is complete (553 of 1,079 ticks over budget), and the second was
refused (the server admitted 388 of 2,939 sessions and refused 2,551), so there is no second run
to compare it with.

The retest at concurrency 64 (`ba20c49`) measures run-to-run repeatability there instead. Its two
ragged runs, on one server process, are complete (2,939 admitted, none refused, in each). Their
modelled tick cost went over the budget on 285 of 2,016 and 248 of 2,001 ticks, and 97 and 33 ticks
ran late (ended after the next tick was due), which is where the card fell behind real time. They
differ on 4 texts (none at word level: punctuation only) and on 4 word timings only; digests
`f7b19c8b…` and `143ad7ac…`. The fixed server at the same concurrency, on a fresh process, did not
hold it: 136 of 247 ticks over budget during the run, and the server admitted 350 sessions and
refused 2,589 ("admissions held at degradation level 1"), so there is no fixed capture at 64.
Before the ragged runs, a first attempt on another fresh process failed: the server admitted 128
sessions and refused 2,811 ("128 live sessions fill the largest bucket"); of the 128 sent, 12 got a
final that covers their audio and 116 did not (96 of them no partial either; the client gave up on
114); 2,927 of the 2,939 recordings have no final that covers their audio (2 of them got a final
that does not cover it) and 2,907 no partial; 335 of 339 ticks were late and 6 over budget. Its
cause was not found. The client's in-flight count times a session from its first audio frame to its
last final, so it covers only the 14 sessions that received a final (`sessions_timed`), not the 128
sent, and its peak of 14 does not say how many of the 128 were in flight at once. The retest's
first ragged run started 236 s after the failed attempt ended. `step1_b300_summary.py` recounts each
failed record's counts from its recordings and refuses any it does not give.
