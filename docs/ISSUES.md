# First Ten Issues

This file contains the exact text of the ten issues to be opened when the repository goes public; they are **not open yet**. The fixed label set is `good-first-issue`, `documentation`, `backend`, `performance`, `research`, `hardware-support`, `protocol`, `harness` and `correctness`.

Every path outside this repository was verified against `NVIDIA-NeMo/Speech@main`, `nvidia-riva/common@main`, `NVIDIA/NeMo-Speech.cpp@HEAD`, `livekit/agents@main` and `pipecat-ai/pipecat@main` on 2026-09-07. Cite the path and treat any line number as a hint. Repository paths under `src/verbatim/`, `bench/`, `docs/`, `corpora/`, `rows/`, `benchmarks/` and `tests/` that these issues ask to add, aside from files already present, do not exist yet. Dated gates govern what gets published; see [docs/SCOPE.md](SCOPE.md).

## 1. [good-first-issue] Ship the paced load-generator corpus manifests (1,000 LibriSpeech test-other + 500 FLEURS)

**Labels:** `good-first-issue`, `harness`

**Size:** S

Every gate is measured against one fixed corpus, which today exists only as a sentence. The day-21 head-to-head, the day-45 invariance gate and every contributed hardware row must replay the same 1,000 LibriSpeech `test-other` and 500 FLEURS utterances in the same order, or nothing compares across dies or arms.

NeMo's manifest shape is already fixed. `prepare_audio_data()` in `nemo/collections/asr/inference/utils/manifest_io.py` reads JSONL keyed on `audio_filepath`, and `dump_output()` in the same file writes `audio_filepath` / `pred_text` back, so a matching manifest feeds arm (b), `examples/asr/asr_streaming_inference/asr_streaming_infer.py`, with no conversion. The trap is `prepare_audio_data(..., sort_by_duration=True)`: it reorders by duration. The load generator must not sort, because arrival order is part of the fixture and duration-sorting is exactly the batch-composition variable Issue 5 is about.

**Acceptance criteria**

- [ ] Commit `corpora/ls-other-1k.jsonl` (not created yet) and `corpora/fleurs-10-by-50.jsonl` (not created yet), with each entry carrying `audio_filepath`, `duration`, `text`, a per-file `sha256` and a stable `stream_id`.
- [ ] Add the `corpora/<id>.yaml` (not created yet) sidecar for each manifest with dataset id plus Hugging Face revision, licence, selection seed and manifest sha256.
- [ ] Add `corpora/build_manifests.py` (not created yet) so a pinned snapshot regenerates both manifests byte-identically, with split, revision and selection seed in the file.
- [ ] Verify audio as 16 kHz mono, with a non-conforming file failing the build loudly.
- [ ] Record the SHA-256 of each manifest in `corpora/CHECKSUMS` (not created yet).
- [ ] Update `benchmarks/README.md` so a row is accepted only against these checksums, with `corpus_id` equal to the manifest sha256 stamped on every row.

**Who this suits:** Someone who has run an ASR eval before, wants a first merge that everything else depends on, and already has LibriSpeech on disk.

## 2. [good-first-issue] Add a streams × p95 × WER × invariance row for the RTX 4090

**Labels:** `good-first-issue`, `harness`, `performance`

**Size:** S

Measured rows are this project's contribution unit, not model classes or features. This project owns four RTX A6000 (`sm_86`) and one B300 and will publish those dies itself; every other die in the table has to come from someone who owns one. This issue is the template, and it requests a row from someone who owns or can access an RTX 4090; it is not a claim that this project owns that die.

The RTX 4090 is a low-cost way to test whether the concurrency ceiling scales with SM count or with host-side launch throughput. NeMo PR #15863 (merged 2026-08-12) established that cache-aware streaming is host-launch-bound at around 1.5k small kernels per step, so a die with fewer SMs on the same host may not lose what is expected. Run `verbatim serve nvidia/nemotron-3.5-asr-streaming-0.6b --chunk 160ms`, then the paced load generator from Issue 1 at rising concurrency until p95 partial latency exceeds chunk + 150 ms; repeat at 560 ms. Both modes map to `streaming.att_context_size` values documented in `examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml`: `[56,13],[56,6],[56,3],[56,1],[56,0]` for the multilingual checkpoint and `[70,*]` for the English one.

**Acceptance criteria**

- [ ] Add `rows/rtx4090/nemotron-3.5-asr-streaming-0.6b/{160ms,560ms}/verbatim-<date>-<handle>/` (not created yet), one directory per chunk mode, produced by `verbatim-bench run` plus `verbatim-bench row` and never hand-written.
- [ ] Report maximum sustained streams, p95 and p99 partial latency, per-stream WER against the corpus reference, eager-step fraction, driver plus CUDA plus NeMo commit, and whether the invariance check passed at concurrency 1 / 32 / max.
- [ ] Use manifest checksums matching `corpora/CHECKSUMS` (not created yet).
- [ ] Attach the raw JSON run log to the PR.
- [ ] Regenerate `benchmarks/ROWS.md` (not created yet) by CI, not by hand.

**Who this suits:** A first-time contributor with a gaming or workstation RTX 4090 and an afternoon; no server internals are needed. This is the contribution on-ramp the project is judged on: whether anyone outside it runs the harness on hardware the author does not own.

## 3. [performance] Drive the eager-step fraction below 2 % by bucketing boundary steps

**Labels:** `performance`, `correctness`

**Size:** L

This is the central engineering claim and what the day-21 gate measures. NeMo's captured encoder step is `nemo/collections/asr/parts/submodules/streaming_encoder_cuda_graphs.py`: `CudaGraphsStreamingEncoderStep.__init__` takes `warmup_steps: int = 3, max_graphs: int = 8`; `_make_key()` keys a graph on input shape, dtype, device, `keep_all_outputs`, `drop_extra_pre_encoded`, `att_context_size`, `last_channel_cache_size` and `valid_out_len`; `stream_step()` falls back to `encoder._cache_aware_stream_step_impl` on any key miss once `len(self._graphs) >= self.max_graphs`; and `_can_use_graphs()` never captures a `keep_all_outputs=True` call at all.

The pipeline then guarantees misses: `CacheAwareRNNTPipeline.transcribe_step_for_frames()` in `nemo/collections/asr/inference/pipelines/cache_aware_rnnt_pipeline.py` splits each tick into a non-final sub-batch called with `keep_all_outputs=False` and a final sub-batch called with `keep_all_outputs=True` plus `right_paddings`. File-driven, final sub-batches are rare; in a server where sessions join and leave every tick, the non-final sub-batch shrinks by however many sessions ended that tick, creating a new shape key each time. Nothing is evicted: `_graphs` only grows until `reset_cuda_graphs_state()`, so each new steady shape burns one of the eight capture slots and, once the budget is full, every further shape runs eager for the life of the process.

The fix in scope is to bucket per-chunk-mode batches into a small fixed set of sizes, route first/last steps into a dedicated small eager side-batch so the steady batch never changes shape, and budget buckets × chunk modes against `max_graphs`. Measure first. Extending capture with per-slot validity masks is the alternative if the side-batch costs more than it saves.

**Acceptance criteria**

- [ ] Keep eager steps at ≤ 2 % of encoder steps at steady state, at 160 ms and 560 ms, at concurrency 32 and at the ceiling.
- [ ] Derive the bucket set and graph budget from `max_graphs` and assert the result at startup rather than relying on an unstated assumption.
- [ ] Record per-tick counters for graph hits, misses, eager side-batch size and evictions.
- [ ] Preserve transcripts, with the invariance suite from Issue 5 still green.
- [ ] Add a design note at `docs/scheduler.md` (not created yet) comparing the side-batch and validity-mask alternatives.

**Who this suits:** An inference-serving scheduler person comfortable with CUDA graph capture, shape bucketing and admission control.

## 4. [performance] Nsight Systems tick-budget profile: where the 160 ms tick actually goes

**Labels:** `performance`, `harness`

**Size:** M

Admission control here is not a heuristic: it admits sessions against a measured tick budget, and that budget is currently a guess. This issue produces the profile that turns it into a number and the repeatable recipe for regenerating it on any die.

The established upstream fact is that the encoder is not the problem. NeMo #15863 reported 3.08–5.14× from CUDA graphs because the step "spends most of its time waiting on the host", launching around 1.5k small kernels. The tick budget is therefore expected to be dominated by everything around the graph replay: audio decode and resample on worker threads, batched GPU mel, per-stream state gather/scatter through `CacheAwareContextManager.get_context()` / `update_cache()` in `nemo/collections/asr/inference/utils/context_manager.py`, the RNNT label-looping decode, endpointing and gRPC serialization. Capture with `nsys profile` plus NVTX ranges around each named tick phase, at concurrency 1, 32 and the ceiling, at 160 ms and 560 ms.

**Acceptance criteria**

- [ ] Instrument every tick phase with NVTX ranges whose names are stable enough to diff across releases.
- [ ] Make `make profile` produce an `.nsys-rep` and a phase-breakdown table in one command.
- [ ] Publish the breakdown at three concurrencies × two chunk modes on the RTX A6000 in `docs/tick-budget.md` (not created yet), calling out the CPU-side gap between graph replays explicitly.
- [ ] Derive the admission-control constant in the scheduler from that table and cite it in a comment.
- [ ] Keep profiling overhead at ≤ 1 % of steady-state p95 when NVTX is compiled out.

**Who this suits:** A GPU serving performance engineer comfortable with Nsight Systems and NVTX.

## 5. [research] Reproduce NeMo #12840's batch-composition WER drift, then propose the upstream fix

**Labels:** `research`, `correctness`

**Size:** L

NVIDIA-NeMo/Speech issue #12840, "Cache Aware Streaming script yields different results for different batch_sizes", was opened 2025-04-01 by `gabitza-tech` and closed by a stale-bot on 2025-06-13 without a fix. The upstream attribution of PR #15863 is `hamuzhan`, and `naymaraq` is the upstream author of `context_manager.py`; these handles cite existing artifacts, not assignment requests.

Verbatim's headline correctness property, transcripts and timestamps bit-identical regardless of concurrency and batch composition, is a claim about that defect, and the day-45 gate stands or falls on it. Any padding artefact of the #12840 class is fixed upstream in NeMo first.

The suspected surface is right-padding. `CacheAwareRNNTPipeline.preprocess()` and its CTC twin in `cache_aware_ctc_pipeline.py` take `right_paddings` and compute `feature_buffer_lens = feature_buffer_lens - right_paddings`, but only the final sub-batch gets a padding list; non-final steps pass `None`. Batch members share one concatenated tensor and one encoder step, so whether a chunk is padded depends on which other sessions are in the tick. `prepare_audio_data(..., sort_by_duration=True)` means NeMo's own entry point silently varies that composition with batch size, a plausible mechanism for the ±1.0 absolute WER swing reported on #12840.

**What is already known, so this issue does not start from zero.** Batch composition has been shown to
change the transcript on the cache-aware streaming path, driving `conformer_stream_step` chunk by chunk
over 2,939 LibriSpeech `test-other` utterances at concurrency 1 against 32. It reproduces two distinct
effects, and separating them is the first thing this issue should preserve:

- **Length coupling, 38 occurrences.** Every one differs only at or after the last word the two arms share.
  A short session beside longer neighbours keeps receiving trailing chunks and its decoder keeps emitting;
  alone, the buffer empties at its own length and the tail is dropped. This is the `right_paddings` class
  of defect described above.
- **Numerical perturbation, 1 occurrence**, once every row is padded to a common length so chunk count is
  identical in both arms. It is mid-sentence, and it is the same utterance and the same two hypotheses the
  offline probe found independently.

The README's batch-invariance section carries both figures. What remains is the root cause, the upstream
filing, and the gate.

**Acceptance criteria**

- [ ] Commit the reproducer on stock NeMo under `tests/invariance/` (not created yet), with the NeMo commit
      pinned, keeping the two effects separated. A report that conflates them will read as a padding
      mistake in the harness rather than the padding defect it demonstrates.
- [x] Show the same utterance, checkpoint and `att_context_size`, differing only in batch composition,
      producing differing transcripts. Done on the streaming path, both effects.
- [ ] Trace the root cause to a named tensor and line rather than asserting it.
- [ ] File an upstream issue or a comment reopening #12840 with the reproducer, and file a PR proposing the fix.
- [ ] Run a CI gate diffing 1,000 LibriSpeech plus 500 FLEURS utterances bit-identically in transcript and timestamps at concurrency 1 / 32 / max.
- [ ] Keep per-stream WER within 0.1 absolute of NeMo's batch-1 streaming reference at the same chunk mode.
- [ ] If the property is unreachable on the graph path, write the finding up, de-scope `docs/SCOPE.md` to "harness plus upstream fix", and publish that outcome.
- [ ] If the reproducer finds no defect at that commit, publish that negative result too and report #12840 as not reproducing rather than quietly dropping it.

**Who this suits:** Someone with numerical-reproducibility instincts, computed references, exact-comparison discipline and patience with padding arithmetic.

## 6. [protocol] Riva StreamingRecognize conformance cases from livekit-plugins-nvidia and Pipecat

**Labels:** `protocol`, `harness`, `correctness`

**Size:** M

The day-60 condition is that `livekit-plugins-nvidia` and Pipecat's `nvidia` STT service complete live calls with zero plugin code. Waiting until day 60 to discover which fields those clients send is how that gate gets missed. This issue turns what both do today into conformance tests that run in CI without LiveKit or Pipecat installed.

From source, as read on 2026-09-07: LiveKit, in `livekit-plugins/livekit-plugins-nvidia/livekit/plugins/nvidia/stt.py`, calls `GetRivaSpeechRecognitionConfig` at setup, iterating `config_response.model_config` and filtering on `parameters["type"] == "online"` and `parameters["language_code"]`. It sends a non-empty `model`, `enable_word_time_offsets=True`, `interim_results=True`, `max_alternatives=1`, `LINEAR_PCM` and `audio_channel_count=1`; it reads `alternatives[0].words` and needs `start_time`/`end_time` in ms. Its `use_ssl=False` branch does not require `NVIDIA_API_KEY`.

Pipecat, in `src/pipecat/services/nvidia/stt.py`, sends `model=""`, `verbatim_transcripts`, `profanity_filter`, and calls `add_endpoint_parameters_to_config` (`EndpointingConfig`, default `stop_history=320`), `add_word_boosting_to_config` (`SpeechContext.phrases` plus `boost`) and `add_custom_configuration_to_config`. Ground truth is `nvidia-riva/common`, `riva/proto/riva_asr.proto`: service `RivaSpeechRecognition`, `StreamingRecognizeRequest` with the `streaming_config`/`audio_content` oneof and the `runtime_config` map carrying `"force_eou"`, `EndpointingConfig`, `RecognitionConfig`, `StreamingRecognitionResult` and `WordInfo`. NVIDIA's reference subset is `NVIDIA/NeMo-Speech.cpp`, `src/services/grpc_asr.cc` / `.h`.

**Acceptance criteria**

- [ ] Add `tests/protocol/riva/test_riva_streaming.py` (not created yet), driving a raw gRPC client built from the vendored, committed stubs with no `riva.client` dependency in-process.
- [ ] Store recorded real-client cases under `tests/protocol/riva/cases/<client>-<n>/` (not created yet), surface (d) in `CONTRIBUTING.md` §1.
- [ ] Cover config-first-message ordering and empty versus non-empty `model`.
- [ ] Cover `GetRivaSpeechRecognitionConfig` returning `parameters["type"]="online"` and `language_code`.
- [ ] Cover interim-then-final ordering, with `WordInfo` in ms on finals only.
- [ ] Cover `EndpointingConfig` accepted and honoured, plus `SpeechContext` boosting.
- [ ] Cover unknown `custom_configuration` keys being ignored rather than fatal.
- [ ] Cover `runtime_config["force_eou"]` accepted without aborting the stream; because NeMo's pipeline exposes no force-EOU hook today, it is accepted-and-ignored with a warning until an upstream hook exists, and the case asserts exactly that.
- [ ] Cover a plaintext, no-TLS, no-API-key channel.
- [ ] Document non-fatal behaviour for every unsupported field; no unsupported field aborts a stream. CI runs the suite on each PR.

**Who this suits:** A gRPC/protobuf contributor who has integrated a voice-agent stack.

## 7. [documentation] Write the Riva StreamingRecognize supported-subset table, field by field

**Labels:** `documentation`, `protocol`

**Size:** M

Verbatim implements a subset of Riva ASR. A self-hoster deciding whether to point an existing client at it needs that subset written down before installing anything. Today the only way to know is to read the server.

Complete `src/verbatim/protocols/riva/conformance.py`, the machine-readable honoured / accepted-and-ignored / rejected table from which `docs/protocols/riva.md` is generated, with a CI check that fails when the two diverge. The module is already in the repository: it defines `FieldStatus`, the frozen `FieldRule`, the `CONFORMANCE` table and `missing_fields()`. What does not exist yet is the generator that renders `docs/protocols/riva.md` from it, the CI divergence check, and rows for any field the table still omits. Cover all three RPCs on service `RivaSpeechRecognition`: `Recognize`, `StreamingRecognize` and `GetRivaSpeechRecognitionConfig`; LiveKit calls the third at setup, so not implemented is not an option.

Walk `RecognitionConfig` field by field: `encoding` (only `LINEAR_PCM`), `sample_rate_hertz`, `language_code`, `max_alternatives`, `profanity_filter`, `speech_contexts`, `enable_word_time_offsets`, `enable_automatic_punctuation`, `model`, `verbatim_transcripts` (ITN on/off), `diarization_config` (out of scope in year 1), `custom_configuration` and `endpointing_config`. Then cover `StreamingRecognitionConfig.interim_results`, `StreamingRecognizeRequest.runtime_config["force_eou"]` (the proto notes it is cache-aware-RNNT-only; accepted but not yet honoured because NeMo's pipeline exposes no hook), and the response side `StreamingRecognitionResult`, including `stability` and `audio_processed`, plus `WordInfo`.

Where behaviour should follow a precedent rather than be invented, the precedent is NVIDIA's own C++ server, `NVIDIA/NeMo-Speech.cpp`, `src/services/grpc_asr.cc`, which maps `speech_contexts`, accepts `endpointing_config.stop_history_eou` with a `custom_configuration` string alias and handles `runtime_config["force_eou"]`.

**Acceptance criteria**

- [ ] Generate `docs/protocols/riva.md` from `src/verbatim/protocols/riva/conformance.py`, updating the existing generated placeholder through that source of truth.
- [ ] Cover every field reachable from `StreamingRecognize`; no field is skipped silently.
- [ ] Give every row the field, proto line, status, Verbatim behaviour and the client that exercises it.
- [ ] Put divergences from `NVIDIA/NeMo-Speech.cpp` in their own section with a reason.
- [ ] Add a CI check that fails when a proto field exists with no row.
- [ ] Link the generated table from the README under "connect an existing client".

**Who this suits:** A documentation-minded contributor who reads protobuf comfortably; a natural pairing with Issue 6.

## 8. [documentation] NeMo checkpoint compatibility matrix: what loads, at which att_context_size, with which decoder

**Labels:** `documentation`

**Size:** M

The claim that any cache-aware FastConformer checkpoint NeMo loads needs a table, because it is not uniformly true and the failure modes are quiet. Two examples are visible in NeMo's own config. `examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml` documents that `att_context_size` values are model-specific: `[70,13],[70,6],[70,1],[70,0]` for `nvidia/nemotron-speech-streaming-en-0.6b` but `[56,13],[56,6],[56,3],[56,1],[56,0]` for the multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b`. The same file notes that `use_cuda_graphs` "requires `use_amp=false`", which the capture path enforces because `_can_use_graphs()` refuses capture under CUDA autocast. A user who enables AMP silently loses the graph path and the concurrency ceiling with it.

The matrix should also record what `CacheAwarePipelineBuilder.build()` in `nemo/collections/asr/inference/factory/cache_aware_pipeline_builder.py` will and will not accept: `ASRDecodingType` is CTC or RNNT only, and cache-aware RNNT is restricted to `greedy_batch` and `malsd_batch`. Anything outside that is out of scope and should say so rather than fail at load.

**Acceptance criteria**

- [ ] Add `docs/checkpoints.md` (not created yet), distinct from the generated stack matrix `COMPATIBILITY.md` (not created yet): Verbatim × NeMo × torch × CUDA × driver × die.
- [ ] Include one row per checkpoint with HF id, family, supported `att_context_size` values and their chunk durations in ms, decoder types, whether prompt-conditioning is required, ITN availability and status: `supported`, `loads-untested` or `unsupported`.
- [ ] Mark both year-one checkpoints `supported` only with the die and NeMo commit on which each row was verified.
- [ ] Add an explicit "not in year 1" section covering Whisper, Canary/AED, buffered non-cache-aware pipelines and diarization.
- [ ] Document the AMP-versus-CUDA-graphs interaction with the startup warning Verbatim emits, quoted verbatim.
- [ ] State that a `loads-untested` row can be promoted by any contributor with a measurement row from Issue 2.

**Who this suits:** A contributor who has already got a NeMo checkpoint running and hit one of these edges. The GPU cost is low and the leverage is high: this is the page NVIDIA would have to read before naming the project on a model card.

## 9. [hardware-support] Validate on L40S, H100 and B200; publish the rows

**Labels:** `hardware-support`, `performance`

**Size:** M

The standing weekly matrix runs on the hardware this project owns: RTX A6000 (`sm_86`) and B300. That spans an Ampere workstation die and a datacentre Blackwell die, which is a wide architectural gap and therefore a portability test, but it misses the dies most operators deploy: L40S, an Ada datacentre SKU, and H100, the die used by NVIDIA's published streaming numbers, and it says nothing about B200. This issue closes that gap with rows from people who own those cards and tests whether the ceiling is portable at all. A die named here is a requested contributed row, not hardware owned by this project.

There is a hypothesis to falsify per die: because the workload is host-launch-bound, as NeMo #15863 reports, the ceiling may track host CPU and PCIe more than SM count. An L40S in a well-provisioned host could land closer to an H100 than FLOPs predict. Publish whatever the numbers say. Each die needs bucket sizes and graph budget re-derived so Issue 3's startup assertion holds, the tick budget re-profiled per Issue 4, and the invariance gate re-run per Issue 5. A die where bit-identity fails is a finding worth its own issue.

**Acceptance criteria**

- [ ] Add `rows/{l40s,h100,b200}/<checkpoint>/{160ms,560ms}/…` (not created yet) for both checkpoints, with each row contributed by someone who owns the named card.
- [ ] Allow a one-off row produced on rented time, flag it `rented`, and keep it out of the standing weekly matrix.
- [ ] State streams at fixed p95, per-stream WER, eager-step fraction and invariance pass/fail at 1 / 32 / max.
- [ ] Report streams as a fraction of the graphed file-driven ceiling on the same die: day-21 arm (b), `asr_streaming_infer.py` with `asr.use_cuda_graphs=true` at batch 32 and 128, never as a bare number.
- [ ] Record any die-specific workaround in `docs/checkpoints.md` (not created yet), or in `COMPATIBILITY.md` (not created yet) when it is a stack-version matter, rather than hiding it in code.
- [ ] Record per-die GPU-hour cost so the weekly matrix stays inside the project's standing hardware budget.
- [ ] Record a die nobody runs as **not measured**, never estimated and never extrapolated.

**Who this suits:** Someone with access to an L40S, an H100 or a B200, such as an operator evaluating Verbatim for deployment or a contributor with rented capacity.

## 10. [backend] Cache-aware CTC pipeline adapter behind the same tick loop (post-MVP)

**Labels:** `backend`

**Size:** L

Post-MVP; do not start before the day-45 invariance gate is green. The MVP is RNNT-only, one chunk mode and one die. A second session family added before invariance is proved makes a failing gate impossible to attribute. It is filed now because it is the first real test of `docs/SCOPE.md`'s growth path: the tick loop should generalise to every streaming pipeline NeMo ships without a server. Cache-aware CTC is the bounded test: it exists upstream as `CacheAwareCTCPipeline` in `nemo/collections/asr/inference/pipelines/cache_aware_ctc_pipeline.py`, is built by the same `CacheAwarePipelineBuilder` through its `ASRDecodingType.CTC` branch, shares `CacheAwareContextManager` slot management and has the same `keep_all_outputs` sub-batch split, so it exercises the graph budget with a second set of shapes rather than a second architecture.

Its shipped config, `examples/asr/conf/asr_streaming_inference/cache_aware_ctc.yaml`, defaults to `batch_size: 256` / `num_slots: 1024` against the RNNT config's `64` / `256`. That is exactly the pressure the bucket budget must survive. The adapter must remain an adapter: no re-implementation of NeMo's pipeline, no new decoders and no new kernels.

**Acceptance criteria**

- [ ] Extend the existing `PipelineAdapter` interface in `src/verbatim/pipelines/base.py`, implement the RNNT adapter against it first with no behaviour change, and keep the invariance suite green.
- [ ] Add the CTC adapter behind that interface, with `verbatim serve --pipeline ctc` working end to end over the same Riva surface.
- [ ] Re-derive the graph budget for CTC shapes, with Issue 3's startup assertion holding for RNNT-only, CTC-only and both loaded.
- [ ] Extend the invariance gate to CTC and keep it green at 1 / 32 / max.
- [ ] Add a CTC measurement row on the RTX A6000 at 160 ms and 560 ms.
- [ ] Explicitly defer Sortformer sessions in the design note, citing `NVIDIA-NeMo/Speech` PRs #16174 and #16210 rather than smuggling them into this issue.

**Who this suits:** A serving-backend engineer who wants a bounded piece and is comfortable waiting for a gate; best done by whoever owned Issue 3.

## Coverage

| Coverage requirement | Issues |
|---|---|
| ≥ 2 `good-first-issue` | 1, 2 |
| ≥ 1 `documentation` | 7, 8 |
| One `backend` implementation, post-MVP | 10 |
| Two `performance` issues | 3, 4 |
| One `research` issue | 5 |
| One `hardware-support` issue | 9 |
| One `protocol`/`harness` issue | 6 |

Issues 1–4 feed the day-21 head-to-head. Issue 5 is the day-45 gate. Issues 6–7 feed the day-60 Riva drop-in. Issues 2 and 9 feed the day-90 demand condition. Issue 10 is post-MVP by construction. Nothing in this set adds a decoder, a kernel, a protocol beyond the Riva subset or a re-implementation of NeMo's pipeline.
