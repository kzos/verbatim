# Verbatim Scope

Verbatim is an Apache-2.0, asynchronous, multi-client, tick-scheduled streaming server for NVIDIA's cache-aware streaming speech pipelines. It is built on NeMo's Apache-2.0 `nemo.collections.asr.inference` package: the continuous-batched cache-aware RNNT/CTC pipelines, the CUDA-graph encoder step, the slot table, the label-looping greedy and MALSD beam decoders, endpointing, per-stream biasing and ITN are used as shipped, and Verbatim re-implements none of it. NeMo ships that pipeline; it does not ship a server, and its entry point takes a file, a directory or a manifest. The year-one checkpoint scope is `nvidia/nemotron-3.5-asr-streaming-0.6b` (multilingual, prompt-conditioned, `att_context_size` family `[56,*]`) and `nvidia/nemotron-speech-streaming-en-0.6b` (English, `[70,*]`).

## Status

Pre-alpha. Verbatim has not been benchmarked; every number describing Verbatim's behaviour in this repository is an angle-bracketed placeholder until a row under `rows/` fills it. The batch-invariance section below reports exploratory probes on stock NeMo, not on Verbatim, which is not implemented. Two external figures are cited to their sources: the checkpoint's monthly download count and the CUDA-graph speedup range. Whether this becomes a project is decided by dated gates against measured rows, not by intent.

## The hardware this runs on

The hardware available to this project is four NVIDIA RTX A6000, 48 GB each, Ampere, compute capability `sm_86`, and one B300, a Blackwell datacentre die. Every published row names its die. A row on any other die comes from whoever owns that card; otherwise the entry reads **not measured**. A slower die lowers the file-driven ceiling and therefore raises the ratio for anything host-bound rather than GPU-bound, so an RTX A6000 result is the weaker test than a result on a faster die. The B300 block is the stricter test.

## In scope — year 1

1. **Tick scheduler.** Asynchronous sessions join and leave; each session has an audio ring buffer. Every chunk period, the scheduler assembles per-chunk-mode batches into fixed bucket sizes whose CUDA graphs are captured and retained. NeMo keys graphs on shape, so buckets multiplied by chunk modes must fit the graph budget. Final steps (`is_last`) are peeled into a small eager side-batch so the steady-state batch never leaves the graph path. First steps are not: NeMo keys the graph on the per-batch pre-encode drop, which a first frame does not change, so a first frame is safe inside the steady batch. Note that the inference pipeline already splits finals into their own sub-batch, so this peeling duplicates work NeMo does and may be removable once measured; if measurement shows that side-batching costs more than it saves, capture is extended with per-slot validity masks. Slot assignment uses NeMo's `CacheAwareContextManager`. Admission control operates against a measured tick budget and yields a hard, published concurrency ceiling per chunk mode per GPU.
2. **Reuse, not rewrite.** NeMo's cache-aware RNNT/CTC pipelines, CUDA-graph encoder step, label-looping greedy and MALSD beam decoders, endpointing, per-stream biasing and ITN are used through the pipeline API. No new decoders and no new kernels.
3. **Batch invariance.** A session's transcript and timestamps are bit-identical regardless of concurrency and batch composition on the graph path. A CI gate runs the same corpus at concurrency `1 / 32 / maximum` and diffs outputs. Any padding artefact found, including the `#12840` class, is fixed upstream in NeMo first.
4. **Host path.** gRPC exposes the Riva `StreamingRecognize` subset used by `livekit-plugins-nvidia`, Pipecat's `nvidia` STT service and NeMo-Speech.cpp's `riva_server` clients. A plain WebSocket serves the demo. Audio decode and resampling run in worker threads, mel features are batched on the GPU, no JSON runs on the tick path, and back-pressure plus a p95 partial-latency SLO is provided per chunk mode.
5. **Harness.** A paced real-time load generator covers LibriSpeech test-other and FLEURS subsets with per-stream WER and latency; the five arms run in one container; and the published table contains streams × p95 × WER × invariance on the dies this project owns: RTX A6000 and B300.

## The MVP bar

`pip install verbatim && verbatim serve nvidia/nemotron-3.5-asr-streaming-0.6b --chunk 160ms` must support sessions at **≥ 0.8× the graphed file-driven ceiling** measured by NeMo's own `asr_streaming_infer.py` with `use_cuda_graphs=true` on the same box on the same day, at **≤ chunk + 150 ms p95**. The invariance gate must be green. A LiveKit Agents call and a Pipecat call must complete against it with zero plugin code. The five-arm table must be published.

## Explicitly out of scope — year 1

1. **Whisper in any mode** — SGLang, vLLM and faster-whisper own those runtimes.
2. **AED decoders and Canary** — NeMo and vLLM own those model paths.
3. **Paged encoder caches** — fixed-size state and NeMo's slot table make them unnecessary here.
4. **CuTe/CUTLASS encoder kernels** — the streaming step is host-launch-bound rather than kernel-bound — NeMo #15863 reports roughly 1.5k small kernels per step and a 3.08-5.14x range from graph capture — so kernel work is not where the cost sits; the split is not measured here.
5. **FP8/NVFP4 encoders** — the models are fp32, graphs skip autocast, and NeMo #16147 is doing mixed precision.
6. **Speculative decoding.**
7. **OpenAI-Realtime and Deepgram protocol compatibility.**
8. **LiveKit/Pipecat plugin packages** — the Riva path replaces them.
9. **Diarization, PII, translation, TTS and speech-to-speech in year 1** — NeMo pipelines own these paths; they are the growth path.
10. **A control plane, dashboards or a hosted service.**
11. **CPU/edge/mobile runtimes** — NeMo-Speech.cpp, parakeet.cpp and sherpa-onnx own those paths.
12. **ROCm.**
13. **Any re-implementation of NeMo's pipeline.**
14. **Any of the killed daemon/schema/company residues from the study that produced this project.**

## Where the boundary is enforced

Scope is a review criterion. A PR that adds an OUT item is closed with a pointer to this file, however good the code is. `docs/not-here.md` carries the redirect for each common case.

## The growth path, and why year 2 is not committed

Year 1 is the reference open server for cache-aware streaming ASR: ship the MVP, publish the table, offer the invariance gate upstream as a NeMo PR, run a day-0 measurement row for each new NVIDIA cache-aware checkpoint, and land the Riva-subset path in LiveKit's and Pipecat's docs as the self-hosted option. The trigger to continue past year 1 is any one of the following by 2027-03-31: NVIDIA names Verbatim on a model card, in NeMo docs or a blog; a named operator runs ≥ 500 concurrent streams on it and says so publicly; or vLLM declines native support in writing while the out-of-tree plugin path is taken.

Absent a trigger, Verbatim stays a maintained utility and year 2 is not committed. That is a successful outcome of this plan and not a failure of it: a small server that does one thing correctly, kept working, is worth more than a large one nobody depends on. Year 2, if it happens, is more session types under the same scheduler, gate and Riva surface as NeMo ships its pipelines.

## Related documents

- [KILL.md](../KILL.md) — the dated conditions under which this stops.
- [docs/not-here.md](not-here.md) — common refusals with redirects.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contribution units and review boundary.
- [README.md](../README.md) — the public first screen and current status.
