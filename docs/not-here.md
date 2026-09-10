# Not Here

Saying no early and in writing is the courtesy. This page exists so that a contributor knows the boundary before starting, not after opening a PR.

## Not a contribution surface

| Not accepted | Why | Where it belongs |
|---|---|---|
| New model classes / checkpoint families | Hugging Face Transformers owns day-0 reference implementations; NVIDIA's ASR lead authors them; model cards name the sanctioned runtimes | `huggingface/transformers`, `NVIDIA-NeMo/Speech` |
| New decoders, new kernels | Out of scope by construction — NeMo #15863 moved the bottleneck to the host, which is where this project works | NeMo, FlashInfer, CUTLASS |
| OpenAI-Realtime / Deepgram protocol compatibility | The Realtime beta was removed 2026-05-12 and legacy families shut down 2027-01-20; Deepgram v2 rejects unknown parameters, so compatibility would have to be exact and kept exact. A four-protocol treadmill is not maintainable by one person | nowhere; use the Riva subset |
| LiveKit / Pipecat plugin packages | Plugin packages live with the framework that loads them, and Pipecat documents its community tier as not maintained by the project itself | a recipe under `docs/recipes/` (not created yet) |
| Wrapper backends (WhisperLiveKit, speaches, LocalAI) | Their backends are in-process Python or C++/ggml across many accelerator families; Verbatim is neither | their repositories, if they want it |
| Whisper in any mode; Canary/AED; paged encoder caches; FP8/NVFP4 encoders; speculative decoding; diarization/PII/translation/TTS in year 1 | `docs/SCOPE.md`'s OUT list | SGLang, vLLM, NeMo |
| Dashboards, control planes, a hosted service | `docs/SCOPE.md`'s OUT list | — |

## Out of scope for year 1

The fourteen-item OUT list has one canonical copy in [docs/SCOPE.md](SCOPE.md). This page does not restate it.

## What is wanted instead

A measured row is the flagship contribution unit. It and the seven other surfaces — what each one produces and what knowledge it needs — are one table away, in [CONTRIBUTING.md §1](../CONTRIBUTING.md#1-the-contribution-units).

## If you think this is wrong

Open an issue with the case. Scope is a review criterion, but it is not secret: the boundary and its written reason are in [docs/SCOPE.md](SCOPE.md), and either can be argued against.
