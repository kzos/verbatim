# Third-party material

What this repository redistributes, under which licence, at which pin — and what it deliberately does
not redistribute. The on-prem operators this project serves run a licence scanner before they run a
server; finding an undisclosed dependency is a worse conversation than reading about a disclosed one.

## 1. Vendored: NVIDIA Riva protocol buffers (MIT)

| | |
|---|---|
| **Upstream** | <https://github.com/nvidia-riva/common> |
| **Commit** | `268890b7286031a6d4950e34f7ce13ed0d4ce621` |
| **Fetched** | 2026-09-08 |
| **Licence** | MIT — `SPDX-License-Identifier: MIT` in every file header; full text at [`proto/riva/LICENSE`](../proto/riva/LICENSE) |
| **Copyright** | Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. |

### Files

| Vendored path | Upstream path | Upstream blob SHA |
|---|---|---|
| `proto/riva/proto/riva_asr.proto` | `riva/proto/riva_asr.proto` | `a60399c9af0ab897dd26fc1641f37cab499afe4d` |
| `proto/riva/proto/riva_audio.proto` | `riva/proto/riva_audio.proto` | `f1a4b17a5eaa35fa565e05d51644fe3c3260e1d6` |
| `proto/riva/proto/riva_common.proto` | `riva/proto/riva_common.proto` | `43ce2c805e7486dbe08a282dd0e10aab9bd40661` |

Content SHA-256 of the vendored copies is recorded in [`proto/riva/CHECKSUMS`](../proto/riva/CHECKSUMS).
The pin itself is recorded in [`proto/riva/UPSTREAM`](../proto/riva/UPSTREAM).

`riva_asr.proto` imports the other two and nothing else; `riva_audio.proto` and `riva_common.proto`
import nothing. The remaining files in the upstream directory (`riva_nlp.proto`, `riva_nmt.proto`,
`riva_tts.proto`, `health.proto`, `BUILD`) are not vendored: NLP, NMT and TTS are out of scope, and the
health surface is plain HTTP on `:8080`, not gRPC.

`riva_asr.proto` additionally carries an upstream notice for material derived from Google LLC's speech
API definitions, licensed Apache-2.0. That notice is retained verbatim in the file header and is
reproduced by inclusion.

### Why this SHA and not the `stable` tag

Three consumers pin three different SHAs, seventeen months apart at the extremes:

| Consumer | Pin | Date |
|---|---|---|
| `nvidia-riva/common` tag `stable` = `r2.19.0` | `766a50d9` | 2025-03-19 |
| `NVIDIA/NeMo-Speech.cpp` submodule `proto/riva-common` | `71df9826` | 2026-05-12 |
| `nvidia-riva/python-clients` submodule | `268890b7` | 2026-08-13 |

`stable` does **not** contain `StreamingRecognizeRequest.runtime_config` and its `force_eou` key — the
one field NVIDIA added (2026-05-05) specifically for cache-aware RNNT models, which is the model family
Verbatim serves. A server implementing "the stable tag" would silently lack it.

Verbatim pins `268890b7` because that is the SHA `nvidia-riva/python-clients` ships, and that client is
what `livekit-plugins-nvidia` and Pipecat's `nvidia` STT service actually install. The clients that
matter define the wire we must speak.

`scripts/check_riva_proto_drift.py` (not yet written) is to diff the pinned files against
`nvidia-riva/common@main` weekly on CPU CI and open an issue on any change. Upstream changes roughly
annually — five content changes to `riva_asr.proto` in four years, every one additive — so this is a
cheap watch that will fire meaningfully when it fires.

### Generated stubs

`src/verbatim/protocols/riva/_gen/` holds `*_pb2.py`, `*_pb2.pyi` and `*_pb2_grpc.py` generated from the
vendored protos by [`scripts/gen_protos.sh`](../scripts/gen_protos.sh) with `grpc_tools.protoc`. They are
**committed** so that `pip install verbatim` never runs protoc, never needs a compiler and never needs
CUDA headers. Do not edit them; CI is to regenerate and fail on any diff.

`nvidia-riva-client` is deliberately **not** a runtime dependency. It pins `protobuf==6.33.5` exactly,
and its build rewrites `from riva.proto import ...` to `from riva.client.proto import ...` while the
descriptors still register the file paths `riva/proto/*.proto` and the `nvidia.riva.asr` symbols — so
importing both it and Verbatim's own stubs in one process is not a supported configuration. It is a
test-only dependency, to be installed into a separate virtualenv and driven as a separate process.

## 2. Depended on, never vendored

| Project | Licence | How it arrives |
|---|---|---|
| NVIDIA NeMo (`nemo-toolkit[asr]`) | Apache-2.0 | optional extra `verbatim[nemo]`, not installed by default |
| PyTorch | BSD-3-Clause | via the same extra |
| `grpcio`, `grpcio-tools` | Apache-2.0 | runtime / dev dependency |
| `protobuf` | BSD-3-Clause | runtime dependency |
| `websockets` | BSD-3-Clause | runtime dependency |
| `numpy` | BSD-3-Clause | runtime dependency |
| `soundfile` | BSD-3-Clause (wraps **libsndfile**, LGPL-2.1) | runtime dependency |
| `pytest`, `pytest-asyncio` | MIT | dev dependency |
| `ruff` | MIT | dev dependency |

`src/verbatim/pipelines/` is the only package permitted to `import nemo`, so no NeMo API break can
spread past one directory. The rule is currently enforced only by CI checking that `nemo-toolkit` and
`torch` are not installed at all; the AST walk that would enforce it properly is not yet written and is
tracked in `docs/BACKLOG.md`. No NeMo source is copied here; a CI check is to
fail the build on any NVIDIA/NeMo copyright header appearing outside this file.

### Transitive LGPL, stated rather than hidden

**libsndfile** (through `soundfile`) and **libsoxr** (through `soxr`, which arrives with
`nemo-toolkit[asr]` → `librosa` whether or not Verbatim declares it) are **LGPL-2.1**. A claim that the
default install is permissive-only would be false, and this project does not make it. PyAV, which
carries an LGPL FFmpeg build, is confined to an optional extra and is not installed by anything by
default.

## 3. Not redistributed

**Model checkpoints.** `nvidia/nemotron-3.5-asr-streaming-0.6b` and
`nvidia/nemotron-speech-streaming-en-0.6b` are **OpenMDW-1.1** (`license: other`,
`license_name: openmdw-1.1`, `license_link: https://openmdw.ai/license/1-1/` in the card front matter),
not Apache-2.0. They are never shipped in a wheel, a container or a release asset; they are fetched
from Hugging Face at first run. Verbatim's Apache-2.0 grant does not extend to them, and the CLI is to
print the model licence and its link on first download.

**Corpus audio.** LibriSpeech (`openslr/librispeech_asr`) and FLEURS (`google/fleurs`) are CC-BY-4.0, so
redistribution would be lawful; the reason not to is that a repository carrying a gigabyte of WAV is a
repository nobody clones, and a pinned dataset revision plus a checksum is a stronger reproducibility
claim than a copy. `corpora/` holds manifests only. `tel-vys` (Vystadial 2013, CC-BY-SA-3.0) is
manifest-only for the additional reason that its ShareAlike terms would attach to a redistribution.

**Arm (c) of the benchmark.** `modal-projects/modal-nvidia-asr` is **unlicensed** (`license: null` on
the GitHub API). It is cloned at container build time on the runner, measured, and never published or
redistributed. Rows produced from it are labelled "unlicensed; measured for comparison only".
