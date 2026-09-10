# Contributing to Verbatim

Thank you for considering it. This document is short on purpose, and the three sections that matter
are [the contribution units](#1-the-contribution-units), [the number rule](#2-the-number-rule) and
[the sign-off](#3-developer-certificate-of-origin-dco).

Verbatim is pre-alpha and openly conditional: dated kill gates decide whether it becomes a project at
all (see `README.md` § Status). Contributing to a project that may correctly stop is a real choice, so
the document says so before asking for anything.

## Before you start

Read [docs/SCOPE.md](docs/SCOPE.md) and [docs/not-here.md](docs/not-here.md). This project has a
narrow, fixed boundary and it says no to good code that sits outside it, so knowing where the edge is
before you write anything is worth the ten minutes.

---

## 0. The one commitment

> **The scheduler core is maintainer-only, and no contribution surface requires reading it.**

`src/verbatim/scheduler/` — the tick loop, CUDA-graph bucketing and capture, boundary side-batching,
slot assignment, admission control — is the one place where a contributor would have to hold the whole
system in their head. So it is the one place that is closed. Every other surface is either **data** (a
measurement row, a corpus, a failing case, an environment capture) or a **named interface**
(`PipelineAdapter`, `SessionTransport`, a Riva conformance case, a compatibility-matrix entry).

This is not a courtesy. A project with a realistic contributor ceiling in the low tens cannot spend any
of them on onboarding into a CUDA-graph scheduler.

---

## 1. The contribution units

Eight surfaces. In every row, the knowledge required excludes the scheduler core.

| # | Surface | What you produce | Knowledge you need |
|---|---|---|---|
| a | **Measurement rows** | `rows/<die>/<checkpoint>/<chunk>/<arm>-<date>-<handle>/` | run one container command |
| b | Per-locale WER rows | `rows/wer/<checkpoint>/<locale>/*.json` plus notes | **the language** |
| c | Invariance failing cases | `tests/invariance/cases/<id>/` | run at two concurrencies, diff |
| d | Riva conformance cases | `tests/protocol/riva/cases/<client>-<n>/` | run your own client |
| e | Pipeline adapters (post-MVP) | `src/verbatim/pipelines/<family>.py` | the NeMo pipeline plus the `PipelineAdapter` methods |
| f | Corpora and pacing patterns | `corpora/<id>.jsonl` + `<id>.yaml`, `bench/src/verbatim_bench/pacing/*.yaml` | **your own traffic shape** |
| g | Docs: compatibility matrix and recipes | `docs/compat/*.yaml`, `docs/recipes/*.md` | the deployment you ran |
| h | Hardware validation | `hardware/<die>/*.md` plus `env.json` | your machine |

**(a) is the flagship unit.** A measured row — a die, a checkpoint, a chunk mode, a concurrency
sustained at a stated p95, with its WER and its invariance verdict — is the thing this project trades
in. It is produced by `verbatim-bench run` (and checked by `verbatim-bench verify`), never hand-written, and it is
validated by CI with no human in the loop.

**"Add a model class" is explicitly NOT a unit.** Hugging Face Transformers owns the day-0 reference
implementations, and checkpoints load through NeMo's pipeline. No model class is requested, ever.

### Scope is a review criterion

The following are out of scope for year 1 and a PR that adds one will be closed with a pointer to this
line, however good the code is: Whisper in any mode; AED decoders and Canary; paged encoder caches;
CuTe/CUTLASS encoder kernels; FP8/NVFP4 encoders; speculative decoding; OpenAI-Realtime or Deepgram
protocol compatibility; LiveKit or Pipecat plugin packages; diarization, PII, translation, TTS or
speech-to-speech; a control plane, a dashboard or a hosted service; CPU/edge/mobile runtimes; ROCm;
and any re-implementation of NeMo's pipeline.

---

## 2. The number rule

> **No number that the harness did not produce.**

Not in this repository's README, not in the demo's captions, not in a talk, a model-card link, an issue
comment or a tweet. **A number is a link to a row.** If you cannot link the row, write the placeholder
in angle brackets and leave it unfilled — that is what `README.md` does today and it is not an
embarrassment, it is the rule working.

Nine more rules follow from it, and they apply to the maintainer exactly as they apply to you:

1. **Min-of-three, never best-of-three.** All three are printed.
2. **Failures are published** at the same size and in the same table as successes, including the day-21
   kill outcome whichever way it goes.
3. **No vendor numbers in the main table.** Vendor-published values live in a separately labelled panel
   with URL and fetch date. Measured NIM/Riva rows only with written permission from NVIDIA.
4. **Comparisons only across identical `(corpus_id, checkpoint@rev, chunk, dtype, profile, X)`.**
5. **The maintainer's server plays by the same rules.** Its rows are marked `self` and are `verified`
   only when an independent reproduction exists.
6. **The harness's own floor is published** with every release; a row whose p95 is within 2x of the
   floor is flagged "client-limited".
7. **Rows age visibly.** A stale row is grey, never gone, never quietly updated.
8. **No "up to".** A row is `S` under stated conditions; the conditions are in the row.
9. **Methodology changes bump the schema** and re-run the owned rows before the site renders the new
   version; old rows stay under the old schema label.

---

## 3. Developer Certificate of Origin (DCO)

Verbatim uses **DCO sign-off, not a CLA**. A CLA deters exactly the contributors this project needs, for
a relicensing optionality the project has decided it does not want.

Sign every commit:

```bash
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Your sign-off certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/) —
in short, that you wrote the contribution or otherwise have the right to submit it under this
repository's licence, and that you understand the contribution and your sign-off are public and
retained indefinitely.

Use your real name and a real email address. `git commit -s` fills both from your git config.

---

## 4. Working on a change

Work in a branch, keep the change to one subsystem, and leave the repository green:

- touch only the files your change needs, plus their tests;
- add no dependency that is not already declared in `pyproject.toml`;
- `ruff check .`, `ruff format --check .` and `python -m pytest -q` must all be clean before you open
  a pull request;
- if you find you need a fact the documentation does not state, say so in the pull request rather than
  guessing. A missing fact is a bug in the documentation.

## 5. Everything else

- **SPDX headers** on every source file: `# SPDX-License-Identifier: Apache-2.0`.
- **Style** is whatever `ruff check` and `ruff format` say. There is no second opinion.
- **Tests are CPU-only by default.** The `gpu` marker exists and runs only on self-hosted runners; a PR
  that needs a GPU to be reviewed waits for the one machine that can review it.
- **Do not vendor NeMo source.** `src/verbatim/pipelines/` is the only package permitted to import
  `nemo` or `torch`. CI checks that neither package is installed in the CPU job; the per-module AST walk test is not written yet, and `docs/BACKLOG.md` tracks it. When NeMo lacks a hook that is needed, the
  pull request goes to NeMo first; only then may a shim exist, and only with an upstream PR number and
  a deletion date.
- **Do not edit `src/verbatim/protocols/riva/_gen/`.** It is generated. Run `scripts/gen_protos.sh`.
- **Be decent.** See `CODE_OF_CONDUCT.md`.
