# Kill Conditions

`KILL.md` records the dated conditions under which Verbatim continues, stops, redirects or de-scopes. The clock is anchored at 2026-09-14 and the repository goes public on 2026-09-15. No gate date moves for any reason — the gate is fixed and the scope is the variable. This file is updated on each date with the verdict, whatever the verdict is, and is linked from the [README first screen](README.md). When a condition fires, the README changes the same day.

An unavailable die is recorded as **not measured**, never estimated or extrapolated.

## How to read this

Each condition carries six fields: **Date**, **What it measures**, **Measurable exit criterion**, **If it fires**, **Status**, and **Where the evidence will live**. **Kill** closes the project or server work named by the condition. **Stop** ends server work while preserving the surviving artifact named by the condition. **Redirect** moves the work to the upstream path named by the condition. **De-scope** removes the failed server scope while preserving the harness or upstream-fix work named by the condition.

## The seven conditions

### 1. Outreach and redirect

**Date:** 2026-09-19 (week 1) — messages sent; 2026-10-05 (day 21) — answers read.

**What it measures:** Whether the relevant upstream paths have stated a competing in-tree server direction or an asynchronous server commitment.

**Measurable exit criterion:** If the vLLM maintainers state in writing that vLLM wants a custom-`ModelState` Nemotron streaming path and would take it in-tree, build it there and keep only the harness standalone. If the NeMo maintainers state that NVIDIA will ship an asynchronous multi-client batched server in NeMo, NeMo-Speech.cpp or labs-Voice-Agent within two quarters, stop the server; contribute the invariance gate and harness upstream. The statement is what counts, from whoever is empowered to make it; this file names projects and not people. Silence is not an answer and triggers neither branch.

**If it fires:** Redirect the build to vLLM and keep only the harness standalone in the first branch. Stop the server and contribute the invariance gate and harness upstream in the second branch.

**Status:** pending — no verdict recorded

**Where the evidence will live:** `outreach/LOG.md` (not created yet), one thread per message with the single question and the answer or no answer by day 21.

### 2. Five-arm head-to-head

**Date:** 2026-10-05 (day 21).

**What it measures:** The five-arm comparison on one RTX A6000 at the two scoped chunk modes, at equal per-stream WER. A slower die lowers the file-driven ceiling and therefore raises the ratio for anything host-bound rather than GPU-bound, so an RTX A6000 pass is a weaker test than a pass on a faster die; the B300 block is the stricter test.

**Measurable exit criterion:** At 160 ms and 560 ms chunk modes, with the same checkpoint and paced real-time audio: (a) the prototype; (b) NeMo's `asr_streaming_infer.py` with `use_cuda_graphs=true` at batch 32 and 128, the file-driven graphed ceiling; (c) `modal-nvidia-asr` multi-stream, measured only and unlicensed; (d) the best existing open server, among NeMo-Speech.cpp `cuda-server`, labs-Voice-Agent and `briancaffey/nemotron-asr-server`; (e) vLLM's Qwen3-ASR realtime as the causal-on-vLLM alternative, streams and p95 only. Kill if (a) < 0.7× (b) in streams at equal per-stream WER, or if (d) already ≥ 0.5× (b). Publish the table either way.

**If it fires:** In the first case, kill the server because a Python host path cannot reach the ceiling and the remaining C++/Rust server should not be started. In the second case, kill the server because the gap is not worth a project. In both cases the table lands at `results/2026-10-05-five-arm-a6000.md` with its five run directories, `bench/` survives as the standalone harness, and the README's Status section records which case fired.

**Status:** pending — no verdict recorded

**Where the evidence will live:** `results/2026-10-05-five-arm-a6000.md` (not created yet), with its five run directories under `rows/`, which is present and empty until a run writes one, including exact reproduction commands and GPU-hours consumed.

### 3. Batch invariance

**Date:** 2026-10-29 (day 45).

**What it measures:** Whether concurrency and batch composition leave transcripts and timestamps bit-identical on the graph path, while preserving the batch-1 streaming reference accuracy.

> **The graph path is not in a released NeMo wheel.** Checked 2026-09-11 against NeMo 2.7.3 and 3.0.0: both halves of PR #15863 exist only in the source tree, so a default install can run this gate eager only. No threshold here moves because of that; a row must name its execution mode instead. See [docs/decisions/0002](docs/decisions/0002-the-graph-path-is-not-in-a-released-wheel.md).

**Measurable exit criterion:** Bit-identical transcripts and timestamps across concurrency 1 / 32 / maximum for 1,000 LibriSpeech test-other and 500 FLEURS utterances with ≤ 2 % of steps eager; per-stream WER equal to NeMo's batch-1 streaming reference at the same chunk mode within 0.1 absolute; and the same corpus re-run with fixed-shape batching disabled, whose divergence count is published either way.

**If it fires:** If unattainable on the graph path, de-scope to **harness plus upstream fix** and stop the server.

**Amended 2026-09-10, before any verdict, and recorded here rather than folded in silently.** Exploratory
probes since this condition was written change what a pass and a fail would mean:

- Transcript divergence is **architecture-dependent**. With the software held fixed it appears on an
  RTX A6000 and not on a B300. A bit-identical result on current silicon may therefore say more about
  the card than about the server, so this gate must name its die and must not generalise from it.
- **Word timing is the exposed channel**, moving 31 times per 2,912 utterances where the text was
  identical, against 4 times that a word changed. The criterion already names timestamps; it should be
  read as the primary channel rather than a secondary one.
- Word timing is **not** architecture-dependent in the same way: it moved with the software version on
  one card. The Blackwell timing channel has never been measured.

None of that changes the date, the corpus, or the threshold.

**Status:** pending — no verdict recorded

**Where the evidence will live:** `benchmarks/gate.json` (not created yet), with corpus size, concurrency levels, eager-step fraction and the first differing byte on failure.

### 4. The Riva drop-in

**Date:** 2026-11-13 (day 60).

**What it measures:** Whether the two named client integrations can complete live calls against the Riva subset without changes to their plugin code.

**Measurable exit criterion:** `livekit-plugins-nvidia` and Pipecat's `nvidia` STT service complete live calls against Verbatim with zero plugin code, at ≤ chunk + 150 ms p95 at the published ceiling. If the required Riva subset is not feature-complete by 2026-11-06, or the work on it passes three weeks of effort (logged in the tracking issue), or both plugins hard-require an NVIDIA API key — fall back to WebSocket-only and halve P(useful).

**If it fires:** The WebSocket surface becomes the only supported one, `docs/RIVA_SUBSET.md` records which RPCs and fields were reached before the fallback, and the halving of P(useful) is written up as a dated decision record under `docs/decisions/` with 0.40 left visible, as `docs/SCOPE.md` requires of any price change.

**Status:** pending — no verdict recorded

**Where the evidence will live:** `docs/RIVA_SUBSET.md` (not created yet), with implemented and deliberately unimplemented RPCs and fields and the status each returns.

### 5. Absorption watch

**Date:** 2026-11-13 (day 60).

**What it measures:** Whether an upstream runtime absorbs the cache-aware cross-session server path before the demand gate.

**Measurable exit criterion:** vLLM merges a Nemotron/Parakeet streaming model with cross-session batching into the Realtime table → redirect: build it there instead. NVIDIA ships an Apache-2.0 asynchronous multi-client batched server → stop.

**If it fires:** Redirect to vLLM when it merges the path; stop when NVIDIA ships the server.

**Status:** pending — no verdict recorded

**Where the evidence will live:** `watch/LOG.md` (not created yet), with a dated weekly entry including a dated no change.

### 6. Demand by name

**Date:** 2026-12-13 (day 90).

**What it measures:** Whether a named operator, NVIDIA or outside contributors create public evidence of demand by name.

**Measurable exit criterion:** At least one of: a named installer running ≥ 100 concurrent streams and reporting numbers publicly; NVIDIA naming the project; three outside contributors with merged measurement rows. None → stop at a respected utility; do not commit year 2.

**If it fires:** Stop at a respected utility and do not commit year 2.

**Status:** pending — no verdict recorded

**Where the evidence will live:** `DEMAND.md` (not created yet), with named installers, NVIDIA mentions and merged rows, each with permalinks and dates.

### 7. Standing cost, every week

**Date:** Standing cost, every week.

**What it measures:** The cost of the standing measurement matrix on owned hardware.

**Measurable exit criterion:** The measurement matrix (2 checkpoints × 5 chunk modes × 2 dies) runs on owned hardware in ≤ 4 GPU-hours/week with no rented CI; if it cannot, cut the matrix, never the gate.

**If it fires:** The cells dropped from the matrix are listed under `rows/` as **not measured**, with the week's GPU-hours beside them; no gate date, threshold or corpus in this file changes.

**Status:** pending — no verdict recorded

**Where the evidence will live:** the run directories under `rows/`, which is present and empty until a run writes one, with GPU-hours printed by the harness at the end of every run.

## What is not evidence

Stars, upvotes, aggregator rank and traffic are not evidence for condition 6. They are recorded separately in `DEMAND.md` (not created yet) and labelled as not evidence.

## Deviations from the source text

Recorded 2026-09-08. Four deviations are visible here:

- **Condition 2 die.** Original wording: condition 2 named a faster die that this project does not own. It is replaced with one RTX A6000 because this project owns four RTX A6000 and one B300, and publishing a claim about an unowned die would be exactly the unearned number the first rule exists to prevent.
- **Condition 4 effort cap.** Original wording: the required Riva subset exceeded an effort cap stated in a person-week unit. The source unit presumes several people and is not applicable here; the replacement is the date the source itself already set, 2026-11-06, plus an effort bar logged in the tracking issue.
- **Dropped parallel lane.** Original wording: a parallel kernel-contribution lane carried an effort cap. That clause was conditional on a person this project does not have, so it is dropped.
- **Condition 3 control arm.** The day-45 criterion now requires the same corpus re-run without fixed-shape batching, so the gate carries both bucketed and ragged arms and publishes the ragged divergence count either way.

The other conditions retain their current wording. Condition 7 in particular stands as written.
