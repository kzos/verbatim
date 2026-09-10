# Benchmark Methodology

## 1. The freeze and its proof

Freezing turns every definition from an output into an input. A wrong-but-frozen definition produces a wrong-but-honest table, which the correction rule below lets us fix with a schema bump and a full re-run. An unfrozen definition is a knob, and after the first run the author knows which way each knob moves the number relative to the kill thresholds. The latency definition alone moves the measured stream count by about one chunk period. The commit date is the only evidence the knobs were not turned.

This methodology is frozen and committed before the first head-to-head comparative row exists, with a target of 2026-09-30. A local commit date is forgeable, so the freeze is proven only by a push to the public repository or by an external timestamp of the commit recorded in a `docs/decisions/` record. This repository is now public, so the push itself is that proof: the commit that introduced this document carries a date GitHub records independently of any local clock.

## 2. Latency

The single primary latency is `word_emission`. For word `k` in a stream's final transcript, `L_k = t_first_stable_appearance(k) - t_audio_end(k)`, where the first term is the receive time of the first partial in which the word appears at that position and is never subsequently removed, and the second is the moment the load generator sends the last sample of that word. This definition is protocol-independent and can score an arm whose wire has no watermark.

The secondary latency is `chunk_watermark`. For chunk `i` covering audio `[a_i, b_i)`, it is `t_recv(first partial whose watermark >= b_i) - t_send(b_i)`. The WebSocket surface carries the watermark as `audio_s`; the Riva surface carries it as `audio_processed`. Arms with a watermark record this secondary, while an arm without one records no secondary value. The pre-declared fallback, if that definition cannot be implemented by the freeze date, is watermark-only: an arm with no watermark on its wire carries no p95 rather than silently choosing another definition.

`first_partial_after_send` is deleted, not demoted and not retained as a third column. It is overload-blind: in the CPU harness demonstration measured against a null server, p95 was approximately 166 ms against a backlog of approximately 3 s. The arrival-order match can pair a send with an older chunk's partial, so the value stays near one chunk period while backlog grows. The surviving implementation is `bench/src/verbatim_bench/client.py::match_partials_by_watermark`.

## 3. Constants

Every value in this section is a declared input, definition, threshold, or procedure constant, not a measurement. The run schema constant is `SCHEMA_VERSION_FOR_RUN = "vb-results/2"`. The latency constants are `LATENCY_PRIMARY = "word_emission"`, `LATENCY_SECONDARY = "chunk_watermark"`, `LATENCY_FALLBACK_IF_PRIMARY_UNAVAILABLE = "chunk_watermark"`, `LATENCY_DELETED = "first_partial_after_send"`, and `X_MS = 150`.

The rung uses `WARM_UP_S = 60`, `WINDOW_S = 180`, `WARM_UP_READING_S = 30`, `WARM_UP_CONVERGENCE = 0.10`, and `WARM_UP_CAP_S = 120`. Two consecutive readings of the reading length within 10 percent open the window; failure to converge by the cap fails the rung as `unstable`.

The ladder uses `LADDER_N0_FRACTION_OF_C = 0.5`, `LADDER_N0_WITHOUT_CEILING = 16`, `LADDER_MULTIPLIER = 1.15`, `LADDER_RESOLUTION = 0.02`, and `LADDER_INVALID_RUNGS_TO_ABORT = 2`. Its seeds are `SEEDS = (20260914, 20260915, 20260916)`, in that order and distinct, and `S_REPEATS = 3`.

The workload constants are `SESSION_PROFILE = "m180"`, `PACING_PROFILE = "uniform"`, `FRAME_MS = 20`, `FRAME_JITTER_MS = 10` in either direction with seeded jitter, and `CHUNK_MODES_DAY21 = (160, 560)`.

The ceiling constants are `CEILING_BATCH_SIZES = (32, 128)`, `CEILING_NUM_SLOTS_EQUALS_BATCH = true`, `CEILING_WARMUP_STEPS = 1`, `CEILING_RUN_STEPS = 3`, `CEILING_MEDIAN_OF = 5`, `CEILING_MIN_INPUT_STREAMS_PER_BATCH_SLOT = 4`, and `CEILING_RTFX_CROSS_CHECK = 0.02`. The kill constants are `KILL_RULE_1_THRESHOLD = 0.7`, `KILL_RULE_2_THRESHOLD = 0.5`, `F_MAINTAINER_RERUN_ABOVE = 1.1`, and `WER_WINDOW_ABSOLUTE = 0.1`.

The validity constants are `CLIENT_CPU_MAX_FRACTION_OF_CPUSET = 0.5`, `PACING_SLIP_P99_MAX_MS = 5.0`, `CLIENT_LIMITED_FLOOR_MULTIPLE = 2.0`, `STEAL_PCT_MAX = 0.0`, `CGROUP_THROTTLED_DELTA_MAX = 0`, `NVML_THROTTLE_EVENTS_MAX = 0`, `FOREIGN_GPU_PROCESSES_MAX = 0`, and `GPU_PERSISTENCE_MODE_REQUIRED = true`. The four calibration values named in section 7 are intentionally still `null`.

The day-21 corpus manifest SHA-256 is `<corpus-manifest-sha256>`, and its stratified-sample seed is `<day-21-stratified-sample-seed>`. The session-builder version and seed are `<session-builder-version>` and `<session-builder-seed>`. The room-noise source revision is `<room-noise-source-revision>`, and the filler/evaluation split is `<filler-evaluation-split>`. The checkpoint revision and checkpoint-file SHA-256 are `<checkpoint-revision>` and `<checkpoint-file-sha256>`. The toolkit commit SHA is `<toolkit-commit-sha>`, the container-digest policy is `<container-digest-policy>`, and the normaliser package version is `<normaliser-package-version>`. These placeholders stay angle-bracketed and unfilled until the corresponding artifacts exist; no value is inferred.

The document path is `benchmarks/METHODOLOGY.md`. The machine-read block after section 13 is the sorted JSON representation of the code source, and is the only block used for mechanical equality.

## 4. The arm registry

The registry is frozen at the pin date. Each row has a committed launch description even when the artifact is not yet available; angle-bracketed entries are deliberately unfilled rather than guessed.

| Arm | Kind | Pinned commit | Dtype class | Committed launch line | Surface | Kill-rule role | Panel | not_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) | candidate | `<arm-a-pinned-commit>` | `<arm-a-dtype-class>` | `<arm-a-committed-launch-line>` | `<arm-a-surface>` | supplies `S` for rule 1 | candidate/comparable | candidate pin or launch artifact unavailable |
| (b) | ceiling | `<arm-b-pinned-commit>` | `<arm-b-dtype-class>` | `<arm-b-committed-launch-line>` | `<arm-b-surface>` | supplies `C`, not `S` | ceiling | ceiling artifact or file-driven launch unavailable |
| (c) | informational | `<arm-c-pinned-commit>` | `<arm-c-dtype-class>` | `<arm-c-committed-launch-line>` | `<arm-c-surface>` | no kill-rule input | informational | informational endpoint or artifact unavailable |
| (d1) | kill-rule-2 candidate | `<arm-d1-pinned-commit>` | `<arm-d1-dtype-class>` | `<arm-d1-committed-launch-line>` | `<arm-d1-surface>` | supplies `S_d` for rule 2 | rule-2 candidate | frozen d1 pin or launch artifact unavailable |
| (d2) | kill-rule-2 candidate | `<arm-d2-pinned-commit>` | `<arm-d2-dtype-class>` | `<arm-d2-committed-launch-line>` | `<arm-d2-surface>` | supplies `S_d` for rule 2 | rule-2 candidate | frozen d2 pin or launch artifact unavailable |
| (d3) | kill-rule-2 candidate | `<arm-d3-pinned-commit>` | `<arm-d3-dtype-class>` | `<arm-d3-committed-launch-line>` | `<arm-d3-surface>` | supplies `S_d` for rule 2 | rule-2 candidate | frozen d3 pin or launch artifact unavailable |
| (e) | informational | `<arm-e-pinned-commit>` | `<arm-e-dtype-class>` | `<arm-e-committed-launch-line>` | `<arm-e-surface>` | no fraction of ceiling | different checkpoint, not comparable | separate checkpoint artifact unavailable |
| (f) | contributed | `<arm-f-pinned-commit>` | `<arm-f-dtype-class>` | `<arm-f-committed-launch-line>` | `<arm-f-surface>` | enters only through contribution path | not measured | contributor pin, launch line, or artifact unavailable |
| (g) | contributed | `<arm-g-pinned-commit>` | `<arm-g-dtype-class>` | `<arm-g-committed-launch-line>` | `<arm-g-surface>` | enters only through contribution path | not measured | contributor pin, launch line, or artifact unavailable |
| (h) | vendor | `<arm-h-pinned-commit>` | `<arm-h-dtype-class>` | `<arm-h-committed-launch-line>` | `<arm-h-surface>` | no unpinned kill-rule input | not measured | vendor access, pin, or launch line unavailable |

The `(d1)`, `(d2)`, and `(d3)` list is fixed at the pin date. A server added later enters through the contributed path and never substitutes for a member after a run. The `(f)`, `(g)`, and `(h)` panels initially contain `not measured` rows and retain their pre-declared reasons. The vendor panel is separate, labelled with its provenance, and is not merged into the comparable panel or used in a kill rule without a matching frozen identity tuple.

## 5. The kill arithmetic

The benchmark measures `S(die, checkpoint, chunk_ms, pacing_profile, X)` as the largest concurrency `N` at which, over a 180-second measurement window following a 60-second warm-up at that concurrency, all four criteria hold: p95 latency is at most `chunk_ms + X`; corpus WER is within `0.1` absolute of the batch-1 reference for the same `(checkpoint, chunk, corpus_id, dtype)`; no stream is refused, dropped, back-pressured into audio loss, or ended without a final transcript, while the load generator remains below 50 percent of its pinned CPU budget; and the GPU reports zero throttle events during the window. The `S` value is the minimum of three seeded repeats.

The fraction is `F = min-of-3 S / median-of-5 best-of-B C`. There is no rounding before comparison: `F = 0.699` kills under rule 1, while exactly `F = 0.7` is on the rule-1 boundary and does not kill under that strict-less-than rule. Exactly `S_d = 0.5 C` kills under rule 2; values below it do not, and the threshold is not rounded. A value above `1.1` triggers the maintainer rerun rule.

Rule 2 is evaluated separately for each `(d1)`, `(d2)`, and `(d3)` member at matched dtype. A measured `S_d >= 0.5 C` kills, but an unrun `(d)` member makes rule 2 not evaluable. Therefore the memo may not claim that no existing server reaches half the ceiling unless all three members were run or a dominance argument for the unrun member was written here before the run. Every ratio uses `S` and `C` from the same box session. Arm `(a)` is judged in the state it has on the run day, and its eager-step fraction, surface, and bucketing state are printed beside the verdict.

## 6. The ceiling

For each `B` in `{32, 128}`, the file-driven ceiling run uses `num_slots = B`, `warmup_steps = 1`, and `run_steps = 3`, and reports the median of five runs. The script's own throughput line and the harness wall timer must cross-check within 2 percent (`CEILING_RTFX_CROSS_CHECK = 0.02`). The file-driven run is given at least `4 × B` concurrent input streams, or `C` is not a ceiling: the canonical session set is far smaller than `4 × 128`, so without replication the batch is under-filled, lowers `C`, and raises `F`.

`C` is the real-time-stream equivalent of the toolkit's graphed file-driven pipeline on the same die. It is a host number as well as a GPU number, so its cpuset sensitivity is printed beside it and held fixed for the matched ratio.

## 7. Validity

The validity thresholds are `CLIENT_CPU_MAX_FRACTION_OF_CPUSET <= 0.5`, `PACING_SLIP_P99_MAX_MS <= 5.0`, `CLIENT_LIMITED_FLOOR_MULTIPLE = 2.0`, `STEAL_PCT_MAX <= 0.0`, `CGROUP_THROTTLED_DELTA_MAX = 0`, `NVML_THROTTLE_EVENTS_MAX = 0`, `FOREIGN_GPU_PROCESSES_MAX = 0`, and `GPU_PERSISTENCE_MODE_REQUIRED = true`. The client CPU threshold is a fraction of its pinned cpuset; the pacing threshold is in milliseconds; the steal, PSI, and floor values are host-pressure or floor checks, not server results.

### Not Yet Frozen

`PSI_CPU_SOME_MAX_PCT`, `PSI_CPU_FULL_MAX_PCT`, `AA_SPREAD_MAX_PCT`, and `NULL_FLOOR_TOLERANCE_PCT` are each `null` on the freeze date because no calibration produced a value. A threshold still `null` on the freeze date is filled from a calibration performed by the harness itself on the box, in a one-line commit made before the first run. A rung measured while any required threshold is `null` is invalid rather than passing.

**an invalid run is re-run, never footnoted.** Two consecutive invalid rungs abort the ladder as `host unfit` and release the box.

## 8. The host and the box

Only one arm runs at a time per box. Every row carries `box_id`. `C` and the null floor are re-measured in every box session. A fixed harness micro-benchmark supplies the single-thread score at the start and end of every box session, and the kill decision is read on that score. `F` is a box property: it is comparable across boxes only with its sensitivity rungs and is never ranked across boxes.

## 9. Schema version and harness tag

The environment-record schema version used by a run is `vb-results/2`. No field is added after the run to explain a result. One committed harness tag runs every arm in a table. If a bug is found mid-run, the harness tag is bumped and every arm is re-run; the fix is never applied to only one arm.

## 10. Comparability

Comparisons are allowed only across identical `(corpus_id, checkpoint@rev, chunk, dtype, profile, X)`. This identity is implemented in the renderer and in the kill calculator, not only in prose. Two rows are same-box comparable only when boot id, GPU UUID, cgroup CPU quota, effective cpuset, container digest, and harness tag are equal, and both null floors agree within the published floor tolerance.

Arm `(e)` renders only in a panel marked **different checkpoint, not comparable**. It has its own min-of-three, no fraction of ceiling, and one ladder rather than two. Different-box pairings are labelled and never ranked.

## 11. The constants test

`tests/harness/test_methodology_freeze.py` locates this document from the repository root, parses its single machine-read JSON block, and compares that mapping exactly with `verbatim_bench.constants.frozen_constants()`. It also checks the section order, null calibration names, latency retirement, die declarations, arm registry, and the no-drift invariants. The document and the instrument are therefore held equal mechanically rather than by intention.

## 12. Refresh cadence and staleness

The owned dies are class `self`, and the standing weekly matrix runs on them within the standing constraint of no more than 4 GPU-hours per week on owned hardware and no rented continuous integration; the GPU-hour meter counts usage per week, while a zero-cost staleness watchdog goes red when the newest invariance row is older than the published interval or a pin has drifted, making it the only watchdog that fires when nobody ran anything. A one-off row on a rented die, if it ever happens, is labelled `rented` and never enters the standing matrix.

## 13. The die the verdict is read on

On the NVIDIA RTX A6000 (`sm_86`), the `0.7` rule-1 and `0.5` rule-2 thresholds are the verdict thresholds.

A slower die lowers `C`, so `F` rises for every host-bound arm; an A6000 pass is therefore easier than a pass on a faster die and is not evidence that the host path reaches the faster die's ceiling. The pre-declared stricter follow-up is the owned B300 confirmation block at 160 and 560 ms, scheduled after day 21 rather than decided after the A6000 result. The `RTX 5090` and `B200` rows promised by the design documents are **not measured**.

<!-- frozen-constants -->
```json
{
  "AA_SPREAD_MAX_PCT": null,
  "CEILING_BATCH_SIZES": [
    32,
    128
  ],
  "CEILING_MEDIAN_OF": 5,
  "CEILING_MIN_INPUT_STREAMS_PER_BATCH_SLOT": 4,
  "CEILING_NUM_SLOTS_EQUALS_BATCH": true,
  "CEILING_RTFX_CROSS_CHECK": 0.02,
  "CEILING_RUN_STEPS": 3,
  "CEILING_WARMUP_STEPS": 1,
  "CGROUP_THROTTLED_DELTA_MAX": 0,
  "CHUNK_MODES_DAY21": [
    160,
    560
  ],
  "CLIENT_CPU_MAX_FRACTION_OF_CPUSET": 0.5,
  "CLIENT_LIMITED_FLOOR_MULTIPLE": 2.0,
  "CONFIRMATION_DIE": "B300",
  "FOREIGN_GPU_PROCESSES_MAX": 0,
  "FRAME_JITTER_MS": 10,
  "FRAME_MS": 20,
  "F_MAINTAINER_RERUN_ABOVE": 1.1,
  "GPU_PERSISTENCE_MODE_REQUIRED": true,
  "KILL_RULE_1_THRESHOLD": 0.7,
  "KILL_RULE_2_THRESHOLD": 0.5,
  "LADDER_INVALID_RUNGS_TO_ABORT": 2,
  "LADDER_MULTIPLIER": 1.15,
  "LADDER_N0_FRACTION_OF_C": 0.5,
  "LADDER_N0_WITHOUT_CEILING": 16,
  "LADDER_RESOLUTION": 0.02,
  "LATENCY_DELETED": "first_partial_after_send",
  "LATENCY_FALLBACK_IF_PRIMARY_UNAVAILABLE": "chunk_watermark",
  "LATENCY_PRIMARY": "word_emission",
  "LATENCY_SECONDARY": "chunk_watermark",
  "NOT_MEASURED_DIES": [
    "RTX 5090",
    "B200"
  ],
  "NULL_FLOOR_TOLERANCE_PCT": null,
  "NVML_THROTTLE_EVENTS_MAX": 0,
  "PACING_PROFILE": "uniform",
  "PACING_SLIP_P99_MAX_MS": 5.0,
  "PSI_CPU_FULL_MAX_PCT": null,
  "PSI_CPU_SOME_MAX_PCT": null,
  "SCHEMA_VERSION_FOR_RUN": "vb-results/2",
  "SEEDS": [
    20260914,
    20260915,
    20260916
  ],
  "SESSION_PROFILE": "m180",
  "STEAL_PCT_MAX": 0.0,
  "S_REPEATS": 3,
  "UNFROZEN_THRESHOLDS": [
    "PSI_CPU_SOME_MAX_PCT",
    "PSI_CPU_FULL_MAX_PCT",
    "AA_SPREAD_MAX_PCT",
    "NULL_FLOOR_TOLERANCE_PCT"
  ],
  "VERDICT_DIE": "NVIDIA RTX A6000",
  "VERDICT_SM": "8.6",
  "WARM_UP_CAP_S": 120,
  "WARM_UP_CONVERGENCE": 0.1,
  "WARM_UP_READING_S": 30,
  "WARM_UP_S": 60,
  "WER_WINDOW_ABSOLUTE": 0.1,
  "WINDOW_S": 180,
  "X_MS": 150
}
```
