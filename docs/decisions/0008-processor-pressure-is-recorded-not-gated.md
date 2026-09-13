# DR-0008 — Processor pressure is recorded on every rung and gates nothing

**Date:** 2026-09-13
**Status:** accepted; the third decision on one gate, and the sequence is the point

## What happened

[DR-0007](0007-the-pressure-thresholds-calibrated-on-the-box.md) calibrated the two pressure
thresholds the frozen methodology left null, from the machine-wide `/proc/pressure/cpu` counters
over null-floor windows. The first capacity search with the host record on then aborted on pressure:
the null floor gave 0.40, real rungs at sixteen streams read 0.42, and the difference was the server
under test, absent from the baseline. The calibration was moved to the server under test at a
reference concurrency, and the reading to this session's own cgroup (`9da518a`), because the
machine-wide file is dominated by background this measurement neither causes nor controls: every
value ever measured on it fell between 0.22 and 0.42 percent while the idle box alone stalled 0.325.
The third capacity search then produced the numbers that settle the question.

| | this session's own cgroup | machine-wide |
|---|---|---|
| idle | 0.04 | 0.57 |
| six streams | 0.12 | 0.65 |
| sixteen streams | **0.23** | 0.64 |

The machine-wide reading barely moves with our load, because it is mostly background. The scoped
reading roughly doubles each time our load doubles, because it measures exactly our own work, which
is the variable a capacity search sweeps. A fixed ceiling on it is a ceiling on concurrency wearing
a validity check's clothes, which is why a threshold calibrated at six streams rejected the rung at
sixteen. Subtracting one reading from the other is not available either: pressure shares are not
additive across cgroups.

## The decision

**Processor pressure is recorded on every rung, scoped and machine-wide, and gates nothing.**
`rung_validity` no longer returns `psi` or `psi_threshold_unfrozen`; the two enum values stay so
that records written before this still name their reason. The two constants keep the last
calibrated values (0.13 and 0.12, scoped, against the server under test) as a record that nothing
reads as a gate; putting them back on the unfrozen list would invalidate every rung through the
other clause, which is the thing being removed. The other five host checks are unchanged and all
work: client processor use, steal, cgroup throttling, GPU throttle events, foreign compute processes.

## The two instruments withdrawn

1. **Machine-wide counters** (DR-0007): exact over the window, but dominated by background. A
   threshold on them encoded the box, not the run.
2. **Cgroup-scoped counters** (`9da518a`): exact and ours, but proportional to our own load. A
   threshold on them capped concurrency.

Both produced honest numbers; neither produced a validity gate. The readings from both are kept on
every rung as the input to whatever instrument replaces this, so the next attempt does not start
from nothing.

## The alternatives that were rejected

**Keep gating on the scoped reading, calibrated higher.** Rejected: any fixed ceiling on a quantity
that scales with N is a ceiling on N, and the calibration would have to be re-taken at every
concurrency a ladder might reach.

**Gate on the difference or the ratio of the two.** Rejected: pressure shares are not additive
across cgroups, so neither has a defined meaning.

**Gate on a relative rise within the run.** Rejected as a fourth instrument chosen without a
measurement; if one is wanted, the recorded readings are the data to design it from.

**Drop the readings with the gate.** Rejected: throwing them away would make the next attempt start
from nothing, and the pair is what shows the background being removed.

## Consequences

- A rung is no longer invalid for pressure, so a capacity search can reach a rung that produces a
  percentile; the two aborts of 2026-09-12 (`client_cpu` misread, then pressure) are both gone.
- `verbatim-bench calibrate-psi` remains, as the instrument that produces the recorded values.
- Section 7 of the methodology says the pair is recorded and not gated, and why, in one sentence.
