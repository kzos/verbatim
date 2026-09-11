# DR-0004 — A rung records which criteria it evaluated, and the schema bumps to carry it

**Date:** 2026-09-11
**Status:** accepted, and it must be implemented before any ladder run is filed

## What happened

The ladder's rung executor evaluates one of the four criteria
[`benchmarks/METHODOLOGY.md`](../../benchmarks/METHODOLOGY.md) section 56 defines. It measures latency,
counts refusals, and writes the rest as literals: `wer_vs_batch1=None`, `sessions_dropped=0`,
`sessions_without_final=0`, `valid=True`, `invalid_reason=None`. Four of the eight `Criterion` values
and all eight `InvalidReason` values are defined in the code and unreachable from it.

So a rung that passes today means "latency was met and nothing was refused", while the document says it
means all four criteria held. The fix is not to implement the three missing criteria first. It is to
stop a rung reporting a pass for a criterion it never evaluated.

## The decision

**A criterion is a confirmed pass only when it was evaluated and met. A rung passes only when all four
are confirmed.** Anything else is not a pass.

To say that truthfully, a rung needs a third state. Evaluated-and-met and evaluated-and-failed are not
enough, because "not evaluated" is neither.

- A new rung field, `criteria_evaluated`, lists the criteria actually established by that rung.
- `passed` is true only when every criterion the document names appears there and was met.
- `first_failing_criterion` keeps its meaning exactly: a criterion that was evaluated and failed. It is
  `null` when nothing evaluated has failed, including when almost nothing was evaluated.
- `valid` and `invalid_reason` are untouched and keep naming host fitness, which is what the eight
  values in section 21 describe. An unevaluated criterion is not a host problem and must not be filed
  as one.

A reader of today's ladder output then sees `passed: false`, `first_failing_criterion: null`,
`criteria_evaluated: ["latency", "integrity:refused"]`, which is exactly what happened.

## The alternative that was rejected

Reporting `passed: false` with `first_failing_criterion` set to the first unevaluated criterion, so a
rung with good latency and no refusals reports a WER failure. That is cheaper, touches no schema, and
**replaces one false claim with another**: a reader sees a rung that failed on word error rate when
word error rate was never computed. It is the same defect this project keeps finding one level over, a
field reporting a measurement nobody took, and it would be introduced deliberately by the very commit
whose purpose is to stop that.

Forcing it into `valid: false` with a ninth `InvalidReason` was also rejected: that vocabulary names
host fitness, and an unevaluated criterion is not a host being unfit.

## Why the schema bumps, and why now

The rung object in `benchmarks/schema/row.schema.v2.json` sets `additionalProperties: false`, so
`criteria_evaluated` cannot be added without a new schema version. Per
[DR-0001](0001-freeze-the-benchmark-methodology.md) that requires a schema bump, a full re-run of every
affected row, and its own decision record. This is that record.

**The re-run costs nothing, because the set of affected rows is empty.** `rows/` contains no rows. The
two files under `rows/exploratory/` are marked in their own README as citable for nothing, and both are
already withdrawn for the separate defect that the ladder never ran the window it reported.

Section 84 of the methodology says **no field is added after the run to explain a result**. That rule is
the reason to act now rather than later. Added today, `criteria_evaluated` is part of the definition of
a rung before any row exists, and cannot be a post-hoc explanation of one. Added after the first row is
filed, it would be exactly the thing that sentence forbids, and the honest representation would then be
unavailable for the life of the project.

The freeze will never be cheaper to amend than it is today. That is the argument, and it expires.

## Consequences

- `SCHEMA_VERSION_FOR_RUN` becomes `vb-results/3`, in `constants.py` and in the methodology document's
  frozen block, in one commit, so the freeze guard never sees them disagree.
- `benchmarks/schema/row.schema.v3.json` carries the rung with `criteria_evaluated`.
- **Every ladder output this project has produced comes back `passed: false`.** That is the correct
  answer for all of them and the point of the change.
- The three criteria that are computable from data the harness already returns, finals from
  `finals_received`, pacing slip against `PACING_SLIP_P99_MAX_MS`, and WER from the final and reference
  text against a batch-1 reference, become the next work rather than the first work. A rung that refuses
  to overclaim is correct while incomplete; a rung that overclaims is not.
