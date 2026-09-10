# DR-0001 — Freeze the benchmark methodology before the first comparative row

**Date:** 2026-09-10
**Status:** accepted

## Decision

Every definition, constant and threshold the day-21 comparative table will be read against is fixed in
[`benchmarks/METHODOLOGY.md`](../../benchmarks/METHODOLOGY.md) and committed **before** any comparative
row exists. `bench/src/verbatim_bench/constants.py` is the single source for those constants, and
`tests/harness/test_methodology_freeze.py` fails if the document and the code disagree.

## The alternative that was rejected

Write the methodology after the first runs, when the definitions can be chosen to fit what the harness
turned out to measure.

That is the normal way and it is why comparative tables are not trusted. Freezing turns every definition
from an output into an input. A wrong-but-frozen definition produces a wrong-but-honest table, which can
be corrected with a schema bump and a full re-run. An unfrozen definition is a knob, and after the first
run the author knows which way each knob moves the number relative to the kill thresholds. The latency
definition alone moves the measured stream count by about one chunk period.

## Why this record exists at all

The methodology document asks for proof that the freeze predates the numbers, and says a local commit
date is forgeable. It named two acceptable proofs: a push to a public remote, or an external timestamp
of the commit recorded here.

The repository went public on 2026-09-10, so the second proof now exists: the commit below carries a
date GitHub records independently of any local clock, and it predates every comparative row because
none exists yet.

**Commit hash of the methodology freeze:** `d3f5a6f8de70717ef84a0f499ce3e2245b8f4e95`
**Public remote:** https://github.com/kzos/verbatim, public since 2026-09-10

The freeze is therefore proven as of the date above. Any later change to a frozen constant requires a
schema bump and a full re-run, recorded as its own decision record, so that amending a definition after
seeing a number is visible in this directory rather than invisible in a diff.

## Consequences

- Changing a frozen constant requires a schema bump and a full re-run of every affected row, not an edit.
- A guard that cannot fail is worse than no guard. The freeze test was found inert on 2026-09-09, when a
  fabricated sentence claiming 412 sustained streams passed all twenty-four of its assertions.
  **Corrected 2026-09-10, by probing the guard rather than trusting this record.** What the guard does
  catch is constant drift: changing a value inside the frozen block reddens
  `test_document_constants_equal_the_code_constants`, and a constant present on one side only reddens
  its own test. What it still does **not** catch is the failure mode named above. Appending
  *"Verbatim sustains 412 concurrent streams at the 160 ms chunk mode on one A6000"* to the document
  passes all twenty-four assertions today. This record previously said that hole was closed. It is not,
  it is now in [`docs/BACKLOG.md`](../BACKLOG.md), and the sentence that claimed otherwise is left above
  rather than deleted so the overclaim is visible.
- Kill-gate thresholds are read against this document, so amending a condition after seeing a number is
  visible in the history of [`KILL.md`](../../KILL.md) rather than invisible in a definition.
