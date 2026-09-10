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

Neither exists yet. This repository is private and its history was restarted on 2026-09-10, so the
freeze is currently **unproven** and the document says so. This record is the placeholder that becomes
the proof: when the repository goes public, the commit hash introducing the methodology gets written
below, and GitHub's own record of that commit's date is the external timestamp.

**Commit hash of the methodology freeze:** `<filled on the first public push>`
**Public remote:** `<filled on the first public push>`

Until both are filled, no comparative row published from this repository may claim its methodology was
frozen in advance. The claim and the proof land together or not at all.

## Consequences

- Changing a frozen constant requires a schema bump and a full re-run of every affected row, not an edit.
- A guard that cannot fail is worse than no guard. The freeze test was found inert on 2026-09-09, when a
  fabricated sentence claiming 412 sustained streams passed all twenty-four of its assertions; it now
  fails on three separate probes.
- Kill-gate thresholds are read against this document, so amending a condition after seeing a number is
  visible in the history of [`KILL.md`](../../KILL.md) rather than invisible in a definition.
