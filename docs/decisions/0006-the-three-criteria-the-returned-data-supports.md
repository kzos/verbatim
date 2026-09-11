# DR-0006 — The three criteria the returned data supports, and the one it does not

**Date:** 2026-09-11
**Status:** accepted and implemented

## What happened

[DR-0004](0004-a-rung-records-which-criteria-it-evaluated.md) stopped a rung reporting a pass for a
criterion nobody evaluated, and left the checks themselves for this record. Of the four criteria
[`benchmarks/METHODOLOGY.md`](../../benchmarks/METHODOLOGY.md) section 56 names, the harness evaluated
latency and counted refusals. Word error rate, dropped streams, streams that ended without a final, and
zero GPU throttle events were written by nobody and claimed by nobody.

Three of those four need no collection that does not exist. `SessionResult` already carries
`pacing_slip_ms` for the validity check that gates all of them, `finals_received` for missing finals,
`server_session_id` to tell a stream the server never admitted from one it admitted and lost, and
`final_text`/`reference_text` for word error rate — the last two only meaningful since the load client
began reading every final a stream carries rather than returning on the first.

## The decision

**A rung evaluates every criterion its own load result supports, and says so; it evaluates nothing
else, and says that too.** The reduction from a load result to a rung moved out of the command into
`ladder.rung_from_run`, a pure function, so what a rung claims can be tested against a run built by
hand rather than against four minutes of wall clock.

- **Pacing slip is a validity threshold, not a criterion.** A rung whose pooled slip p99 exceeds
  `PACING_SLIP_P99_MAX_MS` is `valid=false` with `invalid_reason: pacing_slip`. It establishes no
  criterion and reports no percentile, because the load it would have been taken over is not the load
  the rung names. The threshold has one definition, `ladder.pacing_slip_validity`, which
  `rung_validity` defers to, so a rung executed from its load alone applies the same tolerance as one
  executed against a full host record. **This fires on the configuration the project ships**, which is
  the point: measured against the null server on 2026-09-11, four sessions at the frozen 20 ms framing
  pool to a slip p99 of 10.8 ms over 800 frames, against a tolerance of 5.0.

- **Integrity is three counts, over every session the rung ran.** Refused is a stream that failed
  before the server ever acknowledged it; dropped is one that failed after; without-final is
  `finals_received == 0`. They are not disjoint and are not meant to be. They are counted over the
  warm-up as well as the window because a stream outcome carries no receive timestamp to window it by,
  and because that direction is conservative: it is a superset of the window's failures, so it can
  only refuse a rung the window-only rule would also have refused.

- **Word error rate is evaluated only against a supplied batch-1 reference.** The reference is a
  five-key JSON document naming the checkpoint, chunk, corpus and dtype it was measured at, and the
  number. Two of those coordinates the ladder reads from the run; the other two are declared on the
  command line, because the ladder speaks to a server over a socket and never learns what it loaded.
  Without a reference the criterion is simply left out of `criteria_evaluated`. A reference that was
  supplied and cannot be used is a usage error and the ladder refuses to start.

- **Thermal stays unevaluated, so no rung can pass.** Zero GPU throttle events over the window needs a
  collection nothing in the harness performs. It is absent from every `criteria_evaluated`, and
  `Rung.passed` therefore returns false for every rung this harness can produce. That is the correct
  answer and the thing DR-0004 exists to protect.

The normaliser word error rate is computed with is **this harness's own**, written out in
`wer.normalise`, and is **not** the frozen `<normaliser-package-version>` the methodology names, which
is still an unfilled placeholder because no package is pinned. Both sides of the comparison must come
from it, which is what the reference document's coordinates are for.

## The alternatives that were rejected

**Pinning a normaliser package now, to fill the placeholder.** It would make the criterion citable
against outside numbers immediately. It was rejected because picking one in passing, inside the commit
that implements a criterion, is exactly how an unexamined choice becomes a freeze: contraction
expansion, number spelling and dialect mapping each move a WER by more than some of the differences
this benchmark is meant to resolve. The placeholder stays unfilled and visible until it is chosen
deliberately.

**Taking the batch-1 number from the command line as a bare float.** Simpler by one file. Rejected
because the criterion is defined for one `(checkpoint, chunk, corpus_id, dtype)` and is meaningless
across any of them, and a bare float carries none of that: the first mismatched comparison would be
undetectable and would look exactly like a real result.

**Skipping word error rate silently when the supplied reference does not describe the run.** Rejected
for the same reason DR-0004 rejected naming an unevaluated criterion as failed: an operator who
configured a comparison and got a run with no WER in it would be unable to tell that from having
configured nothing at all.

**Evaluating thermal from whatever `env.GpuFacts` returns at some moment near the run.** Rejected. The
criterion is zero throttle events *during the window*, and a reading taken outside it is a different
measurement wearing its name. `THERMAL` is left unreachable until something samples across the window.

## Consequences

- **Every canonical rung comes back invalid today**, because the load generator misses its own frozen
  pacing tolerance. Two consecutive invalid rungs abort the ladder as host unfit, so the ladder now
  produces no capacity at all rather than a capacity nobody should have believed. `docs/BACKLOG.md`
  carries the open finding that the excess is the generator's own seeded frame jitter, which shapes no
  send and only grades one.
- No schema change: `criteria_evaluated`, `wer_vs_batch1`, `sessions_dropped` and
  `sessions_without_final` are all already in `vb-results/3`. Nothing was added after the run to
  explain a result.
- The ladder gained `--frame-ms`, which it could not express at all and silently took the default for,
  and `--wer-batch1`, `--checkpoint` and `--dtype`. `ladder.json` records the framing that ran and the
  reference that was in force, or null.
