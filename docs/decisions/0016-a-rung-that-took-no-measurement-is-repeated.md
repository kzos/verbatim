# DR-0016: a rung that took no measurement is repeated, not scored

**Status:** accepted, 2026-09-15. Amends the rung procedure frozen by DR-0001.

## Context

The ladder searches one dimension, the stream count, and bisects on whether a rung
passed. A rung can end in three states, and until now only two of them were handled
honestly:

- **It measured, and passed or failed a criterion.** The search moves up or down.
- **It was invalid** — a throttled cgroup, a generator that missed its own schedule.
  The load it names is not the load that ran, so `execute` re-runs it, and a host that
  keeps producing invalid rungs aborts the whole ladder as unfit.
- **Its warm-up never settled**, so no measurement window opened. This was scored as a
  failed rung, and the bisection narrowed downward from it.

The third is the defect. `criteria.py` already says `UNSTABLE` "names a rung that
produced no measurement window" and is deliberately not one of the pass criteria. The
search read it as one anyway, so **"no window opened" became "the server cannot sustain
this many streams"**. Those are different statements and the second does not follow from
the first.

## What the data showed

Two canonical ladders on a B300, 2026-09-14, three seeds each, same server, same corpus.

| arm | S per seed | reported S |
|---|---|---|
| fixed | 20 / 24 / 46 | 20 |
| ragged | 19 / 36 / 59 | 19 |

A 2.3× and a 3.1× spread on an unchanged server, both ending `unstable`. It had been
filed as seed variance and named the project's top open technical item.

It is neither seed variance nor capacity. **In the fixed run every one of the six failing
rungs ended as `UNSTABLE` with `criteria_evaluated: []` and `p95: inf`.** Not one rung
failed on latency, word error rate, integrity or thermal. In the ragged run five of seven
were the same shape, one was invalid (`cgroup_throttled`), and exactly one was a real
criterion failure (`integrity:dropped` at n=20).

Two further readings rule out a capacity explanation.

**The percentile barely moves across the searched range.** At matched stream counts:
n=19 gives 196.7 / 189.2 / 190.9 ms across the three seeds, n=32 gives 195.0 / 194.8, and
n=37 gives 197.6 on a *passing* rung. At n=20 a rung with p95 189.3 ms failed while one at
189.8 ms passed. A capacity ceiling does not look like that.

**The warm-up reading series is dominated by single transients, not by a climbing load.**
The rule opens the window when two consecutive readings agree within
`WARM_UP_CONVERGENCE`, capped at `WARM_UP_CAP_S`, which allows three or four readings.
One stray reading spoils two consecutive pairs and can consume the whole budget. The
failing series:

```
fixed  n=23  [208, 331, 199]        fixed  n=25  [339, 202, 334]
fixed  n=21  [189, 215, 242]        fixed  n=49  [1814, 335, 290]
ragged n=37  [195, 340, 196]        ragged n=60  [239, 200, 252]
```

against series that converged and passed:

```
ragged n=20  [11268, 192, 188]      ragged n=59  [663, 212, 200]
fixed  n=23  [197, 252, 242]        fixed  n=46  [239, 201, 197]
```

The same stream count, n=23, appears in both lists. Whether a transient lands inside the
budget is not a property of the load: one seed failed to converge at 21 and 23 where
another passed 46 comfortably.

## Decision

**A rung whose warm-up never settled is run again, up to `LADDER_UNSTABLE_REPEATS = 2`
attempts. Converging on any attempt makes it a measured rung, scored on the attempt that
converged. Failing to converge at the same stream count on every attempt is the server's
answer and keeps the `unstable` criterion.**

Both attempts are appended to the rung record, in order, so the artifact shows the repeat
rather than hiding it. This is the treatment invalid rungs have always had, extended to
the other kind of rung that carries no number.

## What was rejected

**Treating `UNSTABLE` as an invalid rung.** It is the smallest change — `execute`
already re-runs invalid rungs — but it is wrong at the boundary it matters most. A server
genuinely past its ceiling has a latency that climbs and never settles, so every attempt
fails, and the ladder would abort the whole run with "host unfit". That reports a fit host
as unfit and loses the ceiling the run existed to find.

**Loosening `WARM_UP_CONVERGENCE` or raising `WARM_UP_CAP_S`.** This would make the
symptom rarer without deciding anything, and both are frozen numbers whose effect on
every published rung is hard to reason about. Repeating a rung changes only rungs that
took no measurement.

**Classifying by the shape of the reading series** — treating a monotone climb as a real
ceiling and a wandering series as a flake. It is the most informative option and it was
dropped as too clever: it infers the server's state from three numbers, where re-running
observes it. If the repeat turns out to be too expensive at high stream counts, this is
where to look next, and the series is already recorded on every rung for that purpose.

**Doing nothing and reporting the spread.** The spread was already being reported, as
seed variance, which is what sent the investigation in the wrong direction for two days.

## What this invalidates

**The two canonical B300 arms.** Every failing rung of the fixed arm, and five of seven on
the ragged arm, ended as `UNSTABLE` with no criterion evaluated, so neither S was a capacity
number. Both are re-run under this procedure and the rows replaced.

**Not the A6000, and an earlier draft of this record said otherwise.** Its
2026-09-13 ladder is in `rows/exploratory/ladder-a6000-bf16-eager-2026-09-13-all-criteria.json`
and every failing rung there failed on **`latency`**, with a real percentile: p95 303.8 and
307.4 ms passing at n=22, then 311.7 to 323.8 ms failing at 23 through 26, against a 310 ms
budget. That is a monotone latency ceiling and exactly what the ladder is for. This decision
does not reach it and does not withdraw it.

The A6000 run has a different problem, which this does not fix: seed 16 is non-monotonic
across the boundary, failing at 22 with 312.5 ms and passing at 23 with 305.0, so the search
reported `s: 0`. It also predates the warm-up reading series, so `warm_up_converged` is null
on every rung and nobody can now ask whether any of them settled. A re-run under the current
harness is worth having to put readings behind the number — but it is a confirmation, not a
correction.

It does not touch the invariance results, which do not use the ladder, or the ceiling
measurements, which do not either.
