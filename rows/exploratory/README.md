# `rows/exploratory/` — runs that are NOT rows

A row under `rows/` is a measurement the day-21, day-45, day-60 and day-90 gates are read against. It
must be canonical, comparable and reproducible. Everything in **this** directory failed at least one of
those and is kept because the failure is informative.

Nothing here may be cited as a benchmark result, quoted in a comparison, or used to argue a gate either
way. Each file says in its own record why it is not a row.

> **Both files below carry `canonical_window: true` and neither run was canonical.** The ladder's rung
> executor built its load spec without the frozen window and with the ramp forced to zero, so a rung
> was a single pass of one utterance per slot: fifteen seconds of wall clock and 106 latency samples at
> six streams, against a frozen window of 180 s and a warm-up of 60 s that was applied nowhere. The flag
> compared command-line arguments with constants and attested only that the operator typed no override.
> Four repeats of one identical rung give p95 values of 289.8, 320.3, 302.6 and 301.8 ms against a
> 310 ms threshold: three passes and a failure from the same inputs. Evidence section 25.
> The executor was fixed on 2026-09-11 by
> [DR-0005](../../docs/decisions/0005-the-ladder-runs-the-window-it-reports.md). These two files were
> produced before it and stay withdrawn: a fixed harness does not make an old run canonical.

| file | why it is not a row |
|---|---|
| `ladder-b300-bf16-graphed-2026-09-11.json` | The graph path, reporting `s: 0`, which describes the search and not the server: one seed was non-monotonic, failing at 6 streams and passing at 7, and a ladder search assumes monotonicity. The two clusters read from its p95 values in evidence section 24 were repeat noise; a single rung repeated on one machine covers the whole 30 ms gap that section attributed to the graph path. Sections 24 and 25. |
| `ladder-a6000-bf16-eager-2026-09-11.json` | The three seeds located the boundary at N=4, 6 and 11. The server sits on its 310 ms latency budget at every concurrency tested, so pass/fail is decided by noise rather than by load. The reported `s: 4` is the low end of a spread, not a capacity. Sections 23 and 25. |
