# DR-0005 — The ladder runs the measurement window it reports

**Date:** 2026-09-11
**Status:** accepted, and it must be implemented before any ladder run is filed

## What happened

`bench/src/verbatim_bench/cli.py::_make_rung` built its `LoadSpec` with `ramp_s=0.0` and never passed
`window_s`. In `run_load`, `window_s is None` takes the branch that runs each slot exactly once with no
closed-loop replacement, so a rung was one pass of one utterance down each slot and then stopped.
Measured on 2026-09-11 against the live A6000 bfloat16 server, a six-stream rung was 15.1 s of wall
clock and 106 latency samples, against a frozen window of 180 s. Run with the window applied by hand,
the same rung was 188.6 s and 4,798 samples. Four repeats of one identical rung gave p95 values of
289.8, 320.3, 302.6 and 301.8 ms against a 310 ms threshold: three passes and a failure from the same
inputs.

`warm_up_s` reached no measurement anywhere. Every use of it outside `constants.py` stored, stamped,
compared or serialised it. `Criterion.UNSTABLE` was defined and unreachable.

Every rung was nonetheless stamped `canonical_window: true`, from
`args.warm_up_s == WARM_UP_S and args.window_s == WINDOW_S` — a comparison of command-line arguments
with constants, true of every run nobody overrode, including every run that took no measurement at all.

## The decision

**A rung runs the three phases the frozen document names, in order, and a sample counts only in the
phase it arrived in.**

- The **ramp** ends when every one of the `N` slots has opened its first session. The load generator
  learns this from the server's own session acknowledgement rather than from a timer, so the warm-up
  clock starts with all `N` streams live, which is what section 56 requires.
- The **warm-up at N** takes readings of `WARM_UP_READING_S` and opens the window when two consecutive
  readings agree within `WARM_UP_CONVERGENCE`, never before `WARM_UP_S` and never after
  `WARM_UP_CAP_S`. Reaching the cap without converging fails the rung as `Criterion.UNSTABLE`, which
  makes that value reachable for the first time.
- The **measurement window** runs for `WINDOW_S` with closed-loop replacement holding concurrency at
  `N`. Only samples received inside it reach the rung's percentile.

**A reading is the p95 of the partial latencies received during it** — the same statistic the rung
reports. The document says two readings of the reading length must agree within ten percent; it does
not say of what. Choosing the rung's own statistic means the warm-up ends when the number being
measured has settled, rather than when some other number has.

**`canonical_window` is a property of the run.** It is false whenever the load that executed did not
have the frozen warm-up and window: no window opened, the warm-up never settled, the window opened
before the frozen warm-up had elapsed, the window closed before the frozen length had elapsed, or any
of the five durations was overridden. The argument comparison survives only as a warning printed
before the run, and never stamps anything.

## What this forced

The generator has to read latency **while sessions are still open**. A canonical session is 180 s and
a canonical warm-up is 60 s, so at the moment the window would open, no session has finished and there
is nothing to take a reading of. The watermark match was therefore rewritten as a streaming matcher,
`client.WatermarkMatcher`, which chunks and partials are offered to in arrival order;
`match_partials_by_watermark` is now written in terms of it, so there is one definition and the
methodology's named function still means what section 2 says. Each session carries `partial_recv_s`
beside `partial_ms`, the monotonic moment each sample was matched, which is what phase attribution is
read from.

`Rung` gains `window_s` and `wall_clock_s`: the load it ran, not the load it was asked for, so a reader
can check one against the other. A rung that claims a 180-second window and reports fifteen seconds of
wall clock did not run it.

## The alternatives that were rejected

**Timing the ramp instead of observing it.** Waiting `ramp_s` and declaring the streams live is one
line shorter and silently wrong whenever a connection is slow, which is exactly when the rung matters.

**Defining the reading over completed sessions.** It needs no streaming matcher and produces no
readings at all for the canonical rung, because no 180-second session finishes inside a 60-second
warm-up. Every canonical rung would fail as unstable.

**Bumping the schema to carry the window on the rung.** `row.schema.v3.json` sets
`additionalProperties: false` on the rung object, so `window_s` and `wall_clock_s` cannot appear in a
row's ladder without a bump, and DR-0001 makes that a full re-run with its own record. They are opt-in
instead: `LadderResult.to_json_list()` emits the frozen shape, `ladder.json` asks for the window, and a
test holds the default inside the schema's property set. The schema bumps when there is a row to bump
it for.

## Consequences

- `--ramp-s`, `--warm-up-reading-s`, `--warm-up-convergence` and `--warm-up-cap-s` join `--warm-up-s`
  and `--window-s` on `verbatim-bench ladder`. Every one of them makes the run non-canonical when it
  differs from the frozen value, and each is what lets the test suite run a three-phase rung in
  seconds rather than four minutes.
- A canonical rung now costs at least `WARM_UP_S + WINDOW_S` of wall clock, and a ladder is a multiple
  of that. The 4 GPU-hours-per-week standing constraint in section 12 is now a real budget for the
  ladder rather than a formality.
- `verbatim-bench run` is unchanged: with no window it runs each slot once, which is a smoke run and
  was never a rung.
- The two files under `rows/exploratory/` stay withdrawn. They were produced by the executor this
  record replaces, and nothing here makes them citable.
