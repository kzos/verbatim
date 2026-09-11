# DR-0007 — The pressure thresholds, calibrated on the box, and what the calibration found instead

**Date:** 2026-09-11
**Status:** accepted; it fills the last two thresholds that block a valid row

## What the freeze required

[`benchmarks/METHODOLOGY.md`](../../benchmarks/METHODOLOGY.md) section 7 leaves `PSI_CPU_SOME_MAX_PCT`
and `PSI_CPU_FULL_MAX_PCT` null on the freeze date, and says a threshold still null is filled **from a
calibration performed by the harness itself on the box**, in a one-line commit made before the first
run, with a rung measured while any required threshold is null invalid rather than passing.

So the commit is one line and nobody types the number in it. The value comes from an instrument.

## The decisions taken before it could run

The document specifies the calibration and not its workload, so three choices were this record's.

1. **Calibrate against the null floor, not an idle box.** The gate exists to reject a rung whose host
   was under pressure the measurement cannot survive, not to assert the machine was doing nothing. An
   idle-box threshold would sit below what a clean run of the generator itself produces, so the first
   honest run would fail its own validity gate. That would be a guard that cannot pass, built the same
   day four that could not fail were removed.

   **Amended 2026-09-12: this was right in its reasoning and one step short in its application, and the
   first capacity search failed on it.** The null floor excludes the server under test, which on a
   single box serving and measuring at once is part of the clean baseline rather than foreign to it.
   The calibrated 0.4 is the box's own 0.325 background stall plus about 0.07 from the generator, with
   nothing left for the server; a real run at the same concurrency reads 0.4209 and 0.4221 and is
   rejected by five percent. **Re-calibrate with the complete measurement running, the real server and
   the generator together, at a reference concurrency where the server is independently known healthy.**
   Six streams is that concurrency: it was measured at p95 193.7 ms against a 310 ms budget with 116 ms
   of margin. The circularity is real and is bounded by that reference rather than denied: a threshold
   taken with the server running absorbs whatever pressure that server causes, which is why the
   reference state has to be one that was shown healthy by a different instrument.
2. **Maximum over three concurrencies, not one.** `N = 16` is `LADDER_N0_WITHOUT_CEILING`, the only
   concurrency the document itself names; 32 and 128 are `CEILING_BATCH_SIZES`, and between them they
   cover every concurrency the method contemplates. Client-side pressure was expected to rise with
   concurrency, so a threshold from one N risked rejecting an honest run at another.
3. **No safety factor.** A factor would itself be an unfrozen constant chosen by hand, which is the
   thing being avoided. The threshold is the observed maximum of clean windows, rounded up to 0.01.
4. **Measure the window's own stall time, not a decaying average of it.** The first two attempts read
   `/proc/pressure/cpu`'s `avg60`, which is an exponentially decaying average with a sixty-second time
   constant, so it reports a window plus a smeared memory of whatever preceded it. The third attempt
   read 0.46 against the first attempt's 0.01, from a test suite that had finished seconds earlier; the
   quiet observation recorded 0.27 and falling, which is the decay itself. The gate now reads the
   `total` counters, monotonic microseconds of stall, and takes the delta across the window over the
   window's length, so the reading is the window's own pressure and nothing else.

   **This was not only a calibration problem.** A rung's window opens after the ramp and the warm-up,
   about two time constants, so a transient from before the run would still have read six times the
   threshold when the window opened. **A rung could have been invalidated by pressure that ended before
   it started**, with nothing in the record to show the cause, which is the same shape as every defect
   this project spent the day removing: a gate firing on something other than what it names. The
   five-minute quiet wait that the decaying average required is gone with it; the observation stays for
   steal, throttling, load average and foreign processes, which are the things it can actually settle.

## What it produced

`PSI_CPU_SOME_MAX_PCT = 0.4`, `PSI_CPU_FULL_MAX_PCT = 0.0`, from run 5 on `0cc5f3a` with a clean tree,
recorded in `calibration-psi-2026-09-11-run5-0cc5f3a.json`.

| N | seed | clean | counter `some` % | counter `full` % | `avg60` `some` max | client CPU % | slip p99 |
|---|---|---|---|---|---|---|---|
| 16 | 20260914 | yes | 0.392 | 0.0 | 0.05 | 1.06 | 2.68 ms |
| 16 | 20260915 | yes | 0.398 | 0.0 | 0.00 | 1.07 | 2.80 ms |
| 16 | 20260916 | yes | 0.398 | 0.0 | 0.00 | 1.07 | 2.95 ms |
| 32 | 20260914 | **no** | 0.379 | 0.0 | 0.00 | 1.21 | **6.18 ms** |
| 32 | 20260915 | **no** | 0.375 | 0.0 | 0.00 | 1.21 | **6.99 ms** |
| 32 | 20260916 | **no** | 0.370 | 0.0 | 0.00 | 1.21 | **6.42 ms** |
| 128 | 20260914 | **no** | 0.237 | 0.0 | 0.00 | 1.90 | **37.28 ms** |
| 128 | 20260915 | **no** | 0.221 | 0.0 | 0.00 | 1.92 | **44.61 ms** |
| 128 | 20260916 | **no** | 0.216 | 0.0 | 0.00 | 1.92 | **43.22 ms** |

No session failed anywhere, and full stall is exactly zero in every window and in the idle observation.
The quiet observation read steal 0.0, no cgroup throttling, a load average of 0.90 against a 48-CPU
cpuset, no foreign compute process, and its own counter pressure at `some` 0.325 with `full` 0.0.

**Three things in that table are worth more than the two values.**

**The two estimators disagree by an order of magnitude and only one of them is right.** The counter
reads about 0.4 percent of every window stalled, while the decaying average sampled inside the same
windows reads 0.00 to 0.05. The idle box's own counter pressure is 0.325, so a small background stall
is present on this machine even when nothing of ours runs, and the decaying average does not show it at
all. The threshold is calibrated on the instrument the gate reads, so `0.4` admits the floor and the
idle box alike. Both fields stay in the records for now; **do not read the two numbers against each
other.**

**Pressure falls with concurrency rather than rising**: 0.39 at 16 streams, 0.37 at 32, 0.22 at 128. The
three-concurrency rule bought out a risk that runs in the opposite direction from the one assumed.

**32 streams did not pace cleanly in a single window**, at a slip p99 of 6.2 to 7.0 ms against the
frozen 5.0, so the threshold rests on the three windows at 16 alone. An earlier attempt had two of three
clean at 32, so the generator is marginal there rather than reliably short. That sharpens the capacity
finding: **the first ladder on this box will meet the pacing gate at or before 32 streams.**

## The reason that reasoning was half wrong, recorded because it was

**The risk that justified three concurrencies does not exist.** Pressure sits at the floor across an
eightfold range and full stall is exactly zero in every window. One concurrency would have produced the
same two numbers, and the rule bought nothing it was chosen to buy.

**It bought something else.** Running the range is what showed that the load generator, not the server,
is what stops first on this box: one event-loop process cannot hold a 20 ms send schedule for 128
streams, at a pacing slip p99 of 33 to 38 ms against a frozen 5.0, with nothing failing and pressure
never moving. A capacity search here will be invalidated by its own instrument between 32 and 128
streams, while the server at six streams sat 116 ms inside a 310 ms budget.

That finding is worth more than the thresholds the run was commissioned to produce, and it was
available only because the rule was wrong about why it was needed.

## Consequences

- A zero threshold on full pressure is brittle by construction: one stalled sample invalidates a run. It
  is calibrated rather than chosen, it has precedent in `STEAL_PCT_MAX` and `CGROUP_THROTTLED_DELTA_MAX`,
  and it demonstrably passes. **The first time a clean-looking run is invalidated on full pressure
  alone, look at the box and not at the number.**
- An excluded 128-stream window read above the threshold the clean windows produced. A concurrency the
  generator cannot pace cannot yield a valid rung, so no run that could pass will meet that pressure.
- If a ladder ever climbs past 128, a clean run could exceed a threshold calibrated below it. That
  failure is visible, because rungs go invalid with the pressure reason rather than silently passing.
  Re-calibrate higher and record why; never raise the number by hand.
- **Whether the generator needs splitting across processes is a question for the first ladder run and
  not for this record.** Read which gate binds first. If latency fails before pacing goes invalid, the
  server's ceiling is inside the instrument's range and a second process buys nothing.
- **The threshold encodes this box's background stall, which is most of it.** The idle counter pressure
  is 0.325 and the threshold is 0.4, so what was calibrated is the floor plus the generator's own small
  contribution. A different box with a higher background would fail this gate immediately and would be
  right to: the methodology re-measures the floor every box session, and the answer there is to
  re-calibrate on that box and record it, never to raise the number to fit.
- **Three earlier attempts are retired with their reasons rather than deleted.** Run 1 read 0.01 on a
  dirty tree with the decaying average; run 2 read 0.46 because a test suite had ended seconds earlier
  and the decay was still in the estimator; runs 3 and 4 were stopped before landing when the estimator
  itself was replaced. The sequence is the record: a threshold that changed by a factor of forty across
  attempts is worth nothing until the instrument producing it is right.
