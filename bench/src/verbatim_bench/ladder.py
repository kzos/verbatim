# SPDX-License-Identifier: Apache-2.0
"""The one-dimensional, validity-aware concurrency ladder.

Only the stream count is searched.  Invalid host observations are logged as
discarded rungs and never silently converted into a passing or failing server
measurement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

from verbatim_bench import constants
from verbatim_bench.env import GpuFacts, HostCounters


class Criterion(StrEnum):
    LATENCY = "latency"
    WER = "wer"
    INTEGRITY_REFUSED = "integrity:refused"
    INTEGRITY_DROPPED = "integrity:dropped"
    INTEGRITY_NO_FINAL = "integrity:no_final"
    THERMAL = "thermal"
    UNSTABLE = "unstable"
    INVALID_HOST = "invalid_host"


#: The four criteria the frozen methodology names for a rung to pass, written in the
#: vocabulary of `Criterion`. Its third criterion, integrity, is three values here,
#: because a stream can be refused, dropped, or end without a final transcript.
#: `UNSTABLE` and `INVALID_HOST` are deliberately absent: they name a rung that produced
#: no measurement window and a host that was unfit to measure on, neither of which is a
#: criterion the run is asked to establish.
RUNG_PASS_CRITERIA: Final = frozenset(
    {
        Criterion.LATENCY,
        Criterion.WER,
        Criterion.INTEGRITY_REFUSED,
        Criterion.INTEGRITY_DROPPED,
        Criterion.INTEGRITY_NO_FINAL,
        Criterion.THERMAL,
    }
)


class InvalidReason(StrEnum):
    CLIENT_CPU = "client_cpu"
    STEAL = "steal"
    PSI = "psi"
    PSI_THRESHOLD_UNFROZEN = "psi_threshold_unfrozen"
    CGROUP_THROTTLED = "cgroup_throttled"
    THROTTLE_EVENT = "throttle_event"
    PACING_SLIP = "pacing_slip"
    FOREIGN_GPU_PROCESS = "foreign_gpu_process"


@dataclass(frozen=True, slots=True)
class RungPlan:
    n: int
    seed: int
    warm_up_s: float = constants.WARM_UP_S
    window_s: float = constants.WINDOW_S

    @property
    def canonical_window(self) -> bool:
        """Whether this plan asked for the frozen warm-up and window lengths.

        A property of the request, not of any run. It can only take a rung's canonical
        flag away, never grant it: what the operator typed says nothing about what the
        load generator did, and the rung decides that from the load it executed.
        """
        return self.warm_up_s == constants.WARM_UP_S and self.window_s == constants.WINDOW_S


@dataclass(frozen=True, slots=True)
class Rung:
    """One rung, which reports a pass only for the criteria it actually established.

    `criteria_evaluated` lists what this rung established. `passed` is derived from it
    rather than stored, so a rung cannot be constructed that claims a criterion nobody
    evaluated: a pass needs every criterion in `RUNG_PASS_CRITERIA` evaluated and none
    of them failed.

    `first_failing_criterion` keeps its meaning unchanged. It names a criterion that was
    evaluated and failed, and stays `None` when nothing evaluated has failed, including
    when almost nothing was evaluated. It never reports an unevaluated criterion.

    `valid` and `invalid_reason` are untouched and keep naming host fitness. An
    unevaluated criterion is not a host problem and is never filed as one.

    `warm_up_s`, `window_s` and `wall_clock_s` describe the load this rung ran, not the
    load it was asked for, so a reader can check one against the others: a rung that
    claims a 180-second window and reports fifteen seconds of wall clock did not run it.
    `canonical_window` follows from the same three rather than from the command line.
    """

    n: int
    seed: int
    p95_ms: float
    wer_vs_batch1: float | None
    first_failing_criterion: Criterion | None
    valid: bool
    invalid_reason: InvalidReason | None
    warm_up_s: float
    sessions_refused: int
    sessions_dropped: int
    sessions_without_final: int
    criteria_evaluated: tuple[Criterion, ...] = ()
    canonical_window: bool = True
    window_s: float = constants.WINDOW_S
    wall_clock_s: float = 0.0

    @property
    def criteria_unevaluated(self) -> frozenset[Criterion]:
        """The pass criteria this rung never established."""
        return RUNG_PASS_CRITERIA - set(self.criteria_evaluated)

    @property
    def passed(self) -> bool:
        """True only when every pass criterion was evaluated and none of them failed."""
        return not self.criteria_unevaluated and self.first_failing_criterion is None


@dataclass(frozen=True, slots=True)
class Sensitivity:
    server_cpuset_halved_pass: bool | None
    client_cpuset_halved_pass: bool | None
    c_cpuset_halved_rtfx: float | None

    @property
    def host_bound(self) -> bool | None:
        """True when halving the server cpuset makes the rung fail."""
        if self.server_cpuset_halved_pass is None:
            return None
        return not self.server_cpuset_halved_pass

    @property
    def client_limited(self) -> bool | None:
        """True when halving the client cpuset makes the rung fail."""
        if self.client_cpuset_halved_pass is None:
            return None
        return not self.client_cpuset_halved_pass


@dataclass(frozen=True, slots=True)
class LadderResult:
    rungs: tuple[Rung, ...]
    s: int | None
    ending_criterion: Criterion | None
    s_per_seed: Mapping[int, int]
    aborted: bool
    abort_reason: str | None
    sensitivity: Sensitivity | None

    def to_json_list(self, *, include_window: bool = False) -> list[dict[str, Any]]:
        """Serialize rungs without losing invalid attempts or their reasons.

        The default shape is the rung object the row schema defines, which sets
        `additionalProperties: false`, so the executed window and wall clock are opt-in:
        `ladder.json` is this harness's own artifact and carries them, while a row
        document keeps exactly the rung the frozen schema names.
        """
        rows = [
            {
                "n": rung.n,
                "seed": rung.seed,
                "p95_ms": rung.p95_ms,
                "wer_vs_batch1": rung.wer_vs_batch1,
                "passed": rung.passed,
                "first_failing_criterion": (
                    rung.first_failing_criterion.value
                    if rung.first_failing_criterion is not None
                    else None
                ),
                "criteria_evaluated": [criterion.value for criterion in rung.criteria_evaluated],
                "valid": rung.valid,
                "invalid_reason": rung.invalid_reason.value if rung.invalid_reason else None,
                "warm_up_s": rung.warm_up_s,
                "sessions_refused": rung.sessions_refused,
                "sessions_dropped": rung.sessions_dropped,
                "sessions_without_final": rung.sessions_without_final,
                "canonical_window": rung.canonical_window,
            }
            for rung in self.rungs
        ]
        if include_window:
            for row, rung in zip(rows, self.rungs, strict=True):
                row["window_s"] = rung.window_s
                row["wall_clock_s"] = rung.wall_clock_s
        return rows


RungRunner = Callable[[RungPlan], Rung]


def n0_for(ceiling_c: float | None) -> int:
    """Return the frozen starting rung, using the no-ceiling fallback when needed."""
    if ceiling_c is None:
        return constants.LADDER_N0_WITHOUT_CEILING
    return max(1, round(constants.LADDER_N0_FRACTION_OF_C * ceiling_c))


def next_rung_up(n: int) -> int:
    """Return the strictly increasing frozen-multiplier rung."""
    return max(n + 1, math.ceil(n * constants.LADDER_MULTIPLIER))


def bisection_complete(lo: int, hi: int) -> bool:
    """Return whether the frozen relative resolution has been reached."""
    return hi - lo <= max(1, math.floor(constants.LADDER_RESOLUTION * lo))


def _passed(rung: Rung) -> bool:
    return rung.valid and rung.passed


def _narrow_canonical(rung: Rung, plan: RungPlan) -> Rung:
    """Let a non-canonical plan clear a rung's canonical flag, and nothing set it.

    The rung executor derives `canonical_window` from the load it ran. A plan that asked
    for overridden durations can only make that answer worse, so this narrows and never
    widens. The flag it replaced was the plan's answer alone, which made it true of every
    run nobody overrode, including every run that took no measurement at all.
    """
    if rung.canonical_window and not plan.canonical_window:
        return replace(rung, canonical_window=False)
    return rung


def run_ladder(
    run_rung: RungRunner,
    *,
    n0: int,
    seeds: Sequence[int] = constants.SEEDS,
    warm_up_s: float = constants.WARM_UP_S,
    window_s: float = constants.WINDOW_S,
) -> LadderResult:
    """Climb, bisect, and cheaply repeat the boundary for the remaining seeds.

    A remaining-seed repeat that passes the previously failing boundary resumes
    the climb.  This is important because a transient fail must not become a
    permanent ceiling.
    """
    if not seeds:
        raise ValueError("at least one ladder seed is required")
    if n0 < 1:
        raise ValueError("n0 must be positive")

    all_rungs: list[Rung] = []
    invalid_streak = 0

    def execute(plan: RungPlan) -> tuple[Rung | None, str | None]:
        nonlocal invalid_streak
        while True:
            rung = _narrow_canonical(run_rung(plan), plan)
            all_rungs.append(rung)
            if rung.valid:
                invalid_streak = 0
                return rung, None
            invalid_streak += 1
            if invalid_streak >= constants.LADDER_INVALID_RUNGS_TO_ABORT:
                return None, "host unfit"

    def search_seed(
        seed: int, start_n: int, initial_hi: int | None = None
    ) -> tuple[int, Rung | None, bool]:
        """Search one seed and return its highest pass and ending failing rung."""
        lo = 0
        hi = initial_hi
        ending: Rung | None = None
        if initial_hi is None:
            n = start_n
            while True:
                rung, abort = execute(RungPlan(n, seed, warm_up_s, window_s))
                if abort is not None:
                    return lo, None, True
                assert rung is not None
                if _passed(rung):
                    lo = n
                    n = next_rung_up(n)
                else:
                    hi = n
                    ending = rung
                    break
        else:
            start_pass = True
            if start_n > 0:
                rung, abort = execute(RungPlan(start_n, seed, warm_up_s, window_s))
                if abort is not None:
                    return lo, None, True
                assert rung is not None
                if _passed(rung):
                    lo = start_n
                else:
                    start_pass = False
                    ending = rung
            rung, abort = execute(RungPlan(initial_hi, seed, warm_up_s, window_s))
            if abort is not None:
                return lo, None, True
            assert rung is not None
            if _passed(rung) and start_pass:
                lo = initial_hi
                n = next_rung_up(initial_hi)
                while True:
                    rung, abort = execute(RungPlan(n, seed, warm_up_s, window_s))
                    if abort is not None:
                        return lo, None, True
                    assert rung is not None
                    if _passed(rung):
                        lo = n
                        n = next_rung_up(n)
                    else:
                        hi = n
                        ending = rung
                        break
            elif _passed(rung):
                return lo, ending, False
            else:
                hi = initial_hi
                if ending is None:
                    ending = rung

        assert hi is not None
        while not bisection_complete(lo, hi):
            mid = (lo + hi) // 2
            if mid <= lo:
                break
            rung, abort = execute(RungPlan(mid, seed, warm_up_s, window_s))
            if abort is not None:
                return lo, None, True
            assert rung is not None
            if _passed(rung):
                lo = mid
            else:
                hi = mid
                ending = rung
        return lo, ending, False

    first_seed = seeds[0]
    first_s, first_fail, aborted = search_seed(first_seed, n0)
    s_per_seed: dict[int, int] = {first_seed: first_s}
    if aborted:
        return LadderResult(
            tuple(all_rungs),
            None,
            Criterion.INVALID_HOST,
            s_per_seed,
            True,
            "host unfit",
            None,
        )

    ending_criterion = first_fail.first_failing_criterion if first_fail is not None else None
    boundary = first_fail.n if first_fail is not None else next_rung_up(first_s)

    for seed in seeds[1:]:
        # The cheap repeat always re-runs both sides of the original boundary.
        seed_s, seed_fail, aborted = search_seed(seed, first_s, boundary)
        s_per_seed[seed] = seed_s
        if aborted:
            return LadderResult(
                tuple(all_rungs),
                None,
                Criterion.INVALID_HOST,
                s_per_seed,
                True,
                "host unfit",
                None,
            )
        if seed_fail is not None:
            ending_criterion = seed_fail.first_failing_criterion or ending_criterion

    result_s = min(s_per_seed.values())
    return LadderResult(
        tuple(all_rungs),
        result_s,
        ending_criterion,
        s_per_seed,
        False,
        None,
        None,
    )


def run_sensitivity(
    run_rung: RungRunner,
    *,
    s: int,
    seed: int,
    halve_server_cpuset: Callable[[], AbstractContextManager[None]],
    halve_client_cpuset: Callable[[], AbstractContextManager[None]],
) -> Sensitivity:
    """Run the two causal cpuset controls at the already found ceiling."""
    with halve_server_cpuset():
        server_rung = run_rung(RungPlan(s, seed))
    with halve_client_cpuset():
        client_rung = run_rung(RungPlan(s, seed))
    return Sensitivity(
        server_cpuset_halved_pass=_passed(server_rung) if server_rung.valid else False,
        client_cpuset_halved_pass=_passed(client_rung) if client_rung.valid else False,
        c_cpuset_halved_rtfx=None,
    )


def check_monotone(rungs: Sequence[Rung]) -> tuple[str, ...]:
    """Report each same-seed fail followed by a higher passing rung."""
    messages: list[str] = []
    by_seed: dict[int, list[Rung]] = {}
    for rung in rungs:
        if rung.valid:
            by_seed.setdefault(rung.seed, []).append(rung)
    for seed, entries in by_seed.items():
        ordered = sorted(entries, key=lambda rung: rung.n)
        for lower in ordered:
            if lower.passed:
                continue
            for higher in ordered:
                if higher.n > lower.n and higher.passed:
                    messages.append(
                        f"seed {seed}: rung n={higher.n} passed above failed n={lower.n}"
                    )
    return tuple(messages)


def _cpu_fraction(value: float) -> float:
    """Accept the record's fraction form and tolerate a percentage-form probe."""
    return value / 100.0 if value > 1.0 else value


def rung_validity(
    counters: HostCounters, gpu: GpuFacts, pacing_slip_p99_ms: float
) -> InvalidReason | None:
    """Apply every frozen host validity threshold to one measurement window."""
    if constants.PSI_CPU_SOME_MAX_PCT is None or constants.PSI_CPU_FULL_MAX_PCT is None:
        return InvalidReason.PSI_THRESHOLD_UNFROZEN
    client_fraction = _cpu_fraction(counters.client_cpu_pct_of_cpuset)
    if client_fraction > constants.CLIENT_CPU_MAX_FRACTION_OF_CPUSET:
        return InvalidReason.CLIENT_CPU
    if counters.steal_pct > constants.STEAL_PCT_MAX:
        return InvalidReason.STEAL
    if (
        counters.psi_cpu_some_avg is None
        or counters.psi_cpu_full_avg is None
        or counters.psi_cpu_some_avg > constants.PSI_CPU_SOME_MAX_PCT
        or counters.psi_cpu_full_avg > constants.PSI_CPU_FULL_MAX_PCT
    ):
        return InvalidReason.PSI
    if (
        counters.cgroup_nr_throttled_delta > constants.CGROUP_THROTTLED_DELTA_MAX
        or counters.cgroup_throttled_usec_delta > constants.CGROUP_THROTTLED_DELTA_MAX
    ):
        return InvalidReason.CGROUP_THROTTLED
    if sum(gpu.throttle_reasons.values()) > constants.NVML_THROTTLE_EVENTS_MAX:
        return InvalidReason.THROTTLE_EVENT
    if len(gpu.compute_process_pids) > constants.FOREIGN_GPU_PROCESSES_MAX:
        return InvalidReason.FOREIGN_GPU_PROCESS
    if pacing_slip_p99_ms > constants.PACING_SLIP_P99_MAX_MS:
        return InvalidReason.PACING_SLIP
    return None
