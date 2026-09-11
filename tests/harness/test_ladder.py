# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the validity-aware concurrency ladder."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest
from verbatim_bench import constants
from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.env import HostCounters, fake_gpu_facts
from verbatim_bench.ladder import (
    RUNG_PASS_CRITERIA,
    Criterion,
    InvalidReason,
    LadderResult,
    Rung,
    RungPlan,
    Sensitivity,
    bisection_complete,
    check_monotone,
    n0_for,
    next_rung_up,
    run_ladder,
    run_sensitivity,
    rung_validity,
)
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.results import percentile

pytestmark = pytest.mark.cpu

SEED = 20260914
EVERY_CRITERION = tuple(sorted(RUNG_PASS_CRITERIA, key=lambda criterion: criterion.value))


def _rung(
    n: int,
    seed: int = SEED,
    *,
    passed: bool,
    criterion: Criterion | None = None,
    valid: bool = True,
    invalid_reason: InvalidReason | None = None,
    warm_up_s: float = constants.WARM_UP_S,
    canonical: bool = True,
    criteria_evaluated: tuple[Criterion, ...] | None = None,
) -> Rung:
    """A rung double for the search tests, which need only passes and fails.

    `passed` is derived on `Rung` rather than stored, so the double has to declare what
    it evaluated. A rung that passed, and one that failed on a criterion the methodology
    names, evaluated all of them; a rung that is invalid, or that never converged,
    established nothing. The assertion below keeps the double from claiming a state the
    real type cannot hold. What a partially evaluating rung reports is the subject of the
    criteria tests, not of these.
    """
    if criteria_evaluated is None:
        named = criterion is not None and criterion in RUNG_PASS_CRITERIA
        criteria_evaluated = EVERY_CRITERION if (passed or named) else ()
    rung = Rung(
        n=n,
        seed=seed,
        p95_ms=10.0 if passed else 10000.0,
        wer_vs_batch1=None,
        first_failing_criterion=criterion if not passed else None,
        valid=valid,
        invalid_reason=invalid_reason,
        warm_up_s=warm_up_s,
        sessions_refused=1 if criterion is Criterion.INTEGRITY_REFUSED else 0,
        sessions_dropped=0,
        sessions_without_final=0,
        criteria_evaluated=criteria_evaluated,
        canonical_window=canonical,
    )
    assert rung.passed is passed
    return rung


def _threshold_runner(n_star: int, criterion: Criterion = Criterion.LATENCY):
    def _run(plan: RungPlan) -> Rung:
        if plan.n <= n_star:
            return _rung(plan.n, plan.seed, passed=True)
        return _rung(plan.n, plan.seed, passed=False, criterion=criterion)

    return _run


def _counters(**overrides) -> HostCounters:
    base: dict = {
        "steal_pct": 0.0,
        "psi_cpu_some_avg": 0.0,
        "psi_cpu_full_avg": 0.0,
        "cgroup_nr_throttled_delta": 0,
        "cgroup_throttled_usec_delta": 0,
        "loadavg_start": 0.5,
        "loadavg_end": 0.5,
        "server_cpu_pct_of_cpuset": 10.0,
        "server_hottest_thread_pct": 20.0,
        "server_threads": 4,
        "client_cpu_pct_of_cpuset": 0.1,
        "client_processes": 1,
        "server_cpu_s_per_stream_hour": 1.0,
    }
    base.update(overrides)
    return HostCounters(**base)


def test_n0_is_half_the_ceiling_rounded() -> None:
    assert n0_for(100.0) == 50
    assert n0_for(32.0) == 16
    assert n0_for(33.0) == round(0.5 * 33.0)


def test_n0_without_a_ceiling_is_the_pre_declared_constant() -> None:
    assert n0_for(None) == constants.LADDER_N0_WITHOUT_CEILING


def test_next_rung_up_is_strictly_greater() -> None:
    for n in (1, 2, 5, 16, 100):
        assert next_rung_up(n) > n
    assert next_rung_up(1) == 2


def test_bisection_stops_at_the_frozen_resolution() -> None:
    assert bisection_complete(100, 101) is True
    assert bisection_complete(100, 110) is False
    lo, hi = 100, 200
    assert bisection_complete(lo, hi) is False


def test_ladder_finds_the_analytic_stream_count_within_the_resolution() -> None:
    n_star = 40
    outcome = run_ladder(_threshold_runner(n_star), n0=n0_for(80.0), seeds=(SEED,))
    assert outcome.s is not None
    assert abs(outcome.s - n_star) <= max(1, int(constants.LADDER_RESOLUTION * n_star) + 1)


def test_ladder_labels_the_criterion_that_ended_it() -> None:
    outcome = run_ladder(_threshold_runner(10, Criterion.WER), n0=5, seeds=(SEED,))
    assert outcome.ending_criterion is Criterion.WER


def test_ladder_records_a_refusal_criterion_distinctly_from_a_latency_criterion() -> None:
    refused = run_ladder(_threshold_runner(8, Criterion.INTEGRITY_REFUSED), n0=4, seeds=(SEED,))
    latency = run_ladder(_threshold_runner(8, Criterion.LATENCY), n0=4, seeds=(SEED,))
    assert refused.ending_criterion is Criterion.INTEGRITY_REFUSED
    assert latency.ending_criterion is Criterion.LATENCY
    assert refused.ending_criterion != latency.ending_criterion


def test_a_rung_that_never_converges_fails_as_unstable_at_the_cap() -> None:
    def _run(plan: RungPlan) -> Rung:
        if plan.n <= 4:
            return _rung(plan.n, plan.seed, passed=True)
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.UNSTABLE)

    outcome = run_ladder(_run, n0=2, seeds=(SEED,))
    assert outcome.ending_criterion is Criterion.UNSTABLE


def test_repeats_run_only_the_rung_at_s_and_the_first_failing_rung_above() -> None:
    n_star = 10
    calls: list[tuple[int, int]] = []

    def _run(plan: RungPlan) -> Rung:
        calls.append((plan.n, plan.seed))
        if plan.n <= n_star:
            return _rung(plan.n, plan.seed, passed=True)
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)

    seeds = (SEED, SEED + 1)
    outcome = run_ladder(_run, n0=5, seeds=seeds)
    assert outcome.s is not None
    extra = [(n, s) for (n, s) in calls if s == seeds[1]]
    assert len(extra) >= 2
    assert {n for (n, _) in extra} >= {outcome.s}


def test_s_is_the_minimum_over_seeds() -> None:
    stars = {SEED: 20, SEED + 1: 14, SEED + 2: 18}

    def _run(plan: RungPlan) -> Rung:
        star = stars[plan.seed]
        if plan.n <= star:
            return _rung(plan.n, plan.seed, passed=True)
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)

    outcome = run_ladder(_run, n0=8, seeds=tuple(stars))
    assert outcome.s == 14
    assert outcome.s_per_seed[SEED] >= 14


def test_a_rerun_of_a_failing_rung_that_passes_continues_upward() -> None:
    seen: dict[int, int] = {}

    def _run(plan: RungPlan) -> Rung:
        count = seen.get(plan.n, 0)
        seen[plan.n] = count + 1
        if plan.n <= 10:
            return _rung(plan.n, plan.seed, passed=True)
        if plan.n == 12 and count == 0:
            return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)
        if plan.n <= 16:
            return _rung(plan.n, plan.seed, passed=True)
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)

    outcome = run_ladder(_run, n0=8, seeds=(SEED, SEED + 1))
    assert outcome.s is not None
    assert outcome.s > 10


def test_an_invalid_rung_is_discarded_logged_and_rerun() -> None:
    calls = {"n": 0}

    def _run(plan: RungPlan) -> Rung:
        calls["n"] += 1
        if calls["n"] == 1:
            return _rung(
                plan.n,
                plan.seed,
                passed=False,
                valid=False,
                invalid_reason=InvalidReason.STEAL,
            )
        if plan.n <= 8:
            return _rung(plan.n, plan.seed, passed=True)
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)

    outcome = run_ladder(_run, n0=4, seeds=(SEED,))
    assert outcome.s is not None
    invalid = [r for r in outcome.rungs if not r.valid]
    assert len(invalid) >= 1
    assert invalid[0].invalid_reason is InvalidReason.STEAL


def test_two_consecutive_invalid_rungs_abort_as_host_unfit() -> None:
    def _run(plan: RungPlan) -> Rung:
        return _rung(
            plan.n,
            plan.seed,
            passed=False,
            valid=False,
            invalid_reason=InvalidReason.STEAL,
        )

    outcome = run_ladder(_run, n0=4, seeds=(SEED,))
    assert outcome.aborted is True
    assert outcome.abort_reason == "host unfit"
    assert outcome.s is None


def test_an_unfrozen_pressure_threshold_makes_the_rung_invalid() -> None:
    assert constants.PSI_CPU_SOME_MAX_PCT is None
    reason = rung_validity(_counters(), fake_gpu_facts(), 1.0)
    assert reason is InvalidReason.PSI_THRESHOLD_UNFROZEN


def test_check_monotone_reports_a_pass_above_a_fail() -> None:
    rungs = [
        _rung(8, SEED, passed=False, criterion=Criterion.LATENCY),
        _rung(10, SEED, passed=True),
    ]
    messages = check_monotone(rungs)
    assert len(messages) == 1
    assert "10" in messages[0] and "8" in messages[0]


def test_sensitivity_rung_labels_a_host_bound_row() -> None:
    def _run(plan: RungPlan) -> Rung:
        return _rung(plan.n, plan.seed, passed=True)

    @contextlib.contextmanager
    def _fail():
        yield

    original = run_sensitivity

    def _run_fail(plan: RungPlan) -> Rung:
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)

    calls: list[str] = []

    @contextlib.contextmanager
    def _server_half():
        calls.append("server")
        yield

    @contextlib.contextmanager
    def _client_half():
        calls.append("client")
        yield

    # Simulate: server-halved fails, client-halved passes.
    def _mixed(plan: RungPlan) -> Rung:
        if calls and calls[-1] == "server":
            return _run_fail(plan)
        return _run(plan)

    sensitivity = original(
        _mixed,
        s=8,
        seed=SEED,
        halve_server_cpuset=_server_half,
        halve_client_cpuset=_client_half,
    )
    assert sensitivity.host_bound is True
    assert sensitivity.client_limited is False


def test_sensitivity_rung_labels_a_client_limited_row_as_invalid() -> None:
    @contextlib.contextmanager
    def _noop():
        yield

    def _run(plan: RungPlan) -> Rung:
        return _rung(plan.n, plan.seed, passed=False, criterion=Criterion.LATENCY)

    sensitivity = run_sensitivity(
        _run,
        s=8,
        seed=SEED,
        halve_server_cpuset=_noop,
        halve_client_cpuset=_noop,
    )
    assert sensitivity.client_limited is True
    assert isinstance(sensitivity, Sensitivity)


def test_rung_durations_default_to_the_frozen_window() -> None:
    plan = RungPlan(n=4, seed=SEED)
    assert plan.warm_up_s == constants.WARM_UP_S
    assert plan.window_s == constants.WINDOW_S
    assert plan.canonical_window is True


def test_overridden_durations_mark_the_run_non_canonical() -> None:
    plan = RungPlan(n=4, seed=SEED, warm_up_s=1.0, window_s=2.0)
    assert plan.canonical_window is False


def test_ladder_against_the_capacity_limited_null_server() -> None:
    capacity = 2
    chunk_ms = 160
    threshold_ms = chunk_ms + constants.X_MS
    pcm = bytes(int(0.5 * 16000 * 2))
    chunk = ChunkMode.parse(chunk_ms)

    async def _measure(n: int, seed: int) -> Rung:
        async with NullServer(
            NullServerConfig(capacity=capacity, overload_penalty_ms=800.0)
        ) as server:
            import asyncio

            async def _one(index: int):
                utterance = Utterance(
                    stream_id=f"cap-{index}",
                    audio_path=Path("cap.wav"),
                    duration_s=0.5,
                    text="capacity",
                )
                return await run_session(
                    server.endpoint,
                    session_id=f"s{index:04d}",
                    utterance=utterance,
                    pcm=pcm,
                    chunk=chunk,
                    start_delay_s=0.0,
                )

            sessions = await asyncio.gather(*(_one(i) for i in range(n)))
            samples = [v for s in sessions for v in s.partial_ms]
            p95 = percentile(samples, 95) if samples else float("inf")
            passed = p95 <= threshold_ms
            # A double again: this rung really does measure latency, but it declares the
            # whole criteria set because the assertion below is about the search finding
            # the null server's capacity. A rung declaring only what it measured could
            # not pass, which is the point of the criteria tests.
            return Rung(
                n=n,
                seed=seed,
                p95_ms=float(p95),
                wer_vs_batch1=None,
                first_failing_criterion=None if passed else Criterion.LATENCY,
                valid=True,
                invalid_reason=None,
                warm_up_s=0.2,
                sessions_refused=0,
                sessions_dropped=0,
                sessions_without_final=0,
                criteria_evaluated=EVERY_CRITERION,
                canonical_window=False,
            )

    def _run(plan: RungPlan) -> Rung:
        import asyncio as _asyncio

        return _asyncio.run(_measure(plan.n, plan.seed))

    outcome = run_ladder(_run, n0=1, seeds=(SEED,), warm_up_s=0.2, window_s=0.5)
    assert outcome.s == capacity
    assert outcome.ending_criterion is Criterion.LATENCY
    assert all(not rung.canonical_window for rung in outcome.rungs)


def test_ladder_uses_only_frozen_constants() -> None:
    import pathlib as _pathlib

    text = (
        _pathlib.Path(__file__).resolve().parents[2]
        / "bench"
        / "src"
        / "verbatim_bench"
        / "ladder.py"
    ).read_text()
    assert "0.7" not in text
    assert "1.15" not in text or "LADDER_MULTIPLIER" in text


def _partial_rung(n: int = 8, **overrides) -> Rung:
    """A rung shaped like the one the harness can actually produce today."""
    fields: dict = {
        "n": n,
        "seed": SEED,
        "p95_ms": 10.0,
        "wer_vs_batch1": None,
        "first_failing_criterion": None,
        "valid": True,
        "invalid_reason": None,
        "warm_up_s": float(constants.WARM_UP_S),
        "sessions_refused": 0,
        "sessions_dropped": 0,
        "sessions_without_final": 0,
        "criteria_evaluated": (Criterion.LATENCY, Criterion.INTEGRITY_REFUSED),
        "canonical_window": True,
    }
    fields.update(overrides)
    return Rung(**fields)


def test_rung_pass_criteria_are_the_ones_the_methodology_names() -> None:
    named = {
        Criterion.LATENCY,
        Criterion.WER,
        Criterion.INTEGRITY_REFUSED,
        Criterion.INTEGRITY_DROPPED,
        Criterion.INTEGRITY_NO_FINAL,
        Criterion.THERMAL,
    }
    assert named == RUNG_PASS_CRITERIA
    # Neither names a criterion of the run: one is a rung that produced no window, the
    # other a host that was unfit to measure on.
    assert Criterion.UNSTABLE not in RUNG_PASS_CRITERIA
    assert Criterion.INVALID_HOST not in RUNG_PASS_CRITERIA


def test_a_rung_that_evaluated_only_latency_and_refusals_does_not_pass() -> None:
    rung = _partial_rung()
    assert rung.passed is False
    # Nothing evaluated failed, so nothing is named. Naming an unevaluated criterion here
    # would report a failure for a measurement nobody took.
    assert rung.first_failing_criterion is None
    assert rung.criteria_unevaluated == {
        Criterion.WER,
        Criterion.INTEGRITY_DROPPED,
        Criterion.INTEGRITY_NO_FINAL,
        Criterion.THERMAL,
    }


def test_a_rung_passes_only_once_every_criterion_was_evaluated_and_met() -> None:
    rung = _partial_rung(criteria_evaluated=EVERY_CRITERION)
    assert rung.passed is True
    assert rung.criteria_unevaluated == frozenset()
    one_short = _partial_rung(criteria_evaluated=EVERY_CRITERION[:-1])
    assert one_short.passed is False
    assert one_short.first_failing_criterion is None


def test_an_evaluated_criterion_that_failed_is_still_named() -> None:
    rung = _partial_rung(
        p95_ms=10000.0,
        first_failing_criterion=Criterion.LATENCY,
        criteria_evaluated=EVERY_CRITERION,
    )
    assert rung.passed is False
    assert rung.first_failing_criterion is Criterion.LATENCY


def test_criteria_evaluated_round_trips_through_to_json_list() -> None:
    result = LadderResult(
        rungs=(_partial_rung(), _partial_rung(10, criteria_evaluated=EVERY_CRITERION)),
        s=None,
        ending_criterion=None,
        s_per_seed={},
        aborted=False,
        abort_reason=None,
        sensitivity=None,
    )
    rows = result.to_json_list()
    assert rows[0]["criteria_evaluated"] == ["latency", "integrity:refused"]
    assert rows[0]["passed"] is False
    assert rows[0]["first_failing_criterion"] is None
    assert rows[1]["criteria_evaluated"] == [criterion.value for criterion in EVERY_CRITERION]
    assert rows[1]["passed"] is True
    # The host-fitness vocabulary is untouched by an unevaluated criterion.
    assert rows[0]["valid"] is True
    assert rows[0]["invalid_reason"] is None


def test_a_ladder_of_partially_evaluated_rungs_reports_no_passing_rung() -> None:
    def _run(plan: RungPlan) -> Rung:
        return _partial_rung(plan.n)

    outcome = run_ladder(_run, n0=4, seeds=(SEED,))
    assert outcome.s == 0
    assert outcome.rungs
    assert not any(rung.passed for rung in outcome.rungs)
    assert outcome.ending_criterion is None


async def test_the_rung_executor_declares_only_the_criteria_it_established(tmp_path) -> None:
    """The defect DR-0004 names, checked at the executor that has it.

    The rung this harness can run today measures latency and counts refusals. Word error
    rate, dropped streams, missing finals and throttle events are written as literals by
    nobody: they are simply absent from what the rung claims, so the rung cannot report a
    pass, and it names no failing criterion because nothing it evaluated failed.
    """
    # Same-directory import, not `tests.harness.test_pace`; see the note in test_schema_v2.
    from test_pace import make_manifest, make_wav
    from verbatim_bench.cli import main

    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.5)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    out_dir = tmp_path / "ladder-out"
    async with NullServer(NullServerConfig()) as server:
        argv = [
            "ladder",
            "--endpoint",
            server.endpoint,
            "--manifest",
            str(manifest),
            "--arm",
            "a",
            "--n0",
            "1",
            "--seeds",
            str(SEED),
            "--out",
            str(out_dir),
        ]
        rc = await asyncio.get_running_loop().run_in_executor(None, main, argv)
    assert rc == 0
    payload = json.loads((out_dir / "ladder.json").read_text(encoding="utf-8"))
    assert payload["rungs"]
    for rung in payload["rungs"]:
        assert rung["criteria_evaluated"] == ["latency", "integrity:refused"]
        assert rung["passed"] is False
        assert rung["first_failing_criterion"] is None
        # Host fitness is a separate question and this host was fine.
        assert rung["valid"] is True
        assert rung["invalid_reason"] is None
    assert payload["s"] == 0
