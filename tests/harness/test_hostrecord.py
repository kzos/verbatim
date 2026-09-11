# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The host record of a window: sampled across it, windowed by the generator's own clock,
and the first thing that lets a rung evaluate thermal and, once every threshold is frozen,
pass."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
from test_env import _fake_host_tree, _write_pid_stat
from test_ladder import _counters
from test_rung_criteria import SEED, THRESHOLD_MS, WINDOW_CLOSE_S, WINDOW_OPEN_S, _run, _session
from verbatim_bench import constants
from verbatim_bench.env import FakeGpuProbe, GpuFacts, fake_gpu_facts
from verbatim_bench.hostrecord import (
    GpuSample,
    GpuWindow,
    HostWindow,
    WindowRecorder,
    run_load_recorded,
    summarise_gpu,
    throttling_reasons,
)
from verbatim_bench.ladder import (
    Criterion,
    InvalidReason,
    RungPlan,
    rung_from_run,
    rung_validity,
)
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.pace import ChunkMode, LoadSpec, WindowHooks, run_load

pytestmark = pytest.mark.cpu

IDLE = {
    "clocks_throttle_reason_gpu_idle": 1,
    "clocks_throttle_reason_applications_clocks_setting": 1,
    "clocks_throttle_reason_hw_slowdown": 0,
    "clocks_throttle_reason_sw_thermal_slowdown": 0,
}
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smi_q_x_synthetic.xml"


def _facts(**overrides: object) -> GpuFacts:
    base = {"throttle_reasons": dict(IDLE), "compute_process_pids": (), "persistence_mode": True}
    base.update(overrides)
    return fake_gpu_facts(**base)


def _quiet_window(*, open_s: float = WINDOW_OPEN_S, close_s: float = WINDOW_CLOSE_S) -> GpuWindow:
    stamps = [open_s + i for i in range(int(close_s - open_s) + 1)]
    return summarise_gpu(
        [GpuSample(t, _facts()) for t in stamps],
        open_s=open_s,
        close_s=close_s,
        interval_s=1.0,
        server_pid=None,
    )


def _host(gpu: GpuWindow | None = None, **counter_overrides: object) -> HostWindow:
    return HostWindow(
        counters=_counters(**counter_overrides),
        gpu=gpu if gpu is not None else _quiet_window(),
        open_s=WINDOW_OPEN_S,
        close_s=WINDOW_CLOSE_S,
    )


def _freeze_psi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the day the pressure thresholds are calibrated and committed."""
    monkeypatch.setattr(constants, "PSI_CPU_SOME_MAX_PCT", 100.0)
    monkeypatch.setattr(constants, "PSI_CPU_FULL_MAX_PCT", 100.0)


def test_only_throttling_reasons_are_events() -> None:
    """An idle GPU is not a throttled one, and a clock setting is not a slowdown."""
    assert throttling_reasons(IDLE) == {
        "clocks_throttle_reason_hw_slowdown": 0,
        "clocks_throttle_reason_sw_thermal_slowdown": 0,
    }
    hot = {**IDLE, "clocks_event_reason_sw_power_cap": 1, "clocks_throttle_reason_sync_boost": 1}
    active = throttling_reasons(hot)
    assert active["clocks_event_reason_sw_power_cap"] == 1
    assert "clocks_throttle_reason_sync_boost" not in active
    assert "clocks_throttle_reason_gpu_idle" not in active


def test_events_are_samples_inside_the_window_with_a_throttling_reason_active() -> None:
    def facts_at(t: float) -> GpuFacts:
        reasons = dict(IDLE)
        reasons["clocks_throttle_reason_hw_slowdown"] = 1 if t in (12.0, 13.0, 16.0) else 0
        return _facts(throttle_reasons=reasons)

    samples = [GpuSample(t, facts_at(t)) for t in (9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)]
    gpu = summarise_gpu(samples, open_s=10.0, close_s=15.0, interval_s=1.0, server_pid=None)
    assert gpu.samples == 6  # 9.0 and 16.0 fell outside the window
    assert gpu.throttle_events == 2
    assert gpu.throttle_reasons == {
        "clocks_throttle_reason_hw_slowdown": 2,
        "clocks_throttle_reason_sw_thermal_slowdown": 0,
    }
    assert gpu.covered is True
    assert gpu.max_gap_s == pytest.approx(1.0)


def test_a_window_sampled_only_at_its_start_is_not_covered() -> None:
    once = summarise_gpu(
        [GpuSample(10.0, _facts())], open_s=10.0, close_s=20.0, interval_s=1.0, server_pid=None
    )
    assert once.samples == 1
    assert once.covered is False
    assert once.max_gap_s == pytest.approx(10.0)
    never = summarise_gpu([], open_s=10.0, close_s=20.0, interval_s=1.0, server_pid=None)
    assert never.samples == 0
    assert never.covered is False
    assert never.max_gap_s is None


def test_foreign_pids_are_every_compute_process_but_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [GpuSample(t, _facts(compute_process_pids=(4242, 999))) for t in (10.0, 11.0)]
    gpu = summarise_gpu(samples, open_s=10.0, close_s=11.0, interval_s=1.0, server_pid=4242)
    assert gpu.compute_process_pids == (999,)
    _freeze_psi(monkeypatch)
    assert rung_validity(_counters(), gpu, 1.0) is InvalidReason.FOREIGN_GPU_PROCESS
    alone = summarise_gpu(samples, open_s=10.0, close_s=11.0, interval_s=1.0, server_pid=None)
    assert alone.compute_process_pids == (999, 4242)


def test_a_covered_quiet_window_lets_a_rung_evaluate_thermal_and_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first constructible pass: every criterion evaluated and met, over a window whose
    host record was sampled, on the day the pressure thresholds are frozen."""
    _freeze_psi(monkeypatch)
    host = _host()
    rung = rung_from_run(
        _run([_session(index) for index in range(4)]),
        plan=RungPlan(n=4, seed=SEED),
        threshold_ms=THRESHOLD_MS,
        batch1_wer=0.0,
        host=host,
    )
    assert rung.valid is True
    assert Criterion.THERMAL in rung.criteria_evaluated
    assert rung.criteria_evaluated[-1] is Criterion.THERMAL  # the document's order
    assert rung.criteria_unevaluated == frozenset()
    assert rung.first_failing_criterion is None
    assert rung.passed is True
    assert rung.host is host


def test_without_a_host_record_thermal_stays_unevaluated_and_nothing_passes() -> None:
    rung = rung_from_run(
        _run([_session(index) for index in range(4)]),
        plan=RungPlan(n=4, seed=SEED),
        threshold_ms=THRESHOLD_MS,
        batch1_wer=0.0,
    )
    assert Criterion.THERMAL not in rung.criteria_evaluated
    assert rung.criteria_unevaluated == frozenset({Criterion.THERMAL})
    assert rung.passed is False
    assert rung.host is None


def test_a_throttle_event_inside_the_window_makes_the_rung_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_psi(monkeypatch)
    reasons = {**IDLE, "clocks_throttle_reason_sw_thermal_slowdown": 1}
    stamps = [WINDOW_OPEN_S + i for i in range(int(constants.WINDOW_S) + 1)]
    samples = [
        GpuSample(t, _facts(throttle_reasons=reasons if t == stamps[3] else IDLE)) for t in stamps
    ]
    gpu = summarise_gpu(
        samples, open_s=WINDOW_OPEN_S, close_s=WINDOW_CLOSE_S, interval_s=1.0, server_pid=None
    )
    assert gpu.throttle_events == 1
    rung = rung_from_run(
        _run([_session(index) for index in range(4)]),
        plan=RungPlan(n=4, seed=SEED),
        threshold_ms=THRESHOLD_MS,
        batch1_wer=0.0,
        host=_host(gpu),
    )
    assert rung.valid is False
    assert rung.invalid_reason is InvalidReason.THROTTLE_EVENT
    assert rung.criteria_evaluated == ()
    assert rung.passed is False


def test_an_uncovered_window_leaves_thermal_unevaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_psi(monkeypatch)
    gpu = summarise_gpu(
        [GpuSample(WINDOW_OPEN_S, _facts())],
        open_s=WINDOW_OPEN_S,
        close_s=WINDOW_CLOSE_S,
        interval_s=1.0,
        server_pid=None,
    )
    rung = rung_from_run(
        _run([_session(index) for index in range(4)]),
        plan=RungPlan(n=4, seed=SEED),
        threshold_ms=THRESHOLD_MS,
        batch1_wer=0.0,
        host=_host(gpu),
    )
    assert rung.valid is True
    assert Criterion.THERMAL not in rung.criteria_evaluated
    assert rung.passed is False


def test_unfrozen_pressure_thresholds_keep_a_sampled_rung_invalid_and_recorded() -> None:
    """The frozen document: a rung measured while a required threshold is null is invalid
    rather than passing. The record is still kept, because it is the calibration's input."""
    assert constants.PSI_CPU_SOME_MAX_PCT is None
    host = _host()
    rung = rung_from_run(
        _run([_session(index) for index in range(4)]),
        plan=RungPlan(n=4, seed=SEED),
        threshold_ms=THRESHOLD_MS,
        batch1_wer=0.0,
        host=host,
    )
    assert rung.valid is False
    assert rung.invalid_reason is InvalidReason.PSI_THRESHOLD_UNFROZEN
    assert rung.host is host
    assert rung.host.to_json_dict()["gpu"]["throttle_events"] == 0


class _ScriptedProbe:
    """A GPU probe whose samples are scripted, cycling on the last one."""

    def __init__(self, facts: list[GpuFacts]) -> None:
        self._facts = facts
        self.calls = 0

    def count(self) -> int:
        return 1

    def facts(self, index: int) -> GpuFacts:
        assert index == 0
        self.calls += 1
        return self._facts[min(self.calls - 1, len(self._facts) - 1)]


class _FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_the_recorder_bounds_the_host_sampler_to_the_window_and_windows_the_gpu(
    tmp_path: Path,
) -> None:
    procfs = _fake_host_tree(tmp_path, steal=10, nr=2, usec=5000)
    cgroupfs = tmp_path / "cgroup"
    _write_pid_stat(procfs, 999999, 100, 50)
    clock = _FakeClock(5.0)
    hot = {**IDLE, "clocks_throttle_reason_sw_thermal_slowdown": 1}
    probe = _ScriptedProbe(
        [_facts(throttle_reasons=hot), _facts(), _facts(), _facts(), _facts(throttle_reasons=hot)]
    )
    recorder = WindowRecorder(
        gpu=probe,
        gpu_index=0,
        server_pid=None,
        client_pid=999999,
        interval_s=1.0,
        procfs=procfs,
        cgroupfs=cgroupfs,
        clock=clock,
    )
    assert recorder.finish() is None  # no window yet
    recorder.poll()  # t=5, before the window: a hot sample that must not count
    clock.now = 10.0
    recorder.on_window_open()
    for t in (10.0, 11.0, 12.0):
        clock.now = t
        recorder.poll()
    from test_env import _write

    _write(procfs / "stat", "cpu  200 0 200 1600 0 0 0 20 0 0\n")
    _write(cgroupfs / "cpu.stat", "nr_periods 200\nnr_throttled 5\nthrottled_usec 9000\n")
    _write_pid_stat(procfs, 999999, 100, 50)
    clock.now = 12.5
    recorder.on_window_close()
    clock.now = 13.0
    recorder.poll()  # after the window: a hot sample that must not count either
    host = recorder.finish()
    assert host is not None
    assert (host.open_s, host.close_s) == (10.0, 12.5)
    assert host.gpu.samples == 3
    assert host.gpu.throttle_events == 0
    assert host.gpu.covered is True
    assert host.counters.cgroup_nr_throttled_delta == 3
    assert host.counters.cgroup_throttled_usec_delta == 4000
    assert probe.calls == 5


def test_a_failing_probe_is_counted_not_raised(tmp_path: Path) -> None:
    class _Broken:
        def count(self) -> int:
            return 1

        def facts(self, index: int) -> GpuFacts:
            raise OSError("nvidia-smi: not found")

    procfs = _fake_host_tree(tmp_path)
    recorder = WindowRecorder(
        gpu=_Broken(), client_pid=None, procfs=procfs, cgroupfs=tmp_path / "cgroup"
    )
    recorder.on_window_open()
    recorder.poll()
    recorder.on_window_close()
    host = recorder.finish()
    assert host is not None
    assert recorder.probe_errors == 1
    assert host.gpu.samples == 0 and host.gpu.covered is False


async def test_run_load_fires_the_window_hooks_at_open_and_close(tmp_path: Path) -> None:
    from test_pace import make_manifest, make_wav

    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    stamps: dict[str, float] = {}
    hooks = WindowHooks(
        on_open=lambda: stamps.__setitem__("open", time.monotonic()),
        on_close=lambda: stamps.__setitem__("close", time.monotonic()),
    )
    async with NullServer(NullServerConfig()) as server:
        spec = LoadSpec(
            endpoint=server.endpoint,
            manifest=manifest,
            sessions=1,
            chunk=ChunkMode.parse(160),
            ramp_s=0.0,
            frame_ms=160,
            window_s=0.5,
            warm_up_s=None,
        )
        result = await run_load(spec, hooks=hooks)
    assert set(stamps) == {"open", "close"}
    assert result.window_open_s is not None and result.window_close_s is not None
    assert stamps["open"] == pytest.approx(result.window_open_s, abs=0.05)
    assert stamps["close"] == pytest.approx(result.window_close_s, abs=0.05)


async def test_run_load_recorded_returns_a_window_sampled_across_the_load(tmp_path: Path) -> None:
    from test_pace import make_manifest, make_wav

    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    probe = FakeGpuProbe((_facts(),))
    recorder = WindowRecorder(gpu=probe, client_pid=os.getpid(), interval_s=0.1)
    async with NullServer(NullServerConfig()) as server:
        spec = LoadSpec(
            endpoint=server.endpoint,
            manifest=manifest,
            sessions=1,
            chunk=ChunkMode.parse(160),
            ramp_s=0.0,
            frame_ms=160,
            window_s=0.6,
            warm_up_s=None,
        )
        result = await run_load_recorded(spec, recorder)
    host = recorder.finish()
    assert host is not None
    assert result.window_length_s == pytest.approx(0.6, abs=0.1)
    assert host.gpu.samples >= 3
    assert host.gpu.covered is True
    assert host.gpu.throttle_events == 0
    assert host.counters.client_cpu_pct_of_cpuset >= 0.0


async def test_the_ladder_records_the_host_window_when_asked(tmp_path: Path) -> None:
    """End to end: `--host-record` samples every rung's window from the saved nvidia-smi
    document, the record is in `ladder.json`, and while the pressure thresholds are
    unfrozen the rung is invalid for exactly that reason, as the document says."""
    from test_ladder import FAST_RUNG
    from test_ladder import SEED as LADDER_SEED
    from test_pace import make_manifest, make_wav
    from verbatim_bench.cli import main

    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
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
            str(LADDER_SEED),
            "--out",
            str(out_dir),
            "--host-record",
            "--server-pid",
            str(os.getpid()),
            "--from-smi-xml",
            str(FIXTURE),
            "--gpu-sample-interval-s",
            "0.1",
            *FAST_RUNG,
        ]
        rc = await asyncio.get_running_loop().run_in_executor(None, main, argv)
    del rc  # the outcome depends on whether the fast warm-up converged; the record does not
    payload = json.loads((out_dir / "ladder.json").read_text(encoding="utf-8"))
    assert payload["config"]["host_record"] is True
    assert payload["config"]["server_pid"] == os.getpid()
    assert payload["rungs"]
    # A rung whose warm-up never converged opened no window, so there was nothing to
    # sample and its record is rightly None; every rung that held a window has one.
    windowed = [r for r in payload["rungs"] if r["first_failing_criterion"] != "unstable"]
    if not windowed:
        # The fast warm-up converges by luck of the box; a run that never held a window
        # sampled nothing, which is right, and proves nothing about the record either way.
        pytest.skip("the fast rung's warm-up never converged, so no window was held")
    for rung in payload["rungs"]:
        if rung["first_failing_criterion"] == "unstable":
            assert rung["host"] is None
            continue
        assert rung["host"] is not None
        assert rung["host"]["gpu"]["samples"] >= 1
        assert rung["host"]["gpu"]["covered"] is True
        assert rung["host"]["gpu"]["throttle_events"] == 0
        assert rung["host"]["gpu"]["foreign_pids"] == []
        assert rung["host"]["counters"]["client_cpu_pct_of_cpuset"] >= 0.0
        assert rung["invalid_reason"] == InvalidReason.PSI_THRESHOLD_UNFROZEN.value
        assert rung["valid"] is False


def test_host_record_needs_a_server_pid(tmp_path: Path) -> None:
    from verbatim_bench.cli import main

    rc = main(
        [
            "ladder",
            "--endpoint",
            "ws://127.0.0.1:1/v1/stream",
            "--manifest",
            str(tmp_path / "missing.jsonl"),
            "--arm",
            "a",
            "--n0",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--host-record",
        ]
    )
    assert rc == 1
