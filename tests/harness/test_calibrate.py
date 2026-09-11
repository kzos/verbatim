# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The pressure-threshold calibration: the gate's own estimator, the null floor, a quiet
box, and provenance beside the two numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_env import _fake_box_tree, _fake_host_tree, _write, _write_pid_stat
from verbatim_bench import constants
from verbatim_bench.calibrate import (
    PRECISION_PCT,
    QUIET_OBSERVATION_S,
    CalibrationRefusal,
    WindowReading,
    calibrate,
    ceil_to_precision,
    observe_quiet,
    thresholds_from,
)
from verbatim_bench.env import FakeGpuProbe, fake_gpu_facts

pytestmark = pytest.mark.cpu

PRESSURE = (
    "some avg10=0.00 avg60=0.12 avg300=0.05 total=1\n"
    "full avg10=0.00 avg60=0.03 avg300=0.01 total=1\n"
)


def _window(
    some: float | None, full: float | None, seed: int = 1, *, n: int = 1, clean: bool = True
) -> WindowReading:
    return WindowReading(
        n=n,
        seed=seed,
        window_open_s=10.0,
        window_close_s=11.0,
        warm_up_converged=True,
        psi_some_window_pct=some,
        psi_full_window_pct=full,
        psi_samples=3,
        psi_some_max=some,
        psi_full_max=full,
        psi_some_at_close=some,
        psi_full_at_close=full,
        client_cpu_pct_of_cpuset=1.0,
        pacing_slip_p99_ms=1.0,
        clean=clean,
        unclean_reason=None if clean else "did not drive cleanly",
    )


def test_an_unclean_window_is_recorded_and_excluded_from_the_maximum() -> None:
    """A window the generator did not drive cleanly says so and does not set the bar."""
    windows = [_window(0.10, 0.02, n=16), _window(0.90, 0.40, n=128, clean=False)]
    assert thresholds_from(windows) == (0.10, 0.02)
    with pytest.raises(CalibrationRefusal, match="clean"):
        thresholds_from([_window(0.9, 0.4, clean=False)])


def test_a_threshold_is_the_ceiling_of_the_maximum_over_every_window() -> None:
    windows = [_window(0.121, 0.03), _window(0.09, 0.041), _window(0.117, 0.0)]
    assert thresholds_from(windows) == (0.13, 0.05)
    assert ceil_to_precision(0.12, PRECISION_PCT) == 0.12  # exactly on the grid stays
    assert ceil_to_precision(0.1201) == 0.13
    assert ceil_to_precision(0.0) == 0.0


def test_no_pressure_samples_is_a_refusal_not_a_zero() -> None:
    with pytest.raises(CalibrationRefusal, match="pressure"):
        thresholds_from([_window(None, None)])
    with pytest.raises(CalibrationRefusal):
        thresholds_from([])


def _quiet_tree(tmp_path: Path, *, steal: int = 0, load: str = "0.20 0.10 0.10 1/100 1\n"):
    procfs = _fake_host_tree(tmp_path, steal=steal, nr=2, usec=5000)
    _write(procfs / "loadavg", load)
    _write(procfs / "pressure" / "cpu", PRESSURE)
    return procfs, tmp_path / "cgroup"


def test_observe_quiet_passes_a_quiet_box_and_records_pressure(tmp_path: Path) -> None:
    procfs, cgroupfs = _quiet_tree(tmp_path)
    reading = observe_quiet(duration_s=0.0, procfs=procfs, cgroupfs=cgroupfs, sleep=lambda s: None)
    assert reading.refusal is None
    assert reading.psi_some == pytest.approx(0.12)
    assert reading.psi_full == pytest.approx(0.03)
    assert reading.to_json_dict()["quiet"] is True


def test_observe_quiet_refuses_steal_load_and_a_foreign_gpu_process(tmp_path: Path) -> None:
    procfs, cgroupfs = _quiet_tree(tmp_path, steal=10)
    _write_pid_stat(procfs, 1, 0, 0)
    # steal accrues only across the observation: write a larger steal count after start.
    sampler_seen: list[str] = []

    def sleep(_: float) -> None:
        _write(procfs / "stat", "cpu  200 0 200 1600 0 0 0 20 0 0\n")
        sampler_seen.append("slept")

    stolen = observe_quiet(duration_s=1.0, procfs=procfs, cgroupfs=cgroupfs, sleep=sleep)
    assert stolen.refusal is not None and "steal" in stolen.refusal
    busy_procfs, busy_cgroupfs = _quiet_tree(tmp_path / "busy", load="40.00 30.00 20.00 5/100 1\n")
    busy = observe_quiet(
        duration_s=0.0, procfs=busy_procfs, cgroupfs=busy_cgroupfs, sleep=lambda s: None
    )
    assert busy.refusal is not None and "load average" in busy.refusal
    calm_procfs, calm_cgroupfs = _quiet_tree(tmp_path / "gpu")
    probe = FakeGpuProbe((fake_gpu_facts(compute_process_pids=(4242, 999)),))
    foreign = observe_quiet(
        duration_s=0.0,
        procfs=calm_procfs,
        cgroupfs=calm_cgroupfs,
        gpu=probe,
        server_pid=4242,
        sleep=lambda s: None,
    )
    assert foreign.refusal is not None and "999" in foreign.refusal
    assert foreign.foreign_gpu_pids == (999,)


async def test_calibration_refuses_a_busy_box_before_running_any_load(tmp_path: Path) -> None:
    procfs, cgroupfs = _quiet_tree(tmp_path, load="40.00 30.00 20.00 5/100 1\n")
    with pytest.raises(CalibrationRefusal, match="not quiet"):
        await calibrate(
            manifest=tmp_path / "missing.jsonl",  # never read: the refusal comes first
            procfs=procfs,
            cgroupfs=cgroupfs,
            quiet_s=0.0,
            sleep=lambda s: None,
        )


async def test_calibration_runs_the_null_floor_and_writes_provenance(tmp_path: Path) -> None:
    from test_pace import make_manifest, make_wav

    # The box record needs the box tree (cpuinfo, quota, clocksource); the sampler and
    # the quiet test need the host tree (stat, pressure, loadavg). Both under tmp_path.
    _, sysfs, _ = _fake_box_tree(tmp_path)
    procfs, cgroupfs = _quiet_tree(tmp_path)
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    record = await calibrate(
        manifest=manifest,
        ns=(1, 2),
        seeds=(constants.SEEDS[0], constants.SEEDS[1]),
        window_s=0.6,
        warm_up_s=None,  # no warm-up: the window opens at once
        frame_ms=160,
        interval_s=0.1,
        quiet_s=0.0,
        procfs=procfs,
        cgroupfs=cgroupfs,
        sysfs=sysfs,
        repo_root=Path(__file__).resolve().parents[2],
        sleep=lambda s: None,
    )
    # The fake counters never advance, so the windows' pressure is exactly zero and the
    # thresholds are zero; the avg60 context still shows the fake file's 0.12 and 0.03.
    assert record.psi_cpu_some_max_pct == 0.0
    assert record.psi_cpu_full_max_pct == 0.0
    assert all(w.psi_some_window_pct == pytest.approx(0.0) for w in record.windows)
    assert all(w.psi_some_max == pytest.approx(0.12) for w in record.windows)
    assert len(record.windows) == 4  # two concurrencies, two seeds each
    assert [w.n for w in record.windows] == [1, 1, 2, 2]
    assert all(w.psi_samples >= 3 for w in record.windows)
    assert all(w.clean for w in record.windows)
    assert record.canonical is False
    assert record.quiet.refusal is None
    doc = record.to_json_dict()
    assert doc["schema"] == "vb-psi-calibration/1"
    assert doc["thresholds"]["PSI_CPU_SOME_MAX_PCT"] == 0.0
    assert doc["windows"][0]["psi_some_window_pct"] == pytest.approx(0.0)
    assert doc["windows"][0]["avg60_some_max"] == pytest.approx(0.12)
    assert doc["box"]["boot_id"] == "boot-1234"
    assert doc["load"]["ns"] == [1, 2] and doc["load"]["seeds"] == list(record.seeds)
    assert doc["load"]["clean_windows"] == 4 and doc["load"]["unclean_windows"] == []
    assert doc["harness"]["version"]
    assert "PSI_CPU_SOME_MAX_PCT: Final[float | None] = 0.0" in record.constants_lines()
    json.dumps(doc)


def test_the_cli_writes_the_record_and_prints_the_two_lines(tmp_path: Path, capsys) -> None:
    """Against the real /proc on this box, fast durations: the shape, not the number."""
    from test_pace import make_manifest, make_wav
    from verbatim_bench.cli import main

    if not Path("/proc/pressure/cpu").exists():
        pytest.skip("this box exposes no /proc/pressure/cpu")
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    rc = main(
        [
            "calibrate-psi",
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "cal"),
            "--ns",
            "1",
            "--seeds",
            str(constants.SEEDS[0]),
            "--window-s",
            "0.6",
            "--frame-ms",
            "160",
            "--interval-s",
            "0.1",
            "--quiet-s",
            "0.2",
            "--no-gpu",
        ]
    )
    out = capsys.readouterr().out
    if rc == 2 and "refused" in out:
        pytest.skip(f"this box refused the calibration: {out.strip()}")
    assert rc == 0, out
    doc = json.loads((tmp_path / "cal" / "calibration.json").read_text())
    assert doc["thresholds"]["PSI_CPU_SOME_MAX_PCT"] >= 0.0
    assert doc["quiet"]["quiet"] is True
    assert "PSI_CPU_SOME_MAX_PCT: Final[float | None] = " in out
    assert "not canonical" in out


def test_the_quiet_observation_is_one_load_average_time_constant() -> None:
    """The pressure estimator is a counter delta and needs no wait; the observation's
    length is for the one-minute load average it reads."""
    assert QUIET_OBSERVATION_S == 60.0


def test_observe_quiet_records_both_ends_of_the_pressure_decay(tmp_path: Path) -> None:
    procfs, cgroupfs = _quiet_tree(tmp_path)
    seen: list[str] = []

    def sleep(_: float) -> None:
        seen.append("slept")
        _write(procfs / "pressure" / "cpu", PRESSURE.replace("avg60=0.12", "avg60=0.01"))

    reading = observe_quiet(duration_s=1.0, procfs=procfs, cgroupfs=cgroupfs, sleep=sleep)
    assert seen == ["slept"]
    assert reading.psi_some_start == pytest.approx(0.12)
    assert reading.psi_some == pytest.approx(0.01)
    doc = reading.to_json_dict()
    assert doc["avg60_some_start"] == pytest.approx(0.12)
    assert doc["psi_some_window_pct"] == pytest.approx(0.0)  # a static counter: no stall
