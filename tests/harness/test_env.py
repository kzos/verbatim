# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the machine-read environment record."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from verbatim_bench.env import (
    EnvironmentRefusal,
    FakeGpuProbe,
    HostSampler,
    box_id,
    collect,
    fake_gpu_facts,
    harness_identity,
    parse_smi_xml,
    read_box,
    refuse_if_unfit,
    same_box,
    single_thread_score,
)

pytestmark = pytest.mark.cpu

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smi_q_x_synthetic.xml"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_box_tree(root: Path, *, quota: str = "max 100000") -> tuple[Path, Path, Path]:
    procfs = root / "proc"
    sysfs = root / "sys"
    cgroupfs = root / "cgroup"
    _write(procfs / "sys" / "kernel" / "random" / "boot_id", "boot-1234\n")
    cpuinfo = "".join(
        f"processor\t: {i}\nmodel name\t: Synthetic CPU\nflags\t\t: fpu hypervisor\n"
        f"physical id\t: {i // 4}\ncore id\t\t: {i % 4}\nmicrocode\t: 0x1\n\n"
        for i in range(8)
    )
    _write(procfs / "cpuinfo", cpuinfo)
    _write(procfs / "1" / "cgroup", "0::/\n")
    _write(procfs / "self" / "timerslack_ns", "50000\n")
    (sysfs / "devices" / "system" / "node").mkdir(parents=True, exist_ok=True)
    (sysfs / "devices" / "system" / "node" / "node0").mkdir(parents=True, exist_ok=True)
    (sysfs / "devices" / "system" / "node" / "node1").mkdir(parents=True, exist_ok=True)
    clock_dir = sysfs / "devices" / "system" / "clocksource" / "clocksource0"
    _write(clock_dir / "current_clocksource", "tsc\n")
    _write(cgroupfs / "cpu.max", quota + "\n")
    _write(cgroupfs / "cpuset.cpus.effective", "0-3\n")
    _write(cgroupfs / "memory.max", "max\n")
    return procfs, sysfs, cgroupfs


def _collect_kwargs(root: Path, probe=None, **overrides):
    procfs, sysfs, cgroupfs = _fake_box_tree(root)
    gpu = probe if probe is not None else FakeGpuProbe((fake_gpu_facts(),))
    kwargs: dict = {
        "gpu": gpu,
        "gpu_index": 0,
        "repo_root": Path(__file__).resolve().parents[2],
        "procfs": procfs,
        "sysfs": sysfs,
        "cgroupfs": cgroupfs,
    }
    kwargs.update(overrides)
    return kwargs


def test_parse_smi_xml_reads_every_field_the_record_needs() -> None:
    (gpu,) = parse_smi_xml(FIXTURE.read_text(encoding="utf-8"))
    assert gpu.uuid == "GPU-11111111-2222-3333-4444-555555555555"
    assert gpu.name == "Synthetic Test GPU"
    assert gpu.sm == "8.6"
    assert gpu.vram_total_gb == pytest.approx(49140 / 1024.0)
    assert gpu.driver == "550.144.03"
    assert gpu.cuda_driver == "12.4"
    assert gpu.vbios == "94.02.00.00.00"
    assert gpu.pci_bus_id == "00000000:01:00.0"
    assert gpu.serial == "1523822000000"
    assert gpu.pcie_gen_current == 4
    assert gpu.pcie_gen_max == 4
    assert gpu.pcie_width_current == 16
    assert gpu.pcie_width_max == 16
    assert gpu.persistence_mode is True
    assert gpu.compute_mode == "Default"
    assert gpu.mig_mode == "Disabled"
    assert gpu.ecc_current is True
    assert gpu.ecc_pending is True
    assert gpu.power_limit_w == pytest.approx(300.0)
    assert gpu.power_limit_default_w == pytest.approx(300.0)
    assert gpu.power_limit_min_w == pytest.approx(100.0)
    assert gpu.power_limit_max_w == pytest.approx(300.0)
    assert gpu.clock_sm_mhz == 1500
    assert gpu.clock_mem_mhz == 1200
    assert gpu.clock_sm_max_mhz == 1800
    assert gpu.temperature_c == pytest.approx(45.0)
    assert gpu.temperature_slowdown_c == pytest.approx(85.0)
    assert gpu.temperature_shutdown_c == pytest.approx(92.0)
    assert gpu.throttle_reasons["clocks_throttle_reason_gpu_idle"] == 0
    assert gpu.compute_process_pids == ()
    assert gpu.util_pct == pytest.approx(5.0)


def test_parse_smi_xml_leaves_a_missing_element_as_none() -> None:
    text = FIXTURE.read_text(encoding="utf-8").replace("<serial>1523822000000</serial>", "")
    (gpu,) = parse_smi_xml(text)
    assert gpu.serial is None
    assert gpu.uuid is not None


def test_box_id_is_stable_across_two_collections_on_the_same_inputs() -> None:
    procfs, sysfs, cgroupfs = _fake_box_tree(Path("/tmp/box-stable"))
    box_a = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    box_b = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    gpu = fake_gpu_facts()
    assert box_id(box_a, gpu) == box_id(box_b, gpu)


def test_box_id_changes_when_the_cpuset_changes(tmp_path: Path) -> None:
    procfs, sysfs, cgroupfs = _fake_box_tree(tmp_path / "a")
    box_a = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    (cgroupfs / "cpuset.cpus.effective").write_text("0-1\n", encoding="utf-8")
    box_b = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    assert box_id(box_a, fake_gpu_facts()) != box_id(box_b, fake_gpu_facts())


def test_box_id_changes_when_the_gpu_uuid_changes(tmp_path: Path) -> None:
    procfs, sysfs, cgroupfs = _fake_box_tree(tmp_path)
    box = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    assert box_id(box, fake_gpu_facts(uuid="GPU-A")) != box_id(box, fake_gpu_facts(uuid="GPU-B"))


def test_read_box_parses_a_cgroup_quota_that_is_not_max(tmp_path: Path) -> None:
    procfs, sysfs, cgroupfs = _fake_box_tree(tmp_path, quota="200000 100000")
    box = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    assert box.cgroup_cpu_max == "200000 100000"
    assert box.threads_logical == 8


def test_collect_refuses_when_persistence_mode_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=True, container_digest=None
        ),
    )
    probe = FakeGpuProbe((fake_gpu_facts(persistence_mode=False),))
    with pytest.raises(EnvironmentRefusal, match="persistence"):
        collect(**_collect_kwargs(tmp_path, probe=probe))


def test_collect_refuses_when_a_foreign_compute_process_is_on_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=True, container_digest=None
        ),
    )
    probe = FakeGpuProbe((fake_gpu_facts(compute_process_pids=(1234,)),))
    with pytest.raises(EnvironmentRefusal, match="foreign"):
        collect(**_collect_kwargs(tmp_path, probe=probe))


def test_collect_refuses_on_a_dirty_harness_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=False, container_digest=None
        ),
    )
    with pytest.raises(EnvironmentRefusal, match="dirty"):
        collect(**_collect_kwargs(tmp_path))


def test_collect_refuses_on_a_null_gpu_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=True, container_digest=None
        ),
    )
    with pytest.raises(EnvironmentRefusal, match="null gpu"):
        collect(**_collect_kwargs(tmp_path, probe=FakeGpuProbe(())))


def test_collect_refuses_on_an_unreadable_cgroup_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=True, container_digest=None
        ),
    )
    procfs, sysfs, cgroupfs = _fake_box_tree(tmp_path)
    (cgroupfs / "cpu.max").unlink()
    with pytest.raises(EnvironmentRefusal, match="cgroup quota"):
        collect(
            gpu=FakeGpuProbe((fake_gpu_facts(),)),
            gpu_index=0,
            repo_root=Path(__file__).resolve().parents[2],
            procfs=procfs,
            sysfs=sysfs,
            cgroupfs=cgroupfs,
        )


def test_declared_fields_are_prefixed_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=True, container_digest=None
        ),
    )
    record = collect(**_collect_kwargs(tmp_path, declared={"declared_provider": "test-lab"}))
    assert set(record.declared) == {"declared_provider"}
    payload = record.to_json_dict()
    for block_name, block in payload.items():
        if block_name == "declared":
            continue
        text = str(block)
        assert "declared_provider" not in text or block_name == "declared"


def test_same_box_reads_no_declared_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import verbatim_bench.env as env_module

    monkeypatch.setattr(
        env_module,
        "harness_identity",
        lambda repo_root: env_module.HarnessIdentity(
            version="0.0.1.dev0", git_sha="abc", tree_clean=True, container_digest=None
        ),
    )
    kwargs_a = _collect_kwargs(tmp_path / "a", declared={"declared_provider": "lab-a"})
    kwargs_b = _collect_kwargs(tmp_path / "b", declared={"declared_provider": "lab-b"})
    record_a = collect(**kwargs_a)
    record_b = collect(**kwargs_b)
    object.__setattr__(record_a, "cal", {**record_a.cal, "null_floor_p95_ms": 10.0})
    object.__setattr__(record_b, "cal", {**record_b.cal, "null_floor_p95_ms": 10.0})
    assert same_box(record_a, record_b, floor_tolerance_pct=5.0) is True
    object.__setattr__(record_b, "box", {**record_b.box, "cpuset_effective": "0-1"})
    assert same_box(record_a, record_b, floor_tolerance_pct=5.0) is False


def _fake_host_tree(root: Path, *, steal: int = 10, nr: int = 2, usec: int = 5000) -> Path:
    procfs = root / "proc"
    cgroupfs = root / "cgroup"
    _write(procfs / "stat", f"cpu  100 0 100 800 0 0 0 {steal} 0 0\n")
    _write(
        procfs / "pressure" / "cpu",
        "some avg10=0.10 avg60=0.20 avg300=0.00 total=100\n"
        "full avg10=0.05 avg60=0.10 avg300=0.00 total=50\n",
    )
    _write(procfs / "loadavg", "0.50 0.30 0.20 1/100 1234\n")
    _write(cgroupfs / "cpu.stat", f"nr_periods 100\nnr_throttled {nr}\nthrottled_usec {usec}\n")
    _write(cgroupfs / "cpuset.cpus.effective", "0-3\n")
    return procfs


def _write_pid_stat(procfs: Path, pid: int, utime: int, stime: int) -> None:
    fields = ["1", "(fake)", "R", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
    fields += [str(utime), str(stime)] + ["0"] * 10
    _write(procfs / str(pid) / "stat", " ".join(fields) + "\n")


def test_host_sampler_computes_deltas_not_absolutes(tmp_path: Path) -> None:
    procfs = _fake_host_tree(tmp_path, steal=10, nr=2, usec=5000)
    cgroupfs = tmp_path / "cgroup"
    _write_pid_stat(procfs, 999999, 100, 50)
    sampler = HostSampler(server_pid=None, client_pid=999999, procfs=procfs, cgroupfs=cgroupfs)
    sampler.start()
    _write(procfs / "stat", "cpu  200 0 200 1600 0 0 0 20 0 0\n")
    _write(cgroupfs / "cpu.stat", "nr_periods 200\nnr_throttled 5\nthrottled_usec 9000\n")
    _write_pid_stat(procfs, 999999, 100, 50)
    counters = sampler.stop()
    assert counters.cgroup_nr_throttled_delta == 3
    assert counters.cgroup_throttled_usec_delta == 4000
    assert counters.steal_pct == pytest.approx(10 / 1010 * 100.0, rel=0.05)


def test_hottest_thread_percentage_is_per_thread_not_per_process(tmp_path: Path) -> None:
    procfs = _fake_host_tree(tmp_path)
    cgroupfs = tmp_path / "cgroup"
    pid = 424242
    for tid, (utime, stime) in enumerate([(100, 0), (100, 0), (100, 0)], start=1000):
        task_stat = procfs / str(pid) / "task" / str(tid) / "stat"
        fields = ["1", "(t)", "R", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
        fields += [str(utime), str(stime)] + ["0"] * 10
        _write(task_stat, " ".join(fields) + "\n")
    _write_pid_stat(procfs, pid, 300, 0)
    sampler = HostSampler(server_pid=pid, client_pid=None, procfs=procfs, cgroupfs=cgroupfs)
    sampler.start()
    for tid in (1000, 1001, 1002):
        extra = 100 if tid == 1000 else 1
        fields = ["1", "(t)", "R", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
        fields += [str(100 + extra), "0"] + ["0"] * 10
        _write(procfs / str(pid) / "task" / str(tid) / "stat", " ".join(fields) + "\n")
    _write_pid_stat(procfs, pid, 402, 0)
    import time as _time

    _time.sleep(0.05)
    counters = sampler.stop()
    assert counters.server_threads == 3
    assert counters.server_hottest_thread_pct is not None
    assert counters.server_cpu_pct_of_cpuset is not None
    assert counters.server_hottest_thread_pct > counters.server_cpu_pct_of_cpuset


def test_single_thread_score_is_reproducible_within_its_own_tolerance() -> None:
    first = single_thread_score(duration_s=0.02)
    second = single_thread_score(duration_s=0.02)
    assert first > 0 and second > 0
    spread = abs(first - second) / max(first, second)
    assert spread < 0.75


def test_env_collector_imports_no_optional_dependency() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import verbatim_bench.env"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, verbatim_bench.env; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_smi_fixture_declares_itself_synthetic() -> None:
    header = FIXTURE.read_text(encoding="utf-8")[:1200].lower()
    assert "hand-written" in header or "handwritten" in header
    assert "placeholder" in header
    assert "never" in header and "measured" in header


def test_refuse_if_unfit_rejects_a_dirty_tree_directly() -> None:
    import verbatim_bench.env as env_module

    record = env_module.EnvironmentRecord(
        box={"cgroup_cpu_max": "max 100000"},
        gpu=env_module._gpu_to_dict(fake_gpu_facts()),
        sw={},
        arm={},
        harness={"tree_clean": False},
        checkpoint={},
        corpus={},
        run={},
        cal={},
        host={},
        declared={},
    )
    with pytest.raises(EnvironmentRefusal):
        refuse_if_unfit(record)
    assert harness_identity(Path(__file__).resolve().parents[2]).version
