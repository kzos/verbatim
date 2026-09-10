# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Machine-read environment record for comparable benchmark rows.

Every field the comparability check reads is collected here from NVML
(via ``nvidia-smi -q -x``), ``/proc``, ``/sys``, cgroup files, the runtime and
the launched process. Anything a human would type lives only under ``declared_``
keys that :func:`same_box` never reads.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol


@dataclass(frozen=True, slots=True)
class GpuFacts:
    uuid: str | None
    name: str | None
    sm: str | None
    vram_total_gb: float | None
    driver: str | None
    cuda_driver: str | None
    vbios: str | None
    pci_bus_id: str | None
    serial: str | None
    pcie_gen_current: int | None
    pcie_gen_max: int | None
    pcie_width_current: int | None
    pcie_width_max: int | None
    persistence_mode: bool | None
    compute_mode: str | None
    mig_mode: str | None
    ecc_current: bool | None
    ecc_pending: bool | None
    power_limit_w: float | None
    power_limit_default_w: float | None
    power_limit_min_w: float | None
    power_limit_max_w: float | None
    clock_sm_mhz: int | None
    clock_mem_mhz: int | None
    clock_sm_max_mhz: int | None
    clock_sm_app_mhz: int | None
    temperature_c: float | None
    temperature_slowdown_c: float | None
    temperature_shutdown_c: float | None
    throttle_reasons: Mapping[str, int]
    compute_process_pids: tuple[int, ...]
    energy_counter_available: bool
    util_pct: float | None
    nvml_version: str | None = None


class GpuProbe(Protocol):
    def count(self) -> int: ...

    def facts(self, index: int) -> GpuFacts: ...


def _child_text(parent: ET.Element | None, names: Sequence[str]) -> str | None:
    if parent is None:
        return None
    for name in names:
        child = parent.find(name)
        if child is not None and child.text is not None:
            text = child.text.strip()
            if text:
                return text
    return None


def _find_gpu_parent(root: ET.Element, tag: str) -> ET.Element | None:
    found = root.find(f".//{tag}")
    return found


def _parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.split()[0].strip().rstrip("x")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    cleaned = text.split()[0].strip().rstrip("x")
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _parse_bool_enabled(text: str | None) -> bool | None:
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in ("enabled", "on", "true", "yes"):
        return True
    if lowered in ("disabled", "off", "false", "no", "n/a", "not supported"):
        return False
    return None


def _parse_mib_to_gb(text: str | None) -> float | None:
    value = _parse_float(text)
    if value is None:
        return None
    return value / 1024.0


def _parse_throttle_reasons(gpu: ET.Element) -> dict[str, int]:
    reasons: dict[str, int] = {}
    parent = gpu.find("throttle_reasons")
    if parent is None:
        parent = gpu.find("clocks_event_reasons")
    if parent is None:
        parent = gpu.find("clock_event_reasons")
    if parent is None:
        return reasons
    for child in list(parent):
        tag = str(child.tag).strip()
        if not tag or child.text is None:
            continue
        text = child.text.strip()
        try:
            reasons[tag] = int(text.split()[0])
        except ValueError:
            lowered = text.lower()
            if lowered in ("active", "1", "true", "yes"):
                reasons[tag] = 1
            elif lowered in ("not active", "0", "false", "no", "n/a"):
                reasons[tag] = 0
    return reasons


def _parse_pids(gpu: ET.Element) -> tuple[int, ...]:
    pids: list[int] = []
    for parent_tag in ("processes", "compute_processes"):
        parent = gpu.find(parent_tag)
        if parent is None:
            continue
        for pid_el in parent.findall(".//pid"):
            if pid_el.text is None:
                continue
            try:
                pids.append(int(pid_el.text.strip().split()[0]))
            except ValueError:
                continue
    return tuple(pids)


def _parse_one_gpu(gpu: ET.Element, root: ET.Element) -> GpuFacts:
    driver = _child_text(root, ["driver_version"])
    cuda_driver = _child_text(root, ["cuda_version"])
    nvml_version = _child_text(root, ["nvml_version", "nvidia_smi_version"])
    name = _child_text(gpu, ["product_name", "name"])
    uuid = _child_text(gpu, ["uuid"])
    serial = _child_text(gpu, ["serial"])
    if serial is not None and serial.lower() in ("n/a", "unknown", "not supported", "-"):
        serial = None
    vbios = _child_text(gpu, ["vbios_version", "vbios"])
    pci_bus_id = _child_text(gpu, ["pci_bus_id"])
    if pci_bus_id is None:
        pci_bus_id = _child_text(gpu.find("pci"), ["pci_bus_id", "bus_id"])
    sm = _child_text(gpu, ["compute_cap", "compute_capability", "sm", "cuda_compute_cap"])
    vram_total_gb = _parse_mib_to_gb(_child_text(gpu, ["fb_memory_total_mib", "memory_total_mib"]))
    if vram_total_gb is None:
        mem_parent = gpu.find("fb_memory_usage")
        if mem_parent is None:
            mem_parent = gpu.find("memory_usage")
        vram_total_gb = _parse_mib_to_gb(
            _child_text(mem_parent, ["total"]) if mem_parent is not None else None
        )
    pcie_parent = gpu.find("pcie_gen")
    if pcie_parent is None:
        pcie_parent = gpu.find("pci_gpu_link_info")
    if pcie_parent is None:
        pcie_parent = gpu.find("pci")
    pcie_gen_current = _parse_int(
        _child_text(pcie_parent, ["pcie_gen_current", "current_link_gen"])
        if pcie_parent is not None
        else _child_text(gpu, ["pcie_gen_current"])
    )
    pcie_gen_max = _parse_int(
        _child_text(pcie_parent, ["pcie_gen_max", "max_link_gen"])
        if pcie_parent is not None
        else _child_text(gpu, ["pcie_gen_max"])
    )
    pcie_width_current = _parse_int(
        _child_text(pcie_parent, ["pcie_width_current", "current_link_width"])
        if pcie_parent is not None
        else _child_text(gpu, ["pcie_width_current"])
    )
    pcie_width_max = _parse_int(
        _child_text(pcie_parent, ["pcie_width_max", "max_link_width"])
        if pcie_parent is not None
        else _child_text(gpu, ["pcie_width_max"])
    )
    persistence_mode = _parse_bool_enabled(_child_text(gpu, ["persistence_mode", "persistence"]))
    compute_mode = _child_text(gpu, ["compute_mode"])
    mig_mode = _child_text(gpu, ["mig_mode", "current_mig"])
    if mig_mode is None:
        mig_parent = gpu.find("mig_mode")
        mig_mode = _child_text(mig_parent, ["current_mig"]) if mig_parent is not None else None
    ecc_parent = gpu.find("ecc_mode")
    ecc_current = _parse_bool_enabled(
        _child_text(ecc_parent, ["current_ecc", "ecc_current"])
        if ecc_parent is not None
        else _child_text(gpu, ["ecc_current"])
    )
    ecc_pending = _parse_bool_enabled(
        _child_text(ecc_parent, ["pending_ecc", "ecc_pending"])
        if ecc_parent is not None
        else _child_text(gpu, ["ecc_pending"])
    )
    power_parent = gpu.find("power_readings")
    if power_parent is None:
        power_parent = gpu.find("power_limits")
    if power_parent is None:
        power_parent = gpu
    power_limit_w = _parse_float(
        _child_text(power_parent, ["power_limit", "power_limit_w", "current_power_limit"])
    )
    power_limit_default_w = _parse_float(
        _child_text(power_parent, ["default_power_limit", "power_limit_default_w"])
    )
    power_limit_min_w = _parse_float(
        _child_text(power_parent, ["min_power_limit", "power_limit_min_w"])
    )
    power_limit_max_w = _parse_float(
        _child_text(power_parent, ["max_power_limit", "power_limit_max_w"])
    )
    clocks_parent = gpu.find("clocks")
    clock_sm_mhz = _parse_int(
        _child_text(clocks_parent, ["graphics_clock", "clock_sm_mhz", "sm_clock"])
        if clocks_parent is not None
        else _child_text(gpu, ["clock_sm_mhz"])
    )
    clock_mem_mhz = _parse_int(
        _child_text(clocks_parent, ["mem_clock", "clock_mem_mhz", "mem_clock_current"])
        if clocks_parent is not None
        else _child_text(gpu, ["clock_mem_mhz"])
    )
    clock_sm_max_mhz = _parse_int(
        _child_text(clocks_parent, ["max_graphics_clock", "clock_sm_max_mhz", "max_sm_clock"])
        if clocks_parent is not None
        else _child_text(gpu, ["clock_sm_max_mhz"])
    )
    app_parent = clocks_parent.find("applications_clocks") if clocks_parent is not None else None
    clock_sm_app_mhz = _parse_int(
        _child_text(app_parent, ["graphics_clock", "clock_sm_app_mhz"])
        if app_parent is not None
        else _child_text(gpu, ["clock_sm_app_mhz"])
    )
    temp_parent = gpu.find("temperature")
    temperature_c = _parse_float(
        _child_text(temp_parent, ["gpu_temp", "temperature_c", "current_temp"])
        if temp_parent is not None
        else _child_text(gpu, ["temperature_c"])
    )
    temperature_slowdown_c = _parse_float(
        _child_text(temp_parent, ["gpu_temp_slow_threshold", "temperature_slowdown_c"])
        if temp_parent is not None
        else _child_text(gpu, ["temperature_slowdown_c"])
    )
    temperature_shutdown_c = _parse_float(
        _child_text(temp_parent, ["gpu_temp_max_threshold", "temperature_shutdown_c"])
        if temp_parent is not None
        else _child_text(gpu, ["temperature_shutdown_c"])
    )
    throttle_reasons = _parse_throttle_reasons(gpu)
    compute_process_pids = _parse_pids(gpu)
    energy_text = _child_text(gpu, ["energy_counter_available", "power_draw_available"])
    if energy_text is None:
        energy_counter_available = _child_text(gpu, ["power_draw"]) is not None
    else:
        lowered = energy_text.strip().lower()
        energy_counter_available = lowered in ("true", "yes", "1", "available", "enabled")
    util_parent = gpu.find("utilization")
    util_pct = _parse_float(
        _child_text(util_parent, ["gpu_util", "util_pct", "gpu"])
        if util_parent is not None
        else _child_text(gpu, ["util_pct"])
    )
    return GpuFacts(
        uuid=uuid,
        name=name,
        sm=sm,
        vram_total_gb=vram_total_gb,
        driver=driver,
        cuda_driver=cuda_driver,
        vbios=vbios,
        pci_bus_id=pci_bus_id,
        serial=serial,
        pcie_gen_current=pcie_gen_current,
        pcie_gen_max=pcie_gen_max,
        pcie_width_current=pcie_width_current,
        pcie_width_max=pcie_width_max,
        persistence_mode=persistence_mode,
        compute_mode=compute_mode,
        mig_mode=mig_mode,
        ecc_current=ecc_current,
        ecc_pending=ecc_pending,
        power_limit_w=power_limit_w,
        power_limit_default_w=power_limit_default_w,
        power_limit_min_w=power_limit_min_w,
        power_limit_max_w=power_limit_max_w,
        clock_sm_mhz=clock_sm_mhz,
        clock_mem_mhz=clock_mem_mhz,
        clock_sm_max_mhz=clock_sm_max_mhz,
        clock_sm_app_mhz=clock_sm_app_mhz,
        temperature_c=temperature_c,
        temperature_slowdown_c=temperature_slowdown_c,
        temperature_shutdown_c=temperature_shutdown_c,
        throttle_reasons=throttle_reasons,
        compute_process_pids=compute_process_pids,
        energy_counter_available=bool(energy_counter_available),
        util_pct=util_pct,
        nvml_version=nvml_version,
    )


def parse_smi_xml(text: str) -> tuple[GpuFacts, ...]:
    """Parse ``nvidia-smi -q -x`` output into one fact record per GPU.

    Missing elements become ``None``, never a default that could be mistaken
    for a reading.
    """
    root = ET.fromstring(text)
    gpus = root.findall("gpu")
    if not gpus:
        gpus = root.findall(".//gpu")
    return tuple(_parse_one_gpu(gpu, root) for gpu in gpus)


class SmiProbe:
    """Read GPU facts from ``nvidia-smi -q -x`` output."""

    def __init__(self, gpus: tuple[GpuFacts, ...]) -> None:
        self._gpus = gpus

    @classmethod
    def from_xml(cls, text: str) -> SmiProbe:
        return cls(parse_smi_xml(text))

    @classmethod
    def from_command(cls, argv: Sequence[str] = ("nvidia-smi", "-q", "-x")) -> SmiProbe:
        proc = subprocess.run(list(argv), capture_output=True, text=True, check=False)
        proc.check_returncode()
        return cls.from_xml(proc.stdout)

    def count(self) -> int:
        return len(self._gpus)

    def facts(self, index: int) -> GpuFacts:
        return self._gpus[index]


@dataclass(frozen=True, slots=True)
class FakeGpuProbe:
    gpus: tuple[GpuFacts, ...] = ()

    def count(self) -> int:
        return len(self.gpus)

    def facts(self, index: int) -> GpuFacts:
        return self.gpus[index]


def fake_gpu_facts(**overrides: Any) -> GpuFacts:
    """Return a fully populated synthetic GPU record for tests and local runs."""
    base: dict[str, Any] = {
        "uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "name": "Synthetic Test GPU",
        "sm": "8.6",
        "vram_total_gb": 48.0,
        "driver": "550.144.03",
        "cuda_driver": "12.4",
        "vbios": "94.02.00.00.00",
        "pci_bus_id": "00000000:00:00.0",
        "serial": None,
        "pcie_gen_current": 4,
        "pcie_gen_max": 4,
        "pcie_width_current": 16,
        "pcie_width_max": 16,
        "persistence_mode": True,
        "compute_mode": "Default",
        "mig_mode": "Disabled",
        "ecc_current": True,
        "ecc_pending": True,
        "power_limit_w": 300.0,
        "power_limit_default_w": 300.0,
        "power_limit_min_w": 100.0,
        "power_limit_max_w": 300.0,
        "clock_sm_mhz": 1500,
        "clock_mem_mhz": 1200,
        "clock_sm_max_mhz": 1800,
        "clock_sm_app_mhz": None,
        "temperature_c": 45.0,
        "temperature_slowdown_c": 85.0,
        "temperature_shutdown_c": 92.0,
        "throttle_reasons": {},
        "compute_process_pids": (),
        "energy_counter_available": True,
        "util_pct": 5.0,
        "nvml_version": "12.550.144.03",
    }
    base.update(overrides)
    return GpuFacts(**base)


@dataclass(frozen=True, slots=True)
class BoxFacts:
    boot_id: str
    cpu_model: str
    hypervisor_flag: bool
    microcode: str | None
    sockets: int
    cores_physical: int
    threads_logical: int
    numa_nodes: int
    cgroup_cpu_max: str
    cpuset_effective: str
    memory_max: str
    virtualisation: str
    container_runtime: str | None
    kernel: str
    clocksource: str
    monotonic_resolution_ns: int
    timerslack_ns: int | None
    single_thread_score: float | None


@dataclass(frozen=True, slots=True)
class HostCounters:
    steal_pct: float
    psi_cpu_some_avg: float | None
    psi_cpu_full_avg: float | None
    cgroup_nr_throttled_delta: int
    cgroup_throttled_usec_delta: int
    loadavg_start: float
    loadavg_end: float
    server_cpu_pct_of_cpuset: float | None
    server_hottest_thread_pct: float | None
    server_threads: int | None
    client_cpu_pct_of_cpuset: float
    client_processes: int
    server_cpu_s_per_stream_hour: float | None


@dataclass(frozen=True, slots=True)
class HarnessIdentity:
    version: str
    git_sha: str | None
    tree_clean: bool
    container_digest: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    box: dict[str, Any]
    gpu: dict[str, Any]
    sw: dict[str, Any]
    arm: dict[str, Any]
    harness: dict[str, Any]
    checkpoint: dict[str, Any]
    corpus: dict[str, Any]
    run: dict[str, Any]
    cal: dict[str, Any]
    host: dict[str, Any]
    declared: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "box": dict(self.box),
            "gpu": dict(self.gpu),
            "sw": dict(self.sw),
            "arm": dict(self.arm),
            "harness": dict(self.harness),
            "checkpoint": dict(self.checkpoint),
            "corpus": dict(self.corpus),
            "run": dict(self.run),
            "cal": dict(self.cal),
            "host": dict(self.host),
            "declared": dict(self.declared),
        }


class EnvironmentRefusal(RuntimeError):
    """Raised instead of producing a row: persistence mode off, a foreign compute
    process on the target GPU, a dirty harness tree, a null gpu block, or an
    unreadable cgroup quota."""


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_cpuinfo(text: str) -> tuple[str, bool, str | None, int, int]:
    model = "unknown"
    hypervisor = False
    microcode: str | None = None
    sockets_set: set[str] = set()
    cores_set: set[tuple[str, str]] = set()
    threads = 0
    current_physical: str | None = None
    current_core: str | None = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "model name" and model == "unknown" and value:
            model = value
        elif key == "flags":
            flags = set(value.split())
            if "hypervisor" in flags:
                hypervisor = True
        elif key == "microcode" and microcode is None and value:
            microcode = value
        elif key == "physical id":
            current_physical = value
            sockets_set.add(value)
        elif key == "core id":
            current_core = value
        elif key == "processor":
            threads += 1
            if current_physical is not None and current_core is not None:
                cores_set.add((current_physical, current_core))
    sockets = len(sockets_set) if sockets_set else 1
    cores_physical = len(cores_set) if cores_set else max(threads, 1)
    return model, hypervisor, microcode, sockets, threads or 1, cores_physical


def _count_numa_nodes(sysfs: Path) -> int:
    node_dir = sysfs / "devices" / "system" / "node"
    try:
        entries = list(node_dir.iterdir())
    except OSError:
        return 1
    nodes = [entry for entry in entries if entry.name.startswith("node")]
    return len(nodes) if nodes else 1


def _detect_container_runtime(procfs: Path) -> str | None:
    if Path("/.dockerenv").exists():
        return "docker"
    cgroup_text = _read_text_file(procfs / "1" / "cgroup")
    if cgroup_text is not None:
        lowered = cgroup_text.lower()
        if "docker" in lowered:
            return "docker"
        if "kubepods" in lowered or "containerd" in lowered:
            return "containerd"
        if "lxc" in lowered:
            return "lxc"
    return None


def single_thread_score(*, duration_s: float = 1.0) -> float:
    """Run a fixed single-threaded micro-benchmark and return its score.

    The score is iterations per second of a small integer hash loop, so it is a
    host-CPU number taken in the same session as the ladder it calibrates.
    """
    iterations = max(1000, int(200000 * max(duration_s, 0.001)))
    state = 0x243F6A88
    start = time.perf_counter()
    for _ in range(iterations):
        state = ((state * 1103515245 + 12345) & 0x7FFFFFFF) ^ (state >> 13)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return float(iterations)
    return iterations / elapsed


def read_box(
    *,
    procfs: Path = Path("/proc"),
    sysfs: Path = Path("/sys"),
    cgroupfs: Path = Path("/sys/fs/cgroup"),
) -> BoxFacts:
    """Collect host identity and budget facts from kernel and cgroup files."""
    boot_id = _read_text_file(procfs / "sys" / "kernel" / "random" / "boot_id") or "unknown"
    cpuinfo_text = _read_text_file(procfs / "cpuinfo") or ""
    if cpuinfo_text:
        cpu_model, hypervisor_flag, microcode, sockets, threads_logical, cores = _parse_cpuinfo(
            cpuinfo_text
        )
    else:
        cpu_model, hypervisor_flag, microcode = "unknown", False, None
        sockets, threads_logical, cores = 1, 1, 1
    numa_nodes = _count_numa_nodes(sysfs)
    cpu_max_path = cgroupfs / "cpu.max"
    try:
        cgroup_cpu_max = cpu_max_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise EnvironmentRefusal(f"unreadable cgroup quota at {cpu_max_path}: {exc}") from exc
    if not cgroup_cpu_max:
        raise EnvironmentRefusal(f"unreadable cgroup quota at {cpu_max_path}: empty file")
    cpuset_effective = (
        _read_text_file(cgroupfs / "cpuset.cpus.effective")
        or _read_text_file(cgroupfs / "cpuset.cpus")
        or "unknown"
    )
    memory_max = _read_text_file(cgroupfs / "memory.max") or "unknown"
    virtualisation = "vm" if hypervisor_flag else "bare"
    container_runtime = _detect_container_runtime(procfs)
    kernel = platform.uname().release
    clocksource = (
        _read_text_file(
            sysfs / "devices" / "system" / "clocksource" / "clocksource0" / "current_clocksource"
        )
        or "unknown"
    )
    try:
        monotonic_resolution_ns = time.get_clock_info("monotonic").resolution_ns
    except (AttributeError, ValueError):
        monotonic_resolution_ns = 1
    timerslack_text = _read_text_file(procfs / "self" / "timerslack_ns")
    try:
        timerslack_ns = int(timerslack_text.split()[0]) if timerslack_text else None
    except ValueError:
        timerslack_ns = None
    try:
        score = single_thread_score(duration_s=0.02)
    except OSError:
        score = None
    return BoxFacts(
        boot_id=boot_id,
        cpu_model=cpu_model,
        hypervisor_flag=hypervisor_flag,
        microcode=microcode,
        sockets=sockets,
        cores_physical=cores,
        threads_logical=threads_logical,
        numa_nodes=numa_nodes,
        cgroup_cpu_max=cgroup_cpu_max,
        cpuset_effective=cpuset_effective,
        memory_max=memory_max,
        virtualisation=virtualisation,
        container_runtime=container_runtime,
        kernel=kernel,
        clocksource=clocksource,
        monotonic_resolution_ns=int(monotonic_resolution_ns),
        timerslack_ns=timerslack_ns,
        single_thread_score=score,
    )


def box_id(box: BoxFacts, gpu: GpuFacts) -> str:
    """Return the comparability identity for one box session."""
    material = "|".join(
        [
            str(gpu.uuid or "no-gpu"),
            box.boot_id,
            box.cpu_model,
            box.cgroup_cpu_max,
            box.cpuset_effective,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def harness_identity(repo_root: Path) -> HarnessIdentity:
    """Read the harness build stamp and refuse to bless a dirty tree elsewhere."""
    from verbatim_bench import __version__

    container_digest = os.environ.get("VERBATIM_CONTAINER_DIGEST")
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        git_sha = proc.stdout.strip() or None
        if proc.returncode != 0:
            git_sha = None
    except OSError:
        git_sha = None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        tree_clean = proc.returncode == 0 and not proc.stdout.strip()
    except OSError:
        tree_clean = False
    return HarnessIdentity(
        version=__version__,
        git_sha=git_sha,
        tree_clean=bool(tree_clean),
        container_digest=container_digest,
    )


def _gpu_to_dict(gpu: GpuFacts | None) -> dict[str, Any]:
    if gpu is None:
        return {
            "uuid": None,
            "name": None,
            "sm": None,
            "vram_total_gb": None,
            "driver": None,
            "nvml_version": None,
            "cuda_driver_version": None,
            "vbios": None,
            "pci_bus_id": None,
            "serial": None,
            "pcie_gen_current": None,
            "pcie_gen_max": None,
            "pcie_width_current": None,
            "pcie_width_max": None,
            "persistence_mode": None,
            "compute_mode": None,
            "mig_mode": None,
            "ecc_current": None,
            "ecc_pending": None,
            "power_limit_w": None,
            "power_limit_default_w": None,
            "power_limit_min_w": None,
            "power_limit_max_w": None,
            "clock_sm_mhz": None,
            "clock_mem_mhz": None,
            "clock_sm_max_mhz": None,
            "clock_sm_app_mhz": None,
            "temperature_c": None,
            "temperature_slowdown_c": None,
            "temperature_shutdown_c": None,
            "throttle_reasons": {},
            "compute_process_pids": [],
            "foreign_processes_count": None,
            "energy_counter_available": None,
            "util_pct": None,
        }
    return {
        "uuid": gpu.uuid,
        "name": gpu.name,
        "sm": gpu.sm,
        "vram_total_gb": gpu.vram_total_gb,
        "driver": gpu.driver,
        "nvml_version": gpu.nvml_version,
        "cuda_driver_version": gpu.cuda_driver,
        "vbios": gpu.vbios,
        "pci_bus_id": gpu.pci_bus_id,
        "serial": gpu.serial,
        "pcie_gen_current": gpu.pcie_gen_current,
        "pcie_gen_max": gpu.pcie_gen_max,
        "pcie_width_current": gpu.pcie_width_current,
        "pcie_width_max": gpu.pcie_width_max,
        "persistence_mode": gpu.persistence_mode,
        "compute_mode": gpu.compute_mode,
        "mig_mode": gpu.mig_mode,
        "ecc_current": gpu.ecc_current,
        "ecc_pending": gpu.ecc_pending,
        "power_limit_w": gpu.power_limit_w,
        "power_limit_default_w": gpu.power_limit_default_w,
        "power_limit_min_w": gpu.power_limit_min_w,
        "power_limit_max_w": gpu.power_limit_max_w,
        "clock_sm_mhz": gpu.clock_sm_mhz,
        "clock_mem_mhz": gpu.clock_mem_mhz,
        "clock_sm_max_mhz": gpu.clock_sm_max_mhz,
        "clock_sm_app_mhz": gpu.clock_sm_app_mhz,
        "temperature_c": gpu.temperature_c,
        "temperature_slowdown_c": gpu.temperature_slowdown_c,
        "temperature_shutdown_c": gpu.temperature_shutdown_c,
        "throttle_reasons": dict(gpu.throttle_reasons),
        "compute_process_pids": list(gpu.compute_process_pids),
        "foreign_processes_count": len(gpu.compute_process_pids),
        "energy_counter_available": gpu.energy_counter_available,
        "util_pct": gpu.util_pct,
    }


def _box_to_dict(box: BoxFacts, gpu: GpuFacts | None) -> dict[str, Any]:
    computed_id = box_id(box, gpu) if gpu is not None else None
    return {
        "gpu_uuid": gpu.uuid if gpu is not None else None,
        "gpu_pci_bus_id": gpu.pci_bus_id if gpu is not None else None,
        "gpu_serial_if_readable": gpu.serial if gpu is not None else None,
        "boot_id": box.boot_id,
        "cpu_model": box.cpu_model,
        "cpu_flags_hypervisor": box.hypervisor_flag,
        "microcode": box.microcode,
        "sockets": box.sockets,
        "cores_physical": box.cores_physical,
        "threads_logical": box.threads_logical,
        "numa_nodes": box.numa_nodes,
        "cgroup_cpu_max": box.cgroup_cpu_max,
        "cpuset_effective": box.cpuset_effective,
        "memory_max": box.memory_max,
        "virtualisation": box.virtualisation,
        "container_runtime": box.container_runtime,
        "kernel": box.kernel,
        "clocksource": box.clocksource,
        "monotonic_resolution_ns": box.monotonic_resolution_ns,
        "timerslack_ns": box.timerslack_ns,
        "single_thread_score": box.single_thread_score,
        "id": computed_id,
    }


def collect(
    *,
    gpu: GpuProbe,
    gpu_index: int = 0,
    repo_root: Path,
    procfs: Path = Path("/proc"),
    sysfs: Path = Path("/sys"),
    cgroupfs: Path = Path("/sys/fs/cgroup"),
    declared: Mapping[str, str] | None = None,
) -> EnvironmentRecord:
    """Collect every machine-read field and refuse instead of guessing a row."""
    box = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    gpu_facts: GpuFacts | None = None
    if gpu.count() > gpu_index >= 0:
        try:
            gpu_facts = gpu.facts(gpu_index)
        except (IndexError, ValueError, OSError):
            gpu_facts = None
    identity = harness_identity(repo_root)
    declared_clean: dict[str, Any] = {}
    for key, value in dict(declared or {}).items():
        name = str(key)
        if not name.startswith("declared_"):
            name = f"declared_{name}"
        declared_clean[name] = value
    box_dict = _box_to_dict(box, gpu_facts)
    gpu_dict = _gpu_to_dict(gpu_facts)
    harness_tag = os.environ.get("VERBATIM_HARNESS_TAG", identity.version)
    record = EnvironmentRecord(
        box=box_dict,
        gpu=gpu_dict,
        sw={
            "python": platform.python_version(),
            "torch": None,
            "torch_cuda_build": None,
            "cudnn": None,
            "toolkit_version": None,
            "toolkit_git_sha": None,
            "tf32_matmul": None,
            "tf32_cudnn": None,
            "deterministic_algorithms": None,
            "env_CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pip_freeze_sha256": None,
        },
        arm={
            "commit_sha": identity.git_sha,
            "argv": list(sys.argv),
            "environ_filtered": {
                key: os.environ[key]
                for key in ("CUDA_VISIBLE_DEVICES", "PATH")
                if key in os.environ
            },
            "pid_cpuset": box.cpuset_effective,
            "threads_at_start": box.threads_logical,
            "container_digest": identity.container_digest,
        },
        harness={
            "version": identity.version,
            "git_sha": identity.git_sha,
            "tree_clean": identity.tree_clean,
            "container_digest": identity.container_digest,
            "tag": harness_tag,
        },
        checkpoint={
            "hf_id": None,
            "revision": None,
            "file_sha256": None,
            "converted_sha256": None,
            "converter_version": None,
        },
        corpus={
            "manifest_sha256": None,
            "session_builder_version": None,
            "session_seed": None,
            "noise_source_revision": None,
            "evaluation_stream_ids_sha256": None,
            "filler_seed": None,
        },
        run={
            "wall_start_utc": None,
            "wall_end_utc": None,
            "gpu_hours": None,
            "box_session_index": 0,
            "one_arm_at_a_time": True,
        },
        cal={
            "null_floor_p95_ms": None,
            "null_floor_client_cpu_pct": None,
            "aa_repeat": None,
        },
        host={
            "steal_pct_window": None,
            "psi_cpu_some_avg_window": None,
            "psi_cpu_full_avg_window": None,
            "cgroup_nr_throttled_delta": None,
            "cgroup_throttled_usec_delta": None,
            "loadavg_start": None,
            "loadavg_end": None,
            "server_cpu_pct_of_cpuset": None,
            "server_hottest_thread_pct": None,
            "server_threads": None,
            "client_cpu_pct_of_cpuset": None,
            "client_processes": None,
            "server_cpu_s_per_stream_hour": None,
            "pacing_slip_ms": None,
        },
        declared=declared_clean,
    )
    refuse_if_unfit(record)
    return record


def refuse_if_unfit(record: EnvironmentRecord) -> None:
    """Raise :class:`EnvironmentRefusal` instead of producing an unfit row."""
    gpu = record.gpu
    if not gpu or gpu.get("uuid") is None:
        raise EnvironmentRefusal("null gpu block: no GPU facts were collected")
    if gpu.get("persistence_mode") is False:
        raise EnvironmentRefusal("persistence mode is off on the target GPU")
    foreign = gpu.get("foreign_processes_count")
    pids = gpu.get("compute_process_pids") or []
    if isinstance(foreign, int) and foreign > 0:
        raise EnvironmentRefusal(f"foreign compute processes on the target GPU: {pids!r}")
    if len(list(pids)) > 0:
        raise EnvironmentRefusal(f"foreign compute processes on the target GPU: {list(pids)!r}")
    harness = record.harness
    if harness.get("tree_clean") is False:
        raise EnvironmentRefusal("harness tree is dirty: commit or stash before measuring")
    box = record.box
    quota = box.get("cgroup_cpu_max")
    if quota is None or (isinstance(quota, str) and not quota.strip()):
        raise EnvironmentRefusal("unreadable cgroup quota: cannot establish the CPU budget")


COMPARABILITY_FIELDS: Final = (
    "box.boot_id",
    "box.gpu_uuid",
    "box.cgroup_cpu_max",
    "box.cpuset_effective",
    "harness.container_digest",
    "harness.tag",
)


def _dotted(record: EnvironmentRecord, dotted: str) -> Any:
    block_name, _, key = dotted.partition(".")
    block = {
        "box": record.box,
        "gpu": record.gpu,
        "sw": record.sw,
        "arm": record.arm,
        "harness": record.harness,
        "checkpoint": record.checkpoint,
        "corpus": record.corpus,
        "run": record.run,
        "cal": record.cal,
        "host": record.host,
    }.get(block_name)
    if not isinstance(block, dict):
        return None
    return block.get(key)


def same_box(a: EnvironmentRecord, b: EnvironmentRecord, *, floor_tolerance_pct: float) -> bool:
    """Return true only when every comparability field matches and both null
    floors agree within the tolerance. Never reads ``declared``."""
    for dotted in COMPARABILITY_FIELDS:
        if _dotted(a, dotted) != _dotted(b, dotted):
            return False
    floor_a = a.cal.get("null_floor_p95_ms")
    floor_b = b.cal.get("null_floor_p95_ms")
    if floor_a is None or floor_b is None:
        return False
    try:
        value_a = float(floor_a)
        value_b = float(floor_b)
    except (TypeError, ValueError):
        return False
    if value_a == 0 and value_b == 0:
        return True
    if value_a == 0 or value_b == 0:
        return False
    spread = abs(value_a - value_b) / max(abs(value_a), abs(value_b)) * 100.0
    return spread <= float(floor_tolerance_pct)


def _read_proc_stat_cpu(procfs: Path) -> tuple[int, int]:
    text = _read_text_file(procfs / "stat")
    if not text:
        return 0, 0
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = line.split()
            try:
                numbers = [int(part) for part in parts[1:]]
            except ValueError:
                return 0, 0
            total = sum(numbers)
            steal = numbers[7] if len(numbers) > 7 else 0
            return total, steal
    return 0, 0


def _read_pressure_avg(procfs: Path) -> tuple[float | None, float | None]:
    text = _read_text_file(procfs / "pressure" / "cpu")
    if not text:
        return None, None
    some: float | None = None
    full: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        token_source = stripped.split(None, 1)[1] if " " in stripped else ""
        avg: float | None = None
        for token in token_source.split():
            if token.startswith("avg60="):
                try:
                    avg = float(token.split("=", 1)[1])
                except ValueError:
                    avg = None
                break
        if avg is None:
            for token in token_source.split():
                if token.startswith("avg10="):
                    try:
                        avg = float(token.split("=", 1)[1])
                    except ValueError:
                        avg = None
                    break
        if stripped.startswith("some"):
            some = avg
        elif stripped.startswith("full"):
            full = avg
    return some, full


def _read_cgroup_throttled(cgroupfs: Path) -> tuple[int, int]:
    text = _read_text_file(cgroupfs / "cpu.stat")
    if not text:
        return 0, 0
    nr_throttled = 0
    throttled_usec = 0
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        key = key.strip()
        try:
            number = int(value.strip().split()[0])
        except (ValueError, IndexError):
            continue
        if key == "nr_throttled":
            nr_throttled = number
        elif key == "throttled_usec":
            throttled_usec = number
    return nr_throttled, throttled_usec


def _read_loadavg(procfs: Path) -> float:
    text = _read_text_file(procfs / "loadavg")
    if not text:
        return 0.0
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return 0.0


def _read_proc_cpu_time(stat_path: Path) -> int | None:
    text = _read_text_file(stat_path)
    if not text:
        return None
    parts = text.split()
    if len(parts) < 15:
        return None
    try:
        utime = int(parts[13])
        stime = int(parts[14])
    except ValueError:
        return None
    return utime + stime


def _cpuset_size(cpuset: str | None) -> int:
    if not cpuset or cpuset.strip() in ("unknown", ""):
        return 1
    count = 0
    for part in cpuset.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                low, _, high = part.partition("-")
                count += int(high) - int(low) + 1
            except ValueError:
                count += 1
        else:
            count += 1
    return max(count, 1)


class HostSampler:
    """Sample host tenancy counters over a measurement window.

    The sampler records absolute counters at :meth:`start` and returns only
    deltas at :meth:`stop`, so a window with no pressure reports zeros rather
    than the machine's uptime totals.
    """

    def __init__(
        self,
        *,
        server_pid: int | None = None,
        client_pid: int | None = None,
        procfs: Path = Path("/proc"),
        cgroupfs: Path = Path("/sys/fs/cgroup"),
    ) -> None:
        self._server_pid = server_pid
        self._client_pid = client_pid if client_pid is not None else os.getpid()
        self._procfs = procfs
        self._cgroupfs = cgroupfs
        self._start_wall: float = 0.0
        self._start_total: int = 0
        self._start_steal: int = 0
        self._start_nr: int = 0
        self._start_usec: int = 0
        self._start_load: float = 0.0
        self._start_server: int | None = None
        self._start_client: int | None = None
        self._start_threads: dict[int, int] = {}
        self._started = False

    def _task_pids(self, pid: int) -> list[int]:
        task_dir = self._procfs / str(pid) / "task"
        try:
            entries = list(task_dir.iterdir())
        except OSError:
            return []
        tids: list[int] = []
        for entry in entries:
            try:
                tids.append(int(entry.name))
            except ValueError:
                continue
        return tids

    def start(self) -> None:
        self._start_wall = time.monotonic()
        self._start_total, self._start_steal = _read_proc_stat_cpu(self._procfs)
        self._start_nr, self._start_usec = _read_cgroup_throttled(self._cgroupfs)
        self._start_load = _read_loadavg(self._procfs)
        if self._server_pid is not None:
            self._start_server = _read_proc_cpu_time(self._procfs / str(self._server_pid) / "stat")
            self._start_threads = {
                tid: value
                for tid in self._task_pids(self._server_pid)
                if (
                    value := _read_proc_cpu_time(
                        self._procfs / str(self._server_pid) / "task" / str(tid) / "stat"
                    )
                )
                is not None
            }
        if self._client_pid is not None:
            self._start_client = _read_proc_cpu_time(self._procfs / str(self._client_pid) / "stat")
        self._started = True

    def stop(self, *, stream_hours: float | None = None) -> HostCounters:
        if not self._started:
            self.start()
        wall = time.monotonic() - self._start_wall
        if wall <= 0:
            wall = 1e-9
        end_total, end_steal = _read_proc_stat_cpu(self._procfs)
        end_nr, end_usec = _read_cgroup_throttled(self._cgroupfs)
        end_load = _read_loadavg(self._procfs)
        total_delta = max(end_total - self._start_total, 0)
        steal_delta = max(end_steal - self._start_steal, 0)
        steal_pct = (steal_delta / total_delta * 100.0) if total_delta else 0.0
        psi_some, psi_full = _read_pressure_avg(self._procfs)
        cpuset_text = _read_text_file(self._cgroupfs / "cpuset.cpus.effective")
        cpuset_size = _cpuset_size(cpuset_text)
        try:
            hertz = os.sysconf("SC_CLK_TCK")
        except (AttributeError, ValueError, OSError):
            hertz = 100
        server_pct: float | None = None
        hottest: float | None = None
        server_threads: int | None = None
        server_cpu_s: float | None = None
        if self._server_pid is not None:
            end_server = _read_proc_cpu_time(self._procfs / str(self._server_pid) / "stat")
            pids = self._task_pids(self._server_pid)
            server_threads = len(pids) if pids else None
            if end_server is not None and self._start_server is not None:
                delta_jiffies = max(end_server - self._start_server, 0)
                delta_s = delta_jiffies / float(hertz)
                server_pct = delta_s / wall / float(cpuset_size) * 100.0
                if stream_hours:
                    server_cpu_s = delta_s / float(stream_hours)
            thread_pcts: list[float] = []
            for tid in pids:
                end_value = _read_proc_cpu_time(
                    self._procfs / str(self._server_pid) / "task" / str(tid) / "stat"
                )
                start_value = self._start_threads.get(tid)
                if end_value is None or start_value is None:
                    continue
                thread_delta_s = max(end_value - start_value, 0) / float(hertz)
                thread_pcts.append(thread_delta_s / wall * 100.0)
            hottest = max(thread_pcts) if thread_pcts else None
        client_pct = 0.0
        client_processes = 1
        if self._client_pid is not None:
            end_client = _read_proc_cpu_time(self._procfs / str(self._client_pid) / "stat")
            if end_client is not None and self._start_client is not None:
                delta_jiffies = max(end_client - self._start_client, 0)
                client_pct = delta_jiffies / float(hertz) / wall / float(cpuset_size) * 100.0
        return HostCounters(
            steal_pct=steal_pct,
            psi_cpu_some_avg=psi_some,
            psi_cpu_full_avg=psi_full,
            cgroup_nr_throttled_delta=max(end_nr - self._start_nr, 0),
            cgroup_throttled_usec_delta=max(end_usec - self._start_usec, 0),
            loadavg_start=self._start_load,
            loadavg_end=end_load,
            server_cpu_pct_of_cpuset=server_pct,
            server_hottest_thread_pct=hottest,
            server_threads=server_threads,
            client_cpu_pct_of_cpuset=client_pct,
            client_processes=client_processes,
            server_cpu_s_per_stream_hour=server_cpu_s,
        )


__all__ = [
    "COMPARABILITY_FIELDS",
    "BoxFacts",
    "EnvironmentRecord",
    "EnvironmentRefusal",
    "FakeGpuProbe",
    "GpuFacts",
    "GpuProbe",
    "HarnessIdentity",
    "HostCounters",
    "HostSampler",
    "SmiProbe",
    "box_id",
    "collect",
    "fake_gpu_facts",
    "harness_identity",
    "parse_smi_xml",
    "read_box",
    "refuse_if_unfit",
    "same_box",
    "single_thread_score",
]
