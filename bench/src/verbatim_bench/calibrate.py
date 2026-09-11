# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The calibration the frozen document asks for: the pressure thresholds, from the null
floor, on the box, by the harness itself.

``benchmarks/METHODOLOGY.md`` section 7 leaves ``PSI_CPU_SOME_MAX_PCT`` and
``PSI_CPU_FULL_MAX_PCT`` null on the freeze date: a threshold still null is filled from a
calibration performed by the harness itself on the box, in a one-line commit made before
the first run. Nobody types a number. This module is the instrument, and its record is
the provenance that one-line commit cites.

**Calibrated against the null floor, not an idle box.** The validity gate exists to reject
a rung whose host was under pressure the measurement cannot survive, not to assert the
machine was doing nothing. A clean run of the generator itself produces pressure, so an
idle-box threshold would fail the first honest run on its own validity gate. The null
floor is this methodology's defined clean baseline, the same generator under the same
framing against the harness's own null server, so a run that shows more pressure than
the floor showed is a run whose host was doing more than the floor's host was, and that
is the condition to reject.

**On the gate's own estimator, which is exact.** ``rung_validity`` compares what
``HostSampler.stop`` computed for the window: the delta of ``/proc/pressure/cpu``'s
monotonic ``total`` stall counters over the window, as a percentage of it, for ``some`` and
for ``full`` each from its own counter. That is the window's pressure and nothing else: no
time constant, no decay, no memory of what came before the window opened. The earlier
estimator, ``avg60`` read at the close, was an exponentially decaying average that carried
the minute before the window into it; it let the tail of a test suite that had ended
seconds before a calibration set a threshold of 0.46 where the quiet box gives 0.01, and it
would have let a rung be invalidated by pressure that ended before its window opened. The
``avg60`` samples are still polled inside every window and kept in the record as context
while the two are compared. The threshold is the maximum over every clean window's
pressure, rounded up to ``PRECISION_PCT``. No safety factor: a factor would be an unfrozen
constant chosen by hand, the thing this avoids.

**Over the concurrencies the method contemplates.** Client-side pressure rises with N, so a
threshold calibrated at one low N would reject an honest run at a higher one: a guard that
cannot pass. The floor is driven at ``NULL_FLOOR_NS``, three windows at each, and the
threshold is the maximum over all of them; the constants are the frozen document's own
(``LADDER_N0_WITHOUT_CEILING``, the only concurrency it names for a ladder without a
ceiling, and ``CEILING_BATCH_SIZES``, which bracket every concurrency a ladder will reach),
so no number here is anyone's choice. A window that did not drive cleanly, a session error
or the generator over its own frozen pacing tolerance, is recorded with its reason and
excluded from the maximum rather than dropped: if the box cannot drive the largest N, that is
a finding in the record. The limit to know: a ladder that climbs past the largest N calibrated
can exceed a threshold taken below it; that shows as rungs invalid for pressure, never as a
silent pass, and the answer is to re-calibrate higher and record why, never to raise the
number by hand.

**Or against the server under test, which is where the row is taken.** On a box that
serves and measures at once, the clean baseline is not the null floor: it includes the
server, whose work is what a row exists to measure, and a threshold taken without it has
no room for it. The first capacity search showed exactly that: the null floor gave 0.40,
the idle box alone stalls 0.32, the generator adds about 0.07, and the server under test
about 0.02, so a real rung at 16 streams read 0.42 and was rejected by the five percent
the server costs. Given an ``endpoint``, the calibration runs the complete measurement,
generator and server together, at ``CALIBRATION_REFERENCE_N`` streams, with the same
rule. The reference is what bounds the circularity: a threshold taken with the server
running absorbs whatever pressure that server causes, so it must be taken in a state
shown healthy by a different instrument, not assumed healthy by this one. Six streams
was measured independently at a p95 of 193.7 ms against a 310 ms budget, 116 ms of
margin, before this calibration existed. The record names the mode, the endpoint and
the reference, so a reader can tell a null-floor threshold from one taken under test.

**No warm-up by default.** A rung's warm-up is the convergence protocol at N: the window
opens when two consecutive readings of the server's p95 agree within a fraction. The
null server answers in a fraction of a millisecond, so that fraction is noise and the
protocol converges by luck; the floor is at steady state the moment every stream is live,
which is when its window opens. The window itself is the rung's, in length, framing and
profile. Pass ``warm_up_s`` to run the protocol anyway; the record says which was done.

**It refuses on a box that is not quiet.** A calibration taken under load would silently
raise the bar for every future run and nothing downstream could detect it. Before the
runs, with the generator not running, the box is observed for ``QUIET_OBSERVATION_S``, one
time constant of the load average it reads, and its own pressure over that observation is
recorded from the same counters. During it:
steal must be at most ``STEAL_PCT_MAX``, the cgroup throttling delta at most
``CGROUP_THROTTLED_DELTA_MAX``, the one-minute load average at most
``QUIET_LOADAVG_MAX_FRACTION_OF_CPUSET`` of the effective cpuset (a tenth), and no compute process
but the named server on the GPU. Pressure is recorded during the observation for the
record but is not a gate: the instrument cannot gate on the number it exists to produce.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from verbatim_bench import constants
from verbatim_bench.client import ChunkMode
from verbatim_bench.env import (
    GpuFacts,
    GpuProbe,
    HostSampler,
    _cpuset_size,
    _read_pressure_avg,
    _read_text_file,
    box_id,
    harness_identity,
    own_cgroup,
    read_box,
)
from verbatim_bench.hostrecord import HostWindow, WindowRecorder, run_load_recorded
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.pace import LoadSpec

__all__ = [
    "CALIBRATION_REFERENCE_N",
    "NULL_FLOOR_NS",
    "PRECISION_PCT",
    "QUIET_LOADAVG_MAX_FRACTION_OF_CPUSET",
    "QUIET_OBSERVATION_S",
    "CalibrationRecord",
    "CalibrationRefusal",
    "QuietReading",
    "WindowReading",
    "calibrate",
    "ceil_to_precision",
    "observe_quiet",
    "thresholds_from",
]

PRECISION_PCT: Final = 0.01
#: The concurrencies the null floor is driven at: the frozen document's own numbers.
NULL_FLOOR_NS: Final = (constants.LADDER_N0_WITHOUT_CEILING, *constants.CEILING_BATCH_SIZES)
#: The concurrency a calibration against the server under test runs at: a state shown
#: healthy by a different instrument (six streams at p95 193.7 ms against a 310 ms
#: budget), which is what bounds the circularity of calibrating with the server running.
CALIBRATION_REFERENCE_N: Final = 6
#: One time constant of the one-minute load average the quiet test reads. The pressure
#: estimator no longer needs the wait: a counter delta over a window has no memory.
QUIET_OBSERVATION_S: Final = 60.0
#: A tenth of the cpuset: on a 48-CPU box a load average of 4.8, which the box's own
#: services sit under and a paced load of even a few streams does not.
QUIET_LOADAVG_MAX_FRACTION_OF_CPUSET: Final = 0.1


class CalibrationRefusal(RuntimeError):
    """The instrument would not calibrate: the box was not quiet, or a run did not hold a
    window, or the box exposes no pressure information."""


@dataclass(frozen=True, slots=True)
class QuietReading:
    """What the box did while nothing of ours ran, and whether that passes the quiet test."""

    duration_s: float
    steal_pct: float
    cgroup_nr_throttled_delta: int
    cgroup_throttled_usec_delta: int
    loadavg_1m: float
    cpuset_size: int
    psi_some: float | None
    psi_full: float | None
    foreign_gpu_pids: tuple[int, ...]
    refusal: str | None
    psi_some_start: float | None = None
    psi_full_start: float | None = None
    psi_some_window_pct: float | None = None
    psi_full_window_pct: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "steal_pct": self.steal_pct,
            "cgroup_nr_throttled_delta": self.cgroup_nr_throttled_delta,
            "cgroup_throttled_usec_delta": self.cgroup_throttled_usec_delta,
            "loadavg_1m": self.loadavg_1m,
            "cpuset_size": self.cpuset_size,
            "loadavg_max_fraction_of_cpuset": QUIET_LOADAVG_MAX_FRACTION_OF_CPUSET,
            "psi_some_window_pct": self.psi_some_window_pct,
            "psi_full_window_pct": self.psi_full_window_pct,
            "avg60_some_start": self.psi_some_start,
            "avg60_full_start": self.psi_full_start,
            "avg60_some_end": self.psi_some,
            "avg60_full_end": self.psi_full,
            "foreign_gpu_pids": list(self.foreign_gpu_pids),
            "quiet": self.refusal is None,
            "refusal": self.refusal,
        }


def observe_quiet(
    *,
    duration_s: float = QUIET_OBSERVATION_S,
    procfs: Path = Path("/proc"),
    cgroupfs: Path | None = None,
    gpu: GpuProbe | None = None,
    gpu_index: int = 0,
    server_pid: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> QuietReading:
    """Observe the box for ``duration_s`` with the generator not running, and judge it."""
    if cgroupfs is None:
        cgroupfs = own_cgroup(procfs)
    sampler = HostSampler(server_pid=None, client_pid=os.getpid(), procfs=procfs, cgroupfs=cgroupfs)
    some_start, full_start = _read_pressure_avg(procfs)
    sampler.start()
    sleep(duration_s)
    counters = sampler.stop()
    cpuset = _cpuset_size(_read_text_file(cgroupfs / "cpuset.cpus.effective"))
    some, full = _read_pressure_avg(procfs)
    foreign: tuple[int, ...] = ()
    if gpu is not None:
        try:
            facts = gpu.facts(gpu_index)
        except (IndexError, ValueError, OSError):
            facts = None
        if facts is not None:
            foreign = tuple(sorted(p for p in facts.compute_process_pids if p != server_pid))
    refusal: str | None = None
    if counters.steal_pct > constants.STEAL_PCT_MAX:
        refusal = f"steal {counters.steal_pct:.2f}% over {constants.STEAL_PCT_MAX}"
    elif (
        counters.cgroup_nr_throttled_delta > constants.CGROUP_THROTTLED_DELTA_MAX
        or counters.cgroup_throttled_usec_delta > constants.CGROUP_THROTTLED_DELTA_MAX
    ):
        refusal = (
            f"cgroup throttled {counters.cgroup_nr_throttled_delta} periods, "
            f"{counters.cgroup_throttled_usec_delta} us, "
            f"over {constants.CGROUP_THROTTLED_DELTA_MAX}"
        )
    elif counters.loadavg_end > QUIET_LOADAVG_MAX_FRACTION_OF_CPUSET * cpuset:
        refusal = (
            f"load average {counters.loadavg_end:.2f} over "
            f"{QUIET_LOADAVG_MAX_FRACTION_OF_CPUSET} of a {cpuset}-CPU cpuset"
        )
    elif foreign:
        refusal = f"foreign GPU compute process(es) {list(foreign)}"
    return QuietReading(
        duration_s=duration_s,
        steal_pct=counters.steal_pct,
        cgroup_nr_throttled_delta=counters.cgroup_nr_throttled_delta,
        cgroup_throttled_usec_delta=counters.cgroup_throttled_usec_delta,
        loadavg_1m=counters.loadavg_end,
        cpuset_size=cpuset,
        psi_some=some,
        psi_full=full,
        foreign_gpu_pids=foreign,
        refusal=refusal,
        psi_some_start=some_start,
        psi_full_start=full_start,
        psi_some_window_pct=counters.psi_cpu_some_avg,
        psi_full_window_pct=counters.psi_cpu_full_avg,
    )


@dataclass(frozen=True, slots=True)
class WindowReading:
    """One null-floor window as the calibration saw it."""

    n: int
    seed: int
    window_open_s: float
    window_close_s: float
    warm_up_converged: bool | None
    psi_some_window_pct: float | None
    psi_full_window_pct: float | None
    psi_samples: int
    psi_some_max: float | None
    psi_full_max: float | None
    psi_some_at_close: float | None
    psi_full_at_close: float | None
    client_cpu_pct_of_cpuset: float
    pacing_slip_p99_ms: float | None
    sessions: int = 0
    sessions_failed: int = 0
    clean: bool = True
    unclean_reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "clean": self.clean,
            "unclean_reason": self.unclean_reason,
            "sessions": self.sessions,
            "sessions_failed": self.sessions_failed,
            "seed": self.seed,
            "window_open_s": self.window_open_s,
            "window_close_s": self.window_close_s,
            "warm_up_converged": self.warm_up_converged,
            "psi_some_window_pct": self.psi_some_window_pct,
            "psi_full_window_pct": self.psi_full_window_pct,
            "avg60_samples": self.psi_samples,
            "avg60_some_max": self.psi_some_max,
            "avg60_full_max": self.psi_full_max,
            "avg60_some_at_close": self.psi_some_at_close,
            "avg60_full_at_close": self.psi_full_at_close,
            "client_cpu_pct_of_cpuset": self.client_cpu_pct_of_cpuset,
            "pacing_slip_p99_ms": self.pacing_slip_p99_ms,
        }


def ceil_to_precision(value: float, precision: float = PRECISION_PCT) -> float:
    """Round up to the stated precision; a threshold never sits below what was seen."""
    steps = math.ceil(round(value / precision, 6))
    return round(steps * precision, 6)


def thresholds_from(
    windows: Sequence[WindowReading], *, precision: float = PRECISION_PCT
) -> tuple[float, float]:
    """The two thresholds: the maximum over every clean window's pressure, rounded up."""
    clean = [w for w in windows if w.clean]
    somes = [w.psi_some_window_pct for w in clean if w.psi_some_window_pct is not None]
    fulls = [w.psi_full_window_pct for w in clean if w.psi_full_window_pct is not None]
    if not clean or not somes or not fulls:
        raise CalibrationRefusal(
            "no pressure samples inside any clean window: the box exposes no "
            "/proc/pressure/cpu, no window was held, or no window drove cleanly"
        )
    return ceil_to_precision(max(somes), precision), ceil_to_precision(max(fulls), precision)


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """The two thresholds with everything needed to know where they came from."""

    psi_cpu_some_max_pct: float
    psi_cpu_full_max_pct: float
    precision_pct: float
    taken_at: str
    box_id: str
    boot_id: str
    cpu_model: str
    cgroup_cpu_max: str
    cpuset_effective: str
    gpu_uuid: str | None
    harness_version: str
    harness_git_sha: str | None
    harness_tree_clean: bool
    manifest: str
    ns: tuple[int, ...]
    seeds: tuple[int, ...]
    window_s: float
    warm_up_s: float | None
    frame_ms: int
    pacing_profile: str
    interval_s: float
    canonical: bool
    quiet: QuietReading
    windows: tuple[WindowReading, ...]
    mode: str = "null-floor"
    endpoint: str | None = None
    server_pid: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "vb-psi-calibration/1",
            "thresholds": {
                "PSI_CPU_SOME_MAX_PCT": self.psi_cpu_some_max_pct,
                "PSI_CPU_FULL_MAX_PCT": self.psi_cpu_full_max_pct,
                "precision_pct": self.precision_pct,
                "rule": "ceiling to precision of the maximum over every clean null-floor "
                "window's pressure, the delta of /proc/pressure/cpu's total stall counter over "
                "the window as a percentage of it, some and full each from its own counter, "
                "no factor; avg60 samples recorded as context only",
            },
            "taken_at": self.taken_at,
            "box": {
                "id": self.box_id,
                "boot_id": self.boot_id,
                "cpu_model": self.cpu_model,
                "cgroup_cpu_max": self.cgroup_cpu_max,
                "cpuset_effective": self.cpuset_effective,
                "gpu_uuid": self.gpu_uuid,
            },
            "harness": {
                "version": self.harness_version,
                "git_sha": self.harness_git_sha,
                "tree_clean": self.harness_tree_clean,
            },
            "load": {
                "mode": self.mode,
                "server": (
                    "the harness's own null server (verbatim_bench.nullserver)"
                    if self.endpoint is None
                    else f"the server under test at {self.endpoint}"
                ),
                "endpoint": self.endpoint,
                "server_pid": self.server_pid,
                "reference_n": CALIBRATION_REFERENCE_N if self.endpoint is not None else None,
                "manifest": self.manifest,
                "ns": list(self.ns),
                "windows_per_n": len(self.seeds),
                "seeds": list(self.seeds),
                "clean_windows": sum(1 for w in self.windows if w.clean),
                "unclean_windows": [
                    {"n": w.n, "seed": w.seed, "reason": w.unclean_reason}
                    for w in self.windows
                    if not w.clean
                ],
                "window_s": self.window_s,
                "warm_up_s": self.warm_up_s,
                "frame_ms": self.frame_ms,
                "pacing_profile": self.pacing_profile,
                "gpu_sample_interval_s": self.interval_s,
                "canonical": self.canonical,
            },
            "quiet": self.quiet.to_json_dict(),
            "windows": [w.to_json_dict() for w in self.windows],
        }

    def constants_lines(self) -> str:
        """The one-line commit's two lines, as they go into `constants.py`."""
        return (
            f"PSI_CPU_SOME_MAX_PCT: Final[float | None] = {self.psi_cpu_some_max_pct}\n"
            f"PSI_CPU_FULL_MAX_PCT: Final[float | None] = {self.psi_cpu_full_max_pct}\n"
        )


def _reading(result: Any, host: HostWindow, *, n: int, seed: int) -> WindowReading:
    from verbatim_bench.results import percentile

    slips = [v for s in result.sessions for v in s.pacing_slip_ms]
    failed = sum(1 for s in result.sessions if s.error is not None)
    slip_p99 = percentile(slips, 99) if slips else None
    reason: str | None = None
    if failed:
        reason = f"{failed} of {len(result.sessions)} session(s) failed"
    elif slip_p99 is not None and slip_p99 > constants.PACING_SLIP_P99_MAX_MS:
        reason = (
            f"pacing slip p99 {slip_p99:.2f} ms over the frozen "
            f"{constants.PACING_SLIP_P99_MAX_MS} ms: the generator did not drive N={n} cleanly"
        )
    return WindowReading(
        n=n,
        seed=seed,
        window_open_s=host.open_s,
        window_close_s=host.close_s,
        warm_up_converged=result.warm_up_converged,
        psi_some_window_pct=host.counters.psi_cpu_some_avg,
        psi_full_window_pct=host.counters.psi_cpu_full_avg,
        psi_samples=host.psi.samples,
        psi_some_max=host.psi.some_max,
        psi_full_max=host.psi.full_max,
        psi_some_at_close=host.counters.psi_cpu_some_avg,
        psi_full_at_close=host.counters.psi_cpu_full_avg,
        client_cpu_pct_of_cpuset=host.counters.client_cpu_pct_of_cpuset,
        pacing_slip_p99_ms=slip_p99,
        sessions=len(result.sessions),
        sessions_failed=failed,
        clean=reason is None,
        unclean_reason=reason,
    )


async def calibrate(
    *,
    manifest: Path,
    endpoint: str | None = None,
    ns: Sequence[int] | None = None,
    seeds: Sequence[int] = constants.SEEDS,
    window_s: float = constants.WINDOW_S,
    warm_up_s: float | None = None,
    warm_up_reading_s: float = constants.WARM_UP_READING_S,
    warm_up_convergence: float = constants.WARM_UP_CONVERGENCE,
    warm_up_cap_s: float = constants.WARM_UP_CAP_S,
    ramp_s: float = 0.0,
    frame_ms: int = constants.FRAME_MS,
    interval_s: float = 1.0,
    quiet_s: float = QUIET_OBSERVATION_S,
    gpu: GpuProbe | None = None,
    gpu_index: int = 0,
    server_pid: int | None = None,
    procfs: Path = Path("/proc"),
    cgroupfs: Path | None = None,
    sysfs: Path = Path("/sys"),
    repo_root: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    precision: float = PRECISION_PCT,
) -> CalibrationRecord:
    """Observe the box quiet, run the floor once per seed at each N with the window recorder,
    and reduce the windows' pressure to the two thresholds with their provenance. Without
    an ``endpoint`` the floor is the harness's null server at ``NULL_FLOOR_NS``; with one it
    is the server under test at ``CALIBRATION_REFERENCE_N``, which then needs ``server_pid``."""
    if ns is None:
        ns = NULL_FLOOR_NS if endpoint is None else (CALIBRATION_REFERENCE_N,)
    if endpoint is not None and server_pid is None:
        raise CalibrationRefusal(
            "a calibration against the server under test needs its pid, so its CPU is in "
            "the record and its compute process is not counted foreign"
        )
    if cgroupfs is None:
        cgroupfs = own_cgroup(procfs)
    quiet = observe_quiet(
        duration_s=quiet_s,
        procfs=procfs,
        cgroupfs=cgroupfs,
        gpu=gpu,
        gpu_index=gpu_index,
        server_pid=server_pid,
        sleep=sleep,
    )
    if quiet.refusal is not None:
        raise CalibrationRefusal(f"the box is not quiet: {quiet.refusal}")
    gpu_facts: GpuFacts | None = None
    if gpu is not None:
        try:
            gpu_facts = gpu.facts(gpu_index)
        except (IndexError, ValueError, OSError):
            gpu_facts = None
    windows: list[WindowReading] = []
    async with AsyncExitStack() as stack:
        if endpoint is None:
            target = (await stack.enter_async_context(NullServer(NullServerConfig()))).endpoint
        else:
            target = endpoint
        for n, seed in ((n, seed) for n in ns for seed in seeds):
            spec = LoadSpec(
                endpoint=target,
                manifest=Path(manifest),
                sessions=n,
                chunk=ChunkMode.parse(160),
                profile=constants.PACING_PROFILE,
                seed=seed,
                ramp_s=ramp_s,
                frame_ms=frame_ms,
                window_s=window_s,
                warm_up_s=warm_up_s,
                warm_up_reading_s=warm_up_reading_s,
                warm_up_convergence=warm_up_convergence,
                warm_up_cap_s=warm_up_cap_s,
                arm="null-floor",
            )
            recorder = WindowRecorder(
                gpu=gpu if gpu is not None else _NoGpu(),
                gpu_index=gpu_index,
                server_pid=server_pid,
                client_pid=os.getpid(),
                interval_s=interval_s,
                procfs=procfs,
                cgroupfs=cgroupfs,
            )
            result = await run_load_recorded(spec, recorder)
            host = recorder.finish()
            if host is None or result.warm_up_converged is False:
                raise CalibrationRefusal(
                    f"the calibration run at N={n} with seed {seed} held no window: warm-up "
                    f"converged={result.warm_up_converged}; the calibration measures the "
                    "window a rung measures and nothing else"
                )
            windows.append(_reading(result, host, n=n, seed=seed))
    some, full = thresholds_from(windows, precision=precision)
    box = read_box(procfs=procfs, sysfs=sysfs, cgroupfs=cgroupfs)
    identity = harness_identity(repo_root if repo_root is not None else Path.cwd())
    return CalibrationRecord(
        psi_cpu_some_max_pct=some,
        psi_cpu_full_max_pct=full,
        precision_pct=precision,
        taken_at=datetime.now(UTC).isoformat(),
        box_id=box_id(box, gpu_facts),
        boot_id=box.boot_id,
        cpu_model=box.cpu_model,
        cgroup_cpu_max=box.cgroup_cpu_max,
        cpuset_effective=box.cpuset_effective,
        gpu_uuid=gpu_facts.uuid if gpu_facts is not None else None,
        harness_version=identity.version,
        harness_git_sha=identity.git_sha,
        harness_tree_clean=identity.tree_clean,
        manifest=str(manifest),
        ns=tuple(ns),
        seeds=tuple(seeds),
        window_s=float(window_s),
        warm_up_s=None if warm_up_s is None else float(warm_up_s),
        frame_ms=frame_ms,
        pacing_profile=constants.PACING_PROFILE,
        interval_s=interval_s,
        mode="null-floor" if endpoint is None else "under-test",
        endpoint=endpoint,
        server_pid=server_pid,
        # Canonical: the frozen window, framing and seeds, and the mode's own N. The
        # warm-up is not part of it; the floor has none by design, and the record names
        # what ran.
        canonical=(
            float(window_s) == float(constants.WINDOW_S)
            and frame_ms == constants.FRAME_MS
            and tuple(seeds) == tuple(constants.SEEDS)
            and tuple(ns)
            == (tuple(NULL_FLOOR_NS) if endpoint is None else (CALIBRATION_REFERENCE_N,))
        ),
        quiet=quiet,
        windows=tuple(windows),
    )


class _NoGpu:
    """A probe for a box with no GPU record to sample: every read is an absent device."""

    def count(self) -> int:
        return 0

    def facts(self, index: int) -> GpuFacts:
        raise IndexError("no GPU probe was given to the calibration")
