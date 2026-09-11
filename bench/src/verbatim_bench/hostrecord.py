# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The host record of one measurement window: what the box and the GPU did while a rung ran.

``benchmarks/METHODOLOGY.md`` section 56 names zero GPU throttle events during the window
as the fourth criterion of a rung, and section 7 names the host thresholds a rung must meet
to be valid at all. Both are properties of the window itself, so both need sampling across
it: DR-0006 rejects a reading taken near the run as a different measurement wearing the
name. ``WindowRecorder`` does that sampling. The load generator calls its hooks the moment
the window opens and closes, which bounds the existing ``HostSampler``'s deltas to the
window exactly; the ladder polls the GPU probe at an interval while the load runs; and
every sample is stamped on the monotonic clock the generator stamps the window with, so a
sample outside the window is discarded rather than counted.

Only reasons that are throttling count as events: the power cap and the hardware and
software slowdowns. ``gpu_idle`` is active whenever the GPU has nothing to do, which
between ticks is most of the time, and the applications-clocks, sync-boost and display-clock
reasons are settings; none of them is the GPU refusing to run at speed.

The GPU is read through the ``GpuProbe`` seam the environment record already uses, so a
test drives this with a scripted probe and a fake clock, and the same code runs against
``nvidia-smi`` on a box.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from verbatim_bench.env import GpuFacts, GpuProbe, HostCounters, HostSampler, SmiProbe
from verbatim_bench.pace import LoadSpec, WindowHooks, run_load
from verbatim_bench.results import RunResult

__all__ = [
    "DEFAULT_INTERVAL_S",
    "THROTTLING_SUFFIXES",
    "GpuSample",
    "GpuWindow",
    "HostWindow",
    "LiveSmiProbe",
    "WindowRecorder",
    "run_load_recorded",
    "summarise_gpu",
    "throttling_reasons",
]

#: The reason names, by suffix, that mean the GPU is being held below speed. Both
#: spellings nvidia-smi has used (``clocks_throttle_reason_*`` and
#: ``clocks_event_reason_*``) share these suffixes.
THROTTLING_SUFFIXES: Final = (
    "_sw_power_cap",
    "_hw_slowdown",
    "_hw_thermal_slowdown",
    "_hw_power_brake_slowdown",
    "_sw_thermal_slowdown",
)
#: How often the GPU is polled across the window. One ``nvidia-smi -q -x`` costs tens of
#: milliseconds on a worker thread; a throttle episode lasts far longer than a second.
DEFAULT_INTERVAL_S: Final = 1.0


def throttling_reasons(reasons: Mapping[str, int]) -> dict[str, int]:
    """The subset of a probe's reasons that are throttling, each 1 when active."""
    return {
        name: (1 if value else 0)
        for name, value in reasons.items()
        if name.endswith(THROTTLING_SUFFIXES)
    }


@dataclass(frozen=True, slots=True)
class GpuSample:
    at_s: float
    facts: GpuFacts


@dataclass(frozen=True, slots=True)
class GpuWindow:
    """What the GPU reported inside one window, from the samples that fell in it.

    ``throttle_reasons`` and ``compute_process_pids`` carry the names ``rung_validity``
    reads: the first counts, per throttling reason, the samples on which it was active;
    the second is every compute process seen in the window other than the server's own.
    ``covered`` is whether the sampling spanned the window: no gap between consecutive
    samples, nor from the open to the first or the last to the close, longer than twice
    the interval. Thermal is evaluated only over a covered window.
    """

    samples: int
    interval_s: float
    covered: bool
    max_gap_s: float | None
    throttle_events: int
    throttle_reasons: Mapping[str, int]
    compute_process_pids: tuple[int, ...]
    persistence_mode: bool | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "interval_s": self.interval_s,
            "covered": self.covered,
            "max_gap_s": self.max_gap_s,
            "throttle_events": self.throttle_events,
            "throttle_reasons": dict(self.throttle_reasons),
            "foreign_pids": list(self.compute_process_pids),
            "persistence_mode": self.persistence_mode,
        }


@dataclass(frozen=True, slots=True)
class HostWindow:
    """The host record of one window: the sampler's deltas and the GPU's samples."""

    counters: HostCounters
    gpu: GpuWindow
    open_s: float
    close_s: float

    def to_json_dict(self) -> dict[str, Any]:
        c = self.counters
        return {
            "window_open_s": self.open_s,
            "window_close_s": self.close_s,
            "counters": {
                "steal_pct_window": c.steal_pct,
                "psi_cpu_some_avg_window": c.psi_cpu_some_avg,
                "psi_cpu_full_avg_window": c.psi_cpu_full_avg,
                "cgroup_nr_throttled_delta": c.cgroup_nr_throttled_delta,
                "cgroup_throttled_usec_delta": c.cgroup_throttled_usec_delta,
                "loadavg_start": c.loadavg_start,
                "loadavg_end": c.loadavg_end,
                "server_cpu_pct_of_cpuset": c.server_cpu_pct_of_cpuset,
                "server_hottest_thread_pct": c.server_hottest_thread_pct,
                "server_threads": c.server_threads,
                "client_cpu_pct_of_cpuset": c.client_cpu_pct_of_cpuset,
                "client_processes": c.client_processes,
                "server_cpu_s_per_stream_hour": c.server_cpu_s_per_stream_hour,
            },
            "gpu": self.gpu.to_json_dict(),
        }


def summarise_gpu(
    samples: Sequence[GpuSample],
    *,
    open_s: float,
    close_s: float,
    interval_s: float,
    server_pid: int | None,
) -> GpuWindow:
    """Reduce the samples that fell inside ``[open_s, close_s]`` to the window's record."""
    inside = sorted((s for s in samples if open_s <= s.at_s <= close_s), key=lambda s: s.at_s)
    if not inside:
        return GpuWindow(
            samples=0,
            interval_s=interval_s,
            covered=False,
            max_gap_s=None,
            throttle_events=0,
            throttle_reasons={},
            compute_process_pids=(),
            persistence_mode=None,
        )
    stamps = [s.at_s for s in inside]
    gaps = [stamps[0] - open_s, close_s - stamps[-1]]
    gaps.extend(b - a for a, b in pairwise(stamps))
    max_gap = max(gaps)
    events = 0
    per_reason: dict[str, int] = {}
    pids: set[int] = set()
    persistence: bool | None = None
    for sample in inside:
        active = throttling_reasons(sample.facts.throttle_reasons)
        if any(active.values()):
            events += 1
        for name, value in active.items():
            per_reason[name] = per_reason.get(name, 0) + value
        pids.update(sample.facts.compute_process_pids)
        mode = sample.facts.persistence_mode
        if mode is False:
            persistence = False
        elif mode is True and persistence is None:
            persistence = True
    if server_pid is not None:
        pids.discard(server_pid)
    return GpuWindow(
        samples=len(inside),
        interval_s=interval_s,
        covered=max_gap <= 2.0 * interval_s,
        max_gap_s=max_gap,
        throttle_events=events,
        throttle_reasons=per_reason,
        compute_process_pids=tuple(sorted(pids)),
        persistence_mode=persistence,
    )


class LiveSmiProbe:
    """A probe that runs ``nvidia-smi`` on every call, for polling; ``SmiProbe`` parses one
    document once, which is right for the environment record and wrong for a window."""

    def __init__(self, argv: Sequence[str] = ("nvidia-smi", "-q", "-x")) -> None:
        self._argv = tuple(argv)

    def count(self) -> int:
        return SmiProbe.from_command(self._argv).count()

    def facts(self, index: int) -> GpuFacts:
        return SmiProbe.from_command(self._argv).facts(index)


class WindowRecorder:
    """Records one window: the host sampler between the hooks, the GPU probe at an interval.

    ``on_window_open`` and ``on_window_close`` are the hooks ``run_load`` calls, on the
    loop thread, at the two moments the window is defined by. ``poll`` takes one GPU
    sample now; ``run`` polls until told to stop, each poll on a worker thread because a
    live probe is a subprocess. ``finish`` reduces it all to a ``HostWindow``, or None
    when no window was recorded.
    """

    def __init__(
        self,
        *,
        gpu: GpuProbe,
        gpu_index: int = 0,
        server_pid: int | None = None,
        client_pid: int | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        procfs: Path = Path("/proc"),
        cgroupfs: Path = Path("/sys/fs/cgroup"),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s!r}")
        self._gpu = gpu
        self._gpu_index = gpu_index
        self._server_pid = server_pid
        self._interval_s = interval_s
        self._clock = clock
        self._sampler = HostSampler(
            server_pid=server_pid, client_pid=client_pid, procfs=procfs, cgroupfs=cgroupfs
        )
        self._samples: list[GpuSample] = []
        self._open_s: float | None = None
        self._close_s: float | None = None
        self._counters: HostCounters | None = None
        self.probe_errors = 0

    @property
    def hooks(self) -> WindowHooks:
        return WindowHooks(on_open=self.on_window_open, on_close=self.on_window_close)

    def on_window_open(self) -> None:
        self._open_s = self._clock()
        self._sampler.start()

    def on_window_close(self) -> None:
        self._counters = self._sampler.stop()
        self._close_s = self._clock()

    def poll(self) -> None:
        at = self._clock()
        try:
            facts = self._gpu.facts(self._gpu_index)
        except (IndexError, ValueError, OSError, subprocess.SubprocessError):
            self.probe_errors += 1
            return
        self._samples.append(GpuSample(at_s=at, facts=facts))

    async def run(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            await loop.run_in_executor(None, self.poll)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                continue

    def finish(self) -> HostWindow | None:
        if self._open_s is None or self._close_s is None or self._counters is None:
            return None
        gpu = summarise_gpu(
            self._samples,
            open_s=self._open_s,
            close_s=self._close_s,
            interval_s=self._interval_s,
            server_pid=self._server_pid,
        )
        return HostWindow(
            counters=self._counters, gpu=gpu, open_s=self._open_s, close_s=self._close_s
        )


async def run_load_recorded(spec: LoadSpec, recorder: WindowRecorder) -> RunResult:
    """``run_load`` with the recorder's hooks on the window and its poller alongside."""
    stop = asyncio.Event()
    poller = asyncio.create_task(recorder.run(stop))
    try:
        return await run_load(spec, hooks=recorder.hooks)
    finally:
        stop.set()
        await poller
