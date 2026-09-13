# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The paced real-time load generator: wall-clock, not as-fast-as-possible.

``N`` concurrent sessions, each streaming its corpus session in real time in
fixed-size PCM frames, with a stagger profile and seeded jitter. Timestamps come
from one monotonic clock in the generator process; the server is never asked for
its own latency figures.

A windowed load runs three phases in order, and a sample belongs to the phase it
arrived in:

* the **ramp**, which ends when every one of the ``N`` slots has opened its first
  session, so the warm-up clock starts with all ``N`` streams live rather than with
  the first of them;
* the **warm-up** at ``N``, which takes readings of ``warm_up_reading_s`` and opens
  the window once two consecutive readings agree within ``warm_up_convergence``,
  never before ``warm_up_s`` has elapsed and never after ``warm_up_cap_s``, at which
  point the load never settled and no window opens at all;
* the **measurement window** of ``window_s``, throughout which closed-loop
  replacement keeps concurrency at ``N``: a slot whose session finishes sleeps
  ``replace_gap_s`` and starts the next one.

Only window samples are a measurement. Warm-up samples stay on the sessions that
produced them and are excluded from the window by ``window_partial_samples``. Without
``window_s`` the generator runs each slot once and expresses no window at all, which
is a smoke run and never a rung.

One rung is not always one process. ``spec.slots`` names the rung slots this process
drives and ``spec.total_sessions`` the rung's whole stream count, and ``WindowGate``
takes the decision of when the window opens out of this process and gives it to the
rung. ``verbatim_bench.multiproc`` is the coordinator that uses both; everything here
runs one event loop and does not know how many others there are.

"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from verbatim_bench import constants
from verbatim_bench.client import ChunkMode, SessionResult, run_session
from verbatim_bench.corpus import (
    load_manifest,
    manifest_corpus_id,
    read_pcm16,
    validate_pcm_sha256,
)
from verbatim_bench.results import RunResult, percentile

Profile = Literal["uniform", "burst", "bursty", "diurnal"]

#: The stagger used when the caller names none. Not a frozen methodology constant: the
#: document fixes the warm-up and the window and requires only that all ``N`` streams
#: are live before the warm-up clock starts, which holds for any ramp length.
DEFAULT_RAMP_S: Final = 60.0


@dataclass(frozen=True, slots=True)
class LoadSpec:
    endpoint: str
    manifest: Path
    sessions: int
    chunk: ChunkMode
    profile: Profile = "uniform"
    seed: int = 20260914
    ramp_s: float = DEFAULT_RAMP_S
    burst_window_s: float = 2.0
    words: bool = False
    lang: str = "en-US"
    arm: str = "unknown"
    x_ms: int = constants.X_MS
    frame_ms: int = constants.FRAME_MS
    replace_gap_s: float = 0.0
    window_s: float | None = None
    warm_up_s: float | None = None
    warm_up_reading_s: float = constants.WARM_UP_READING_S
    warm_up_convergence: float = constants.WARM_UP_CONVERGENCE
    warm_up_cap_s: float = constants.WARM_UP_CAP_S
    #: The rung slots this process drives, when it drives only some of them; ``None`` is
    #: the whole rung, ``range(sessions)``. A share of a rung names its slots by their
    #: index in the RUNG and never by their index in the share, because everything a slot
    #: seeds -- its place in the ramp, which utterance it replays, its frame jitter -- has
    #: to come out the same however many processes the rung was split across.
    slots: tuple[int, ...] | None = None
    #: The rung's whole stream count when this spec drives a share of it; ``None`` when
    #: ``sessions`` is already the whole rung.
    total_sessions: int | None = None

    def __post_init__(self) -> None:
        if self.slots is None:
            return
        if self.total_sessions is None:
            raise ValueError("slots names rung slots, so total_sessions must say how many")
        if len(self.slots) != self.sessions:
            raise ValueError(
                f"slots holds {len(self.slots)} entries but sessions is {self.sessions}"
            )
        if len(set(self.slots)) != len(self.slots):
            raise ValueError(f"slots repeats a rung slot: {self.slots!r}")
        if any(not 0 <= slot < self.total_sessions for slot in self.slots):
            raise ValueError(
                f"slots {self.slots!r} leaves the rung, which has {self.total_sessions} slots"
            )

    @property
    def rung_sessions(self) -> int:
        """The rung's whole stream count, whether or not this spec drives all of it."""
        return self.total_sessions if self.total_sessions is not None else self.sessions

    @property
    def slot_indices(self) -> tuple[int, ...]:
        """The rung slots this spec drives, in the order it drives them."""
        return self.slots if self.slots is not None else tuple(range(self.sessions))


@dataclass(frozen=True, slots=True)
class WindowHooks:
    """Called by `run_load` on the loop thread at the two moments a window is defined by:
    the instant it opens and the instant it closes. They bound whatever a caller samples
    across the window to the window itself, and they must not block."""

    on_open: Callable[[], None]
    on_close: Callable[[], None]


@dataclass(slots=True)
class _Phases:
    """What the load did with its clock, filled in as each phase ends."""

    wall_start_s: float
    all_live_at_s: float | None = None
    warm_up_end_s: float | None = None
    window_open_s: float | None = None
    window_close_s: float | None = None
    converged: bool | None = None
    readings: list[float | None] = field(default_factory=list)


def _is_burst(profile: str) -> bool:
    return profile in ("burst", "bursty")


def plan_start_offsets(spec: LoadSpec) -> list[float]:
    """Per-session start delay in seconds, deterministic for a given seed.

    uniform: `i * ramp_s / max(1, sessions - 1)` (all zero when sessions == 1),
             plus a seeded uniform phase offset in [0, chunk.ms / 1000).
    burst/bursty: a seeded uniform draw in [0, burst_window_s), plus the same
             phase offset (`bursty` is the accepted alias for `burst`).
    diurnal: declared in the vocabulary only; the arrival pattern is not
             implemented here, so planning falls back to the uniform stagger.

    The phase offset is what stops a server from hiding a tick-alignment wait.

    The plan is always the whole rung's, drawn in rung slot order, and a spec that drives
    only a share of the rung takes its own entries out of it. That is what keeps a rung
    split across several processes on the schedule the same rung runs on in one: the
    draws are the same draws in the same order, nothing is re-seeded per process, and the
    returned list is still one entry per session this spec runs.
    """
    rng = random.Random(spec.seed)
    period_s = spec.chunk.ms / 1000.0
    total = spec.rung_sessions
    offsets: list[float] = []
    for i in range(total):
        phase = rng.random() * period_s
        if _is_burst(spec.profile):
            base = rng.random() * spec.burst_window_s
        else:
            base = i * spec.ramp_s / max(1, total - 1)
        offsets.append(base + phase)
    return [offsets[slot] for slot in spec.slot_indices]


def pacing_slip_samples(result: RunResult) -> list[list[float]]:
    """Return the per-session pacing-slip samples carried by a load result."""
    return [list(session.pacing_slip_ms) for session in result.sessions]


def window_partial_samples(result: RunResult) -> list[float]:
    """Return only the partial-latency samples that arrived inside the window.

    A run whose window never opened took no measurement, so this is empty rather than
    a percentile over the warm-up.
    """
    open_s, close_s = result.window_open_s, result.window_close_s
    if open_s is None or close_s is None:
        return []
    return [
        latency
        for session in result.sessions
        for latency, recv in zip(session.partial_ms, session.partial_recv_s, strict=True)
        if open_s <= recv <= close_s
    ]


def executed_canonical_window(result: RunResult) -> bool:
    """Whether the load that ran had the frozen warm-up and measurement window.

    Read from the run, never from the command line. A load that expressed no window, or
    whose warm-up never settled, or whose window opened before the frozen warm-up had
    elapsed or closed before the frozen window had, is not canonical however the
    operator spelled the arguments. Comparing arguments with constants is a guard that
    cannot fail: it was true of every rung produced on 2026-09-11, none of which ran a
    measurement window at all.
    """
    if result.warm_up_converged is not True:
        return False
    window_s = result.window_length_s
    warm_up_s = result.warm_up_length_s
    if window_s is None or warm_up_s is None:
        return False
    if window_s < constants.WINDOW_S:
        return False
    if warm_up_s < constants.WARM_UP_S or warm_up_s > constants.WARM_UP_CAP_S:
        return False
    spec = result.spec_dict
    return (
        spec.get("window_s") == constants.WINDOW_S
        and spec.get("warm_up_s") == constants.WARM_UP_S
        and spec.get("warm_up_reading_s") == constants.WARM_UP_READING_S
        and spec.get("warm_up_convergence") == constants.WARM_UP_CONVERGENCE
        and spec.get("warm_up_cap_s") == constants.WARM_UP_CAP_S
    )


def _readings_agree(previous: float | None, current: float | None, tolerance: float) -> bool:
    """Two readings agree when they differ by at most `tolerance` of the larger.

    A reading over an interval that produced no sample is `None`: nothing was measured,
    so nothing agrees with it, and a load that emits nothing never opens a window.
    """
    if previous is None or current is None:
        return False
    return abs(current - previous) <= tolerance * max(previous, current)


class WindowGate(Protocol):
    """Decides, for the whole rung, the one instant its measurement window opens.

    A rung is not always one process. When it is several, the window has to open at the
    same instant on the same clock in all of them: a percentile pooled over windows that
    did not overlap is not a percentile of anything the server did. So the decision does
    not belong to whichever process happens to be holding the samples, and ``run_load``
    does not make it. It reports what it saw through these three calls and is told the
    answer.

    All three are awaited on the load's own event loop, and every one of them must be
    bounded: a gate that waits on another process has to give up and answer rather than
    hold a rung open for as long as that process stays silent.
    """

    async def all_live(self, at_s: float) -> float:
        """Report that this process's slots are all live at ``at_s``.

        Returns the instant the RUNG's warm-up begins, which is when the last of its
        processes came live. Every process reads its warm-up floor and its reading grid
        from that one instant, so the readings the gate pools cover the same intervals.
        """

    async def reading(self, values: Sequence[float], *, start_s: float, end_s: float) -> None:
        """Offer the partial latencies this process matched in one reading interval.

        The raw samples, not their percentile: the rung's reading is the percentile of
        the pooled sample, and percentiles do not pool.
        """

    async def opens_at(self) -> float | None:
        """Block until the rung decides, then return the instant its window opens on the
        shared monotonic clock, or ``None`` if the rung never settled."""


class LocalGate:
    """The gate of a rung that is one process, which therefore decides on its own readings.

    This is the protocol ``run_load`` applied inline before a rung could be more than one
    process, moved behind the seam and otherwise unchanged: two consecutive readings that
    agree within ``warm_up_convergence`` open the window, never before ``warm_up_s`` has
    passed since all streams went live and never after ``warm_up_cap_s``, at which point
    the load never settled and no window opens at all.
    """

    def __init__(self, spec: LoadSpec) -> None:
        self._warm_up_s = spec.warm_up_s
        self._convergence = spec.warm_up_convergence
        self._cap_s = spec.warm_up_cap_s
        self._all_live_at_s: float | None = None
        self._previous: float | None = None
        self._opens_at: float | None = None
        self._decided = asyncio.Event()

    async def all_live(self, at_s: float) -> float:
        self._all_live_at_s = at_s
        if self._warm_up_s is None:
            # No warm-up was asked for, so the window opens the moment the load is at N.
            self._opens_at = at_s
            self._decided.set()
        return at_s

    async def reading(self, values: Sequence[float], *, start_s: float, end_s: float) -> None:
        value = percentile(values, 95) if values else None
        if not self._decided.is_set() and _readings_agree(self._previous, value, self._convergence):
            floor_at = (self._all_live_at_s or end_s) + (self._warm_up_s or 0.0)
            self._opens_at = max(end_s, floor_at)
            self._decided.set()
        self._previous = value

    async def opens_at(self) -> float | None:
        if self._all_live_at_s is None:
            raise RuntimeError("opens_at() was awaited before all_live()")
        if not self._decided.is_set():
            # Bounded by the cap: the warm-up gets `warm_up_cap_s` from all streams live
            # to settle, and a load that has not settled by then opens no window.
            remaining = self._all_live_at_s + self._cap_s - time.monotonic()
            if remaining > 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._decided.wait(), timeout=remaining)
        return self._opens_at if self._decided.is_set() else None


def spec_dict(spec: LoadSpec) -> dict[str, Any]:
    """The record of the load a result was produced by.

    One definition, because a rung combined from several processes has to write the same
    shape as the same rung run in one.
    """
    return {
        "endpoint": spec.endpoint,
        "chunk_ms": spec.chunk.ms,
        "seed": spec.seed,
        "sessions": spec.sessions,
        "profile": spec.profile,
        "words": spec.words,
        "arm": spec.arm,
        "x_ms": spec.x_ms,
        "frame_ms": spec.frame_ms,
        "ramp_s": spec.ramp_s,
        "replace_gap_s": spec.replace_gap_s,
        "window_s": spec.window_s,
        "warm_up_s": spec.warm_up_s,
        "warm_up_reading_s": spec.warm_up_reading_s,
        "warm_up_convergence": spec.warm_up_convergence,
        "warm_up_cap_s": spec.warm_up_cap_s,
        "slots": list(spec.slots) if spec.slots is not None else None,
        "total_sessions": spec.total_sessions,
    }


async def run_load(
    spec: LoadSpec, *, hooks: WindowHooks | None = None, gate: WindowGate | None = None
) -> RunResult:
    """Open every session per the plan, gather the results, and reduce them.

    With ``window_s`` set this runs the ramp, the warm-up and the measurement window
    described in this module's docstring, and the returned result carries the wall-clock
    moment each phase began and ended. Without it each slot runs once and no window is
    expressed. Every session carries its own pacing-slip samples for the row's validity
    accounting.

    A slot in flight when the window closes is allowed to finish rather than cancelled,
    so ``wall_clock_s`` exceeds the ramp, warm-up and window by up to one session. The
    window itself is exactly the two timestamps, and nothing outside it is a sample.

    ``gate`` decides when the window opens; the default is ``LocalGate``, which is this
    process deciding on its own readings and is the protocol this function applied inline
    before a rung could be split. A rung that IS split hands every process the same gate
    implementation talking to one coordinator, and the window opens at one instant in all
    of them. The two phase timestamps this returns are that instant and that instant plus
    ``window_s`` exactly, not the clock read after the sleep that waited for them, because
    every process has to record the same window down to the bit for their samples to pool.

    ``spec.slots`` names the rung slots this process drives. Everything a slot seeds is
    keyed off the RUNG slot index, never off the position in this process's own list.
    """
    utterances = load_manifest(spec.manifest)
    if not utterances:
        raise ValueError(f"manifest {spec.manifest} has no entries")
    if gate is not None and spec.window_s is None:
        raise ValueError("a window gate decides when a window opens; this spec has none")
    pcm_cache: dict[str, bytes] = {}
    for utterance in utterances:
        pcm = read_pcm16(utterance.audio_path)
        validate_pcm_sha256(pcm, utterance.sha256, path=utterance.audio_path)
        pcm_cache[utterance.stream_id] = pcm
    offsets = plan_start_offsets(spec)
    slot_indices = spec.slot_indices
    window_gate: WindowGate = gate if gate is not None else LocalGate(spec)
    started_at = datetime.now(UTC)
    wall_start = time.monotonic()
    phases = _Phases(wall_start_s=wall_start)
    samples: list[tuple[float, float]] = []
    ramp_done = asyncio.Event()
    stop = asyncio.Event()
    resolved = 0
    reading = spec.warm_up_s is not None

    def _resolve_slot() -> None:
        nonlocal resolved
        resolved += 1
        if resolved >= spec.sessions:
            ramp_done.set()

    def _on_sample(latency_ms: float, recv_s: float) -> None:
        # Only the warm-up reads samples as they arrive. Window samples are read back
        # off the sessions that carry them, so this buffer holds one reading at a time
        # and is dropped the moment the window opens.
        if reading:
            samples.append((latency_ms, recv_s))

    async def _one(
        slot: int,
        ordinal: int,
        delay: float,
        on_open: Callable[[], None] | None = None,
    ) -> SessionResult:
        utterance = utterances[(slot + ordinal) % len(utterances)]
        return await run_session(
            spec.endpoint,
            session_id=f"s{slot:04d}-{ordinal:04d}",
            utterance=utterance,
            pcm=pcm_cache[utterance.stream_id],
            chunk=spec.chunk,
            start_delay_s=delay,
            words=spec.words,
            lang=spec.lang,
            frame_ms=spec.frame_ms,
            frame_seed=spec.seed + slot,
            on_open=on_open,
            on_sample=_on_sample,
        )

    async def _slot(position: int) -> list[SessionResult]:
        # `slot` is the slot's index in the RUNG. It names the session, picks the
        # utterance and seeds the frame jitter, so a rung split four ways runs the same
        # sessions in the same slots as the same rung run in one process.
        slot = slot_indices[position]
        collected: list[SessionResult] = []
        if spec.window_s is None:
            collected.append(await _one(slot, 0, offsets[position]))
            return collected
        opened = False

        def _on_open() -> None:
            nonlocal opened
            if not opened:
                opened = True
                _resolve_slot()

        ordinal = 0
        delay = offsets[position]
        while True:
            session = await _one(slot, ordinal, delay, on_open=_on_open)
            collected.append(session)
            if ordinal == 0 and not opened:
                # The first attempt never reached the server. The ramp is over for this
                # slot all the same, or one slot that cannot connect would hold the
                # warm-up clock for the whole run.
                opened = True
                _resolve_slot()
            if stop.is_set():
                break
            if session.error is not None and session.chunks == 0:
                # A slot that cannot open a session is not holding concurrency at N, and
                # retrying as fast as the kernel can refuse only loads the client.
                break
            ordinal += 1
            delay = spec.replace_gap_s
        return collected

    async def _take_readings(origin_s: float, decision: asyncio.Task[float | None]) -> None:
        """Offer the gate one reading per interval on the rung's grid, until it decides.

        The grid runs from `origin_s`, the instant the rung's last process came live, so
        every process's k-th reading covers the same interval and the gate pools samples
        that were taken at the same time.
        """
        index = 0
        while not decision.done():
            start_at = origin_s + index * spec.warm_up_reading_s
            end_at = start_at + spec.warm_up_reading_s
            samples.clear()
            remaining = end_at - time.monotonic()
            if remaining > 0:
                done, _ = await asyncio.wait({decision}, timeout=remaining)
                if done:
                    return
            values = [latency for latency, recv in samples if start_at <= recv < end_at]
            # This process's own reading, kept on this process's own record. The rung's
            # reading is the gate's, over every process's samples.
            phases.readings.append(percentile(values, 95) if values else None)
            await window_gate.reading(values, start_s=start_at, end_s=end_at)
            index += 1

    async def _supervise() -> None:
        nonlocal reading
        await ramp_done.wait()
        origin_s = await window_gate.all_live(time.monotonic())
        phases.all_live_at_s = origin_s
        decision: asyncio.Task[float | None] = asyncio.create_task(window_gate.opens_at())
        try:
            if spec.warm_up_s is not None:
                await _take_readings(origin_s, decision)
            opens_at = await decision
        finally:
            reading = False
            samples.clear()
            if not decision.done():
                decision.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await decision
        phases.converged = None if spec.warm_up_s is None else opens_at is not None
        if opens_at is None:
            phases.warm_up_end_s = time.monotonic()
            stop.set()
            return
        lead = opens_at - time.monotonic()
        if lead > 0:
            await asyncio.sleep(lead)
        phases.warm_up_end_s = opens_at
        phases.window_open_s = opens_at
        if hooks is not None:
            hooks.on_open()
        closes_at = opens_at + (spec.window_s or 0.0)
        remaining = closes_at - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        phases.window_close_s = closes_at
        if hooks is not None:
            hooks.on_close()
        stop.set()

    supervisor: asyncio.Task[None] | None = None
    if spec.window_s is not None:
        supervisor = asyncio.create_task(_supervise())
    try:
        nested = list(await asyncio.gather(*(_slot(i) for i in range(spec.sessions))))
    finally:
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await supervisor
    sessions = [session for group in nested for session in group]
    wall_clock_s = time.monotonic() - wall_start
    return RunResult(
        spec_dict=spec_dict(spec),
        sessions=sessions,
        wall_clock_s=wall_clock_s,
        corpus_id=manifest_corpus_id(spec.manifest),
        manifest_name=Path(spec.manifest).name,
        started_at=started_at,
        wall_start_s=phases.wall_start_s,
        all_live_at_s=phases.all_live_at_s,
        warm_up_end_s=phases.warm_up_end_s,
        window_open_s=phases.window_open_s,
        window_close_s=phases.window_close_s,
        warm_up_converged=phases.converged,
        warm_up_readings=tuple(phases.readings),
    )
