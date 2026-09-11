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

"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

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
    """
    rng = random.Random(spec.seed)
    period_s = spec.chunk.ms / 1000.0
    offsets: list[float] = []
    for i in range(spec.sessions):
        phase = rng.random() * period_s
        if _is_burst(spec.profile):
            base = rng.random() * spec.burst_window_s
        else:
            base = i * spec.ramp_s / max(1, spec.sessions - 1)
        offsets.append(base + phase)
    return offsets


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


async def run_load(spec: LoadSpec, *, hooks: WindowHooks | None = None) -> RunResult:
    """Open every session per the plan, gather the results, and reduce them.

    With ``window_s`` set this runs the ramp, the warm-up and the measurement window
    described in this module's docstring, and the returned result carries the wall-clock
    moment each phase began and ended. Without it each slot runs once and no window is
    expressed. Every session carries its own pacing-slip samples for the row's validity
    accounting.

    A slot in flight when the window closes is allowed to finish rather than cancelled,
    so ``wall_clock_s`` exceeds the ramp, warm-up and window by up to one session. The
    window itself is exactly the two timestamps, and nothing outside it is a sample.
    """
    utterances = load_manifest(spec.manifest)
    if not utterances:
        raise ValueError(f"manifest {spec.manifest} has no entries")
    pcm_cache: dict[str, bytes] = {}
    for utterance in utterances:
        pcm = read_pcm16(utterance.audio_path)
        validate_pcm_sha256(pcm, utterance.sha256, path=utterance.audio_path)
        pcm_cache[utterance.stream_id] = pcm
    offsets = plan_start_offsets(spec)
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

    async def _slot(slot: int) -> list[SessionResult]:
        collected: list[SessionResult] = []
        if spec.window_s is None:
            collected.append(await _one(slot, 0, offsets[slot]))
            return collected
        opened = False

        def _on_open() -> None:
            nonlocal opened
            if not opened:
                opened = True
                _resolve_slot()

        ordinal = 0
        delay = offsets[slot]
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

    async def _warm_up(start: float) -> float | None:
        """Return the moment the window may open, or None if the load never settled."""
        cap_at = start + spec.warm_up_cap_s
        previous: float | None = None
        converged_at: float | None = None
        while converged_at is None:
            samples.clear()
            reading_start = time.monotonic()
            if reading_start + spec.warm_up_reading_s > cap_at:
                return None
            await asyncio.sleep(spec.warm_up_reading_s)
            reading_end = time.monotonic()
            values = [latency for latency, recv in samples if reading_start <= recv < reading_end]
            reading = percentile(values, 95) if values else None
            phases.readings.append(reading)
            if _readings_agree(previous, reading, spec.warm_up_convergence):
                converged_at = reading_end
            previous = reading
        floor_at = start + (spec.warm_up_s or 0.0)
        if converged_at < floor_at:
            await asyncio.sleep(floor_at - converged_at)
            return time.monotonic()
        return converged_at

    async def _supervise() -> None:
        nonlocal reading
        await ramp_done.wait()
        phases.all_live_at_s = time.monotonic()
        if spec.warm_up_s is None:
            phases.warm_up_end_s = phases.all_live_at_s
            phases.converged = None
        else:
            opens_at = await _warm_up(phases.all_live_at_s)
            reading = False
            samples.clear()
            phases.warm_up_end_s = opens_at if opens_at is not None else time.monotonic()
            phases.converged = opens_at is not None
            if opens_at is None:
                stop.set()
                return
        phases.window_open_s = phases.warm_up_end_s
        if hooks is not None:
            hooks.on_open()
        remaining = phases.window_open_s + (spec.window_s or 0.0) - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        phases.window_close_s = time.monotonic()
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
        spec_dict={
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
        },
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
