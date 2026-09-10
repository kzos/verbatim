# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The paced real-time load generator: wall-clock, not as-fast-as-possible.

``N`` concurrent sessions, each streaming its corpus session in real time in
fixed-size PCM frames, with a stagger profile and seeded jitter. Timestamps come
from one monotonic clock in the generator process; the server is never asked for
its own latency figures.

Closed-loop replacement keeps concurrency at ``N`` through the window: when a
session in a slot finishes before the window ends, the slot sleeps
``replace_gap_s`` and starts the next session, so the server always sees ``N``
live streams. Without a window the generator runs each slot once, as before.

"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from verbatim_bench import constants
from verbatim_bench.client import ChunkMode, SessionResult, run_session
from verbatim_bench.corpus import (
    load_manifest,
    manifest_corpus_id,
    read_pcm16,
    validate_pcm_sha256,
)
from verbatim_bench.results import RunResult

Profile = Literal["uniform", "burst", "bursty", "diurnal"]


@dataclass(frozen=True, slots=True)
class LoadSpec:
    endpoint: str
    manifest: Path
    sessions: int
    chunk: ChunkMode
    profile: Profile = "uniform"
    seed: int = 20260914
    ramp_s: float = 60.0
    burst_window_s: float = 2.0
    words: bool = False
    lang: str = "en-US"
    arm: str = "unknown"
    x_ms: int = constants.X_MS
    frame_ms: int = constants.FRAME_MS
    replace_gap_s: float = 0.0
    window_s: float | None = None


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


async def run_load(spec: LoadSpec) -> RunResult:
    """Open every session per the plan, gather the results, and reduce them.

    With ``window_s`` set, each of the ``sessions`` slots replaces a finished
    session after ``replace_gap_s`` until the window ends, holding concurrency
    at ``N``; without it each slot runs once. Every session carries its own
    pacing-slip samples for the row's validity accounting.
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
    counter = 0

    async def _one(slot: int, ordinal: int, delay: float) -> SessionResult:
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
        )

    async def _slot(slot: int) -> list[SessionResult]:
        collected: list[SessionResult] = []
        if spec.window_s is None:
            collected.append(await _one(slot, 0, offsets[slot]))
            return collected
        ordinal = 0
        first = True
        while True:
            elapsed = time.monotonic() - wall_start
            if elapsed >= (spec.window_s or 0.0):
                break
            delay = offsets[slot] if first else spec.replace_gap_s
            first = False
            collected.append(await _one(slot, ordinal, delay))
            ordinal += 1
            if time.monotonic() - wall_start >= (spec.window_s or 0.0):
                break
        return collected

    nested = list(await asyncio.gather(*(_slot(i) for i in range(spec.sessions))))
    sessions = [session for group in nested for session in group]
    wall_clock_s = time.monotonic() - wall_start
    _ = counter
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
            "replace_gap_s": spec.replace_gap_s,
            "window_s": spec.window_s,
        },
        sessions=sessions,
        wall_clock_s=wall_clock_s,
        corpus_id=manifest_corpus_id(spec.manifest),
        manifest_name=Path(spec.manifest).name,
        started_at=started_at,
    )
