# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The seeded frame jitter is on the wire: the sender sleeps to the jittered deadline
and is graded against the same one, so the slip measures the generator's lateness
and nothing else."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from verbatim_bench import constants
from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.nullserver import NullServer, NullServerConfig

pytestmark = pytest.mark.cpu

_FRAME_BYTES = 20 * 16000 // 1000 * 2


class _PerfectSleeper:
    """A clock that a sleep advances exactly: the generator keeps every deadline it sets
    itself, so any slip it then records is the grading disagreeing with the schedule."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay_s: float) -> None:
        self.sleeps.append(delay_s)
        self.now += delay_s


async def _one_session(seed: int) -> tuple[list[float], list[float]]:
    pcm = bytes(_FRAME_BYTES * 24)  # three 160 ms chunks, 24 wire frames
    utterance = Utterance(stream_id="jit-0", audio_path=Path("jit.wav"), duration_s=0.48, text="")
    sleeper = _PerfectSleeper()
    async with NullServer(NullServerConfig()) as server:
        result = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterance,
            pcm=pcm,
            chunk=ChunkMode.parse("160ms"),
            start_delay_s=0.0,
            clock=sleeper.clock,
            sleep=sleeper.sleep,
            frame_ms=20,
            frame_seed=seed,
        )
    assert result.error is None
    assert result.chunks == 3
    return result.pacing_slip_ms, sleeper.sleeps


async def test_a_generator_that_keeps_its_own_schedule_records_no_slip() -> None:
    """With a perfect sleeper every send lands exactly on the deadline the sender slept
    to. Graded against a different deadline, one carrying a jitter the sleep did not,
    up to FRAME_JITTER_MS of slip appears out of nothing; graded against the same one,
    none does."""
    slips, sleeps = await _one_session(seed=20260915)
    assert len(slips) == 24
    assert slips == [0.0] * 24
    assert len(sleeps) == 23


async def test_the_seeded_jitter_shapes_the_wire_schedule() -> None:
    """The sleeps between frames carry the seeded draw, one per frame in frame order,
    within +-FRAME_JITTER_MS of the nominal period, and the same seed reproduces them."""
    _, sleeps = await _one_session(seed=20260915)
    _, again = await _one_session(seed=20260915)
    assert sleeps == again
    jitter_s = constants.FRAME_JITTER_MS / 1000.0
    rng = random.Random(20260915)
    expected = []
    previous = 0.0
    for index in range(1, 24):
        deadline = index * 0.020 + rng.uniform(-jitter_s, jitter_s)
        expected.append(deadline - previous)
        previous = deadline
    assert sleeps == pytest.approx(expected, abs=1e-9)
    assert any(abs(s - 0.020) > 0.001 for s in sleeps)  # the jitter is really there
    assert all(abs(s - 0.020) <= 2 * jitter_s + 1e-9 for s in sleeps)


async def test_chunk_sized_framing_draws_no_jitter() -> None:
    """Non-canonical framing, one frame per chunk, is unjittered: the draw is a property
    of the frozen 20 ms workload, not of the client."""
    pcm = bytes(160 * 16000 // 1000 * 2 * 3)
    utterance = Utterance(stream_id="jit-1", audio_path=Path("jit.wav"), duration_s=0.48, text="")
    sleeper = _PerfectSleeper()
    async with NullServer(NullServerConfig()) as server:
        result = await run_session(
            server.endpoint,
            session_id="s0001",
            utterance=utterance,
            pcm=pcm,
            chunk=ChunkMode.parse("160ms"),
            start_delay_s=0.0,
            clock=sleeper.clock,
            sleep=sleeper.sleep,
            frame_ms=None,
        )
    assert result.error is None
    assert result.pacing_slip_ms == [0.0] * 3
    assert sleeper.sleeps == pytest.approx([0.160, 0.160], abs=1e-9)
