# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Tick-loop acceptance tests: fixed boundaries, steady shape under churn, eager budget."""

from __future__ import annotations

import random

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session, SessionState
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.tick import TickLoop

CHUNK = ChunkMode(160)
N = CHUNK.samples
PERIOD = CHUNK.period_s


def _harness(
    bucket: int = 8, edge_batch: int = 8, ceiling: int | None = None
) -> tuple[TickLoop, FakePipelineAdapter, SimulatedClock]:
    config = EngineConfig(
        chunk=CHUNK,
        buckets=(bucket,),
        edge_batch=edge_batch,
        calibrated_ceiling=bucket if ceiling is None else ceiling,
    )
    pipeline = FakePipelineAdapter(CHUNK, buckets=(bucket,))
    clock = SimulatedClock()
    loop = TickLoop(config, pipeline, SessionRegistry(), clock=clock)
    return loop, pipeline, clock


def _audio(rng: np.random.Generator, chunks: int, seed_offset: int = 0) -> np.ndarray:
    del seed_offset
    return rng.uniform(-0.5, 0.5, size=chunks * N).astype(np.float32)


def _join(
    loop: TickLoop,
    session_id: int,
    audio: np.ndarray,
    *,
    drain: bool = True,
    ring_seconds: float = 8.0,
) -> Session:
    session = Session(session_id, CHUNK, ring_seconds=ring_seconds)
    session.configure()
    decision = loop.admit_session(session)
    assert decision.admitted, f"session {session_id} refused: {decision.reason}"
    written = session.ring.write(audio)
    assert written == len(audio)
    if drain:
        session.begin_draining()
    return session


def test_boundaries_are_fixed_multiples_of_the_period() -> None:
    loop, _, clock = _harness()
    loop.run_for(10)
    assert loop.boundaries == pytest.approx([i * PERIOD for i in range(10)])
    assert clock.waits == pytest.approx([PERIOD] * 10)


def test_a_late_tick_does_not_shift_later_boundaries() -> None:
    loop, _, clock = _harness()
    loop.run_for(3)
    clock.advance(2.5 * PERIOD)
    loop.run_for(3)
    assert loop.boundaries == pytest.approx([i * PERIOD for i in range(6)])
    # The catch-up ticks waited nothing; the grid itself never moved.
    assert clock.waits[3] == pytest.approx(0.0)


def test_one_row_per_session_per_tick() -> None:
    loop, _, _ = _harness()
    rng = np.random.default_rng(11)
    sessions = [_join(loop, i + 1, _audio(rng, 5)) for i in range(3)]
    del sessions
    results = loop.run_for(6)
    seen: set[tuple[int, int]] = set()
    for result in results:
        key = (result.tick_id, result.stream_id)
        assert key not in seen, f"two rows for session {result.stream_id} on tick {result.tick_id}"
        seen.add(key)


def test_starved_sessions_are_skipped_without_synthetic_audio() -> None:
    loop, _, _ = _harness()
    rng = np.random.default_rng(12)
    _join(loop, 1, _audio(rng, 2))
    starved = Session(2, CHUNK)
    starved.configure()
    assert loop.admit_session(starved).admitted
    results = loop.run_for(3)
    assert [r for r in results if r.stream_id == 2] == []
    assert starved.audio_processed_s == 0.0
    assert starved.ring.available == 0
    assert all(stat.starved >= 1 for stat in loop.stats)


def test_pad_row_results_are_dropped() -> None:
    loop, _, _ = _harness()
    rng = np.random.default_rng(13)
    _join(loop, 1, _audio(rng, 2))
    _join(loop, 2, _audio(rng, 2))
    for result in loop.run_for(3):
        assert result.stream_id > 0


def test_steady_shape_never_changes_across_churn() -> None:
    bucket = 16
    loop, pipeline, _ = _harness(bucket=bucket, ceiling=bucket)
    rng = random.Random(20240501)
    audio_rng = np.random.default_rng(14)
    open_sessions: dict[int, Session] = {}
    next_id = 1
    for _ in range(200):
        for sid in [s for s, sess in open_sessions.items() if sess.state is SessionState.CLOSED]:
            del open_sessions[sid]
        if len(open_sessions) < 10 and rng.random() < 0.5:
            for _ in range(rng.randint(1, 2)):
                session = _join(loop, next_id, _audio(audio_rng, rng.randint(5, 30)))
                open_sessions[next_id] = session
                next_id += 1
        loop.run_tick()
    steady_shapes = {shape for shape, keep_all in pipeline.shapes_seen if not keep_all}
    assert steady_shapes == {bucket}


def test_eager_fraction_stays_within_budget() -> None:
    loop, _, _ = _harness(bucket=8, edge_batch=8)
    rng = np.random.default_rng(15)
    for index in range(4):
        loop.run_for(index * 10 - loop.tick_id if index else 0)
        _join(loop, index + 1, _audio(rng, 60), ring_seconds=12.0)
    loop.run_for(250 - loop.tick_id)
    fraction = loop.admission.eager_step_fraction
    assert fraction <= 0.02


def test_eager_fraction_is_reported_even_when_high() -> None:
    loop, _, _ = _harness(bucket=8, edge_batch=8)
    rng = np.random.default_rng(16)
    for index in range(6):
        _join(loop, index + 1, _audio(rng, 1))
        loop.run_for(2)
    fraction = loop.admission.eager_step_fraction
    assert fraction == pytest.approx(1 / 3)
    assert fraction > 0.02


def test_slot_accounting_reconciles_over_a_long_run() -> None:
    bucket = 8
    loop, _, _ = _harness(bucket=bucket, ceiling=bucket)
    rng = random.Random(77)
    audio_rng = np.random.default_rng(17)
    open_sessions: dict[int, Session] = {}
    next_id = 1
    for _ in range(500):
        for sid in [s for s, sess in open_sessions.items() if sess.state is SessionState.CLOSED]:
            del open_sessions[sid]
        if len(open_sessions) < 6 and rng.random() < 0.6:
            session = _join(loop, next_id, _audio(audio_rng, rng.randint(3, 12)))
            open_sessions[next_id] = session
            next_id += 1
        loop.run_tick()
    live = sum(1 for s in open_sessions.values() if s.state is not SessionState.CLOSED)
    assert live == loop.registry.live
    assert loop.slots.reserved == loop.registry.live + loop.scheduler.config.effective_pad


def test_tick_stats_are_recorded_per_tick() -> None:
    loop, _, _ = _harness(bucket=8)
    rng = np.random.default_rng(18)
    _join(loop, 1, _audio(rng, 4))
    loop.run_for(5)
    assert [s.tick_id for s in loop.stats] == [0, 1, 2, 3, 4]
    assert all(s.steady_rows == 8 for s in loop.stats)


def test_uncalibrated_loop_never_overfills_the_steady_batch() -> None:
    config = EngineConfig(chunk=CHUNK)
    pipeline = FakePipelineAdapter(CHUNK, buckets=config.buckets or (8,))
    loop = TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock())
    rng = np.random.default_rng(19)
    for session_id in range(1, 13):
        session = Session(session_id, CHUNK, ring_seconds=8.0)
        session.configure()
        decision = loop.admit_session(session)
        if decision.admitted:
            assert session.ring.write(_audio(rng, 3)) == 3 * N
    loop.run_for(5)
    assert all(s.steady_rows == 8 for s in loop.stats)
