# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Tick-loop acceptance tests: fixed boundaries, steady shape under churn, eager budget."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import ErrorCode
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session, SessionState
from verbatim.core.types import PcmFrame, StepResult
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.tick import STATS_RETAINED, TickLoop

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


class _RecordingPipeline(FakePipelineAdapter):
    """The CPU fake, plus a record of closes, a failing open and a failing step."""

    def __init__(
        self,
        chunk: ChunkMode,
        *,
        buckets: Sequence[int],
        fail_open: Sequence[int] = (),
        fail_on_call: int | None = None,
    ) -> None:
        super().__init__(chunk, buckets=buckets)
        self.closed: list[int] = []
        self.stepped: list[int] = []
        self.calls = 0
        self._fail_open = set(fail_open)
        self._fail_on_call = fail_on_call

    def open_stream(self, stream_id: int, options: SessionOptions | None) -> None:
        if stream_id in self._fail_open:
            raise RuntimeError("open failed")
        super().open_stream(stream_id, options)

    def close_stream(self, stream_id: int) -> None:
        self.closed.append(stream_id)
        super().close_stream(stream_id)

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        self.calls += 1
        if self.calls == self._fail_on_call:
            raise RuntimeError("step failed")
        self.stepped.extend(f.stream_id for f in frames if f.stream_id >= 0)
        return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)


def _recording_loop(**pipeline_kwargs: object) -> tuple[TickLoop, _RecordingPipeline, EngineConfig]:
    config = EngineConfig(chunk=CHUNK, buckets=(8,), calibrated_ceiling=8)
    pipeline = _RecordingPipeline(CHUNK, buckets=(8,), **pipeline_kwargs)  # type: ignore[arg-type]
    return TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock()), pipeline, config


def test_a_step_failure_fails_and_closes_every_live_session() -> None:
    """A failed step delivers INTERNAL to every live session and closes it in the same
    tick: slot released, registry entry gone, `close_stream` called. Before this the
    sessions stayed registered, were fed again on the next tick, and held their slots
    until a transport happened to abort them."""
    loop, pipeline, config = _recording_loop(fail_on_call=2)
    rng = np.random.default_rng(21)
    sessions = [_join(loop, i + 1, _audio(rng, 4), drain=False) for i in range(3)]
    assert loop.run_tick()
    assert loop.run_tick() == []
    errors = dict(loop.drain_errors())
    assert sorted(errors) == [1, 2, 3]
    assert all(error.code is ErrorCode.INTERNAL for error in errors.values())
    assert all(str(error) == "step failed" for error in errors.values())
    assert all(session.state is SessionState.CLOSED for session in sessions)
    assert loop.registry.live == 0
    assert loop.slots.reserved == config.effective_pad
    assert sorted(pipeline.closed) == [1, 2, 3]
    # Nothing is fed again and no second error is queued.
    assert loop.run_tick() == []
    assert loop.drain_errors() == []


def test_a_session_whose_open_fails_is_closed_in_the_same_tick_and_never_stepped() -> None:
    """The never-opened-stream decision. NeMo has no state for a stream whose open
    failed, so any frame of it, including the synthetic abort frame, would dereference
    None inside a batch and fail every session in it. The session closes in the tick
    its open failed, its slot is released, and no frame of it ever reaches the adapter.
    """
    loop, pipeline, config = _recording_loop(fail_open=(1,))
    rng = np.random.default_rng(22)
    session = _join(loop, 1, _audio(rng, 3), drain=False)
    healthy = _join(loop, 2, _audio(rng, 6), drain=False)
    results = loop.run_tick()
    assert [r.stream_id for r in results] == [2]
    assert dict(loop.drain_errors())[1].code is ErrorCode.INTERNAL
    assert session.state is SessionState.CLOSED
    assert 1 not in loop.registry
    assert loop.slots.reserved == config.effective_pad + 1  # only the healthy session
    assert pipeline.closed == [1]
    assert 1 not in pipeline.stepped
    for _ in range(3):
        loop.run_tick()
    assert 1 not in pipeline.stepped
    assert healthy.state is SessionState.RUNNING


def test_a_draining_session_whose_final_was_consumed_outside_the_loop_is_closed() -> None:
    """The safety net in `run_tick` for a DRAINING session with no frame left. No
    tick-loop path produces the state today, so the test builds it directly: consume
    the final frame through the session's own API, then let a tick find the session.
    Deleting the branch would leave such a session registered with its slot forever."""
    loop, pipeline, config = _recording_loop()
    rng = np.random.default_rng(23)
    session = _join(loop, 1, _audio(rng, 1)[:100])
    consumed = session.next_frame()
    assert consumed is not None and consumed.is_last
    assert session.state is SessionState.DRAINING
    assert loop.run_tick() == []
    assert session.state is SessionState.CLOSED
    assert 1 not in loop.registry
    assert loop.slots.reserved == config.effective_pad
    assert pipeline.closed == [1]


def test_tick_costs_are_read_after_the_step() -> None:
    """An adapter that measures its own step reports this tick's cost, not the last one's."""

    class MeasuringPipeline(FakePipelineAdapter):
        def transcribe_step(self, frames, *, keep_all_outputs):
            self._step_ms = 42.0
            return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)

    config = EngineConfig(chunk=CHUNK, buckets=(8,), calibrated_ceiling=8)
    pipeline = MeasuringPipeline(CHUNK, buckets=(8,), step_ms=1.0)
    loop = TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock())
    loop.run_tick()
    assert loop.stats[-1].step_ms == 42.0


def test_tick_stats_and_boundaries_are_bounded() -> None:
    loop, _, _ = _harness()
    loop.run_for(STATS_RETAINED + 50)
    assert loop.tick_id == STATS_RETAINED + 50
    assert len(loop.stats) == STATS_RETAINED
    assert len(loop.boundaries) == STATS_RETAINED
    assert loop.stats[0].tick_id == 50
    assert loop.stats[-1].tick_id == STATS_RETAINED + 49
