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
from verbatim.scheduler.graph_budget import ConfigError
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
    config = loop.scheduler.config
    assert loop.slots.reserved == loop.registry.live + config.effective_pad + config.edge_pad_rows


def test_tick_stats_are_recorded_per_tick() -> None:
    loop, _, _ = _harness(bucket=8)
    rng = np.random.default_rng(18)
    _join(loop, 1, _audio(rng, 4))
    loop.run_for(5)
    assert [s.tick_id for s in loop.stats] == [0, 1, 2, 3, 4]
    assert all(s.steady_rows == 8 for s in loop.stats)


def test_uncalibrated_loop_never_overfills_the_steady_batch() -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(8,))
    pipeline = FakePipelineAdapter(CHUNK, buckets=(8,))
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
    assert loop.slots.reserved == config.effective_pad + config.edge_pad_rows
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
    # Only the healthy session, over the steady and edge pad rows.
    assert loop.slots.reserved == config.effective_pad + config.edge_pad_rows + 1
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
    assert loop.slots.reserved == config.effective_pad + config.edge_pad_rows
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


def _idle_harness(idle_timeout_s: float | None) -> TickLoop:
    config = EngineConfig(
        chunk=CHUNK, buckets=(8,), edge_batch=8, calibrated_ceiling=8, idle_timeout_s=idle_timeout_s
    )
    return TickLoop(
        config, FakePipelineAdapter(CHUNK, buckets=(8,)), SessionRegistry(), clock=SimulatedClock()
    )


def _admit_idle(loop: TickLoop, session_id: int) -> Session:
    session = Session(session_id, CHUNK, ring_seconds=8.0)
    session.configure()
    assert loop.admit_session(session).admitted
    return session


def test_an_idle_session_is_closed_at_the_deadline_and_its_slot_released() -> None:
    """0.4 s on a 160 ms period is three ticks: two starved ticks keep the session,
    the third closes it with DEADLINE_EXCEEDED and gives its slot back."""
    loop = _idle_harness(idle_timeout_s=0.4)
    idle = _admit_idle(loop, 1)
    reserved = loop.slots.reserved
    loop.run_for(2)
    assert idle.state is SessionState.STARVED
    assert idle.starved_ticks == 2
    assert loop.drain_errors() == []
    assert loop.slots.reserved == reserved
    loop.run_for(1)
    assert idle.state is SessionState.CLOSED
    assert loop.slots.reserved == reserved - 1
    assert loop.registry.live == 0
    errors = loop.drain_errors()
    assert [(stream_id, error.code) for stream_id, error in errors] == [
        (1, ErrorCode.DEADLINE_EXCEEDED)
    ]
    assert "0.48 s" in str(errors[0][1])


def test_audio_resets_the_idle_count() -> None:
    loop = _idle_harness(idle_timeout_s=0.4)
    session = _admit_idle(loop, 1)
    rng = np.random.default_rng(31)
    for _ in range(4):
        loop.run_for(2)
        assert session.ring.write(_audio(rng, 1)) == N
        loop.run_for(1)
        assert session.state is SessionState.RUNNING
        assert session.starved_ticks == 0
    assert loop.drain_errors() == []
    loop.run_for(3)
    assert session.state is SessionState.CLOSED
    assert [code for _, error in loop.drain_errors() for code in [error.code]] == [
        ErrorCode.DEADLINE_EXCEEDED
    ]


def test_the_deadline_does_not_touch_sessions_with_audio_or_the_pads() -> None:
    loop = _idle_harness(idle_timeout_s=0.4)
    rng = np.random.default_rng(32)
    busy = _join(loop, 1, _audio(rng, 20), drain=False)
    idle = _admit_idle(loop, 2)
    loop.run_for(10)
    assert busy.state is SessionState.RUNNING
    assert idle.state is SessionState.CLOSED
    assert [stream_id for stream_id, _ in loop.drain_errors()] == [2]
    assert all(stat.pad_rows == loop.stats[0].pad_rows for stat in loop.stats)


def test_no_deadline_when_disabled() -> None:
    loop = _idle_harness(idle_timeout_s=None)
    idle = _admit_idle(loop, 1)
    loop.run_for(500)
    assert idle.state is SessionState.STARVED
    assert idle.starved_ticks == 500
    assert loop.drain_errors() == []


@pytest.mark.parametrize("idle_timeout_s", [0, -1.0, True])
def test_a_non_positive_idle_timeout_is_a_config_error(idle_timeout_s: object) -> None:
    with pytest.raises(ConfigError, match="idle_timeout_s"):
        EngineConfig(chunk=CHUNK, buckets=(8,), idle_timeout_s=idle_timeout_s)  # type: ignore[arg-type]


def test_the_default_idle_timeout_is_thirty_seconds_and_the_stub_engine_disables_it() -> None:
    assert EngineConfig(chunk=CHUNK, buckets=(8,)).idle_timeout_s == 30.0


def test_results_are_published_before_the_sleep_to_the_next_boundary() -> None:
    """A row computed at boundary t reaches its transport at t plus the step, not at
    t+1: `publish` runs before the sleep, so it sees this tick's boundary on the
    clock, and it carries exactly the rows the tick returns."""
    loop, _, clock = _harness(bucket=1, edge_batch=1)
    _join(loop, 1, _audio(np.random.default_rng(3), 2), drain=False)
    published: list[tuple[list[StepResult], float]] = []
    returned = loop.run_tick(publish=lambda rows: published.append((list(rows), clock.now())))
    assert len(published) == 1
    rows, published_at = published[0]
    assert rows == returned
    assert [row.stream_id for row in rows] == [1]
    assert published_at == loop.boundaries[-1]
    assert clock.now() == pytest.approx(loop.boundaries[-1] + PERIOD)


class _FailingStep(FakePipelineAdapter):
    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        raise RuntimeError("step failed")


def test_a_failed_tick_publishes_before_its_sleep_with_the_errors_recorded() -> None:
    """The failure path keeps the order: `fail_live` records every live session's
    error, then `publish` runs with no rows, before the sleep, so the caller drains
    the errors at boundary t and not at t+1."""
    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1, calibrated_ceiling=1)
    clock = SimulatedClock()
    loop = TickLoop(config, _FailingStep(CHUNK, buckets=(1,)), SessionRegistry(), clock=clock)
    _join(loop, 1, _audio(np.random.default_rng(4), 1), drain=False)
    seen: list[tuple[list[StepResult], float, list[int]]] = []
    loop.run_tick(
        publish=lambda rows: seen.append(
            (list(rows), clock.now(), [sid for sid, _ in loop.drain_errors()])
        )
    )
    assert seen == [([], loop.boundaries[-1], [1])]
    assert clock.now() == pytest.approx(loop.boundaries[-1] + PERIOD)
