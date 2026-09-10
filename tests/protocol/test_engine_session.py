# SPDX-License-Identifier: Apache-2.0
"""Engine facade tests: ingress ownership, FIFO delivery and tick-thread seams."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import pytest

from verbatim.audio.pcm import decode_pcm16
from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import ErrorCode, ResourceExhausted, VerbatimError
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.core.types import PcmFrame, StepResult
from verbatim.engine import Engine, stub_engine
from verbatim.pipelines.base import PipelineAdapter
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.base import Hypothesis, SessionHandle, SessionOptions
from verbatim.scheduler.clock import ScaledMonotonicClock, SimulatedClock
from verbatim.scheduler.tick import TickLoop

CHUNK = ChunkMode(160)
OPTIONS = SessionOptions(chunk_ms=160)


async def _drain(session: SessionHandle) -> list[Hypothesis]:
    results = session.results()
    return [hypothesis async for hypothesis in results]


async def _collect(session: SessionHandle) -> list[Hypothesis]:
    return await asyncio.wait_for(_drain(session), timeout=5.0)


def _engine(*, ring_seconds: float = 3.0) -> Engine:
    return stub_engine(ring_seconds=ring_seconds, clock=ScaledMonotonicClock(100.0))


def _rows(frames: Sequence[PcmFrame], *, eager: bool) -> list[StepResult]:
    return [
        StepResult(
            stream_id=frame.stream_id,
            tick_id=-1,
            partial_text="row",
            final_text=None,
            audio_processed_s=123.0,
            eager=eager,
        )
        for frame in frames
    ]


def test_decode_pcm16_carries_an_odd_byte_between_payloads() -> None:
    samples, carry = decode_pcm16(b"\x01\x02\x03")
    assert samples.shape == (1,)
    assert samples[0] == pytest.approx(0x0201 / 32768)
    assert carry == b"\x03"

    samples, carry = decode_pcm16(b"\x04", carry)
    assert samples.shape == (1,)
    assert samples[0] == pytest.approx(0x0403 / 32768)
    assert carry == b""


class _StampingPipeline(PipelineAdapter):
    def __init__(self) -> None:
        self._open: set[int] = set()

    @property
    def chunk(self) -> ChunkMode:
        return CHUNK

    def supported_buckets(self) -> tuple[int, ...]:
        return (1,)

    def open_stream(self, stream_id: int, options: SessionOptions | None) -> None:
        self._open.add(stream_id)

    def close_stream(self, stream_id: int) -> None:
        self._open.discard(stream_id)

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        return _rows(frames, eager=keep_all_outputs)


class _FailingPipeline(_StampingPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("step failed")
        return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)


class _OpenFailingPipeline(_StampingPipeline):
    def open_stream(self, stream_id: int, options: SessionOptions | None) -> None:
        if stream_id == 2:
            raise RuntimeError("open failed")
        super().open_stream(stream_id, options)


def _induce_dead_tick_thread(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run_tick = engine._tick.run_tick

    def run_tick(*, lock: object = None) -> list[StepResult]:
        if engine.tick_id >= 2:
            raise MemoryError("tick thread died")
        return real_run_tick(lock=lock)

    monkeypatch.setattr(engine._tick, "run_tick", run_tick)


@pytest.mark.asyncio
async def test_feed_is_non_blocking_and_accepts_any_size() -> None:
    """Every offered size is accepted while the ring has room, including odd bytes."""
    engine = _engine()
    session = engine.open_session(OPTIONS)
    async with engine:
        one = b"\x01"
        odd = b"\x02\x03\x04"
        multi_chunk = b"\x00" * (OPTIONS.chunk_bytes * 2 + 3)
        assert session.feed(one) == len(one)
        assert session.feed(odd) == len(odd)
        assert session.feed(multi_chunk) == len(multi_chunk)
        session.end()
        hypotheses = await _collect(session)
    assert hypotheses[-1].audio_processed_s == pytest.approx(5123 / 16000)


@pytest.mark.asyncio
async def test_feed_returns_short_only_when_the_ring_is_full_and_never_drops() -> None:
    engine = _engine(ring_seconds=0.2)
    session = engine.open_session(OPTIONS)
    capacity = int(16000 * 0.2)
    offer = b"\x00" * ((capacity + 100) * 2)
    async with engine:
        accepted = session.feed(offer)
        assert accepted == capacity * 2
        assert len(offer) - accepted == 100 * 2
        while engine.buffered_samples(1) > capacity - 100:
            await engine.wait_for_ticks(1)
        remainder = offer[accepted:]
        assert session.feed(remainder) == len(remainder)
        assert session.feed(b"\x01") == 1
        assert session.feed(b"\x02\x03\x04") == 3


@pytest.mark.asyncio
async def test_one_loop_wake_per_tick() -> None:
    engine = _engine()
    sessions = [engine.open_session(OPTIONS) for _ in range(3)]
    start_tick_id = engine.tick_id
    start_loop_wakes = engine.loop_wakes
    await engine.start()
    for session in sessions:
        assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    await engine.wait_for_ticks(6)
    await engine.stop()
    assert engine.loop_wakes - start_loop_wakes == engine.tick_id - start_tick_id


@pytest.mark.asyncio
async def test_results_are_per_session_fifo() -> None:
    scripts = {
        "model-a": ("alpha", "one"),
        "model-b": ("beta", "two"),
        "model-c": ("gamma", "three"),
    }

    def script_for(options: SessionOptions) -> Sequence[str]:
        return scripts[options.model]

    engine = stub_engine(script_for=script_for, clock=ScaledMonotonicClock(100.0))
    options = [SessionOptions(chunk_ms=160, model=model) for model in scripts]
    sessions = [engine.open_session(session_options) for session_options in options]
    async with engine:
        for session in sessions:
            assert session.feed(b"\x00" * (OPTIONS.chunk_bytes * 2)) == OPTIONS.chunk_bytes * 2
            session.end()
        outputs = await asyncio.gather(*[_collect(session) for session in sessions])
    expected = [
        [
            ("alpha", False, 0.16),
            ("alpha one", False, 0.32),
            ("alpha one", True, 0.32),
        ],
        [("beta", False, 0.16), ("beta two", False, 0.32), ("beta two", True, 0.32)],
        [
            ("gamma", False, 0.16),
            ("gamma three", False, 0.32),
            ("gamma three", True, 0.32),
        ],
    ]
    for hypotheses, expected_rows in zip(outputs, expected, strict=True):
        assert [(h.text, h.is_final, h.audio_processed_s) for h in hypotheses] == expected_rows


@pytest.mark.asyncio
async def test_results_ends_after_the_is_last_final() -> None:
    engine = _engine()
    session = engine.open_session(OPTIONS)
    async with engine:
        assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
        session.end()
        hypotheses = await _collect(session)
    assert hypotheses[-1].is_final is True


@pytest.mark.asyncio
async def test_results_ends_after_an_abort_and_yields_nothing_for_the_abort_frame() -> None:
    engine = _engine()
    session = engine.open_session(OPTIONS)
    async with engine:
        consumer = asyncio.create_task(_collect(session))
        await engine.wait_for_ticks(1)
        session.abort()
        hypotheses = await consumer
    assert all(not hypothesis.is_final for hypothesis in hypotheses)


@pytest.mark.asyncio
async def test_open_session_refused_raises_resource_exhausted_and_reserves_nothing() -> None:
    """The default bucket refuses the ninth session at eight live sessions.

    Part B's two eight-session tests sit exactly on this boundary with no headroom,
    so raising the default bucket is not a free change.
    """
    engine = _engine()
    async with engine:
        sessions = [engine.open_session(OPTIONS) for _ in range(8)]
        del sessions
        live = engine._registry.live
        reserved = engine._tick.slots.reserved
        with pytest.raises(ResourceExhausted) as raised:
            engine.open_session(OPTIONS)
        assert raised.value.code is ErrorCode.RESOURCE_EXHAUSTED
        assert raised.value.retry_after_ms == CHUNK.ms
        assert engine._registry.live == live
        assert engine._tick.slots.reserved == reserved


@pytest.mark.asyncio
async def test_audio_processed_s_comes_from_the_session_not_from_the_adapter() -> None:
    pipeline = _StampingPipeline()
    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1)
    engine = Engine(config, pipeline, clock=ScaledMonotonicClock(100.0))
    session = engine.open_session(OPTIONS)
    async with engine:
        assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
        session.end()
        hypotheses = await _collect(session)
    assert hypotheses
    assert all(hypothesis.audio_processed_s == pytest.approx(0.16) for hypothesis in hypotheses)


@pytest.mark.asyncio
async def test_a_step_exception_reaches_every_live_session_as_internal() -> None:
    pipeline = _FailingPipeline()
    config = EngineConfig(chunk=CHUNK, buckets=(2,), edge_batch=1)
    engine = Engine(config, pipeline, clock=ScaledMonotonicClock(100.0))
    sessions = [engine.open_session(OPTIONS) for _ in range(2)]
    async with engine:
        for session in sessions:
            assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
        consumers = [asyncio.create_task(_collect(session)) for session in sessions]
        await engine.wait_for_ticks(2)
        for consumer in consumers:
            with pytest.raises(VerbatimError) as raised:
                await consumer
            assert raised.value.code is ErrorCode.INTERNAL
            assert str(raised.value) == "step failed"
        assert engine.tick_id >= 2


def test_the_scheduler_suite_clock_is_untouched() -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1)
    pipeline = FakePipelineAdapter(CHUNK, buckets=(1,))
    loop = TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock())
    session = Session(1, CHUNK)
    session.configure()
    assert loop.admit_session(session).admitted
    assert session.ring.write(np.zeros(CHUNK.samples, dtype=np.float32)) == CHUNK.samples
    assert loop.run_tick()[0].audio_processed_s == pytest.approx(CHUNK.period_s)
    assert loop.tick_id == 1


@pytest.mark.asyncio
async def test_pipeline_step_does_not_run_under_the_engine_lock() -> None:
    observed: list[bool] = []
    engine: Engine

    class ProbePipeline(_StampingPipeline):
        def transcribe_step(
            self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
        ) -> list[StepResult]:
            acquired = engine._lock.acquire(blocking=False)
            observed.append(acquired)
            if acquired:
                engine._lock.release()
            return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)

    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1)
    engine = Engine(config, ProbePipeline(), clock=ScaledMonotonicClock(100.0))
    session = engine.open_session(OPTIONS)
    async with engine:
        assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
        await engine.wait_for_ticks(3)
        assert len(observed) >= 3
        assert all(observed)


@pytest.mark.asyncio
async def test_an_open_stream_failure_is_isolated_to_the_failing_session() -> None:
    pipeline = _OpenFailingPipeline()
    config = EngineConfig(chunk=CHUNK, buckets=(2,), edge_batch=1)
    engine = Engine(config, pipeline, clock=ScaledMonotonicClock(100.0))
    sessions = [engine.open_session(OPTIONS) for _ in range(2)]
    assert sessions[0].feed(b"\x00" * (OPTIONS.chunk_bytes * 2)) == OPTIONS.chunk_bytes * 2
    assert sessions[1].feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    sessions[0].end()

    async with engine:
        consumers = [asyncio.create_task(_collect(session)) for session in sessions]
        with pytest.raises(VerbatimError) as raised:
            await consumers[1]
        assert raised.value.code is ErrorCode.INTERNAL
    assert str(raised.value) == "open failed"
    healthy = await consumers[0]
    assert [(hypothesis.text, hypothesis.is_final) for hypothesis in healthy] == [
        ("row", False),
        ("row", False),
    ]
    assert [hypothesis.audio_processed_s for hypothesis in healthy] == pytest.approx([0.16, 0.32])


@pytest.mark.asyncio
async def test_a_dead_tick_thread_fails_every_live_session_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout is only a backstop for a result queue that would otherwise hang."""
    engine = _engine()
    sessions = [engine.open_session(OPTIONS) for _ in range(2)]
    for session in sessions:
        assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    _induce_dead_tick_thread(engine, monkeypatch)

    async with engine:
        consumers = [asyncio.create_task(_collect(session)) for session in sessions]
        for consumer in consumers:
            with pytest.raises(VerbatimError) as raised:
                await asyncio.wait_for(consumer, timeout=5.0)
            assert raised.value.code is ErrorCode.INTERNAL
        assert engine._dead


@pytest.mark.asyncio
async def test_feed_and_open_session_are_refused_once_the_tick_thread_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    session = engine.open_session(OPTIONS)
    assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    _induce_dead_tick_thread(engine, monkeypatch)

    async with engine:
        consumer = asyncio.create_task(_collect(session))
        with pytest.raises(VerbatimError) as raised:
            await consumer
        assert raised.value.code is ErrorCode.INTERNAL

        with pytest.raises(VerbatimError) as feed_error:
            session.feed(b"\x00" * OPTIONS.chunk_bytes)
        assert feed_error.value.code is ErrorCode.INTERNAL

        with pytest.raises(ResourceExhausted) as open_error:
            engine.open_session(OPTIONS)
        assert open_error.value.retry_after_ms == 0


@pytest.mark.asyncio
async def test_finished_sessions_are_dropped_from_the_engine() -> None:
    """A session that has delivered its last row costs the engine nothing. Before this
    the two per-session dictionaries kept every closed session's ring buffer, about
    188 KiB each, for the life of the process."""
    engine = _engine()
    async with engine:
        sessions = [engine.open_session(OPTIONS) for _ in range(5)]
        assert len(engine._sessions) == 5
        assert len(engine._queues) == 5
        for session in sessions:
            assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
            session.end()
        outputs = await asyncio.gather(*[_collect(session) for session in sessions])
        assert all(hypotheses[-1].is_final for hypotheses in outputs)
        assert engine._sessions == {}
        assert engine._queues == {}
        assert engine._registry.live == 0


@pytest.mark.asyncio
async def test_a_failed_session_is_dropped_from_the_engine_too() -> None:
    pipeline = _OpenFailingPipeline()
    config = EngineConfig(chunk=CHUNK, buckets=(2,), edge_batch=1)
    engine = Engine(config, pipeline, clock=ScaledMonotonicClock(100.0))
    sessions = [engine.open_session(OPTIONS) for _ in range(2)]
    assert sessions[0].feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    assert sessions[1].feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    async with engine:
        with pytest.raises(VerbatimError):
            await _collect(sessions[1])
        assert 2 not in engine._queues
        assert 2 not in engine._sessions
        assert 1 in engine._queues
        sessions[0].end()
        assert await _collect(sessions[0])
        assert engine._queues == {}
