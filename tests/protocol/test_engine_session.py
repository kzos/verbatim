# SPDX-License-Identifier: Apache-2.0
"""Engine facade tests: ingress ownership, FIFO delivery and tick-thread seams."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import numpy as np
import pytest

from verbatim.audio.pcm import decode_pcm16
from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import (
    DeadlineExceeded,
    ErrorCode,
    InvalidArgument,
    ResourceExhausted,
    Unavailable,
    VerbatimError,
)
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.core.types import PcmFrame, StepResult
from verbatim.engine import Engine, stub_engine
from verbatim.pipelines.base import PipelineAdapter
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.pipelines.nemo_fake import FakeCacheAwarePipeline, boundary_for
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
    """The second step raises. Both sessions are fed while the thread is parked before
    its first tick, so the step that fails has two live sessions to reach; fed after
    the thread was running, the failing tick could land between the two feeds, and did
    once in a soak under load."""
    pipeline = _FailingPipeline()
    clock = _HeldClock()
    config = EngineConfig(chunk=CHUNK, buckets=(2,), edge_batch=1)
    engine = Engine(config, pipeline, clock=clock)
    sessions = [engine.open_session(OPTIONS) for _ in range(2)]
    async with engine:
        try:
            await _until(lambda: clock.sleeps >= 1)
            for session in sessions:
                assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
            consumers = [asyncio.create_task(_collect(session)) for session in sessions]
        finally:
            clock.open()  # released whatever happened above, so stop() can join the thread
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
        # The healthy session ended itself, so its results run out on their own; they
        # are awaited while the engine is still running, because stopping the engine
        # the moment the failure arrives would cut them short at whatever tick that was.
        healthy = await consumers[0]
    assert str(raised.value) == "open failed"
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


@pytest.mark.asyncio
async def test_endpointing_delivers_a_final_before_the_client_half_closes() -> None:
    """LiveKit and Pipecat never half-close during a live call: the final for an
    utterance must arrive from NeMo's endpointer, on silence, with the session still
    open. Wired end to end: SessionOptions.stop_history_eou_ms into the adapter, into
    NeMo's request options, and the step where the fake detects the end of utterance
    back out as an is_final hypothesis with word timings."""
    config = EngineConfig(chunk=CHUNK, buckets=(2,), edge_batch=1, pipeline="cache_aware_rnnt")
    nemo = FakeCacheAwarePipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareRNNTAdapter(
        CHUNK, boundary_for(nemo), buckets=(2,), required_slots=config.num_slots
    )
    engine = Engine(config, adapter, clock=ScaledMonotonicClock(100.0))
    session = engine.open_session(
        SessionOptions(chunk_ms=160, word_timestamps=True, stop_history_eou_ms=2 * CHUNK.ms)
    )
    speech = (b"\x10\x00" * OPTIONS.chunk_samples) * 2
    silence = b"\x00" * (OPTIONS.chunk_bytes * 2)
    received: list[Hypothesis] = []

    async def consume() -> None:
        async for hypothesis in session.results():
            received.append(hypothesis)
            if hypothesis.is_final:
                return

    async with engine:
        assert session.feed(speech) == len(speech)
        assert session.feed(silence) == len(silence)
        await asyncio.wait_for(consume(), timeout=5.0)
        # The final arrived while the session was still open: nobody called end().
        assert engine._registry.live == 1
        final = received[-1]
        assert final.is_final and final.text.count(" ") == 1
        assert [(w.start_ms, w.end_ms) for w in final.words] == [(0, 160), (160, 320)]
        assert final.audio_processed_s == pytest.approx(4 * 0.16)
        session.end()
        rest = await _collect(session)
    assert rest[-1].is_final is True


@pytest.mark.asyncio
async def test_open_session_refuses_a_chunk_mode_the_engine_does_not_serve() -> None:
    """One engine serves one chunk mode. A session asking for another is refused by
    name and reserves nothing, rather than being run on this engine's period."""
    engine = _engine()
    async with engine:
        live = engine._registry.live
        reserved = engine._tick.slots.reserved
        with pytest.raises(InvalidArgument, match="560"):
            engine.open_session(SessionOptions(chunk_ms=560))
        assert engine._registry.live == live
        assert engine._tick.slots.reserved == reserved


@pytest.mark.asyncio
async def test_an_idle_session_is_closed_with_deadline_exceeded_and_dropped() -> None:
    """A session that never sends audio is closed by the engine at the idle deadline:
    its consumer gets DEADLINE_EXCEEDED, its slot is released and the engine keeps
    nothing of it. Both transports inherit this without a timer of their own."""
    engine = stub_engine(idle_timeout_s=0.5, clock=ScaledMonotonicClock(100.0))
    async with engine:
        reserved = engine._tick.slots.reserved
        session = engine.open_session(OPTIONS)
        assert engine._tick.slots.reserved == reserved + 1
        with pytest.raises(DeadlineExceeded) as raised:
            await _collect(session)
        assert raised.value.code is ErrorCode.DEADLINE_EXCEEDED
        assert "no audio for" in str(raised.value)
        assert engine._tick.slots.reserved == reserved
        assert engine._sessions == {}
        assert engine._queues == {}


@pytest.mark.asyncio
async def test_a_session_that_keeps_sending_is_not_closed_by_the_deadline() -> None:
    """One chunk every two ticks against a deadline of three starved ticks. The thread
    is stepped one tick at a time on the held clock, so exactly one starved tick
    separates the feeds whatever the machine is doing; on a real clock at a hundred
    times, a stall of a few milliseconds between wake and feed reached the deadline."""
    clock = _HeldClock()
    engine = stub_engine(idle_timeout_s=0.5, clock=clock)  # 0.5 s is three ticks of 160 ms
    async with engine:
        try:
            await _until(lambda: clock.sleeps >= 1)  # parked before tick 1
            session = engine.open_session(OPTIONS)
            consumer = asyncio.create_task(_collect(session))
            for _ in range(6):
                assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
                for _ in range(2):
                    await _one_tick(engine, clock)
            session.end()
        finally:
            clock.open()  # the drain runs freely, and stop() can join whatever happened
        hypotheses = await consumer
    assert hypotheses[-1].is_final
    assert hypotheses[-1].audio_processed_s == pytest.approx(6 * 0.16)


class _SlowStep(FakePipelineAdapter):
    def __init__(self, seconds: float) -> None:
        super().__init__(CHUNK, buckets=(1,))
        self.seconds = seconds

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        time.sleep(self.seconds)
        return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)


@pytest.mark.asyncio
async def test_stop_does_not_block_the_event_loop_while_joining_the_tick_thread() -> None:
    """The join waits for the tick thread's current step, up to a period plus a step.
    A heartbeat task on the loop must keep running through it; with the join on the
    loop it would stall for the whole step, and so would every live socket."""
    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1)
    engine = Engine(config, _SlowStep(0.3), clock=ScaledMonotonicClock(100.0))
    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            await asyncio.sleep(0.01)
            beats += 1

    await engine.start()
    await asyncio.sleep(0.05)  # the tick thread is inside a slow step by now
    task = asyncio.create_task(heartbeat())
    started = time.monotonic()
    await engine.stop()
    elapsed = time.monotonic() - started
    task.cancel()
    assert elapsed >= 0.1, "the join returned before the step ended; nothing was measured"
    assert beats >= 5, f"the loop ran {beats} heartbeat(s) during a {elapsed:.2f} s join"


class _HeldClock:
    """Real time for `now`; every sleep to the next boundary blocks until the test
    releases it, so the tick thread parks exactly where the test says. `open()` ends
    the holding: every sleep from then on returns at once, so a thread parked when a
    test fails is not left there for `stop()` to join forever."""

    def __init__(self) -> None:
        self._releases = threading.Semaphore(0)
        self._open = False
        self.sleeps = 0

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        self.sleeps += 1
        if not self._open:
            self._releases.acquire()

    def release(self) -> None:
        self._releases.release()

    def open(self) -> None:
        self._open = True
        self._releases.release()


async def _until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


async def _one_tick(engine: Engine, clock: _HeldClock) -> None:
    """Release the parked thread for exactly one tick and wait for that tick's wake."""
    wakes = engine._processed_wakes
    clock.release()
    await _until(lambda: engine._processed_wakes > wakes)


@asynccontextmanager
async def _held_engine() -> AsyncIterator[tuple[Engine, _HeldClock]]:
    """A started engine whose thread has finished tick 1 and is parked in its sleep,
    with tick 0's wake already processed on the loop. On exit the clock is opened and
    the engine stopped whatever the test did, so a failed assertion is a failure and
    not a process hung on the join of a thread parked in a sleep nobody releases."""
    clock = _HeldClock()
    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1)
    engine = Engine(config, _StampingPipeline(), clock=clock)
    await engine.start()
    try:
        await _until(lambda: clock.sleeps >= 1)
        clock.release()
        await _until(lambda: engine._processed_wakes >= 1)
        await _until(lambda: clock.sleeps >= 2)
        yield engine, clock
    finally:
        clock.open()
        await asyncio.wait_for(engine.stop(), timeout=5.0)


@pytest.mark.asyncio
async def test_a_waiter_parked_for_ticks_is_released_when_the_engine_stops() -> None:
    """A transport reader parked in back-pressure waits for ticks; once `stop()` has
    begun, only one more wake will ever come from the thread. A waiter for more than
    that must still be released, and told, or the server would never shut down."""
    async with _held_engine() as (engine, clock):
        waiter = asyncio.create_task(engine.wait_for_ticks(3))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        stopping = asyncio.create_task(engine.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done(), "stop() joins the thread, which is parked in its sleep"
        assert not waiter.done()
        clock.release()  # the thread finishes tick 1, sends its last wake, and exits
        await asyncio.wait_for(stopping, timeout=5.0)
        with pytest.raises(RuntimeError, match="not running"):
            await asyncio.wait_for(waiter, timeout=5.0)


@pytest.mark.asyncio
async def test_wait_for_ticks_during_stop_raises_at_once() -> None:
    async with _held_engine() as (engine, clock):
        stopping = asyncio.create_task(engine.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done()
        with pytest.raises(RuntimeError, match="not running"):
            await asyncio.wait_for(engine.wait_for_ticks(1), timeout=1.0)
        clock.release()
        await asyncio.wait_for(stopping, timeout=5.0)


@pytest.mark.asyncio
async def test_a_reader_parked_in_back_pressure_is_released_by_stop() -> None:
    """Bench's probe, pinned: audio pending against a small ring, the reader parked in
    `wait_for_ticks`, then `stop()`. The reader is released with the error the
    transports already handle, and its next `feed` refuses the closed session."""
    async with _held_engine() as (engine, clock):
        session = engine.open_session(OPTIONS)
        pending = b"\x00" * (OPTIONS.chunk_bytes * 40)
        accepted = session.feed(pending)
        assert 0 < accepted < len(pending), "the ring must be full for the reader to park"
        outcomes: list[str] = []

        async def reader() -> None:
            try:
                await engine.wait_for_ticks(1)
            except RuntimeError as exc:
                outcomes.append(f"released: {exc}")
                return
            outcomes.append("returned")

        parked = asyncio.create_task(reader())
        await asyncio.sleep(0.05)
        assert not parked.done()
        stopping = asyncio.create_task(engine.stop())
        await asyncio.sleep(0.05)
        clock.release()
        await asyncio.wait_for(stopping, timeout=5.0)
        await asyncio.wait_for(parked, timeout=5.0)
        assert outcomes == ["released: engine tick loop is not running"]
        with pytest.raises(VerbatimError, match="CLOSED"):
            session.feed(pending[accepted:])


@pytest.mark.asyncio
async def test_a_session_open_at_stop_is_told_unavailable_not_given_a_clean_end() -> None:
    """A session still open when the engine stops never received its final: the engine
    is going away mid-stream. Its result stream ends with UNAVAILABLE rather than a
    clean return, because a caller cannot otherwise tell a finished utterance from an
    abandoned one, and that is what decides whether it retries. Partials the engine
    had already produced still arrive first."""
    engine = _engine()
    async with engine:
        session = engine.open_session(OPTIONS)
        assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
        await engine.wait_for_ticks(2)  # the chunk has been stepped: a partial is queued
    # the context exit stopped the engine with the session still open
    seen: list[Hypothesis] = []

    async def drain() -> None:
        async for hypothesis in session.results():
            seen.append(hypothesis)

    with pytest.raises(Unavailable) as raised:
        await asyncio.wait_for(drain(), timeout=5.0)
    assert raised.value.code is ErrorCode.UNAVAILABLE
    assert "shutting down" in str(raised.value)
    assert seen, "the partial produced before stop() must still be delivered"
    assert all(not hypothesis.is_final for hypothesis in seen)


@pytest.mark.asyncio
async def test_a_final_produced_by_the_last_tick_arrives_as_a_final_not_as_unavailable() -> None:
    """`stop()` tells a session UNAVAILABLE only after the join, so the thread's last
    tick has run and its wake has been processed (the loop runs that callback before
    the join's completion, in FIFO order; `stop()`'s own drain covers a wake still
    queued). A session whose final that last tick produced finished its utterance: the
    caller gets the final and a clean end, and only a stream still open after the
    join is told the server went away. Told before the join, a finished utterance
    would be reported abandoned. The scripted fake delivers the final two ticks after
    `end()`, so the thread is held with the final exactly one tick away when `stop()`
    begins."""
    clock = _HeldClock()
    engine = stub_engine(clock=clock)
    async with engine:
        try:
            await _until(lambda: clock.sleeps >= 1)  # parked before tick 1
            session = engine.open_session(OPTIONS)
            assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
            await _one_tick(engine, clock)  # the chunk is stepped
            session.end()
            await _one_tick(engine, clock)  # the drain has begun; the final is one tick away
            assert 1 in engine._queues, "the final must still be in flight when stop() begins"
            stopping = asyncio.create_task(engine.stop())
            await asyncio.sleep(0.05)
            assert not stopping.done(), "stop() joins the thread, which is parked in its sleep"
            clock.release()  # the thread's last tick produces the final, then it exits
            await asyncio.wait_for(stopping, timeout=5.0)
        finally:
            clock.open()
    hypotheses = await _collect(session)
    assert hypotheses[-1].is_final
    assert hypotheses[-1].audio_processed_s == pytest.approx(0.16)
