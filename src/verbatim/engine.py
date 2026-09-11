# SPDX-License-Identifier: Apache-2.0
"""The asynchronous engine facade over the CPU tick loop.

The engine runs whichever ``PipelineAdapter`` it is built with, the NeMo adapter
in ``verbatim.pipelines.cache_aware_rnnt`` or the CPU fake in
``verbatim.pipelines.fake``, and keeps all protocol-independent result delivery
on the asyncio loop. Both transports run every session through it.

Threading is deliberately coarse and temporary: exactly one lock is constructed
here. It guards ``open_session``, ``feed``, ``end``/``abort``/``close``, and the
collect and stamping phases of ``TickLoop.run_tick``. It is not held across the
pipeline step or the sleep to the next boundary. The later thread-ownership brief
replaces this placeholder with session commands and owner assertions. The emit
deque is the deliberate exception: the tick thread appends and the loop popleft's
without another lock.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Final

from verbatim.audio.pcm import decode_pcm16
from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import (
    ErrorCode,
    InvalidArgument,
    ResourceExhausted,
    Unavailable,
    VerbatimError,
)
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session, SessionState
from verbatim.core.types import StepResult
from verbatim.obs.counters import Counters
from verbatim.obs.latency import LatencySketch
from verbatim.obs.metrics import MetricsSnapshot
from verbatim.pipelines.base import PipelineAdapter
from verbatim.pipelines.fake import FakePipelineAdapter, ScriptSource
from verbatim.protocols.base import EngineHandle, Hypothesis, SessionHandle, SessionOptions
from verbatim.protocols.emit import emissions_for
from verbatim.scheduler.clock import Clock, ScaledMonotonicClock
from verbatim.scheduler.tick import TickLoop

__all__ = ["Engine", "EngineSession", "stub_engine"]

logger = logging.getLogger(__name__)

# Test-harness inputs used only by ``stub_engine``. Neither is a measurement.
STUB_ENGINE_CLOCK_SCALE: Final = 100.0
STUB_ENGINE_BUCKET: Final = 8


@dataclass(frozen=True, slots=True)
class _ErrorEvent:
    stream_id: int
    error: VerbatimError


_EmitItem = StepResult | _ErrorEvent
_QueueItem = StepResult | _ErrorEvent


def _is_pure_partial(item: _QueueItem) -> bool:
    """A partial carries a running prefix the next one supersedes: droppable under
    backlog. A final (``final_text``), an ``is_last`` row and an error are not."""
    return isinstance(item, StepResult) and not item.is_last and item.final_text is None


class _ResultQueue:
    """A single-producer single-consumer queue, both on the event loop, that bounds its
    backlog by dropping the oldest pure partials. Terminals are never dropped, so the
    bound can be temporarily exceeded when the backlog is all finals, which never
    happens: one utterance has one terminal. ``put`` returns how many it dropped."""

    def __init__(self, max_backlog: int) -> None:
        self._items: deque[_QueueItem] = deque()
        self._event = asyncio.Event()
        self._max = max_backlog

    def put(self, item: _QueueItem) -> int:
        self._items.append(item)
        dropped = 0
        while len(self._items) > self._max:
            for index, held in enumerate(self._items):
                if _is_pure_partial(held):
                    del self._items[index]
                    dropped += 1
                    break
            else:
                break  # nothing droppable; keep the terminal-only backlog intact
        self._event.set()
        return dropped

    async def get(self) -> _QueueItem:
        while not self._items:
            self._event.clear()
            if self._items:
                break
            await self._event.wait()
        return self._items.popleft()


class Engine(EngineHandle):
    """One chunk mode's ``TickLoop`` on a dedicated thread.

    The pipeline is whatever adapter the engine is built with; ``stub_engine`` builds
    one over the scripted CPU fake. The single coarse lock is a temporary
    placeholder for the later thread-ownership brief; it is never held across
    ``PipelineAdapter.transcribe_step``.
    """

    def __init__(
        self,
        config: EngineConfig,
        pipeline: PipelineAdapter,
        *,
        clock: Clock | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._config = config
        self._pipeline = pipeline
        self._registry = SessionRegistry()
        self._lock = threading.Lock()
        self._tick = TickLoop(config, pipeline, self._registry, clock=clock)
        self._emit: deque[_EmitItem] = deque()
        self._sessions: dict[int, EngineSession] = {}
        self._queues: dict[int, _ResultQueue] = {}
        self._next_session_id = 1
        self._loop = loop
        self._wake_event: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._dead = False
        self._loop_wakes = 0
        self._processed_wakes = 0
        self._started = False
        self._results_dropped_total = 0
        self._counters = Counters()
        self._tick_costs = LatencySketch()
        self._partial_latency = LatencySketch()
        # (end time, lateness_ms) per completed tick, appended by the tick thread and
        # drained by the loop's wake, which is where the partial becomes visible.
        self._tick_ends: deque[tuple[float, float]] = deque()
        self._last_tick_end: float | None = None
        self._last_refusal_reason: str | None = None

    @property
    def chunk_ms(self) -> int:
        return self._config.chunk.ms

    @property
    def tick_id(self) -> int:
        """Ticks completed; exposed for deterministic tests."""
        return self._tick.tick_id

    @property
    def loop_wakes(self) -> int:
        """Number of per-tick ``call_soon_threadsafe`` calls made."""
        return self._loop_wakes

    def buffered_samples(self, session_id: int) -> int:
        """Return the unread sample count while holding the engine bookkeeping lock."""
        with self._lock:
            return self._registry.get(session_id).ring.available

    async def start(self) -> None:
        """Bind to the running asyncio loop and start the tick thread."""
        running_loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = running_loop
        elif self._loop is not running_loop:
            raise RuntimeError("Engine.start must run on the engine's configured event loop")
        if self._thread is not None and self._thread.is_alive():
            return
        self._wake_event = asyncio.Event()
        self._running = True
        self._started = True
        self._thread = threading.Thread(
            target=self._run_ticks,
            name="verbatim-tick-loop",
            daemon=True,
        )
        self._thread.start()

    async def stop(self) -> None:
        """Stop the tick thread and finish every open results iterator.

        The join happens in an executor: the tick thread may be inside a pipeline
        step, up to a period plus that step, and blocking the event loop for that
        long during shutdown would stall every live socket with it.
        """
        self._running = False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            await asyncio.get_running_loop().run_in_executor(None, thread.join)
        self._thread = None
        if self._loop is None:
            return

        # A completed tick may have queued a wake that has not run yet. Drain it
        # before appending shutdown markers so final rows retain FIFO order. The same
        # wake releases any waiter that parked after the thread's last one.
        self._wake()
        with self._lock:
            live = self._registry.by_state(
                SessionState.ADMITTED,
                SessionState.RUNNING,
                SessionState.STARVED,
                SessionState.DRAINING,
            )
            for session in live:
                handle = self._sessions.get(session.session_id)
                if handle is not None:
                    handle._drop_carry()
                session.begin_draining(aborted=True)
                session.close()
                self._registry.remove(session.session_id)
                self._tick.slots.release(1)
            # A session still open here never received its final: the engine is
            # going away mid-stream. Tell each one so, rather than closing its result
            # stream as if the utterance had completed. A caller cannot distinguish a
            # clean end from an abandoned one, and that decides whether it retries.
            for stream_id, queue in self._queues.items():
                queue.put(_ErrorEvent(stream_id, Unavailable("the server is shutting down")))
            self._queues.clear()
            self._sessions.clear()

    async def __aenter__(self) -> Engine:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def open_session(self, options: SessionOptions) -> SessionHandle:
        """Create, admit and register one session under the engine lock.

        One engine serves one chunk mode. A session asking for another is refused
        with INVALID_ARGUMENT rather than run on this engine's period: the transcript
        would then be attributed to a configuration that never produced it.
        """
        if options.chunk_ms != self._config.chunk.ms:
            raise InvalidArgument(
                f"invalid chunk_ms {options.chunk_ms!r}: this engine serves "
                f"{self._config.chunk.ms} ms"
            )
        session_id = self._next_session_id
        session = Session(
            session_id,
            self._config.chunk,
            ring_seconds=self._config.ring_seconds,
            options=options,
        )
        session.configure()
        with self._lock:
            if self._dead:
                self._counters.observe_admission(False)
                self._last_refusal_reason = "engine is not running"
                raise ResourceExhausted("engine is not running", retry_after_ms=0)
            decision = self._tick.admit_session(session)
            self._counters.observe_admission(decision.admitted)
            if not decision.admitted:
                self._last_refusal_reason = decision.reason or "session admission refused"
                raise ResourceExhausted(
                    decision.reason or "session admission refused",
                    retry_after_ms=decision.retry_after_ms,
                )
            queue = _ResultQueue(self._config.max_result_backlog)
            handle = EngineSession(self, session, queue)
            self._sessions[session_id] = handle
            self._queues[session_id] = queue
            self._next_session_id += 1
            return handle

    async def wait_for_ticks(self, n: int) -> None:
        """Wait for ``n`` further completed ticks using the per-tick wake.

        Raises RuntimeError once the engine is stopping or its tick thread is gone. A
        transport reader parked here in back-pressure is released by the thread's last
        wake or by ``stop()``'s own, and told, rather than left waiting for a tick that
        will never come; ``stop()`` sets ``_running`` false before it joins, so the
        thread's last wake is enough to release a waiter of any count.
        """
        if n < 0:
            raise ValueError(f"tick count must be >= 0, got {n!r}")
        if n == 0:
            return
        if self._thread is None or self._wake_event is None:
            raise RuntimeError("Engine.wait_for_ticks requires a started engine")
        if self._dead or not self._running:
            raise RuntimeError("engine tick loop is not running")
        target = self._processed_wakes + n
        event = self._wake_event
        while self._processed_wakes < target:
            event.clear()
            if self._processed_wakes >= target:
                break
            await event.wait()
            if self._dead or not self._running:
                raise RuntimeError("engine tick loop is not running")

    def _run_ticks(self) -> None:
        """Run ticks; `_publish` queues each tick's results and wakes the loop once,
        before that tick's sleep to the next boundary."""
        while self._running:
            try:
                self._tick.run_tick(lock=self._lock, publish=self._publish)
            except Exception as exc:
                logger.exception("tick loop stopped after an unexpected scheduler failure")
                self._running = False
                self._dead = True
                with self._lock:
                    self._tick.fail_live(exc)
                    for stream_id, error in self._tick.drain_errors():
                        self._emit.append(_ErrorEvent(stream_id, error))
                loop = self._loop
                if loop is not None:
                    with suppress(RuntimeError):
                        loop.call_soon_threadsafe(self._wake)
                return

    def _publish(self, results: list[StepResult]) -> None:
        """The tail of a tick, on the tick thread, between its stamping phase and its
        sleep: queue the results and the errors, fold the tick into the counters, and
        wake the loop once. A partial computed at boundary t therefore reaches the
        transport at t plus the step, not at t+1. Called by `run_tick` outside the
        engine lock; a lost loop ends the thread after this tick's sleep."""
        for result in results:
            self._emit.append(result)
        for stream_id, error in self._tick.drain_errors():
            self._emit.append(_ErrorEvent(stream_id, error))
        self._observe_tick()
        loop = self._loop
        if loop is None:
            self._running = False
            self._dead = True
            return
        try:
            loop.call_soon_threadsafe(self._wake)
        except RuntimeError:
            self._running = False
            self._dead = True
            return
        self._loop_wakes += 1

    def _observe_tick(self) -> None:
        """Fold the tick just completed into the counters and the cost window, on the
        tick thread, and hand its end time to the loop for the partial latency."""
        stats = self._tick.last_stats
        if stats is None:
            return
        end = self._tick.clock.now()
        with self._lock:
            self._counters.observe_tick(
                stats, budget_ms=self._config.budget_ms, period_ms=float(self._config.chunk.ms)
            )
            self._tick_costs.observe(stats.step_ms + stats.edge_ms)
        self._tick_ends.append((end, stats.lateness_ms))
        self._last_tick_end = end

    def snapshot(self) -> MetricsSnapshot:
        """What the engine knows about itself right now, under its lock. Data only:
        readiness and liveness are judged by the health reporter from this."""
        with self._lock:
            admission = self._tick.admission
            slots = self._tick.slots
            now = self._tick.clock.now()
            age = None if self._last_tick_end is None else max(0.0, now - self._last_tick_end)
            buckets = self._config.buckets or (0,)
            return MetricsSnapshot(
                chunk_ms=self._config.chunk.ms,
                period_ms=float(self._config.chunk.ms),
                budget_ms=self._config.budget_ms,
                bucket=max(buckets),
                started=self._started,
                running=self._running,
                dead=self._dead,
                tick_id=self._tick.tick_id,
                last_tick_age_s=age,
                live_sessions=self._registry.live,
                slots_capacity=slots.capacity,
                slots_reserved=slots.reserved,
                slots_free=slots.free(),
                ceiling=admission.ceiling,
                calibrated_ceiling=admission.calibrated_ceiling,
                degradation_level=admission.degradation_level,
                consecutive_overruns=admission.consecutive_overruns,
                p95_tick_ms=admission.p95_ms,
                eager_step_fraction=admission.eager_step_fraction,
                counters=replace(self._counters),
                latency_count=self._partial_latency.count,
                partial_latency_ms=(
                    self._partial_latency.p50,
                    self._partial_latency.p95,
                    self._partial_latency.p99,
                ),
                tick_cost_ms=(self._tick_costs.p50, self._tick_costs.p95, self._tick_costs.p99),
                last_refusal_reason=self._last_refusal_reason,
                results_partials_dropped_total=self._results_dropped_total,
            )

    def _wake(self) -> None:
        """Move the lock-free emit deque into per-session asyncio queues.

        A terminal item, the ``is_last`` row or an error, also drops the engine's
        references to the session: the handle keeps its own queue and ring, so
        ``results()`` still drains what was queued, and once the transport lets the
        handle go the session costs nothing. Before this the two dictionaries kept
        every closed session's ring buffer for the life of the process.
        """
        now = self._tick.clock.now()
        while self._tick_ends:
            end, lateness_ms = self._tick_ends.popleft()
            self._partial_latency.observe(lateness_ms + max(0.0, now - end) * 1000.0)
        while True:
            try:
                item = self._emit.popleft()
            except IndexError:
                break
            queue = self._queues.get(item.stream_id)
            if queue is not None:
                self._results_dropped_total += queue.put(item)
                if isinstance(item, _ErrorEvent) or item.is_last:
                    del self._queues[item.stream_id]
                    self._sessions.pop(item.stream_id, None)
        if self._wake_event is not None:
            self._processed_wakes += 1
            self._wake_event.set()

    def _feed(self, handle: EngineSession, payload: bytes) -> int:
        """Decode and accept an owned prefix while holding the sole engine lock."""
        with self._lock:
            if self._dead:
                raise VerbatimError("engine tick loop is not running", ErrorCode.INTERNAL)
            session = handle._session
            if session.state not in (
                SessionState.ADMITTED,
                SessionState.RUNNING,
                SessionState.STARVED,
            ):
                raise InvalidArgument(
                    f"cannot feed session {session.session_id} in state {session.state.value}"
                )
            old_carry = handle._carry
            samples, new_carry = decode_pcm16(payload, old_carry)
            accepted_samples = session.ring.write(samples)
            if accepted_samples == len(samples):
                handle._carry = new_carry
                return len(payload)

            # The accepted sample prefix consumes two bytes per sample from the
            # combined old-carry-plus-payload stream. A previous carry is already
            # owned, so only the current payload contributes to this return value.
            consumed_payload = max(0, accepted_samples * 2 - len(old_carry))
            handle._carry = old_carry if accepted_samples == 0 else b""
            return min(len(payload), consumed_payload)


class EngineSession(SessionHandle):
    """One session's transport-facing ring and asynchronous result queue."""

    def __init__(
        self,
        engine: Engine,
        session: Session,
        queue: _ResultQueue,
    ) -> None:
        self._engine = engine
        self._session = session
        self._queue = queue
        self._carry = b""

    @property
    def options(self) -> SessionOptions:
        options = self._session.options
        assert options is not None
        return options

    def feed(self, pcm: bytes) -> int:
        """Accept a non-blocking owned prefix of PCM16LE bytes."""
        payload = bytes(pcm)
        return self._engine._feed(self, payload)

    def end(self) -> None:
        """Half-close the session; the tick loop drains its remaining ring audio."""
        with self._engine._lock:
            self._drop_carry()
            if self._session.state in (
                SessionState.ADMITTED,
                SessionState.RUNNING,
                SessionState.STARVED,
                SessionState.DRAINING,
            ):
                self._session.begin_draining()

    def abort(self) -> None:
        """Discard buffered audio through the normal zero-valid drain frame."""
        with self._engine._lock:
            self._drop_carry()
            if self._session.state in (
                SessionState.ADMITTED,
                SessionState.RUNNING,
                SessionState.STARVED,
                SessionState.DRAINING,
            ):
                self._session.begin_draining(aborted=True)

    async def results(self) -> AsyncIterator[Hypothesis]:
        """Yield this session's FIFO hypotheses until its last row or failure."""
        while True:
            item = await self._queue.get()
            if isinstance(item, _ErrorEvent):
                raise item.error
            emission = emissions_for(item, self.options)
            for hypothesis in emission.hypotheses:
                yield hypothesis
            if emission.end_of_session:
                return

    def _drop_carry(self) -> None:
        if self._carry:
            logger.warning(
                "dropping incomplete PCM16 sample at session %s end",
                self._session.session_id,
            )
            self._carry = b""


def stub_engine(
    *,
    script: Sequence[str] | None = None,
    partial_every: int = 1,
    script_for: ScriptSource | None = None,
    chunk_ms: int = 160,
    bucket: int | None = None,
    ring_seconds: float = 3.0,
    clock: Clock | None = None,
    idle_timeout_s: float | None = None,
) -> Engine:
    """Build a ready-to-run engine over the scripted CPU fake.

    ``bucket``, ``ring_seconds``, ``idle_timeout_s`` and the clock scale are
    test-harness inputs, not measurements of hardware. The default clock keeps
    protocol tests on the same boundary arithmetic without waiting for real audio
    time, which is also why the idle deadline is off unless a test asks for it: on
    the scaled clock thirty engine seconds pass in a third of a real one.
    """
    chunk = ChunkMode(chunk_ms)
    config = EngineConfig(
        chunk=chunk,
        buckets=(bucket if bucket is not None else STUB_ENGINE_BUCKET,),
        ring_seconds=ring_seconds,
        idle_timeout_s=idle_timeout_s,
    )
    source = script_for
    if source is None:

        def source(_options: SessionOptions) -> Sequence[str] | None:
            return script

    pipeline = FakePipelineAdapter(
        chunk,
        buckets=config.buckets or (),
        scripted=True,
        script_for=source,
        partial_every=partial_every,
    )
    selected_clock = clock if clock is not None else ScaledMonotonicClock(STUB_ENGINE_CLOCK_SCALE)
    return Engine(config, pipeline, clock=selected_clock)
