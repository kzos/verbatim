# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The tick loop. Runs on a dedicated non-asyncio thread that owns every NeMo object.

Per tick, per mode: collect exactly one chunk per ready session; plan the steady
batch (padded to exactly ``B``) and the edge batches; one ``transcribe_step`` for
the steady batch, one per edge batch; push ``StepResult`` records to the emit
queue; record ``TickStats``; sleep to the next boundary.

Final (``is_last``) frames are never mixed into the steady call: NeMo's pipeline
splits them into a ``keep_all_outputs=True`` sub-batch, which would shrink the
steady sub-batch and change its graph key.

The loop runs any ``PipelineAdapter``: the NeMo adapter in
``verbatim.pipelines.cache_aware_rnnt`` in a server, and in the scheduler suite the
deterministic CPU fake with an injectable clock, synchronously on the caller's
thread, so the whole suite runs in milliseconds with no asyncio and no threads.
This module must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import replace
from typing import Final

from verbatim.config import EngineConfig
from verbatim.core.errors import DeadlineExceeded, ErrorCode, VerbatimError
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session, SessionState
from verbatim.core.types import PcmFrame, StepResult, TickStats
from verbatim.pipelines.base import PipelineAdapter
from verbatim.scheduler.admission import AdmissionController, AdmissionDecision
from verbatim.scheduler.buckets import BucketPlan, BucketScheduler
from verbatim.scheduler.clock import Clock, MonotonicClock
from verbatim.scheduler.slots import SlotTable

__all__ = ["TickLoop"]

logger = logging.getLogger(__name__)

#: How many ticks of ``TickStats`` and boundaries the loop keeps. Unbounded lists
#: grew by one entry per tick for the life of the process, about 540,000 a day at
#: the 160 ms grid; the admission window needs 200 and diagnostics need a little more.
STATS_RETAINED: Final = 1024


def _cost(pipeline: PipelineAdapter, name: str) -> float:
    """A pipeline's configured per-step cost as a plain number (an input, never a
    measurement). Adapters without costs contribute zero."""
    value = getattr(pipeline, name, 0.0)
    return float(value)


class TickLoop:
    """One chunk mode's collect/plan/step/emit/stats/wait cycle, driven synchronously
    against an injectable clock. Boundaries are fixed multiples of the period since
    start, so a late tick does not shift the ones after it."""

    def __init__(
        self,
        config: EngineConfig,
        pipeline: PipelineAdapter,
        registry: SessionRegistry,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._config = config
        self._pipeline = pipeline
        self._registry = registry
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._slots = SlotTable(config.num_slots)
        self._scheduler = BucketScheduler(config)
        self._admission = AdmissionController(config, self._slots)
        # Warm-up: pad rows hold slots for the life of the process and are never shed.
        self._pad_count = config.effective_pad
        self._slots.reserve(self._pad_count)
        # Edge pad rows are one-shot, but the slots they take during an edge step are
        # real. Reserving the most that can be in flight at once keeps `free()` the
        # truth, so admission never lends those slots to a session.
        if config.edge_pad_rows > 0:
            self._slots.reserve(config.edge_pad_rows)
        # The idle deadline in ticks. A live session starved for this many consecutive
        # ticks is closed with DEADLINE_EXCEEDED in the collect phase, where its slot
        # is held. Counted on the engine's clock, so both transports inherit it
        # without a timer of their own.
        self._idle_limit_ticks: int | None = (
            None
            if config.idle_timeout_s is None
            else max(1, math.ceil(config.idle_timeout_s / config.chunk.period_s))
        )
        self._start = self._clock.now()
        self._tick_id = 0
        self._stats: deque[TickStats] = deque(maxlen=STATS_RETAINED)
        self._boundaries: deque[float] = deque(maxlen=STATS_RETAINED)
        self._errors: dict[int, VerbatimError] = {}

    @property
    def tick_id(self) -> int:
        """Ticks completed so far."""
        return self._tick_id

    @property
    def stats(self) -> list[TickStats]:
        """One ``TickStats`` per completed tick, in order, for the most recent
        ``STATS_RETAINED`` ticks. ``tick_id`` keeps counting past that."""
        return list(self._stats)

    @property
    def admission(self) -> AdmissionController:
        return self._admission

    @property
    def slots(self) -> SlotTable:
        return self._slots

    @property
    def scheduler(self) -> BucketScheduler:
        return self._scheduler

    @property
    def registry(self) -> SessionRegistry:
        return self._registry

    @property
    def boundaries(self) -> list[float]:
        """Each tick's scheduled boundary, fixed multiples of the period since start,
        for the most recent ``STATS_RETAINED`` ticks."""
        return list(self._boundaries)

    @property
    def period_s(self) -> float:
        return self._config.chunk.period_s

    def admit_session(self, session: Session) -> AdmissionDecision:
        """Admit through the admission controller: on success the session is admitted,
        registered and holding a slot; on refusal it is rejected and holds nothing."""
        decision = self._admission.decide(self._registry.live)
        if not decision.admitted:
            session.reject()
            return decision
        session.admit()
        self._registry.add(session)
        self._slots.reserve(1)
        return decision

    def _close_session(self, session: Session) -> None:
        session.close()
        self._registry.remove(session.session_id)
        self._slots.release(1)

    def drain_errors(self) -> list[tuple[int, VerbatimError]]:
        """The per-stream step failures since the last drain, then clear.

        The tick thread appends these to the engine-owned emit deque alongside the
        rows, so every live session's transport is told. Called on the tick thread
        only, immediately after `run_tick` returns on that same thread.
        """
        if not self._errors:
            return []
        items = list(self._errors.items())
        self._errors.clear()
        return items

    def fail_live(self, exc: Exception) -> None:
        """Fail every currently live session with INTERNAL, and close it.

        Each transport's `results()` raises the error, and the tick thread does not
        die silently. Closing is part of failing: a session left registered after
        its error would be fed again on the next tick, hold its slot until its
        transport noticed, and have a second error queued to a consumer that had
        already gone. Its slot is released, its registry entry removed and the
        adapter's `close_stream` called. Marking the engine unhealthy, refusing new
        admissions and exiting non-zero are a later brief.
        """
        err = VerbatimError(str(exc), ErrorCode.INTERNAL)
        for session in self._registry.by_state(
            SessionState.ADMITTED,
            SessionState.RUNNING,
            SessionState.STARVED,
            SessionState.DRAINING,
        ):
            self._errors.setdefault(session.session_id, err)
            session.begin_draining(aborted=True)
            self._close_session(session)
            try:
                self._pipeline.close_stream(session.session_id)
            except Exception:
                logger.exception(
                    "close_stream failed for stream %s while failing live sessions",
                    session.session_id,
                )

    def _record(
        self, tick_id: int, plan: BucketPlan, starved: int, step_ms: float, edge_ms: float
    ) -> None:
        """The stamping-and-bookkeeping tail shared by the success and failure paths."""
        stats = TickStats(
            tick_id=tick_id,
            steady_rows=len(plan.steady),
            live_rows=plan.live_rows,
            pad_rows=plan.pad_rows,
            edge_batches=len(plan.edge_batches),
            eager_rows=plan.eager_rows,
            starved=starved,
            step_ms=step_ms,
            edge_ms=edge_ms,
        )
        self._stats.append(stats)
        self._admission.observe(stats, tick_ms=step_ms + edge_ms)
        self._tick_id += 1

    def run_tick(self, *, lock: threading.Lock | None = None) -> list[StepResult]:
        """One tick, synchronously, on the caller's thread. Boundaries are fixed multiples
        of the period since start, so a late tick does not shift the ones after it.

        When `lock` is supplied (the engine's one lock, and only that one), the
        collect phase and the stamping and bookkeeping phase run under it; the
        `PipelineAdapter.transcribe_step` calls and the sleep to the next boundary
        never do. With `lock=None` the tick path takes no lock at all.
        """
        tick_id = self._tick_id
        boundary = self._start + tick_id * self._config.chunk.period_s
        self._boundaries.append(boundary)

        # Collect phase, under the engine lock when one is supplied: the registry
        # snapshot, `next_frame()`, the batch assembly and the audio-clock read.
        # `open_stream` runs here, on the tick thread, for each first frame, so no
        # adapter state is ever created on the asyncio loop thread.
        non_final: list[PcmFrame] = []
        final: list[PcmFrame] = []
        starved = 0
        closing: list[Session] = []
        audio_by_stream: dict[int, float] = {}
        frame_by_stream: dict[int, PcmFrame] = {}
        aborted_streams: set[int] = set()
        plan: BucketPlan | None = None
        step_ms = 0.0
        edge_ms = 0.0
        try:
            if lock is not None:
                lock.acquire()
            try:
                for session in self._registry.by_state(
                    SessionState.ADMITTED,
                    SessionState.RUNNING,
                    SessionState.STARVED,
                    SessionState.DRAINING,
                ):
                    frame = session.next_frame()
                    if frame is None:
                        if session.state is SessionState.DRAINING:
                            # A DRAINING session whose final frame was already consumed
                            # without a close. No tick-loop path produces this today:
                            # a failed open closes in its own tick and a failed step
                            # closes every live session. Kept as a safety net so the
                            # state can never leak a slot; the test that reaches it
                            # consumes the final frame through the session directly.
                            closing.append(session)
                        else:
                            # STARVED: not scheduled, slot kept, no synthetic audio inserted.
                            starved += 1
                            if (
                                self._idle_limit_ticks is not None
                                and session.starved_ticks >= self._idle_limit_ticks
                            ):
                                # Past the idle deadline. A client that opened a session
                                # and stopped sending would otherwise hold its slot for
                                # ever, and a bucket of them denies service to everyone.
                                idle_s = session.starved_ticks * self._config.chunk.period_s
                                self._errors.setdefault(
                                    session.session_id,
                                    DeadlineExceeded(
                                        f"no audio for {idle_s:.2f} s; the idle deadline "
                                        f"is {self._config.idle_timeout_s} s"
                                    ),
                                )
                                session.begin_draining(aborted=True)
                                closing.append(session)
                    else:
                        # Read the audio clock immediately after `next_frame` returned
                        # this tick's frame, before the step; only `next_frame` mutates
                        # it, so the stamp below carries this tick's value.
                        audio_by_stream[frame.stream_id] = session.audio_processed_s
                        frame_by_stream[frame.stream_id] = frame
                        if session._aborted:
                            aborted_streams.add(frame.stream_id)
                        if frame.is_first:
                            try:
                                self._pipeline.open_stream(frame.stream_id, session.options)
                            except Exception as exc:
                                # The never-opened-stream decision: a session whose open
                                # failed never steps. NeMo has no state for it, so any
                                # frame of it, including the synthetic abort frame, would
                                # dereference None inside the batch and fail everyone.
                                # It closes in this tick; its transport gets the error.
                                self._errors.setdefault(
                                    frame.stream_id,
                                    VerbatimError(str(exc), ErrorCode.INTERNAL),
                                )
                                session.begin_draining(aborted=True)
                                closing.append(session)
                                continue
                        if frame.is_last:
                            final.append(frame)
                            closing.append(session)
                        else:
                            non_final.append(frame)

                plan = self._scheduler.plan(non_final, final)
            finally:
                if lock is not None:
                    lock.release()

            # The step runs without the lock: holding a non-reentrant engine lock
            # across a whole pipeline batch would deadlock the first tick, and even if
            # it did not it would make every `feed()` on the asyncio loop wait for a
            # whole step.
            steady_rows = self._pipeline.transcribe_step(plan.steady, keep_all_outputs=False)
            edge_rows: list[list[StepResult]] = []
            for batch in plan.edge_batches:
                edge_rows.append(self._pipeline.transcribe_step(batch, keep_all_outputs=True))

            # Costs are read after the steps, so an adapter that measures its own
            # wall time reports this tick's, not the previous tick's.
            step_ms = _cost(self._pipeline, "step_ms")
            edge_step_ms = _cost(self._pipeline, "edge_step_ms")
            edge_ms = edge_step_ms * len(plan.edge_batches)

            def _stamp(rows: list[StepResult], *, eager: bool) -> list[StepResult]:
                """Stamp each real row from the frame that produced it: the session's own
                audio clock carried from collect, plus that frame's valid count and
                lastness. The engine and the emitter never compute a clock and never
                re-stamp one; an adapter that stamped its own could not tell real bytes
                from the padding the server invented."""
                stamped: list[StepResult] = []
                for result in rows:
                    if result.stream_id < 0:
                        continue  # Pad-row results are dropped here.
                    frame = frame_by_stream.get(result.stream_id)
                    if frame is None:
                        continue
                    if result.stream_id in aborted_streams:
                        if any(row.stream_id == result.stream_id for row in stamped):
                            continue
                        stamped.append(
                            replace(
                                result,
                                tick_id=tick_id,
                                partial_text="",
                                final_text=None,
                                eager=eager,
                                audio_processed_s=audio_by_stream[result.stream_id],
                                words=(),
                                valid_samples=0,
                                is_last=True,
                            )
                        )
                        continue
                    stamped.append(
                        replace(
                            result,
                            tick_id=tick_id,
                            eager=eager,
                            audio_processed_s=audio_by_stream[result.stream_id],
                            valid_samples=frame.valid_samples,
                            is_last=frame.is_last,
                        )
                    )
                return stamped

            # Stamping and bookkeeping phase, under the engine lock when one is
            # supplied. `close_stream` runs here, on the tick thread, after each
            # stream's `is_last` frame.
            if lock is not None:
                lock.acquire()
            try:
                results: list[StepResult] = []
                results.extend(_stamp(steady_rows, eager=False))
                for rows in edge_rows:
                    results.extend(_stamp(rows, eager=True))

                stamped_aborts = {
                    row.stream_id for row in results if row.stream_id in aborted_streams
                }
                for stream_id in sorted(aborted_streams - stamped_aborts):
                    results.append(
                        StepResult(
                            stream_id=stream_id,
                            tick_id=tick_id,
                            partial_text="",
                            final_text=None,
                            audio_processed_s=audio_by_stream[stream_id],
                            eager=True,
                            valid_samples=0,
                            is_last=True,
                        )
                    )

                for session in closing:
                    self._close_session(session)
                    self._pipeline.close_stream(session.session_id)

                self._record(tick_id, plan, starved, step_ms, edge_ms)
            finally:
                if lock is not None:
                    lock.release()
        except Exception as exc:
            if lock is not None:
                lock.acquire()
            try:
                self.fail_live(exc)
                if plan is None:
                    plan = self._scheduler.plan([], [])
                self._record(tick_id, plan, starved, step_ms, edge_ms)
            finally:
                if lock is not None:
                    lock.release()
            self._clock.sleep_until(self._start + self._tick_id * self._config.chunk.period_s)
            return []

        self._clock.sleep_until(self._start + self._tick_id * self._config.chunk.period_s)
        return results

    def run_for(self, ticks: int) -> list[StepResult]:
        """`ticks` ticks against the clock. With a SimulatedClock this returns immediately,
        which is why the whole scheduler suite runs in milliseconds on a laptop."""
        out: list[StepResult] = []
        for _ in range(ticks):
            out.extend(self.run_tick())
        return out
