# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""What the engine reports about itself: counters, the latency windows, admissions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import ResourceExhausted
from verbatim.core.types import PcmFrame, StepResult
from verbatim.engine import Engine, stub_engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.clock import ScaledMonotonicClock

pytestmark = pytest.mark.cpu

CHUNK = ChunkMode(160)
OPTIONS = SessionOptions(chunk_ms=160)


def _chunk() -> bytes:
    return b"\x00" * OPTIONS.chunk_bytes


async def test_before_start_the_snapshot_says_so() -> None:
    engine = stub_engine()
    snapshot = engine.snapshot()
    assert snapshot.started is False and snapshot.running is False and snapshot.dead is False
    assert snapshot.tick_id == 0 and snapshot.last_tick_age_s is None
    assert snapshot.live_sessions == 0 and snapshot.counters.ticks_total == 0
    assert snapshot.chunk_ms == 160 and snapshot.bucket == 8
    assert snapshot.ceiling is None and snapshot.calibrated_ceiling is None


async def test_ticks_costs_and_partial_latency_are_observed_while_running() -> None:
    engine = stub_engine()
    async with engine:
        session = engine.open_session(OPTIONS)
        assert session.feed(_chunk()) == OPTIONS.chunk_bytes
        await engine.wait_for_ticks(3)
        snapshot = engine.snapshot()
        assert snapshot.started and snapshot.running and not snapshot.dead
        assert snapshot.tick_id >= 3
        assert snapshot.counters.ticks_total >= 3
        assert snapshot.counters.steady_steps_total >= 1
        assert snapshot.live_sessions == 1
        assert snapshot.slots_free == snapshot.slots_capacity - snapshot.slots_reserved
        assert snapshot.latency_count >= 1, "each processed wake observes one partial latency"
        p50, p95, p99 = snapshot.partial_latency_ms
        assert 0.0 <= p50 <= p95 <= p99
        assert snapshot.last_tick_age_s is not None and snapshot.last_tick_age_s >= 0.0
    assert engine.snapshot().running is False


async def test_admissions_and_the_last_refusal_reason_are_recorded() -> None:
    engine = stub_engine(bucket=2)
    async with engine:
        engine.open_session(OPTIONS)
        engine.open_session(OPTIONS)
        with pytest.raises(ResourceExhausted):
            engine.open_session(OPTIONS)
        snapshot = engine.snapshot()
    assert snapshot.counters.sessions_admitted_total == 2
    assert snapshot.counters.sessions_refused_total == 1
    assert snapshot.last_refusal_reason is not None
    assert "fill the largest bucket" in snapshot.last_refusal_reason


class _SlowAndCostly(FakePipelineAdapter):
    """Sleeps for real, so the tick ends after its boundary, and declares a modelled
    step cost over the budget, which is the number the admission controller and the
    counters observe; the two are different measurements on purpose."""

    def __init__(self, seconds: float, declared_ms: float) -> None:
        super().__init__(CHUNK, buckets=(1,))
        self.seconds = seconds
        self.declared_ms = declared_ms

    @property
    def step_ms(self) -> float:
        return self.declared_ms

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        time.sleep(self.seconds)
        return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)


async def test_a_tick_past_its_boundary_is_late_and_a_costly_one_is_over_budget() -> None:
    """A 30 ms step on a clock running a hundred times real time is 3 s of engine time
    against a 160 ms period, so the tick ends long after the next boundary; a declared
    cost of 200 ms exceeds the 112 ms budget."""
    config = EngineConfig(chunk=CHUNK, buckets=(1,), edge_batch=1)
    pipeline = _SlowAndCostly(0.03, declared_ms=200.0)
    engine = Engine(config, pipeline, clock=ScaledMonotonicClock(100.0))
    async with engine:
        session = engine.open_session(OPTIONS)
        assert session.feed(_chunk()) == OPTIONS.chunk_bytes
        await engine.wait_for_ticks(2)
        snapshot = engine.snapshot()
    assert snapshot.counters.ticks_late_total >= 1
    assert snapshot.counters.ticks_over_budget_total >= 1
    assert snapshot.tick_cost_ms[1] == pytest.approx(200.0)


async def test_a_dead_tick_loop_is_visible_in_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline step that raises fails the sessions and the loop goes on; only a
    scheduler failure kills the thread, which is what the snapshot must show."""
    engine = stub_engine()
    real_run_tick = engine._tick.run_tick

    def run_tick(*, lock: object = None, publish: object = None) -> list[StepResult]:
        # The signature has to match the real one, publish included. Without it the
        # loop died on a TypeError at the first tick instead of the MemoryError at the
        # third, so this passed for the wrong reason and did so only sometimes.
        if engine.tick_id >= 2:
            raise MemoryError("tick thread died")
        return real_run_tick(lock=lock, publish=publish)

    monkeypatch.setattr(engine._tick, "run_tick", run_tick)
    async with engine:
        session = engine.open_session(OPTIONS)
        assert session.feed(_chunk()) == OPTIONS.chunk_bytes
        for _ in range(500):
            if engine.snapshot().dead:
                break
            await asyncio.sleep(0.01)
        snapshot = engine.snapshot()
    assert snapshot.dead is True and snapshot.running is False
