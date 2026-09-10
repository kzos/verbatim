# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Admission-control acceptance tests: ceiling, slots, the drop-and-return rule, the ladder."""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.types import PcmFrame, TickStats
from verbatim.scheduler.admission import OVERRUN_TRIP, WINDOW_TICKS, AdmissionController
from verbatim.scheduler.buckets import BucketScheduler
from verbatim.scheduler.slots import SlotTable

CHUNK = ChunkMode(160)
BUCKET = 8


def _config(**overrides) -> EngineConfig:
    args: dict = {"chunk": CHUNK, "buckets": (BUCKET,)}
    args.update(overrides)
    return EngineConfig(**args)


def _controller(config: EngineConfig, capacity: int | None = None) -> AdmissionController:
    return AdmissionController(config, SlotTable(capacity or config.num_slots))


def _stats(
    *, step_ms: float = 1.0, edge_ms: float = 0.0, edge_batches: int = 0, live_rows: int = 1
) -> TickStats:
    return TickStats(
        tick_id=0,
        steady_rows=BUCKET,
        live_rows=live_rows,
        pad_rows=BUCKET - live_rows,
        edge_batches=edge_batches,
        eager_rows=0,
        starved=0,
        step_ms=step_ms,
        edge_ms=edge_ms,
    )


def test_no_ceiling_when_uncalibrated() -> None:
    controller = _controller(_config(calibrated_ceiling=None))
    assert controller.ceiling is None
    # No ceiling: bounded by the largest bucket (and slots), not by a ceiling.
    assert controller.decide(BUCKET - 1).admitted is True
    decision = controller.decide(BUCKET)
    assert decision.admitted is False
    assert decision.reason == f"session refused: {BUCKET} live sessions fill the largest bucket"


def test_admits_below_the_ceiling() -> None:
    controller = _controller(_config(calibrated_ceiling=4))
    assert controller.decide(3).admitted is True


def test_rejects_at_the_ceiling() -> None:
    controller = _controller(_config(calibrated_ceiling=4))
    decision = controller.decide(4)
    assert decision.admitted is False
    assert decision.reason == "session refused: 4 live sessions at the ceiling of 4"
    assert decision.retry_after_ms > 0


def test_rejects_when_slots_are_exhausted_even_below_the_ceiling() -> None:
    config = _config(calibrated_ceiling=100)
    slots = SlotTable(config.num_slots)
    slots.reserve(config.num_slots)
    controller = AdmissionController(config, slots)
    decision = controller.decide(0)
    assert decision.admitted is False
    assert decision.reason == "session refused: no free slot"


def test_ceiling_never_exceeds_the_calibrated_value() -> None:
    controller = _controller(_config(calibrated_ceiling=8))
    for tick in range(500):
        controller.observe(_stats(step_ms=1.0), tick_ms=1.0)
        assert controller.ceiling == 8, f"ceiling moved on cheap tick {tick}"
    assert controller.decide(7).admitted is True


def test_ceiling_moves_down_under_overrun() -> None:
    controller = _controller(_config(calibrated_ceiling=8))
    # No ceiling decision is taken on a partial window: fill it with over-budget ticks.
    for _ in range(WINDOW_TICKS):
        controller.observe(_stats(step_ms=200.0), tick_ms=200.0)
    assert controller.ceiling is not None
    assert controller.ceiling < 8


def _calibrated_128() -> AdmissionController:
    return _controller(_config(buckets=(128,), calibrated_ceiling=128))


def _observe(
    controller: AdmissionController, *, live_rows: int, tick_ms: float, ticks: int
) -> None:
    for _ in range(ticks):
        controller.observe(_stats(step_ms=tick_ms, live_rows=live_rows), tick_ms=tick_ms)


def test_a_short_overrun_burst_does_not_collapse_the_ceiling() -> None:
    """The cold read's reproduction: 128 calibrated, a full healthy window, twelve ticks
    at 200 ms against a 112 ms budget, then healthy again.

    The window p95 stays over budget for about 190 ticks after such a burst. The rule
    this replaces decremented the ceiling on every one of them, so the ceiling read 1
    after 150 healthy ticks and stayed there for the life of the process while every
    other health signal recovered. The guard: the ceiling drops once, to one below the
    live count that was running, and returns to the calibrated value.
    """
    controller = _calibrated_128()
    _observe(controller, live_rows=64, tick_ms=60.0, ticks=WINDOW_TICKS)
    _observe(controller, live_rows=64, tick_ms=200.0, ticks=12)
    assert controller.degradation_level == 0
    lowest = 128
    for _ in range(2 * WINDOW_TICKS):
        controller.observe(_stats(step_ms=60.0, live_rows=64), tick_ms=60.0)
        assert controller.ceiling is not None
        assert controller.ceiling <= 128
        lowest = min(lowest, controller.ceiling)
    assert 63 <= lowest < 128
    assert controller.ceiling == 128
    assert controller.decide(8).admitted is True


def test_a_drop_is_taken_at_most_once_per_window() -> None:
    controller = _calibrated_128()
    _observe(controller, live_rows=64, tick_ms=60.0, ticks=WINDOW_TICKS)
    _observe(controller, live_rows=64, tick_ms=200.0, ticks=60)
    # Sixty over-budget ticks, one drop: to one below the live count, not sixty below.
    assert controller.ceiling == 63


def test_the_ladder_counts_this_ticks_cost_not_the_windows_memory() -> None:
    controller = _calibrated_128()
    _observe(controller, live_rows=64, tick_ms=60.0, ticks=WINDOW_TICKS)
    _observe(controller, live_rows=64, tick_ms=200.0, ticks=12)
    assert controller.consecutive_overruns == 12
    _observe(controller, live_rows=64, tick_ms=60.0, ticks=1)
    # The window still remembers the burst; the ladder does not.
    assert controller.p95_ms > controller.config.budget_ms
    assert controller.consecutive_overruns == 0
    assert controller.degradation_level == 0
    assert controller.decide(8).admitted is True


def test_sustained_overrun_drops_the_ceiling_below_the_live_count_then_restores_it() -> None:
    controller = _calibrated_128()
    _observe(controller, live_rows=64, tick_ms=60.0, ticks=WINDOW_TICKS)
    _observe(controller, live_rows=64, tick_ms=200.0, ticks=3 * OVERRUN_TRIP)
    assert controller.degradation_level == 3
    assert controller.ceiling is not None
    assert controller.ceiling <= 63
    assert controller.decide(8).admitted is False
    _observe(controller, live_rows=8, tick_ms=60.0, ticks=3 * WINDOW_TICKS)
    assert controller.degradation_level == 0
    assert controller.ceiling == 128
    assert controller.decide(8).admitted is True


def test_degradation_ladder_stops_admitting_after_20_consecutive_overruns() -> None:
    controller = _controller(_config(calibrated_ceiling=32))
    for _ in range(19):
        controller.observe(_stats(step_ms=200.0), tick_ms=200.0)
    assert controller.consecutive_overruns == 19
    assert controller.degradation_level == 0
    assert controller.decide(0).admitted is True
    controller.observe(_stats(step_ms=200.0), tick_ms=200.0)
    assert controller.consecutive_overruns == 20
    assert controller.degradation_level == 1
    assert controller.decide(0).admitted is False


def test_degradation_never_sheds_pad_rows() -> None:
    config = _config(calibrated_ceiling=32)
    scheduler = BucketScheduler(config)
    live = [
        PcmFrame(
            stream_id=i + 1,
            samples=np.zeros(config.chunk.samples, dtype=np.float32),
            is_first=False,
            is_last=False,
            valid_samples=config.chunk.samples,
        )
        for i in range(2)
    ]
    for overruns in (0, 20, 40, 60):
        controller = _controller(config)
        for _ in range(overruns):
            controller.observe(_stats(step_ms=200.0), tick_ms=200.0)
        plan = scheduler.plan(live, [])
        assert len(plan.steady) == BUCKET
        assert plan.pad_rows == BUCKET - 2


def test_budget_is_the_period_times_the_utilisation_target() -> None:
    assert _config().budget_ms == pytest.approx(112.0)


def test_ewma_alpha_is_applied() -> None:
    controller = _controller(_config(calibrated_ceiling=8))
    controller.observe(_stats(step_ms=1.0), tick_ms=1.0)
    controller.observe(_stats(step_ms=11.0), tick_ms=11.0)
    assert controller.step_ewma(BUCKET) == pytest.approx(2.0)


def test_p95_is_the_95th_percentile_not_a_lower_one() -> None:
    controller = _controller(_config(calibrated_ceiling=8))
    for _ in range(160):
        controller.observe(_stats(step_ms=1.0), tick_ms=1.0)
    for _ in range(40):
        controller.observe(_stats(step_ms=500.0), tick_ms=500.0)
    assert controller.p95_ms == pytest.approx(500.0)
    assert controller.consecutive_overruns >= 1
    assert controller.ceiling is not None
    assert controller.ceiling < 8


def test_uncalibrated_admission_still_stops_at_the_largest_bucket() -> None:
    controller = _controller(_config(calibrated_ceiling=None))
    assert controller.decide(BUCKET - 1).admitted is True
    decision = controller.decide(BUCKET)
    assert decision.admitted is False
    assert decision.reason == f"session refused: {BUCKET} live sessions fill the largest bucket"
