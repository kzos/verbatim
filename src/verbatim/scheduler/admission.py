# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Measured tick budget -> published concurrency ceiling, per chunk mode, per GPU.

``budget_ms = period_ms * utilisation_target`` (default 0.70). The ceiling starts
from the harness's calibration row for this GPU and mode and never exceeds it. A
refused session gets ``RESOURCE_EXHAUSTED`` with a retry-after hint.

Two runtime signals move admission, and they are kept separate on purpose:

- The **degradation ladder** counts consecutive ticks whose own cost exceeded the
  budget. Twenty in a row stop admissions; one tick back inside the budget resets
  the count. It answers "is the box overloaded now".
- The **ceiling** follows the p95 over the last 200 ticks. When that p95 is over
  budget and the current tick is too, the ceiling drops to one below the live
  count that was running, at most once per window. Once a full window has passed
  since the last drop and the p95 is back inside the budget, the ceiling returns
  to the calibrated value. It never goes above it.

Why the separation matters: a burst of over-budget ticks keeps the window p95
over budget for the next 190 ticks after the burst ends. An earlier rule
decremented the ceiling on every one of those ticks and keyed the ladder on the
same window, so a two-second hiccup on a 128-stream calibration left a
one-stream server for the life of the process, with every health signal clean,
and refused admissions for thirty seconds after the hiccup was over.

Here ``gpu_step``, ``host``, ``edge_step`` and ``emit`` arrive as plain numbers on
``TickStats``: there is no timing measurement and no benchmark here, only the
EWMA and window arithmetic the runtime values feed. This module must not depend
on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from verbatim.config import EngineConfig
from verbatim.core.types import TickStats
from verbatim.scheduler.boundary import EagerCounter
from verbatim.scheduler.slots import SlotTable

__all__ = ["AdmissionController", "AdmissionDecision"]

#: EWMA weight for each newly observed tick.
ALPHA: float = 0.1

#: Rolling p95 window length in ticks. Also the least spacing between two drops of
#: the ceiling, and the wait after a drop before the ceiling can return.
WINDOW_TICKS: int = 200

#: Consecutive over-budget ticks before the ladder stops admitting.
OVERRUN_TRIP: int = 20


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Admit, or refuse with the protocol code and a retry-after hint."""

    admitted: bool
    reason: str = ""
    retry_after_ms: int = 0


class AdmissionController:
    """Admits iff ``live < ceiling`` and a slot is free, under a ceiling that starts at
    the calibrated value, never exceeds it, drops below the running live count on a
    sustained over-budget window and returns to the calibrated value after a clean
    one."""

    def __init__(self, config: EngineConfig, slots: SlotTable) -> None:
        self._config = config
        self._slots = slots
        self._calibrated = config.calibrated_ceiling
        self._ceiling = config.calibrated_ceiling
        self._step_ewma: dict[int, float] = {}
        self._edge_ewma: dict[int, float] = {}
        self._window: deque[float] = deque(maxlen=WINDOW_TICKS)
        self._consecutive_overruns = 0
        # Counted as if a full window had already passed, so the first over-budget
        # window can drop the ceiling as soon as the window itself is full.
        self._ticks_since_drop = WINDOW_TICKS
        self._eager = EagerCounter()

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def calibrated_ceiling(self) -> int | None:
        """The value from the calibration row. The runtime ceiling never exceeds it."""
        return self._calibrated

    @property
    def ceiling(self) -> int | None:
        """The runtime ceiling, or ``None`` when uncalibrated. A ceiling nobody
        measured is not invented: without one, admission is bounded only by slots
        and the largest bucket."""
        return self._ceiling

    @property
    def consecutive_overruns(self) -> int:
        """Consecutive ticks whose own cost exceeded the budget; zero after any tick
        inside it."""
        return self._consecutive_overruns

    @property
    def degradation_level(self) -> int:
        """0 = healthy, 1 = stop admitting, 2 = stop burst rounds as well,
        3 = also shrink the bucket when elastic buckets are enabled."""
        if self._consecutive_overruns >= 3 * OVERRUN_TRIP:
            return 3
        if self._consecutive_overruns >= 2 * OVERRUN_TRIP:
            return 2
        if self._consecutive_overruns >= OVERRUN_TRIP:
            return 1
        return 0

    @property
    def eager_step_fraction(self) -> float:
        """``eager_steps / total_steps`` over ticks observed so far."""
        return self._eager.fraction

    @property
    def p95_ms(self) -> float:
        """p95 of the modelled tick time over the rolling window; 0.0 when empty."""
        if not self._window:
            return 0.0
        ordered = sorted(self._window)
        rank = max(0, -(-len(ordered) * 95 // 100) - 1)
        return ordered[rank]

    def step_ewma(self, bucket: int) -> float:
        """The current EWMA of the steady-step cost for a bucket; 0.0 when unseen."""
        return self._step_ewma.get(bucket, 0.0)

    def edge_ewma(self, bucket: int) -> float:
        """The current EWMA of the per-edge-batch cost for a bucket; 0.0 when unseen."""
        return self._edge_ewma.get(bucket, 0.0)

    def observe(self, stats: TickStats, *, tick_ms: float) -> None:
        """Fold one tick into the EWMAs, the ladder and the rolling p95 window."""
        self._fold(self._step_ewma, stats.steady_rows, stats.step_ms)
        self._fold(self._edge_ewma, stats.steady_rows, stats.edge_ms)
        self._window.append(tick_ms)
        self._eager.record(steps=1 + stats.edge_batches, eager_steps=stats.edge_batches)
        over_budget = tick_ms > self._config.budget_ms
        # The ladder counts this tick's own cost, never the window's memory of an
        # earlier one, so it reacts to overload that is happening now.
        if over_budget:
            self._consecutive_overruns += 1
        else:
            self._consecutive_overruns = 0
        self._ticks_since_drop += 1
        self._move_ceiling(stats.live_rows, over_budget)

    def _move_ceiling(self, live_rows: int, over_budget: bool) -> None:
        """One drop per window at most, and a return only after a window has passed.

        Both waits are what stop one burst from becoming two hundred decrements: the
        window p95 stays over budget for 190 ticks after a 12-tick burst, and a rule
        that acted on every one of those ticks collapsed the ceiling to 1.
        """
        if self._ceiling is None or self._calibrated is None:
            return
        if len(self._window) < WINDOW_TICKS or self._ticks_since_drop < WINDOW_TICKS:
            return
        if self.p95_ms > self._config.budget_ms:
            if over_budget:
                # Drop below the live count that was running when the window
                # tripped, and only while the overload is current: a window that
                # merely remembers a burst does not drop the ceiling again.
                target = max(1, min(self._ceiling - 1, live_rows - 1))
                if target < self._ceiling:
                    self._ceiling = target
                    self._ticks_since_drop = 0
        elif self._ceiling < self._calibrated:
            # A full window inside the budget since the last drop: the calibrated
            # value is a measurement and the drop was a reaction, so return to it.
            self._ceiling = self._calibrated

    @staticmethod
    def _fold(table: dict[int, float], bucket: int, value: float) -> None:
        previous = table.get(bucket)
        table[bucket] = value if previous is None else ALPHA * value + (1 - ALPHA) * previous

    def decide(self, live: int) -> AdmissionDecision:
        """Admit iff ``live < ceiling`` (when calibrated) and a slot is free and the
        degradation ladder is not holding admissions; otherwise a refusal whose
        ``reason`` names which limit bound, with a retry-after hint of one tick
        period. The reason reaches the wire as the error text, so it says what
        happened rather than repeating the status code."""
        hint = self._config.chunk.ms
        if self._slots.free() < 1:
            return AdmissionDecision(
                admitted=False, reason="session refused: no free slot", retry_after_ms=hint
            )
        buckets = self._config.buckets
        assert buckets is not None
        if live >= max(buckets):
            return AdmissionDecision(
                admitted=False,
                reason=f"session refused: {live} live sessions fill the largest bucket",
                retry_after_ms=hint,
            )
        if self.degradation_level >= 1:
            return AdmissionDecision(
                admitted=False,
                reason=f"session refused: admissions held at degradation level "
                f"{self.degradation_level}",
                retry_after_ms=hint,
            )
        if self._ceiling is not None and live >= self._ceiling:
            return AdmissionDecision(
                admitted=False,
                reason=f"session refused: {live} live sessions at the ceiling of {self._ceiling}",
                retry_after_ms=hint,
            )
        return AdmissionDecision(admitted=True)
