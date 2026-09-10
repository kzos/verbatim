# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Measured tick budget -> published concurrency ceiling, per chunk mode, per GPU.

``budget_ms = period_ms * utilisation_target`` (default 0.70). The ceiling is the
largest live count whose p95 modelled tick time fits the budget over the last 200
ticks. It starts from the harness's calibration row for this GPU and mode and
moves only downward at runtime. A refused session gets ``RESOURCE_EXHAUSTED``
with a retry-after hint.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. Here ``gpu_step``, ``host``, ``edge_step`` and ``emit`` arrive as plain
numbers on ``TickStats``: there is no timing measurement and no benchmark here,
only the EWMA and window arithmetic the runtime values will feed. This module
must not depend on the NeMo toolkit or on PyTorch.
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

#: Rolling p95 window length in ticks.
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
    """Admits iff ``live < ceiling`` and a slot is free, under a ceiling that starts
    at the calibrated value and moves only downward at runtime."""

    def __init__(self, config: EngineConfig, slots: SlotTable) -> None:
        self._config = config
        self._slots = slots
        self._ceiling = config.calibrated_ceiling
        self._step_ewma: dict[int, float] = {}
        self._edge_ewma: dict[int, float] = {}
        self._window: deque[float] = deque(maxlen=WINDOW_TICKS)
        self._consecutive_overruns = 0
        self._eager = EagerCounter()

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def ceiling(self) -> int | None:
        """The runtime ceiling, or ``None`` when uncalibrated. A ceiling nobody
        measured is not invented: without one, admission is bounded only by slots."""
        return self._ceiling

    @property
    def consecutive_overruns(self) -> int:
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
        """Fold one tick into the EWMAs and the rolling p95 window (200 ticks, alpha 0.1)."""
        self._fold(self._step_ewma, stats.steady_rows, stats.step_ms)
        self._fold(self._edge_ewma, stats.steady_rows, stats.edge_ms)
        self._window.append(tick_ms)
        self._eager.record(steps=1 + stats.edge_batches, eager_steps=stats.edge_batches)
        if self.p95_ms > self._config.budget_ms:
            self._consecutive_overruns += 1
            if self._ceiling is not None and self._ceiling > 1:
                # The runtime ceiling moves only downward: cheap ticks never raise it.
                self._ceiling -= 1
        else:
            self._consecutive_overruns = 0

    @staticmethod
    def _fold(table: dict[int, float], bucket: int, value: float) -> None:
        previous = table.get(bucket)
        table[bucket] = value if previous is None else ALPHA * value + (1 - ALPHA) * previous

    def decide(self, live: int) -> AdmissionDecision:
        """Admit iff ``live < ceiling`` (when calibrated) and a slot is free and the
        degradation ladder is not holding admissions; otherwise ``RESOURCE_EXHAUSTED``
        with a retry-after hint of one tick period."""
        hint = self._config.chunk.ms
        if self._slots.free() < 1:
            return AdmissionDecision(
                admitted=False, reason="RESOURCE_EXHAUSTED", retry_after_ms=hint
            )
        buckets = self._config.buckets
        assert buckets is not None
        if live >= max(buckets):
            return AdmissionDecision(
                admitted=False, reason="RESOURCE_EXHAUSTED", retry_after_ms=hint
            )
        if self.degradation_level >= 1:
            return AdmissionDecision(
                admitted=False, reason="RESOURCE_EXHAUSTED", retry_after_ms=hint
            )
        if self._ceiling is not None and live >= self._ceiling:
            return AdmissionDecision(
                admitted=False, reason="RESOURCE_EXHAUSTED", retry_after_ms=hint
            )
        return AdmissionDecision(admitted=True)
