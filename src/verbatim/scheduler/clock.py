# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Real and simulated clocks. The simulated one is why the scheduler tests run on CPU.

The GPU is needed to find out what the tick budget *is*; it is not needed to test
what the scheduler does with it.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This module must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

import time
from typing import Protocol

__all__ = ["Clock", "MonotonicClock", "ScaledMonotonicClock", "SimulatedClock"]


class Clock(Protocol):
    """The tick loop's only view of time: read now, wait for a boundary."""

    def now(self) -> float: ...
    def sleep_until(self, deadline: float) -> None: ...


class MonotonicClock:
    """time.monotonic() and a real sleep."""

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class ScaledMonotonicClock:
    """A monotonic clock whose waits are divided by `scale`, for tests that need a real
    thread and a real tick grid without real time.

    Boundary arithmetic is identical to MonotonicClock's -- the loop still computes
    `start + tick_id * period_s` -- so a scaled run exercises the same code path as a
    real one. `scale` is a test-harness input, not a measurement of anything.
    """

    def __init__(self, scale: float = 1.0) -> None:
        if scale < 1.0:
            raise ValueError(f"scale must be >= 1.0, got {scale!r}")
        self._scale = scale
        self._start_wall = time.monotonic()
        self._start = self._start_wall

    @property
    def scale(self) -> float:
        """The wait divisor: a test-harness input, not a measurement of anything."""
        return self._scale

    def now(self) -> float:
        return self._start + (time.monotonic() - self._start_wall) * self._scale

    def sleep_until(self, deadline: float) -> None:
        delay = (deadline - self.now()) / self._scale
        if delay > 0:
            time.sleep(delay)


class SimulatedClock:
    """Advances only when told to. This is why the scheduler tests run on CPU in milliseconds."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._waits: list[float] = []

    def now(self) -> float:
        return self._now

    def sleep_until(self, deadline: float) -> None:
        """Jump to the deadline, recording the wait. A deadline already past records
        a zero wait and never moves time backwards: a late tick delays audio, it does
        not shift later boundaries."""
        wait = deadline - self._now
        if wait > 0:
            self._waits.append(wait)
            self._now = deadline
        else:
            self._waits.append(0.0)

    def advance(self, seconds: float) -> None:
        """Move time forward outside the tick loop, e.g. to simulate an overrunning tick."""
        self._now += seconds

    @property
    def waits(self) -> list[float]:
        """One entry per ``sleep_until`` call, in order."""
        return list(self._waits)
