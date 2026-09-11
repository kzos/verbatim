# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""A bounded window of latency samples with exact quantiles over the window.

One sketch per quantity; the engine keeps one for the tick cost and one for the
server-side partial latency (the tick's lateness past its boundary plus the delay
until the loop routed its rows). Quantiles are nearest-rank over the retained
window, so they are exact for what was retained and say nothing about what was not.
"""

from __future__ import annotations

from collections import deque

__all__ = ["LatencySketch"]


class LatencySketch:
    def __init__(self, window: int = 1000) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        self._samples: deque[float] = deque(maxlen=window)

    def observe(self, ms: float) -> None:
        self._samples.append(float(ms))

    @property
    def count(self) -> int:
        return len(self._samples)

    def quantile(self, q: float) -> float:
        """Nearest-rank quantile over the window; 0.0 when empty."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {q!r}")
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        rank = max(0, -(-len(ordered) * int(q * 100) // 100) - 1)
        return ordered[rank]

    @property
    def p50(self) -> float:
        return self.quantile(0.5)

    @property
    def p95(self) -> float:
        return self.quantile(0.95)

    @property
    def p99(self) -> float:
        return self.quantile(0.99)
