# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``PcmFrame``, ``StepResult``, ``TickStats`` -- plain dataclasses.

No protobuf and no JSON on the tick path. Protocol encoding happens on the
asyncio loop, in the emitter, from these types.

These types are the stable boundary between the scheduler and any adapter: the
NeMo adapter in ``verbatim.pipelines.cache_aware_rnnt`` and the CPU fake in
``verbatim.pipelines.fake`` both produce them. This module must not depend on
the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["PAD_STREAM_ID_BASE", "PcmFrame", "StepResult", "TickStats", "Word"]

PAD_STREAM_ID_BASE: Final = -1


@dataclass(frozen=True, slots=True)
class Word:
    """One word with integer-millisecond timings. Moved here from protocols/base.py so a
    StepResult can carry words without the tick path importing anything under protocols/."""

    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class PcmFrame:
    """One row of one step. `samples` is float32 in [-1, 1], length == chunk samples always."""

    stream_id: int
    samples: np.ndarray
    is_first: bool
    is_last: bool
    valid_samples: int

    @property
    def is_pad(self) -> bool:
        """Pad rows use negative ids (-1, -2, ...); real sessions use positive ints."""
        return self.stream_id < 0


@dataclass(frozen=True, slots=True)
class StepResult:
    """One row's output for one tick. Pad-row results are dropped by the tick loop."""

    stream_id: int
    tick_id: int
    partial_text: str
    final_text: str | None
    audio_processed_s: float
    eager: bool
    words: tuple[Word, ...] = ()
    confidence: float = 1.0
    valid_samples: int = 0  # stamped by TickLoop from the frame this row came from
    is_last: bool = False  # stamped by TickLoop from the frame this row came from


@dataclass(frozen=True, slots=True)
class TickStats:
    """Per-tick bookkeeping: shapes that were stepped and what they cost as inputs."""

    tick_id: int
    steady_rows: int
    live_rows: int
    pad_rows: int
    edge_batches: int
    eager_rows: int
    starved: int
    step_ms: float
    edge_ms: float
