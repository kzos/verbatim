# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""First/last steps segregated into a small eager side-batch (default ``B_edge = 8``).

The day-45 gate requires the eager step fraction to stay at or below 2 %.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU: its final (``is_last``) sub-batch is never captured, which is why
finals are peeled off here instead of mixed into the steady batch. This module is
batch assembly and counters only. This module must not depend on the NeMo toolkit
or on PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from verbatim.config import ChunkMode
from verbatim.core.types import PAD_STREAM_ID_BASE, PcmFrame

__all__ = ["EagerCounter", "PadPool", "partition_edge_batches"]


class PadPool:
    """Persistent pad rows with reserved negative ids, registered once at warm-up and never
    marked is_last. They hold slots for the life of the process; their outputs are dropped.
    They exist to keep the steady batch's shape constant, and are never shed under load."""

    def __init__(self, size: int, chunk: ChunkMode) -> None:
        if size < 0:
            raise ValueError(f"pad pool size must be >= 0, got {size!r}")
        zeros = np.zeros(chunk.samples, dtype=np.float32)
        self._rows: tuple[PcmFrame, ...] = tuple(
            PcmFrame(
                stream_id=PAD_STREAM_ID_BASE - i,
                samples=zeros,
                is_first=False,
                is_last=False,
                valid_samples=chunk.samples,
            )
            for i in range(size)
        )
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._rows)

    def reset(self) -> None:
        """Rewind the per-tick cursor. Every tick starts from the first pad row, so
        a pad row is handed out at most once per tick across all batches of the plan."""
        self._cursor = 0

    def take(self, n: int) -> tuple[PcmFrame, ...]:
        """The next ``n`` persistent rows. The pool is sized for the largest demand
        (the full bucket plus a full edge batch), so running short is a caller bug."""
        if n < 0 or self._cursor + n > len(self._rows):
            raise ValueError(f"pad take of {n} out of range for a pool of {len(self._rows)} rows")
        out = self._rows[self._cursor : self._cursor + n]
        self._cursor += n
        return out


def partition_edge_batches(
    finals: Sequence[PcmFrame],
    edge_batch: int,
    pads: PadPool,
) -> tuple[tuple[PcmFrame, ...], ...]:
    """Split final rows into ``ceil(len(finals) / edge_batch)`` batches of exactly
    ``edge_batch`` rows, padded with pad rows. Empty input gives no batches."""
    if edge_batch < 1:
        raise ValueError(f"edge_batch must be >= 1, got {edge_batch!r}")
    batches: list[tuple[PcmFrame, ...]] = []
    for start in range(0, len(finals), edge_batch):
        group = tuple(finals[start : start + edge_batch])
        batches.append(group + pads.take(edge_batch - len(group)))
    return tuple(batches)


class EagerCounter:
    """The eager-step fraction: ``eager_steps / total_steps``, where an eager step is
    one edge-batch call. A counter computed from the plan, never a claim about hardware."""

    def __init__(self) -> None:
        self._steps = 0
        self._eager_steps = 0

    @property
    def steps(self) -> int:
        """Total pipeline calls recorded (steady plus edge)."""
        return self._steps

    @property
    def eager_steps(self) -> int:
        """Edge-batch calls recorded."""
        return self._eager_steps

    @property
    def fraction(self) -> float:
        """``eager_steps / total_steps``; 0.0 before anything ran. Reported even when
        high: a gate you can only pass is not a gate."""
        if self._steps == 0:
            return 0.0
        return self._eager_steps / self._steps

    def record(self, *, steps: int, eager_steps: int) -> None:
        """Fold one tick's pipeline calls into the counters."""
        if steps < 0 or eager_steps < 0 or eager_steps > steps:
            raise ValueError(
                f"invalid step counts steps={steps!r} eager_steps={eager_steps!r}: "
                f"require 0 <= eager_steps <= steps"
            )
        self._steps += steps
        self._eager_steps += eager_steps
