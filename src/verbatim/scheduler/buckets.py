# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Fixed bucket sizes per chunk mode, assembled from the live registry.

The batch dimension is part of NeMo's graph key, so the steady batch never
shrinks: it is padded to exactly ``B`` with persistent pad sessions holding
reserved negative stream ids. Default policy is ``n = 1``, ``B_1 = ceiling(mode)``
-- the fixed-shape default. Elastic buckets are opt-in and must pass the
invariance gate for that bucket list before a release enables them.

This module is batch-shape arithmetic only, tested against the CPU fake in
``verbatim.pipelines.fake``. This module must not depend on the NeMo toolkit or
on PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verbatim.config import EngineConfig
from verbatim.core.types import PcmFrame
from verbatim.scheduler.boundary import PadPool, partition_edge_batches
from verbatim.scheduler.graph_budget import ConfigError

__all__ = ["BucketPlan", "BucketScheduler"]


@dataclass(frozen=True, slots=True)
class BucketPlan:
    """One tick's batch layout. ``steady`` is always exactly ``bucket`` rows and never
    holds an ``is_last`` frame; every edge batch is always exactly ``edge_batch`` rows."""

    bucket: int
    steady: tuple[PcmFrame, ...]
    edge_batches: tuple[tuple[PcmFrame, ...], ...]
    live_rows: int
    pad_rows: int
    eager_rows: int


class BucketScheduler:
    """Pads the steady batch to exactly ``B`` and peels finals into edge batches."""

    def __init__(self, config: EngineConfig) -> None:
        if config.buckets is None or len(config.buckets) == 0:
            raise ConfigError("BucketScheduler requires a non-empty bucket list")
        self._config = config
        # Steady padding needs a full bucket; each edge batch needs a full edge batch.
        self._pads = PadPool(max(config.buckets) + config.edge_batch, config.chunk)

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def pads(self) -> PadPool:
        """The persistent pad rows. Never shed: they are what keeps the shape."""
        return self._pads

    def select_bucket(self, live_rows: int) -> int:
        """The smallest configured bucket that fits. With one bucket (the default) always
        that one. Raises when live_rows exceeds the largest bucket -- admission should have
        made that unreachable."""
        buckets = self._config.buckets
        assert buckets is not None
        if live_rows < 0:
            raise ValueError(f"live_rows must be >= 0, got {live_rows!r}")
        if live_rows > max(buckets):
            raise RuntimeError(
                f"live rows {live_rows} exceed the largest bucket {max(buckets)}: "
                f"admission should have made this unreachable"
            )
        if not self._config.elastic_buckets:
            # Fixed-shape policy: one shape at any occupancy, so the graph key is stable.
            return max(buckets)
        for bucket in buckets:
            if live_rows <= bucket:
                return bucket
        raise RuntimeError(  # Unreachable: the guard above already rejected the overflow.
            f"live rows {live_rows} exceed the largest bucket {max(buckets)}: "
            f"admission should have made this unreachable"
        )

    def plan(self, non_final: Sequence[PcmFrame], final: Sequence[PcmFrame]) -> BucketPlan:
        """Steady rows padded to exactly the bucket; finals partitioned into edge batches of
        exactly edge_batch rows, padded. `is_last` frames NEVER enter the steady batch."""
        self._pads.reset()
        for frame in non_final:
            if frame.is_last:
                raise ValueError(
                    f"non-final row for stream {frame.stream_id} has is_last=True: "
                    f"finals must be passed separately so they never enter the steady batch"
                )
        live = sorted(non_final, key=lambda f: f.stream_id)
        bucket = self.select_bucket(len(live))
        steady = tuple(live) + self._pads.take(bucket - len(live))
        edge_batches = partition_edge_batches(tuple(final), self._config.edge_batch, self._pads)
        return BucketPlan(
            bucket=bucket,
            steady=steady,
            edge_batches=edge_batches,
            live_rows=len(live) + len(final),
            pad_rows=bucket - len(live),
            eager_rows=sum(len(batch) for batch in edge_batches),
        )
