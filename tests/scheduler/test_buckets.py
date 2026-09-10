# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Bucket-planner acceptance tests: fixed steady shape, edge segregation, pad rows."""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.types import PcmFrame
from verbatim.scheduler.buckets import BucketScheduler

CHUNK = ChunkMode(160)
N = CHUNK.samples
BUCKET = 8


def _config(**overrides) -> EngineConfig:
    args: dict = {"chunk": CHUNK, "buckets": (BUCKET,), "calibrated_ceiling": BUCKET}
    args.update(overrides)
    return EngineConfig(**args)


def _live_frame(stream_id: int, *, is_last: bool = False) -> PcmFrame:
    return PcmFrame(
        stream_id=stream_id,
        samples=np.zeros(N, dtype=np.float32),
        is_first=False,
        is_last=is_last,
        valid_samples=N,
    )


def _scheduler(**overrides) -> BucketScheduler:
    return BucketScheduler(_config(**overrides))


@pytest.mark.parametrize("live_count", [0, 1, 7, BUCKET - 1, BUCKET])
def test_steady_batch_is_always_exactly_the_bucket(live_count: int) -> None:
    plan = _scheduler().plan([_live_frame(i + 1) for i in range(live_count)], [])
    assert len(plan.steady) == BUCKET
    assert plan.bucket == BUCKET


def test_finals_never_enter_the_steady_batch() -> None:
    plan = _scheduler().plan(
        [_live_frame(1), _live_frame(2), _live_frame(3)],
        [_live_frame(4, is_last=True), _live_frame(5, is_last=True)],
    )
    assert not any(frame.is_last for frame in plan.steady)
    assert sorted(f.stream_id for f in plan.steady if not f.is_pad) == [1, 2, 3]
    assert plan.edge_batches != ()


def test_finals_are_partitioned_into_edge_batches_of_exactly_b_edge() -> None:
    plan = _scheduler().plan([], [_live_frame(i + 1, is_last=True) for i in range(9)])
    assert len(plan.edge_batches) == 2
    assert all(len(batch) == 8 for batch in plan.edge_batches)
    first, second = plan.edge_batches
    assert sum(1 for f in first if not f.is_pad) == 8
    assert sum(1 for f in second if not f.is_pad) == 1


def test_no_edge_batches_when_there_are_no_finals() -> None:
    plan = _scheduler().plan([_live_frame(1)], [])
    assert plan.edge_batches == ()
    assert plan.eager_rows == 0


def test_pad_rows_have_negative_stream_ids_and_are_flagged() -> None:
    plan = _scheduler().plan([], [])
    assert len(plan.steady) == BUCKET
    for frame in plan.steady:
        assert frame.stream_id < 0
        assert frame.is_pad
        assert frame.is_last is False


def test_row_order_is_live_ascending_then_pad() -> None:
    plan = _scheduler().plan([_live_frame(9), _live_frame(2), _live_frame(5)], [])
    live_ids = [f.stream_id for f in plan.steady if not f.is_pad]
    pad_ids = [f.stream_id for f in plan.steady if f.is_pad]
    assert live_ids == [2, 5, 9]
    assert live_ids + pad_ids == [f.stream_id for f in plan.steady]


def test_select_bucket_default_is_the_single_bucket() -> None:
    scheduler = _scheduler()
    for live_rows in (0, 1, BUCKET - 1, BUCKET):
        assert scheduler.select_bucket(live_rows) == BUCKET


def test_select_bucket_elastic_picks_the_smallest_that_fits() -> None:
    scheduler = _scheduler(buckets=(4, 8, 16), elastic_buckets=True, calibrated_ceiling=16)
    assert scheduler.select_bucket(0) == 4
    assert scheduler.select_bucket(4) == 4
    assert scheduler.select_bucket(5) == 8
    assert scheduler.select_bucket(16) == 16
    with pytest.raises(RuntimeError):
        scheduler.select_bucket(17)
    # Without the opt-in the shape stays fixed at the largest bucket.
    fixed = _scheduler(buckets=(4, 8, 16), calibrated_ceiling=16)
    assert fixed.select_bucket(5) == 16


def test_elastic_buckets_are_off_by_default() -> None:
    config = EngineConfig(chunk=CHUNK, calibrated_ceiling=16)
    assert len(config.buckets or ()) == 1
    assert config.elastic_buckets is False


def test_select_bucket_raises_above_the_largest_bucket_in_fixed_mode() -> None:
    with pytest.raises(RuntimeError, match="admission should have made this unreachable"):
        _scheduler().select_bucket(BUCKET + 1)


def test_pad_rows_are_not_shared_between_the_steady_and_edge_batches() -> None:
    plan = _scheduler().plan(
        [_live_frame(1), _live_frame(2)],
        [_live_frame(3, is_last=True)],
    )
    steady_pads = [f.stream_id for f in plan.steady if f.is_pad]
    edge_pads = [[f.stream_id for f in batch if f.is_pad] for batch in plan.edge_batches]
    assert len(plan.edge_batches) == 1
    groups = [steady_pads, *edge_pads]
    for i, group in enumerate(groups):
        for other in groups[i + 1 :]:
            assert set(group).isdisjoint(other)
    seen = [sid for group in groups for sid in group]
    assert len(set(seen)) == len(seen)
