# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The two config rules a server makes reachable: no bucket is ever assumed, and a pad
pool that cannot fill the steady shape is refused as the invariance break it is. Plus
the slot arithmetic, term by term."""

from __future__ import annotations

import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import STUB_ENGINE_BUCKET, stub_engine
from verbatim.scheduler.graph_budget import ConfigError

pytestmark = pytest.mark.cpu

CHUNK = ChunkMode(160)


def test_a_config_without_a_bucket_or_a_ceiling_is_refused() -> None:
    with pytest.raises(ConfigError, match="no bucket") as raised:
        EngineConfig(chunk=CHUNK)
    assert "stub_engine" in str(raised.value)


def test_a_ceiling_alone_is_the_single_bucket() -> None:
    config = EngineConfig(chunk=CHUNK, calibrated_ceiling=24)
    assert config.buckets == (24,)
    assert config.calibrated_ceiling == 24


def test_the_stub_engine_carries_its_own_bucket_and_says_it_is_not_a_measurement() -> None:
    assert stub_engine()._config.buckets == (STUB_ENGINE_BUCKET,)
    assert stub_engine(bucket=3)._config.buckets == (3,)


@pytest.mark.parametrize("pad_pool", [1, 7, 31])
def test_a_pad_pool_that_cannot_fill_the_steady_shape_is_an_invariance_break(pad_pool: int) -> None:
    with pytest.raises(ConfigError, match="invariance break by configuration") as raised:
        EngineConfig(chunk=CHUNK, buckets=(32,), pad_pool=pad_pool)
    message = str(raised.value)
    assert f"pad_pool={pad_pool}" in message
    assert "largest bucket is 32" in message
    assert "at least 32" in message


@pytest.mark.parametrize(("pad_pool", "effective"), [(0, 32), (32, 32), (40, 40)])
def test_zero_or_at_least_the_bucket_is_accepted(pad_pool: int, effective: int) -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(32,), pad_pool=pad_pool)
    assert config.effective_pad == effective


def test_num_slots_is_the_sum_of_its_named_terms() -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(32,))
    assert config.edge_pad_rows == 7
    assert config.num_slots == 32 + 32 + 7 + 8
    assert config.num_slots == (
        max(config.buckets or ())
        + config.effective_pad
        + config.edge_pad_rows
        + config.drain_margin
    )


def test_an_edge_batch_of_one_has_no_edge_pad_rows() -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(8,), edge_batch=1, drain_margin=0)
    assert config.edge_pad_rows == 0
    assert config.num_slots == 8 + 8
