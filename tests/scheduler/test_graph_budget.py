# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Graph-budget acceptance tests: the capture-key count, the config-time assert."""

from __future__ import annotations

import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.scheduler.graph_budget import (
    ConfigError,
    assert_graph_budget,
    graph_keys_required,
)


def test_single_mode_single_bucket_fits() -> None:
    assert graph_keys_required({160: (16,)}) == 2
    assert_graph_budget({160: (16,)}, 8)


def test_budget_is_exceeded_and_named() -> None:
    buckets = {80: (4, 8), 160: (4, 8), 560: (4, 8), 1120: (4, 8)}
    assert graph_keys_required(buckets) == 9
    with pytest.raises(ConfigError, match=r"9.*max_graphs=8"):
        assert_graph_budget(buckets, 8)


def test_budget_is_checked_at_config_time() -> None:
    with pytest.raises(ConfigError):
        EngineConfig(
            chunk=ChunkMode(80),
            buckets=(8, 16, 32),
            elastic_buckets=True,
            calibrated_ceiling=32,
            max_graphs=3,
        )


def test_edge_key_is_reserved() -> None:
    # Eight buckets exactly fill eight graphs, but the edge shape needs one more key.
    with pytest.raises(ConfigError, match=r"9.*max_graphs=8"):
        EngineConfig(
            chunk=ChunkMode(160),
            buckets=(1, 2, 3, 4, 5, 6, 7, 8),
            elastic_buckets=True,
            calibrated_ceiling=8,
            max_graphs=8,
        )
