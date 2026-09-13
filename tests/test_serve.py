# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""`ServeSettings` and `engine_config`, the layer under the CLI: what argparse refuses
is refused here too, so a programmatic caller gets the same rules, and the operator's
sizing choice becomes exactly the engine configuration it names."""

from __future__ import annotations

import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import Engine
from verbatim.pipelines.base import GraphCapability
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.scheduler.graph_budget import ConfigError
from verbatim.serve import ServeSettings, engine_config

pytestmark = pytest.mark.cpu

CHUNK = ChunkMode(160)
MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def test_exactly_one_of_ceiling_and_bucket_is_required() -> None:
    with pytest.raises(ConfigError, match="never assumed"):
        ServeSettings(model=MODEL, chunk=CHUNK)
    with pytest.raises(ConfigError, match="exactly one"):
        ServeSettings(model=MODEL, chunk=CHUNK, ceiling=32, bucket=8)


@pytest.mark.parametrize("field", ["ceiling", "bucket"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_sizing_must_be_a_positive_integer(field: str, value: object) -> None:
    with pytest.raises(ConfigError, match=f"--{field}"):
        ServeSettings(model=MODEL, chunk=CHUNK, **{field: value})  # type: ignore[arg-type]


def test_a_server_never_runs_without_an_idle_deadline() -> None:
    with pytest.raises(ConfigError, match="deadline"):
        ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, idle_timeout_s=0)


def test_ports_and_hosts_are_checked() -> None:
    with pytest.raises(ConfigError, match="both 7000"):
        ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, ws_port=7000, grpc_port=7000)
    with pytest.raises(ConfigError, match="--ws-port"):
        ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, ws_port=70000)
    with pytest.raises(ConfigError, match="--host"):
        ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, host="")
    with pytest.raises(ConfigError, match="--pipeline"):
        ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, pipeline="whisper")


def test_both_cache_aware_branches_and_the_fake_are_servable_and_nothing_else() -> None:
    for name in ("cache_aware_rnnt", "cache_aware_ctc", "fake"):
        settings = ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, pipeline=name)
        assert engine_config(settings).pipeline == name
    with pytest.raises(ConfigError, match="cache_aware_ctc"):
        ServeSettings(model=MODEL, chunk=CHUNK, bucket=8, pipeline="ctc")


def test_a_ceiling_is_the_calibrated_ceiling_and_the_single_bucket() -> None:
    config = engine_config(ServeSettings(model=MODEL, chunk=CHUNK, ceiling=24, idle_timeout_s=45))
    assert config.buckets == (24,)
    assert config.calibrated_ceiling == 24
    assert config.idle_timeout_s == 45
    assert config.pipeline == "cache_aware_rnnt"


def test_a_named_bucket_is_uncalibrated() -> None:
    settings = ServeSettings(
        model=MODEL, chunk=CHUNK, bucket=8, ring_seconds=1.5, stop_history_eou_ms=640
    )
    config = engine_config(settings)
    assert config.buckets == (8,)
    assert config.calibrated_ceiling is None
    assert settings.calibrated is False
    assert config.ring_seconds == 1.5
    assert config.stop_history_eou_ms == 640


def test_the_execution_label_comes_from_the_capture_controller_not_the_flag() -> None:
    """The label and the fact must not travel by different routes.

    `execution` is what a health endpoint reports, what a metrics label carries, and what
    a published row will be checked against. It used to be computed from `settings.eager`
    -- the operator's request -- while whether the encoder step actually runs graphed is
    decided by the capture controller from the adapter's reported capability, and then
    proved by warming up against the runtime's own retained-graph count.

    Through the CLI the two agreed and could not be made to disagree. They were still two
    answers to one question, and only one of them is checked. This pins the checked one.
    """
    config = EngineConfig(chunk=CHUNK, buckets=(8,), calibrated_ceiling=8)
    graphed = FakePipelineAdapter(CHUNK, buckets=(8,), graphs=GraphCapability.graphed())
    assert Engine(config, graphed).execution == "graph path"

    eager = FakePipelineAdapter(
        CHUNK, buckets=(8,), graphs=GraphCapability.eager("this fake runs no graphs")
    )
    assert Engine(config, eager).execution == "eager"
