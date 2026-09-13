# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""A row must observe the server, not quote the operator."""

from __future__ import annotations

import json

import pytest
from verbatim_bench.serverfacts import (
    ArmContradiction,
    ServerFacts,
    check_arm,
    health_url,
    read_server_facts,
)

pytestmark = pytest.mark.cpu

ENDPOINT = "ws://127.0.0.1:8765/v1/stream"


def _facts(**overrides: object) -> ServerFacts:
    args: dict = {
        "ready": True,
        "model": "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
        "pipeline": "cache_aware_rnnt",
        "chunk_ms": 160,
        "precision": "bfloat16",
        "execution": "eager",
        "tick_id": 4211,
    }
    args.update(overrides)
    return ServerFacts(**args)  # type: ignore[arg-type]


def test_the_health_url_sits_beside_the_stream_endpoint() -> None:
    """Both listeners share a port, so the harness does not need a second flag."""
    assert health_url(ENDPOINT) == "http://127.0.0.1:8765/readyz"
    assert health_url("wss://host.example:443/v1/stream") == "https://host.example:443/readyz"


def test_a_server_that_does_not_answer_is_no_reading_rather_than_a_disagreement() -> None:
    """None is not the same as a contradiction: a server too old to carry these fields
    is a different situation from one contradicting its own label, and only the second
    should stop a run."""

    def refuse(_url: str) -> str:
        raise OSError("connection refused")

    assert read_server_facts(ENDPOINT, fetch=refuse) is None
    check_arm(None, arm="anything-at-all", declared_dtype="float32")


def test_the_reading_comes_back_as_the_server_reported_it() -> None:
    body = json.dumps(
        {
            "ready": True,
            "model": "m",
            "pipeline": "cache_aware_ctc",
            "chunk_ms": 560,
            "precision": "float32",
            "execution": "graph path",
            "tick_id": 9,
        }
    )
    facts = read_server_facts(ENDPOINT, fetch=lambda _url: body)
    assert facts is not None
    assert (facts.pipeline, facts.chunk_ms, facts.precision, facts.execution) == (
        "cache_aware_ctc",
        560,
        "float32",
        "graph path",
    )


def test_a_declared_precision_the_server_is_not_running_is_refused() -> None:
    """The exact comparison. --dtype is what a row's WER reference is matched on, so a
    run declaring float32 against a bfloat16 server would attach real numbers to the
    wrong precision."""
    with pytest.raises(ArmContradiction, match="precision"):
        check_arm(_facts(precision="bfloat16"), arm="run-4", declared_dtype="float32")


def test_a_declared_chunk_mode_the_server_is_not_running_is_refused() -> None:
    with pytest.raises(ArmContradiction, match="160 ms"):
        check_arm(_facts(chunk_ms=160), arm="run-4", declared_chunk_ms=560)


def test_an_arm_naming_graphs_against_an_eager_server_is_refused() -> None:
    """The whole reason this module exists. Every B300 arm this project has run is named
    in exactly this style, and until now nothing checked the name against the server."""
    with pytest.raises(ArmContradiction, match="graph"):
        check_arm(_facts(execution="eager"), arm="verbatim-b300-bf16-graphed")


def test_an_arm_naming_eager_against_a_graphed_server_is_refused() -> None:
    with pytest.raises(ArmContradiction, match="eager"):
        check_arm(_facts(execution="graph path"), arm="b300-eager-control")


def test_an_arm_that_names_no_mode_is_left_alone() -> None:
    """This refuses a contradiction; it does not impose a naming convention. An arm
    called "run-4" says nothing about execution and must not be second-guessed."""
    check_arm(_facts(execution="graph path"), arm="run-4", declared_dtype="bfloat16")
    check_arm(_facts(execution="eager"), arm="2026-09-14-sweep", declared_chunk_ms=160)


def test_an_arm_that_agrees_with_the_server_passes() -> None:
    """The positive control: without it every test above would pass on a function that
    refused everything."""
    check_arm(
        _facts(execution="graph path", precision="bfloat16", chunk_ms=160),
        arm="verbatim-b300-bf16-graphed",
        declared_dtype="bfloat16",
        declared_chunk_ms=160,
    )
