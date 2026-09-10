# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the vb-results/1 JSON Schema and its validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verbatim_bench.canonical import stamp_checksum
from verbatim_bench.results import percentile
from verbatim_bench.schema import SCHEMA_ID, load_schema, validate


def _session(
    session_id: str,
    stream_id: str,
    text: str,
    *,
    first: float | None = 10.0,
    partials: list[float] | None = None,
    final: float | None = 20.0,
) -> dict[str, Any]:
    partials = [10.0, 12.0] if partials is None else partials
    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "server_session_id": "null-1",
        "started_at_s": 0.0,
        "chunks": 2,
        "audio_s": 0.32,
        "first_partial_ms": first,
        "final_ms": final,
        "partial_ms": list(partials),
        "final_text": text,
        "reference_text": text,
        "words": [{"w": text, "s": 0, "e": 160}],
        "error": None,
    }


def _block(values: list[float]) -> dict[str, Any]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "n": len(values),
    }


def make_doc(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid vb-results/1 document, stamped with a checksum."""
    sessions = [
        _session("s0000", "utt-1", "hello world"),
        _session("s0001", "utt-2", "goodbye world", first=11.0, partials=[11.0], final=22.0),
    ]
    firsts = [s["first_partial_ms"] for s in sessions if s["first_partial_ms"] is not None]
    partials_all = [v for s in sessions for v in s["partial_ms"]]
    finals = [s["final_ms"] for s in sessions if s["final_ms"] is not None]
    doc: dict[str, Any] = {
        "schema": "vb-results/1",
        "harness": {"version": "0.0.1.dev0", "git": None, "client_floor_p95_ms": None},
        "arm": {"name": "test", "version": None, "container_digest": None, "surface": "websocket"},
        "checkpoint": {"hf_id": None, "revision": None, "file_sha256": None},
        "env": {
            "die": None,
            "sm": None,
            "vram_gb": None,
            "driver": None,
            "cuda": None,
            "nemo": None,
            "torch": None,
            "cpu_sets": None,
        },
        "config": {
            "chunk_ms": 160,
            "frame_ms": 160,
            "corpus_id": "sha256:" + "ab" * 32,
            "manifest": "m.jsonl",
            "seed": 1,
            "sessions": 2,
            "pacing_profile": "uniform",
            "paced_real_time": True,
            "x_ms": 150,
            "endpoint": "ws://127.0.0.1:1/v1/stream",
            "words": False,
            "compute_dtype": None,
            "att_context": None,
        },
        "result": {
            "streams": None,
            "streams_runs": None,
            "latency_ms": {
                "first_partial": _block(firsts),
                "partial": _block(partials_all),
                "final": _block(finals),
            },
            "sessions_started": 2,
            "sessions_completed": 2,
            "sessions_failed": 0,
            "chunks_sent": 4,
            "partials_received": 3,
            "finals_received": 2,
            "wall_clock_s": 1.5,
            "pacing_slip_ms": {"p50": 0.5, "p95": 1.0, "max": 1.5},
            "invariance": {
                "hash_1": None,
                "hash_32a": None,
                "hash_32b": None,
                "hash_max": None,
                "equal": None,
            },
            "eager_step_fraction": None,
            "fraction_of_ceiling": None,
        },
        "sessions": sessions,
        "date": "2026-09-08T00:00:00+00:00",
    }
    doc.update(overrides)
    return stamp_checksum(doc)


def test_minimal_valid_document_passes() -> None:
    assert validate(make_doc()) == []


def test_missing_required_field_is_reported_with_a_path() -> None:
    doc = make_doc()
    del doc["config"]["chunk_ms"]
    errors = validate(doc)
    assert errors
    assert any(error.path == "config.chunk_ms" for error in errors)


def test_wrong_type_is_reported() -> None:
    doc = make_doc()
    doc["config"]["chunk_ms"] = "160"
    errors = validate(doc)
    assert errors
    assert any(error.path == "config.chunk_ms" for error in errors)


def test_null_is_allowed_where_the_schema_says_so() -> None:
    doc = make_doc()
    doc["result"]["streams"] = None
    assert validate(doc) == []
    doc["config"]["chunk_ms"] = None
    errors = validate(doc)
    assert any(error.path == "config.chunk_ms" for error in errors)


def test_unknown_top_level_key_is_reported() -> None:
    doc = make_doc()
    doc["surprise"] = 1
    errors = validate(doc)
    assert any("surprise" in error.path for error in errors)


def test_enum_is_enforced() -> None:
    doc = make_doc()
    doc["arm"]["surface"] = "carrier-pigeon"
    errors = validate(doc)
    assert errors
    assert any(error.path == "arm.surface" for error in errors)
    assert any("websocket" in error.message for error in errors)


def test_pattern_is_enforced() -> None:
    doc = make_doc()
    doc["config"]["corpus_id"] = "deadbeef"
    errors = validate(doc)
    assert any(error.path == "config.corpus_id" for error in errors)


def test_all_errors_are_returned_not_just_the_first() -> None:
    doc = make_doc()
    del doc["config"]["chunk_ms"]
    doc["arm"]["surface"] = "carrier-pigeon"
    doc["surprise"] = 1
    assert len(validate(doc)) >= 3


def test_schema_file_is_valid_json_and_declares_its_id() -> None:
    assert SCHEMA_ID == "vb-results/1"
    schema = load_schema()
    assert isinstance(schema, dict)
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "benchmarks" / "schema" / "row.schema.json"
        if candidate.is_file():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            break
    else:
        raise AssertionError("benchmarks/schema/row.schema.json not found")
    assert raw == schema
    assert "vb-results/1" in str(schema.get("$id", "")) + str(schema.get("title", ""))
    assert schema.get("$id", "").startswith("urn:")


def test_per_session_partials_received_is_optional_but_nonnegative() -> None:
    doc = make_doc()
    assert validate(doc) == []
    doc["sessions"][0]["partials_received"] = 1
    assert validate(doc) == []
    doc["sessions"][0]["partials_received"] = -1
    errors = validate(doc)
    assert any(error.path == "sessions[0].partials_received" for error in errors)
