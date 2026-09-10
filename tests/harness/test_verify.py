# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for `verbatim-bench verify`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from verbatim_bench.canonical import stamp_checksum
from verbatim_bench.cli import main
from verbatim_bench.client import ChunkMode
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.pace import LoadSpec, run_load
from verbatim_bench.results import percentile, write_results
from verbatim_bench.schema import validate
from verbatim_bench.verify import Level, verify_document, verify_file


def _session(
    session_id: str,
    stream_id: str,
    text: str,
    *,
    first: float | None = 10.0,
    partials: list[float] | None = None,
    final: float | None = 20.0,
    error: str | None = None,
    chunks: int = 2,
    partials_received: int | None = None,
) -> dict[str, Any]:
    partials = [10.0, 12.0] if partials is None else partials
    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "server_session_id": None if error else "null-1",
        "started_at_s": 0.0,
        "chunks": chunks,
        "audio_s": 0.32,
        "first_partial_ms": first,
        "final_ms": final,
        "partial_ms": list(partials),
        "partials_received": len(partials) if partials_received is None else partials_received,
        "final_text": text,
        "reference_text": text,
        "words": [] if error else [{"w": text, "s": 0, "e": 160}],
        "error": error,
    }


def _block(values: list[float]) -> dict[str, Any]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "n": len(values),
    }


def build_doc(sessions: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """Assemble a self-consistent document from `sessions` and stamp its checksum."""
    firsts = [s["first_partial_ms"] for s in sessions if s["first_partial_ms"] is not None]
    partials_all = [v for s in sessions for v in s["partial_ms"]]
    finals = [s["final_ms"] for s in sessions if s["final_ms"] is not None]
    completed = sum(1 for s in sessions if s["error"] is None)
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
            "sessions": len(sessions),
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
            "sessions_started": len(sessions),
            "sessions_completed": completed,
            "sessions_failed": len(sessions) - completed,
            "chunks_sent": sum(s["chunks"] for s in sessions),
            "partials_received": sum(len(s["partial_ms"]) for s in sessions),
            "finals_received": len(finals),
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


def default_sessions() -> list[dict[str, Any]]:
    return [
        _session("s0000", "utt-1", "hello world"),
        _session("s0001", "utt-2", "goodbye world", first=11.0, partials=[11.0], final=22.0),
    ]


def make_doc(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid document: verifies cleanly."""
    return build_doc(default_sessions(), **overrides)


def _codes(report: Any) -> list[str]:
    return [finding.code for finding in report.findings]


def _write(path: Path, doc: dict[str, Any]) -> Path:
    target = path / "results.json" if path.is_dir() or path.suffix != ".json" else path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return target


def test_valid_document_verifies() -> None:
    report = verify_document(make_doc())
    assert report.ok
    assert report.errors == ()


def test_percentile_mismatch_is_an_error() -> None:
    doc = make_doc()
    doc["result"]["latency_ms"]["partial"]["p95"] += 5.0
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "PERCENTILE_MISMATCH" in _codes(report)
    assert any(
        finding.code == "PERCENTILE_MISMATCH" and finding.path == "result.latency_ms.partial.p95"
        for finding in report.findings
    )


def test_percentile_within_tolerance_passes() -> None:
    doc = make_doc()
    doc["result"]["latency_ms"]["partial"]["p95"] -= 0.05
    stamp_checksum(doc)
    assert verify_document(doc).ok


def test_n_mismatch_is_an_error() -> None:
    doc = make_doc()
    doc["result"]["latency_ms"]["partial"]["n"] = 999
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "N_MISMATCH" in _codes(report)


def test_counter_mismatch_is_an_error() -> None:
    doc = make_doc()
    doc["result"]["sessions_completed"] += 1
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "COUNTER_MISMATCH" in _codes(report)


def test_out_of_order_percentiles_are_an_error() -> None:
    doc = make_doc()
    block = doc["result"]["latency_ms"]["first_partial"]
    block["p50"], block["p95"] = block["p95"], block["p50"]
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "ORDER" in _codes(report)


def test_bad_corpus_id_is_an_error() -> None:
    doc = make_doc()
    doc["config"]["corpus_id"] = "deadbeef"
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "CORPUS_ID" in _codes(report)


def test_bad_date_is_an_error() -> None:
    doc = make_doc()
    doc["date"] = "not-a-date"
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "DATE" in _codes(report)


def test_invariance_equal_true_with_differing_hashes_is_an_error() -> None:
    doc = make_doc()
    doc["result"]["invariance"] = {
        "hash_1": "ab" * 32,
        "hash_32a": "cd" * 32,
        "hash_32b": "cd" * 32,
        "hash_max": "cd" * 32,
        "equal": True,
    }
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "INVARIANCE_INCONSISTENT" in _codes(report)
    message = next(
        finding.message for finding in report.findings if finding.code == "INVARIANCE_INCONSISTENT"
    )
    assert "hash_1" in message and "hash_32a" in message


def test_invariance_equal_false_with_identical_hashes_is_an_error() -> None:
    doc = make_doc()
    doc["result"]["invariance"] = {
        "hash_1": "ab" * 32,
        "hash_32a": "ab" * 32,
        "hash_32b": "ab" * 32,
        "hash_max": "ab" * 32,
        "equal": False,
    }
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "INVARIANCE_INCONSISTENT" in _codes(report)


def test_invariance_absent_is_fine() -> None:
    doc = make_doc()
    del doc["result"]["invariance"]
    stamp_checksum(doc)
    assert verify_document(doc).ok


def test_streams_must_be_min_of_three() -> None:
    doc = make_doc()
    doc["result"]["streams_runs"] = [100, 90, 95]
    doc["result"]["streams"] = 100
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "NOT_MIN_OF_THREE" in _codes(report)
    doc["result"]["streams"] = 90
    stamp_checksum(doc)
    assert verify_document(doc).ok


def test_missing_checksum_is_a_warning_not_an_error() -> None:
    doc = make_doc()
    del doc["checksum"]
    report = verify_document(doc)
    assert report.ok
    assert any(
        finding.code == "CHECKSUM" and finding.level is Level.WARNING for finding in report.findings
    )


def test_bad_checksum_is_an_error() -> None:
    doc = make_doc()
    doc["result"]["wall_clock_s"] = 999.0
    report = verify_document(doc)
    assert not report.ok
    assert "CHECKSUM" in _codes(report)


def test_strict_promotes_warnings(tmp_path: Path) -> None:
    doc = make_doc()
    del doc["checksum"]
    assert verify_document(doc).ok
    target = _write(tmp_path / "results.json", doc)
    assert main(["verify", str(target)]) == 0
    assert main(["verify", str(target), "--strict"]) == 2


def test_errored_session_does_not_break_verification() -> None:
    failed = _session(
        "s0002",
        "utt-3",
        "",
        first=None,
        partials=[],
        final=None,
        error="TimeoutError: no final received after end",
        chunks=0,
    )
    doc = build_doc([*default_sessions(), failed])
    assert doc["result"]["sessions_started"] == 3
    assert doc["result"]["sessions_completed"] == 2
    assert doc["result"]["sessions_failed"] == 1
    report = verify_document(doc)
    assert report.ok, [f"{f.code} {f.path}: {f.message}" for f in report.findings]


def test_verify_file_accepts_a_directory(tmp_path: Path) -> None:
    _write(tmp_path / "results.json", make_doc())
    assert verify_file(tmp_path).ok


def test_cli_exit_codes(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.json", make_doc())
    assert main(["verify", str(good)]) == 0
    bad_doc = make_doc()
    bad_doc["result"]["sessions_completed"] = 999
    bad = _write(tmp_path / "bad.json", bad_doc)
    assert main(["verify", str(bad)]) == 2
    assert main(["verify", str(tmp_path / "missing.json")]) == 1


def test_cli_json_output_is_parseable(tmp_path: Path, capsys: Any) -> None:
    target = _write(tmp_path / "results.json", make_doc())
    assert main(["verify", str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload["findings"], list)
    assert isinstance(payload["ok"], bool) and payload["ok"] is True


def test_verify_needs_no_optional_dependency() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import verbatim_bench.verify; "
            "assert 'torch' not in sys.modules, 'torch leaked'; "
            "assert 'numpy' not in sys.modules, 'numpy leaked'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_partials_received_reconciles_against_per_session_counts() -> None:
    # One partial message per utterance, seven sent chunks with samples: the
    # honest artifact reports partials_received=1, not len(partial_ms)=7.
    session = _session(
        "s0000", "utt-1", "hello", partials=[float(v) for v in range(7)], partials_received=1
    )
    doc = build_doc([session])
    doc["result"]["partials_received"] = 1
    stamp_checksum(doc)
    report = verify_document(doc)
    assert report.ok, [f"{f.code} {f.path}: {f.message}" for f in report.findings]


def test_partials_received_check_skipped_when_sessions_lack_the_field() -> None:
    doc = make_doc()
    for session in doc["sessions"]:
        del session["partials_received"]
    doc["result"]["partials_received"] = 999
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not any(
        finding.code == "COUNTER_MISMATCH" and finding.path == "result.partials_received"
        for finding in report.findings
    )


@pytest.mark.parametrize("runs", [[None, None, None], [1, "a", 2], [1, True, 3]])
def test_malformed_streams_runs_yields_findings_not_traceback(runs: Any) -> None:
    doc = make_doc()
    doc["result"]["streams_runs"] = runs
    doc["result"]["streams"] = 1
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "NOT_MIN_OF_THREE" in _codes(report)


def test_cli_malformed_streams_runs_exits_2(tmp_path: Path) -> None:
    doc = make_doc()
    doc["result"]["streams_runs"] = [None, None, None]
    doc["result"]["streams"] = 1
    target = _write(tmp_path / "malformed.json", doc)
    assert main(["verify", str(target)]) == 2


def test_two_malformed_hashes_yield_two_findings() -> None:
    doc = make_doc()
    doc["result"]["invariance"] = {
        "hash_1": "zzz",
        "hash_32a": "not-hex-either",
        "hash_32b": None,
        "hash_max": None,
        "equal": True,
    }
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert sum(1 for finding in report.findings if finding.code == "INVARIANCE_INCONSISTENT") == 2


def test_uppercase_hash_is_a_finding() -> None:
    doc = make_doc()
    doc["result"]["invariance"] = {
        "hash_1": "AB" * 32,
        "hash_32a": None,
        "hash_32b": None,
        "hash_max": None,
        "equal": True,
    }
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "INVARIANCE_INCONSISTENT" in _codes(report)


def test_single_hash_with_equal_null_is_valid() -> None:
    doc = make_doc()
    doc["result"]["invariance"] = {
        "hash_1": "ab" * 32,
        "hash_32a": None,
        "hash_32b": None,
        "hash_max": None,
        "equal": None,
    }
    stamp_checksum(doc)
    report = verify_document(doc)
    assert "INVARIANCE_INCONSISTENT" not in _codes(report)
    assert report.ok


async def test_writer_output_round_trips_through_validate_and_verify(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "utt.wav"
    samples = np.sin(2 * np.pi * 440.0 * np.arange(8000) / 16000).astype(np.float32)
    sf.write(str(wav), 0.5 * samples, 16000)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": wav.name,
                "duration": 0.5,
                "text": "reference for utt-0",
                "stream_id": "utt-0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    async with NullServer(NullServerConfig()) as server:
        spec = LoadSpec(
            endpoint=server.endpoint,
            manifest=manifest,
            sessions=2,
            chunk=ChunkMode.parse(160),
            seed=11,
            ramp_s=0.0,
        )
        result = await run_load(spec)
    out = write_results(result, tmp_path / "out")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert validate(doc) == []
    report = verify_document(doc)
    assert report.ok, [f"{f.code} {f.path}: {f.message}" for f in report.findings]
