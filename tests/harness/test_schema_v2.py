# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for vb-results/2 validation and verification."""

from __future__ import annotations

from typing import Any

import pytest
from verbatim_bench import constants
from verbatim_bench.canonical import stamp_checksum
from verbatim_bench.results import percentile
from verbatim_bench.schema import SCHEMA_ID_V2, load_schema, validate
from verbatim_bench.verify import Level, verify_document, verify_table

pytestmark = pytest.mark.cpu

HEX_A = "ab" * 32
HEX_C = "cd" * 32


def _block(values: list[float]) -> dict[str, Any]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "n": len(values),
    }


def _session(
    session_id: str = "s0000",
    stream_id: str = "utt-1",
    text: str = "hello world",
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "server_session_id": "null-1",
        "started_at_s": 0.0,
        "chunks": 2,
        "audio_s": 0.32,
        "first_partial_ms": 10.0,
        "final_ms": 20.0,
        "partial_ms": [10.0, 12.0],
        "partials_received": 2,
        "final_text": text,
        "reference_text": text,
        "words": [{"w": text, "s": 0, "e": 160}],
        "error": None,
        "pacing_slip_samples": [0.1, 0.2],
    }


def _ladder_rung(
    n: int, seed: int, *, passed: bool, criterion: str | None = None
) -> dict[str, Any]:
    return {
        "n": n,
        "seed": seed,
        "p95_ms": 10.0 if passed else 5000.0,
        "wer_vs_batch1": None,
        "passed": passed,
        "first_failing_criterion": criterion if not passed else None,
        "valid": True,
        "invalid_reason": None,
        "warm_up_s": float(constants.WARM_UP_S),
        "sessions_refused": 0,
        "sessions_dropped": 0,
        "sessions_without_final": 0,
        "canonical_window": True,
    }


def make_v2_doc(**overrides: Any) -> dict[str, Any]:
    sessions = [_session("s0000", "utt-1"), _session("s0001", "utt-2", "goodbye world")]
    firsts = [10.0, 10.0]
    partials_all = [10.0, 12.0, 10.0, 12.0]
    finals = [20.0, 20.0]
    ladder = [
        _ladder_rung(8, constants.SEEDS[0], passed=True),
        _ladder_rung(10, constants.SEEDS[0], passed=False, criterion="latency"),
    ]
    doc: dict[str, Any] = {
        "schema": "vb-results/2",
        "harness": {
            "version": "0.0.1.dev0",
            "git": "abc123",
            "client_floor_p95_ms": None,
            "tag": "harness-1",
            "tree_clean": True,
            "container_digest": None,
        },
        "arm": {
            "name": "test",
            "version": None,
            "container_digest": None,
            "surface": "websocket",
            "arm_id": "a",
            "status": "measured",
            "reason": None,
            "panel": "main",
            "dtype_class": "fp32_tf32",
            "kill_rule": 1,
        },
        "checkpoint": {
            "hf_id": None,
            "revision": None,
            "file_sha256": None,
            "converted_sha256": None,
            "converter_version": None,
        },
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
            "frame_ms": 20,
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
            "session_profile": "m180",
            "canonical_window": True,
        },
        "result": {
            "streams": 8,
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
            "partials_received": 4,
            "finals_received": 2,
            "wall_clock_s": 1.5,
            "pacing_slip_ms": {"p50": 0.1, "p95": 0.2, "max": 0.2},
            "invariance": {
                "hash_1": HEX_A,
                "hash_32a": HEX_A,
                "hash_32b": HEX_A,
                "hash_max": HEX_A,
                "equal": True,
            },
            "eager_step_fraction": None,
            "fraction_of_ceiling": None,
        },
        "sessions": sessions,
        "date": "2026-09-08T00:00:00+00:00",
        "box": {
            "boot_id": "boot-1",
            "gpu_uuid": "GPU-1",
            "cgroup_cpu_max": "max 100000",
            "cpuset_effective": "0-3",
            "container_digest": None,
            "id": "box-123",
        },
        "gpu": {
            "uuid": "GPU-1",
            "name": constants.VERDICT_DIE,
            "sm": constants.VERDICT_SM,
            "vram_total_gb": 48.0,
            "driver": "550.144.03",
            "nvml_version": "12.550.144.03",
            "cuda_driver_version": "12.4",
            "vbios": "94.02",
            "pci_bus_id": "00000000:01:00.0",
            "serial": None,
            "pcie_gen_current": 4,
            "pcie_gen_max": 4,
            "pcie_width_current": 16,
            "pcie_width_max": 16,
            "persistence_mode": True,
            "compute_mode": "Default",
            "mig_mode": "Disabled",
            "ecc_current": True,
            "ecc_pending": True,
            "power_limit_w": 300.0,
            "power_limit_default_w": 300.0,
            "power_limit_min_w": 100.0,
            "power_limit_max_w": 300.0,
            "clock_sm_mhz": 1500,
            "clock_mem_mhz": 1200,
            "clock_sm_max_mhz": 1800,
            "clock_sm_app_mhz": None,
            "temperature_c": 45.0,
            "temperature_slowdown_c": 85.0,
            "temperature_shutdown_c": 92.0,
            "throttle_reasons": {},
            "compute_process_pids": [],
            "foreign_processes_count": 0,
            "energy_counter_available": True,
            "util_pct": 5.0,
        },
        "sw": {"python": "3.11.0"},
        "host": {"client_cpu_pct_of_cpuset": 0.1},
        "cal": {"null_floor_p95_ms": 5.0, "null_floor_client_cpu_pct": 1.0},
        "sensitivity": {},
        "ladder": ladder,
        "ceiling": {},
        "declared": {},
    }
    doc.update(overrides)
    return stamp_checksum(doc)


def _codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def test_v1_documents_still_validate_and_verify_unchanged() -> None:
    # Same-directory import, not `tests.harness.test_verify`. There is no `tests` package: the
    # dotted form only resolved because `python -m pytest` puts the working directory on sys.path,
    # and CI invokes `pytest` directly, where it does not.
    from test_verify import make_doc as make_v1_doc

    doc = make_v1_doc()
    assert validate(doc) == []
    assert verify_document(doc).ok


def test_v2_minimal_document_validates() -> None:
    schema = load_schema(SCHEMA_ID_V2)
    assert schema["properties"]["schema"] == {"const": "vb-results/2"}
    assert validate(make_v2_doc(), schema) == []


def test_v2_rejects_a_notes_key_at_any_depth() -> None:
    doc = make_v2_doc()
    doc["result"]["notes"] = "post-hoc narrative"
    assert validate(doc, load_schema(SCHEMA_ID_V2))
    doc2 = make_v2_doc()
    doc2["sessions"][0]["notes"] = "x"
    assert validate(doc2, load_schema(SCHEMA_ID_V2))
    doc3 = make_v2_doc()
    doc3["notes"] = "x"
    assert validate(doc3, load_schema(SCHEMA_ID_V2))


def test_v2_requires_a_reason_when_arm_status_is_not_measured() -> None:
    doc = make_v2_doc()
    doc["arm"]["status"] = "not_run"
    doc["arm"]["reason"] = None
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "ARM_STATUS_MISSING_REASON" in _codes(report)


def test_v2_pacing_profile_enum_is_uniform_burst_diurnal() -> None:
    for profile in ("uniform", "burst", "diurnal"):
        doc = make_v2_doc()
        doc["config"]["pacing_profile"] = profile
        assert validate(doc, load_schema(SCHEMA_ID_V2)) == []
    doc = make_v2_doc()
    doc["config"]["pacing_profile"] = "bursty"
    assert validate(doc, load_schema(SCHEMA_ID_V2))


def test_verify_flags_a_null_gpu_block() -> None:
    doc = make_v2_doc()
    doc["gpu"] = None
    stamp_checksum(doc)
    assert "ENV_INCOMPLETE" in _codes(verify_document(doc))


def test_verify_flags_a_die_and_sm_that_disagree() -> None:
    doc = make_v2_doc()
    doc["gpu"]["name"] = constants.VERDICT_DIE
    doc["gpu"]["sm"] = "9.0"
    stamp_checksum(doc)
    assert "SM_DIE_MISMATCH" in _codes(verify_document(doc))


def test_verify_flags_a_nonzero_throttle_count() -> None:
    doc = make_v2_doc()
    doc["gpu"]["throttle_reasons"] = {"clocks_throttle_reason_sw_thermal": 3}
    stamp_checksum(doc)
    assert "THROTTLE_NONZERO" in _codes(verify_document(doc))


def test_verify_flags_client_cpu_over_budget() -> None:
    doc = make_v2_doc()
    doc["host"]["client_cpu_pct_of_cpuset"] = 0.9
    stamp_checksum(doc)
    assert "CLIENT_CPU_OVER_BUDGET" in _codes(verify_document(doc))


def test_verify_flags_a_missing_null_floor() -> None:
    doc = make_v2_doc()
    doc["cal"]["null_floor_p95_ms"] = None
    stamp_checksum(doc)
    assert "FLOOR_MISSING" in _codes(verify_document(doc))


def test_verify_flags_a_non_monotone_ladder() -> None:
    doc = make_v2_doc()
    doc["ladder"] = [
        _ladder_rung(8, constants.SEEDS[0], passed=False, criterion="latency"),
        _ladder_rung(10, constants.SEEDS[0], passed=True),
    ]
    doc["result"]["streams"] = 10
    stamp_checksum(doc)
    assert "LADDER_NOT_MONOTONE" in _codes(verify_document(doc))


def test_verify_flags_s_that_is_not_the_highest_passing_rung() -> None:
    doc = make_v2_doc()
    doc["result"]["streams"] = 4
    stamp_checksum(doc)
    assert "S_NOT_HIGHEST_PASSING" in _codes(verify_document(doc))


def test_verify_recomputes_the_fraction_of_ceiling() -> None:
    doc = make_v2_doc()
    doc["result"]["streams_runs"] = [90, 95, 100]
    doc["ceiling"] = {
        "c_runs": [100.0, 100.0, 100.0, 100.0, 100.0],
        "box_id": "box-123",
        "dtype_class": "fp32_tf32",
    }
    doc["result"]["fraction_of_ceiling"] = 0.5
    stamp_checksum(doc)
    assert "F_MISMATCH" in _codes(verify_document(doc))


def test_verify_flags_a_ratio_from_two_box_sessions() -> None:
    doc = make_v2_doc()
    doc["result"]["streams_runs"] = [90, 90, 90]
    doc["ceiling"] = {
        "c_runs": [100.0, 100.0, 100.0, 100.0, 100.0],
        "box_id": "box-other",
        "dtype_class": "fp32_tf32",
    }
    doc["result"]["fraction_of_ceiling"] = 0.9
    stamp_checksum(doc)
    assert "CROSS_BOX_RATIO" in _codes(verify_document(doc))


def test_verify_flags_a_ceiling_with_fewer_than_four_input_streams_per_slot() -> None:
    doc = make_v2_doc()
    doc["ceiling"] = {"input_streams_per_slot": 2}
    stamp_checksum(doc)
    assert "CEILING_UNDERFILLED" in _codes(verify_document(doc))


def test_verify_flags_repeated_seeds_in_streams_runs() -> None:
    doc = make_v2_doc()
    doc["ladder"] = [
        _ladder_rung(8, 111, passed=True),
        _ladder_rung(8, 111, passed=True),
    ]
    stamp_checksum(doc)
    assert "SEEDS_NOT_DISTINCT" in _codes(verify_document(doc))


def test_verify_flags_a_table_mixing_harness_tags() -> None:
    first = make_v2_doc()
    second = make_v2_doc()
    second["harness"]["tag"] = "harness-2"
    stamp_checksum(second)
    report = verify_table([first, second])
    assert "HARNESS_TAG_MIXED" in _codes(report)


def test_verify_rejects_equal_true_with_fewer_than_four_hashes() -> None:
    doc = make_v2_doc()
    doc["result"]["invariance"] = {
        "hash_1": HEX_A,
        "hash_32a": None,
        "hash_32b": None,
        "hash_max": None,
        "equal": True,
    }
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "INVARIANCE_UNDER_FOUR_HASHES" in _codes(report)


def test_verify_warns_when_no_arm_ever_failed_on_latency() -> None:
    doc = make_v2_doc()
    doc["ladder"] = [
        _ladder_rung(8, constants.SEEDS[0], passed=True),
        _ladder_rung(10, constants.SEEDS[0], passed=False, criterion="wer"),
    ]
    stamp_checksum(doc)
    report = verify_document(doc)
    warnings = [f for f in report.findings if f.level is Level.WARNING]
    assert any(f.code == "NO_LATENCY_CRITERION_ANYWHERE" for f in warnings)


def test_verify_rejects_a_non_canonical_window_row() -> None:
    doc = make_v2_doc()
    doc["config"]["canonical_window"] = False
    stamp_checksum(doc)
    assert "NON_CANONICAL_WINDOW" in _codes(verify_document(doc))


def test_v2_document_with_all_four_hashes_and_equal_true_passes() -> None:
    report = verify_document(make_v2_doc())
    assert "INVARIANCE_UNDER_FOUR_HASHES" not in _codes(report)
    assert report.ok
