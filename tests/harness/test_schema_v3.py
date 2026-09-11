# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for vb-results/3: a rung records which criteria it evaluated.

The version exists for one field. `criteria_evaluated` lists what a rung actually
established, `passed` is a claim only about criteria that appear there, and
`first_failing_criterion` still names a criterion that was evaluated and failed rather
than one nobody measured. These tests hold the schema, the validator and the verifier to
that, and check that `vb-results/2` documents are verified exactly as before.
"""

from __future__ import annotations

from typing import Any

import pytest
from verbatim_bench import constants
from verbatim_bench.canonical import stamp_checksum
from verbatim_bench.ladder import RUNG_PASS_CRITERIA
from verbatim_bench.schema import SCHEMA_ID_V2, SCHEMA_ID_V3, load_schema, validate
from verbatim_bench.verify import verify_document

pytestmark = pytest.mark.cpu

EVERY_CRITERION = sorted(criterion.value for criterion in RUNG_PASS_CRITERIA)
PARTIAL_CRITERIA = ["latency", "integrity:refused"]


def _rung(
    n: int,
    seed: int,
    *,
    passed: bool,
    criterion: str | None = None,
    criteria_evaluated: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "n": n,
        "seed": seed,
        "p95_ms": 10.0 if passed else 5000.0,
        "wer_vs_batch1": None,
        "passed": passed,
        "first_failing_criterion": criterion,
        "criteria_evaluated": (
            list(EVERY_CRITERION) if criteria_evaluated is None else list(criteria_evaluated)
        ),
        "valid": True,
        "invalid_reason": None,
        "warm_up_s": float(constants.WARM_UP_S),
        "sessions_refused": 0,
        "sessions_dropped": 0,
        "sessions_without_final": 0,
        "canonical_window": True,
    }


def make_v3_doc(**overrides: Any) -> dict[str, Any]:
    """A v3 row: the v2 fixture, restamped, with rungs that say what they evaluated."""
    # Same-directory import, not `tests.harness.test_schema_v2`; see the note in that file.
    from test_schema_v2 import make_v2_doc

    doc = make_v2_doc()
    doc["schema"] = SCHEMA_ID_V3
    doc["ladder"] = [
        _rung(8, constants.SEEDS[0], passed=True),
        _rung(10, constants.SEEDS[0], passed=False, criterion="latency"),
    ]
    doc.update(overrides)
    return stamp_checksum(doc)


def _codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def test_the_frozen_run_version_is_the_schema_this_repository_carries() -> None:
    assert constants.SCHEMA_VERSION_FOR_RUN == SCHEMA_ID_V3
    assert load_schema(SCHEMA_ID_V3)["properties"]["schema"] == {"const": "vb-results/3"}


def test_v3_minimal_document_validates_and_verifies() -> None:
    doc = make_v3_doc()
    assert validate(doc, load_schema(SCHEMA_ID_V3)) == []
    assert verify_document(doc).ok


def test_v2_cannot_carry_the_new_field_and_v3_can() -> None:
    from test_schema_v2 import make_v2_doc

    doc = make_v2_doc()
    doc["ladder"][0]["criteria_evaluated"] = PARTIAL_CRITERIA
    # This is why the version bumped: the v2 rung sets additionalProperties false.
    assert validate(doc, load_schema(SCHEMA_ID_V2))
    assert validate(make_v3_doc(), load_schema(SCHEMA_ID_V3)) == []


def test_v3_rejects_a_criterion_outside_the_vocabulary() -> None:
    doc = make_v3_doc()
    doc["ladder"][0]["criteria_evaluated"] = ["latency", "vibes"]
    assert validate(doc, load_schema(SCHEMA_ID_V3))


def test_v2_documents_are_still_verified_against_v2() -> None:
    from test_schema_v2 import make_v2_doc

    doc = make_v2_doc()
    # The v2 rungs carry no criteria_evaluated and are not asked for one.
    assert "criteria_evaluated" not in doc["ladder"][0]
    assert verify_document(doc).ok


def test_verify_rejects_a_v3_rung_that_passed_without_evaluating_every_criterion() -> None:
    doc = make_v3_doc()
    doc["ladder"][0]["criteria_evaluated"] = PARTIAL_CRITERIA
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "PASSED_WITHOUT_EVALUATING_EVERY_CRITERION" in _codes(report)


def test_verify_rejects_a_v3_rung_that_fails_on_a_criterion_it_never_evaluated() -> None:
    """The alternative DR-0004 rejected, caught at verification.

    A rung with good latency and no refusals that reports a word error rate failure is
    replacing one false claim with another: the word error rate was never computed.
    """
    doc = make_v3_doc()
    doc["ladder"] = [
        _rung(
            8,
            constants.SEEDS[0],
            passed=False,
            criterion="wer",
            criteria_evaluated=PARTIAL_CRITERIA,
        )
    ]
    doc["result"]["streams"] = 0
    stamp_checksum(doc)
    report = verify_document(doc)
    assert not report.ok
    assert "CRITERION_NOT_EVALUATED" in _codes(report)


def test_verify_accepts_a_v3_rung_that_did_not_pass_because_evaluation_was_partial() -> None:
    doc = make_v3_doc()
    doc["ladder"] = [
        _rung(8, constants.SEEDS[0], passed=False, criteria_evaluated=PARTIAL_CRITERIA)
    ]
    doc["result"]["streams"] = 0
    stamp_checksum(doc)
    report = verify_document(doc)
    assert "CRITERION_MISSING" not in _codes(report)
    assert "CRITERION_NOT_EVALUATED" not in _codes(report)
    assert report.ok


def test_verify_still_flags_a_rung_that_evaluated_everything_and_failed_unlabelled() -> None:
    doc = make_v3_doc()
    doc["ladder"] = [_rung(8, constants.SEEDS[0], passed=False)]
    doc["result"]["streams"] = 0
    stamp_checksum(doc)
    assert "CRITERION_MISSING" in _codes(verify_document(doc))


def test_an_unevaluated_criterion_is_never_filed_as_a_host_problem() -> None:
    doc = make_v3_doc()
    doc["ladder"] = [
        _rung(8, constants.SEEDS[0], passed=False, criteria_evaluated=PARTIAL_CRITERIA)
    ]
    doc["result"]["streams"] = 0
    stamp_checksum(doc)
    assert verify_document(doc).ok
    # `valid` and `invalid_reason` name host fitness, and a criterion nobody evaluated is
    # not a host being unfit.
    assert doc["ladder"][0]["valid"] is True
    assert doc["ladder"][0]["invalid_reason"] is None
