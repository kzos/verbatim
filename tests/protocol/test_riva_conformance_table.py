# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The conformance table is the reviewable artefact: every descriptor field needs a row."""

from __future__ import annotations

import pytest

from verbatim.protocols.riva.conformance import (
    CONFORMANCE,
    FieldStatus,
    missing_fields,
    rules_for,
)

pytestmark = pytest.mark.cpu


def test_every_recognition_config_field_has_a_rule() -> None:
    assert missing_fields().get("RecognitionConfig", ()) == ()


def test_every_streaming_request_field_has_a_rule() -> None:
    missing = missing_fields()
    assert missing.get("StreamingRecognizeRequest", ()) == ()
    assert missing.get("StreamingRecognitionConfig", ()) == ()
    assert missing.get("EndpointingConfig", ()) == ()


def test_rules_are_unique() -> None:
    keys = [(rule.message, rule.field) for rule in CONFORMANCE]
    assert len(keys) == len(set(keys))


def test_rejected_fields_name_a_status() -> None:
    rejected = [rule for rule in CONFORMANCE if rule.status is FieldStatus.REJECTED]
    assert rejected
    for rule in rejected:
        statuses = ("INVALID_ARGUMENT", "NOT_FOUND", "UNIMPLEMENTED")
        assert any(status in rule.behaviour for status in statuses), (
            f"{rule.message}.{rule.field} names no status"
        )


def test_force_eou_rule_says_accepted_and_ignored() -> None:
    force_eou = [
        rule for rule in rules_for("StreamingRecognizeRequest") if "force_eou" in rule.field
    ]
    assert force_eou
    assert all(rule.status is FieldStatus.ACCEPTED_AND_IGNORED for rule in force_eou)
