# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the frozen arm registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from verbatim_bench.arms.registry import (
    REGISTRY,
    ArmOutcome,
    ArmStatus,
    Comparability,
    NotRunReason,
    Panel,
    arms_in_kill_rule,
    main_table_arms,
    missing_rows,
    not_measured_entries,
)

pytestmark = pytest.mark.cpu


def test_registry_covers_every_pre_declared_arm() -> None:
    assert sorted(REGISTRY) == ["a", "b", "c", "d1", "d2", "d3", "e", "f", "g", "h"]


def test_kill_rule_membership_is_exactly_as_ruled() -> None:
    assert {spec.arm_id for spec in arms_in_kill_rule(1)} == {"a"}
    assert {spec.arm_id for spec in arms_in_kill_rule(2)} == {"d1", "d2", "d3"}
    ceiling = REGISTRY["b"]
    assert ceiling.kill_rule is None
    assert all(spec.kill_rule != 1 or spec.arm_id == "a" for spec in REGISTRY.values())


def test_arm_e_is_not_comparable_and_carries_no_fraction_of_ceiling() -> None:
    arm_e = REGISTRY["e"]
    assert arm_e.comparability is Comparability.NOT_COMPARABLE
    assert arm_e.carries_fraction_of_ceiling is False
    assert arm_e.kill_rule is None


def test_arm_e_renders_only_in_the_not_comparable_panel() -> None:
    assert REGISTRY["e"].panel is Panel.NOT_COMPARABLE
    assert REGISTRY["e"].arm_id not in {spec.arm_id for spec in main_table_arms()}
    assert REGISTRY["e"].arm_id not in {spec.arm_id for spec in not_measured_entries()}


def test_contributed_and_vendor_arms_are_not_measured_entries() -> None:
    entries = {spec.arm_id for spec in not_measured_entries()}
    assert entries == {"f", "g", "h"}
    assert all(REGISTRY[arm_id].kill_rule is None for arm_id in entries)
    assert all(REGISTRY[arm_id].carries_fraction_of_ceiling is False for arm_id in entries)


def test_every_arm_has_at_least_one_pre_declared_not_run_reason() -> None:
    for arm_id, spec in REGISTRY.items():
        assert len(spec.not_run_reasons) >= 1, arm_id
        assert all(isinstance(reason, NotRunReason) for reason in spec.not_run_reasons)
    assert NotRunReason.BUDGET in REGISTRY["e"].not_run_reasons


def test_missing_rows_reports_an_arm_with_no_outcome() -> None:
    outcomes = [
        ArmOutcome(arm_id=arm_id, status=ArmStatus.MEASURED)
        for arm_id in ("a", "b", "c", "d1", "d2", "d3", "e", "f", "g")
    ]
    assert missing_rows(outcomes) == ("h",)
    full = [
        *outcomes,
        ArmOutcome(
            arm_id="h", status=ArmStatus.NOT_RUN, reason=NotRunReason.PERMISSION_NOT_GRANTED
        ),
    ]
    assert missing_rows(full) == ()


def test_not_run_outcome_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        ArmOutcome(arm_id="a", status=ArmStatus.NOT_RUN, reason=None)
    with pytest.raises(ValueError):
        ArmOutcome(arm_id="a", status=ArmStatus.INVALID, reason=None)
    ok = ArmOutcome(arm_id="a", status=ArmStatus.NOT_RUN, reason=NotRunReason.BUDGET)
    assert ok.reason is NotRunReason.BUDGET


def test_pins_file_lists_every_registry_arm() -> None:
    pins_path = (
        Path(__file__).resolve().parents[2]
        / "bench"
        / "src"
        / "verbatim_bench"
        / "arms"
        / "PINS.md"
    )
    text = pins_path.read_text(encoding="utf-8")
    for arm_id, spec in REGISTRY.items():
        assert f"## ({arm_id})" in text
        if spec.repo is not None:
            assert spec.repo in text


def test_unpinned_commits_are_none_not_a_placeholder_string() -> None:
    pins_path = (
        Path(__file__).resolve().parents[2]
        / "bench"
        / "src"
        / "verbatim_bench"
        / "arms"
        / "PINS.md"
    )
    text = pins_path.read_text(encoding="utf-8")
    for spec in REGISTRY.values():
        assert spec.pinned_commit is None
    assert "<arm-a-pinned-commit>" in text
    assert "0000000" not in text
    assert "deadbeef" not in text.lower()
