# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the frozen kill-rule calculator."""

from __future__ import annotations

import pytest
from verbatim_bench import constants
from verbatim_bench.arms.registry import ArmStatus
from verbatim_bench.killrule import (
    ArmMeasurement,
    ComparabilityRefusal,
    RuleOutcome,
    evaluate_rule_1,
    evaluate_rule_2,
    fraction_of_ceiling,
)

pytestmark = pytest.mark.cpu

BOX = "box-123"
OTHER_BOX = "box-456"
DIE = constants.VERDICT_DIE
DTYPE = "fp32_tf32"


def _c(values=(100.0, 102.0, 98.0, 101.0, 99.0)) -> list[float]:
    return list(values)


def test_fraction_of_ceiling_is_min_of_three_over_median_of_five() -> None:
    assert fraction_of_ceiling([90, 100, 95], _c()) == pytest.approx(90 / 100.0)
    with pytest.raises(ValueError):
        fraction_of_ceiling([90, 100], _c())
    with pytest.raises(ValueError):
        fraction_of_ceiling([90, 100, 95], [100.0, 101.0])


def test_exactly_the_threshold_does_not_kill() -> None:
    verdict = evaluate_rule_1(
        s_runs=[70, 70, 70],
        c_runs=[100.0, 100.0, 100.0, 100.0, 100.0],
        dtype_s=DTYPE,
        dtype_c=DTYPE,
        box_id_s=BOX,
        box_id_c=BOX,
        die=DIE,
    )
    assert verdict.outcome is RuleOutcome.CONTINUE
    assert verdict.f == pytest.approx(0.7)


def test_just_below_the_threshold_kills() -> None:
    verdict = evaluate_rule_1(
        s_runs=[699, 700, 700],
        c_runs=[1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        dtype_s=DTYPE,
        dtype_c=DTYPE,
        box_id_s=BOX,
        box_id_c=BOX,
        die=DIE,
    )
    assert verdict.f == pytest.approx(0.699)
    assert verdict.outcome is RuleOutcome.KILL


def test_rule_two_kills_at_exactly_half_the_ceiling() -> None:
    members = {
        "d1": ArmMeasurement(
            arm_id="d1",
            status=ArmStatus.MEASURED,
            s_runs=(50, 50, 50),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
        "d2": ArmMeasurement(
            arm_id="d2",
            status=ArmStatus.MEASURED,
            s_runs=(10, 10, 10),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
        "d3": ArmMeasurement(
            arm_id="d3",
            status=ArmStatus.MEASURED,
            s_runs=(10, 10, 10),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
    }
    verdict = evaluate_rule_2(
        members=members, c_runs=[100.0] * 5, dtype_c=DTYPE, box_id_c=BOX, die=DIE
    )
    assert verdict.outcome is RuleOutcome.KILL


def test_rule_two_with_an_unrun_member_is_not_evaluable() -> None:
    members = {
        "d1": ArmMeasurement(
            arm_id="d1",
            status=ArmStatus.MEASURED,
            s_runs=(10, 10, 10),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
        "d2": ArmMeasurement(
            arm_id="d2",
            status=ArmStatus.NOT_RUN,
            s_runs=(),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
        "d3": ArmMeasurement(
            arm_id="d3",
            status=ArmStatus.MEASURED,
            s_runs=(10, 10, 10),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
    }
    verdict = evaluate_rule_2(
        members=members, c_runs=[100.0] * 5, dtype_c=DTYPE, box_id_c=BOX, die=DIE
    )
    assert verdict.outcome is RuleOutcome.NOT_EVALUABLE


def test_rule_two_not_evaluable_never_becomes_a_continue() -> None:
    members = {
        "d1": ArmMeasurement(
            arm_id="d1",
            status=ArmStatus.MEASURED,
            s_runs=(10, 10, 10),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
        "d2": {"status": "not_run", "s_runs": (), "dtype_class": DTYPE, "box_id": BOX, "die": DIE},
        "d3": ArmMeasurement(
            arm_id="d3",
            status=ArmStatus.MEASURED,
            s_runs=(10, 10, 10),
            dtype_class=DTYPE,
            box_id=BOX,
            die=DIE,
        ),
    }
    verdict = evaluate_rule_2(
        members=members,
        c_runs=[100.0] * 5,
        dtype_c=DTYPE,
        box_id_c=BOX,
        die=DIE,  # type: ignore[arg-type]
    )
    assert verdict.outcome is not RuleOutcome.CONTINUE
    assert verdict.outcome is RuleOutcome.NOT_EVALUABLE


def test_a_ratio_across_two_box_sessions_is_refused() -> None:
    with pytest.raises(ComparabilityRefusal):
        evaluate_rule_1(
            s_runs=[90, 90, 90],
            c_runs=[100.0] * 5,
            dtype_s=DTYPE,
            dtype_c=DTYPE,
            box_id_s=BOX,
            box_id_c=OTHER_BOX,
            die=DIE,
        )


def test_a_ratio_across_two_dtype_classes_is_refused() -> None:
    with pytest.raises(ComparabilityRefusal):
        evaluate_rule_1(
            s_runs=[90, 90, 90],
            c_runs=[100.0] * 5,
            dtype_s=DTYPE,
            dtype_c="bf16",
            box_id_s=BOX,
            box_id_c=BOX,
            die=DIE,
        )


def test_a_die_other_than_the_pre_declared_one_is_refused() -> None:
    with pytest.raises(ComparabilityRefusal):
        evaluate_rule_1(
            s_runs=[90, 90, 90],
            c_runs=[100.0] * 5,
            dtype_s=DTYPE,
            dtype_c=DTYPE,
            box_id_s=BOX,
            box_id_c=BOX,
            die="RTX 5090",
        )


def test_verdict_carries_the_die_and_the_box_id() -> None:
    verdict = evaluate_rule_1(
        s_runs=[90, 90, 90],
        c_runs=[100.0] * 5,
        dtype_s=DTYPE,
        dtype_c=DTYPE,
        box_id_s=BOX,
        box_id_c=BOX,
        die=DIE,
    )
    assert verdict.die == DIE
    assert verdict.box_id == BOX
    assert verdict.rule == 1
