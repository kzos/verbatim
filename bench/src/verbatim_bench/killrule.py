# SPDX-License-Identifier: Apache-2.0
"""The two frozen kill-rule calculations.

The calculator refuses to manufacture a cross-box or cross-dtype ratio.  That
keeps a visually plausible fraction from becoming an invalid comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from verbatim_bench import constants
from verbatim_bench.arms.registry import ArmStatus


class RuleOutcome(StrEnum):
    KILL = "kill"
    CONTINUE = "continue"
    NOT_EVALUABLE = "not_evaluable"


class ComparabilityRefusal(ValueError):
    """The proposed numerator and denominator cannot be compared."""


@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    """The measured portion of an arm row needed by rule 2."""

    arm_id: str = ""
    status: ArmStatus = ArmStatus.MEASURED
    s_runs: tuple[int, ...] = ()
    dtype_class: str = "<unknown>"
    box_id: str = ""
    die: str = constants.VERDICT_DIE


@dataclass(frozen=True, slots=True)
class KillVerdict:
    rule: int
    outcome: RuleOutcome
    f: float | None
    detail: str
    die: str
    box_id: str


def _as_status(value: ArmStatus | str) -> ArmStatus:
    return value if isinstance(value, ArmStatus) else ArmStatus(value)


def _median_five(values: Sequence[float]) -> float:
    if len(values) != constants.CEILING_MEDIAN_OF:
        raise ValueError(f"ceiling sample must contain exactly {constants.CEILING_MEDIAN_OF} runs")
    ordered = sorted(float(value) for value in values)
    return ordered[len(ordered) // 2]


def fraction_of_ceiling(s_runs: Sequence[int], c_runs: Sequence[float]) -> float:
    """Return min-of-three ``S`` divided by median-of-five ``C``."""
    if len(s_runs) != constants.S_REPEATS:
        raise ValueError(f"stream sample must contain exactly {constants.S_REPEATS} runs")
    ceiling = _median_five(c_runs)
    if ceiling == 0:
        raise ValueError("ceiling sample cannot have a zero median")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in s_runs):
        raise ValueError("stream runs must be integers")
    return min(s_runs) / ceiling


def evaluate_rule_1(
    *,
    s_runs: Sequence[int],
    c_runs: Sequence[float],
    dtype_s: str,
    dtype_c: str,
    box_id_s: str,
    box_id_c: str,
    die: str,
) -> KillVerdict:
    """Evaluate the candidate fraction against the strict rule-1 threshold."""
    if box_id_s != box_id_c:
        raise ComparabilityRefusal("rule 1 requires S and C from the same box session")
    if dtype_s != dtype_c:
        raise ComparabilityRefusal("rule 1 requires matching dtype classes")
    if die != constants.VERDICT_DIE:
        raise ComparabilityRefusal(f"rule 1 is frozen to die {constants.VERDICT_DIE!r}")
    fraction = fraction_of_ceiling(s_runs, c_runs)
    outcome = (
        RuleOutcome.KILL if fraction < constants.KILL_RULE_1_THRESHOLD else RuleOutcome.CONTINUE
    )
    detail = (
        f"F={fraction:.12g} compared with rule-1 threshold {constants.KILL_RULE_1_THRESHOLD:.12g}"
    )
    return KillVerdict(1, outcome, fraction, detail, die, box_id_s)


def _measurement_value(measurement: ArmMeasurement | Mapping[str, object], name: str) -> object:
    if isinstance(measurement, Mapping):
        return measurement.get(name)
    return getattr(measurement, name)


def evaluate_rule_2(
    *,
    members: Mapping[str, ArmMeasurement],
    c_runs: Sequence[float],
    dtype_c: str,
    box_id_c: str,
    die: str,
) -> KillVerdict:
    """Evaluate d1/d2/d3, preserving rule 2's unrun-member asymmetry."""
    if die != constants.VERDICT_DIE:
        raise ComparabilityRefusal(f"rule 2 is frozen to die {constants.VERDICT_DIE!r}")
    ceiling = _median_five(c_runs)
    if ceiling == 0:
        raise ValueError("ceiling sample cannot have a zero median")

    required = ("d1", "d2", "d3")
    unavailable: list[str] = []
    measured_fractions: dict[str, float] = {}
    for arm_id in required:
        measurement = members.get(arm_id)
        if measurement is None:
            unavailable.append(arm_id)
            continue
        status = _as_status(_measurement_value(measurement, "status"))
        if status is not ArmStatus.MEASURED:
            unavailable.append(arm_id)
            continue
        measurement_dtype = _measurement_value(measurement, "dtype_class")
        if measurement_dtype != dtype_c:
            raise ComparabilityRefusal(f"{arm_id} and C have different dtype classes")
        measurement_box = _measurement_value(measurement, "box_id")
        if measurement_box != box_id_c:
            raise ComparabilityRefusal(f"{arm_id} and C are from different box sessions")
        measurement_die = _measurement_value(measurement, "die")
        if measurement_die != die:
            raise ComparabilityRefusal(f"{arm_id} was measured on a different die")
        runs = _measurement_value(measurement, "s_runs")
        if not isinstance(runs, Sequence) or isinstance(runs, str):
            raise ValueError(f"{arm_id} stream runs are not a sequence")
        if len(runs) != constants.S_REPEATS:
            expected = constants.S_REPEATS
            raise ValueError(f"{arm_id} stream sample must hold exactly {expected} runs")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in runs):
            raise ValueError(f"{arm_id} stream runs must be integers")
        measured_fractions[arm_id] = min(runs) / ceiling

    best = max(measured_fractions.values(), default=None)
    if best is not None and best >= constants.KILL_RULE_2_THRESHOLD:
        outcome = RuleOutcome.KILL
    elif unavailable:
        outcome = RuleOutcome.NOT_EVALUABLE
    else:
        outcome = RuleOutcome.CONTINUE
    if best is None:
        detail = f"no measured rule-2 member; unavailable={','.join(unavailable)}"
    else:
        detail = f"best measured F={best:.12g}"
        if unavailable:
            detail += f"; unavailable={','.join(unavailable)}"
    return KillVerdict(2, outcome, best, detail, die, box_id_c)
