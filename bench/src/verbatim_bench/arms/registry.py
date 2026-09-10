# SPDX-License-Identifier: Apache-2.0
"""The frozen arm inventory used to build every comparative result row.

Keeping the inventory executable prevents an unavailable arm from disappearing
from a table merely because its server was not runnable on a particular day.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ArmRole(StrEnum):
    CANDIDATE = "candidate"
    CEILING = "ceiling"
    INFORMATIONAL = "informational"
    CONTRIBUTED = "contributed"
    VENDOR = "vendor"


class ArmStatus(StrEnum):
    MEASURED = "measured"
    NOT_RUN = "not_run"
    INVALID = "invalid"


class NotRunReason(StrEnum):
    PLATFORM_BOUND = "platform_bound"
    CHECKPOINT_UNLOADABLE = "checkpoint_unloadable"
    CONVERTED_ARTIFACT_UNAVAILABLE = "converted_artifact_unavailable"
    ADAPTER_NOT_BUILT = "adapter_not_built"
    ENGINE_BUILD_UNAVAILABLE = "engine_build_unavailable"
    HARNESS_INCOMPLETE = "harness_incomplete"
    CONTRIBUTED_NOT_BUILT = "contributed_not_built"
    PERMISSION_NOT_GRANTED = "permission_not_granted"
    BUDGET = "budget"


class Comparability(StrEnum):
    CANONICAL = "canonical"
    DIFFERENT_TOOLKIT = "different_toolkit"
    CONVERTED_CHECKPOINT = "converted_checkpoint"
    NOT_COMPARABLE = "not_comparable"


class Panel(StrEnum):
    MAIN = "main"
    NOT_COMPARABLE = "not_comparable"
    NOT_MEASURED = "not_measured"
    VENDOR = "vendor"


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    description: str
    repo: str | None
    surface: str
    role: ArmRole
    kill_rule: int | None
    dtype_class: str
    pinned_commit: str | None
    launch_argv: tuple[str, ...]
    comparability: Comparability
    panel: Panel
    not_run_reasons: tuple[NotRunReason, ...]
    carries_fraction_of_ceiling: bool


_A_REASON = (NotRunReason.ADAPTER_NOT_BUILT, NotRunReason.HARNESS_INCOMPLETE)
_B_REASON = (NotRunReason.HARNESS_INCOMPLETE,)
_C_REASON = (NotRunReason.PERMISSION_NOT_GRANTED, NotRunReason.HARNESS_INCOMPLETE)
_D1_REASON = (NotRunReason.CONVERTED_ARTIFACT_UNAVAILABLE, NotRunReason.ADAPTER_NOT_BUILT)
_D2_REASON = (NotRunReason.CHECKPOINT_UNLOADABLE, NotRunReason.ADAPTER_NOT_BUILT)
_D3_REASON = (NotRunReason.CHECKPOINT_UNLOADABLE, NotRunReason.ADAPTER_NOT_BUILT)
_E_REASON = (NotRunReason.BUDGET,)
_F_REASON = (NotRunReason.CONTRIBUTED_NOT_BUILT,)
_G_REASON = (NotRunReason.CONTRIBUTED_NOT_BUILT,)
_H_REASON = (NotRunReason.PERMISSION_NOT_GRANTED,)


REGISTRY: Final[Mapping[str, ArmSpec]] = {
    "a": ArmSpec(
        arm_id="a",
        description="this project's prototype server",
        repo="Verbatim",
        surface="websocket",
        role=ArmRole.CANDIDATE,
        kill_rule=1,
        dtype_class="fp32_tf32",
        pinned_commit=None,
        launch_argv=("<arm-a-committed-launch-line>",),
        comparability=Comparability.CANONICAL,
        panel=Panel.MAIN,
        not_run_reasons=_A_REASON,
        carries_fraction_of_ceiling=True,
    ),
    "b": ArmSpec(
        arm_id="b",
        description="the toolkit's graphed file-driven ceiling and batch-1 reference",
        repo="NeMo",
        surface="file",
        role=ArmRole.CEILING,
        kill_rule=None,
        dtype_class="bf16",
        pinned_commit=None,
        launch_argv=("<arm-b-committed-launch-line>",),
        comparability=Comparability.DIFFERENT_TOOLKIT,
        panel=Panel.MAIN,
        not_run_reasons=_B_REASON,
        carries_fraction_of_ceiling=True,
    ),
    "c": ArmSpec(
        arm_id="c",
        description="the unlicensed multi-stream WebSocket server",
        repo="modal-nvidia-asr",
        surface="websocket",
        role=ArmRole.INFORMATIONAL,
        kill_rule=None,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-c-committed-launch-line>",),
        comparability=Comparability.DIFFERENT_TOOLKIT,
        panel=Panel.MAIN,
        not_run_reasons=_C_REASON,
        carries_fraction_of_ceiling=False,
    ),
    "d1": ArmSpec(
        arm_id="d1",
        description="the C++ runtime cuda-server over its gRPC binary",
        repo=None,
        surface="riva-grpc",
        role=ArmRole.CANDIDATE,
        kill_rule=2,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-d1-committed-launch-line>",),
        comparability=Comparability.CONVERTED_CHECKPOINT,
        panel=Panel.MAIN,
        not_run_reasons=_D1_REASON,
        carries_fraction_of_ceiling=True,
    ),
    "d2": ArmSpec(
        arm_id="d2",
        description="the batch-1-per-session voice-agent server",
        repo=None,
        surface="openai-realtime",
        role=ArmRole.CANDIDATE,
        kill_rule=2,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-d2-committed-launch-line>",),
        comparability=Comparability.CANONICAL,
        panel=Panel.MAIN,
        not_run_reasons=_D2_REASON,
        carries_fraction_of_ceiling=True,
    ),
    "d3": ArmSpec(
        arm_id="d3",
        description="the community WebSocket server",
        repo=None,
        surface="websocket",
        role=ArmRole.CANDIDATE,
        kill_rule=2,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-d3-committed-launch-line>",),
        comparability=Comparability.CANONICAL,
        panel=Panel.MAIN,
        not_run_reasons=_D3_REASON,
        carries_fraction_of_ceiling=True,
    ),
    "e": ArmSpec(
        arm_id="e",
        description="the language-model engine realtime path on a different checkpoint",
        repo="vLLM Qwen3-ASR",
        surface="openai-realtime",
        role=ArmRole.INFORMATIONAL,
        kill_rule=None,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-e-committed-launch-line>",),
        comparability=Comparability.NOT_COMPARABLE,
        panel=Panel.NOT_COMPARABLE,
        not_run_reasons=_E_REASON,
        carries_fraction_of_ceiling=False,
    ),
    "f": ArmSpec(
        arm_id="f",
        description="contributed inference-server sequence-batching path",
        repo=None,
        surface="none",
        role=ArmRole.CONTRIBUTED,
        kill_rule=None,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-f-committed-launch-line>",),
        comparability=Comparability.NOT_COMPARABLE,
        panel=Panel.NOT_MEASURED,
        not_run_reasons=_F_REASON,
        carries_fraction_of_ceiling=False,
    ),
    "g": ArmSpec(
        arm_id="g",
        description="contributed engine model-state path",
        repo=None,
        surface="none",
        role=ArmRole.CONTRIBUTED,
        kill_rule=None,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-g-committed-launch-line>",),
        comparability=Comparability.NOT_COMPARABLE,
        panel=Panel.NOT_MEASURED,
        not_run_reasons=_G_REASON,
        carries_fraction_of_ceiling=False,
    ),
    "h": ArmSpec(
        arm_id="h",
        description="the vendor's managed container",
        repo=None,
        surface="none",
        role=ArmRole.VENDOR,
        kill_rule=None,
        dtype_class="<unknown>",
        pinned_commit=None,
        launch_argv=("<arm-h-committed-launch-line>",),
        comparability=Comparability.NOT_COMPARABLE,
        panel=Panel.VENDOR,
        not_run_reasons=_H_REASON,
        carries_fraction_of_ceiling=False,
    ),
}


def arms_in_kill_rule(rule: int) -> tuple[ArmSpec, ...]:
    """Return numerator arms for a rule, excluding its ceiling denominator."""
    return tuple(spec for spec in REGISTRY.values() if spec.kill_rule == rule)


def main_table_arms() -> tuple[ArmSpec, ...]:
    """Return arms rendered in the comparable/main panel."""
    return tuple(spec for spec in REGISTRY.values() if spec.panel is Panel.MAIN)


def not_measured_entries() -> tuple[ArmSpec, ...]:
    """Return registry-generated placeholders for contributed and vendor arms."""
    return tuple(
        spec for spec in REGISTRY.values() if spec.panel in (Panel.NOT_MEASURED, Panel.VENDOR)
    )


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    arm_id: str
    status: ArmStatus
    reason: NotRunReason | None = None

    def __post_init__(self) -> None:
        if self.status is not ArmStatus.MEASURED and self.reason is None:
            raise ValueError(f"arm {self.arm_id!r} requires a reason when status is {self.status}")
        if self.status is ArmStatus.MEASURED and self.reason is not None:
            raise ValueError("a measured arm cannot carry a not-run reason")


def missing_rows(outcomes: Iterable[ArmOutcome]) -> tuple[str, ...]:
    """Return pre-declared arm ids that have no outcome at all."""
    present = {outcome.arm_id for outcome in outcomes}
    return tuple(arm_id for arm_id in REGISTRY if arm_id not in present)
