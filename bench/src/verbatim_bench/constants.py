# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Frozen benchmark inputs shared by the harness and its published method."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION_FOR_RUN: Final = "vb-results/3"

# Latency
LATENCY_PRIMARY: Final = "word_emission"
LATENCY_SECONDARY: Final = "chunk_watermark"
LATENCY_FALLBACK_IF_PRIMARY_UNAVAILABLE: Final = "chunk_watermark"
LATENCY_DELETED: Final = "first_partial_after_send"
X_MS: Final = 150

# The rung
WARM_UP_S: Final = 60
WINDOW_S: Final = 180
WARM_UP_READING_S: Final = 30
WARM_UP_CONVERGENCE: Final = 0.10
WARM_UP_CAP_S: Final = 120

# The ladder
LADDER_N0_FRACTION_OF_C: Final = 0.5
LADDER_N0_WITHOUT_CEILING: Final = 16
LADDER_MULTIPLIER: Final = 1.15
LADDER_RESOLUTION: Final = 0.02
LADDER_INVALID_RUNGS_TO_ABORT: Final = 2
SEEDS: Final = (20260914, 20260915, 20260916)
S_REPEATS: Final = 3

# The workload
SESSION_PROFILE: Final = "m180"
PACING_PROFILE: Final = "uniform"
FRAME_MS: Final = 20
FRAME_JITTER_MS: Final = 10
CHUNK_MODES_DAY21: Final = (160, 560)

# The ceiling
CEILING_BATCH_SIZES: Final = (32, 128)
CEILING_NUM_SLOTS_EQUALS_BATCH: Final = True
CEILING_WARMUP_STEPS: Final = 1
CEILING_RUN_STEPS: Final = 3
CEILING_MEDIAN_OF: Final = 5
CEILING_MIN_INPUT_STREAMS_PER_BATCH_SLOT: Final = 4
CEILING_RTFX_CROSS_CHECK: Final = 0.02

# The kill arithmetic
KILL_RULE_1_THRESHOLD: Final = 0.7
KILL_RULE_2_THRESHOLD: Final = 0.5
F_MAINTAINER_RERUN_ABOVE: Final = 1.1
WER_WINDOW_ABSOLUTE: Final = 0.1

# Validity
CLIENT_CPU_MAX_FRACTION_OF_CPUSET: Final = 0.5
PACING_SLIP_P99_MAX_MS: Final = 5.0
CLIENT_LIMITED_FLOOR_MULTIPLE: Final = 2.0
STEAL_PCT_MAX: Final = 0.0
CGROUP_THROTTLED_DELTA_MAX: Final = 0
NVML_THROTTLE_EVENTS_MAX: Final = 0
FOREIGN_GPU_PROCESSES_MAX: Final = 0
GPU_PERSISTENCE_MODE_REQUIRED: Final = True

# Calibrated by the harness on the box, per section 7 and DR-0007. Never guess these.
PSI_CPU_SOME_MAX_PCT: Final[float | None] = 0.4
PSI_CPU_FULL_MAX_PCT: Final[float | None] = 0.0
# Not yet frozen: no measurement exists. Never guess these.
AA_SPREAD_MAX_PCT: Final[float | None] = None
NULL_FLOOR_TOLERANCE_PCT: Final[float | None] = None
UNFROZEN_THRESHOLDS: Final = (
    "AA_SPREAD_MAX_PCT",
    "NULL_FLOOR_TOLERANCE_PCT",
)

# The die the verdict is read on
VERDICT_DIE: Final = "NVIDIA RTX A6000"
VERDICT_SM: Final = "8.6"
CONFIRMATION_DIE: Final = "B300"
NOT_MEASURED_DIES: Final = ("RTX 5090", "B200")

METHODOLOGY_PATH: Final = Path("benchmarks/METHODOLOGY.md")
FROZEN_BLOCK_MARKER: Final = "<!-- frozen-constants -->"

_HELPER_CONSTANTS = frozenset({"METHODOLOGY_PATH", "FROZEN_BLOCK_MARKER"})


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def frozen_constants() -> dict[str, Any]:
    """Return the JSON-comparable source used by the document's frozen block.

    Deriving the mapping from the module namespace means a newly declared public
    constant cannot silently avoid the document drift check.
    """
    return {
        name: _json_value(value)
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_") and name not in _HELPER_CONSTANTS
    }


def unfrozen_names() -> tuple[str, ...]:
    """Return the declared thresholds whose calibration values are still null."""
    return tuple(name for name in UNFROZEN_THRESHOLDS if globals()[name] is None)
