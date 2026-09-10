# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the frozen benchmark constants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from verbatim_bench import constants

pytestmark = pytest.mark.cpu


def test_frozen_constants_round_trips_through_json() -> None:
    values = constants.frozen_constants()
    assert json.loads(json.dumps(values)) == values


def test_every_public_constant_is_in_frozen_constants() -> None:
    helper_names = {"METHODOLOGY_PATH", "FROZEN_BLOCK_MARKER"}
    public_names = {
        name
        for name in vars(constants)
        if name.isupper() and not name.startswith("_") and name not in helper_names
    }
    assert public_names <= set(constants.frozen_constants())


def test_unfrozen_thresholds_are_none_not_a_guess() -> None:
    assert all(getattr(constants, name) is None for name in constants.UNFROZEN_THRESHOLDS)
    assert constants.unfrozen_names() == constants.UNFROZEN_THRESHOLDS


def test_seeds_are_three_and_distinct() -> None:
    assert len(constants.SEEDS) == 3
    assert len(set(constants.SEEDS)) == 3


def test_kill_thresholds_are_exact_floats_not_rounded() -> None:
    assert constants.KILL_RULE_1_THRESHOLD == 0.7
    assert constants.KILL_RULE_1_THRESHOLD > 0.699
    assert constants.KILL_RULE_2_THRESHOLD == 0.5


def test_ceiling_stream_floor_is_four_times_the_batch() -> None:
    assert (
        constants.CEILING_MIN_INPUT_STREAMS_PER_BATCH_SLOT * max(constants.CEILING_BATCH_SIZES)
        == 512
    )


def test_constants_import_pulls_in_no_optional_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    bench_src = str(root / "bench" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (bench_src, environment.get("PYTHONPATH")) if part
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import verbatim_bench.constants; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_not_measured_dies_are_named() -> None:
    assert constants.NOT_MEASURED_DIES == ("RTX 5090", "B200")
    assert constants.VERDICT_DIE not in constants.NOT_MEASURED_DIES
