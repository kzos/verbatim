# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Slot accounting acceptance tests: capacity, reconciliation, the unreachable reserve."""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.slots import SlotTable
from verbatim.scheduler.tick import TickLoop

CHUNK = ChunkMode(160)


def _config(**overrides) -> EngineConfig:
    args: dict = {"chunk": CHUNK, "buckets": (16,), "calibrated_ceiling": 16}
    args.update(overrides)
    return EngineConfig(**args)


def test_capacity_is_the_configured_num_slots() -> None:
    config = _config()
    assert config.num_slots == 16 + 16 + 8 + 8
    assert SlotTable(config.num_slots).capacity == config.num_slots


def test_reserve_and_release_reconcile() -> None:
    table = SlotTable(8)
    table.reserve(3)
    assert table.reserved == 3
    assert table.free() == 5
    table.release(2)
    assert table.reserved == 1
    table.release(1)
    assert table.reserved == 0
    assert table.free() == 8


def test_reserving_beyond_capacity_raises() -> None:
    table = SlotTable(2)
    table.reserve(2)
    with pytest.raises(RuntimeError, match="unreachable"):
        table.reserve(1)


def test_pad_rows_hold_slots_for_the_process_lifetime() -> None:
    config = _config()
    loop = TickLoop(
        config,
        FakePipelineAdapter(CHUNK, buckets=(16,)),
        SessionRegistry(),
        clock=SimulatedClock(),
    )
    # Pads are reserved at warm-up, before any session exists.
    assert loop.slots.reserved == config.effective_pad
    session = Session(1, CHUNK)
    session.configure()
    assert loop.admit_session(session).admitted
    session.ring.write(np.zeros(CHUNK.samples, dtype=np.float32))
    session.begin_draining()
    loop.run_for(3)
    assert session.state.value == "CLOSED"
    # The drain released the session's slot; the pads were never released.
    assert loop.slots.reserved == config.effective_pad
