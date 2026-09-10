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
    assert config.num_slots == 16 + 16 + 7 + 8  # bucket, steady pads, edge pads, margin
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
    # Pads are reserved at warm-up, before any session exists: the steady pad rows and
    # the edge pad rows that can be in flight during one edge step.
    assert loop.slots.reserved == config.effective_pad + config.edge_pad_rows
    session = Session(1, CHUNK)
    session.configure()
    assert loop.admit_session(session).admitted
    session.ring.write(np.zeros(CHUNK.samples, dtype=np.float32))
    session.begin_draining()
    loop.run_for(3)
    assert session.state.value == "CLOSED"
    # The drain released the session's slot; the pads were never released.
    assert loop.slots.reserved == config.effective_pad + config.edge_pad_rows


def test_edge_pad_rows_are_reserved_at_warm_up_so_free_is_the_truth() -> None:
    """An edge batch of eight holds at most seven one-shot pad rows, and they take NeMo
    slots for the step. The table reserves them up front, so `free()` never counts a
    slot an edge step is about to use, and the sum of the named terms is the capacity."""
    config = _config(edge_batch=8)
    loop = TickLoop(
        config,
        FakePipelineAdapter(CHUNK, buckets=(16,)),
        SessionRegistry(),
        clock=SimulatedClock(),
    )
    assert config.edge_pad_rows == 7
    assert loop.slots.reserved == config.effective_pad + 7
    assert loop.slots.free() == max(config.buckets or ()) + config.drain_margin
    single = _config(edge_batch=1)
    loop = TickLoop(
        single, FakePipelineAdapter(CHUNK, buckets=(16,)), SessionRegistry(), clock=SimulatedClock()
    )
    assert loop.slots.reserved == single.effective_pad
