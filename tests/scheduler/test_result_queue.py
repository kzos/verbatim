# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The bounded result queue's drop policy, on its own."""

from __future__ import annotations

import asyncio

import pytest

from verbatim.core.errors import ErrorCode, Unavailable
from verbatim.core.types import StepResult
from verbatim.engine import _ErrorEvent, _ResultQueue

pytestmark = pytest.mark.cpu


def _partial(tick: int) -> StepResult:
    return StepResult(
        stream_id=1,
        tick_id=tick,
        partial_text=f"p{tick}",
        final_text=None,
        audio_processed_s=0.0,
        eager=False,
    )


def _final(tick: int, *, is_last: bool) -> StepResult:
    return StepResult(
        stream_id=1,
        tick_id=tick,
        partial_text="",
        final_text=f"f{tick}",
        audio_processed_s=0.0,
        eager=False,
        is_last=is_last,
    )


async def test_the_oldest_partials_are_dropped_to_hold_the_bound() -> None:
    queue = _ResultQueue(3)
    assert all(queue.put(_partial(i)) == 0 for i in range(3))
    assert queue.put(_partial(3)) == 1  # over the bound: the oldest partial goes
    texts = [(await queue.get()).partial_text for _ in range(3)]
    assert texts == ["p1", "p2", "p3"], "p0 was dropped, the rest kept in order"


async def test_a_final_is_never_dropped_even_under_a_flood_of_partials() -> None:
    queue = _ResultQueue(2)
    queue.put(_final(0, is_last=False))  # a mid-stream final (endpointing)
    for i in range(1, 20):
        queue.put(_partial(i))
    drained = []
    for _ in range(2):
        drained.append(await queue.get())
    assert drained[0].final_text == "f0", "the final survived the flood"
    assert drained[1].partial_text == "p19", "only the newest partial remained beside it"


async def test_an_error_and_the_last_row_are_never_dropped() -> None:
    queue = _ResultQueue(1)
    queue.put(_partial(0))
    queue.put(_final(1, is_last=True))  # the terminal
    queue.put(_ErrorEvent(1, Unavailable("gone")))
    items = [await queue.get() for _ in range(2)]
    assert isinstance(items[0], StepResult) and items[0].is_last
    assert isinstance(items[1], _ErrorEvent) and items[1].error.code is ErrorCode.UNAVAILABLE


def _terminal(tick: int) -> StepResult:
    """An `is_last` row with no final text: the aborted or drain terminal that ends the
    stream without a final hypothesis. Not a pure partial, so never dropped; if it were,
    `results()` would wait forever on a terminal the backlog threw away."""
    return StepResult(
        stream_id=1,
        tick_id=tick,
        partial_text="",
        final_text=None,
        audio_processed_s=0.0,
        eager=False,
        is_last=True,
    )


async def test_a_bare_last_row_with_no_final_text_is_never_dropped() -> None:
    queue = _ResultQueue(2)
    queue.put(_terminal(0))  # the terminal: is_last, final_text None
    for i in range(1, 20):
        queue.put(_partial(i))
    drained = [await queue.get() for _ in range(2)]
    assert drained[0].is_last and drained[0].final_text is None, "the bare terminal survived"
    assert drained[1].partial_text == "p19", "only the newest partial remained beside it"


async def test_get_blocks_until_an_item_arrives() -> None:
    queue = _ResultQueue(4)
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0.02)
    assert not getter.done()
    queue.put(_partial(7))
    assert (await asyncio.wait_for(getter, timeout=1.0)).partial_text == "p7"
