# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Ring buffer acceptance tests: exact reads, no overwrite, zero-padded drain, exact counters."""

from __future__ import annotations

import numpy as np

from verbatim.core.ring import RingBuffer


def test_pop_returns_none_when_short() -> None:
    ring = RingBuffer(2560)
    ring.write(np.ones(100, dtype=np.float32))
    assert ring.pop(2560) is None
    # The short read consumed nothing.
    assert ring.available == 100


def test_pop_returns_exactly_n_and_advances() -> None:
    ring = RingBuffer(4096)
    audio = np.arange(3000, dtype=np.float32)
    assert ring.write(audio) == 3000
    out = ring.pop(1000)
    assert out is not None
    assert len(out) == 1000
    assert list(out) == list(audio[:1000])
    assert ring.available == 2000
    rest = ring.pop(2000)
    assert rest is not None
    assert list(rest) == list(audio[1000:])
    assert ring.available == 0


def test_write_never_overwrites_unread_audio() -> None:
    ring = RingBuffer(1000)
    first = np.arange(1000, dtype=np.float32)
    assert ring.write(first) == 1000
    accepted = ring.write(np.full(500, -1.0, dtype=np.float32))
    assert accepted < 500
    assert accepted == 0
    # Audio is never dropped server-side: the buffered samples are unchanged.
    out = ring.pop(1000)
    assert out is not None
    assert list(out) == list(first)


def test_drain_zero_pads_and_reports_valid_count() -> None:
    ring = RingBuffer(4096)
    ring.write(np.ones(100, dtype=np.float32))
    out, valid = ring.drain(2560)
    assert valid == 100
    assert len(out) == 2560
    assert list(out[:100]) == [1.0] * 100
    assert list(out[100:]) == [0.0] * 2460
    assert ring.available == 0


def test_sample_counters_are_exact_over_many_wraps() -> None:
    capacity = 5000
    chunk = 256
    ring = RingBuffer(capacity)
    expected: list[float] = []
    received: list[float] = []
    cursor = 0
    for _ in range(10_000):
        block = np.arange(cursor, cursor + chunk, dtype=np.float32)
        cursor += chunk
        assert ring.write(block) == chunk
        expected.extend(block.tolist())
        out = ring.pop(chunk)
        assert out is not None
        received.extend(out.tolist())
    assert ring.total_written == 10_000 * chunk
    assert ring.total_popped == 10_000 * chunk
    assert received == expected
