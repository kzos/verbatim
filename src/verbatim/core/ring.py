# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Per-session lock-free PCM ring buffer: fixed capacity, no allocation on the hot path.

Single producer (a decode worker thread), single consumer (the tick thread).
Chunk boundaries are cut from the session's own sample counter, never from the
wall clock -- that is what makes a session's frame sequence a pure function of its
audio, and therefore what makes batch invariance reachable at all.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This buffer is clock-free bookkeeping with no model behind it.
This module must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RingBuffer"]


class RingBuffer:
    """Fixed-capacity float32 ring. Single producer, single consumer, no allocation on pop.

    Chunk boundaries come from this object's own sample counter, never from a clock:
    that is what makes a session's frame sequence a pure function of its audio.

    This is the one and only place padding is invented (`drain` zero-pads the last
    chunk to exactly `n` samples). `write`'s partial accept is a back-pressure
    signal -- the caller retains the remainder and stops reading its socket -- and
    never a drop: audio is never dropped server-side.
    """

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples < 1:
            raise ValueError(f"capacity_samples must be >= 1, got {capacity_samples!r}")
        self._capacity = capacity_samples
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._head = 0
        self._size = 0
        self._written = 0
        self._popped = 0

    @property
    def available(self) -> int:
        """Buffered (unread) samples."""
        return self._size

    @property
    def capacity(self) -> int:
        """Total capacity in samples."""
        return self._capacity

    @property
    def total_written(self) -> int:
        """Samples ever accepted."""
        return self._written

    @property
    def total_popped(self) -> int:
        """Samples ever consumed, including drain reads."""
        return self._popped

    def write(self, samples: np.ndarray) -> int:
        """Append. Returns the number accepted. Never overwrites unread audio: when full
        it accepts fewer than offered, and the caller stops reading its socket. Audio is
        never dropped server-side."""
        arr = np.asarray(samples, dtype=np.float32).ravel()
        room = self._capacity - self._size
        n = min(len(arr), room)
        if n <= 0:
            return 0
        tail = (self._head + self._size) % self._capacity
        first = min(n, self._capacity - tail)
        self._buf[tail : tail + first] = arr[:first]
        if n > first:
            self._buf[: n - first] = arr[first:n]
        self._size += n
        self._written += n
        return n

    def pop(self, n: int) -> np.ndarray | None:
        """Exactly n samples, or None when fewer are available. Never a short read."""
        if n < 0:
            raise ValueError(f"pop count must be >= 0, got {n!r}")
        if self._size < n:
            return None
        out = np.empty(n, dtype=np.float32)
        first = min(n, self._capacity - self._head)
        out[:first] = self._buf[self._head : self._head + first]
        if n > first:
            out[first:] = self._buf[: n - first]
        self._head = (self._head + n) % self._capacity
        self._size -= n
        self._popped += n
        return out

    def drain(self, n: int) -> tuple[np.ndarray, int]:
        """The last chunk: up to n samples zero-padded to exactly n, plus the valid count."""
        if n < 0:
            raise ValueError(f"drain count must be >= 0, got {n!r}")
        valid = min(self._size, n)
        out = np.zeros(n, dtype=np.float32)
        if valid > 0:
            taken = self.pop(valid)
            assert taken is not None
            out[:valid] = taken
        return out, valid
