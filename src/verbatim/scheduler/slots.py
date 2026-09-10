# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Accounting over NeMo's ``CacheAwareContextManager`` slot table.

Verbatim never touches NeMo's tables directly. ``SlotTable`` guarantees
``len(reserved) <= num_slots`` *before* a first frame reaches ``get_context``, so
NeMo's ``RuntimeError("No free slots available")`` is unreachable in steady
operation and is treated as a bug when it happens.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This is integer bookkeeping only. This module must not depend on
the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

__all__ = ["SlotTable"]


class SlotTable:
    """Reserved-slot accounting. ``reserved`` is live sessions plus pad rows plus
    draining sessions; pad rows are reserved at warm-up and never released."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"slot capacity must be >= 1, got {capacity!r}")
        self._capacity = capacity
        self._reserved = 0

    @property
    def capacity(self) -> int:
        """The configured ``num_slots``."""
        return self._capacity

    @property
    def reserved(self) -> int:
        """Slots currently held."""
        return self._reserved

    def free(self) -> int:
        """Slots still available for admission."""
        return self._capacity - self._reserved

    def reserve(self, n: int = 1) -> None:
        """Hold ``n`` slots. Exceeding capacity means admission let someone through
        it should not have, so reaching it is a bug, raised loudly."""
        if n < 1:
            raise ValueError(f"reserve count must be >= 1, got {n!r}")
        if self._reserved + n > self._capacity:
            raise RuntimeError(
                f"slot reservation of {n} exceeds capacity {self._capacity} "
                f"(reserved={self._reserved}): admission should have made this "
                f"unreachable; treating it as a bug"
            )
        self._reserved += n

    def release(self, n: int = 1) -> None:
        """Free ``n`` slots. Releasing what was never reserved is a caller bug."""
        if n < 1:
            raise ValueError(f"release count must be >= 1, got {n!r}")
        if n > self._reserved:
            raise RuntimeError(
                f"slot release of {n} exceeds reserved {self._reserved}: "
                f"releasing a slot that was never reserved"
            )
        self._reserved -= n
