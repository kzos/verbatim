# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Counters the engine keeps from what the tick loop already computes.

``ticks_over_budget_total`` counts ticks whose own cost exceeded the tick budget
(``chunk_ms * utilisation_target``); ``ticks_late_total`` counts ticks that ended after
the next boundary, so a tick was missed. ``steady_steps_total`` and ``eager_steps_total``
count batches stepped, not rows. There is no ``graph_replays_total``: the engine cannot
observe whether NeMo replayed a graph, and a counter nobody observed would be a number
presented as observed. Whether the run is eager or graphed is a label the operator's
choice supplies to the exposition, not a count.
"""

from __future__ import annotations

from dataclasses import dataclass

from verbatim.core.types import TickStats

__all__ = ["Counters"]


@dataclass(slots=True)
class Counters:
    ticks_total: int = 0
    ticks_over_budget_total: int = 0
    ticks_late_total: int = 0
    steady_steps_total: int = 0
    eager_steps_total: int = 0
    sessions_admitted_total: int = 0
    sessions_refused_total: int = 0

    def observe_tick(self, stats: TickStats, *, budget_ms: float, period_ms: float) -> None:
        """Fold one completed tick in. A tick is over budget when its modelled cost, the
        number the admission controller observes, exceeds the budget; it is late when
        it ended after the next boundary, which is measured on the engine's clock."""
        self.ticks_total += 1
        if stats.step_ms + stats.edge_ms > budget_ms:
            self.ticks_over_budget_total += 1
        if stats.lateness_ms > period_ms:
            self.ticks_late_total += 1
        if stats.steady_rows > 0:
            self.steady_steps_total += 1
        self.eager_steps_total += stats.edge_batches

    def observe_admission(self, admitted: bool) -> None:
        if admitted:
            self.sessions_admitted_total += 1
        else:
            self.sessions_refused_total += 1
