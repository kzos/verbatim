# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The metrics snapshot and its Prometheus text exposition.

``MetricsSnapshot`` is data the engine hands out under its lock; ``render`` turns it
into the text exposition format (version 0.0.4). Every series carries the labels the
operator's choices fix for the process: chunk mode, precision, execution and model,
because a number without them cannot be read against decision record 0003.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verbatim.obs.counters import Counters

__all__ = ["MetricsSnapshot", "render"]


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    chunk_ms: int
    period_ms: float
    budget_ms: float
    bucket: int
    started: bool
    running: bool
    dead: bool
    tick_id: int
    last_tick_age_s: float | None
    live_sessions: int
    slots_capacity: int
    slots_reserved: int
    slots_free: int
    ceiling: int | None
    calibrated_ceiling: int | None
    degradation_level: int
    consecutive_overruns: int
    p95_tick_ms: float
    eager_step_fraction: float
    counters: Counters
    latency_count: int
    partial_latency_ms: tuple[float, float, float]
    tick_cost_ms: tuple[float, float, float]
    last_refusal_reason: str | None
    results_partials_dropped_total: int = 0


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(labels: Mapping[str, str], **extra: str) -> str:
    items = {**labels, **extra}
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_escape(str(v))}"' for k, v in items.items()) + "}"


def render(snapshot: MetricsSnapshot, *, labels: Mapping[str, str]) -> str:
    """Prometheus text exposition of one snapshot, with ``labels`` on every series."""
    c = snapshot.counters
    lines: list[str] = []

    def series(name: str, kind: str, help_text: str, value: float | int, **extra: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name}{_labels(labels, **extra)} {value}")

    series("verbatim_up", "gauge", "1 while the tick loop is running.", int(snapshot.running))
    series(
        "verbatim_tick_loop_dead",
        "gauge",
        "1 after the tick loop stopped on a failure.",
        int(snapshot.dead),
    )
    series("verbatim_ticks_total", "counter", "Ticks completed.", c.ticks_total)
    series(
        "verbatim_ticks_over_budget_total",
        "counter",
        "Ticks whose own cost exceeded the tick budget.",
        c.ticks_over_budget_total,
    )
    series(
        "verbatim_ticks_late_total",
        "counter",
        "Ticks that ended after the next boundary.",
        c.ticks_late_total,
    )
    series(
        "verbatim_steady_steps_total", "counter", "Steady batches stepped.", c.steady_steps_total
    )
    series(
        "verbatim_eager_steps_total",
        "counter",
        "Edge batches stepped eagerly.",
        c.eager_steps_total,
    )
    series(
        "verbatim_eager_step_fraction",
        "gauge",
        "Eager steps over all steps, as the admission controller counts them.",
        snapshot.eager_step_fraction,
    )
    series(
        "verbatim_sessions_admitted_total",
        "counter",
        "Sessions admitted.",
        c.sessions_admitted_total,
    )
    series(
        "verbatim_sessions_refused_total", "counter", "Sessions refused.", c.sessions_refused_total
    )
    series(
        "verbatim_result_partials_dropped_total",
        "counter",
        "Partials dropped from a session's result backlog to hold its bound.",
        snapshot.results_partials_dropped_total,
    )
    series("verbatim_live_sessions", "gauge", "Sessions live now.", snapshot.live_sessions)
    series(
        "verbatim_slots_capacity",
        "gauge",
        "NeMo slots the pipeline was built with.",
        snapshot.slots_capacity,
    )
    series("verbatim_slots_free", "gauge", "NeMo slots free now.", snapshot.slots_free)
    if snapshot.ceiling is not None:
        series("verbatim_ceiling", "gauge", "The runtime admission ceiling.", snapshot.ceiling)
    series(
        "verbatim_degradation_level",
        "gauge",
        "0 healthy; 1 admissions held; 2 bursts held; 3 bucket shrunk.",
        snapshot.degradation_level,
    )
    series(
        "verbatim_tick_p95_ms",
        "gauge",
        "p95 of the modelled tick time over the admission window.",
        snapshot.p95_tick_ms,
    )
    for q, value in zip(("0.5", "0.95", "0.99"), snapshot.tick_cost_ms, strict=True):
        series(
            "verbatim_tick_cost_ms",
            "gauge",
            "Tick cost, step plus edge, over the retained window.",
            value,
            quantile=q,
        )
    for q, value in zip(("0.5", "0.95", "0.99"), snapshot.partial_latency_ms, strict=True):
        series(
            "verbatim_partial_latency_ms",
            "gauge",
            "Server-side partial latency: lateness past the tick boundary plus the delay "
            "to the loop.",
            value,
            quantile=q,
        )
    if snapshot.last_tick_age_s is not None:
        series(
            "verbatim_last_tick_age_seconds",
            "gauge",
            "Seconds since the last completed tick.",
            round(snapshot.last_tick_age_s, 6),
        )
    return "\n".join(lines) + "\n"
