# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``/healthz``, ``/readyz``, ``/metrics``, ``/admission`` on the WebSocket listener.

Served through the ``websockets`` library's ``process_request`` hook, no web
framework: a plain HTTP GET to one of these paths gets a JSON body (the text
exposition for ``/metrics``) and never a WebSocket handshake. The reporter reads one
``MetricsSnapshot`` from the engine per request and judges from it:

- ``/readyz`` is 200 once the pipeline is built, the engine started, the tick loop
  running, and at least one tick has completed; 503 otherwise, with the reason.
- ``/healthz`` is 200 while the tick loop is running and its last tick is recent,
  within ``STALE_TICKS`` periods; 503 when it is not running, dead, or stalled.
- ``/admission`` is always 200: the runtime ceiling, the live count, the slot table,
  the degradation level, and the last refusal's reason with the counts.
- ``/metrics`` is the Prometheus text exposition, labelled with the process facts.

The README once said ``/readyz`` would also wait for every bucket key to be captured
and an invariance self-test to pass. Neither exists; this reporter says what it
checks, not what a document hoped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, Protocol

from verbatim.obs.metrics import MetricsSnapshot, render

__all__ = ["STALE_TICKS", "HealthReporter", "HealthResponse", "HealthSource", "ServiceFacts"]

#: A tick loop whose last tick is older than this many periods is reported not alive.
STALE_TICKS: Final = 3.0
JSON: Final = "application/json; charset=utf-8"
PROMETHEUS: Final = "text/plain; version=0.0.4; charset=utf-8"


class HealthSource(Protocol):
    def snapshot(self) -> MetricsSnapshot: ...


@dataclass(frozen=True, slots=True)
class ServiceFacts:
    """What the operator's choices fixed for this process, carried as labels."""

    model: str
    chunk_ms: int
    precision: str
    execution: str  # "eager", "graph path" or "fake"
    pipeline: str

    def labels(self) -> dict[str, str]:
        return {
            "chunk_ms": str(self.chunk_ms),
            "precision": self.precision,
            "execution": self.execution,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class HealthResponse:
    status: int
    content_type: str
    body: str


class HealthReporter:
    ROUTES: Final = frozenset({"/healthz", "/readyz", "/admission", "/metrics"})

    def __init__(self, source: HealthSource, facts: ServiceFacts) -> None:
        self._source = source
        self._facts = facts

    @property
    def facts(self) -> ServiceFacts:
        return self._facts

    def route(self, path: str) -> HealthResponse | None:
        """The response for a health path, or ``None`` when the path is not one."""
        if path not in self.ROUTES:
            return None
        snapshot = self._source.snapshot()
        if path == "/metrics":
            return HealthResponse(200, PROMETHEUS, render(snapshot, labels=self._facts.labels()))
        if path == "/healthz":
            alive, reason = _liveness(snapshot)
            return _json(
                200 if alive else 503,
                {
                    "alive": alive,
                    "reason": reason,
                    "running": snapshot.running,
                    "dead": snapshot.dead,
                    "tick_id": snapshot.tick_id,
                    "last_tick_age_s": snapshot.last_tick_age_s,
                },
            )
        if path == "/readyz":
            ready, reason = _readiness(snapshot)
            return _json(
                200 if ready else 503,
                {
                    "ready": ready,
                    "reason": reason,
                    "model": self._facts.model,
                    "pipeline": self._facts.pipeline,
                    "chunk_ms": self._facts.chunk_ms,
                    "precision": self._facts.precision,
                    "execution": self._facts.execution,
                    "tick_id": snapshot.tick_id,
                },
            )
        counters = snapshot.counters
        return _json(
            200,
            {
                "ceiling": snapshot.ceiling,
                "calibrated_ceiling": snapshot.calibrated_ceiling,
                "bucket": snapshot.bucket,
                "live": snapshot.live_sessions,
                "slots": {
                    "capacity": snapshot.slots_capacity,
                    "reserved": snapshot.slots_reserved,
                    "free": snapshot.slots_free,
                },
                "degradation_level": snapshot.degradation_level,
                "consecutive_overruns": snapshot.consecutive_overruns,
                "admitted_total": counters.sessions_admitted_total,
                "refused_total": counters.sessions_refused_total,
                "last_refusal_reason": snapshot.last_refusal_reason,
                "p95_tick_ms": snapshot.p95_tick_ms,
                "eager_step_fraction": snapshot.eager_step_fraction,
            },
        )


def _json(status: int, body: dict[str, object]) -> HealthResponse:
    return HealthResponse(status, JSON, json.dumps(body, separators=(",", ":")) + "\n")


def _liveness(snapshot: MetricsSnapshot) -> tuple[bool, str | None]:
    if snapshot.dead:
        return False, "tick loop dead"
    if not snapshot.running:
        return False, "tick loop not running"
    age = snapshot.last_tick_age_s
    limit = STALE_TICKS * snapshot.period_ms / 1000.0
    if age is not None and age > limit:
        return False, f"last tick {age:.2f} s ago, more than {STALE_TICKS:g} periods"
    return True, None


def _readiness(snapshot: MetricsSnapshot) -> tuple[bool, str | None]:
    if snapshot.dead:
        return False, "tick loop dead"
    if not snapshot.started:
        return False, "engine not started"
    if not snapshot.running:
        return False, "tick loop not running"
    if snapshot.tick_id < 1:
        return False, "no tick completed yet"
    return True, None
