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

``/readyz`` ends with three keys added after every earlier one, whose keys and values
are unchanged: ``word_confidence``, the ``--word-confidence`` value whose configuration
matches the decoding configuration NeMo applied (null when none matches or it cannot be
read; see ``ServiceFacts.word_confidence``), ``observed``, the facts in
``OBSERVED_KEYS`` read off the built model and decoder at the moment of the request
(``verbatim.pipelines.observed``), and ``code``, the directories of the packages this
process imported (``code_paths``). A reporter given nothing to observe reports null
for each observed fact, which is "not observed" and never a default.

``code`` exists because a server answers with the code it imported, which is not always
the code of the checkout it was started from: an editable install elsewhere in the same
virtualenv shadows a checkout's ``src`` unless something puts that first on the path. A
capture that names the checkout it meant to test names what was asked for; ``code``
names what answered.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Protocol

from verbatim.obs.metrics import MetricsSnapshot, render

__all__ = [
    "CODE_KEYS",
    "OBSERVED_KEYS",
    "STALE_TICKS",
    "HealthReporter",
    "HealthResponse",
    "HealthSource",
    "ServiceFacts",
    "code_paths",
]

#: The keys of ``/readyz``'s ``observed`` object, in wire order. Each is read off the
#: built pipeline when ``/readyz`` is asked, not taken from a flag or a spec.
OBSERVED_KEYS: Final = ("att_context_size", "decoder_step_confidence", "decoder_graphs")

#: Each key of ``/readyz``'s ``code`` object, in wire order, with the package it names.
_CODE_PACKAGES: Final = (("verbatim_path", "verbatim"), ("bench_path", "verbatim_bench"))
#: The keys of ``/readyz``'s ``code`` object, in wire order.
CODE_KEYS: Final = tuple(key for key, _ in _CODE_PACKAGES)

#: A tick loop whose last tick is older than this many periods is reported not alive.
STALE_TICKS: Final = 3.0
JSON: Final = "application/json; charset=utf-8"
PROMETHEUS: Final = "text/plain; version=0.0.4; charset=utf-8"


def _package_dir(module: ModuleType) -> str | None:
    """``os.path.dirname(module.__file__)`` with every symlink resolved; None for a module
    with no file, which names no directory."""
    file = getattr(module, "__file__", None)
    if not isinstance(file, str):
        return None
    return os.path.realpath(os.path.dirname(file))


def code_paths() -> dict[str, str | None]:
    """``/readyz``'s ``code`` object: the directory of each package this process imports.

    ``verbatim_path`` is the ``verbatim`` package that is serving, and ``bench_path`` the
    ``verbatim_bench`` package, or null when that cannot be imported: the server does not
    need it. Both are read off the module objects the import system returns at the moment
    of the request, never computed from a checkout, a flag or this file's own location.
    """
    paths: dict[str, str | None] = {}
    for key, name in _CODE_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            paths[key] = None
            continue
        paths[key] = _package_dir(module)
    return paths


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
    #: Whether this server serves per-session phrase lists. It labels every metric and
    #: appears on ``/readyz`` because turning biasing on changes the decoder's arithmetic
    #: for every row, biased or not: a transcript digest from a biasing server is not
    #: comparable with one from a server without it, and a harness that could not read
    #: this would file both under the same arm.
    biasing: bool = False
    #: The runtime this server is actually running on, reported so a row can be attributed
    #: to it. None where there is nothing to report, which is the fake pipeline.
    #:
    #: It is on ``/readyz`` rather than only in the startup banner because a banner is read
    #: by whoever was watching. On 2026-09-15 two A6000 ladders on the same card, the same
    #: checkpoint and the same bucket gave boundaries of 22 and 17, and neither record
    #: carried a NeMo version, so the two could not be compared and the difference could
    #: not be attributed to anything. A capacity number without the runtime that produced
    #: it is not a measurement of a server, it is a measurement of an afternoon.
    nemo_version: str | None = None
    torch_version: str | None = None
    device_name: str | None = None
    #: The ``--word-confidence`` value this server's pipeline was built with: "off",
    #: "nemo-shipped" or "paper-best", read back from the decoding configuration NeMo
    #: applied (``verbatim.pipelines.observed.configured_word_confidence``). None where it
    #: could not be read. Finals carry each word's ``"c"`` only when it is not "off".
    #: Not a metrics label: the label set of every series stays what it was.
    word_confidence: str | None = None

    def labels(self) -> dict[str, str]:
        return {
            "chunk_ms": str(self.chunk_ms),
            "precision": self.precision,
            "execution": self.execution,
            "biasing": "on" if self.biasing else "off",
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class HealthResponse:
    status: int
    content_type: str
    body: str


class HealthReporter:
    ROUTES: Final = frozenset({"/healthz", "/readyz", "/admission", "/metrics"})

    def __init__(
        self,
        source: HealthSource,
        facts: ServiceFacts,
        *,
        observe: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        """``observe`` reads the built pipeline and returns the ``OBSERVED_KEYS``; it is
        called on every ``/readyz``. Without it every observed key is null."""
        self._source = source
        self._facts = facts
        self._observe = observe

    @property
    def facts(self) -> ServiceFacts:
        return self._facts

    def observed(self) -> dict[str, object]:
        """``/readyz``'s ``observed`` object: exactly ``OBSERVED_KEYS``, null for any
        key the reading does not carry."""
        reading = self._observe() if self._observe is not None else {}
        return {key: reading.get(key) for key in OBSERVED_KEYS}

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
                    "biasing": self._facts.biasing,
                    "nemo_version": self._facts.nemo_version,
                    "torch_version": self._facts.torch_version,
                    "device_name": self._facts.device_name,
                    "tick_id": snapshot.tick_id,
                    # Added after every earlier key, which keep their order and values.
                    "word_confidence": self._facts.word_confidence,
                    "observed": self.observed(),
                    "code": code_paths(),
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
                "result_partials_dropped": snapshot.results_partials_dropped_total,
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
