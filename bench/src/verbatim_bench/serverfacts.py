# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""What the server says it is running, read from the server rather than from the operator.

A row's execution mode -- graphed or eager, which precision, which pipeline -- was until
now a free-text string typed into ``--arm`` and copied verbatim into the artifact. Nothing
in the harness ever asked the server. That is the same defect class as a warm-up recording
a capture it never made (``docs/decisions/0011``): a label that is an input to the run
rather than an observation of it.

The server has exposed the answer all along. ``/readyz`` on the WebSocket listener returns
the model, the pipeline, the chunk mode, the precision and the execution mode, and since
``docs/decisions/0011`` the execution mode is read from the capture controller, which
proved it by warming up. This module fetches it with the standard library only, because
``verify`` must keep running on a reviewer's machine with nothing installed.

A contradiction between what was declared and what the server reports is refused rather
than recorded. A run mislabelled at the top is worse than no run: its numbers are real and
attached to the wrong arm, and nothing downstream can tell.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

__all__ = ["ArmContradiction", "ServerFacts", "check_arm", "health_url", "read_server_facts"]


class ArmContradiction(RuntimeError):
    """The declared arm and the server's own report disagree."""


@dataclass(frozen=True, slots=True)
class ServerFacts:
    """One reading of ``/readyz``. ``None`` fields are ones the server did not report."""

    ready: bool
    model: str | None
    pipeline: str | None
    chunk_ms: int | None
    precision: str | None
    execution: str | None
    tick_id: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model": self.model,
            "pipeline": self.pipeline,
            "chunk_ms": self.chunk_ms,
            "precision": self.precision,
            "execution": self.execution,
            "tick_id": self.tick_id,
        }


def health_url(endpoint: str, path: str = "/readyz") -> str:
    """The health URL beside a ``ws://host:port/v1/stream`` endpoint.

    The two listeners share a port, so this is the same host and port over http. A
    ``wss://`` endpoint maps to ``https://``; anything else is taken as already http.
    """
    parsed = urlparse(endpoint)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme or "http")
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


Fetcher = Callable[[str], str]


def _fetch(url: str, *, timeout_s: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def read_server_facts(endpoint: str, *, fetch: Fetcher | None = None) -> ServerFacts | None:
    """Read ``/readyz`` beside ``endpoint``. None when the server does not answer.

    None is "no reading", which is different from a reading that disagrees, and is left
    for the caller to judge: a server too old to carry these fields is a different
    situation from one contradicting its own label.
    """
    fetch = fetch or _fetch
    try:
        body = json.loads(fetch(health_url(endpoint)))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    if not isinstance(body, dict):
        return None
    return ServerFacts(
        ready=bool(body.get("ready")),
        model=body.get("model"),
        pipeline=body.get("pipeline"),
        chunk_ms=body.get("chunk_ms"),
        precision=body.get("precision"),
        execution=body.get("execution"),
        tick_id=body.get("tick_id"),
    )


#: Tokens an arm name uses for each execution mode. An arm that names none of them gets
#: no opinion: this refuses a contradiction, it does not impose a naming convention.
_EXECUTION_TOKENS: dict[str, tuple[str, ...]] = {
    "graph path": ("graphed", "graphs", "cudagraph", "cuda-graph"),
    "eager": ("eager",),
}


def check_arm(
    facts: ServerFacts | None,
    *,
    arm: str,
    declared_dtype: str | None = None,
    declared_chunk_ms: int | None = None,
) -> None:
    """Raise ``ArmContradiction`` when the declaration and the server disagree.

    Three comparisons, each only made when both sides are present:

    * the declared dtype against the server's reported precision, which is exact;
    * the declared chunk mode against the server's, which is exact;
    * the arm name against the server's execution mode, but only when the arm names a
      mode at all. An arm called "b300-bf16-graphed" on an eager server is a
      contradiction; an arm called "run-4" says nothing about execution and is left alone.
    """
    if facts is None:
        return
    if declared_dtype and facts.precision and declared_dtype != facts.precision:
        raise ArmContradiction(
            f"--dtype {declared_dtype!r} and the server reports precision "
            f"{facts.precision!r}: the row would name a precision this server is not "
            "running"
        )
    if declared_chunk_ms and facts.chunk_ms and int(declared_chunk_ms) != int(facts.chunk_ms):
        raise ArmContradiction(
            f"the run is configured for {declared_chunk_ms} ms chunks and the server "
            f"reports {facts.chunk_ms} ms"
        )
    execution = facts.execution
    if not execution:
        return
    lowered = arm.lower()
    for mode, tokens in _EXECUTION_TOKENS.items():
        if mode == execution:
            continue
        if any(token in lowered for token in tokens):
            raise ArmContradiction(
                f"the arm {arm!r} names {mode!r} and the server reports "
                f"execution {execution!r}: a row mislabelled at the top carries real "
                "numbers under the wrong arm, and nothing downstream can tell"
            )
