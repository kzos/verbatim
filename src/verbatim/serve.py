# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""One engine behind both wires: what ``verbatim serve`` runs once it has a pipeline.

``ServeSettings`` is the validated operator input. ``engine_config`` turns it into
the ``EngineConfig`` the scheduler runs on; there is no bucket unless the operator
named one, either a ceiling from a measured row or a bucket named as uncalibrated
for the ladder run. ``run_server`` binds the Riva and WebSocket listeners over one
``Engine`` and holds them open until ``shutdown`` is set, then closes the listeners
first and the engine last, so no socket outlives the engine that fed it.

A session mid-stream at shutdown is told the server is going away: the engine raises
`UNAVAILABLE` on every result stream still open when it stops, which the Riva wire
returns as the `UNAVAILABLE` status and the WebSocket as close code 1001. A clean end
is reserved for an utterance that actually finished, so a caller can tell the two
apart and decide whether to retry. That is the engine's rule, not this module's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import Engine
from verbatim.pipelines.base import PipelineAdapter
from verbatim.protocols.health import HealthReporter, ServiceFacts
from verbatim.protocols.riva.server import RivaServer, RivaServerConfig
from verbatim.protocols.ws.server import WsServer, WsServerConfig
from verbatim.scheduler.graph_budget import ConfigError

__all__ = ["PIPELINES", "Endpoints", "ServeSettings", "engine_config", "run_server"]

#: What ``--pipeline`` accepts, in the order the help lists them. The two cache-aware
#: names are NeMo's two ``ASRDecodingType`` branches for its cache-aware builder; the
#: adapter each names refuses a pipeline built for the other.
PIPELINES: tuple[str, ...] = ("cache_aware_rnnt", "cache_aware_ctc", "fake")


@dataclass(frozen=True, slots=True)
class ServeSettings:
    """What the operator asked for. Validated once, here, so every later message can
    trust it. Exactly one of ``ceiling`` and ``bucket`` is set."""

    model: str
    chunk: ChunkMode
    ceiling: int | None = None
    bucket: int | None = None
    host: str = "0.0.0.0"
    ws_port: int = 8080
    grpc_port: int = 50051
    idle_timeout_s: float = 30.0
    ring_seconds: float = 3.0
    stop_history_eou_ms: int = 800
    pipeline: str = "cache_aware_rnnt"
    eager: bool = False
    padding: str = "fixed"
    biasing: bool = False
    decoder_graphs: bool = False
    att_context_left: int | None = None
    language_code: str = "en-US"
    compute_dtype: str = "bfloat16"
    device_id: int = 0

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("a checkpoint name or .nemo path is required")
        if (self.ceiling is None) == (self.bucket is None):
            raise ConfigError(
                "exactly one of --ceiling (from a measured row) and --bucket (named as "
                "uncalibrated, for the ladder run) is required: a bucket nobody measured is "
                "never assumed"
            )
        for name in ("ceiling", "bucket"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 1):
                raise ConfigError(f"--{name} must be a positive integer, got {value!r}")
        for name in ("ws_port", "grpc_port"):
            port = getattr(self, name)
            if isinstance(port, bool) or not 0 <= port <= 65535:
                raise ConfigError(f"--{name.replace('_', '-')} must be 0..65535, got {port!r}")
        if self.ws_port and self.ws_port == self.grpc_port:
            raise ConfigError(f"--ws-port and --grpc-port are both {self.ws_port}")
        if not self.idle_timeout_s > 0:
            raise ConfigError(
                f"--idle-timeout must be positive seconds, got {self.idle_timeout_s!r}; a "
                "server does not run without a deadline for sessions that send nothing"
            )
        if not self.ring_seconds > 0:
            raise ConfigError(f"--ring-seconds must be positive, got {self.ring_seconds!r}")
        if self.stop_history_eou_ms < 0:
            raise ConfigError(
                f"--stop-history-eou-ms must be >= 0, got {self.stop_history_eou_ms!r}"
            )
        if self.pipeline not in PIPELINES:
            named = ", ".join(PIPELINES)
            raise ConfigError(f"--pipeline must be one of {named}, got {self.pipeline!r}")
        if self.biasing and self.pipeline != "cache_aware_rnnt":
            raise ConfigError(
                f"--biasing needs --pipeline cache_aware_rnnt, got {self.pipeline!r}: NeMo "
                "reaches per-stream biasing through the RNNT decoding computer and no other "
                "pipeline here has one, so the server would accept phrase lists and "
                "transcribe every session unbiased"
            )
        if not self.host:
            raise ConfigError("--host must not be empty")

    @property
    def calibrated(self) -> bool:
        return self.ceiling is not None


def engine_config(settings: ServeSettings) -> EngineConfig:
    """The scheduler configuration for these settings. A ceiling is the calibrated
    ceiling and the single bucket; a named bucket is the single bucket with no ceiling,
    which is what the ladder run measures against."""
    return EngineConfig(
        chunk=settings.chunk,
        buckets=None if settings.calibrated else (settings.bucket,),
        calibrated_ceiling=settings.ceiling,
        ring_seconds=settings.ring_seconds,
        pipeline=settings.pipeline,
        stop_history_eou_ms=settings.stop_history_eou_ms,
        idle_timeout_s=settings.idle_timeout_s,
        padding=settings.padding,
        biasing=settings.biasing,
    )


@dataclass(frozen=True, slots=True)
class Endpoints:
    """Where the two wires are listening, with the ports actually bound."""

    ws_endpoint: str
    grpc_target: str
    ws_port: int
    grpc_port: int


async def run_server(
    settings: ServeSettings,
    adapter: PipelineAdapter,
    *,
    shutdown: asyncio.Event,
    on_ready: Callable[[Endpoints], None] | None = None,
    execution: str | None = None,
    runtime: Mapping[str, str | None] | None = None,
) -> None:
    """Serve until ``shutdown`` is set. Listeners close before the engine stops.

    ``execution`` names how the encoder step runs, "eager", "graph path" or "fake",
    for the health endpoints and the metrics labels; the CLI knows it exactly and
    passes it, and the default only derives it from the settings.

    ``runtime`` is what the installed stack turned out to be -- ``nemo``, ``torch`` and
    ``device`` -- which the CLI has already probed to decide the graph path. It reaches
    ``/readyz`` so a harness can put it on the row: two ladders of the same checkpoint on
    the same card are not comparable without it.
    """
    engine = Engine(engine_config(settings), adapter)
    if execution is None:
        # Read from the capture controller, which decided the mode from the adapter's
        # capability and then proved it by warming up, rather than re-derived from the
        # flag the operator passed. The two agreed through the CLI path and could not be
        # made to disagree there -- but they were two answers to one question, and the
        # one that is checked is this one.
        execution = "fake" if settings.pipeline == "fake" else engine.execution
    facts = ServiceFacts(
        model=settings.model,
        chunk_ms=settings.chunk.ms,
        precision="none" if settings.pipeline == "fake" else settings.compute_dtype,
        execution=execution,
        pipeline=settings.pipeline,
        # Read from the adapter, not from the flag: the adapter refused to start unless
        # the built decoder actually carried the biasing arena, so this is what the
        # server can do rather than what was asked of it.
        biasing=bool(getattr(adapter, "biasing", False)),
        nemo_version=(runtime or {}).get("nemo"),
        torch_version=(runtime or {}).get("torch"),
        device_name=(runtime or {}).get("device"),
    )
    riva = RivaServer(
        engine,
        RivaServerConfig(
            host=settings.host,
            port=settings.grpc_port,
            served_models=(settings.model,),
            language_code=settings.language_code,
        ),
    )
    ws = WsServer(
        engine,
        WsServerConfig(host=settings.host, port=settings.ws_port),
        health=HealthReporter(engine, facts),
    )
    async with engine, riva, ws:
        if on_ready is not None:
            on_ready(
                Endpoints(
                    ws_endpoint=ws.endpoint,
                    grpc_target=f"{settings.host}:{riva.port}",
                    ws_port=ws.port,
                    grpc_port=riva.port,
                )
            )
        await shutdown.wait()
