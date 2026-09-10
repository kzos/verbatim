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
from collections.abc import Callable
from dataclasses import dataclass

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import Engine
from verbatim.pipelines.base import PipelineAdapter
from verbatim.protocols.riva.server import RivaServer, RivaServerConfig
from verbatim.protocols.ws.server import WsServer, WsServerConfig
from verbatim.scheduler.graph_budget import ConfigError

__all__ = ["Endpoints", "ServeSettings", "engine_config", "run_server"]


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
        if self.pipeline not in ("cache_aware_rnnt", "fake"):
            raise ConfigError(f"--pipeline must be cache_aware_rnnt or fake, got {self.pipeline!r}")
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
) -> None:
    """Serve until ``shutdown`` is set. Listeners close before the engine stops."""
    engine = Engine(engine_config(settings), adapter)
    riva = RivaServer(
        engine,
        RivaServerConfig(
            host=settings.host,
            port=settings.grpc_port,
            served_models=(settings.model,),
            language_code=settings.language_code,
        ),
    )
    ws = WsServer(engine, WsServerConfig(host=settings.host, port=settings.ws_port))
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
