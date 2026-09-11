# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The null server: accepts audio, echoes a fixed partial per chunk, nothing else.

This is the harness's own floor: the p95 measured against it is the latency the
client itself costs, and a real row whose p95 is within 2x of the floor is flagged
"client-limited" rather than believed. It is also the test double for the paced
load generator -- no real server, no GPU, no model.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

import websockets
from websockets.asyncio.server import Server, ServerConnection

_SAMPLE_RATE_HZ = 16000
_BYTES_PER_SECOND = _SAMPLE_RATE_HZ * 2


@dataclass
class NullServerConfig:
    chunk_ms: int = 160
    partial_text: str = "null"
    final_text: str = "null server"
    emit_words: bool = False
    partial_delay_ms: float = 0.0
    fail_after_chunks: int | None = None
    capacity: int | None = None
    overload_penalty_ms: float = 160.0
    #: Milliseconds added to `partial_delay_ms` per second since the server started.
    #: A server whose latency climbs without bound never settles, which is the only way
    #: to exercise the warm-up cap without waiting for a real one to misbehave.
    partial_delay_growth_ms_per_s: float = 0.0
    word_script: tuple[str, ...] | None = None
    retract_mode: bool = False


@dataclass
class _SessionState:
    id: str
    received_bytes: int = 0
    chunks_emitted: int = 0


class NullServer:
    """Accepts audio, echoes a fixed partial per chunk. Nothing else.

    This is the harness's own floor: the p95 measured against it is the latency the
    client itself costs, and a real row whose p95 is within 2x of the floor is flagged
    "client-limited" rather than believed. It is also the test double for every test in
    this task -- no real server, no GPU, no model.

    Usage:
        async with NullServer(NullServerConfig()) as server:
            ...  server.endpoint -> "ws://127.0.0.1:<ephemeral port>/v1/stream"
    """

    def __init__(self, config: NullServerConfig | None = None) -> None:
        self.config = config or NullServerConfig()
        self.endpoint: str = ""
        self.sessions_seen: int = 0
        self.bytes_received: int = 0
        self._server: Server | None = None
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._live = 0
        self._started_at = time.monotonic()

    async def __aenter__(self) -> NullServer:
        self._started_at = time.monotonic()
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        sock = self._server.sockets
        if not sock:
            raise RuntimeError("null server bound no sockets")
        port = sock[0].getsockname()[1]
        self.endpoint = f"ws://127.0.0.1:{port}/v1/stream"
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _chunk_bytes(self, chunk_ms: int) -> int:
        return chunk_ms * _SAMPLE_RATE_HZ // 1000 * 2

    async def _handle(self, ws: ServerConnection) -> None:
        query = parse_qs(urlsplit(ws.request.path).query)
        try:
            chunk_ms = int(query.get("chunk_ms", [str(self.config.chunk_ms)])[0])
        except ValueError:
            chunk_ms = self.config.chunk_ms
        words = query.get("words", ["0"])[0] == "1"
        session = _SessionState(id=f"null-{next(self._ids)}")
        async with self._lock:
            self.sessions_seen += 1
            self._live += 1
        frame_bytes = self._chunk_bytes(chunk_ms)
        await ws.send(
            json.dumps(
                {
                    "type": "session",
                    "id": session.id,
                    "chunk_ms": chunk_ms,
                    "invariance_class": "null",
                }
            )
        )
        try:
            await self._serve_session(ws, session, frame_bytes, words)
        finally:
            async with self._lock:
                self._live -= 1

    def _partial_text_for(self, chunks_emitted: int) -> str:
        if self.config.word_script is not None:
            shown = self.config.word_script[:chunks_emitted]
            return " ".join(shown)
        if self.config.retract_mode:
            words = self.config.partial_text.split()
            if chunks_emitted == 2 and len(words) > 1:
                return " ".join(words[:-1])
            return self.config.partial_text
        return self.config.partial_text

    def _final_text_for(self) -> str:
        if self.config.word_script is not None:
            return " ".join(self.config.word_script)
        return self.config.final_text

    async def _chunk_delay_s(self) -> float:
        delay_ms = self.config.partial_delay_ms
        if self.config.partial_delay_growth_ms_per_s:
            elapsed_s = time.monotonic() - self._started_at
            delay_ms += self.config.partial_delay_growth_ms_per_s * elapsed_s
        if self.config.capacity is not None:
            async with self._lock:
                live = self._live
            overload = max(0, live - self.config.capacity)
            delay_ms += overload * self.config.overload_penalty_ms
        return delay_ms / 1000.0

    async def _serve_session(
        self, ws: ServerConnection, session: _SessionState, frame_bytes: int, words: bool
    ) -> None:
        pending = 0

        async def _emit_partial(audio_s: float) -> None:
            delay_s = await self._chunk_delay_s()
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            await ws.send(
                json.dumps(
                    {
                        "type": "partial",
                        "text": self._partial_text_for(session.chunks_emitted),
                        "audio_s": audio_s,
                    }
                )
            )

        async def maybe_fail() -> bool:
            if (
                self.config.fail_after_chunks is not None
                and session.chunks_emitted >= self.config.fail_after_chunks
            ):
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "null_injected_failure",
                            "message": "fail_after_chunks reached",
                        }
                    )
                )
                await ws.close()
                return True
            return False

        try:
            async for message in ws:
                if isinstance(message, str):
                    try:
                        event: Any = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "end":
                        if pending > 0:
                            pending = 0
                            session.chunks_emitted += 1
                            audio_s = session.received_bytes / _BYTES_PER_SECOND
                            await _emit_partial(audio_s)
                        final: dict[str, Any] = {
                            "type": "final",
                            "text": self._final_text_for(),
                            "audio_s": session.received_bytes / _BYTES_PER_SECOND,
                        }
                        if words or self.config.emit_words:
                            final["words"] = []
                        await ws.send(json.dumps(final))
                        await ws.close()
                        return
                    continue
                session.received_bytes += len(message)
                async with self._lock:
                    self.bytes_received += len(message)
                pending += len(message)
                while pending >= frame_bytes:
                    pending -= frame_bytes
                    session.chunks_emitted += 1
                    audio_s = session.received_bytes / _BYTES_PER_SECOND
                    await _emit_partial(audio_s)
                    if await maybe_fail():
                        return
        except websockets.exceptions.ConnectionClosed:
            pass
