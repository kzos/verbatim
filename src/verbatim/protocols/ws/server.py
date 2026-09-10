# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Plain WebSocket server for the demo: binary PCM in, JSON out.

``ws://host:8080/v1/stream?chunk_ms=160&lang=en-US&words=1``. The same listener
serves the demo page and the health endpoints. The day-21 prototype row is taken
over this surface, because the Riva subset is month-2 work; the row's ``surface``
field says so.

Chunk boundaries are cut from the session's own sample counter, never from the
wall clock: inbound bytes are buffered and the recognizer is handed exactly one
``chunk_ms`` chunk at a time, so the chunk sequence is a pure function of the
audio. The trailing tail is zero-padded to a full chunk and passed as the final
chunk, but the session's audio clock advances by the real bytes handed to the
recognizer -- the server still pads the tail for the recognizer and still never
counts the bytes it invented into ``audio_s``. The session therefore stamps
every outgoing frame from its own real-byte counter rather than forwarding
the recognizer's view, which cannot tell real bytes from padding.

Back-pressure: no audio is ever dropped. If a session's buffer exceeds its cap
(``ring_seconds`` of audio), the server stops reading that socket until the
recognizer drains it, so the peer's own flow control slows the client down.
With the synchronous recognizers used here every full chunk is consumed before
the next read, so the retained buffer stays below one chunk; the cap is the
guard that keeps that promise explicit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, replace
from urllib.parse import parse_qsl

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from verbatim.core.errors import ErrorCode, InvalidArgument, VerbatimError
from verbatim.protocols.base import (
    SAMPLE_RATE_HZ,
    VALID_CHUNK_MS,
    Hypothesis,
    Recognizer,
    RecognizerFactory,
    SessionOptions,
)
from verbatim.protocols.ws.frames import (
    ErrorFrame,
    FinalFrame,
    PartialFrame,
    SessionFrame,
    parse_client_text,
    parse_query,
)

__all__ = ["WsServer", "WsServerConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WsServerConfig:
    """Listener configuration. `invariance_class` stays the `<unmeasured>`
    placeholder until a measured value exists; never fabricate one."""

    host: str = "127.0.0.1"
    port: int = 8080
    path: str = "/v1/stream"
    invariance_class: str = "<unmeasured>"  # a placeholder, never a fabricated value
    ring_seconds: float = 3.0  # back-pressure cap
    default_chunk_ms: int = 160

    def __post_init__(self) -> None:
        if self.default_chunk_ms not in VALID_CHUNK_MS:
            valid = ", ".join(str(v) for v in VALID_CHUNK_MS)
            raise ValueError(
                f"invalid default_chunk_ms {self.default_chunk_ms!r}: must be one of {valid}"
            )


class _SessionClosed(Exception):
    """Control flow: the session was closed after telling the client why."""


class WsServer:
    """The plain WebSocket demo surface. Binary PCM in, JSON out.

    Usage:
        server = WsServer(stub_recognizer_factory(), WsServerConfig(port=0))
        async with server:
            ...  server.endpoint -> "ws://127.0.0.1:<bound port>/v1/stream"
    """

    def __init__(self, factory: RecognizerFactory, config: WsServerConfig | None = None) -> None:
        self._factory = factory
        self._config = config if config is not None else WsServerConfig()
        self._server: Server | None = None
        self._live = 0
        self._total = 0

    @property
    def endpoint(self) -> str:
        return f"ws://{self._config.host}:{self.port}{self._config.path}"

    @property
    def port(self) -> int:
        """The bound port; useful when config.port == 0."""
        if self._server is not None and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self._config.port

    @property
    def live_sessions(self) -> int:
        return self._live

    @property
    def sessions_total(self) -> int:
        return self._total

    @property
    def _cap_bytes(self) -> int:
        return int(self._config.ring_seconds * SAMPLE_RATE_HZ * 2)

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle,
            self._config.host,
            self._config.port,
            process_request=self._process_request,
            max_size=None,
        )

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    async def __aenter__(self) -> WsServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        """Serve the endpoint at `config.path` only; anything else is HTTP 404."""
        path, _, _ = request.path.partition("?")
        if path != self._config.path:
            return connection.respond(404, "not found")
        return None

    async def _send_hypothesis(
        self, ws: ServerConnection, hypo: Hypothesis, options: SessionOptions, audio_s: float
    ) -> None:
        """Emit one hypothesis, stamped with the session's own audio clock."""
        if hypo.is_final:
            words = tuple(hypo.words) if options.word_timestamps else None
            await ws.send(FinalFrame(text=hypo.text, audio_s=audio_s, words=words).to_json())
        elif options.interim_results:
            await ws.send(PartialFrame(text=hypo.text, audio_s=audio_s).to_json())

    async def _consume_chunk(
        self,
        ws: ServerConnection,
        recognizer: Recognizer,
        chunk: bytes,
        options: SessionOptions,
        audio_s: float,
    ) -> None:
        """Pass one full chunk to the recognizer and emit what is ready.

        A recognizer failure becomes an `ErrorFrame` and a closed session; the
        client is always told why. Raises _SessionClosed after closing.
        """
        try:
            hypos = recognizer.add_chunk(chunk)
        except VerbatimError as exc:
            await ws.send(ErrorFrame(code=exc.code, message=str(exc)).to_json())
            await ws.close()
            raise _SessionClosed from exc
        except Exception as exc:  # every failure must reach the client
            logger.exception("recognizer failed; closing session")
            await ws.send(ErrorFrame(code=ErrorCode.INTERNAL, message=str(exc)).to_json())
            await ws.close()
            raise _SessionClosed from exc
        for hypo in hypos:
            await self._send_hypothesis(ws, hypo, options, audio_s)

    async def _finish(
        self,
        ws: ServerConnection,
        recognizer: Recognizer,
        options: SessionOptions,
        audio_s: float,
    ) -> None:
        """Flush the recognizer, send the remaining hypotheses, and close cleanly."""
        try:
            hypos = recognizer.finalize()
        except VerbatimError as exc:
            await ws.send(ErrorFrame(code=exc.code, message=str(exc)).to_json())
            await ws.close()
            raise _SessionClosed from exc
        except Exception as exc:  # every failure must reach the client
            logger.exception("recognizer finalize failed; closing session")
            await ws.send(ErrorFrame(code=ErrorCode.INTERNAL, message=str(exc)).to_json())
            await ws.close()
            raise _SessionClosed from exc
        for hypo in hypos:
            await self._send_hypothesis(ws, hypo, options, audio_s)
        await ws.close()

    async def _handle(self, ws: ServerConnection) -> None:
        target = ws.request.path
        _, _, raw_query = target.partition("?")
        try:
            options = parse_query(raw_query)
        except InvalidArgument as exc:
            # Never a crash, never a silent substitution: one error frame, clean close.
            await ws.send(ErrorFrame(code=ErrorCode.INVALID_ARGUMENT, message=str(exc)).to_json())
            await ws.close()
            return
        params = dict(parse_qsl(raw_query, keep_blank_values=True))
        if "chunk_ms" not in params and self._config.default_chunk_ms != options.chunk_ms:
            options = replace(options, chunk_ms=self._config.default_chunk_ms)

        session_id = uuid.uuid4().hex[:16]
        self._live += 1
        self._total += 1
        try:
            await ws.send(
                SessionFrame(
                    id=session_id,
                    chunk_ms=options.chunk_ms,
                    invariance_class=self._config.invariance_class,
                ).to_json()
            )
            try:
                recognizer = self._factory(options)
            except VerbatimError as exc:
                await ws.send(ErrorFrame(code=exc.code, message=str(exc)).to_json())
                await ws.close()
                return
            except Exception as exc:  # every failure must reach the client
                logger.exception("recognizer factory failed; closing session")
                await ws.send(ErrorFrame(code=ErrorCode.INTERNAL, message=str(exc)).to_json())
                await ws.close()
                return

            buffer = bytearray()
            cap_bytes = self._cap_bytes
            real_bytes = 0  # real bytes handed to the recognizer; the audio clock
            while True:
                # Back-pressure: stop reading this socket while the buffer is over
                # its cap instead of discarding; drain first, read after.
                while len(buffer) > cap_bytes and len(buffer) >= options.chunk_bytes:
                    chunk = bytes(buffer[: options.chunk_bytes])
                    del buffer[: options.chunk_bytes]
                    real_bytes += options.chunk_bytes
                    await self._consume_chunk(
                        ws, recognizer, chunk, options, real_bytes / 2 / SAMPLE_RATE_HZ
                    )
                    await asyncio.sleep(0)
                try:
                    message = await ws.recv()
                except ConnectionClosed:
                    # Aborted without `end`: drop the session, emit nothing, and
                    # never call finalize() -- an aborted session has no final.
                    return
                if isinstance(message, bytes):
                    buffer += message
                    while len(buffer) >= options.chunk_bytes:
                        chunk = bytes(buffer[: options.chunk_bytes])
                        del buffer[: options.chunk_bytes]
                        real_bytes += options.chunk_bytes
                        await self._consume_chunk(
                            ws, recognizer, chunk, options, real_bytes / 2 / SAMPLE_RATE_HZ
                        )
                else:
                    try:
                        parse_client_text(message)
                    except InvalidArgument as exc:
                        await ws.send(
                            ErrorFrame(code=ErrorCode.INVALID_ARGUMENT, message=str(exc)).to_json()
                        )
                        continue
                    if len(buffer):
                        # The tail completes the chunk grid: pad it, pass it as
                        # the final chunk, and stamp what comes back with the
                        # clock including the tail's real bytes.
                        tail = bytes(buffer)
                        buffer.clear()
                        padded = tail + b"\x00" * (options.chunk_bytes - len(tail))
                        real_bytes += len(tail)
                        clock_s = real_bytes / 2 / SAMPLE_RATE_HZ
                        await self._consume_chunk(ws, recognizer, padded, options, clock_s)
                        await self._finish(ws, recognizer, options, clock_s)
                    else:
                        clock_s = real_bytes / 2 / SAMPLE_RATE_HZ
                        await self._finish(ws, recognizer, options, clock_s)
                    return
        except _SessionClosed:
            pass
        except ConnectionClosed:
            pass
        except Exception as exc:  # a session never dies unexplained
            logger.exception("unexpected session failure; closing session")
            with contextlib.suppress(ConnectionClosed):
                await ws.send(ErrorFrame(code=ErrorCode.INTERNAL, message=str(exc)).to_json())
            with contextlib.suppress(ConnectionClosed):
                await ws.close()
        finally:
            self._live -= 1
