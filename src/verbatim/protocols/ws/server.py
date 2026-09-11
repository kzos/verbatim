# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Plain WebSocket server for the demo: binary PCM in, JSON out, on the engine.

``ws://host:8080/v1/stream?chunk_ms=160&lang=en-US&words=1``. The same listener
serves the demo page and the health endpoints. The day-21 prototype row is taken
over this surface, because the Riva subset is month-2 work; the row's ``surface``
field says so.

The transport owns nothing about recognition. A connection opens one engine
session, feeds it whatever bytes arrive, and forwards the hypotheses the engine
delivers, each stamped with the engine's own audio clock. Chunk boundaries are cut
by the session's ring from its sample counter, so the chunk sequence is a pure
function of the audio and never of packetisation or arrival time.

Refusal comes before any acknowledgement: at capacity the client gets one ``error``
frame and a clean close, and never a ``session`` frame.

Back-pressure is live. ``feed`` returns short only when the session's ring is full;
the reader then keeps the remainder and stops reading the socket until the next
tick has drained the ring, so the peer's own flow control slows the client down.
Audio is never dropped server-side. One message larger than ``max_message_bytes``
is refused by the framing layer with close code 1009 before its payload is read;
no error frame can precede that, because sending one would mean reading the
payload the cap exists to refuse.
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

from verbatim.core.errors import ErrorCode, InvalidArgument, ResourceExhausted, VerbatimError
from verbatim.protocols.base import SAMPLE_RATE_HZ, EngineHandle, SessionHandle, SessionOptions
from verbatim.protocols.ws.frames import (
    ErrorFrame,
    FinalFrame,
    PartialFrame,
    SessionFrame,
    parse_client_text,
    parse_query,
)

__all__ = ["DEFAULT_MAX_MESSAGE_BYTES", "WsServer", "WsServerConfig"]

logger = logging.getLogger(__name__)

#: The largest single binary message accepted: three seconds of PCM16 at 16 kHz,
#: the default ring. A message the ring cannot hold at once is fed over several
#: ticks from memory, so this cap bounds what one connection can make the server
#: hold for it.
DEFAULT_MAX_MESSAGE_BYTES = 3 * SAMPLE_RATE_HZ * 2


@dataclass(frozen=True, slots=True)
class WsServerConfig:
    """Listener configuration. `invariance_class` stays the `<unmeasured>`
    placeholder until a measured value exists; never fabricate one."""

    host: str = "127.0.0.1"
    port: int = 8080
    path: str = "/v1/stream"
    invariance_class: str = "<unmeasured>"  # a placeholder, never a fabricated value
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES

    def __post_init__(self) -> None:
        if self.max_message_bytes < 2:
            raise ValueError(
                f"invalid max_message_bytes {self.max_message_bytes!r}: must hold one sample"
            )


class WsServer:
    """The plain WebSocket demo surface over an engine. Binary PCM in, JSON out.

    The server does not own the engine: one engine may sit behind both surfaces,
    so starting and stopping it is the caller's.

    Usage:
        async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
            ...  server.endpoint -> "ws://127.0.0.1:<bound port>/v1/stream"
    """

    def __init__(self, engine: EngineHandle, config: WsServerConfig | None = None) -> None:
        self._engine = engine
        self._config = config if config is not None else WsServerConfig()
        self._server: Server | None = None
        self._handlers: set[asyncio.Task[None]] = set()
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

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle,
            self._config.host,
            self._config.port,
            process_request=self._process_request,
            max_size=self._config.max_message_bytes,
        )

    async def stop(self, grace: float = 5.0) -> None:
        """Close the listener and every connection, then wait up to ``grace`` seconds
        for the handlers to return; whatever is still running after that is cancelled.

        A handler ends when its session's result stream ends, which the engine
        guarantees. The bound is for the day it does not: a listener whose shutdown
        waits on every session ending by itself would hold the process open past
        SIGTERM, and a test of such an engine would hang instead of fail.
        """
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=grace)
        except TimeoutError:
            logger.warning(
                "%d session handler(s) still running %.1f s after close; cancelling",
                len(self._handlers),
                grace,
            )
            for task in list(self._handlers):
                task.cancel()
            await asyncio.wait_for(server.wait_closed(), timeout=grace)

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

    @staticmethod
    async def _refuse(ws: ServerConnection, code: ErrorCode, message: str) -> None:
        """One error frame and a clean close: never a crash, never a silent substitution."""
        with contextlib.suppress(ConnectionClosed):
            await ws.send(ErrorFrame(code=code, message=message).to_json())
        with contextlib.suppress(ConnectionClosed):
            await ws.close()

    async def _handle(self, ws: ServerConnection) -> None:
        _, _, raw_query = ws.request.path.partition("?")
        try:
            options = parse_query(raw_query)
        except InvalidArgument as exc:
            await self._refuse(ws, ErrorCode.INVALID_ARGUMENT, str(exc))
            return
        params = dict(parse_qsl(raw_query, keep_blank_values=True))
        if "chunk_ms" not in params:
            # No mode asked for: the engine's. A mode that was asked for goes to the
            # engine as is, and an engine serving another refuses it by name.
            options = replace(options, chunk_ms=self._engine.chunk_ms)

        # Admission first: a refusal is one error frame and a close, with no session
        # frame, so a client never holds an acknowledgement for a session it has not got.
        try:
            session = self._engine.open_session(options)
        except ResourceExhausted as exc:
            await self._refuse(ws, exc.code, f"{exc}; retry after {exc.retry_after_ms} ms")
            return
        except VerbatimError as exc:
            await self._refuse(ws, exc.code, str(exc))
            return

        self._live += 1
        self._total += 1
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        writer: asyncio.Task[None] | None = None
        try:
            await ws.send(
                SessionFrame(
                    id=uuid.uuid4().hex[:16],
                    chunk_ms=options.chunk_ms,
                    invariance_class=self._config.invariance_class,
                ).to_json()
            )
            writer = asyncio.create_task(self._write_results(ws, session, options))
            await self._read_audio(ws, session)
            # end() or abort() has been called, so the engine's results end after the
            # drain frame and the writer closes the socket when it has sent the last.
            await writer
        except ConnectionClosed:
            pass
        except asyncio.CancelledError:
            # The listener gave up waiting for this session to end (see `stop`).
            session.abort()
            raise
        except Exception as exc:  # a session never dies unexplained
            logger.exception("unexpected session failure; closing session")
            session.abort()
            await self._refuse(ws, ErrorCode.INTERNAL, str(exc))
        finally:
            if writer is not None and not writer.done():
                writer.cancel()
            if task is not None:
                self._handlers.discard(task)
            self._live -= 1

    async def _read_audio(self, ws: ServerConnection, session: SessionHandle) -> None:
        """Feed the socket into the session until the client ends or leaves.

        Returns only after `end()` or `abort()` has been called on the session, so the
        caller can wait for the results to run out.
        """
        while True:
            try:
                message = await ws.recv()
            except ConnectionClosed:
                # Gone without `end`: drop the session and emit no final.
                session.abort()
                return
            if isinstance(message, bytes):
                pending = bytes(message)
                try:
                    while pending:
                        accepted = session.feed(pending)
                        pending = pending[accepted:]
                        if pending:
                            # The ring is full. Hold the remainder and do not read the
                            # socket again until a tick has drained it: live
                            # back-pressure, through the peer's own flow control.
                            await self._engine.wait_for_ticks(1)
                except (VerbatimError, RuntimeError):
                    # The session is over from the engine's side: a step failure
                    # (INTERNAL), the idle deadline (DEADLINE_EXCEEDED) or the engine
                    # stopping under a parked reader (UNAVAILABLE). The engine has
                    # already queued that outcome and the writer tells the client.
                    # Aborting here would race the thread's last tick into a clean
                    # close that reads as a finished utterance.
                    return
            else:
                try:
                    parse_client_text(message)
                except InvalidArgument as exc:
                    await ws.send(
                        ErrorFrame(code=ErrorCode.INVALID_ARGUMENT, message=str(exc)).to_json()
                    )
                    continue
                session.end()
                return

    @staticmethod
    async def _write_results(
        ws: ServerConnection, session: SessionHandle, options: SessionOptions
    ) -> None:
        """Forward every hypothesis the engine delivers, then close cleanly.

        The engine stamps `audio_processed_s`; the transport never keeps a clock of
        its own, because bytes received are not bytes recognised once a ring sits
        between the socket and the pipeline.

        A normal end closes with 1000. When the engine is going away under a live
        session it raises `UNAVAILABLE`; that closes with 1001 (going away), the code
        a client reads as "the server left", not "your utterance finished".
        """
        close_code = 1000
        try:
            async for hypothesis in session.results():
                if hypothesis.is_final:
                    words = tuple(hypothesis.words) if options.word_timestamps else None
                    await ws.send(
                        FinalFrame(
                            text=hypothesis.text,
                            audio_s=hypothesis.audio_processed_s,
                            words=words,
                        ).to_json()
                    )
                elif options.interim_results:
                    await ws.send(
                        PartialFrame(
                            text=hypothesis.text, audio_s=hypothesis.audio_processed_s
                        ).to_json()
                    )
        except VerbatimError as exc:
            # The client is always told why a session dies.
            if exc.code is ErrorCode.UNAVAILABLE:
                close_code = 1001
            with contextlib.suppress(ConnectionClosed):
                await ws.send(ErrorFrame(code=exc.code, message=str(exc)).to_json())
        except ConnectionClosed:
            return
        with contextlib.suppress(ConnectionClosed):
            await ws.close(close_code)
