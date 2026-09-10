# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``grpc.aio`` servicer for ``nvidia.riva.asr.RivaSpeechRecognition``, on the engine.

Implemented: ``StreamingRecognize`` (the whole product) and
``GetRivaSpeechRecognitionConfig`` (LiveKit's ``log_asr_models`` calls it at setup;
omitting it breaks the plugin). ``Recognize`` (unary) returns ``UNIMPLEMENTED``
in year 1 -- Verbatim is a streaming server.

One RPC is one engine session. A reader task consumes the request stream: the
first message's config opens the session, every ``audio_content`` is fed with live
back-pressure, and the client's half-close ends the session. The handler itself
forwards the engine's hypotheses as responses, stamped with the engine's own audio
clock. Refusal at capacity is ``RESOURCE_EXHAUSTED`` with a ``retry-after-ms``
trailer and no response written.

A client cancel reaches ``grpc.aio`` as a half-close followed by cancellation of
the handler: the request iterator ends normally and ``context.cancelled()`` is
still false at that moment (probed, not assumed). So the reader ends the session,
the cancelled handler then aborts it, and the abort wins for whatever the ring
still held: no further audio of a departed client is recognised.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final

import grpc

from verbatim.core.errors import ErrorCode, ResourceExhausted, VerbatimError
from verbatim.protocols.base import EngineHandle, SessionHandle
from verbatim.protocols.riva._gen import riva_asr_pb2, riva_asr_pb2_grpc
from verbatim.protocols.riva.mapping import (
    RivaSessionConfig,
    config_response,
    final_response,
    options_from_config,
    partial_response,
)

__all__ = ["RivaServer", "RivaServerConfig", "RivaSpeechRecognitionServicer"]

logger = logging.getLogger(__name__)

_STATUS_BY_CODE: Final = {
    ErrorCode.INVALID_ARGUMENT: grpc.StatusCode.INVALID_ARGUMENT,
    ErrorCode.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    ErrorCode.UNIMPLEMENTED: grpc.StatusCode.UNIMPLEMENTED,
    ErrorCode.RESOURCE_EXHAUSTED: grpc.StatusCode.RESOURCE_EXHAUSTED,
    ErrorCode.DEADLINE_EXCEEDED: grpc.StatusCode.DEADLINE_EXCEEDED,
    ErrorCode.UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    ErrorCode.INTERNAL: grpc.StatusCode.INTERNAL,
}

#: Trailer carrying the admission controller's retry hint on a refusal.
RETRY_AFTER_TRAILER: Final = "retry-after-ms"


@dataclass(frozen=True, slots=True)
class RivaServerConfig:
    """Listener configuration. Port 0 binds an ephemeral port for tests. Admission
    and the chunk mode are the engine's, so neither is repeated here."""

    host: str = "0.0.0.0"
    port: int = 50051
    served_models: tuple[str, ...] = ("verbatim-stub",)
    language_code: str = "en-US"


class _ProtocolError(Exception):
    """The client broke the request grammar: INVALID_ARGUMENT."""


_Opened = tuple[RivaSessionConfig, SessionHandle] | None


async def _abort(context: grpc.aio.ServicerContext, exc: BaseException) -> None:
    """End the RPC with a real gRPC status: never a stack trace, never a silent hang."""
    if isinstance(exc, ResourceExhausted):
        await context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            str(exc),
            trailing_metadata=((RETRY_AFTER_TRAILER, str(exc.retry_after_ms)),),
        )
    if isinstance(exc, VerbatimError):
        code = _STATUS_BY_CODE[exc.code]
    elif isinstance(exc, _ProtocolError):
        code = grpc.StatusCode.INVALID_ARGUMENT
    else:
        logger.exception("session failed; aborting stream")
        code = grpc.StatusCode.INTERNAL
    await context.abort(code, str(exc))


class RivaSpeechRecognitionServicer(riva_asr_pb2_grpc.RivaSpeechRecognitionServicer):
    """The Riva subset: streaming recognition plus the model-config lookup."""

    def __init__(self, engine: EngineHandle, config: RivaServerConfig) -> None:
        self._engine = engine
        self._config = config

    def _served(self) -> list[tuple[str, int, str]]:
        return [
            (name, self._engine.chunk_ms, self._config.language_code)
            for name in self._config.served_models
        ]

    async def _read_requests(
        self,
        request_iterator: AsyncIterator[riva_asr_pb2.StreamingRecognizeRequest],
        opened: asyncio.Future[_Opened],
    ) -> None:
        """Consume the request stream: config opens the session, audio feeds it with
        live back-pressure, half-close ends it. Errors before the session is open land
        on `opened`; errors after it abort the session and propagate to the awaiter."""
        session: RivaSessionConfig | None = None
        handle: SessionHandle | None = None
        try:
            async for request in request_iterator:
                if "force_eou" in request.runtime_config:
                    logger.warning(
                        "runtime_config force_eou is accepted and ignored: "
                        "honouring it waits on an upstream change"
                    )
                which = request.WhichOneof("streaming_request")
                if which == "streaming_config":
                    if session is not None:
                        raise _ProtocolError(
                            "invalid StreamingRecognizeRequest.streaming_config: "
                            "a second streaming_config on the same stream"
                        )
                    session = options_from_config(
                        request.streaming_config.config,
                        interim_results=request.streaming_config.interim_results,
                        request_id=request.id.value,
                        served_models=self._config.served_models,
                        default_chunk_ms=self._engine.chunk_ms,
                    )
                    handle = self._engine.open_session(session.options)
                    opened.set_result((session, handle))
                elif which == "audio_content":
                    if handle is None:
                        raise _ProtocolError(
                            "invalid StreamingRecognizeRequest.audio_content: "
                            "audio before streaming_config"
                        )
                    pending = bytes(request.audio_content)
                    try:
                        while pending:
                            accepted = handle.feed(pending)
                            pending = pending[accepted:]
                            if pending:
                                # The ring is full: hold the remainder and stop reading
                                # the request stream until a tick has drained it, so
                                # HTTP/2 flow control slows the client down.
                                await self._engine.wait_for_ticks(1)
                    except (VerbatimError, RuntimeError):
                        # The session is over from the engine's side: a step failure
                        # (INTERNAL), the idle deadline (DEADLINE_EXCEEDED) or the
                        # engine stopping under a parked reader (UNAVAILABLE). The
                        # engine has already queued that outcome on the result stream.
                        # Aborting here would race the thread's last tick into a clean
                        # end that reads as a finished utterance.
                        return
                # Unknown oneof branches are ignored per proto3 semantics.
            if handle is None:
                opened.set_result(None)
            else:
                handle.end()
        except asyncio.CancelledError:
            if not opened.done():
                opened.cancel()
            elif handle is not None:
                handle.abort()
            raise
        except Exception as exc:
            if not opened.done():
                # Delivered through `opened`; the handler reports it. Not re-raised,
                # so the task does not also end with an exception nobody retrieves.
                opened.set_exception(exc)
                return
            if handle is not None:
                handle.abort()
            raise

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[riva_asr_pb2.StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[riva_asr_pb2.StreamingRecognizeResponse]:
        """Config first, then PCM16 chunks; half-close flushes the final."""
        opened: asyncio.Future[_Opened] = asyncio.get_running_loop().create_future()
        reader = asyncio.create_task(self._read_requests(request_iterator, opened))
        handle: SessionHandle | None = None
        try:
            try:
                started = await opened
            except (VerbatimError, _ProtocolError) as exc:
                await _abort(context, exc)
                return
            if started is None:
                return  # half-closed without a config: nothing to say
            session, handle = started
            try:
                async for hypothesis in handle.results():
                    if hypothesis.is_final:
                        yield final_response(
                            hypothesis,
                            request_id=session.request_id,
                            word_timestamps=session.options.word_timestamps,
                        )
                    elif session.options.interim_results:
                        yield partial_response(hypothesis, request_id=session.request_id)
            except VerbatimError as exc:
                await _abort(context, exc)
                return
            try:
                await reader
            except (VerbatimError, _ProtocolError) as exc:
                await _abort(context, exc)
                return
        except asyncio.CancelledError:
            # Client went away mid-stream: drop the session, emit nothing, and let
            # the cancellation propagate so the server stays usable.
            if handle is not None:
                handle.abort()
            raise
        finally:
            if not reader.done():
                reader.cancel()
            elif not reader.cancelled():
                # A reader that failed after the session died has already been acted
                # on; retrieving its exception keeps the loop's logs honest.
                reader.exception()

    async def GetRivaSpeechRecognitionConfig(
        self,
        request: riva_asr_pb2.RivaSpeechRecognitionConfigRequest,
        context: grpc.aio.ServicerContext,
    ) -> riva_asr_pb2.RivaSpeechRecognitionConfigResponse:
        """One Config per served mode; what LiveKit's log_asr_models() reads."""
        try:
            return config_response(self._served(), model_name_filter=request.model_name)
        except VerbatimError as exc:
            await _abort(context, exc)
            raise AssertionError("unreachable") from exc

    async def Recognize(
        self,
        request: riva_asr_pb2.RecognizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> riva_asr_pb2.RecognizeResponse:
        """UNIMPLEMENTED. Verbatim is a streaming server; point the caller at NeMo's
        offline path rather than pretending to have a batch surface."""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Recognize (unary) is not implemented: Verbatim is a streaming server; "
            "use StreamingRecognize, or NeMo's offline transcription path for batch audio",
        )
        raise AssertionError("unreachable")


class RivaServer:
    """Wraps grpc.aio.server(). Binds port 0 when config.port == 0 (tests).

    The server does not own the engine: one engine may sit behind both surfaces,
    so starting and stopping it is the caller's.

    Usage:
        async with engine, RivaServer(engine, RivaServerConfig(port=0)) as s:
            ...  s.target -> "127.0.0.1:<bound port>"
    """

    def __init__(self, engine: EngineHandle, config: RivaServerConfig | None = None) -> None:
        self._engine = engine
        self._config = config if config is not None else RivaServerConfig()
        self._server: grpc.aio.Server | None = None
        self._bound_port: int | None = None

    @property
    def target(self) -> str:
        """Dial this: 127.0.0.1 plus the bound port."""
        return f"127.0.0.1:{self.port}"

    @property
    def port(self) -> int:
        """The bound port; useful when config.port == 0."""
        if self._bound_port is not None:
            return self._bound_port
        return self._config.port

    async def start(self) -> None:
        if self._server is not None:
            return
        server = grpc.aio.server()
        servicer = RivaSpeechRecognitionServicer(self._engine, self._config)
        riva_asr_pb2_grpc.add_RivaSpeechRecognitionServicer_to_server(servicer, server)
        self._bound_port = server.add_insecure_port(f"{self._config.host}:{self._config.port}")
        await server.start()
        self._server = server

    async def stop(self, grace: float = 0.5) -> None:
        server, self._server = self._server, None
        if server is not None:
            await server.stop(grace)

    async def __aenter__(self) -> RivaServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
