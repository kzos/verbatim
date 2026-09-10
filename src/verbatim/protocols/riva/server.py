# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``grpc.aio`` servicer for ``nvidia.riva.asr.RivaSpeechRecognition``.

Implemented: ``StreamingRecognize`` (the whole product) and
``GetRivaSpeechRecognitionConfig`` (LiveKit's ``log_asr_models`` calls it at setup;
omitting it breaks the plugin). ``Recognize`` (unary) returns ``UNIMPLEMENTED``
in year 1 -- Verbatim is a streaming server.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Final

import grpc

from verbatim.core.errors import ErrorCode, VerbatimError
from verbatim.protocols.base import SAMPLE_RATE_HZ, Hypothesis, Recognizer, RecognizerFactory
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
    ErrorCode.INTERNAL: grpc.StatusCode.INTERNAL,
}


@dataclass(frozen=True, slots=True)
class RivaServerConfig:
    """Listener configuration. Port 0 binds an ephemeral port for tests."""

    host: str = "0.0.0.0"
    port: int = 50051
    served_models: tuple[str, ...] = ("verbatim-stub",)
    default_chunk_ms: int = 160
    language_code: str = "en-US"
    max_concurrent_streams: int | None = None


async def _abort(context: grpc.aio.ServicerContext, exc: BaseException) -> None:
    """End the RPC with a real gRPC status: never a stack trace, never a silent hang."""
    if isinstance(exc, VerbatimError):
        code = _STATUS_BY_CODE[exc.code]
    else:
        logger.exception("recognizer failed; aborting stream")
        code = grpc.StatusCode.INTERNAL
    await context.abort(code, str(exc))


class RivaSpeechRecognitionServicer(riva_asr_pb2_grpc.RivaSpeechRecognitionServicer):
    """The Riva subset: streaming recognition plus the model-config lookup."""

    def __init__(self, factory: RecognizerFactory, config: RivaServerConfig) -> None:
        self._factory = factory
        self._config = config

    def _served(self) -> list[tuple[str, int, str]]:
        return [
            (name, self._config.default_chunk_ms, self._config.language_code)
            for name in self._config.served_models
        ]

    async def _emit(
        self,
        session: RivaSessionConfig,
        recognizer: Recognizer,
        hypos: list[Hypothesis],
        audio_s: float,
    ) -> AsyncIterator[riva_asr_pb2.StreamingRecognizeResponse]:
        """Yield one response per hypothesis, stamped with the session audio clock."""
        for hypo in hypos:
            if hypo.audio_processed_s == audio_s:
                stamped = hypo
            else:
                stamped = replace(hypo, audio_processed_s=audio_s)
            if stamped.is_final:
                yield final_response(
                    stamped,
                    request_id=session.request_id,
                    word_timestamps=session.options.word_timestamps,
                )
            elif session.options.interim_results:
                yield partial_response(stamped, request_id=session.request_id)

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[riva_asr_pb2.StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[riva_asr_pb2.StreamingRecognizeResponse]:
        """Config first, then PCM16 chunks; half-close flushes the final."""
        session: RivaSessionConfig | None = None
        recognizer: Recognizer | None = None
        buffer = bytearray()
        real_bytes = 0  # real bytes handed to the recognizer; the audio clock
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
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "invalid StreamingRecognizeRequest.streaming_config: "
                            "a second streaming_config on the same stream",
                        )
                    try:
                        session = options_from_config(
                            request.streaming_config.config,
                            interim_results=request.streaming_config.interim_results,
                            request_id=request.id.value,
                            served_models=self._config.served_models,
                            default_chunk_ms=self._config.default_chunk_ms,
                        )
                    except VerbatimError as exc:
                        await _abort(context, exc)
                        return
                    try:
                        recognizer = self._factory(session.options)
                    except VerbatimError as exc:
                        await _abort(context, exc)
                        return
                    except Exception as exc:
                        await _abort(context, exc)
                        return
                elif which == "audio_content":
                    if session is None or recognizer is None:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "invalid StreamingRecognizeRequest.audio_content: "
                            "audio before streaming_config",
                        )
                    buffer += request.audio_content
                    chunk_bytes = session.options.chunk_bytes
                    while len(buffer) >= chunk_bytes:
                        chunk = bytes(buffer[:chunk_bytes])
                        del buffer[:chunk_bytes]
                        real_bytes += chunk_bytes
                        audio_s = real_bytes / 2 / SAMPLE_RATE_HZ
                        try:
                            hypos = recognizer.add_chunk(chunk)
                        except VerbatimError as exc:
                            await _abort(context, exc)
                            return
                        except Exception as exc:
                            await _abort(context, exc)
                            return
                        async for response in self._emit(session, recognizer, hypos, audio_s):
                            yield response
                # Unknown oneof branches are ignored per proto3 semantics.
            if session is None or recognizer is None:
                return
            if buffer:
                tail = bytes(buffer)
                buffer.clear()
                padded = tail + b"\x00" * (session.options.chunk_bytes - len(tail))
                real_bytes += len(tail)
                clock_s = real_bytes / 2 / SAMPLE_RATE_HZ
                try:
                    hypos = recognizer.add_chunk(padded)
                except VerbatimError as exc:
                    await _abort(context, exc)
                    return
                except Exception as exc:
                    await _abort(context, exc)
                    return
                # The padded tail completes the chunk grid: what comes back is
                # stamped with the clock including the tail's real bytes, never
                # the bytes the server invented for padding.
                async for response in self._emit(session, recognizer, hypos, clock_s):
                    yield response
            else:
                clock_s = real_bytes / 2 / SAMPLE_RATE_HZ
            try:
                flushed = recognizer.finalize()
            except VerbatimError as exc:
                await _abort(context, exc)
                return
            except Exception as exc:
                await _abort(context, exc)
                return
            async for response in self._emit(session, recognizer, flushed, clock_s):
                yield response
        except asyncio.CancelledError:
            # Client went away mid-stream: drop the session, emit nothing, and let
            # the cancellation propagate so the server stays usable.
            raise

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

    Usage:
        async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as s:
            ...  s.target -> "127.0.0.1:<bound port>"
    """

    def __init__(self, factory: RecognizerFactory, config: RivaServerConfig | None = None) -> None:
        self._factory = factory
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
        servicer = RivaSpeechRecognitionServicer(self._factory, self._config)
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
