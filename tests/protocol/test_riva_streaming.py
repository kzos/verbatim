# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The Riva subset over grpc.aio on 127.0.0.1, over a real engine.

Every test binds port 0 so tests never collide on a fixed port. The `_collect`
helper opens a channel, sends a config message then N chunks of silence, then
half-closes, and gathers every response. Every RPC is one engine session: what
these tests pin is the wire behaviour of the engine path, refusal at capacity
before any response, live back-pressure and the idle deadline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from itertools import pairwise

import grpc
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.types import PcmFrame, StepResult
from verbatim.engine import Engine, stub_engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.base import EngineHandle, Hypothesis, SessionHandle, SessionOptions
from verbatim.protocols.riva._gen import riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2
from verbatim.protocols.riva.server import RETRY_AFTER_TRAILER, RivaServer, RivaServerConfig
from verbatim.scheduler.clock import ScaledMonotonicClock

pytestmark = pytest.mark.cpu

ENC = riva_audio_pb2.AudioEncoding
FULL_SCRIPT = "the quick brown fox jumps over the lazy dog"
CHUNK = ChunkMode(160)


def _chunk(chunk_ms: int = 160) -> bytes:
    return b"\x00" * (chunk_ms * 32)


class _RecordingSession(SessionHandle):
    def __init__(self, inner: SessionHandle, log: list[tuple[str, int]]) -> None:
        self._inner = inner
        self._log = log

    @property
    def options(self) -> SessionOptions:
        return self._inner.options

    def feed(self, pcm: bytes) -> int:
        accepted = self._inner.feed(pcm)
        self._log.append(("feed", accepted))
        return accepted

    def end(self) -> None:
        self._log.append(("end", 0))
        self._inner.end()

    def abort(self) -> None:
        self._log.append(("abort", 0))
        self._inner.abort()

    def results(self) -> AsyncIterator[Hypothesis]:
        return self._inner.results()


class _RecordingEngine(EngineHandle):
    def __init__(self, inner: Engine) -> None:
        self.inner = inner
        self.log: list[tuple[str, int]] = []

    @property
    def chunk_ms(self) -> int:
        return self.inner.chunk_ms

    def open_session(self, options: SessionOptions) -> SessionHandle:
        return _RecordingSession(self.inner.open_session(options), self.log)

    async def wait_for_ticks(self, n: int) -> None:
        self.log.append(("wait", n))
        await self.inner.wait_for_ticks(n)

    def calls(self, name: str) -> list[int]:
        return [value for kind, value in self.log if kind == name]


@asynccontextmanager
async def _server(
    *, engine: Engine | None = None, **engine_kwargs: object
) -> AsyncIterator[RivaServer]:
    inner = engine if engine is not None else stub_engine(**engine_kwargs)  # type: ignore[arg-type]
    async with inner, RivaServer(inner, RivaServerConfig(port=0)) as server:
        yield server


@asynccontextmanager
async def _recording_server(
    **engine_kwargs: object,
) -> AsyncIterator[tuple[RivaServer, _RecordingEngine]]:
    inner = stub_engine(**engine_kwargs)  # type: ignore[arg-type]
    engine = _RecordingEngine(inner)
    async with inner, RivaServer(engine, RivaServerConfig(port=0)) as server:
        yield server, engine


def _config_message(
    *,
    interim_results: bool = True,
    request_id: str = "",
    runtime_config: dict[str, str] | None = None,
    recognition: dict[str, object] | None = None,
) -> riva_asr_pb2.StreamingRecognizeRequest:
    fields: dict[str, object] = {
        "encoding": ENC.LINEAR_PCM,
        "sample_rate_hertz": 16000,
        "language_code": "en-US",
        "model": "",
        "max_alternatives": 1,
        "audio_channel_count": 1,
    }
    if recognition:
        fields.update(recognition)
    request = riva_asr_pb2.StreamingRecognizeRequest()
    for key, value in fields.items():
        setattr(request.streaming_config.config, key, value)
    request.streaming_config.interim_results = interim_results
    if request_id:
        request.id.value = request_id
    if runtime_config:
        for key, value in runtime_config.items():
            request.runtime_config[key] = value
    return request


def _audio_message(payload: bytes) -> riva_asr_pb2.StreamingRecognizeRequest:
    return riva_asr_pb2.StreamingRecognizeRequest(audio_content=payload)


async def _collect(
    target: str,
    requests: list[riva_asr_pb2.StreamingRecognizeRequest],
    *,
    metadata: list[tuple[str, str]] | None = None,
) -> list[riva_asr_pb2.StreamingRecognizeResponse]:
    async with grpc.aio.insecure_channel(target) as channel:
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

        async def gen():  # type: ignore[no-untyped-def]
            for request in requests:
                yield request

        call = stub.StreamingRecognize(gen(), metadata=metadata)
        return [response async for response in call]


async def _happy_collect(server: RivaServer, n: int = 5) -> list:
    chunks = [_audio_message(_chunk()) for _ in range(n)]
    return await _collect(server.target, [_config_message(), *chunks])


def _partials(
    responses: list[riva_asr_pb2.StreamingRecognizeResponse],
) -> list[riva_asr_pb2.StreamingRecognizeResponse]:
    return [r for r in responses if not r.results[0].is_final]


def _finals(
    responses: list[riva_asr_pb2.StreamingRecognizeResponse],
) -> list[riva_asr_pb2.StreamingRecognizeResponse]:
    return [r for r in responses if r.results[0].is_final]


async def test_recognize_unary_is_unimplemented() -> None:
    async with _server() as server, grpc.aio.insecure_channel(server.target) as channel:
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub.Recognize(riva_asr_pb2.RecognizeRequest())
    assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED


async def test_get_config_returns_online_models() -> None:
    async with _server() as server, grpc.aio.insecure_channel(server.target) as channel:
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        response = await stub.GetRivaSpeechRecognitionConfig(
            riva_asr_pb2.RivaSpeechRecognitionConfigRequest()
        )
    online = [c for c in response.model_config if c.parameters.get("type") == "online"]
    assert len(online) >= 1
    assert all(c.model_name for c in online)


async def test_streaming_happy_path() -> None:
    async with _server() as server:
        responses = await _happy_collect(server, n=5)
    partials = _partials(responses)
    finals = _finals(responses)
    assert len(partials) == 5
    assert all(not r.results[0].is_final for r in partials)
    assert [r.results[0].alternatives[0].transcript for r in partials] == [
        "the",
        "the quick",
        "the quick brown",
        "the quick brown fox",
        "the quick brown fox jumps",
    ]
    assert len(finals) == 1
    assert responses[-1].results[0].is_final is True
    assert finals[0].results[0].alternatives[0].transcript == FULL_SCRIPT
    times = [r.results[0].audio_processed for r in responses]
    assert all(b >= a for a, b in pairwise(times))


async def test_audio_before_config_is_invalid_argument() -> None:
    async with _server() as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [_audio_message(_chunk()), _config_message()])
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_second_config_message_is_invalid_argument() -> None:
    async with _recording_server() as (server, engine):
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [_config_message(), _config_message()])
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    # The session opened by the first config is aborted, not left to the deadline.
    assert engine.calls("abort") == [0]


async def test_interim_results_false_suppresses_partials() -> None:
    async with _server() as server:
        responses = await _collect(
            server.target,
            [_config_message(interim_results=False), *[_audio_message(_chunk()) for _ in range(3)]],
        )
    assert len(responses) == 1
    assert responses[0].results[0].is_final is True


async def test_request_id_is_echoed_on_every_response() -> None:
    async with _server() as server:
        responses = await _collect(
            server.target,
            [_config_message(request_id="abc123"), *[_audio_message(_chunk()) for _ in range(2)]],
        )
    assert len(responses) > 1
    assert all(r.id.value == "abc123" for r in responses)


async def test_no_request_id_gives_empty_string() -> None:
    async with _server() as server:
        responses = await _happy_collect(server, n=2)
    assert len(responses) > 0
    for response in responses:
        assert response.id.value == ""


async def test_unknown_metadata_is_ignored() -> None:
    async with _server() as server:
        responses = await _collect(
            server.target,
            [_config_message(), *[_audio_message(_chunk()) for _ in range(2)]],
            metadata=[("function-id", "test-function"), ("authorization", "Bearer nope")],
        )
    assert len(_finals(responses)) == 1
    assert _finals(responses)[0].results[0].alternatives[0].transcript == FULL_SCRIPT


async def test_force_eou_is_accepted_and_does_not_abort() -> None:
    async with _server() as server, grpc.aio.insecure_channel(server.target) as channel:
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

        async def gen():  # type: ignore[no-untyped-def]
            yield _config_message(runtime_config={"force_eou": "true"})
            yield _audio_message(_chunk())
            yield _audio_message(_chunk())

        call = stub.StreamingRecognize(gen())
        responses = [response async for response in call]
        assert await call.code() == grpc.StatusCode.OK
    assert len(_finals(responses)) == 1
    assert _finals(responses)[0].results[0].alternatives[0].transcript == FULL_SCRIPT


async def test_partial_audio_content_is_buffered_until_a_chunk_completes() -> None:
    async with _server() as server:
        tiny = [_audio_message(bytes(512)) for _ in range(10)]
        responses = await _collect(server.target, [_config_message(), *tiny])
    assert len(_partials(responses)) == 1
    assert len(_finals(responses)) == 1


async def test_trailing_partial_chunk_is_padded_but_only_real_bytes_are_counted() -> None:
    async with _server() as server:
        responses = await _collect(
            server.target,
            [
                _config_message(),
                _audio_message(_chunk()),
                _audio_message(_chunk()),
                _audio_message(bytes(1000)),
            ],
        )
    finals = _finals(responses)
    assert len(finals) == 1
    assert finals[0].results[0].audio_processed == pytest.approx(0.35125)


async def test_words_on_final_only_when_requested() -> None:
    async with _server() as server:
        with_words = await _collect(
            server.target,
            [
                _config_message(recognition={"enable_word_time_offsets": True}),
                _audio_message(_chunk()),
            ],
        )
        without_words = await _collect(
            server.target,
            [
                _config_message(recognition={"enable_word_time_offsets": False}),
                _audio_message(_chunk()),
            ],
        )
    finals = _finals(with_words)
    assert len(finals) == 1
    assert len(finals[0].results[0].alternatives[0].words) > 0
    assert all(len(r.results[0].alternatives[0].words) == 0 for r in _partials(with_words))
    assert all(len(r.results[0].alternatives[0].words) == 0 for r in without_words)


async def test_unknown_model_aborts_with_not_found() -> None:
    async with _server() as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [_config_message(recognition={"model": "nope"})])
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


async def test_bad_encoding_aborts_with_invalid_argument() -> None:
    async with _server() as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(
                server.target,
                [_config_message(recognition={"encoding": ENC.ENCODING_UNSPECIFIED})],
            )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "encoding" in excinfo.value.details()


async def test_diarization_aborts_with_unimplemented() -> None:
    async with _server() as server:
        request = _config_message()
        request.streaming_config.config.diarization_config.enable_speaker_diarization = True
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [request])
    assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED


async def test_concurrent_streams_do_not_interfere() -> None:
    scripts = {f"lang-{i}": (f"alpha{i}", f"beta{i}", f"gamma{i}") for i in range(8)}

    def script_for(options: SessionOptions) -> Sequence[str]:
        return scripts[options.language_code]

    async with _server(script_for=script_for) as server:

        async def run_stream(i: int) -> str:
            responses = await _collect(
                server.target,
                [
                    _config_message(recognition={"language_code": f"lang-{i}"}),
                    *[_audio_message(_chunk()) for _ in range(3)],
                ],
            )
            finals = _finals(responses)
            assert len(finals) == 1
            return finals[0].results[0].alternatives[0].transcript

        texts = await asyncio.gather(*(run_stream(i) for i in range(8)))
    for i, text in enumerate(texts):
        assert text == " ".join(scripts[f"lang-{i}"])


async def test_client_cancel_aborts_the_session_and_the_server_stays_usable() -> None:
    """A cancel reaches grpc.aio as a half-close then a cancellation, so the reader
    may record `end` first; what matters is that `abort` follows and the engine keeps
    nothing of the departed client, and that the next stream is served."""
    async with (
        _recording_server() as (server, engine),
        grpc.aio.insecure_channel(server.target) as channel,
    ):
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

        async def gen():  # type: ignore[no-untyped-def]
            yield _config_message()
            yield _audio_message(_chunk())
            await asyncio.sleep(30)

        call = stub.StreamingRecognize(gen())
        first = await call.read()
        assert first.results[0].is_final is False
        call.cancel()
        for _ in range(500):
            if engine.calls("abort"):
                break
            await asyncio.sleep(0.01)
        assert engine.calls("abort") == [0]
        for _ in range(500):
            if not engine.inner._sessions:
                break
            await asyncio.sleep(0.01)
        assert engine.inner._sessions == {}
        responses = await _collect(server.target, [_config_message(), _audio_message(_chunk())])
    assert len(_finals(responses)) == 1


async def test_refusal_at_capacity_is_resource_exhausted_with_a_retry_trailer() -> None:
    """The ninth stream on the default bucket of eight is refused before any response
    is written: RESOURCE_EXHAUSTED, and the admission controller's retry hint rides
    in the `retry-after-ms` trailer."""
    async with _server() as server, grpc.aio.insecure_channel(server.target) as channel:
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        holds: list[asyncio.Future[None]] = []

        def held_stream():  # type: ignore[no-untyped-def]
            release: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            holds.append(release)

            async def gen():  # type: ignore[no-untyped-def]
                yield _config_message()
                yield _audio_message(_chunk())
                await release

            return stub.StreamingRecognize(gen())

        async def ninth():  # type: ignore[no-untyped-def]
            yield _config_message()

        calls = [held_stream() for _ in range(8)]
        try:
            # A held stream proves it was admitted by delivering its first partial;
            # probing before all eight have would let a probe take a slot ahead of a
            # held stream's config and be the one admitted instead.
            for call in calls:
                first = await call.read()
                assert first.results[0].is_final is False
            refused = stub.StreamingRecognize(ninth())
            with pytest.raises(grpc.aio.AioRpcError) as excinfo:
                await refused.read()
            assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert "session refused" in excinfo.value.details()
            trailers = dict(await refused.trailing_metadata())
            assert trailers[RETRY_AFTER_TRAILER] == "160"
        finally:
            for release in holds:
                release.set_result(None)
            for call in calls:
                # A streaming call's status resolves only once its responses are
                # read to EOF (grpc.aio client behaviour, probed): read, then check.
                while await call.read() is not grpc.aio.EOF:
                    pass
                assert await call.code() == grpc.StatusCode.OK


async def test_back_pressure_holds_the_remainder_and_drops_nothing() -> None:
    """One `audio_content` larger than the session's ring is fed over several ticks:
    short accepts, a one-tick wait between offers, and every sample recognised."""
    message = b"\x00" * (2 * 16000 * 2)  # two seconds against a half-second ring
    async with _recording_server(ring_seconds=0.5) as (server, engine):
        responses = await _collect(server.target, [_config_message(), _audio_message(message)])
    finals = _finals(responses)
    assert len(finals) == 1
    assert finals[0].results[0].audio_processed == pytest.approx(2.0)
    assert len(_partials(responses)) == 13  # 12 full chunks + the tail
    feeds = engine.calls("feed")
    assert sum(feeds) == len(message)
    assert feeds[0] < len(message)
    assert len(engine.calls("wait")) >= len(feeds) - 1


async def test_an_idle_stream_is_closed_with_deadline_exceeded() -> None:
    async with (
        _server(idle_timeout_s=0.5) as server,
        grpc.aio.insecure_channel(server.target) as channel,
    ):
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

        async def gen():  # type: ignore[no-untyped-def]
            yield _config_message()
            await asyncio.sleep(30)

        call = stub.StreamingRecognize(gen())
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await asyncio.wait_for(call.read(), timeout=5.0)
    assert excinfo.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "no audio" in excinfo.value.details()


class _Flaky(FakePipelineAdapter):
    """Raises inside the second step that carries a real (non-pad) row."""

    def __init__(self) -> None:
        super().__init__(CHUNK, buckets=(8,), scripted=True, script_for=lambda _o: None)
        self.real_steps = 0

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        if any(frame.stream_id > 0 for frame in frames):
            self.real_steps += 1
            if self.real_steps == 2:
                raise RuntimeError("boom")
        return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)


async def test_a_pipeline_exception_aborts_with_internal() -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(8,))
    engine = Engine(config, _Flaky(), clock=ScaledMonotonicClock(100.0))
    async with _server(engine=engine) as server:
        requests = [_config_message(), _audio_message(_chunk()), _audio_message(_chunk())]
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, requests)
    assert excinfo.value.code() == grpc.StatusCode.INTERNAL
    assert excinfo.value.details() == "boom"


class _StoppingEngine(_RecordingEngine):
    """The engine as a parked reader sees it once `stop()` has begun: the tick wait
    raises, and by then the engine has ended the session itself."""

    async def wait_for_ticks(self, n: int) -> None:
        self.log.append(("wait", n))
        await self.inner.stop()
        raise RuntimeError("engine tick loop is not running")


async def _read_to_end(
    call,  # type: ignore[no-untyped-def]
) -> tuple[list[riva_asr_pb2.StreamingRecognizeResponse], grpc.StatusCode, str]:
    """Responses, then the status: a streaming call's status resolves only once its
    responses are read to EOF, and a failed call may surface that as an error on read."""
    responses = []
    try:
        while (response := await asyncio.wait_for(call.read(), timeout=5.0)) is not grpc.aio.EOF:
            responses.append(response)
    except grpc.aio.AioRpcError as exc:
        return responses, exc.code(), exc.details() or ""
    return responses, await call.code(), (await call.details()) or ""


async def test_a_reader_parked_in_back_pressure_when_the_engine_stops_is_told_unavailable() -> None:
    """The reader's tick wait raises during shutdown. The reader does not abort the
    session, because the engine has already ended it with UNAVAILABLE and an abort
    could race the thread's last tick into a clean end; the call ends UNAVAILABLE with
    no final. A partial from audio the engine had already stepped may precede it."""
    inner = stub_engine(ring_seconds=0.5)
    engine = _StoppingEngine(inner)
    async with (
        inner,
        RivaServer(engine, RivaServerConfig(port=0)) as server,
        grpc.aio.insecure_channel(server.target) as channel,
    ):
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

        async def gen():  # type: ignore[no-untyped-def]
            yield _config_message()
            yield _audio_message(b"\x00" * (2 * 16000 * 2))  # larger than the ring

        responses, code, details = await _read_to_end(stub.StreamingRecognize(gen()))
    assert code == grpc.StatusCode.UNAVAILABLE
    assert "shutting down" in details
    assert _finals(responses) == []
    assert engine.calls("wait") == [1]
    assert engine.calls("abort") == []


async def test_a_client_mid_stream_is_told_unavailable_when_the_engine_stops() -> None:
    """The engine stops under an open stream: the call ends with UNAVAILABLE, not OK,
    so the caller knows its utterance was abandoned rather than finished and can
    decide to retry. Partials already produced still arrive first; no final does."""
    inner = stub_engine()
    async with (
        inner,
        RivaServer(inner, RivaServerConfig(port=0)) as server,
        grpc.aio.insecure_channel(server.target) as channel,
    ):
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        release: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def gen():  # type: ignore[no-untyped-def]
            yield _config_message()
            yield _audio_message(_chunk())
            await release  # the client stays mid-stream until the test lets go

        call = stub.StreamingRecognize(gen())
        try:
            first = await asyncio.wait_for(call.read(), timeout=5.0)
            assert first.results[0].is_final is False, "the stream must be live and mid-utterance"
            await inner.stop()
            rest, code, details = await _read_to_end(call)
        finally:
            release.set_result(None)
    assert code == grpc.StatusCode.UNAVAILABLE
    assert "shutting down" in details
    assert _finals([first, *rest]) == []


def test_every_error_code_has_a_grpc_status() -> None:
    """A code without a status would surface as an unhandled KeyError in `_abort`,
    which grpc reports as UNKNOWN: the one status this taxonomy must never produce."""
    from verbatim.core.errors import ErrorCode
    from verbatim.protocols.riva.server import _STATUS_BY_CODE

    assert set(_STATUS_BY_CODE) == set(ErrorCode)
    assert _STATUS_BY_CODE[ErrorCode.UNAVAILABLE] is grpc.StatusCode.UNAVAILABLE
