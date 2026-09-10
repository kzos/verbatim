# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The Riva subset over grpc.aio on 127.0.0.1: streaming, config lookup, unary refusal.

Every test binds port 0 so tests never collide on a fixed port. The `_collect`
helper opens a channel, sends a config message then N chunks of silence, then
half-closes, and gathers every response.
"""

from __future__ import annotations

import asyncio
from itertools import pairwise

import grpc
import pytest

from verbatim.pipelines.fake import StubRecognizer, stub_recognizer_factory
from verbatim.protocols.base import Hypothesis, Recognizer, SessionOptions
from verbatim.protocols.riva._gen import riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2
from verbatim.protocols.riva.server import RivaServer, RivaServerConfig

pytestmark = pytest.mark.cpu

ENC = riva_audio_pb2.AudioEncoding
FULL_SCRIPT = "the quick brown fox jumps over the lazy dog"


def _chunk(chunk_ms: int = 160) -> bytes:
    return b"\x00" * (chunk_ms * 32)


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
    async with (
        RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server,
        grpc.aio.insecure_channel(server.target) as channel,
    ):
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub.Recognize(riva_asr_pb2.RecognizeRequest())
    assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED


async def test_get_config_returns_online_models() -> None:
    async with (
        RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server,
        grpc.aio.insecure_channel(server.target) as channel,
    ):
        stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
        response = await stub.GetRivaSpeechRecognitionConfig(
            riva_asr_pb2.RivaSpeechRecognitionConfigRequest()
        )
    online = [c for c in response.model_config if c.parameters.get("type") == "online"]
    assert len(online) >= 1
    assert all(c.model_name for c in online)


async def test_streaming_happy_path() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
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
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [_audio_message(_chunk()), _config_message()])
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_second_config_message_is_invalid_argument() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [_config_message(), _config_message()])
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_interim_results_false_suppresses_partials() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        responses = await _collect(
            server.target,
            [_config_message(interim_results=False), *[_audio_message(_chunk()) for _ in range(3)]],
        )
    assert len(responses) == 1
    assert responses[0].results[0].is_final is True


async def test_request_id_is_echoed_on_every_response() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        responses = await _collect(
            server.target,
            [_config_message(request_id="abc123"), *[_audio_message(_chunk()) for _ in range(2)]],
        )
    assert len(responses) > 1
    assert all(r.id.value == "abc123" for r in responses)


async def test_no_request_id_gives_empty_string() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        responses = await _happy_collect(server, n=2)
    assert len(responses) > 0
    for response in responses:
        assert response.id.value == ""


async def test_unknown_metadata_is_ignored() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        responses = await _collect(
            server.target,
            [_config_message(), *[_audio_message(_chunk()) for _ in range(2)]],
            metadata=[("function-id", "test-function"), ("authorization", "Bearer nope")],
        )
    assert len(_finals(responses)) == 1
    assert _finals(responses)[0].results[0].alternatives[0].transcript == FULL_SCRIPT


async def test_force_eou_is_accepted_and_does_not_abort() -> None:
    async with (
        RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server,
        grpc.aio.insecure_channel(server.target) as channel,
    ):
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
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        tiny = [_audio_message(bytes(512)) for _ in range(10)]
        responses = await _collect(server.target, [_config_message(), *tiny])
    assert len(_partials(responses)) == 1
    assert len(_finals(responses)) == 1


async def test_trailing_partial_chunk_is_padded_but_only_real_bytes_are_counted() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
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
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
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
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [_config_message(recognition={"model": "nope"})])
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


async def test_bad_encoding_aborts_with_invalid_argument() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(
                server.target,
                [_config_message(recognition={"encoding": ENC.ENCODING_UNSPECIFIED})],
            )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "encoding" in excinfo.value.details()


async def test_diarization_aborts_with_unimplemented() -> None:
    async with RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server:
        request = _config_message()
        request.streaming_config.config.diarization_config.enable_speaker_diarization = True
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, [request])
    assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED


async def test_concurrent_streams_do_not_interfere() -> None:
    scripts = {f"lang-{i}": (f"alpha{i}", f"beta{i}", f"gamma{i}") for i in range(8)}

    def factory(options: SessionOptions) -> Recognizer:
        return StubRecognizer(options, script=scripts[options.language_code])

    async with RivaServer(factory, RivaServerConfig(port=0)) as server:

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


async def test_client_cancel_is_clean() -> None:
    async with (
        RivaServer(stub_recognizer_factory(), RivaServerConfig(port=0)) as server,
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
        await asyncio.sleep(0.1)
        responses = await _collect(server.target, [_config_message(), _audio_message(_chunk())])
    assert len(_finals(responses)) == 1


async def test_recognizer_exception_aborts_with_internal() -> None:
    class Flaky(Recognizer):
        def __init__(self, options: SessionOptions) -> None:
            self._options = options
            self._chunks = 0

        @property
        def options(self) -> SessionOptions:
            return self._options

        def add_chunk(self, pcm: bytes) -> list[Hypothesis]:
            self._chunks += 1
            if self._chunks == 2:
                raise RuntimeError("boom")
            return []

        def finalize(self) -> list[Hypothesis]:
            return []

    async with RivaServer(Flaky, RivaServerConfig(port=0)) as server:
        requests = [_config_message(), _audio_message(_chunk()), _audio_message(_chunk())]
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await _collect(server.target, requests)
    assert excinfo.value.code() == grpc.StatusCode.INTERNAL
