# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The WebSocket demo surface, driven with the `websockets` client on 127.0.0.1.

Every test binds port 0 so tests can run in parallel. `interim_results` off is
exercised through the `interim_results=0` query parameter (documented in
`parse_query`): with it set, the transport suppresses partials but still sends
the final.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedOK, InvalidStatus

from verbatim.pipelines.fake import StubRecognizer, stub_recognizer_factory
from verbatim.protocols.base import Hypothesis, Recognizer, SessionOptions
from verbatim.protocols.ws.server import WsServer, WsServerConfig

pytestmark = pytest.mark.cpu

FULL_SCRIPT = "the quick brown fox jumps over the lazy dog"


def _chunk(chunk_ms: int = 160) -> bytes:
    return b"\x00" * (chunk_ms * 16 * 2)


def _server() -> WsServer:
    return WsServer(stub_recognizer_factory(), WsServerConfig(port=0))


async def _recv_json(ws: ClientConnection) -> dict:
    raw = await ws.recv()
    assert isinstance(raw, str)
    return json.loads(raw)


async def _collect_until_close(ws: ClientConnection) -> list[dict]:
    frames = []
    try:
        while True:
            raw = await ws.recv()
            assert isinstance(raw, str)
            frames.append(json.loads(raw))
    except ConnectionClosedOK:
        pass
    return frames


async def test_session_frame_is_first() -> None:
    config = WsServerConfig(port=0, invariance_class="test-class")
    server = WsServer(stub_recognizer_factory(), config)
    async with server, connect(f"{server.endpoint}?chunk_ms=560") as ws:
        frame = await _recv_json(ws)
    assert frame["type"] == "session"
    assert frame["id"]
    assert frame["chunk_ms"] == 560
    assert frame["invariance_class"] == "test-class"


async def test_one_partial_per_chunk() -> None:
    server = _server()
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        partials = []
        for _ in range(5):
            await ws.send(_chunk())
            partials.append(await _recv_json(ws))
    assert [p["type"] for p in partials] == ["partial"] * 5
    assert [p["text"] for p in partials] == [
        "the",
        "the quick",
        "the quick brown",
        "the quick brown fox",
        "the quick brown fox jumps",
    ]
    assert [p["audio_s"] for p in partials] == pytest.approx([0.16, 0.32, 0.48, 0.64, 0.80])


async def test_partial_bytes_are_buffered_until_a_chunk_completes() -> None:
    server = _server()
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        for _ in range(9):
            await ws.send(b"\x00" * 512)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=0.05)
        await ws.send(b"\x00" * 512)
        frame = await asyncio.wait_for(ws.recv(), timeout=1.0)
    assert isinstance(frame, str)
    assert json.loads(frame)["type"] == "partial"


async def test_end_produces_exactly_one_final_and_closes() -> None:
    server = _server()
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await _recv_json(ws)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
        assert ws.close_code == 1000
    assert [f["type"] for f in frames] == ["final"]
    assert frames[0]["text"] == FULL_SCRIPT


async def test_trailing_partial_chunk_is_padded_but_only_real_bytes_are_counted() -> None:
    server = _server()
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await ws.send(_chunk())
        await _recv_json(ws)
        await _recv_json(ws)
        await ws.send(b"\x00" * 1000)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["audio_s"] == pytest.approx(0.35125)


async def test_words_only_when_requested() -> None:
    server = _server()
    async with server:
        async with connect(f"{server.endpoint}?words=1") as ws:
            await _recv_json(ws)
            await ws.send(_chunk())
            await _recv_json(ws)
            await ws.send('{"type": "end"}')
            frames = await _collect_until_close(ws)
        async with connect(f"{server.endpoint}?words=0") as ws:
            raw_session = await ws.recv()
            assert isinstance(raw_session, str)
            await ws.send(_chunk())
            await _recv_json(ws)
            await ws.send('{"type": "end"}')
            raws = []
            try:
                while True:
                    raw = await ws.recv()
                    assert isinstance(raw, str)
                    raws.append(raw)
            except ConnectionClosedOK:
                pass
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1
    assert len(finals[0]["words"]) > 0
    assert all('"words"' not in raw for raw in raws)


async def test_interim_results_false_suppresses_partials() -> None:
    server = _server()
    async with server, connect(f"{server.endpoint}?interim_results=0") as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await ws.send(_chunk())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.recv(), timeout=0.05)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
    assert [f["type"] for f in frames] == ["final"]


async def test_bad_chunk_ms_gets_an_error_frame_then_a_clean_close() -> None:
    server = _server()
    async with server, connect(f"{server.endpoint}?chunk_ms=100") as ws:
        frame = await _recv_json(ws)
        with pytest.raises(ConnectionClosedOK):
            await ws.recv()
        await ws.wait_closed()
        assert ws.close_code == 1000
    assert frame["type"] == "error"
    assert frame["code"] == "INVALID_ARGUMENT"


async def test_junk_text_frame_errors_without_closing() -> None:
    server = _server()
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send('{"type":"start"}')
        error = await _recv_json(ws)
        await ws.send(_chunk())
        partial = await _recv_json(ws)
    assert error["type"] == "error"
    assert error["code"] == "INVALID_ARGUMENT"
    assert partial["type"] == "partial"


async def test_wrong_path_is_rejected() -> None:
    server = _server()
    async with server:
        with pytest.raises(InvalidStatus):
            async with connect(f"ws://127.0.0.1:{server.port}/nope"):
                pass


async def test_concurrent_sessions_do_not_interfere() -> None:
    scripts = {f"lang-{i}": (f"alpha{i}", f"beta{i}", f"gamma{i}") for i in range(8)}

    def factory(options: SessionOptions) -> Recognizer:
        return StubRecognizer(options, script=scripts[options.language_code])

    server = WsServer(factory, WsServerConfig(port=0))
    async with server:

        async def run_session(i: int) -> str:
            async with connect(f"{server.endpoint}?lang=lang-{i}") as ws:
                await _recv_json(ws)
                for _ in range(3):
                    await ws.send(_chunk())
                    await _recv_json(ws)
                await ws.send('{"type": "end"}')
                frames = await _collect_until_close(ws)
            finals = [f for f in frames if f["type"] == "final"]
            assert len(finals) == 1
            return finals[0]["text"]

        texts = await asyncio.gather(*(run_session(i) for i in range(8)))
    for i, text in enumerate(texts):
        assert text == " ".join(scripts[f"lang-{i}"])


async def test_disconnect_without_end_emits_no_final() -> None:
    finalize_calls: list[str] = []

    class RecordingStub(StubRecognizer):
        def finalize(self) -> list[Hypothesis]:
            finalize_calls.append("finalize")
            return super().finalize()

    def factory(options: SessionOptions) -> Recognizer:
        return RecordingStub(options)

    server = WsServer(factory, WsServerConfig(port=0))
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        partial = await _recv_json(ws)
        await ws.close()
        await ws.wait_closed()
    for _ in range(100):
        if server.live_sessions == 0:
            break
        await asyncio.sleep(0.01)
    assert server.live_sessions == 0
    assert server.sessions_total == 1
    assert partial["type"] == "partial"
    assert finalize_calls == []


async def test_a_large_binary_frame_is_accepted() -> None:
    server = _server()
    async with server, connect(server.endpoint, max_size=None) as ws:
        await _recv_json(ws)
        await ws.send(b"\x00" * (5120 * 300))
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
        assert ws.close_code == 1000
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1


async def test_recognizer_exception_becomes_an_error_frame() -> None:
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

    server = WsServer(Flaky, WsServerConfig(port=0))
    async with server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await ws.send(_chunk())
        frames = await _collect_until_close(ws)
    errors = [f for f in frames if f["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "INTERNAL"
