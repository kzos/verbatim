# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The WebSocket demo surface over a real engine, driven with the `websockets` client.

Every test binds port 0 so tests can run in parallel. Every session runs through
`Engine`: the transport owns no recogniser, so what these tests pin is the wire
behaviour of one session on the engine, admission before acknowledgement, live
back-pressure against the session's ring, the message-size cap and the idle
deadline. `interim_results` off is exercised through the `interim_results=0`
query parameter (documented in `parse_query`).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, InvalidStatus

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.types import PcmFrame, StepResult
from verbatim.engine import Engine, stub_engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.base import EngineHandle, Hypothesis, SessionHandle, SessionOptions
from verbatim.protocols.ws.server import DEFAULT_MAX_MESSAGE_BYTES, WsServer, WsServerConfig
from verbatim.scheduler.clock import ScaledMonotonicClock

pytestmark = pytest.mark.cpu

FULL_SCRIPT = "the quick brown fox jumps over the lazy dog"
CHUNK = ChunkMode(160)


def _chunk(chunk_ms: int = 160) -> bytes:
    return b"\x00" * (chunk_ms * 16 * 2)


class _RecordingSession(SessionHandle):
    """A pass-through handle that records what the transport asked the engine for."""

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
    """The stub engine behind a recorder: the transport is tested through the
    public handle interface and nothing else."""

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
    config: WsServerConfig | None = None,
    *,
    engine: Engine | None = None,
    **engine_kwargs: object,
) -> AsyncIterator[WsServer]:
    inner = engine if engine is not None else stub_engine(**engine_kwargs)  # type: ignore[arg-type]
    async with inner, WsServer(inner, config or WsServerConfig(port=0)) as server:
        yield server


@asynccontextmanager
async def _recording_server(
    config: WsServerConfig | None = None, **engine_kwargs: object
) -> AsyncIterator[tuple[WsServer, _RecordingEngine]]:
    inner = stub_engine(**engine_kwargs)  # type: ignore[arg-type]
    engine = _RecordingEngine(inner)
    async with inner, WsServer(engine, config or WsServerConfig(port=0)) as server:
        yield server, engine


async def _recv_json(ws: ClientConnection) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
    assert isinstance(raw, str)
    return json.loads(raw)


async def _collect_until_close(ws: ClientConnection) -> list[dict]:
    frames = []
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert isinstance(raw, str)
            frames.append(json.loads(raw))
    except ConnectionClosedOK:
        pass
    return frames


async def _settle(server: WsServer) -> None:
    for _ in range(500):
        if server.live_sessions == 0:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{server.live_sessions} session(s) still live")


async def test_session_frame_is_first() -> None:
    config = WsServerConfig(port=0, invariance_class="test-class")
    async with _server(config) as server, connect(f"{server.endpoint}?chunk_ms=160") as ws:
        frame = await _recv_json(ws)
    assert frame["type"] == "session"
    assert frame["id"]
    assert frame["chunk_ms"] == 160
    assert frame["invariance_class"] == "test-class"


async def test_a_chunk_ms_the_engine_does_not_serve_is_refused_before_a_session_frame() -> None:
    """One engine serves one chunk mode; a session asking for another gets the error
    the protocol has, not a silent substitution and not a session it cannot have."""
    async with _server() as server, connect(f"{server.endpoint}?chunk_ms=560") as ws:
        frame = await _recv_json(ws)
        with pytest.raises(ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5.0)
    assert frame["type"] == "error"
    assert frame["code"] == "INVALID_ARGUMENT"
    assert "560" in frame["message"]


async def test_one_partial_per_chunk() -> None:
    async with _server() as server, connect(server.endpoint) as ws:
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
    async with _server() as server, connect(server.endpoint) as ws:
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
    async with _server() as server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await _recv_json(ws)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
        assert ws.close_code == 1000
    assert [f["type"] for f in frames] == ["final"]
    assert frames[0]["text"] == FULL_SCRIPT


async def test_trailing_partial_chunk_is_padded_but_only_real_bytes_are_counted() -> None:
    async with _server() as server, connect(server.endpoint) as ws:
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


async def test_audio_s_is_the_engines_clock_not_a_byte_counter() -> None:
    """The transport keeps no clock. Bytes received are not bytes recognised once a
    ring sits between the socket and the pipeline: after `end`, the final is stamped
    with what the engine consumed, which for a mid-chunk tail is every real sample."""
    async with _recording_server() as (server, engine), connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk() + b"\x00" * 1000)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["audio_s"] == pytest.approx((2560 + 500) / 16000)
    assert sum(engine.calls("feed")) == len(_chunk()) + 1000


async def test_words_only_when_requested() -> None:
    async with _server() as server:
        async with connect(f"{server.endpoint}?words=1") as ws:
            await _recv_json(ws)
            await ws.send(_chunk())
            await _recv_json(ws)
            await ws.send('{"type": "end"}')
            frames = await _collect_until_close(ws)
        async with connect(f"{server.endpoint}?words=0") as ws:
            raw_session = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert isinstance(raw_session, str)
            await ws.send(_chunk())
            await _recv_json(ws)
            await ws.send('{"type": "end"}')
            raws = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    assert isinstance(raw, str)
                    raws.append(raw)
            except ConnectionClosedOK:
                pass
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1
    assert len(finals[0]["words"]) > 0
    assert all('"words"' not in raw for raw in raws)


async def test_interim_results_false_suppresses_partials() -> None:
    async with _server() as server, connect(f"{server.endpoint}?interim_results=0") as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await ws.send(_chunk())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.recv(), timeout=0.05)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
    assert [f["type"] for f in frames] == ["final"]


async def test_bad_chunk_ms_gets_an_error_frame_then_a_clean_close() -> None:
    async with _server() as server, connect(f"{server.endpoint}?chunk_ms=100") as ws:
        frame = await _recv_json(ws)
        with pytest.raises(ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5.0)
        await ws.wait_closed()
        assert ws.close_code == 1000
    assert frame["type"] == "error"
    assert frame["code"] == "INVALID_ARGUMENT"


async def test_junk_text_frame_errors_without_closing() -> None:
    async with _server() as server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send('{"type":"start"}')
        error = await _recv_json(ws)
        await ws.send(_chunk())
        partial = await _recv_json(ws)
    assert error["type"] == "error"
    assert error["code"] == "INVALID_ARGUMENT"
    assert partial["type"] == "partial"


async def test_wrong_path_is_rejected() -> None:
    async with _server() as server:
        with pytest.raises(InvalidStatus):
            async with connect(f"ws://127.0.0.1:{server.port}/nope"):
                pass


async def test_concurrent_sessions_do_not_interfere() -> None:
    scripts = {f"lang-{i}": (f"alpha{i}", f"beta{i}", f"gamma{i}") for i in range(8)}

    def script_for(options: SessionOptions) -> Sequence[str]:
        return scripts[options.language_code]

    async with _server(script_for=script_for) as server:

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


async def test_disconnect_without_end_aborts_the_session_and_emits_no_final() -> None:
    """A client that leaves without `end` gets no final: the transport aborts, never
    ends, and the engine's abort frame yields nothing (pinned in the engine suite)."""
    async with _recording_server() as (server, engine), connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        partial = await _recv_json(ws)
        await ws.close()
        await ws.wait_closed()
        await _settle(server)
    assert server.live_sessions == 0
    assert server.sessions_total == 1
    assert partial["type"] == "partial"
    assert engine.calls("abort") == [0]
    assert engine.calls("end") == []


async def test_refusal_at_capacity_is_one_error_frame_and_no_session_frame() -> None:
    """Admission comes before acknowledgement. The ninth connection on the default
    bucket of eight gets RESOURCE_EXHAUSTED with the retry hint, then a clean close,
    and never a `session` frame it could mistake for an open session."""
    async with _server() as server:
        held = [await connect(server.endpoint) for _ in range(8)]
        try:
            for ws in held:
                assert (await _recv_json(ws))["type"] == "session"
            async with connect(server.endpoint) as refused:
                frames = await _collect_until_close(refused)
                assert refused.close_code == 1000
        finally:
            for ws in held:
                await ws.close()
        await _settle(server)
    assert [f["type"] for f in frames] == ["error"]
    assert frames[0]["code"] == "RESOURCE_EXHAUSTED"
    assert frames[0]["message"].startswith("session refused: ")
    assert "retry after 160 ms" in frames[0]["message"]
    assert server.sessions_total == 8


async def test_back_pressure_holds_the_remainder_and_drops_nothing() -> None:
    """A message larger than the session's ring is fed over several ticks: `feed`
    accepts short, the reader waits one tick and offers the rest, and every sample
    reaches the engine. The wait is what stops the socket being read meanwhile."""
    config = WsServerConfig(port=0, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
    message = b"\x00" * (2 * 16000 * 2)  # two seconds against a half-second ring
    async with (
        _recording_server(config, ring_seconds=0.5) as (server, engine),
        connect(server.endpoint) as ws,
    ):
        await _recv_json(ws)
        await ws.send(message)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["audio_s"] == pytest.approx(2.0)
    assert len([f for f in frames if f["type"] == "partial"]) == 13  # 12 full chunks + tail
    feeds = engine.calls("feed")
    assert sum(feeds) == len(message)
    assert feeds[0] < len(message), "the first feed should have been short: the ring is smaller"
    assert len(engine.calls("wait")) >= len(feeds) - 1


async def test_a_message_at_the_cap_is_accepted() -> None:
    async with _server() as server, connect(server.endpoint, max_size=None) as ws:
        await _recv_json(ws)
        await ws.send(b"\x00" * DEFAULT_MAX_MESSAGE_BYTES)
        await ws.send('{"type": "end"}')
        frames = await _collect_until_close(ws)
        assert ws.close_code == 1000
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["audio_s"] == pytest.approx(3.0)


async def test_a_message_over_the_cap_closes_with_1009_and_frees_the_session() -> None:
    """The framing layer refuses the message before its payload is read, with close
    code 1009 (message too big). No error frame can precede that: sending one would
    mean reading the payload the cap exists to refuse. The session is aborted."""
    config = WsServerConfig(port=0, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
    async with _recording_server(config) as (server, engine):
        async with connect(server.endpoint, max_size=None) as ws:
            await _recv_json(ws)
            await ws.send(b"\x00" * (DEFAULT_MAX_MESSAGE_BYTES + 2))
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert ws.close_code == 1009
        await _settle(server)
    assert engine.calls("abort") == [0]
    assert engine.calls("feed") == []


@pytest.mark.parametrize("max_message_bytes", [0, 1])
def test_a_cap_below_one_sample_is_a_config_error(max_message_bytes: int) -> None:
    with pytest.raises(ValueError, match="max_message_bytes"):
        WsServerConfig(port=0, max_message_bytes=max_message_bytes)


async def test_an_idle_session_is_closed_with_deadline_exceeded() -> None:
    """A client that opens a session and never sends audio is told DEADLINE_EXCEEDED
    and closed; the engine enforces it, so the deadline is the same on both wires."""
    async with _server(idle_timeout_s=0.5) as server, connect(server.endpoint) as ws:
        assert (await _recv_json(ws))["type"] == "session"
        frames = await _collect_until_close(ws)
        assert ws.close_code == 1000
    assert [f["type"] for f in frames] == ["error"]
    assert frames[0]["code"] == "DEADLINE_EXCEEDED"
    await _settle(server)


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


async def test_a_pipeline_exception_becomes_an_error_frame() -> None:
    config = EngineConfig(chunk=CHUNK, buckets=(8,))
    engine = Engine(config, _Flaky(), clock=ScaledMonotonicClock(100.0))
    async with _server(engine=engine) as server, connect(server.endpoint) as ws:
        await _recv_json(ws)
        await ws.send(_chunk())
        await ws.send(_chunk())
        frames = await _collect_until_close(ws)
    errors = [f for f in frames if f["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "INTERNAL"


class _StoppingEngine(_RecordingEngine):
    """The engine as a parked reader sees it once `stop()` has begun: the tick wait
    raises, and by then the engine has ended the session itself."""

    async def wait_for_ticks(self, n: int) -> None:
        self.log.append(("wait", n))
        await self.inner.stop()
        raise RuntimeError("engine tick loop is not running")


async def _collect_until_any_close(ws: ClientConnection) -> list[dict]:
    """Like `_collect_until_close`, but a close with any code ends the collection;
    the caller asserts on `ws.close_code`."""
    frames = []
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert isinstance(raw, str)
            frames.append(json.loads(raw))
    except ConnectionClosed:
        pass
    return frames


async def test_a_reader_parked_in_back_pressure_when_the_engine_stops_is_told_unavailable() -> None:
    """The reader's tick wait raises during shutdown. The reader does not abort the
    session, because the engine has already ended it with UNAVAILABLE and an abort
    could race the thread's last tick into a clean close; the client gets an error
    frame naming UNAVAILABLE and close code 1001, going away, with no final. A partial
    from audio the engine had already stepped may precede the error frame."""
    inner = stub_engine(ring_seconds=0.5)
    engine = _StoppingEngine(inner)
    async with (
        inner,
        WsServer(engine, WsServerConfig(port=0)) as server,
        connect(server.endpoint) as ws,
    ):
        assert (await _recv_json(ws))["type"] == "session"
        await ws.send(b"\x00" * (2 * 16000 * 2))  # larger than the ring: the first feed is short
        frames = await _collect_until_any_close(ws)
        assert ws.close_code == 1001
        await _settle(server)
    assert [frame["type"] for frame in frames[:-1] if frame["type"] != "partial"] == []
    assert frames[-1]["type"] == "error"
    assert frames[-1]["code"] == "UNAVAILABLE"
    assert engine.calls("wait") == [1]
    assert engine.calls("abort") == []


async def test_a_client_mid_stream_gets_unavailable_and_close_1001_when_the_engine_stops() -> None:
    """The engine stops under an open session: the client is told with an error frame
    naming UNAVAILABLE and a 1001 close, going away, never a 1000 close that would read
    as a finished utterance. Partials already produced still arrive first; no final."""
    inner = stub_engine()
    async with (
        inner,
        WsServer(inner, WsServerConfig(port=0)) as server,
        connect(server.endpoint) as ws,
    ):
        assert (await _recv_json(ws))["type"] == "session"
        await ws.send(_chunk())
        first = await _recv_json(ws)
        assert first["type"] == "partial", "the session must be live and mid-utterance"
        await inner.stop()
        frames = await _collect_until_any_close(ws)
        assert ws.close_code == 1001
        await _settle(server)
    assert all(frame["type"] == "partial" for frame in frames[:-1])
    assert frames[-1] == {"type": "error", "code": "UNAVAILABLE", "message": frames[-1]["message"]}
    assert "shutting down" in frames[-1]["message"]


class _StuckSession(_RecordingSession):
    """A session whose result stream never ends: the engine failed to end it."""

    async def results(self) -> AsyncIterator[Hypothesis]:
        await asyncio.Event().wait()
        yield Hypothesis(text="", is_final=False, audio_processed_s=0.0)  # pragma: no cover


class _StuckEngine(_RecordingEngine):
    def open_session(self, options: SessionOptions) -> SessionHandle:
        return _StuckSession(self.inner.open_session(options), self.log)


async def test_the_listener_stops_within_its_grace_when_a_session_never_ends() -> None:
    """A handler whose result stream never ends must not hold the listener's shutdown
    open: after the grace the handler is cancelled, its session aborted, and `stop()`
    returns. Without the bound an engine that failed to end one session would keep
    the process alive past SIGTERM, and a test of that engine hangs instead of
    failing, which is how round 5's mutation table lost a row."""
    engine = _StuckEngine(stub_engine())
    server = WsServer(engine, WsServerConfig(port=0))
    await server.start()
    async with connect(server.endpoint) as ws:
        assert (await _recv_json(ws))["type"] == "session"
        await ws.send(_chunk())
        await ws.send('{"type": "end"}')  # the reader ends the session and returns
        for _ in range(500):
            if engine.calls("end"):
                break
            await asyncio.sleep(0.01)
        assert engine.calls("end") == [0], "the reader must have half-closed before stop()"
        started = time.monotonic()
        await asyncio.wait_for(server.stop(grace=0.5), timeout=5.0)
        elapsed = time.monotonic() - started
        frames = await _collect_until_any_close(ws)
        assert ws.close_code == 1001  # the listener's own going-away close
    assert 0.5 <= elapsed < 5.0, f"stop() took {elapsed:.2f} s against a 0.5 s grace"
    assert frames == []
    assert server.live_sessions == 0
    assert engine.calls("abort") == [0], "the cancelled handler drops the session it was holding"
