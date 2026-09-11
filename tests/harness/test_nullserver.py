# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the null server, spoken to over real loopback WebSockets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import websockets
from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.nullserver import NullServer, NullServerConfig
from websockets.exceptions import ConnectionClosed

pytestmark = pytest.mark.cpu

CHUNK_BYTES_160MS = 160 * 16000 // 1000 * 2


async def recv_json(ws):
    message = await ws.recv()
    assert isinstance(message, str)
    return json.loads(message)


async def test_null_server_sends_session_first() -> None:
    async with (
        NullServer(NullServerConfig()) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        first = await recv_json(ws)
        assert first["type"] == "session"
        assert first["chunk_ms"] == 160
        assert isinstance(first["id"], str)


async def test_null_server_emits_one_partial_per_chunk() -> None:
    async with (
        NullServer(NullServerConfig()) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await recv_json(ws)
        partials = []
        for _ in range(5):
            await ws.send(b"\x01\x02" * (CHUNK_BYTES_160MS // 2))
            partials.append(await recv_json(ws))
        assert [p["type"] for p in partials] == ["partial"] * 5
        assert partials[-1]["audio_s"] == pytest.approx(0.8)
        await ws.send(json.dumps({"type": "end"}))
        final = await recv_json(ws)
        assert final["type"] == "final"


async def test_null_server_finalises_on_end() -> None:
    async with (
        NullServer(NullServerConfig()) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await recv_json(ws)
        await ws.send(json.dumps({"type": "end"}))
        final = await recv_json(ws)
        assert final["type"] == "final"
        with pytest.raises(ConnectionClosed):
            await ws.recv()


async def test_segment_finals_emit_mid_stream_and_partials_continue_after_them() -> None:
    config = NullServerConfig(segment_finals=((2, "hello"),), final_text="again")
    async with (
        NullServer(config) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await recv_json(ws)
        frames = []
        for _ in range(3):
            await ws.send(b"\x00" * CHUNK_BYTES_160MS)
            frames.append(await recv_json(ws))
            if len(frames) == 2:
                frames.append(await recv_json(ws))
        assert [f["type"] for f in frames] == ["partial", "partial", "final", "partial"]
        assert frames[2]["text"] == "hello"
        assert frames[2]["audio_s"] == pytest.approx(0.32)
        await ws.send(json.dumps({"type": "end"}))
        terminal = await recv_json(ws)
        assert terminal["type"] == "final"
        assert terminal["text"] == "again"


async def test_null_server_reports_error_when_configured() -> None:
    async with (
        NullServer(NullServerConfig(fail_after_chunks=2)) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await recv_json(ws)
        for _ in range(3):
            await ws.send(b"\x00" * CHUNK_BYTES_160MS)
        seen_error = None
        try:
            for _ in range(6):
                event = await recv_json(ws)
                if event["type"] == "error":
                    seen_error = event
                    break
        except ConnectionClosed:
            pass
        assert seen_error is not None, "expected an error frame before the session ends"
        assert "code" in seen_error


async def test_short_tail_emits_a_final_partial_so_the_floor_matches_the_server() -> None:
    pcm = bytes(range(256)) * 125
    assert len(pcm) == 32000
    utterance = Utterance(
        stream_id="utt-0",
        audio_path=Path("utt.wav"),
        duration_s=1.0,
        text="reference for utt-0",
    )
    async with NullServer(NullServerConfig()) as server:
        session = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterance,
            pcm=pcm,
            chunk=ChunkMode.parse("160ms"),
            start_delay_s=0.0,
        )
    assert session.error is None
    assert session.chunks == 7
    assert session.partials_received == 7
    assert session.audio_s == pytest.approx(1.0)
    # Every chunk, the 40 ms tail included, is acknowledged by a partial whose
    # watermark covers it: seven matched samples, not six and an unmatched tail.
    assert len(session.partial_ms) == 7


async def test_the_tail_partial_is_stamped_with_every_byte_sent() -> None:
    """The floor's last partial carries the tail's audio, as the server's padded final
    chunk counts the real samples in it; a stamp at the last full chunk would leave
    the client's tail chunk unmatched and the floor a chunk short of the server."""
    async with (
        NullServer(NullServerConfig()) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await recv_json(ws)
        await ws.send(b"\x00" * (CHUNK_BYTES_160MS * 2 + CHUNK_BYTES_160MS // 4))  # 2.25 chunks
        first = await recv_json(ws)
        second = await recv_json(ws)
        await ws.send(json.dumps({"type": "end"}))
        tail = await recv_json(ws)
        final = await recv_json(ws)
    assert [first["audio_s"], second["audio_s"]] == pytest.approx([0.16, 0.32])
    assert tail["type"] == "partial"
    assert tail["audio_s"] == pytest.approx(0.36)
    assert final["type"] == "final"
    assert final["audio_s"] == pytest.approx(0.36)


async def test_a_partial_is_stamped_with_the_chunk_it_acknowledges_not_the_bytes_received() -> None:
    """A message that runs past a chunk boundary has been received past it, not
    recognised past it. The server stamps its audio clock, which advances one chunk
    per step; the floor must stamp the same, or a watermark match against the floor
    measures a different thing from the same match against the server."""
    async with (
        NullServer(NullServerConfig()) as server,
        websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await recv_json(ws)
        await ws.send(b"\x00" * (CHUNK_BYTES_160MS + CHUNK_BYTES_160MS // 2))  # 1.5 chunks
        first = await recv_json(ws)
        await ws.send(b"\x00" * (CHUNK_BYTES_160MS // 2))  # completes the second
        second = await recv_json(ws)
        await ws.send(json.dumps({"type": "end"}))
        final = await recv_json(ws)
    assert first["audio_s"] == pytest.approx(0.16)  # not 0.24, the bytes received
    assert second["audio_s"] == pytest.approx(0.32)
    assert final["type"] == "final"  # nothing pending: no tail partial
    assert final["audio_s"] == pytest.approx(0.32)


async def test_capacity_limited_mode_adds_delay_only_above_capacity() -> None:
    import asyncio

    pcm = bytes(16000)
    utterance = Utterance(
        stream_id="cap-0",
        audio_path=Path("cap.wav"),
        duration_s=0.5,
        text="capacity",
    )
    config = NullServerConfig(capacity=8, overload_penalty_ms=800.0)
    async with NullServer(config) as server:

        async def _one(index: int):
            return await run_session(
                server.endpoint,
                session_id=f"s{index:04d}",
                utterance=utterance,
                pcm=pcm,
                chunk=ChunkMode.parse("160ms"),
                start_delay_s=0.0,
            )

        sessions = await asyncio.gather(*(_one(i) for i in range(2)))
    assert all(s.error is None for s in sessions)


async def test_word_script_mode_emits_known_words_with_known_progression() -> None:
    import json as _json

    import websockets as _websockets

    script = ("hello", "world", "again")
    config = NullServerConfig(word_script=script)
    async with (
        NullServer(config) as server,
        _websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        raw = await ws.recv()
        assert isinstance(raw, str)
        assert _json.loads(raw)["type"] == "session"
        await ws.send(b"\x00" * CHUNK_BYTES_160MS)
        first = _json.loads(await ws.recv())
        await ws.send(b"\x00" * CHUNK_BYTES_160MS)
        second = _json.loads(await ws.recv())
    assert first["text"] == "hello"
    assert second["text"] == "hello world"


async def test_partial_retraction_mode_removes_a_word_then_restores_it() -> None:
    import json as _json

    import websockets as _websockets

    config = NullServerConfig(partial_text="hello world", retract_mode=True)
    async with (
        NullServer(config) as server,
        _websockets.connect(f"{server.endpoint}?chunk_ms=160") as ws,
    ):
        await ws.recv()
        await ws.send(b"\x00" * CHUNK_BYTES_160MS)
        first = _json.loads(await ws.recv())
        await ws.send(b"\x00" * CHUNK_BYTES_160MS)
        second = _json.loads(await ws.recv())
    assert first["text"] == "hello world"
    assert second["text"] == "hello"
