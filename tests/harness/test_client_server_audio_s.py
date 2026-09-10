# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Client/server `audio_s` agreement on a short tail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from verbatim_bench.client import ChunkMode, _split_chunks, run_session
from verbatim_bench.corpus import Utterance
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from verbatim.engine import stub_engine
from verbatim.protocols.ws.server import WsServer, WsServerConfig

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("pcm_len", [32000, 32002])
async def test_client_and_real_server_agree_on_audio_s(pcm_len: int) -> None:
    pcm = (bytes(range(256)) * ((pcm_len // 256) + 1))[:pcm_len]
    assert len(pcm) == pcm_len
    expected_audio_s = pcm_len / 2 / 16000
    utterance = Utterance(
        stream_id="utt-0",
        audio_path=Path("utt.wav"),
        duration_s=expected_audio_s,
        text="reference for utt-0",
    )
    chunk = ChunkMode.parse("160ms")
    engine = stub_engine()
    server = WsServer(engine, WsServerConfig(port=0))
    async with engine, server:
        session = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterance,
            pcm=pcm,
            chunk=chunk,
            start_delay_s=0.0,
        )
        assert session.error is None
        assert session.chunks == 7
        assert session.audio_s == pytest.approx(expected_audio_s)
        assert session.final_text

        async with connect(server.endpoint + "?chunk_ms=160") as ws:
            raw = await ws.recv()
            assert isinstance(raw, str)
            for frame in _split_chunks(pcm, chunk.bytes):
                await ws.send(frame)
            await ws.send(json.dumps({"type": "end"}))
            frames = []
            try:
                while True:
                    msg = await ws.recv()
                    assert isinstance(msg, str)
                    frames.append(json.loads(msg))
            except ConnectionClosedOK:
                pass
        finals = [f for f in frames if f["type"] == "final"]
        assert len(finals) == 1
        assert finals[0]["text"] == session.final_text
        server_audio_s = finals[0]["audio_s"]
        # Both frame classes round audio_s to six decimals on the wire,
        # so the tolerance cannot be tighter than about 1e-6.
        assert server_audio_s == pytest.approx(session.audio_s, abs=1e-6)
        assert server_audio_s == pytest.approx(len(pcm) / 2 / 16000, abs=1e-6)
        if pcm_len == 32000:
            assert session.audio_s == pytest.approx(1.0)
            assert server_audio_s == pytest.approx(1.0)
