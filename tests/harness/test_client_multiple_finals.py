# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The load client reads every final a stream carries, not only the first.

The server marks a hypothesis final at each endpoint it detects, so a stream with
an internal silence of the endpointing length arrives as several finals with
partials continuing after each. A reader that stops at the first one records a
fragment of the transcript, a final count that cannot exceed one, and no watermark
for any chunk sent after it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from verbatim_bench.client import ChunkMode, SessionResult, join_final_texts, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.results import RunResult
from verbatim_bench.schema import validate

pytestmark = pytest.mark.cpu

# 1.0 s of audio in 160 ms chunks: six full chunks and a 40 ms tail, so the
# server emits six partials while streaming and a seventh when `end` flushes.
_PCM_1S = bytes(32000)
_CHUNKS = 7


def _utterance() -> Utterance:
    return Utterance(
        stream_id="finals-0",
        audio_path=Path("finals.wav"),
        duration_s=1.0,
        text="hello world again",
    )


async def _run(config: NullServerConfig) -> SessionResult:
    async with NullServer(config) as server:
        return await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=_utterance(),
            pcm=_PCM_1S,
            chunk=ChunkMode.parse("160ms"),
            start_delay_s=0.0,
        )


def _segmented() -> NullServerConfig:
    """Two mid-stream finals at chunks 2 and 4, then the terminal one on `end`."""
    return NullServerConfig(segment_finals=((2, "hello"), (4, "world")), final_text="again")


async def test_a_stream_with_three_finals_is_counted_and_joined() -> None:
    session = await _run(_segmented())

    assert session.error is None
    assert session.finals_received == 3
    assert session.final_text == "hello world again"


async def test_partials_after_an_early_final_are_still_matched_to_their_chunks() -> None:
    session = await _run(_segmented())

    assert session.error is None
    assert session.chunks == _CHUNKS
    # The first final lands after chunk 2. A reader that stopped there would see
    # two partials and two matched chunks out of seven.
    assert session.partials_received == _CHUNKS
    assert len(session.partial_ms) == _CHUNKS


async def test_final_latency_is_measured_to_the_last_final_not_the_first() -> None:
    # The early final arrives about 160 ms into a one-second stream, well before
    # the last chunk is sent, so timing to it would put `final_ms` deeply
    # negative instead of measuring what completed the stream.
    session = await _run(NullServerConfig(segment_finals=((1, "hello"),), final_text="again"))

    assert session.error is None
    assert session.finals_received == 2
    assert session.final_ms is not None
    assert 0.0 <= session.final_ms < 80.0, session.final_ms


async def test_a_single_final_stream_behaves_exactly_as_before() -> None:
    session = await _run(NullServerConfig())

    assert session.error is None
    assert session.finals_received == 1
    assert session.final_text == "null server"
    assert session.chunks == session.partials_received == len(session.partial_ms) == _CHUNKS
    assert session.final_ms is not None and session.final_ms < 80.0


async def test_the_run_document_carries_the_finals_of_every_stream() -> None:
    session = await _run(_segmented())
    doc = RunResult(
        spec_dict={
            "arm": "test",
            "chunk_ms": 160,
            "seed": 0,
            "sessions": 1,
            "profile": "uniform",
            "endpoint": "ws://127.0.0.1:1/v1/stream",
            "words": False,
            "x_ms": 150,
        },
        sessions=[session],
        corpus_id="sha256:" + "ab" * 32,
    ).to_json_dict()

    assert doc["sessions"][0]["finals_received"] == 3
    assert doc["sessions"][0]["final_text"] == "hello world again"
    assert doc["result"]["finals_received"] == 3
    assert validate(doc) == []


def test_join_final_texts_joins_in_arrival_order_and_skips_the_empty() -> None:
    assert join_final_texts([]) == ""
    assert join_final_texts([{"text": "hello"}]) == "hello"
    assert join_final_texts([{"text": "hello"}, {"text": "world"}]) == "hello world"
    # An endpoint the server found nothing in, a frame with no text at all, and a
    # frame whose text is not a string: each is still a final, none is a segment.
    assert join_final_texts([{"text": "hello"}, {"text": ""}, {"text": "world"}]) == "hello world"
    assert join_final_texts([{"text": " hello "}, {"text": "  "}, {}]) == "hello"
    assert join_final_texts([{"text": None}, {"text": "world"}]) == "world"
    assert join_final_texts([{"text": "b"}, {"text": "a"}]) == "b a"
