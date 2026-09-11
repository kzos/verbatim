# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The four demo-protocol message shapes and the client-side parsers."""

from __future__ import annotations

import pytest

from verbatim.core.errors import ErrorCode, InvalidArgument
from verbatim.protocols.base import Word
from verbatim.protocols.ws.frames import (
    ErrorFrame,
    FinalFrame,
    PartialFrame,
    SessionFrame,
    parse_client_text,
    parse_query,
)

pytestmark = pytest.mark.cpu


def test_frame_json_is_compact_and_ordered() -> None:
    assert (
        SessionFrame(id="abc", chunk_ms=160, invariance_class="x").to_json()
        == '{"type":"session","id":"abc","chunk_ms":160,"invariance_class":"x"}'
    )
    assert (
        PartialFrame(text="the quick", audio_s=0.32).to_json()
        == '{"type":"partial","text":"the quick","audio_s":0.32}'
    )
    assert (
        FinalFrame(
            text="the",
            audio_s=0.16,
            words=(Word(word="the", start_ms=0, end_ms=160),),
        ).to_json()
        == '{"type":"final","text":"the","words":[{"w":"the","s":0,"e":160}],"audio_s":0.16}'
    )
    assert (
        ErrorFrame(code=ErrorCode.INVALID_ARGUMENT, message="bad chunk").to_json()
        == '{"type":"error","code":"INVALID_ARGUMENT","message":"bad chunk"}'
    )


def test_final_frame_omits_words_when_none() -> None:
    raw = FinalFrame(text="the", audio_s=0.16).to_json()
    assert '"words"' not in raw
    assert raw == '{"type":"final","text":"the","audio_s":0.16}'


def test_final_frame_word_times_are_integers() -> None:
    raw = FinalFrame(
        text="the",
        audio_s=0.16,
        words=(Word(word="the", start_ms=0, end_ms=160),),
    ).to_json()
    assert '"s":0,"e":160' in raw
    assert "0.0" not in raw


def test_parse_client_text_accepts_end() -> None:
    assert parse_client_text('{"type": "end"}') == "end"


@pytest.mark.parametrize(
    "payload", ["{}", '{"type":"start"}', "not json", '["end"]', '{"type": 1}', ""]
)
def test_parse_client_text_rejects_junk(payload: str) -> None:
    with pytest.raises(InvalidArgument):
        parse_client_text(payload)


def test_parse_query_defaults() -> None:
    options = parse_query("")
    assert options.chunk_ms == 160
    assert options.language_code == "en-US"
    assert options.word_timestamps is False


def test_parse_query_reads_all_three() -> None:
    options = parse_query("chunk_ms=560&lang=de-DE&words=1")
    assert options.chunk_ms == 560
    assert options.language_code == "de-DE"
    assert options.word_timestamps is True


def test_parse_query_ignores_unknown_parameters() -> None:
    options = parse_query("foo=bar&chunk_ms=160")
    assert options.chunk_ms == 160


def test_parse_query_defaults_the_wire_to_pcm16_at_16k() -> None:
    options = parse_query("chunk_ms=160")
    assert options.wire_encoding == "LINEAR_PCM"
    assert options.wire_sample_rate_hz == 16000


def test_parse_query_reads_encoding_and_sample_rate() -> None:
    options = parse_query("encoding=MuLaw&sample_rate_hz=8000")
    assert options.wire_encoding == "MULAW"
    assert options.wire_sample_rate_hz == 8000
    assert options.sample_rate_hz == 16000  # the recognizer's rate, not the wire's
    assert parse_query("encoding=alaw").wire_encoding == "ALAW"
    assert parse_query("encoding=linear-pcm").wire_encoding == "LINEAR_PCM"


@pytest.mark.parametrize("query", ["encoding=flac", "encoding=", "encoding=oggopus"])
def test_parse_query_rejects_an_encoding_not_served(query: str) -> None:
    with pytest.raises(InvalidArgument, match="encoding"):
        parse_query(query)


@pytest.mark.parametrize(
    "query",
    ["sample_rate_hz=8001", "sample_rate_hz=016000", "sample_rate_hz=16k", "sample_rate_hz="],
)
def test_parse_query_rejects_an_unserved_or_non_canonical_rate(query: str) -> None:
    with pytest.raises(InvalidArgument, match="sample_rate_hz"):
        parse_query(query)


def test_parse_query_rejects_a_duplicated_encoding() -> None:
    with pytest.raises(InvalidArgument, match="duplicated"):
        parse_query("encoding=mulaw&encoding=alaw")


@pytest.mark.parametrize("query", ["chunk_ms=100", "chunk_ms=abc"])
def test_parse_query_rejects_a_bad_chunk_ms(query: str) -> None:
    with pytest.raises(InvalidArgument, match="chunk_ms"):
        parse_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "chunk_ms=1_60",
        "chunk_ms=+160",
        "chunk_ms= 160 ",
        "chunk_ms=\uff11\uff16\uff10",
        "chunk_ms=160&chunk_ms=160",
    ],
)
def test_chunk_ms_rejects_non_canonical_spellings(query: str) -> None:
    with pytest.raises(InvalidArgument, match="chunk_ms"):
        parse_query(query)


def test_parse_query_accepts_canonical_chunk_ms() -> None:
    assert parse_query("chunk_ms=160").chunk_ms == 160
