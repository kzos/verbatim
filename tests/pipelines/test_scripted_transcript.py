# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""`ScriptedTranscript`, the per-stream script behind the fake adapter's scripted mode.

Nothing on the wire calls it; the engine path drives it through
`FakePipelineAdapter`. These pin the arithmetic the protocol tests rely on.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from verbatim.core.errors import InvalidArgument
from verbatim.pipelines.fake import ScriptedTranscript
from verbatim.protocols.base import SessionOptions

pytestmark = pytest.mark.cpu


def _script(
    chunk_ms: int = 160,
    script: Sequence[str] | None = None,
    partial_every: int = 1,
) -> ScriptedTranscript:
    return ScriptedTranscript(
        SessionOptions(chunk_ms=chunk_ms), script=script, partial_every=partial_every
    )


def test_add_chunk_rejects_a_short_buffer() -> None:
    source = _script()
    with pytest.raises(InvalidArgument, match="5120"):
        source.add_chunk(b"\x00" * (source.options.chunk_bytes - 1))


def test_add_chunk_rejects_a_long_buffer() -> None:
    source = _script()
    with pytest.raises(InvalidArgument, match="5120"):
        source.add_chunk(b"\x00" * (source.options.chunk_bytes + 1))


def test_partials_are_the_running_prefix() -> None:
    source = _script()
    texts = [source.add_chunk(b"\x00" * source.options.chunk_bytes)[0].text for _ in range(4)]
    assert texts == ["the", "the quick", "the quick brown", "the quick brown fox"]


def test_audio_processed_is_exact() -> None:
    source = _script(chunk_ms=160)
    hypos = [source.add_chunk(b"\x00" * source.options.chunk_bytes)[0] for _ in range(5)]
    assert hypos[-1].audio_processed_s == pytest.approx(0.8)
    source = _script(chunk_ms=1120)
    hypos = [source.add_chunk(b"\x00" * source.options.chunk_bytes)[0] for _ in range(3)]
    assert hypos[-1].audio_processed_s == pytest.approx(3.36)


def test_finalize_emits_one_final_with_the_whole_script() -> None:
    source = _script()
    source.add_chunk(b"\x00" * source.options.chunk_bytes)
    finals = source.finalize()
    assert len(finals) == 1
    assert finals[0].is_final is True
    assert finals[0].text == "the quick brown fox jumps over the lazy dog"


def test_add_chunk_after_finalize_raises() -> None:
    source = _script()
    source.add_chunk(b"\x00" * source.options.chunk_bytes)
    source.finalize()
    with pytest.raises(InvalidArgument):
        source.add_chunk(b"\x00" * source.options.chunk_bytes)


def test_words_are_derived_from_the_chunk_index() -> None:
    source = _script(chunk_ms=160, script=("the", "quick", "brown"))
    source.add_chunk(b"\x00" * source.options.chunk_bytes)
    final = source.finalize()[0]
    assert [(w.word, w.start_ms, w.end_ms) for w in final.words] == [
        ("the", 0, 160),
        ("quick", 160, 320),
        ("brown", 320, 480),
    ]


def test_partial_every_throttles_partials() -> None:
    source = _script(partial_every=2)
    partials = [h for _ in range(4) for h in source.add_chunk(b"\x00" * source.options.chunk_bytes)]
    assert len(partials) == 2


def test_session_options_rejects_an_unsupported_chunk_ms() -> None:
    with pytest.raises(InvalidArgument, match="80, 160, 560, 1120"):
        SessionOptions(chunk_ms=100)


@pytest.mark.parametrize(
    ("chunk_ms", "chunk_bytes"),
    [(80, 2560), (160, 5120), (560, 17920), (1120, 35840)],
)
def test_chunk_bytes_arithmetic(chunk_ms: int, chunk_bytes: int) -> None:
    assert SessionOptions(chunk_ms=chunk_ms).chunk_bytes == chunk_bytes
