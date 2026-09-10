# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the engine result cadence, without a server or a thread."""

from __future__ import annotations

from verbatim.core.types import StepResult, Word
from verbatim.protocols.base import SessionOptions
from verbatim.protocols.emit import emissions_for

OPTIONS = SessionOptions(chunk_ms=160)


def _result(
    *,
    partial: str = "hello",
    final: str | None = None,
    audio: float = 0.16,
    valid: int = 2560,
    is_last: bool = False,
    confidence: float = 1.0,
    words: tuple[Word, ...] = (),
) -> StepResult:
    return StepResult(
        stream_id=1,
        tick_id=3,
        partial_text=partial,
        final_text=final,
        audio_processed_s=audio,
        eager=False,
        words=words,
        confidence=confidence,
        valid_samples=valid,
        is_last=is_last,
    )


def test_partial_for_a_consumed_chunk() -> None:
    emission = emissions_for(_result(), OPTIONS)
    assert [(h.text, h.is_final, h.audio_processed_s) for h in emission.hypotheses] == [
        ("hello", False, 0.16)
    ]


def test_no_partial_when_interim_results_is_false() -> None:
    options = SessionOptions(chunk_ms=160, interim_results=False)
    emission = emissions_for(_result(final="done"), options)
    assert [(h.text, h.is_final) for h in emission.hypotheses] == [("done", True)]


def test_is_last_with_a_real_tail_emits_partial_then_final_with_the_same_stamp() -> None:
    emission = emissions_for(_result(final="done", is_last=True, valid=100), OPTIONS)
    assert [(h.text, h.is_final) for h in emission.hypotheses] == [
        ("hello", False),
        ("done", True),
    ]
    assert {h.audio_processed_s for h in emission.hypotheses} == {0.16}


def test_is_last_with_zero_valid_samples_emits_only_a_final() -> None:
    emission = emissions_for(_result(final="done", is_last=True, valid=0), OPTIONS)
    assert [(h.text, h.is_final) for h in emission.hypotheses] == [("done", True)]


def test_an_abort_frame_emits_nothing() -> None:
    emission = emissions_for(_result(partial="", valid=0, is_last=True), OPTIONS)
    assert emission.hypotheses == ()
    assert emission.end_of_session is True


def test_a_mid_stream_final_is_followed_by_an_empty_partial_and_the_is_last_final_is_not() -> None:
    mid_stream = emissions_for(_result(final="utterance"), OPTIONS)
    assert [(h.text, h.is_final) for h in mid_stream.hypotheses] == [
        ("hello", False),
        ("utterance", True),
        ("", False),
    ]
    last = emissions_for(_result(final="utterance", is_last=True), OPTIONS)
    assert [(h.text, h.is_final) for h in last.hypotheses] == [
        ("hello", False),
        ("utterance", True),
    ]


def test_an_unchanged_partial_is_still_emitted() -> None:
    emission = emissions_for(_result(partial="same", audio=0.32), OPTIONS)
    assert len(emission.hypotheses) == 1
    assert emission.hypotheses[0].text == "same"
    assert emission.hypotheses[0].audio_processed_s == 0.32


def test_words_and_confidence_ride_on_the_final_only() -> None:
    words = (Word("hello", 0, 160, confidence=0.75),)
    emission = emissions_for(
        _result(final="hello", words=words, confidence=0.75),
        OPTIONS,
    )
    partial, final, empty_partial = emission.hypotheses
    assert partial.words == ()
    assert partial.confidence == 1.0
    assert final.words == words
    assert final.confidence == 0.75
    assert empty_partial.text == ""


def test_end_of_session_is_exactly_is_last() -> None:
    assert emissions_for(_result(), OPTIONS).end_of_session is False
    assert emissions_for(_result(is_last=True), OPTIONS).end_of_session is True
