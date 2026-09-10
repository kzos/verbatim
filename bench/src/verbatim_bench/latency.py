# SPDX-License-Identifier: Apache-2.0
"""Pure latency reductions over the events recorded by the load generator.

The harness keeps the event-to-sample rules here so a result can be recomputed
without reconnecting to the server.  In particular, arrival order alone is not
enough to identify the audio represented by a partial under backlog.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PartialEvent:
    t_recv_s: float
    watermark_s: float | None
    text: str


@dataclass(frozen=True, slots=True)
class ChunkSend:
    index: int
    t_send_s: float
    audio_end_s: float


@dataclass(frozen=True, slots=True)
class ReferenceWord:
    position: int
    word: str
    audio_end_s: float


def watermark_latencies(
    sends: Sequence[ChunkSend], partials: Sequence[PartialEvent]
) -> list[float]:
    """Return milliseconds until the first partial that covers each sent chunk.

    A partial without a watermark is deliberately skipped.  The cursor is left
    on a matching partial so one response that advances the watermark may cover
    several chunks, which is possible when a server emits coalesced updates.
    """
    samples: list[float] = []
    partial_index = 0
    for send in sends:
        while partial_index < len(partials):
            event = partials[partial_index]
            watermark = event.watermark_s
            if watermark is not None and watermark >= send.audio_end_s:
                samples.append((event.t_recv_s - send.t_send_s) * 1000.0)
                break
            partial_index += 1
    return samples


def unmatched_chunks(sends: Sequence[ChunkSend], partials: Sequence[PartialEvent]) -> int:
    """Count sent chunks for which no received watermark reaches their end."""
    matched = len(watermark_latencies(sends, partials))
    return len(sends) - matched


def _words(text: str) -> list[str]:
    return text.split()


def word_emission_latencies(
    words: Sequence[ReferenceWord],
    partials: Sequence[PartialEvent],
    *,
    t_audio_start_s: float,
) -> list[float]:
    """Return milliseconds from each word's audio end to its first stable partial.

    A candidate appearance is stable only when every later partial keeps the
    same word at the same position.  This lets a later, corrected appearance
    score while an earlier transient hypothesis is excluded.
    """
    tokenised = [_words(event.text) for event in partials]
    latencies: list[float] = []
    for reference in words:
        if reference.position < 0:
            continue
        for candidate, tokens in enumerate(tokenised):
            if reference.position >= len(tokens) or tokens[reference.position] != reference.word:
                continue
            stable = all(
                reference.position < len(later) and later[reference.position] == reference.word
                for later in tokenised[candidate:]
            )
            if stable:
                t_audio_end = t_audio_start_s + reference.audio_end_s
                latencies.append((partials[candidate].t_recv_s - t_audio_end) * 1000.0)
                break
    return latencies


def retraction_rate(partials: Sequence[PartialEvent], final_text: str) -> float:
    """Return retracted partial words per 100 words in the final transcript."""
    tokenised = [_words(event.text) for event in partials]
    retracted: set[tuple[int, str]] = set()
    for tokens_index, tokens in enumerate(tokenised):
        for position, word in enumerate(tokens):
            if any(
                position >= len(later) or later[position] != word
                for later in tokenised[tokens_index + 1 :]
            ):
                retracted.add((position, word))
    final_words = len(_words(final_text))
    if final_words == 0:
        return 0.0
    return len(retracted) * 100.0 / final_words


def first_partial_latency_ms(
    partials: Sequence[PartialEvent], *, t_first_sample_s: float
) -> float | None:
    """Return latency to the first non-empty partial, or ``None`` if none arrived."""
    for event in partials:
        if _words(event.text):
            return (event.t_recv_s - t_first_sample_s) * 1000.0
    return None


def final_latency_ms(*, t_recv_final_s: float, t_last_sample_s: float) -> float:
    """Return milliseconds from the last sent sample to the received final."""
    return (t_recv_final_s - t_last_sample_s) * 1000.0
