# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Per-stream and corpus WER against the batch-1 reference, with a pinned normaliser.

The methodology's second criterion is that corpus WER is within `WER_WINDOW_ABSOLUTE`
absolute of the batch-1 reference for the same checkpoint, chunk, corpus and dtype. Two
things are needed for it: a way to turn a stream's transcript and its reference text into
an error count, and a way for a run to be told what the batch-1 reference was.

**The normaliser here is this harness's own, and it is not the frozen
`<normaliser-package-version>` the methodology names.** That placeholder is unfilled
because no normaliser package is pinned yet, so the rule applied to both sides of the
comparison is the one written in `normalise` below and nothing else. A reference number
produced by some other normaliser is not comparable with a number produced by this one,
which is why `Batch1Reference` carries the coordinates it was measured at and a run
refuses a reference that does not describe it.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verbatim_bench import constants


class ReferenceError(ValueError):
    """A batch-1 reference document that cannot be read or does not describe a WER."""


#: Apostrophes are deleted rather than replaced by a space, so a contraction stays one
#: word. ASCII, the right single quotation mark, and the modifier letter apostrophe.
_APOSTROPHES: str = "'\u2019\u02bc"

_REFERENCE_KEYS: tuple[str, ...] = ("checkpoint", "chunk_ms", "corpus_id", "dtype", "wer")


def normalise(text: str) -> list[str]:
    """Return the words of `text` under the harness's pinned normalisation.

    The whole rule, in order, and nothing else: compatibility-normalise to NFKC;
    casefold; delete every apostrophe so a contraction stays one word; replace every
    other punctuation or symbol character with a space; split on whitespace.

    What it deliberately does not do: expand contractions, spell out or parse numbers,
    map spellings between dialects, or drop filler words. Each of those is a decision a
    real normaliser package makes deliberately, and making them here in passing would
    bury the choice inside a benchmark instead of pinning it where it can be cited.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    kept: list[str] = []
    for character in folded:
        if character in _APOSTROPHES:
            continue
        kept.append(" " if unicodedata.category(character)[0] in ("P", "S") else character)
    return "".join(kept).split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Word-level Levenshtein distance with unit substitution, insertion and deletion.

    Two rows rather than a full matrix: a corpus is many short utterances, so the
    quadratic term is only ever over one utterance at a time.
    """
    previous = list(range(len(hypothesis) + 1))
    for index, reference_word in enumerate(reference, start=1):
        current = [index]
        for position, hypothesis_word in enumerate(hypothesis, start=1):
            substitution = previous[position - 1] + (0 if reference_word == hypothesis_word else 1)
            current.append(min(substitution, previous[position] + 1, current[position - 1] + 1))
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class WerCount:
    """Errors and reference words, kept apart so several streams can be pooled.

    A corpus WER is the pooled ratio, never the mean of per-stream ratios: a one-word
    utterance transcribed wrongly would otherwise weigh as much as a hundred-word one
    transcribed perfectly.
    """

    errors: int
    reference_words: int

    @property
    def wer(self) -> float | None:
        """The ratio, or `None` when there are no reference words to divide by.

        Undefined is not zero. A corpus whose references are all empty says nothing
        about any transcriber's accuracy, and returning 0.0 there would award a perfect
        score for measuring nothing.
        """
        if self.reference_words == 0:
            return None
        return self.errors / self.reference_words

    def __add__(self, other: WerCount) -> WerCount:
        return WerCount(self.errors + other.errors, self.reference_words + other.reference_words)


def stream_wer_count(reference_text: str, hypothesis_text: str) -> WerCount:
    """Return one stream's error count, under the pinned normaliser."""
    reference = normalise(reference_text)
    hypothesis = normalise(hypothesis_text)
    return WerCount(edit_distance(reference, hypothesis), len(reference))


def word_error_rate(reference_text: str, hypothesis_text: str) -> float | None:
    """Return one stream's WER, or `None` when its reference has no words."""
    return stream_wer_count(reference_text, hypothesis_text).wer


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> WerCount:
    """Pool `(reference_text, hypothesis_text)` pairs into one corpus error count.

    A stream whose transcript is empty, because it was refused or dropped or simply
    produced no final, contributes its whole reference as deletions. That is the honest
    reading: the words were spoken and none came back. The integrity criterion counts
    those same streams separately, so a rung never has to infer a broken stream from a
    WER that moved.
    """
    total = WerCount(0, 0)
    for reference_text, hypothesis_text in pairs:
        total = total + stream_wer_count(reference_text, hypothesis_text)
    return total


def within_window(measured: float, reference: float) -> bool:
    """Whether a measured corpus WER is inside the frozen window around the reference."""
    return abs(measured - reference) <= constants.WER_WINDOW_ABSOLUTE


@dataclass(frozen=True, slots=True)
class Batch1Reference:
    """The batch-1 corpus WER a rung compares itself against, and where it came from.

    The criterion is defined for one `(checkpoint, chunk, corpus_id, dtype)` and means
    nothing across any of them, so the number travels with all four. A rung that cannot
    show the four agree does not evaluate WER at all.
    """

    checkpoint: str
    chunk_ms: int
    corpus_id: str
    dtype: str
    wer: float

    def describes(self, *, checkpoint: str, chunk_ms: int, corpus_id: str, dtype: str) -> bool:
        """Whether this reference was measured at exactly the given coordinates."""
        return (
            self.checkpoint == checkpoint
            and self.chunk_ms == chunk_ms
            and self.corpus_id == corpus_id
            and self.dtype == dtype
        )

    def to_json_dict(self) -> dict[str, Any]:
        """The document form, so a run can record which reference was in force."""
        return {
            "checkpoint": self.checkpoint,
            "chunk_ms": self.chunk_ms,
            "corpus_id": self.corpus_id,
            "dtype": self.dtype,
            "wer": self.wer,
        }


def load_batch1_reference(path: Path) -> Batch1Reference:
    """Read a batch-1 reference document: a JSON object of exactly five keys.

    `checkpoint`, `chunk_ms`, `corpus_id` and `dtype` are the coordinates the criterion
    is defined at; `wer` is the corpus WER measured there, by this module's normaliser.
    Unknown keys are rejected rather than ignored, because a misspelled `corpus_id` that
    fell through would leave that coordinate unchecked, and the comparison would then be
    made against a reference nobody verified.
    """
    reference_path = Path(path)
    try:
        raw = reference_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReferenceError(f"cannot read batch-1 reference {reference_path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReferenceError(
            f"batch-1 reference {reference_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ReferenceError(f"batch-1 reference {reference_path} is not a JSON object")
    missing = [key for key in _REFERENCE_KEYS if key not in document]
    if missing:
        raise ReferenceError(f"batch-1 reference {reference_path} is missing {', '.join(missing)}")
    unknown = sorted(set(document) - set(_REFERENCE_KEYS))
    if unknown:
        raise ReferenceError(
            f"batch-1 reference {reference_path} has unknown keys {', '.join(unknown)}"
        )
    for name in ("checkpoint", "corpus_id", "dtype"):
        value = document[name]
        if not isinstance(value, str) or not value:
            raise ReferenceError(
                f"batch-1 reference {reference_path}: {name} must be a non-empty string"
            )
    chunk_ms = document["chunk_ms"]
    if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int) or chunk_ms <= 0:
        raise ReferenceError(
            f"batch-1 reference {reference_path}: chunk_ms must be a positive integer"
        )
    wer = document["wer"]
    if isinstance(wer, bool) or not isinstance(wer, (int, float)) or wer < 0.0:
        raise ReferenceError(
            f"batch-1 reference {reference_path}: wer must be a non-negative number"
        )
    return Batch1Reference(
        checkpoint=document["checkpoint"],
        chunk_ms=chunk_ms,
        corpus_id=document["corpus_id"],
        dtype=document["dtype"],
        wer=float(wer),
    )
