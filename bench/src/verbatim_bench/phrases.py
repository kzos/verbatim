# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The phrase lists a biasing run sends, and which stream gets which.

With biasing on, a transcript is a function of the audio, the checkpoint, the phrase
list and its weight. A record that names only the first two has quietly weakened the
claim from "same audio, same transcript" to "same audio, and whoever last edited the
phrase list", so a book carries a digest and the digest goes on the row.

Assignment is a pure function of a stream's position in the corpus, and a stream keeps
the SAME list at every concurrency level. That is not a detail: the gate's question is
whether a transcript changes when the batch composition changes, so the only other input
has to be held fixed. What does vary across levels is which streams are in flight
together, which is what the levels are.

Half the corpus carries a list and half carries none, interleaved, so a biased row and an
unbiased row sit next to each other in the same batch -- the case where a per-row fusion
that was not really per-row would show.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PhraseBook",
    "PhraseBookError",
    "assign",
    "load_phrase_book",
    "missed_words",
    "rare_words",
]


class PhraseBookError(ValueError):
    """The phrase book could not be read, or says something it cannot mean."""


@dataclass(frozen=True, slots=True)
class PhraseBook:
    """Named phrase lists, the weight to send them at, and an unrelated control list."""

    name: str
    lists: tuple[tuple[str, tuple[str, ...]], ...]
    boost: float | None = None
    #: A list with no relation to the corpus, for the negative control: none of these
    #: words should appear in a transcript that did not already contain them.
    unrelated: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise PhraseBookError("a phrase book needs a name; it goes on the row")
        if not self.lists:
            raise PhraseBookError(f"phrase book {self.name!r} has no lists")
        for label, phrases in self.lists:
            if not label:
                raise PhraseBookError(f"phrase book {self.name!r} has an unnamed list")
            if not phrases:
                raise PhraseBookError(f"list {label!r} in phrase book {self.name!r} is empty")
            for phrase in phrases:
                if not phrase.strip():
                    raise PhraseBookError(f"list {label!r} has an empty phrase")

    @property
    def digest(self) -> str:
        """The book's identity: every list, in order, plus the weight."""
        hasher = hashlib.sha256()
        hasher.update(b"vb-phrase-book/1\n")
        hasher.update(f"boost={'' if self.boost is None else float(self.boost)!r}\n".encode())
        for label, phrases in self.lists:
            hasher.update(f"[{label}]\n".encode())
            for phrase in phrases:
                hasher.update(f"{len(phrase)}:{phrase}\n".encode())
        for phrase in self.unrelated:
            hasher.update(f"unrelated {len(phrase)}:{phrase}\n".encode())
        return hasher.hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "boost": self.boost,
            "lists": {label: list(phrases) for label, phrases in self.lists},
            "unrelated": list(self.unrelated),
        }


def load_phrase_book(path: Path) -> PhraseBook:
    """Read a phrase book: ``{"name": ..., "boost": ..., "lists": {label: [...]}}``."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PhraseBookError(f"cannot read phrase book {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise PhraseBookError(f"phrase book {path} must be a JSON object")
    raw_lists = body.get("lists")
    if not isinstance(raw_lists, dict):
        raise PhraseBookError(f"phrase book {path} has no 'lists' object")
    lists: list[tuple[str, tuple[str, ...]]] = []
    for label, phrases in raw_lists.items():
        if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
            raise PhraseBookError(f"list {label!r} in {path} must be a list of strings")
        lists.append((str(label), tuple(phrases)))
    boost = body.get("boost")
    if boost is not None and not isinstance(boost, int | float):
        raise PhraseBookError(f"'boost' in {path} must be a number or absent")
    unrelated = body.get("unrelated", [])
    if not isinstance(unrelated, list) or not all(isinstance(p, str) for p in unrelated):
        raise PhraseBookError(f"'unrelated' in {path} must be a list of strings")
    return PhraseBook(
        name=str(body.get("name") or Path(path).stem),
        lists=tuple(lists),
        boost=None if boost is None else float(boost),
        unrelated=tuple(unrelated),
    )


def assign(stream_ids: Sequence[str], book: PhraseBook) -> dict[str, tuple[str, ...]]:
    """Which stream carries which list: half of them, interleaved with unbiased streams.

    A pure function of position in the corpus, so the same corpus and the same book
    produce the same assignment in every level, in every process and on every re-run.
    Streams with no list are simply absent from the mapping.
    """
    slots: list[tuple[str, ...] | None] = [phrases for _, phrases in book.lists]
    slots.extend([None] * len(book.lists))
    out: dict[str, tuple[str, ...]] = {}
    for index, stream_id in enumerate(stream_ids):
        chosen = slots[index % len(slots)]
        if chosen is not None:
            out[stream_id] = chosen
    return out


#: Words shorter than this are too common to be worth boosting, and a list built from
#: them would be testing the model's grip on "the" rather than on a rare term.
RARE_WORD_CHARS = 7

_PUNCTUATION = ".,;:!?\"'()[]"


def _clean(word: str) -> str:
    return word.strip(_PUNCTUATION)


def rare_words(text: str, *, limit: int = 16) -> tuple[str, ...]:
    """The longer words of a reference transcript, in order, without duplicates."""
    seen: list[str] = []
    for word in text.split():
        cleaned = _clean(word)
        if len(cleaned) < RARE_WORD_CHARS or cleaned in seen:
            continue
        seen.append(cleaned)
        if len(seen) >= limit:
            break
    return tuple(seen)


def missed_words(reference: str, transcript: str, *, limit: int = 32) -> tuple[str, ...]:
    """Reference words the transcript did not produce, in order, without duplicates.

    This is what the positive control boosts, and the distinction from ``rare_words``
    was learned the hard way. A control built from a clip's reference regardless of
    whether the model got those words right boosts words the model already emitted, and
    a correct boosting implementation then changes nothing -- so the control reported
    "biasing reached nothing" on a server where biasing demonstrably worked. Measured on
    a B300, 2026-09-14: three clips, every boosted word already present in the bare
    transcript, zero change; the same server with the words it had MISSED boosted moved
    six transcripts of twelve and recovered `risdongram` to `risdon graeme`,
    `ray stroke` to `raystoke` and `important` to `importance`.

    Comparison is case-insensitive and ignores surrounding punctuation, because the
    question is whether the word was recognised, not how it was typeset.
    """
    said = {_clean(word).lower() for word in transcript.split()}
    seen: list[str] = []
    for word in reference.split():
        cleaned = _clean(word)
        if not cleaned or cleaned.lower() in said or cleaned in seen:
            continue
        seen.append(cleaned)
        if len(seen) >= limit:
            break
    return tuple(seen)
