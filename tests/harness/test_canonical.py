# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for canonical digests and the keyed checksum."""

from __future__ import annotations

import json
import re

import pytest
from verbatim_bench.canonical import (
    FinalRecord,
    canonicalise_finals,
    compute_checksum,
    errored_sessions,
    finals_digest,
    finals_from_results,
    partials_digest,
    stamp_checksum,
    verify_checksum,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _finals() -> list[FinalRecord]:
    return [
        FinalRecord("utt-2", "goodbye world", (("goodbye", 0, 200), ("world", 200, 400))),
        FinalRecord("utt-1", "hello world", (("hello", 0, 160), ("world", 160, 320))),
    ]


def test_digest_is_stable() -> None:
    first = finals_digest(_finals())
    second = finals_digest(_finals())
    assert first == second
    assert _HEX64.match(first) is not None


def test_digest_is_order_independent() -> None:
    assert finals_digest(_finals()) == finals_digest(list(reversed(_finals())))


def test_digest_changes_on_a_single_character() -> None:
    left = finals_digest([FinalRecord("s", "hello", ())])
    right = finals_digest([FinalRecord("s", "hellp", ())])
    assert left != right


def test_digest_changes_on_a_word_timestamp() -> None:
    left = finals_digest([FinalRecord("s", "w", (("w", 0, 160),))])
    right = finals_digest([FinalRecord("s", "w", (("w", 0, 161),))])
    assert left != right


def test_digest_does_not_normalise_case() -> None:
    left = finals_digest([FinalRecord("s", "The dog", ())])
    right = finals_digest([FinalRecord("s", "the dog", ())])
    assert left != right


def test_digest_does_not_normalise_punctuation() -> None:
    left = finals_digest([FinalRecord("s", "hello, world", ())])
    right = finals_digest([FinalRecord("s", "hello world", ())])
    assert left != right


def test_digest_does_not_normalise_whitespace() -> None:
    left = finals_digest([FinalRecord("s", "a  b", ())])
    right = finals_digest([FinalRecord("s", "a b", ())])
    assert left != right


def test_digest_does_not_normalise_unicode() -> None:
    nfc = "caf\u00e9"
    nfd = "cafe\u0301"
    assert nfc != nfd
    assert finals_digest([FinalRecord("s", nfc, ())]) != finals_digest([FinalRecord("s", nfd, ())])


def test_digest_sorts_by_byte_order() -> None:
    finals = [
        FinalRecord("a9", "nine", ()),
        FinalRecord("A1", "upper", ()),
        FinalRecord("a10", "ten", ()),
    ]
    serialised = canonicalise_finals(finals)
    assert [row["stream_id"] for row in json.loads(serialised)] == ["A1", "a10", "a9"]
    assert finals_digest(finals) == finals_digest(list(reversed(finals)))


def test_word_times_must_be_integers() -> None:
    with pytest.raises(TypeError):
        canonicalise_finals([FinalRecord("s", "w", (("w", 0, 160.5),))])


def test_errored_sessions_are_excluded_and_counted() -> None:
    doc = {
        "sessions": [
            {"stream_id": "a", "final_text": "one", "words": [], "error": None},
            {"stream_id": "b", "final_text": "two", "words": [], "error": None},
            {"stream_id": "c", "final_text": "three", "words": [], "error": "TimeoutError: x"},
        ]
    }
    assert len(finals_from_results(doc)) == 2
    assert errored_sessions(doc) == 1


def test_partials_digest_is_separate_from_finals_digest() -> None:
    partials = partials_digest({"a": ["x"]})
    finals = finals_digest([FinalRecord("a", "x", ())])
    assert partials != finals
    assert partials == partials_digest({"a": ["x"]})
    assert partials != partials_digest({"a": ["y"]})


def test_checksum_round_trips() -> None:
    doc: dict = {"x": 1.0, "nested": {"a": [1, 2, 3]}}
    stamp_checksum(doc)
    assert verify_checksum(doc) is True


def test_checksum_detects_an_edit() -> None:
    doc: dict = {"latency": {"p95": 10.0}}
    stamp_checksum(doc)
    doc["latency"]["p95"] = 11.0
    assert verify_checksum(doc) is False


def test_checksum_ignores_the_checksum_field_itself() -> None:
    doc: dict = {"x": 1.0}
    stamp_checksum(doc)
    first = compute_checksum(doc)
    stamp_checksum(doc)
    assert compute_checksum(doc) == first
    assert verify_checksum(doc) is True
