# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The scoring rules behind every published number about the words that changed.

Two things are pinned. The rules themselves, on cases small enough to check by eye, including
the ones an earlier, simpler rule got wrong. And the derived record: it must be exactly what the
script produces from the committed probe records today, so it cannot go stale silently.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "rows" / "exploratory"

_spec = importlib.util.spec_from_file_location(
    "unstable_words", ROOT / "scripts" / "unstable_words.py"
)
assert _spec is not None and _spec.loader is not None
uw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uw)

NO_WORDS: frozenset[str] = frozenset()


def one(reference: str, alone: str, in_batch: str) -> dict:
    return uw.analyse(
        [{"librispeech_id": "x", "reference": reference, "alone": alone, "in_batch": in_batch}],
        NO_WORDS,
    )


def test_a_split_word_is_a_multi_word_place_and_the_split_side_is_right() -> None:
    r = one(
        "i told you a fisher boy cried",
        "i told you a fisherboy cried",
        "i told you a fisher boy cried",
    )
    assert (r["places"], r["alone_right"], r["batch_right"], r["neither_right"]) == (1, 0, 1, 0)
    assert r["place_shapes"] == {"one_for_one": 0, "multi_word": 1, "one_side_empty": 0}


def test_an_empty_side_is_not_right_when_it_also_drops_the_reference_word() -> None:
    # The earlier rule scored the empty side right because the other side was wrong.
    r = one("heyday here are tripes", "haydy here are tripes", "here are tripes")
    assert (r["alone_right"], r["batch_right"], r["neither_right"]) == (0, 0, 1)


def test_an_empty_side_is_right_when_the_other_side_only_inserted() -> None:
    r = one("the cat sat", "the cat sat", "the the cat sat")
    assert (r["alone_right"], r["batch_right"], r["neither_right"]) == (1, 0, 0)


def test_matching_words_around_a_missing_reference_word_are_not_right() -> None:
    # "big dog" matches word for word, but the reference's "red" is missing between them.
    r = one("the big red dog ran", "the big dog ran", "the bigdog ran")
    assert (r["alone_right"], r["batch_right"], r["neither_right"]) == (0, 0, 1)


def test_both_sides_can_never_be_right() -> None:
    r = one("seven foals", "seven foals", "sevenfolds")
    assert r["both_right"] == 0 and r["alone_right"] == 1


def test_word_error_counts_are_levenshtein_over_the_reference() -> None:
    r = one("a b c d", "a x c d", "a c d")
    assert (r["word_errors_alone"], r["word_errors_in_batch"], r["reference_words"]) == (1, 1, 4)


def test_a_negation_changing_counts_and_a_spelling_change_does_not() -> None:
    assert (
        len(
            one("didn't mean nowt sir", "didn't mean not sir", "didn't mean nought sir")[
                "negation_or_number_places"
            ]
        )
        == 1
    )
    assert one("cried archy", "cried archie", "cried archy")["negation_or_number_places"] == []


def test_sign_test_is_two_sided_and_capped() -> None:
    assert uw.sign_test(5, 5) == 1.0
    assert uw.sign_test(119, 96) == pytest.approx(0.1333, abs=5e-4)


def _record() -> dict:
    return json.loads((ROWS / "unstable-words-2026-09-24.json").read_text(encoding="utf-8"))


def test_the_derived_record_names_the_inputs_it_was_built_from() -> None:
    record = _record()
    for name, digest in record["inputs"].items():
        assert hashlib.sha256((ROWS / name).read_bytes()).hexdigest() == digest, name


def test_the_derived_record_is_what_the_script_produces_today() -> None:
    record = _record()
    runs = json.loads((ROWS / uw.B300).read_text(encoding="utf-8"))["runs"]["bfloat16"]
    arms = json.loads((ROWS / uw.A6000_BF16).read_text(encoding="utf-8"))["arms"]
    sources = {
        "b300_bf16_equalised": runs["equalised"]["divergences"],
        "b300_bf16_ragged": runs["ragged"]["divergences"],
        "a6000_bf16_equalised": arms["controlled"]["divergences"],
        "a6000_bf16_varying": arms["uncontrolled"]["divergences"],
    }
    wordlist = Path(record["wordlist"]["path"])
    have_list = (
        wordlist.is_file()
        and hashlib.sha256(wordlist.read_bytes()).hexdigest() == record["wordlist"]["sha256"]
    )
    words = (
        frozenset(
            w.strip().lower()
            for w in wordlist.read_text(encoding="utf-8").splitlines()
            if w.strip()
        )
        if have_list
        else NO_WORDS
    )
    for arm, divergences in sources.items():
        fresh = uw.analyse(divergences, words)
        published = record["arms"][arm]
        if not have_list:
            # Everything but the dictionary figures is independent of the word list.
            fresh = {k: v for k, v in fresh.items() if k != "out_of_dictionary"}
            published = {k: v for k, v in published.items() if k != "out_of_dictionary"}
        assert fresh == published, arm
    if not have_list:
        pytest.skip(
            f"word list {wordlist} with the recorded SHA-256 is absent;"
            " dictionary figures not re-derived"
        )
