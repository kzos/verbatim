# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The pinned normaliser, the corpus WER it feeds, and the batch-1 reference document."""

from __future__ import annotations

import json

import pytest
from verbatim_bench import constants
from verbatim_bench.wer import (
    Batch1Reference,
    ReferenceError,
    corpus_wer,
    edit_distance,
    load_batch1_reference,
    normalise,
    stream_wer_count,
    within_window,
    word_error_rate,
)

pytestmark = pytest.mark.cpu

GOOD_REFERENCE = {
    "checkpoint": "some-checkpoint",
    "chunk_ms": 160,
    "corpus_id": "sha256:" + "0" * 64,
    "dtype": "fp32_tf32",
    "wer": 0.0725,
}


def _write(tmp_path, document: object, name: str = "batch1.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_normaliser_folds_case_and_collapses_whitespace() -> None:
    assert normalise("  The   QUICK\tbrown\nfox  ") == ["the", "quick", "brown", "fox"]


def test_the_normaliser_strips_punctuation_but_keeps_contractions_whole() -> None:
    """Punctuation becomes a separator; an apostrophe does not, or a word would split."""
    assert normalise("Hello, world -- don't stop!") == ["hello", "world", "dont", "stop"]
    # The typographic apostrophe and the modifier letter are the same character here, so
    # a transcript that spells a contraction differently is not scored as an error.
    assert normalise("don\u2019t") == normalise("don't") == normalise("don\u02bct")


def test_the_normaliser_leaves_digits_alone() -> None:
    """Numbers are not spelled out, and the docstring says so rather than the code lying."""
    assert normalise("route 66") == ["route", "66"]
    assert normalise("sixty six") != normalise("66")


def test_the_normaliser_is_unicode_aware() -> None:
    """NFKC first, so a decomposed accent and a composed one are the same word."""
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    assert normalise(decomposed) == normalise(composed) == [composed]


def test_edit_distance_counts_substitutions_insertions_and_deletions() -> None:
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert edit_distance(["a", "b", "c"], ["a", "b"]) == 1
    assert edit_distance(["a", "b"], ["a", "b", "c"]) == 1
    assert edit_distance(["a", "b", "c"], []) == 3
    assert edit_distance([], ["a", "b"]) == 2


def test_one_stream_scores_against_its_own_reference() -> None:
    assert word_error_rate("one two three four", "one two three four") == 0.0
    assert word_error_rate("one two three four", "one two THREE five") == pytest.approx(0.25)
    # Nothing came back at all: every reference word is a deletion.
    assert word_error_rate("one two three four", "") == pytest.approx(1.0)


def test_a_stream_with_no_reference_words_has_no_word_error_rate() -> None:
    """Undefined is not zero. A reference of nothing scores nobody."""
    assert word_error_rate("", "whatever the server said") is None
    assert stream_wer_count("", "anything").reference_words == 0


def test_corpus_wer_pools_errors_and_words_rather_than_averaging_streams() -> None:
    """The distinguishing case: one short utterance wrong, one long utterance right.

    Averaging the two per-stream rates gives 0.5. Pooling gives one error in eleven
    words. A benchmark that averaged would let a single short stream dominate a corpus.
    """
    ten_words = "one two three four five six seven eight nine ten"
    pooled = corpus_wer([("yes", "no"), (ten_words, ten_words)])
    assert (pooled.errors, pooled.reference_words) == (1, 11)
    assert pooled.wer == pytest.approx(1.0 / 11.0)


def test_a_corpus_of_empty_references_has_no_wer_at_all() -> None:
    assert corpus_wer([("", "something"), ("", "")]).wer is None
    assert corpus_wer([]).wer is None


def test_a_stream_that_returned_nothing_contributes_its_whole_reference() -> None:
    """A refused, dropped or final-less stream is deletions, not an exclusion."""
    pooled = corpus_wer([("one two three", "one two three"), ("four five six", "")])
    assert (pooled.errors, pooled.reference_words) == (3, 6)
    assert pooled.wer == pytest.approx(0.5)


def test_the_window_is_the_frozen_absolute_one() -> None:
    assert within_window(0.5, 0.5) is True
    assert within_window(0.5, 0.5 - constants.WER_WINDOW_ABSOLUTE / 2) is True
    assert within_window(0.5, 0.5 - constants.WER_WINDOW_ABSOLUTE * 1.5) is False
    assert within_window(0.5, 0.5 + constants.WER_WINDOW_ABSOLUTE * 1.5) is False


def test_a_reference_document_round_trips(tmp_path) -> None:
    reference = load_batch1_reference(_write(tmp_path, GOOD_REFERENCE))
    assert reference == Batch1Reference(
        checkpoint=GOOD_REFERENCE["checkpoint"],
        chunk_ms=GOOD_REFERENCE["chunk_ms"],
        corpus_id=GOOD_REFERENCE["corpus_id"],
        dtype=GOOD_REFERENCE["dtype"],
        wer=GOOD_REFERENCE["wer"],
    )
    assert reference.to_json_dict() == GOOD_REFERENCE


@pytest.mark.parametrize("missing", sorted(GOOD_REFERENCE))
def test_a_reference_missing_any_coordinate_is_refused(tmp_path, missing: str) -> None:
    """All four coordinates plus the number, or the comparison cannot be justified."""
    document = {key: value for key, value in GOOD_REFERENCE.items() if key != missing}
    with pytest.raises(ReferenceError, match=missing):
        load_batch1_reference(_write(tmp_path, document))


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path) -> None:
    """A misspelled `corpus_id` that fell through would leave the coordinate unchecked."""
    document = dict(GOOD_REFERENCE)
    document["corpus"] = document.pop("corpus_id")
    with pytest.raises(ReferenceError):
        load_batch1_reference(_write(tmp_path, document))


@pytest.mark.parametrize(
    "override",
    [
        {"wer": "0.07"},
        {"wer": -0.1},
        {"wer": True},
        {"chunk_ms": "160"},
        {"chunk_ms": 0},
        {"checkpoint": ""},
        {"dtype": 3},
        {"corpus_id": None},
    ],
)
def test_a_reference_with_an_unusable_value_is_refused(tmp_path, override: dict) -> None:
    document = dict(GOOD_REFERENCE)
    document.update(override)
    with pytest.raises(ReferenceError):
        load_batch1_reference(_write(tmp_path, document))


def test_an_unreadable_or_malformed_reference_is_refused(tmp_path) -> None:
    with pytest.raises(ReferenceError):
        load_batch1_reference(tmp_path / "does-not-exist.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReferenceError):
        load_batch1_reference(broken)
    listy = tmp_path / "listy.json"
    listy.write_text("[]", encoding="utf-8")
    with pytest.raises(ReferenceError):
        load_batch1_reference(listy)
