# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The step-1 scorer: normalisation, contractions, flags and their nulls, the confidence
comparison, timing shifts, record checks and controls.

Every case is small enough to check by eye. Each record check has a test that starts from a
valid arm, changes one field, and matches that check's own message. A check with two halves
(side a and side b of a text or word comparison) has one input per half.

The arms are built the way ``probes/stock_divergence.py`` now writes them: every checked
recording also has an ``every_recording`` entry, whose words carry a confidence (0.0 unless a
test gives one), and ``confidence_only_divergent`` counts the recordings whose confidences alone
differ. ``every=False`` builds an arm without them, as the finished record of 2026-09-24 is.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import random
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("step1_places", ROOT / "scripts" / "step1_places.py")
assert _spec is not None and _spec.loader is not None
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

NO_WORDS: frozenset[str] = frozenset()


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """A temporary directory removed when its test ends; pytest's own is kept for three runs."""
    with tempfile.TemporaryDirectory(prefix="step1-places-test-") as name:
        yield Path(name)


PRESENT = "present: ragged diverged, so the harness can see a difference on this run"
ABSENT = "ABSENT: no two-shape arm diverged, so a zero in any arm cannot be told from a blind"


def words_of(text: str, confs: list[float] | None = None, *, ms: bool = False) -> list[list]:
    """A transcript's words as ``[word, start, end, conf]``, 80 ms apart."""
    ws = text.split()
    confs = [0.0] * len(ws) if confs is None else confs
    step = 80 if ms else 0.08
    return [
        [w, round(step * i, 4), round(step * (i + 1), 4), c]
        for i, (w, c) in enumerate(zip(ws, confs, strict=True))
    ]


def arm_of(
    *rows: tuple[str, str, str],
    timings: dict | None = None,
    confs: dict | None = None,
    every: bool = True,
    ms: bool = False,
) -> tuple[dict, dict]:
    """Build a one-arm record from (reference, a, b) rows, the way the probe stores them."""
    timings = timings or {}
    confs = confs or {}
    arm: dict[str, Any] = {"checked": 0, "text_divergent": 0, "timing_only_divergent": 0}
    arm["divergences"], arm["transcripts"] = [], {}
    every_recording: dict[str, Any] = {}
    confidence_only = 0
    refs = {}
    for n, (ref, a, b) in enumerate(rows):
        rid = f"r{n}"
        refs[rid] = ref
        ca, cb = confs.get(rid, (None, None))
        if rid in timings:
            at, bt = timings[rid]
            aw = [[*t[:3], t[3] if len(t) > 3 else 0.0] for t in at]
            bw = [[*t[:3], t[3] if len(t) > 3 else 0.0] for t in bt]
        else:
            aw, bw = words_of(a, ca, ms=ms), words_of(b, cb, ms=ms)
            at, bt = copy.deepcopy(aw), copy.deepcopy(bw)  # the probe stores the same tuples twice
        arm["checked"] += 1
        arm["transcripts"][rid] = {"a": a, "b": b} if a != b else {"a": a}
        timing_differs = sp.comparable(at) != sp.comparable(bt)
        if a != b:
            arm["text_divergent"] += 1
        elif timing_differs:
            arm["timing_only_divergent"] += 1
        if a != b or timing_differs:
            arm["divergences"].append(
                {
                    "n": n,
                    "librispeech_id": rid,
                    "text_differs": a != b,
                    "a": a,
                    "b": b,
                    "a_timings": at,
                    "b_timings": bt,
                }
            )
        elif aw != bw:
            confidence_only += 1
        every_recording[rid] = {"a_text": a, "b_text": b, "a_words": aw, "b_words": bw}
    if every:
        arm["every_recording"] = every_recording
        arm["confidence_only_divergent"] = confidence_only
    return arm, refs


def score(
    *rows,
    normalise=sp.normalise,
    timings=None,
    words=NO_WORDS,
    draws=0,
    confs=None,
    every=True,
    **kw,
) -> dict:
    arm, refs = arm_of(*rows, timings=timings, confs=confs, every=every)
    return sp.score_arm("ragged", arm, refs, words, normalise, draws=draws, **kw)


#: The finished record's repeat counts: 8 of 8 identical alone and in the batch.
STOCK_REPEAT = {"alone_identical": 8, "batch_identical": 8, "checked": 8}


def stock_record(
    arm: dict, refs: dict, *, name="ragged", verdict=None, pc=PRESENT, repeat=STOCK_REPEAT
) -> dict:
    run = {
        "repeat_verdict": sp.SAME_SHAPE_VERDICT if verdict is None else verdict,
        "repeat": copy.deepcopy(repeat),
        "arms": {},
    }
    if pc is not None:
        run["positive_control"] = pc
    run["arms"][name] = arm
    return {
        "chunk_ms": 1120,
        "att_context_size": [70, 13],
        "batch": 32,
        "matmul_precision": "highest",
        "references": refs,
        "runs": {"bfloat16": run},
    }


def captures_record(arm: dict, refs: dict, **top) -> dict:
    record = {
        "source": sp.CAPTURES,
        "timing_unit": "ms",
        "chunk_ms": 160,
        "att_context_size": [70, 1],
        "matmul_precision": "high",
        "batch": 64,
        "execution": "graph path",
        "decoder_graphs": False,
        "word_confidence": "off",
        "references": refs,
        "runs": {
            "bfloat16": {
                "repeat_verdict": sp.SAME_SHAPE_VERDICT,
                "repeat": {
                    "recordings": arm["checked"],
                    "identical": arm["checked"],
                    "digests_match": True,
                },
                "positive_control": "present: the ragged capture differs from the fixed one",
                "arms": {sp.CAPTURES_ARM: arm},
            }
        },
    }
    record.update(top)
    return record


def captures_arm(record: dict) -> dict:
    return record["runs"]["bfloat16"]["arms"][sp.CAPTURES_ARM]


def probe_words():
    # Pull words() out of the probe without running it (it loads a GPU model at import).
    tree = ast.parse((ROOT / "probes" / "stock_divergence.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "words")
    scope: dict = {"re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "probe", "exec"), scope)
    return scope["words"]


# --- normalisation -------------------------------------------------------------------------


def test_normalise_lowercases_and_strips_punctuation_but_keeps_apostrophes() -> None:
    assert sp.normalise("Don't cry, he said. I was obliged to come.") == (
        "don't cry he said i was obliged to come"
    )
    assert sp.normalise("Well-known?  'Tis so!") == "well known 'tis so"


def test_normalise_matches_the_probes_own_word_rule_for_plain_apostrophes() -> None:
    words = probe_words()
    for text in (
        "I'm from the cutter lying off the coast.",
        "And and you have not found out anything came in quick, frightened tones.",
        'Yes, sir? We--we did; "no" (never).',
    ):
        assert sp.normalise(text) == " ".join(words(text))


def test_a_typographic_apostrophe_is_where_the_scorer_and_the_probe_differ() -> None:
    text = "I" + chr(0x2019) + "m here."  # the typographic apostrophe
    assert probe_words()(text) == ["i", "m", "here"]
    assert sp.normalise(text) == "i'm here"


def test_a_punctuation_only_difference_is_not_a_place() -> None:
    r = score(("no sir", "No, sir.", "No sir."))
    assert r["text_divergent"] == 1
    assert r["normalisation"]["text_divergent_raw"] == 1
    assert r["normalisation"]["punctuation_or_case_only"] == 1
    assert r["normalisation"]["text_divergent_after_normalisation"] == 0
    assert r["flag"]["places"] == 0 and r["places"] is None
    assert (r["corpus"]["word_errors_a"], r["corpus"]["word_errors_b"]) == (0, 0)


def test_mutation_without_normalisation_the_punctuation_case_becomes_a_place() -> None:
    # The same case as above with the normalisation removed. If this did not become a place,
    # the test above could not tell a normalising scorer from a blind one.
    r = score(("no sir", "No, sir.", "No sir."), normalise=lambda s: s, every=False)
    assert r["normalisation"]["text_divergent_after_normalisation"] == 1
    assert r["flag"]["places"] == 1 and r["places"]["places"] == 1
    assert r["places"]["place_shapes"]["one_for_one"] == 1


def test_a_real_word_change_survives_normalisation_as_one_place() -> None:
    r = score(("no sir", "No, sir.", "Now, sir."))
    assert r["places"]["places"] == 1
    assert (r["places"]["alone_right"], r["places"]["batch_right"]) == (1, 0)
    assert r["flag"]["b"]["precision"] == 1.0
    assert r["flag"]["a"]["precision"] == 0.0


# --- contractions and digits ----------------------------------------------------------------


def test_a_contraction_against_its_expansion_is_counted_separately() -> None:
    r = score(("i am here", "I'm here.", "I am here."))
    assert r["places"]["places"] == 1
    assert r["normalisation"]["places_differing_only_by_a_contraction"] == 1
    assert r["normalisation"]["contraction_only_places"] == [
        {"librispeech_id": "r0", "a": "i'm", "b": "i am"}
    ]
    # Not expanded: "i'm" against "i am" is still two word errors, and b is right there.
    assert (r["corpus"]["word_errors_a"], r["corpus"]["word_errors_b"]) == (2, 0)
    assert r["places"]["batch_right"] == 1
    # ... and both of those errors are contraction-form only.
    assert r["corpus"]["contraction_form_word_errors_a"] == 2
    assert r["corpus"]["wer_a_contractions_tolerated"] == 0.0


def test_contraction_form_errors_are_counted_on_side_b_too() -> None:
    r = score(("i am here", "I am here.", "I'm here."))
    assert (r["corpus"]["word_errors_a"], r["corpus"]["word_errors_b"]) == (0, 2)
    assert r["corpus"]["contraction_form_word_errors_a"] == 0
    assert r["corpus"]["contraction_form_word_errors_b"] == 2
    assert (r["corpus"]["wer_a"], r["corpus"]["wer_b"]) == (0.0, round(2 / 3, 5))


def test_a_different_word_is_not_a_contraction() -> None:
    r = score(("he was here", "He's here.", "He was here."))
    assert r["places"]["places"] == 1
    assert r["normalisation"]["places_differing_only_by_a_contraction"] == 0


def test_expansions() -> None:
    assert sp.expansions("don't") == (("do", "not"),)
    assert sp.expansions("can't") == (("can", "not"), ("cannot",))
    assert sp.expansions("he'd") == (("he", "would"), ("he", "had"), ("he", "did"))
    assert sp.expansions("it's") == (("it", "is"), ("it", "has"))
    assert sp.expansions("o'clock") == ()
    assert sp.expansions("cat") == ()


def test_contraction_distance_is_edit_distance_when_there_is_no_contraction() -> None:
    rng = random.Random(7)
    vocab = ["a", "b", "c", "d"]
    for _ in range(300):
        x = [rng.choice(vocab) for _ in range(rng.randint(0, 7))]
        y = [rng.choice(vocab) for _ in range(rng.randint(0, 7))]
        assert sp.contraction_distance(x, y) == sp.uw.edit_distance(x, y)
    assert sp.contraction_distance(["i", "am", "from"], ["i'm", "from"]) == 0
    assert sp.contraction_distance(["it's", "done"], ["it", "has", "done"]) == 0


def test_tokens_with_digits_are_counted_per_side() -> None:
    r = score(("in nineteen twenty", "In 1920.", "In nineteen twenty."))
    assert r["normalisation"]["tokens_with_digits"] == {"a": 1, "b": 0}


# --- out of dictionary ----------------------------------------------------------------------


def test_out_of_dictionary_counts_reference_words_absent_from_the_list() -> None:
    words = frozenset({"the", "cat", "sat", "on", "a", "mat"})
    r = score(
        ("the cat sat on the gurr", "The cat sat on the gurr.", "The cat sat on the ger."),
        words=words,
    )
    c = r["corpus"]
    assert (c["reference_words"], c["reference_words_out_of_dictionary"]) == (6, 1)
    assert c["out_of_dictionary_rate"] == round(1 / 6, 5)
    o = r["out_of_dictionary_at_places"]
    assert (o["reference_words_at_places"], o["absent_from_word_list"]) == (1, 1)
    assert (o["share_absent"], o["corpus_share_absent"]) == (1.0, round(1 / 6, 5))
    assert o["label"].startswith("reference words at places absent from the word list")


# --- counts and rates of an arm -------------------------------------------------------------

#: A timing-only divergence: "D e." with both words' times moved.
R1_TIMINGS = (
    [["D", 0.0, 0.08], ["e.", 0.08, 0.16]],
    [["D", 0.0, 0.16], ["e.", 0.16, 0.24]],
)


def test_divergence_rates_are_over_the_checked_recordings() -> None:
    r = score(
        ("a b", "A b.", "A c."),
        ("d e", "D e.", "D e."),
        ("f g", "F g.", "F x."),
        ("h i", "H i.", "H i."),
        timings={"r1": R1_TIMINGS},
    )
    assert (r["checked"], r["text_divergent"], r["timing_only_divergent"]) == (4, 2, 1)
    assert (r["text_divergent_rate"], r["timing_only_divergent_rate"]) == (0.5, 0.25)
    assert r["confidence_only_divergent"] == 0


# --- the place flag: recall, precision --------------------------------------------------------


def test_wildcard_distance_is_edit_distance_without_wildcards() -> None:
    rng = random.Random(11)
    vocab = ["a", "b", "c"]
    for _ in range(300):
        x = [rng.choice(vocab) for _ in range(rng.randint(0, 7))]
        y = [rng.choice(vocab) for _ in range(rng.randint(0, 7))]
        assert sp.wildcard_distance(x, y) == sp.uw.edit_distance(x, y)


def test_recall_counts_only_the_errors_a_place_can_remove() -> None:
    # Both sides share the error "q" for "d"; they differ only at "x"/"y" for "b".
    f = score(("a b c d e", "a x c q e", "a y c q e"))["flag"]
    assert (f["a"]["corpus_word_errors"], f["a"]["word_errors_removable"]) == (2, 1)
    assert (f["b"]["corpus_word_errors"], f["b"]["word_errors_removable"]) == (2, 1)
    assert (f["a"]["recall"], f["b"]["recall"]) == (0.5, 0.5)
    assert f["reference_words_at_places"] == 1


def test_the_wildcard_credits_a_place_with_errors_at_words_it_does_not_touch() -> None:
    # Why reference words at places is no baseline for recall: the place touches one
    # reference word ("cat") but is credited with three of b's four errors.
    f = score(("the cat sat on the mat the end", "The cow sat the end.", "The dog sat the end."))[
        "flag"
    ]
    assert f["reference_words_at_places"] == 1
    assert (f["b"]["corpus_word_errors"], f["b"]["word_errors_removable"]) == (4, 3)


def test_a_two_word_place_touches_two_reference_words() -> None:
    f = score(("a b c d", "a x y d", "a p q d"))["flag"]
    assert f["places"] == 1
    assert f["reference_words_at_places"] == 2
    assert f["b"]["words_flagged"] == 2


def test_recall_denominator_includes_recordings_that_never_diverged() -> None:
    r = score(("a b c", "a x c", "a y c"), ("d e f", "d q f", "d q f"))
    assert r["corpus"]["word_errors_a"] == 2
    assert r["flag"]["a"]["word_errors_removable"] == 1
    assert r["flag"]["a"]["recall"] == 0.5
    assert r["corpus"]["wer_a"] == round(2 / 6, 5)


def test_a_stock_arm_without_every_recording_withholds_recall_and_says_why() -> None:
    rows = [("a b c", "a x c", "a y c"), ("d e f", "d q f", "d q f")]
    r = score(*rows, every=False, draws=20)
    for s in "ab":
        f = r["flag"][s]
        assert f["recall"] is None and "no every_recording" in f["recall_withheld"]
        assert f["null"]["same_recordings"]["recall_mean"] is None
        assert f["null"]["random_recordings"]["recall_mean"] is None
    assert any(w.startswith("recall: the record has no every_recording") for w in r["withheld"])
    # The same rows with every_recording: recall is given, and so is each null's recall_mean.
    given = score(*rows, draws=20)
    assert given["flag"]["a"]["recall"] == 0.5 and "recall_withheld" not in given["flag"]["a"]
    assert given["flag"]["a"]["null"]["random_recordings"]["recall_mean"] is not None
    assert not any(w.startswith("recall:") for w in given["withheld"])


def test_a_captures_arm_gives_recall_without_every_recording() -> None:
    arm, refs = arm_of(("a b c", "a x c", "a y c"), ("d e f", "d q f", "d q f"), every=False)
    r = sp.score_arm(sp.CAPTURES_ARM, arm, refs, NO_WORDS, draws=0, captures=True)
    assert r["flag"]["a"]["recall"] == 0.5 and "recall_withheld" not in r["flag"]["a"]


def test_an_empty_side_place_takes_the_other_sides_inserted_word() -> None:
    r = score(("the cat sat", "The cat sat.", "The the cat sat."))
    assert r["places"]["places"] == 1 and r["places"]["place_shapes"]["one_side_empty"] == 1
    f = r["flag"]
    assert (f["a"]["corpus_word_errors"], f["b"]["corpus_word_errors"]) == (0, 1)
    assert f["b"]["word_errors_removable"] == 1 and f["b"]["recall"] == 1.0
    assert f["a"]["recall"] is None  # no errors to recall
    assert (f["a"]["empty_spans"], f["a"]["words_flagged"]) == (1, 0)


def test_a_dropped_word_is_recalled_at_the_empty_side() -> None:
    f = score(("the big red dog", "the big red dog", "the big dog"))["flag"]
    assert f["b"]["word_errors_removable"] == 1 and f["b"]["recall"] == 1.0


def test_both_right_is_subtracted_from_each_sides_precision() -> None:
    # Each side is one correct word; the misses lie outside the place.
    r = score(("z x x z", "x", "z"))
    assert (r["places"]["places"], r["places"]["both_right"]) == (1, 1)
    assert (r["flag"]["both_right"], r["flag"]["both_right_by_analyse"]) == (1, 1)
    assert (r["flag"]["a"]["precision"], r["flag"]["b"]["precision"]) == (0.0, 0.0)
    # analyse's own count takes both_right off each side too
    f = r["flag"]
    assert (f["a"]["not_right_by_analyse"], f["b"]["not_right_by_analyse"]) == (0, 0)
    bound = r["flag"]["precision_bound"]
    assert bound.startswith("both_right is 1")
    assert "does not hold by construction" in bound


def test_the_precision_bound_is_stated_when_both_right_is_zero() -> None:
    f = score(("no sir", "No, sir.", "Now, sir."))["flag"]
    assert f["both_right"] == 0
    assert "precision_a + precision_b >= 1 holds by construction" in f["precision_bound"]
    assert f["a"]["precision"] + f["b"]["precision"] >= 1
    assert "extra decode" in f["cost"]


def test_places_are_judged_by_the_span_rule_and_analyses_count_is_kept_apart() -> None:
    # a = "no" has a gap where b says "yes". By a's own alignment "no" stands in for "yes", so
    # no reference word is missing at the gap: the span rule, the one the nulls and the
    # confidence flag use, calls the gap right. analyse calls it wrong because b's word there
    # is correct. Scoring the places by analyse's rule would favour them over their nulls.
    r = score(("yes", "No.", "Yes, no."))
    f = r["flag"]
    assert (f["a"]["empty_spans"], f["a"]["not_right"], f["a"]["precision"]) == (1, 0, 0.0)
    assert f["a"]["not_right_by_analyse"] == 1
    assert (f["b"]["not_right"], f["b"]["not_right_by_analyse"]) == (0, 0)
    assert (f["both_right"], f["both_right_by_analyse"]) == (1, 0)
    assert f["precision_bound"].startswith("both_right is 1")
    aligned = sp.uw.Aligned(["no"], ["yes"])
    assert sp.span_not_right(aligned, 0, 0) is False


# --- nulls ----------------------------------------------------------------------------------


def test_an_empty_span_is_not_right_where_a_reference_word_is_missing() -> None:
    aligned = sp.uw.Aligned(["the", "big", "dog"], ["the", "big", "red", "dog"])
    assert sp.span_not_right(aligned, 2, 2) is True  # "red" is missing before "dog"
    assert sp.span_not_right(aligned, 1, 1) is False
    assert sp.span_not_right(aligned, 0, 2) is False
    wrong = sp.uw.Aligned(["the", "bog", "dog"], ["the", "big", "dog"])
    assert sp.span_not_right(wrong, 1, 2) is True


def test_place_randomly_keeps_lengths_and_never_overlaps() -> None:
    rng = random.Random(3)
    for _ in range(500):
        lengths = [rng.randint(0, 3) for _ in range(rng.randint(1, 4))]
        n = sum(lengths) + rng.randint(0, 5)
        spans = sp.place_randomly(n, lengths, rng)
        assert sorted(i2 - i1 for i1, i2 in spans) == sorted(lengths)
        assert all(0 <= i1 <= i2 <= n for i1, i2 in spans)
        assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))


def test_place_randomly_reaches_every_position() -> None:
    rng = random.Random(5)
    assert {sp.place_randomly(3, [1], rng)[0] for _ in range(200)} == {(0, 1), (1, 2), (2, 3)}
    assert {sp.place_randomly(2, [0], rng)[0][0] for _ in range(200)} == {0, 1, 2}


def one_error_rows(k: int) -> tuple[str, str, str]:
    ref = ["a", "b", "c", "d", "e", "f", "g", "h"]
    b = list(ref)
    b[k % 8] = "q"
    return " ".join(ref), " ".join(ref), " ".join(b)


def test_a_flag_on_every_error_beats_both_nulls() -> None:
    rows = [one_error_rows(k) for k in range(20)] + [("a b c d e f g h",) * 3] * 20
    f = score(*rows, draws=200)["flag"]["b"]
    assert (f["word_errors_removable"], f["not_right"], f["precision"]) == (20, 20, 1.0)
    for name in ("same_recordings", "random_recordings"):
        n = f["null"][name]
        assert n["p_value_word_errors_removable"] == round(1 / 201, 5)
        assert n["p_value_not_right"] == round(1 / 201, 5)
        assert n["word_errors_removable_mean"] < 5 and n["not_right_rate_mean"] < 0.25


def test_a_flag_on_a_corpus_that_is_wrong_everywhere_does_not_beat_its_null() -> None:
    # Every word of b is wrong, so a span anywhere removes exactly what the place removes.
    f = score(*[("a b c d", "w x y z", "w x y q")] * 5, draws=50)["flag"]["b"]
    for name in ("same_recordings", "random_recordings"):
        n = f["null"][name]
        assert n["p_value_word_errors_removable"] == 1.0
        assert n["observed_over_null_mean"] == 1.0
        assert n["not_right_rate_mean"] == 1.0


def test_the_not_right_rate_is_per_span_not_per_recording() -> None:
    # Two places in each recording, and every word of b is wrong: every random span is not
    # right, so the rate is 1.0. Per recording it would read 2.0.
    f = score(*[("a b c d e", "v w x y z", "q w r y z")] * 5, draws=20)["flag"]["b"]
    assert (f["spans"], f["recordings_flagged"]) == (10, 5)
    for name in ("same_recordings", "random_recordings"):
        assert f["null"][name]["not_right_mean"] == 10.0
        assert f["null"][name]["not_right_rate_mean"] == 1.0


class StubRec:
    """What ``nulls`` asks of a recording, with fixed answers."""

    def __init__(self, n: int, removable: int, not_right: int) -> None:
        self.hyp = ["w"] * n
        self._removable, self._not_right = removable, not_right

    def removable(self, spans) -> int:
        return self._removable

    def not_right(self, spans) -> int:
        return self._not_right


def test_each_null_figure_has_its_own_numerator_and_denominator() -> None:
    # One flagged recording with two spans; every draw removes 2 errors and is not right once.
    # Observed: 5 errors removed, 2 spans not right, of 10 errors in all.
    recs = {"r0": StubRec(6, 2, 1), "r1": StubRec(6, 2, 1)}
    out = sp.nulls({"r0": [(0, 1), (2, 3)]}, recs, 5, 2, 10, draws=9, seed="s")
    for name in ("same_recordings", "random_recordings"):
        n = out[name]
        assert n["word_errors_removable_mean"] == 2.0
        assert n["recall_mean"] == 0.2  # 2 of 10 errors; over the 2 spans it would read 1.0
        assert n["observed_over_null_mean"] == 2.5  # 5 over 2, not 2 over 5
        assert n["p_value_word_errors_removable"] == 0.1  # no draw removes 5
        assert n["not_right_mean"] == 1.0
        assert n["not_right_rate_mean"] == 0.5  # 1 of 2 spans; per recording it would be 1.0
        # No draw is not right twice. Against the removable draws (2 each) it would read 1.0.
        assert n["p_value_not_right"] == 0.1
    assert out["same_recordings"]["recordings_drawn_from"] == 1
    assert out["random_recordings"]["recordings_drawn_from"] == 2
    withheld = sp.nulls({"r0": [(0, 1)]}, recs, 5, 2, None, draws=9, seed="s")
    assert withheld["same_recordings"]["recall_mean"] is None


def test_a_flags_null_figures_are_computed_from_its_own_observed_counts() -> None:
    # Side b removes 3 errors at one place that is not right once: removable and not-right
    # differ, so a null that took one for the other would show here.
    rows = [("the cat sat on the mat the end", "The cow sat the end.", "The dog sat the end.")]
    rows += [("a b c d e f g h",) * 3] * 9
    f = score(*rows, draws=200)["flag"]["b"]
    assert (f["word_errors_removable"], f["not_right"], f["spans"]) == (3, 1, 1)
    total = f["corpus_word_errors"]
    assert total == 4
    for name in ("same_recordings", "random_recordings"):
        n = f["null"][name]
        mean = n["word_errors_removable_mean"]
        assert n["observed_over_null_mean"] == round(3 / mean, 5)
        assert n["recall_mean"] == round(mean / total, 5)
        assert n["not_right_rate_mean"] == round(n["not_right_mean"] / 1, 5)
    assert f["null"]["random_recordings"]["observed_over_null_mean"] > 1


def test_the_random_recordings_null_draws_from_the_whole_corpus() -> None:
    rows = [("a b c d", "w x y z", "w x y q")] + [("a b c d",) * 3] * 9
    n = score(*rows, draws=200)["flag"]["b"]["null"]
    assert n["same_recordings"]["word_errors_removable_mean"] == 1.0
    assert n["random_recordings"]["recordings_drawn_from"] == 10
    assert n["random_recordings"]["word_errors_removable_mean"] < 0.5


def test_the_random_recordings_null_only_draws_recordings_that_can_hold_the_spans() -> None:
    rows = [("a b c", "a b c", "x y z")] + [("a", "a", "a")] * 5
    n = score(*rows, draws=30)["flag"]["b"]["null"]
    assert n["random_recordings"]["word_errors_removable_mean"] == 3.0


def test_nulls_are_seeded_and_report_their_draws() -> None:
    rows = [one_error_rows(k) for k in range(6)] + [("a b c d e f g h",) * 3] * 6
    first = score(*rows, draws=40, seed="7")["flag"]["b"]["null"]
    again = score(*rows, draws=40, seed="7")["flag"]["b"]["null"]
    assert first == again
    assert first["draws"] == 40 and first["seed"].startswith("7/")
    other = score(*rows, draws=40, seed="8")["flag"]["b"]["null"]
    assert other["same_recordings"] != first["same_recordings"]
    assert score(*rows, draws=0)["flag"]["b"]["null"] is None


# --- the confidence flag, side by side with the places --------------------------------------

CONF_ROWS = (
    ("the cat sat on the mat", "The cat sat on the mat.", "The cat sat on a mat."),
    ("a dog ran home", "A dog ran hum.", "A dog ran hum."),
)
FLAT = {"r0": ([0.9] * 6, [0.9] * 6), "r1": ([0.9] * 4, [0.9] * 4)}


def conf(side: str, *, ms: bool = True, **per_rid: list[float]) -> dict:
    """A --confidence input for CONF_ROWS: one side's words with a confidence each."""
    texts = {
        "a": {"r0": "The cat sat on the mat.", "r1": "A dog ran hum."},
        "b": {"r0": "The cat sat on a mat.", "r1": "A dog ran hum."},
    }[side]
    return {
        "side": side,
        "source": "test",
        "words": {rid: words_of(texts[rid], per_rid[rid], ms=ms) for rid in texts},
    }


def captures_score(*rows, confidence, draws=0, **kw) -> dict:
    arm, refs = arm_of(*rows, every=False, ms=True, **kw)
    return sp.score_arm(
        sp.CAPTURES_ARM, arm, refs, NO_WORDS, draws=draws, confidence=confidence, captures=True
    )


def test_confidence_unavailable_is_said_not_invented() -> None:
    c = score(*CONF_ROWS, every=False)["confidence"]
    for s in "ab":
        assert c[s]["status"] == "confidence unavailable"
        assert "no every_recording" in c[s]["reason"]
    r = score(*CONF_ROWS, every=False)
    assert sum(w.startswith("confidence flag, side") for w in r["withheld"]) == 2


def test_both_sides_are_ranked_from_every_recording() -> None:
    confs = {
        "r0": ([0.9] * 6, [0.9, 0.9, 0.9, 0.9, 0.1, 0.9]),
        "r1": ([0.9, 0.9, 0.9, 0.2], [0.9] * 4),
    }
    r = score(*CONF_ROWS, confs=confs)
    assert r["confidence_only_divergent"] == 1  # r1: same words and times, other confidences
    for s in "ab":
        assert r["confidence"][s]["status"] == "scored"
        assert r["confidence"][s]["side"] == s and r["confidence"][s]["source"] == "every_recording"
    assert r["confidence"]["a"]["flagged_words"] == {"r1": [[3, 4]]}
    assert r["confidence"]["b"]["flagged_words"] == {"r0": [[4, 5]]}
    assert r["withheld"] == []


def test_confidence_flags_the_k_lowest_words_where_k_is_what_the_places_flag() -> None:
    # The place on side a is a word a got right; a's only error ("hum") has the lowest confidence.
    confs = {"r0": ([0.9] * 6, [0.9] * 6), "r1": ([0.9, 0.9, 0.9, 0.2], [0.9] * 4)}
    c = score(*CONF_ROWS, confs=confs)["confidence"]["a"]
    assert c["status"] == "scored" and c["side"] == "a"
    assert c["k_words"] == 1 and c["cutoff_confidence"] == 0.2
    assert (c["flag"]["recordings_flagged"], c["flag"]["words_flagged"]) == (1, 1)
    assert (c["flag"]["precision"], c["flag"]["recall"]) == (1.0, 1.0)
    side_by_side = c["side_by_side"]
    assert (side_by_side["places"]["precision"], side_by_side["places"]["recall"]) == (0.0, 0.0)
    assert (side_by_side["confidence"]["precision"], side_by_side["confidence"]["recall"]) == (
        1.0,
        1.0,
    )


def test_k_counts_the_words_the_places_flag_and_adjacent_flagged_words_form_one_span() -> None:
    rows = (
        ("the cat sat on the mat", "The cat sat on the mat.", "The cat sat in a mat."),
        CONF_ROWS[1],
    )
    confs = {"r0": ([0.9] * 6, [0.9] * 6), "r1": ([0.9, 0.9, 0.2, 0.3], [0.9] * 4)}
    c = score(*rows, confs=confs)["confidence"]["a"]
    assert c["k_words"] == 2
    assert c["flagged_words"] == {"r1": [[2, 4]]}
    # The cutoff is the K-th lowest confidence (0.3), not the lowest (0.2).
    assert c["cutoff_confidence"] == 0.3
    assert c["tied_at_cutoff"] == {"flagged": 1, "all": 1}
    assert (c["flag"]["spans"], c["flag"]["precision"], c["flag"]["recall"]) == (1, 1.0, 1.0)


K_ROWS = (
    ("the cat sat", "The cat sat.", "The the cat sat."),  # a: a gap (0 words); b: "the"
    ("a b c d", "A x c d.", "A y c d."),  # one word each side
)


def test_k_is_the_place_flags_word_count_on_the_same_side() -> None:
    confs = {
        "r0": ([0.5, 0.9, 0.9], [0.9, 0.9, 0.9, 0.8]),
        "r1": ([0.9, 0.3, 0.9, 0.9], [0.9, 0.2, 0.9, 0.9]),
    }
    r = score(*K_ROWS, confs=confs)
    assert (r["flag"]["a"]["words_flagged"], r["flag"]["b"]["words_flagged"]) == (1, 2)
    a, b = r["confidence"]["a"], r["confidence"]["b"]
    assert (a["k_words"], b["k_words"]) == (1, 2)
    assert a["k_rule"].endswith("side a") and b["k_rule"].endswith("side b")
    assert a["flagged_words"] == {"r1": [[1, 2]]}
    assert b["flagged_words"] == {"r0": [[3, 4]], "r1": [[1, 2]]}
    assert (a["place_flag_empty_spans"], b["place_flag_empty_spans"]) == (1, 0)


def test_a_side_whose_places_mark_no_word_has_nothing_to_compare() -> None:
    rows = (K_ROWS[0], ("d e f", "D e f.", "D e f."))
    confs = {
        "r0": ([0.5, 0.9, 0.9], [0.9, 0.9, 0.9, 0.8]),
        "r1": ([0.9, 0.4, 0.9], [0.9, 0.4, 0.9]),
    }
    r = score(*rows, confs=confs)
    assert r["flag"]["a"]["words_flagged"] == 0
    assert r["confidence"]["a"]["status"] == "nothing to compare"
    assert r["confidence"]["a"]["reason"] == "the place flag marks 0 words on side a"
    assert r["confidence"]["b"]["status"] == "scored" and r["confidence"]["b"]["k_words"] == 1


SBS_ROWS = (
    ("a b c d e f", "A b c d e f.", "A b x d e f."),  # b wrong at the place
    ("g h i j", "G q i j.", "G h i j."),  # a wrong at the place
    ("k l m n", "K z m y.", "K l m y."),  # a wrong at the place; both wrong at "y"
    *[("o p q r", "O p q r.", "O p q r.")] * 7,
)


def expected_brief(flag: dict) -> dict:
    return {
        "words_flagged": flag["words_flagged"],
        "spans": flag["spans"],
        "precision": flag["precision"],
        "recall": flag["recall"],
        "not_right_rate_same_recordings_null": flag["null"]["same_recordings"][
            "not_right_rate_mean"
        ],
        "not_right_rate_random_recordings_null": flag["null"]["random_recordings"][
            "not_right_rate_mean"
        ],
        "p_value_removable_random_recordings": flag["null"]["random_recordings"][
            "p_value_word_errors_removable"
        ],
    }


def test_side_by_side_puts_each_sides_own_figures_in_their_own_columns() -> None:
    rng = random.Random(4)
    confs = {}
    for n, (_, a, b) in enumerate(SBS_ROWS):
        confs[f"r{n}"] = tuple([round(rng.uniform(0.3, 1.0), 3) for _ in t.split()] for t in (a, b))
    r = score(*SBS_ROWS, confs=confs, draws=200)
    places = r["flag"]
    # The two sides differ, and each side's two nulls differ, so a column taken from the
    # wrong side or the wrong null cannot match.
    assert places["a"]["precision"] != places["b"]["precision"]
    for s in "ab":
        same, rand = places[s]["null"]["same_recordings"], places[s]["null"]["random_recordings"]
        assert same["not_right_rate_mean"] != rand["not_right_rate_mean"]
        c = r["confidence"][s]
        assert c["status"] == "scored"
        assert c["side_by_side"]["places"] == expected_brief(places[s])
        assert c["side_by_side"]["confidence"] == expected_brief(c["flag"])


def test_confidence_stored_in_the_divergences_is_not_used_and_the_output_says_why() -> None:
    at = [["Hi", 0.08, 0.4, 0.0]]
    bt = [["Hi", 0.16, 0.4, 0.0]]
    r = score(("hi", "Hi.", "Hi."), timings={"r0": (at, bt)}, every=False)
    assert r["confidence"]["a"]["status"] == "confidence unavailable"
    assert "only for divergent recordings" in r["confidence"]["a"]["reason"]
    assert r["timing_only_shift_ms"]["max"] == 80


def test_ties_at_the_cutoff_are_broken_by_the_seed_not_by_position() -> None:
    rows = [("a b c", "a b c", "a x c")] + [("a b c",) * 3] * 9
    confs = {f"r{n}": ([0.5, 0.9, 0.9], [0.9, 0.9, 0.9]) for n in range(10)}
    arm, refs = arm_of(*rows, confs=confs)
    picked = set()
    for seed in range(20):
        c = sp.score_arm("ragged", arm, refs, NO_WORDS, draws=0, seed=str(seed))["confidence"]
        assert c["a"]["tied_at_cutoff"] == {"flagged": 1, "all": 10}
        picked |= set(c["a"]["flagged_words"])
    assert len(picked) > 1


def test_the_confidence_flag_is_scored_by_the_place_rules() -> None:
    # Confidence lowest exactly at the place on side b: both flags mark the same span, so the
    # same rules must give the same precision, removable errors and recall.
    confs = {"r0": ([0.9] * 6, [0.9, 0.9, 0.9, 0.9, 0.1, 0.9]), "r1": ([0.9] * 4, [0.9] * 4)}
    r = score(*CONF_ROWS, confs=confs)
    places, confidence = r["flag"]["b"], r["confidence"]["b"]["flag"]
    for key in ("spans", "words_flagged", "not_right", "precision", "word_errors_removable"):
        assert places[key] == confidence[key], key
    assert (confidence["precision"], confidence["recall"]) == (1.0, 0.5)


def test_constant_confidence_is_not_scored() -> None:
    c = score(*CONF_ROWS)["confidence"]
    for s in "ab":
        assert c[s]["status"] == "confidence constant" and c[s]["value"] == 0.0
        assert "flag" not in c[s]


def test_ties_at_the_cutoff_are_reported() -> None:
    confs = {
        "r0": ([0.5, 0.5, 0.9, 0.9, 0.9, 0.9], [0.9] * 6),
        "r1": ([0.5, 0.9, 0.9, 0.9], [0.9] * 4),
    }
    c = score(*CONF_ROWS, confs=confs)["confidence"]["a"]
    assert c["tied_at_cutoff"] == {"flagged": 1, "all": 3}


def test_confidence_words_are_normalised_like_the_transcript() -> None:
    toks, confs = sp.confidence_tokens(
        [["It", 0, 80, 0.9], ["is", 80, 160, 0.7], ["well-known.", 160, 320, 0.1]],
        "r0",
        lambda t: sp.normalise(t).split(),
    )
    assert toks == ["it", "is", "well", "known"]
    assert confs == [0.9, 0.7, 0.1, 0.1]


def test_confidence_that_does_not_spell_the_transcript_is_refused() -> None:
    given = conf("a", r0=[0.9] * 6, r1=[0.9] * 4)
    given["words"]["r1"][3][0] = "home."
    with pytest.raises(ValueError, match="the words of r1 do not spell side a"):
        captures_score(*CONF_ROWS, confidence=given)


def test_every_recording_words_that_do_not_spell_the_transcript_are_refused() -> None:
    arm, refs = arm_of(*CONF_ROWS, confs=FLAT)
    for e in (arm["every_recording"]["r1"],):
        e["b_words"][3][0] = "home."
        e["a_words"][3][0] = "home."
    arm["confidence_only_divergent"] = 0
    with pytest.raises(ValueError, match="the words of r1 do not spell side a"):
        sp.score_arm("ragged", arm, refs, NO_WORDS, draws=0)


def test_confidence_missing_a_recording_is_refused() -> None:
    given = conf("a", r0=[0.9] * 6, r1=[0.9] * 4)
    del given["words"]["r1"]
    with pytest.raises(ValueError, match="no words for transcript r1"):
        captures_score(*CONF_ROWS, confidence=given)


def test_confidence_entries_for_other_recordings_are_counted() -> None:
    given = conf("a", r0=[0.9] * 6, r1=[0.9, 0.9, 0.9, 0.2])
    given["words"]["r9"] = [["Extra", 0, 80, 0.5]]
    c = captures_score(*CONF_ROWS, confidence=given)["confidence"]
    assert c["a"]["status"] == "scored" and c["a"]["entries_not_in_this_arm"] == 1
    assert c["b"]["status"] == "confidence unavailable"
    assert c["b"]["reason"].startswith("no confidence input covers side b")


@pytest.mark.parametrize("entry", [["A", 0, 80], ["A", 0, 80, 0.9, 0.1]])
def test_a_word_entry_of_other_than_four_elements_is_refused(entry) -> None:
    # Three elements carry no confidence. Five would be ranked on an element that is not the
    # confidence (the fifth, if the last were read), so they are refused, not guessed at.
    given = conf("a", r0=[0.9] * 6, r1=[0.9] * 4)
    given["words"]["r1"][0] = entry
    with pytest.raises(ValueError, match=f"entry 0 of r1 has {len(entry)} elements"):
        captures_score(*CONF_ROWS, confidence=given)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.5"])
def test_a_confidence_that_is_not_a_finite_number_is_refused(bad) -> None:
    confs = {"r0": ([0.9] * 6, [0.9] * 6), "r1": ([bad, 0.9, 0.9, 0.9], [0.9] * 4)}
    arm, refs = arm_of(*CONF_ROWS, confs=confs)
    with pytest.raises(ValueError, match="word 0 of r1 has confidence"):
        sp.score_arm("ragged", arm, refs, NO_WORDS, draws=0)


def test_a_confidence_side_other_than_a_or_b_is_refused() -> None:
    given = conf("a", r0=[0.9] * 6, r1=[0.9] * 4)
    given["side"] = "served"
    with pytest.raises(ValueError, match="side 'served' is not"):
        captures_score(*CONF_ROWS, confidence=given)


# --- every_recording is checked before it is used: one test per check -----------------------


def every_arm() -> dict:
    """r0 text-divergent, r1 timing-only divergent, r2 confidence-only, r3 identical."""
    arm, _ = arm_of(
        ("a b", "A b.", "A c."),
        ("d e", "D e.", "D e."),
        ("f g", "F g.", "F g."),
        ("h i", "H i.", "H i."),
        timings={"r1": R1_TIMINGS},
        confs={"r2": ([0.5, 0.5], [0.5, 0.6])},
    )
    return arm


def _every(arm: dict, rid: str) -> dict:
    return arm["every_recording"][rid]


EVERY_CHECKS = {
    "not a mapping": (
        lambda arm: arm.update(every_recording=[]),
        "every_recording is not a mapping of recording id",
    ),
    "lacks a checked recording": (
        lambda arm: arm["every_recording"].pop("r3"),
        "every_recording lacks 1 checked recordings, e.g. r3",
    ),
    "holds an unchecked recording": (
        lambda arm: arm["every_recording"].update(r9=copy.deepcopy(_every(arm, "r3"))),
        "every_recording holds 1 unchecked recordings, e.g. r9",
    ),
    "a text": (
        lambda arm: _every(arm, "r0").update(a_text="A z."),
        "every_recording r0 text disagrees with its transcript",
    ),
    "b text": (
        lambda arm: _every(arm, "r0").update(b_text="A z."),
        "every_recording r0 text disagrees with its transcript",
    ),
    "a word of three elements": (
        lambda arm: _every(arm, "r3")["a_words"].__setitem__(0, ["H", 0.0, 0.08]),
        "every_recording r3 a_words entry 0 is not [word, start_s, end_s, conf]",
    ),
    "a word of five elements": (
        lambda arm: _every(arm, "r3")["b_words"].__setitem__(0, ["H", 0.0, 0.08, 0.0, 0.0]),
        "every_recording r3 b_words entry 0 is not [word, start_s, end_s, conf]",
    ),
    "b words not a list": (
        lambda arm: _every(arm, "r3").update(b_words=None),
        "every_recording r3 b_words is not a list",
    ),
    "differs but no divergence": (
        lambda arm: _every(arm, "r3")["b_words"][0].__setitem__(2, 0.16),
        "every_recording r3 differs but no divergence lists it",
    ),
    "agrees but a divergence": (
        lambda arm: _every(arm, "r1").update(b_words=copy.deepcopy(_every(arm, "r1")["a_words"])),
        "every_recording r1 agrees but a divergence lists it",
    ),
    "b words disagree with the divergence": (
        lambda arm: _every(arm, "r0")["b_words"][1].__setitem__(1, 0.16),
        "every_recording r0 words disagree with its divergence's timings",
    ),
    "a words disagree with the divergence": (
        lambda arm: _every(arm, "r0")["a_words"][0].__setitem__(1, 0.04),
        "every_recording r0 words disagree with its divergence's timings",
    ),
    "no confidence_only_divergent": (
        lambda arm: arm.pop("confidence_only_divergent"),
        "every_recording without confidence_only_divergent",
    ),
    "confidence_only_divergent wrong": (
        lambda arm: arm.update(confidence_only_divergent=0),
        "confidence_only_divergent 0 but 1 recordings differ only in confidence",
    ),
}


def test_the_valid_every_recording_is_accepted() -> None:
    arm = every_arm()
    assert arm["confidence_only_divergent"] == 1
    sp.check_every_recording(arm)
    del arm["every_recording"], arm["confidence_only_divergent"]
    sp.check_every_recording(arm)  # an arm with neither passes; the scorer withholds instead


@pytest.mark.parametrize("count", [7, 0, None])
def test_a_confidence_only_count_without_every_recording_is_refused(count) -> None:
    # The count is of every_recording's recordings; without them nothing can check it, so it
    # is refused rather than reported. Refused on the scored path and the withheld one.
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."), every=False)
    arm["confidence_only_divergent"] = count
    message = f"record refused: confidence_only_divergent {count!r} without every_recording"
    with pytest.raises(sp.Refused, match=re.escape(message)):
        sp.check_every_recording(arm)
    with pytest.raises(sp.Refused, match=re.escape(message)):
        sp.derive(stock_record(arm, refs), NO_WORDS, draws=0)
    with pytest.raises(sp.Refused, match=re.escape(message)):
        sp.derive(stock_record(arm, refs, verdict="not measured"), NO_WORDS, draws=0)


def test_an_arm_with_neither_reads_its_confidence_only_count_as_null() -> None:
    # An arm that measured nothing reads null, not a measured-looking 0.
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."), every=False)
    scored = sp.derive(stock_record(arm, refs), NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert scored["arms"]["ragged"]["confidence_only_divergent"] is None
    record = stock_record(every_arm(), {f"r{n}": "x" for n in range(4)})
    scored = sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert scored["arms"]["ragged"]["confidence_only_divergent"] == 1


@pytest.mark.parametrize("check", list(EVERY_CHECKS))
def test_each_every_recording_check_refuses_its_own_defect(check: str) -> None:
    arm = every_arm()
    change, message = EVERY_CHECKS[check]
    change(arm)
    with pytest.raises(ValueError, match=re.escape(message)):
        sp.check_every_recording(arm)


# --- timing --------------------------------------------------------------------------------

AT = [["Hi", 0.08, 0.4], ["there.", 0.4, 0.56]]


def timing_only(bt, at=AT, every=True) -> dict:
    return score(("hi there", "Hi there.", "Hi there."), timings={"r0": (at, bt)}, every=every)


def test_timing_only_shift_is_the_largest_start_or_end_move_in_ms() -> None:
    # start moves 80 ms, end moves 160 ms: the larger of the two is the shift
    r = timing_only([["Hi", 0.08, 0.4], ["there.", 0.48, 0.72]])
    t = r["timing_only_shift_ms"]
    assert r["timing_only_divergent"] == 1 and r["flag"]["places"] == 0
    assert (t["paired"], t["min"], t["max"], t["largest_shift_counts"]) == (1, 160, 160, {"160": 1})
    assert (t["segments_shifted"], t["segments_in_paired_recordings"]) == (1, 2)


def test_a_start_move_larger_than_the_end_move_is_the_shift() -> None:
    # start moves 240 ms, end moves 80 ms
    t = timing_only([["Hi", 0.08, 0.4], ["there.", 0.64, 0.64]])["timing_only_shift_ms"]
    assert (t["min"], t["max"]) == (240, 240)


def test_equal_length_timings_with_different_words_are_unpairable() -> None:
    bt = [["Hi", 0.08, 0.4], ["their.", 0.4, 0.56]]
    t = timing_only(bt, every=False)["timing_only_shift_ms"]
    assert (t["unpairable"], t["paired"]) == (["r0"], 0)


def test_timings_of_different_lengths_are_unpairable_even_when_the_shared_words_agree() -> None:
    t = timing_only([["Hi", 0.08, 0.4]], every=False)["timing_only_shift_ms"]
    assert (t["unpairable"], t["paired"]) == (["r0"], 0)


def test_a_recording_whose_text_differs_is_not_timing_measured() -> None:
    bt = [["Hi", 0.32, 0.4], ["where.", 0.4, 0.56]]
    r = score(("hi there", "Hi there.", "Hi where."), timings={"r0": (AT, bt)})
    t = r["timing_only_shift_ms"]
    assert (t["recordings"], t["paired"], t["max"]) == (0, 0, None)


def test_the_median_and_p90_shift_are_nearest_rank_over_the_recordings() -> None:
    rows, timings = [], {}
    for n, k in enumerate((7, 1, 10, 4, 2, 9, 3, 6, 8, 5)):
        rows.append(("hi there", "Hi there.", "Hi there."))
        end = round(0.56 + 0.08 * k, 4)
        timings[f"r{n}"] = (AT, [["Hi", 0.08, 0.4], ["there.", 0.4, end]])
    t = score(*rows, timings=timings)["timing_only_shift_ms"]
    assert t["largest_shift_counts"] == {str(80 * k): 1 for k in range(1, 11)}
    # n = 10 shifts of 80..800 ms. Nearest rank: the median is the 5th value (400), not the
    # middle index's 6th (480); p90 is the 9th (720), not the floor index's 10th (800).
    assert (t["min"], t["median"], t["p90"], t["max"]) == (80, 400, 720, 800)


def test_nearest_rank_quantiles() -> None:
    v = [80, 80, 160, 240, 640]
    assert (sp.nearest_rank(v, 0.5), sp.nearest_rank(v, 0.9)) == (160, 640)
    # n = 4, q = 0.5: nearest rank is the 2nd value; a floor-based index would give the 3rd
    assert sp.nearest_rank([10, 20, 30, 40], 0.5) == 20
    assert sp.nearest_rank([], 0.5) is None


def test_times_are_rounded_to_whole_milliseconds_not_truncated() -> None:
    # Frame-aligned seconds that a float carries a hair under the millisecond: 16.08 * 1000 is
    # 16079.999..., so truncating would put each 1 ms early.
    frames = [16.08, 16.24, 32.16, 32.48]
    assert [sp.to_ms(t) for t in frames] == [16080, 16240, 32160, 32480]
    assert [int(t * 1000) for t in frames] == [16079, 16239, 32159, 32479]  # what truncation gives
    assert sp.largest_shift_ms([["x", 16.0, 16.08]], [["x", 16.0, 16.16]]) == (80, 1)
    assert sp.largest_shift_ms([["x", 0.0, 0.08]], [["x", 0.0, 0.0806]]) == (1, 1)
    assert sp.to_ms(80, "ms") == 80


# --- the record is checked before it is scored: one test per check --------------------------


def valid_arm() -> tuple[dict, dict]:
    """r0 text-divergent, r1 timing-only divergent, r2 identical."""
    return arm_of(
        ("a b", "A b.", "A c."),
        ("d e", "D e.", "D e."),
        ("f g", "F g.", "F g."),
        timings={"r1": ([["D", 0.0, 0.08]], [["D", 0.0, 0.16]])},
    )


def test_the_valid_arm_is_accepted() -> None:
    arm, refs = valid_arm()
    sp.check_arm(arm, refs)


def _div(arm: dict, rid: str) -> dict:
    return next(d for d in arm["divergences"] if d["librispeech_id"] == rid)


CHECKS = {
    "checked": (lambda arm, refs: arm.update(checked=4), "checked 4 but 3 transcripts"),
    "text_divergent": (
        lambda arm, refs: arm.update(text_divergent=2),
        "text_divergent 2 but 1 transcripts store b",
    ),
    "timing_only_divergent": (
        lambda arm, refs: arm.update(timing_only_divergent=2),
        "timing_only_divergent 2 but 1 divergences have text_differs false",
    ),
    "duplicate": (
        lambda arm, refs: arm["divergences"].append(copy.deepcopy(_div(arm, "r0"))),
        "divergence r0 appears 2 times",
    ),
    "no transcript": (
        lambda arm, refs: _div(arm, "r0").update(librispeech_id="r9"),
        "divergence r9 has no transcript",
    ),
    "a text disagrees": (
        lambda arm, refs: _div(arm, "r0").update(a="A z."),
        "divergence r0 text disagrees with its transcript",
    ),
    "b text disagrees": (
        lambda arm, refs: _div(arm, "r0").update(b="A z."),
        "divergence r0 text disagrees with its transcript",
    ),
    "text_differs flag": (
        lambda arm, refs: _div(arm, "r1").update(text_differs=True),
        "divergence r1 text_differs flag disagrees with its texts",
    ),
    "timing-only with equal timings": (
        lambda arm, refs: _div(arm, "r1").update(b_timings=[["D", 0.0, 0.08]]),
        "divergence r1 is timing-only but its timings are equal",
    ),
    "timing-only differing only in confidence": (
        lambda arm, refs: _div(arm, "r1").update(b_timings=[["D", 0.0, 0.08, 0.7]]),
        "divergence r1 is timing-only but its timings are equal",
    ),
    "b equal to a": (
        lambda arm, refs: arm["transcripts"]["r2"].update(b="F g."),
        "transcript r2 stores a b equal to its a",
    ),
    "b without a divergence": (
        lambda arm, refs: arm["transcripts"]["r2"].update(b="F h."),
        "transcript r2 stores a b but has no text divergence",
    ),
    "reference": (lambda arm, refs: refs.pop("r2"), "1 transcripts without a reference, e.g. r2"),
}


@pytest.mark.parametrize("check", list(CHECKS))
def test_each_record_check_refuses_its_own_defect(check: str) -> None:
    arm, refs = valid_arm()
    change, message = CHECKS[check]
    change(arm, refs)
    with pytest.raises(ValueError, match=re.escape(message)):
        sp.check_arm(arm, refs)


def test_a_truncated_record_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "partial.json"
    p.write_text('{"runs": {', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        sp.load_record(p)


def test_the_digest_is_of_the_bytes_parsed(tmp_path: Path) -> None:
    p = tmp_path / "whole.json"
    p.write_text(json.dumps({"runs": {}}), encoding="utf-8")
    record, digest = sp.load_record(p)
    assert record == {"runs": {}}
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()


# --- controls ------------------------------------------------------------------------------

#: Every repeat_verdict the two writers can put in a record other than the accepted one: the
#: stock probe's, then each of scripts/compare_captures.py's (build_places) near misses.
NOT_ACCEPTED = [
    "NOT run-to-run deterministic: divergence counts below include noise",
    "",
    "not measured: no second fixed capture was given (--repeat)",
    "NOT run-to-run identical: two fixed captures differ on 1 in text and 0 in word timings only",
    "identical at one concurrency (32) only: not shown across occupancy",
    "identical at two concurrencies, but the padding is only declared in fixed.json: not the"
    " same-shape verdict",
    "identical, but the observed peak in flight is not recorded in repeat.json (configured 32 and"
    " 8): not shown across occupancy",
    sp.SAME_SHAPE_VERDICT + ".",
]


@pytest.mark.parametrize("verdict", NOT_ACCEPTED)
def test_scores_are_withheld_unless_the_repeat_verdict_is_the_identical_one(verdict) -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    out = sp.derive(stock_record(arm, refs, verdict=verdict), NO_WORDS, draws=0)
    run = out["runs"]["bfloat16"]
    scored = run["arms"]["ragged"]
    assert run["repeat_verdict_accepted"] is False
    assert scored["places"] is None and scored["flag"] is None and scored["confidence"] is None
    assert repr(verdict) in scored["scores_withheld"]
    assert scored["text_divergent"] == 1


def test_scores_are_given_when_the_repeat_verdict_is_the_identical_one() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    out = sp.derive(stock_record(arm, refs), NO_WORDS, draws=0)
    scored = out["runs"]["bfloat16"]["arms"]["ragged"]
    assert out["runs"]["bfloat16"]["repeat_verdict_accepted"] is True
    assert scored["flag"]["places"] == 1 and "scores_withheld" not in scored
    # The stock probe serves nothing: neither side is a served answer.
    assert scored["served_side"] is None


#: Repeat counts that do not show the same-shape verdict, one defect each. A bool or a float
#: equal to the denominator (True == 1, 1.0 == 1) is not a count: each count has its own case.
#: A numerator above its denominator is not "every one identical" either: a check that read
#: "at least n" would take it.
STOCK_REPEAT_AGAINST = {
    "none checked": {"alone_identical": 0, "batch_identical": 0, "checked": 0},
    "alone differs": {"alone_identical": 7, "batch_identical": 8, "checked": 8},
    "batch differs": {"alone_identical": 8, "batch_identical": 7, "checked": 8},
    "counts are bools": {"alone_identical": True, "batch_identical": True, "checked": True},
    "checked is a bool": {"alone_identical": 1, "batch_identical": 1, "checked": True},
    "checked is a float": {"alone_identical": 1, "batch_identical": 1, "checked": 1.0},
    "alone is a bool": {"alone_identical": True, "batch_identical": 1, "checked": 1},
    "alone is a float": {"alone_identical": 1.0, "batch_identical": 1, "checked": 1},
    "alone above checked": {"alone_identical": 2, "batch_identical": 1, "checked": 1},
    "batch is a bool": {"alone_identical": 1, "batch_identical": True, "checked": 1},
    "batch is a float": {"alone_identical": 1, "batch_identical": 1.0, "checked": 1},
    "batch above checked": {"alone_identical": 1, "batch_identical": 2, "checked": 1},
    "not a mapping": "8/8 identical",
    "absent": None,
}
CAPTURES_REPEAT_AGAINST = {
    "none compared": {"recordings": 0, "identical": 0, "digests_match": True},
    "one differs": {"recordings": 1, "identical": 0, "digests_match": True},
    "digests differ": {"recordings": 1, "identical": 1, "digests_match": False},
    "digests_match missing": {"recordings": 1, "identical": 1},
    "digests_match is 1": {"recordings": 1, "identical": 1, "digests_match": 1},
    "recordings is a bool": {"recordings": True, "identical": 1, "digests_match": True},
    "recordings is a float": {"recordings": 1.0, "identical": 1, "digests_match": True},
    "identical is a bool": {"recordings": 1, "identical": True, "digests_match": True},
    "identical is a float": {"recordings": 1, "identical": 1.0, "digests_match": True},
    "identical above recordings": {"recordings": 1, "identical": 2, "digests_match": True},
    "not a mapping": "1/1 identical",
    "absent": None,
}


def test_the_repeat_cases_differ_from_an_accepted_repeat_only_in_their_defect() -> None:
    # The cases above are withheld; these, the nearest accepted counts, are not. Without them
    # a check that refused every repeat would pass the cases above.
    for captures, repeat in (
        (False, {"alone_identical": 1, "batch_identical": 1, "checked": 1}),
        (True, {"recordings": 1, "identical": 1, "digests_match": True}),
    ):
        assert sp.repeat_contradicts(repeat, captures) is None


@pytest.mark.parametrize("case", list(STOCK_REPEAT_AGAINST))
def test_a_same_shape_verdict_the_stock_repeat_counts_do_not_show_is_not_accepted(case) -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    record = stock_record(arm, refs, repeat=STOCK_REPEAT_AGAINST[case])
    run = sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["repeat_verdict_accepted"] is False
    scored = run["arms"]["ragged"]
    assert scored["flag"] is None and scored["places"] is None
    assert "not what the run's own repeat counts show" in scored["scores_withheld"]


@pytest.mark.parametrize("case", list(CAPTURES_REPEAT_AGAINST))
def test_a_same_shape_verdict_the_captures_repeat_counts_do_not_show_is_not_accepted(
    case,
) -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."), every=False, ms=True)
    record = captures_record(arm, refs)
    record["runs"]["bfloat16"]["repeat"] = CAPTURES_REPEAT_AGAINST[case]
    run = sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["repeat_verdict_accepted"] is False
    assert (
        "not what the run's own repeat counts show"
        in captures_arm({"runs": {"bfloat16": run}})["scores_withheld"]
    )


def test_a_withheld_run_still_refuses_a_malformed_arm() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    arm["checked"] = 2
    with pytest.raises(ValueError, match="checked 2 but 1 transcripts"):
        sp.derive(stock_record(arm, refs, verdict="NOT run-to-run"), NO_WORDS, draws=0)
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    arm["confidence_only_divergent"] = 3
    with pytest.raises(ValueError, match="confidence_only_divergent 3 but 0 recordings"):
        sp.derive(stock_record(arm, refs, verdict="NOT run-to-run"), NO_WORDS, draws=0)


def test_a_scored_run_refuses_a_malformed_every_recording() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    arm["confidence_only_divergent"] = 3
    with pytest.raises(ValueError, match="confidence_only_divergent 3 but 0 recordings"):
        sp.derive(stock_record(arm, refs), NO_WORDS, draws=0)


@pytest.mark.parametrize("pc", [ABSENT, None])
def test_a_fixed_arm_zero_is_uninterpretable_without_the_positive_control(pc) -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    out = sp.derive(stock_record(arm, refs, name="fixed", pc=pc), NO_WORDS, draws=0)
    assert out["runs"]["bfloat16"]["arms"]["fixed"]["zero_reading"].startswith("uninterpretable")


def zero_and_diverged(name: str, *, control_arm: str = "ragged", pc=PRESENT) -> dict:
    """A stock record whose arm ``name`` has no divergence and whose ``control_arm`` has one,
    with the positive control reading ``pc``."""
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs, name=name, pc=pc)
    record["runs"]["bfloat16"]["arms"][control_arm] = arm_of(("no sir", "No, sir.", "Now, sir."))[0]
    return record


@pytest.mark.parametrize("pc", [ABSENT, None, "NOT present: nothing diverged", "Present: x"])
def test_a_zero_is_uninterpretable_when_the_control_does_not_read_present(pc) -> None:
    # The ragged arm diverged, so the arms alone would show the control; the run's own
    # reading of it is not "present", and that is not overruled.
    run = sp.derive(zero_and_diverged("fixed", pc=pc), NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["positive_control_seen_in_arms"] is True
    assert run["arms"]["fixed"]["zero_reading"].startswith("uninterpretable")


def test_a_fixed_arm_zero_with_the_control_present_names_the_shape_axis() -> None:
    run = sp.derive(zero_and_diverged("fixed"), NO_WORDS, draws=0)["runs"]["bfloat16"]
    reading = run["arms"]["fixed"]["zero_reading"]
    assert "shape axis" in reading and "neighbour content" in reading
    assert "shape change" in run["positive_control_axis"]
    assert run["positive_control_seen_in_arms"] is True


def test_a_two_shape_arm_zero_with_the_control_present_is_a_plain_zero() -> None:
    run = sp.derive(zero_and_diverged("equalised"), NO_WORDS, draws=0)["runs"]
    reading = run["bfloat16"]["arms"]["equalised"]["zero_reading"]
    assert reading == "zero, with the positive control present on the shape axis"


@pytest.mark.parametrize("control_arm", [None, "fixed"])
def test_a_positive_control_string_no_two_shape_arm_shows_is_not_taken(control_arm) -> None:
    # The run says "present", but no ragged or equalised arm has a divergence in the record
    # (none at all, or only the fixed arm, which is not a two-shape arm). The string alone
    # would make this zero read as a plain zero.
    if control_arm is None:
        arm, refs = arm_of(("no sir", "No sir.", "No sir."))
        record = stock_record(arm, refs, name="equalised")
    else:
        record = zero_and_diverged("equalised", control_arm=control_arm)
    run = sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["positive_control_seen_in_arms"] is False
    reading = run["arms"]["equalised"]["zero_reading"]
    assert reading.startswith("uninterpretable")
    assert "but no two-shape arm of this run has a divergence in the record" in reading


def test_a_captures_arm_with_no_divergence_is_its_own_absent_control() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    record = captures_record(arm, refs)  # its positive_control reads "present: ..."
    scored = captures_arm(sp.derive(record, NO_WORDS, draws=0))
    assert scored["zero_reading"].startswith("uninterpretable")
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."), every=False, ms=True)
    run = sp.derive(captures_record(arm, refs), NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["positive_control_seen_in_arms"] is True


def test_an_arm_that_diverged_has_no_zero_reading() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    out = sp.derive(stock_record(arm, refs, name="fixed", pc=ABSENT), NO_WORDS, draws=0)
    assert out["runs"]["bfloat16"]["arms"]["fixed"]["zero_reading"] is None


def test_a_timing_only_divergence_is_a_divergence_for_the_zero_and_the_control() -> None:
    # No text differs anywhere; the ragged arm's one divergence is in the word timings. It is
    # not a zero, and it is the control the equalised arm's zero is read against.
    timing_only_arm, refs = arm_of(("d e", "D e.", "D e."), timings={"r0": R1_TIMINGS})
    zero, _ = arm_of(("d e", "D e.", "D e."))
    record = stock_record(timing_only_arm, refs)
    record["runs"]["bfloat16"]["arms"]["equalised"] = zero
    run = sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["arms"]["ragged"]["zero_reading"] is None
    assert run["positive_control_seen_in_arms"] is True
    assert run["arms"]["equalised"]["zero_reading"] == (
        "zero, with the positive control present on the shape axis"
    )


def test_each_dtype_draws_its_own_nulls() -> None:
    arm, refs = arm_of(*[one_error_rows(k) for k in range(4)], ("a b c d e f g h",) * 3)
    record = stock_record(arm, refs)
    record["runs"]["float32"] = copy.deepcopy(record["runs"]["bfloat16"])
    out = sp.derive(record, NO_WORDS, draws=10, seed=5)
    for dtype in ("bfloat16", "float32"):
        null = out["runs"][dtype]["arms"]["ragged"]["flag"]["b"]["null"]
        assert null["seed"] == f"5/{dtype}/ragged/places/b"


def test_complete_says_whether_the_record_finished() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    record = stock_record(arm, refs)
    assert sp.derive(record, NO_WORDS, draws=0)["complete"] is False
    record["finished"] = "2026-09-24T18:22:49+0300"
    assert sp.derive(record, NO_WORDS, draws=0)["complete"] is True


#: What a stock record says it asked for and what it read off the built pipeline. Named here,
#: not taken from sp.COPIED, so dropping one from COPIED turns this test red.
REQUESTED_AND_OBSERVED = {
    "use_cuda_graphs": False,
    "word_confidence": "nemo-shipped",
    "decoder_graphs_observed": True,
    "decoder_graphs_mode_observed": "full_graph",
    "word_confidence_observed": {
        "mode_requested": "nemo-shipped",
        "decoder_step_confidence": True,
        "nonzero_conf_words_on_guard_recording": 8,
    },
}


def test_a_stock_records_requested_and_observed_readings_are_copied() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    record = {**stock_record(arm, refs), **copy.deepcopy(REQUESTED_AND_OBSERVED)}
    run = record["runs"]["bfloat16"]
    run["decoder_graphs_observed"] = False
    run["word_confidence_observed"] = {"decoder_step_confidence": False}
    out = sp.derive(record, NO_WORDS, draws=0)
    for key, value in REQUESTED_AND_OBSERVED.items():
        assert out["record_settings"][key] == value, key
    derived_run = out["runs"]["bfloat16"]
    assert derived_run["decoder_graphs_observed"] is False
    assert derived_run["word_confidence_observed"] == {"decoder_step_confidence": False}


#: A server-captures record's provenance and evidence. Named here, not taken from sp.COPIED, so
#: dropping one from COPIED turns the test red; each value differs from the fixture's default.
CAPTURES_EVIDENCE = {
    "model": "model-x",
    "model_revision": "rev-1",
    "matmul_precision": "highest",
    "att_context_size": [70, 13],
    "chunk_ms": 1120,
    "bucket": 128,
    "batch": 128,
    "execution": "eager",
    "decoder_graphs": True,
    "word_confidence": "paper-best",
    "timing_unit": "ms",
    "tracked_files_modified": False,
    "verbatim_commit": "c0ffee",
    "gpu_uuid": "GPU-0",
    "captures": {
        "a_fixed": {"path": "fixed.json", "sha256": "1" * 64, "finals_digest": "2" * 64},
        "b_ragged": {"path": "ragged.json", "sha256": "3" * 64, "finals_digest": "4" * 64},
    },
}


def test_the_output_carries_the_records_provenance_and_the_repeat_it_was_judged_by() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "Now sir."), every=False, ms=True)
    record = captures_record(arm, refs, **copy.deepcopy(CAPTURES_EVIDENCE))
    out = sp.derive(record, NO_WORDS, draws=0)
    for key, value in CAPTURES_EVIDENCE.items():
        assert out["record_settings"][key] == value, key
    run = out["runs"]["bfloat16"]
    assert run["repeat_verdict_accepted"] is True
    assert run["repeat"] == {"recordings": 1, "identical": 1, "digests_match": True}
    # A stock run's repeat counts, on the scored path and on the withheld one.
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    for verdict, accepted in ((None, True), ("not measured", False)):
        out = sp.derive(stock_record(arm, refs, verdict=verdict), NO_WORDS, draws=0)
        run = out["runs"]["bfloat16"]
        assert (run["repeat_verdict_accepted"], run["repeat"]) == (accepted, STOCK_REPEAT)


# --- what a record must say about itself ---------------------------------------------------


@pytest.mark.parametrize("fake", [True, "true", None, 1])
def test_a_record_from_the_fake_pipeline_is_refused(fake) -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    record = stock_record(arm, refs)
    record["fake_pipeline"] = fake
    with pytest.raises(ValueError, match=re.escape(f"fake_pipeline is {fake!r}")):
        sp.derive(record, NO_WORDS, draws=0)
    record["fake_pipeline"] = False
    assert sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]["repeat_verdict_accepted"]


@pytest.mark.parametrize("fake", [True, "true", None, 1])
def test_a_captures_record_carrying_the_fake_marker_is_refused(fake) -> None:
    # scripts/compare_captures.py's build_places writes "fake_pipeline": true into a places
    # record built from a capture probes/server_frozen_answers.py took through its test seams.
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."), every=False, ms=True)
    record = captures_record(arm, refs, fake_pipeline=fake)
    with pytest.raises(ValueError, match=re.escape(f"record refused: fake_pipeline is {fake!r}")):
        sp.derive(record, NO_WORDS, draws=0)
    record["fake_pipeline"] = False
    assert sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]["repeat_verdict_accepted"]


@pytest.mark.parametrize("fake", [True, None])
def test_a_confidence_capture_stamped_as_a_test_double_is_refused(tmp_path, fake) -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    record = captures_record(arm, refs)
    path = tmp_path / "capture.json"
    words = {"r0": [["The", 0, 80, 0.5]]}
    path.write_text(
        json.dumps({"fake_pipeline": fake, "recordings": {"r0": {"words": words["r0"]}}})
    )
    with pytest.raises(ValueError, match=re.escape(f"has fake_pipeline {fake!r}; a ")):
        sp.load_confidence_capture(path, "a", record)
    path.write_text(
        json.dumps({"fake_pipeline": False, "recordings": {"r0": {"words": words["r0"]}}})
    )
    assert sp.load_confidence_capture(path, "a", record)["words"] == words


def test_a_confidence_capture_that_is_not_a_json_object_is_refused(tmp_path) -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    path = tmp_path / "capture.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        sp.load_confidence_capture(path, "a", captures_record(arm, refs))


def test_a_captures_record_measures_timings_in_ms() -> None:
    arm, refs = arm_of(
        ("hi there", "Hi there.", "Hi there."),
        timings={
            "r0": (
                [["Hi", 80, 400], ["there.", 400, 560]],
                [["Hi", 320, 400], ["there.", 400, 560]],
            )
        },
        every=False,
    )
    scored = captures_arm(sp.derive(captures_record(arm, refs), NO_WORDS, draws=0))
    assert scored["timing_only_shift_ms"]["max"] == 240
    assert scored["served_side"] == "a"
    assert scored["sides"]["a"].startswith("the served answer")


@pytest.mark.parametrize("unit", [None, "s", "us"])
def test_a_captures_record_must_say_its_timings_are_ms(unit) -> None:
    # scripts/compare_captures.py writes the wire's integer milliseconds unchanged, with
    # timing_unit "ms". A record without it is refused, not read as seconds.
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    record = captures_record(arm, refs)
    if unit is None:
        del record["timing_unit"]
    else:
        record["timing_unit"] = unit
    with pytest.raises(ValueError, match=re.escape(f"must say timing_unit 'ms'; it says {unit!r}")):
        sp.derive(record, NO_WORDS, draws=0)


def test_score_arm_refuses_a_timing_unit_it_cannot_convert() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    with pytest.raises(ValueError, match="timing_unit 'us' is not one of 's', 'ms'"):
        sp.score_arm("ragged", arm, refs, NO_WORDS, timing_unit="us", draws=0)


def test_a_stock_record_that_says_its_timings_are_not_seconds_is_refused() -> None:
    arm, refs = arm_of(("no sir", "No, sir.", "Now, sir."))
    record = stock_record(arm, refs)
    record["timing_unit"] = "s"
    sp.derive(record, NO_WORDS, draws=0)
    record["timing_unit"] = "ms"
    with pytest.raises(ValueError, match="stock record's timings are seconds; it says"):
        sp.derive(record, NO_WORDS, draws=0)


def test_a_captures_record_whose_bucket_and_batch_disagree_is_refused() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    sp.derive(captures_record(arm, refs, bucket=64), NO_WORDS, draws=0)
    with pytest.raises(ValueError, match="bucket 128 and batch 64 disagree"):
        sp.derive(captures_record(arm, refs, bucket=128), NO_WORDS, draws=0)


def test_a_captures_arm_must_be_fixed_vs_ragged() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    record = captures_record(arm, refs)
    record["runs"]["bfloat16"]["arms"] = {"ragged_vs_fixed": arm}
    with pytest.raises(ValueError, match="server-captures arm 'ragged_vs_fixed'"):
        sp.derive(record, NO_WORDS, draws=0)


def test_an_unknown_source_is_refused() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    with pytest.raises(ValueError, match="source 'somewhere'"):
        sp.derive(captures_record(arm, refs, source="somewhere"), NO_WORDS, draws=0)


def test_a_confidence_target_that_is_not_in_the_record_is_refused() -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    given = conf("a", r0=[0.9] * 6, r1=[0.9] * 4)
    with pytest.raises(ValueError, match="no arm bfloat16/ragged"):
        sp.derive(
            captures_record(arm, refs),
            NO_WORDS,
            confidence_at=("bfloat16", "ragged"),
            confidence=given,
        )


def test_a_confidence_capture_is_refused_for_a_stock_record() -> None:
    arm, refs = arm_of(*CONF_ROWS)
    given = conf("a", r0=[0.9] * 6, r1=[0.9] * 4, ms=False)
    with pytest.raises(ValueError, match="stock record carries its confidence in every_recording"):
        sp.derive(
            stock_record(arm, refs),
            NO_WORDS,
            confidence_at=("bfloat16", "ragged"),
            confidence=given,
        )
    with pytest.raises(ValueError, match="goes with a server-captures record"):
        sp.parse_confidence_at("bfloat16/ragged/b", stock_record(arm, refs))


def test_two_confidence_inputs_for_one_arm_are_refused() -> None:
    arm, refs = arm_of(*CONF_ROWS, ms=True)  # carries every_recording
    with pytest.raises(ValueError, match="two confidence inputs for this arm"):
        sp.derive(
            captures_record(arm, refs),
            NO_WORDS,
            confidence_at=("bfloat16", sp.CAPTURES_ARM),
            confidence=conf("a", r0=[0.9] * 6, r1=[0.9] * 4),
        )


def test_a_confidence_capture_is_attached_only_at_its_own_run_and_arm() -> None:
    # Two runs of the same captures arm; the capture belongs to the bfloat16 one. The float32
    # run must not be scored with it.
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    record = captures_record(arm, refs)
    record["runs"]["float32"] = copy.deepcopy(record["runs"]["bfloat16"])
    out = sp.derive(
        record,
        NO_WORDS,
        draws=0,
        confidence_at=("bfloat16", sp.CAPTURES_ARM),
        confidence=conf("a", r0=[0.9] * 6, r1=[0.9, 0.9, 0.9, 0.2]),
    )
    assert out["runs"]["bfloat16"]["arms"][sp.CAPTURES_ARM]["confidence"]["a"]["status"] == (
        "scored"
    )
    other = out["runs"]["float32"]["arms"][sp.CAPTURES_ARM]["confidence"]["a"]
    assert other["status"] == "confidence unavailable"
    assert other["reason"].startswith("no confidence input covers side a")


def test_the_confidence_target_defaults_to_the_served_side_of_a_captures_record() -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    record = captures_record(arm, refs)
    assert sp.parse_confidence_at(None, record) == ("bfloat16", sp.CAPTURES_ARM, "a")
    assert sp.parse_confidence_at("bfloat16/x/b", record) == ("bfloat16", "x", "b")
    record["runs"]["float32"] = record["runs"]["bfloat16"]
    with pytest.raises(ValueError, match="required when the record has more than one run"):
        sp.parse_confidence_at(None, record)


# --- the serving setting -------------------------------------------------------------------

SERVING = {
    "chunk_ms": 160,
    "att_context_size": [70, 1],
    "dtype": "bfloat16",
    "matmul_precision": "high",
    "bucket": 64,
    "execution": "graph path",
    "decoder_graphs": "off",
    "word_confidence": "off",
}


def test_a_stock_record_at_another_setting_lists_every_difference() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["runs"]["float32"] = record["runs"]["bfloat16"]
    diffs = sp.setting_differences(record, {**SERVING, "bucket": None})
    text = "\n".join(diffs)
    assert "chunk_ms: record 1120, serving 160" in text
    # The probe asked for these; nothing in this record reads them off the built pipeline.
    assert "att_context_size (requested, not observed): record [70, 13], serving [70, 1]" in diffs
    assert "matmul_precision (requested, not observed): record 'highest', serving 'high'" in diffs
    assert "dtype: record 'float32', serving 'bfloat16'" in text
    assert "dtype: record 'bfloat16'" not in text
    assert "batch: record 32 rows in the stock pipeline, serving bucket <placeholder" in text
    assert "execution (requested, not observed): not recorded in the record" in text
    assert "word_confidence (requested, not observed): not recorded in the record" in text
    for dtype in ("bfloat16", "float32"):
        assert (
            f"decoder_graphs (observed in the {dtype} run): not recorded in the record,"
            " serving 'off'" in diffs
        )
        assert (
            f"decoder_step_confidence (observed in the {dtype} run): not recorded in the"
            " record, serving 'off'" in diffs
        )
    assert any(d.startswith("sessions:") for d in diffs)


def test_execution_is_the_encoders_request_and_decoder_graphs_are_their_own_line() -> None:
    # Two settings. execution is how the encoder step runs: in a stock record only the spec's
    # use_cuda_graphs, a request. The decoder's CUDA graphs are read from the built pipeline
    # (decoder_graphs_observed). Each input below is one where taking one for the other shows:
    # the first would report an execution difference that does not exist and hide the decoder
    # one; the second would hide a real encoder difference.
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    serving = {**SERVING, "execution": "eager", "decoder_graphs": "off"}
    record = stock_record(arm, refs)
    run = record["runs"]["bfloat16"]
    record["use_cuda_graphs"], run["decoder_graphs_observed"] = False, True
    diffs = sp.setting_differences(record, serving)
    assert not any(d.startswith("execution") for d in diffs)
    assert "decoder_graphs (observed in the bfloat16 run): record 'on', serving 'off'" in diffs
    record["use_cuda_graphs"], run["decoder_graphs_observed"] = True, False
    diffs = sp.setting_differences(record, serving)
    assert "execution (requested, not observed): record 'graph path', serving 'eager'" in diffs
    assert not any(d.startswith("decoder_graphs") for d in diffs)
    diffs = sp.setting_differences(record, {**serving, "decoder_graphs": None})
    assert (
        "decoder_graphs (observed in the bfloat16 run): record 'off', serving <placeholder: not"
        " given>" in diffs
    )


def test_a_runs_own_decoder_graphs_reading_is_used_before_the_top_level_one() -> None:
    # The probe stamps each run's reading, and at the top level the last dtype built.
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["decoder_graphs_observed"] = True
    record["runs"]["float32"] = copy.deepcopy(record["runs"]["bfloat16"])
    record["runs"]["bfloat16"]["decoder_graphs_observed"] = False
    diffs = sp.setting_differences(record, {**SERVING, "dtype": None})
    assert not any(d.startswith("decoder_graphs (observed in the bfloat16") for d in diffs)
    # The float32 run carries no reading of its own: the top-level one is used, and it says so
    # rather than calling it the float32 run's.
    assert (
        "decoder_graphs (observed, the record's top-level reading; the float32 run has none):"
        " record 'on', serving 'off'" in diffs
    )
    assert not any(d.startswith("decoder_graphs (observed in the float32 run)") for d in diffs)


def test_a_stock_records_observed_decoder_step_confidence_is_compared() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["word_confidence"] = "off"
    record["word_confidence_observed"] = {
        "mode_requested": "off",
        "decoder_step_confidence": True,
        "nonzero_conf_words_on_guard_recording": 0,
    }
    diffs = sp.setting_differences(record, SERVING)
    assert not any(d.startswith("word_confidence") for d in diffs)  # the request matches
    line = (
        "decoder_step_confidence (observed, the record's top-level reading; the bfloat16 run has"
        " none): record 'on', serving 'off'"
    )
    assert line in diffs
    # The run's own reading comes first.
    record["runs"]["bfloat16"]["word_confidence_observed"] = {
        **record["word_confidence_observed"],
        "decoder_step_confidence": False,
    }
    diffs = sp.setting_differences(record, SERVING)
    assert not any(d.startswith("decoder_step_confidence") for d in diffs)
    diffs = sp.setting_differences(record, {**SERVING, "word_confidence": "paper-best"})
    assert "word_confidence (requested, not observed): record 'off', serving 'paper-best'" in diffs
    line = "decoder_step_confidence (observed in the bfloat16 run): record 'off', serving 'on'"
    assert line in diffs


def test_a_stock_records_observed_att_context_is_compared_instead_of_its_request() -> None:
    # The request matches the serving setting and the built encoder does not: only the reading
    # can show it. Where a run has no reading, the top-level one is used and labelled so.
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["att_context_size"] = [70, 1]
    record["runs"]["float32"] = copy.deepcopy(record["runs"]["bfloat16"])
    record["runs"]["bfloat16"]["att_context_size_observed"] = [70, 13]
    serving = {**SERVING, "dtype": None}
    diffs = sp.setting_differences(record, serving)
    assert (
        "att_context_size (observed in the bfloat16 run): record [70, 13], serving [70, 1]" in diffs
    )
    assert (
        "att_context_size (observed in the float32 run): not recorded in the record, serving"
        " [70, 1]" in diffs
    )
    assert not any(d.startswith("att_context_size (requested") for d in diffs)
    record["att_context_size_observed"] = [70, 13]
    diffs = sp.setting_differences(record, serving)
    top = "att_context_size (observed, the record's top-level reading; the float32 run has none)"
    assert f"{top}: record [70, 13], serving [70, 1]" in diffs
    # Observed equal to serving: no line, whatever the request says.
    for run in record["runs"].values():
        run["att_context_size_observed"] = [70, 1]
    record["att_context_size"] = [70, 13]
    assert not any(
        d.startswith("att_context_size") for d in sp.setting_differences(record, serving)
    )
    out = sp.derive(record, NO_WORDS, draws=0)
    assert out["record_settings"]["att_context_size_observed"] == [70, 13]
    assert out["runs"]["bfloat16"]["att_context_size_observed"] == [70, 1]


def test_a_stock_records_top_level_att_context_reading_alone_is_compared() -> None:
    # Only the top level carries a reading; no run has one. The request matches the serving
    # setting and the reading does not, so a check that looked for readings in the runs only
    # would compare the request and hide the difference.
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["att_context_size"] = [70, 1]
    record["att_context_size_observed"] = [70, 13]
    assert not any("att_context_size_observed" in run for run in record["runs"].values())
    diffs = sp.setting_differences(record, SERVING)
    assert (
        "att_context_size (observed, the record's top-level reading; the bfloat16 run has none):"
        " record [70, 13], serving [70, 1]" in diffs
    )
    assert [d for d in diffs if d.startswith("att_context_size")] == [
        "att_context_size (observed, the record's top-level reading; the bfloat16 run has none):"
        " record [70, 13], serving [70, 1]"
    ]


def test_a_runs_own_null_reading_is_not_replaced_by_the_top_level_one() -> None:
    # The top-level reading is the last dtype built. A run that carries the key as null was
    # read and gave nothing; another build's value is not put in its place.
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["decoder_graphs_observed"] = False
    record["runs"]["bfloat16"]["decoder_graphs_observed"] = None
    diffs = sp.setting_differences(record, {**SERVING, "decoder_graphs": "off"})
    assert (
        "decoder_graphs (observed in the bfloat16 run): not recorded in the record, serving 'off'"
        in diffs
    )
    assert not any("top-level" in d for d in diffs)


def test_a_stock_records_matmul_is_a_request_even_beside_an_observed_att_context() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."))
    record = stock_record(arm, refs)
    record["runs"]["bfloat16"]["att_context_size_observed"] = [70, 1]
    diffs = sp.setting_differences(record, SERVING)
    assert "matmul_precision (requested, not observed): record 'highest', serving 'high'" in diffs
    assert not any(d.startswith("att_context_size") for d in diffs)


def test_a_captures_record_at_the_serving_setting_lists_no_difference() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    assert sp.setting_differences(captures_record(arm, refs), SERVING) == []
    diffs = sp.setting_differences(captures_record(arm, refs, bucket=128), SERVING)
    assert diffs == ["bucket: record 128, serving 64"]
    diffs = sp.setting_differences(
        captures_record(arm, refs, execution=None), {**SERVING, "execution": None}
    )
    assert diffs == ["execution: not recorded in the record, serving <placeholder: not given>"]


@pytest.mark.parametrize(
    ("field", "value", "line"),
    [
        ("chunk_ms", 1120, "chunk_ms: record 1120, serving 160"),
        ("att_context_size", [70, 13], "att_context_size: record [70, 13], serving [70, 1]"),
        (
            "matmul_precision",
            "highest",
            "matmul_precision (declared, not observed): record 'highest', serving 'high'",
        ),
        ("word_confidence", "paper-best", "word_confidence: record 'paper-best', serving 'off'"),
        ("execution", "eager", "execution: record 'eager', serving 'graph path'"),
        ("decoder_graphs", True, "decoder_graphs: record 'on', serving 'off'"),
        ("decoder_graphs", None, "decoder_graphs: not recorded in the record, serving 'off'"),
    ],
)
def test_a_captures_record_lists_each_setting_it_differs_in(field, value, line) -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    assert sp.setting_differences(captures_record(arm, refs, **{field: value}), SERVING) == [line]


def test_a_captures_record_in_another_dtype_lists_it() -> None:
    arm, refs = arm_of(("no sir", "No sir.", "No sir."), every=False, ms=True)
    record = captures_record(arm, refs)
    record["runs"] = {"float32": record["runs"]["bfloat16"]}
    assert sp.setting_differences(record, SERVING) == [
        "dtype: record 'float32', serving 'bfloat16'"
    ]


# --- end to end ----------------------------------------------------------------------------


def test_main_scores_a_captures_record_with_and_without_a_confidence_capture(tmp_path) -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    record = tmp_path / "places.json"
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("the\ncat\n", encoding="utf-8")
    capture = tmp_path / "capture.json"
    served = {"r0": "The cat sat on the mat.", "r1": "A dog ran hum."}
    low = {"r0": [0.9] * 6, "r1": [0.9, 0.9, 0.9, 0.2]}
    capture.write_text(
        json.dumps(
            {
                "recordings": {
                    rid: {"text": text, "words": words_of(text, low[rid], ms=True)}
                    for rid, text in served.items()
                }
            }
        ),
        encoding="utf-8",
    )
    named = {"a_fixed": {"sha256": hashlib.sha256(capture.read_bytes()).hexdigest()}}
    record.write_text(json.dumps(captures_record(arm, refs, captures=named)), encoding="utf-8")
    common = ["--wordlist", str(wordlist), "--draws", "10", "--serving-bucket", "64"]
    common += ["--serving-execution", "graph path", "--serving-decoder-graphs", "off"]

    def run(*extra: str) -> dict:
        out = tmp_path / f"derived-{len(list(tmp_path.glob('derived-*')))}.json"
        assert sp.main([str(record), "--out", str(out), *common, *extra]) == 0
        return json.loads(out.read_text(encoding="utf-8"))

    without = run()
    scored = captures_arm(without)
    assert scored["confidence"]["a"]["status"] == "confidence unavailable"
    assert without["not_a_row"]["setting_differences"] == []
    assert without["input"]["sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()

    with_conf = run("--confidence", str(capture))
    c = captures_arm(with_conf)["confidence"]["a"]
    assert (c["status"], c["side"], c["k_words"]) == ("scored", "a", 1)
    assert c["flag"]["precision"] == 1.0 and str(capture) in c["source"]


#: Every serving flag, each at a value the captures_record fixture does not have.
EVERY_FLAG = [
    "--serving-chunk-ms", "1120",
    "--serving-att-context", "70,13",
    "--serving-dtype", "float32",
    "--serving-matmul", "highest",
    "--serving-bucket", "128",
    "--serving-execution", "eager",
    "--serving-decoder-graphs", "on",
    "--serving-word-confidence", "paper-best",
]  # fmt: skip


def places_file(tmp_path: Path, **top) -> Path:
    """A captures record over CONF_ROWS, and a word list beside it, in ``tmp_path``."""
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    path = tmp_path / "places.json"
    path.write_text(json.dumps(captures_record(arm, refs, **top)), encoding="utf-8")
    (tmp_path / "words.txt").write_text("the\ncat\n", encoding="utf-8")
    return path


def scorer(record: Path, *argv: str) -> int:
    """``main`` on ``record`` with the word list beside it and no null draws."""
    words = record.with_name("words.txt")
    return sp.main([str(record), "--wordlist", str(words), "--draws", "0", *argv])


def derived(tmp_path: Path, record: Path, *argv: str) -> dict:
    out = tmp_path / "derived.json"
    assert scorer(record, "--out", str(out), *argv) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_serving_flag_reaches_the_setting_comparison(tmp_path) -> None:
    got = derived(tmp_path, places_file(tmp_path), *EVERY_FLAG)["not_a_row"]
    assert got["serving_setting_compared"] == {
        "chunk_ms": 1120,
        "att_context_size": [70, 13],
        "dtype": "float32",
        "matmul_precision": "highest",
        "bucket": 128,
        "execution": "eager",
        "decoder_graphs": "on",
        "word_confidence": "paper-best",
    }
    assert got["setting_differences"] == [
        "chunk_ms: record 160, serving 1120",
        "att_context_size: record [70, 1], serving [70, 13]",
        "matmul_precision (declared, not observed): record 'high', serving 'highest'",
        "dtype: record 'bfloat16', serving 'float32'",
        "word_confidence: record 'off', serving 'paper-best'",
        "bucket: record 64, serving 128",
        "execution: record 'graph path', serving 'eager'",
        "decoder_graphs: record 'off', serving 'on'",
    ]


def test_the_serving_defaults_are_the_ones_stated(tmp_path) -> None:
    got = derived(tmp_path, places_file(tmp_path))["not_a_row"]["serving_setting_compared"]
    assert got == {
        "chunk_ms": 160,
        "att_context_size": [70, 1],
        "dtype": "bfloat16",
        "matmul_precision": "high",
        "bucket": None,
        "execution": None,
        "decoder_graphs": None,
        "word_confidence": "off",
    }


# --- a refusal, and --out -------------------------------------------------------------------


def test_a_refusal_is_one_line_on_stderr_and_exit_1(tmp_path, capsys) -> None:
    # The integrator's case: a record stamped by a test double. No traceback, nothing written.
    record = places_file(tmp_path, fake_pipeline=True)
    out = tmp_path / "derived.json"
    assert scorer(record, "--out", str(out)) == 1
    err = capsys.readouterr().err
    assert err.startswith("step1_places: record refused: fake_pipeline is True; a test double")
    assert err.count("\n") == 1 and "Traceback" not in err
    assert not out.exists()
    assert scorer(tmp_path / "missing.json") == 1
    assert "No such file or directory" in capsys.readouterr().err


@pytest.mark.parametrize("which", ["record", "wordlist", "confidence"])
def test_an_out_that_is_an_input_is_refused(tmp_path, capsys, which) -> None:
    record = places_file(tmp_path)
    wordlist = tmp_path / "words.txt"
    capture = tmp_path / "capture.json"
    write_capture(capture, {"r0": [["The", 0, 80]], "r1": [["A", 0, 80]]})
    given = {"record": record, "wordlist": wordlist, "confidence": capture}
    before = {name: path.read_bytes() for name, path in given.items()}
    argv = [str(record), "--wordlist", str(wordlist), "--confidence", str(capture)]
    assert sp.main([*argv, "--out", str(given[which])]) == 1
    assert f"--out {given[which]} is the input {given[which]}" in capsys.readouterr().err
    assert {name: path.read_bytes() for name, path in given.items()} == before


@pytest.mark.parametrize("which", ["out", "tmp"])
def test_an_out_that_exists_is_not_written_over(tmp_path, capsys, which) -> None:
    record = places_file(tmp_path)
    out = tmp_path / "derived.json"
    there = {"out": out, "tmp": tmp_path / "derived.json.tmp"}[which]
    there.write_text("sentinel", encoding="utf-8")
    assert scorer(record, "--out", str(out)) == 1
    assert f"--out: {there} exists; it is not written over" in capsys.readouterr().err
    assert there.read_text(encoding="utf-8") == "sentinel"
    assert which == "out" or not out.exists()


@pytest.mark.parametrize("which", ["out", "tmp"])
def test_an_out_that_appears_while_scoring_is_not_written_over(
    tmp_path, capsys, monkeypatch, which
) -> None:
    record = places_file(tmp_path)
    out = tmp_path / "derived.json"
    there = {"out": out, "tmp": tmp_path / "derived.json.tmp"}[which]
    derive = sp.derive

    def appears(*args: Any, **kwargs: Any) -> dict:
        there.write_text("sentinel", encoding="utf-8")
        return derive(*args, **kwargs)

    monkeypatch.setattr(sp, "derive", appears)
    assert scorer(record, "--out", str(out)) == 1
    assert f"--out: {there} appeared while scoring" in capsys.readouterr().err
    assert there.read_text(encoding="utf-8") == "sentinel"
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == sorted(["places.json", "words.txt", there.name])


def test_a_new_out_is_written_whole_and_leaves_no_temporary_file(tmp_path) -> None:
    record = places_file(tmp_path)
    out = derived(tmp_path, record)
    assert out["input"]["sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["derived.json", "places.json", "words.txt"]


def write_capture(path: Path, words: dict) -> str:
    path.write_text(json.dumps({"recordings": {r: {"words": w} for r, w in words.items()}}))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_capture_with_no_confidence_element_gives_confidence_unavailable(tmp_path) -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    record = captures_record(arm, refs)
    path = tmp_path / "capture.json"
    write_capture(path, {"r0": [["The", 0, 80]], "r1": [["A", 0, 80]]})
    given = sp.load_confidence_capture(path, "a", record)
    out = sp.derive(
        record, NO_WORDS, draws=0, confidence_at=("bfloat16", sp.CAPTURES_ARM), confidence=given
    )
    c = captures_arm(out)["confidence"]["a"]
    assert c["status"] == "confidence unavailable"
    assert "no word entry" in c["reason"] and "carries a confidence" in c["reason"]


def test_a_capture_where_some_words_carry_a_confidence_and_some_do_not_is_refused(
    tmp_path,
) -> None:
    # probes/server_frozen_answers.py appends the confidence only where the wire carried one,
    # so it can write this. It is refused when loaded, whatever the record's verdict, rather
    # than read as "confidence unavailable" or scored on the recordings that have one.
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    record = captures_record(arm, refs, repeat_verdict="not measured")
    path = tmp_path / "capture.json"
    write_capture(
        path,
        {
            "r0": words_of("The cat sat on the mat.", [0.9] * 6, ms=True),
            "r1": [w[:3] for w in words_of("A dog ran hum.", ms=True)],
        },
    )
    message = f"{path} is mixed: 6 word entries carry a confidence and 4 do not"
    with pytest.raises(ValueError, match=re.escape(message)):
        sp.load_confidence_capture(path, "a", record)


@pytest.mark.parametrize(
    ("entry", "size"), [(["The", 0, 80, 0.9, 0.1], "5"), (["The", 0], "2"), ("The", "str")]
)
def test_a_capture_word_entry_of_another_shape_is_refused_when_loaded(
    tmp_path, entry, size
) -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    path = tmp_path / "capture.json"
    write_capture(path, {"r0": [["A", 0, 80, 0.5], entry]})
    with pytest.raises(ValueError, match=re.escape(f"has word entries of {size} elements")):
        sp.load_confidence_capture(path, "a", captures_record(arm, refs))


def test_a_confidence_capture_that_is_not_the_named_capture_is_refused(tmp_path) -> None:
    arm, refs = arm_of(*CONF_ROWS, every=False, ms=True)
    path = tmp_path / "capture.json"
    digest = write_capture(path, {"r0": [["The", 0, 80, 0.5]]})
    named = {"a_fixed": {"sha256": digest}, "b_ragged": {"sha256": "0" * 64}}
    record = captures_record(arm, refs, captures=named)
    assert sp.load_confidence_capture(path, "a", record)["side"] == "a"
    with pytest.raises(ValueError, match="is not the side b capture"):
        sp.load_confidence_capture(path, "b", record)


# --- the contract with scripts/compare_captures.py -----------------------------------------


def _compare_captures() -> Any:
    try:
        import verbatim_bench.canonical  # noqa: F401  (compare_captures needs it)
    except ImportError:
        pytest.skip("verbatim_bench is not importable; the contract test needs bench/src")
    spec = importlib.util.spec_from_file_location(
        "compare_captures_for_places", ROOT / "scripts" / "compare_captures.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CC_REFS = {"r0": "i'm from the cutter", "r1": "no sir", "r2": "a dog ran home"}
CC_FIXED = {
    "r0": ("I'm from the cutter.", [("I'm", 0, 160, 0.9), ("from", 160, 320, 0.8),
                                    ("the", 320, 400, 0.9), ("cutter.", 400, 640, 0.7)]),
    "r1": ("No, sir.", [("No,", 0, 160, 0.9), ("sir.", 160, 320, 0.9)]),
    "r2": ("A dog ran hum.", [("A", 0, 80, 0.9), ("dog", 80, 160, 0.9), ("ran", 160, 240, 0.9),
                              ("hum.", 240, 320, 0.2)]),
}  # fmt: skip
#: The ragged side: r1 differs in a word, r0 only in the end of its last word.
CC_RAGGED = {
    **CC_FIXED,
    "r0": (CC_FIXED["r0"][0], [*CC_FIXED["r0"][1][:3], ("cutter.", 400, 720, 0.7)]),
    "r1": ("Now, sir.", [("Now,", 0, 160, 0.9), ("sir.", 160, 320, 0.9)]),
}


def cc_capture(
    cc: Any,
    path: Path,
    padding: str,
    concurrency: int,
    answers: dict,
    *,
    observed: bool = True,
    peak: bool = True,
    fake: bool = False,
    matmul: str = "high",
    confidence: str = "nemo-shipped",
    tweak: Callable[[dict], None] | None = None,
    digest: str | None = None,
) -> Any:
    """A capture in the shape probes/server_frozen_answers.py writes, saved and loaded; its
    client peaked at min(concurrency, recordings) in flight, and every recording's
    ``in_flight_s`` gives that peak (the comparator recounts it); with
    ``peak`` False its client records no observed peak in flight; with ``fake`` it is stamped
    as the probe stamps a capture taken through its test seams. ``matmul`` is declared, and is
    the installed default, as the probe stamps it: the server does not report it. The server
    ran with word confidence ``confidence``: "off" drops each word's confidence, as the wire
    then carries none. ``tweak`` edits the record before its digest is stamped; ``digest``
    stamps that value instead of the recordings' own."""
    record: dict[str, Any] = {"fake_pipeline": True} if fake else {}
    most = min(concurrency, len(answers))
    # Groups of ``most`` sessions in flight together, each group ending before the next starts.
    spans = [[k // most + (k % most) / (8 * most), k // most + 0.875] for k in range(len(answers))]
    record |= {
        "record": cc.RECORD_KIND,
        "success": True,
        "failures": [],
        "started": f"{path.stem}-started",
        "finished": f"{path.stem}-finished",
        "server": {
            "model": {"reported_by_server": "m"},
            "model_revision": {
                "observed_in_hf_cache": "rev",
                "hub_dir_source": "the server's environment (/proc/<pid>/environ), by "
                "huggingface_hub's rule",
            },
            "chunk_ms": {"reported_by_server": 160},
            "dtype": {"reported_by_server": "bfloat16"},
            "execution": {"reported_by_server": "eager"},
            "matmul_precision": {
                "reported_by_server": None,
                "declared": matmul,
                "installed_default": matmul,
            },
            "bucket": {"reported_by_server": 8},
            "padding": {
                "declared": padding,
                "observed_in_server_log": padding if observed else None,
            },
            "att_context": {"observed_in_readyz": [70, 1]},
            "word_confidence": {
                "reported_by_server": confidence,
                "decoder_step_confidence": confidence != "off",
                "declared": confidence,
            },
            "gpu": {"before": {"uuid": "GPU-test"}},
            # Contract C7: where the server process imported its code from.
            "code": {
                "reported_by_server": {
                    "verbatim_path": "checkout/src/verbatim",
                    "bench_path": "checkout/bench/src/verbatim_bench",
                }
            },
        },
        "client": {
            "concurrency": {"configured": concurrency}
            | ({"observed_peak_in_flight": most} if peak else {}),
            "verbatim_commit": "c",
            "probe_sha256": "p",
            "bench_client_sha256": "b",
            "tracked_files_modified": False,
            "code_outside_checkout": [],
        },
        "recordings": {
            rid: {
                "n": n,
                "text": text,
                "words": [list(w if confidence != "off" else w[:3]) for w in words],
                "pcm_sha256": f"audio-of-{rid}",
                "reference": CC_REFS[rid],
                "in_flight_s": spans[n],
            }
            for n, (rid, (text, words)) in enumerate(answers.items())
        },
    }
    if tweak is not None:
        tweak(record)
    record["finals_digest"] = cc.digest_of(record) if digest is None else digest
    path.write_text(json.dumps(record), encoding="utf-8")
    return cc.load_capture(path)


def places_from_captures(tmp_path: Path, repeat: str | None) -> tuple[Any, dict, Path]:
    """scripts/compare_captures.build_places on a fixed and a ragged capture, with a second
    fixed capture at another concurrency ("other"), the same one ("same"), a differing one
    ("differs"), another concurrency with its padding only declared ("declared"), another
    concurrency with no observed peak in flight recorded ("unrecorded"), or none; the record
    comes back through JSON, as --places-out writes it."""
    cc = _compare_captures()
    fixed_path = tmp_path / "fixed.json"
    fixed = cc_capture(cc, fixed_path, "fixed", 32, CC_FIXED)
    ragged = cc_capture(cc, tmp_path / "ragged.json", "ragged", 32, CC_RAGGED)
    second = None
    if repeat is not None:
        answers = CC_RAGGED if repeat == "differs" else CC_FIXED
        second = cc_capture(
            cc,
            tmp_path / "repeat.json",
            "fixed",
            32 if repeat == "same" else 1,
            answers,
            observed=repeat != "declared",
            peak=repeat != "unrecorded",
        )
    same = None if second is None else cc.identity(fixed.record, second.record)
    record = cc.build_places(fixed, ragged, repeat=second, fixed_vs_repeat=same)
    return cc, json.loads(json.dumps(record)), fixed_path


def test_a_compare_captures_places_record_is_scored_as_the_contract_says(tmp_path) -> None:
    cc, record, fixed_path = places_from_captures(tmp_path, "other")
    assert (record["source"], record["timing_unit"]) == (sp.CAPTURES, "ms")
    assert record["bucket"] == record["batch"] == 8
    given = sp.load_confidence_capture(fixed_path, "a", record)
    out = sp.derive(
        record, NO_WORDS, draws=20, confidence_at=("bfloat16", cc.ARM), confidence=given
    )
    run = out["runs"]["bfloat16"]
    assert run["repeat_verdict_accepted"] is True
    scored = run["arms"][cc.ARM]
    assert scored["served_side"] == "a"
    assert (scored["checked"], scored["text_divergent"], scored["timing_only_divergent"]) == (
        3,
        1,
        1,
    )
    assert scored["timing_only_shift_ms"]["largest_shift_counts"] == {"80": 1}
    assert scored["flag"]["places"] == 1
    assert scored["flag"]["a"]["recall"] is not None  # a captures record holds every recording
    c = scored["confidence"]
    assert (c["a"]["status"], c["a"]["k_words"], c["a"]["flagged_words"]) == (
        "scored",
        1,
        {"r2": [[3, 4]]},
    )
    assert c["b"]["status"] == "confidence unavailable"
    assert out["record_settings"]["timing_unit"] == "ms"
    # Every capture recorded a clean tree; the record says so and the output carries it.
    assert out["record_settings"]["tracked_files_modified"] is False


#: Every repeat_verdict build_places writes other than the accepted one, by the second capture
#: that gives it (see places_from_captures).
NEAR_MISS = {
    None: "not measured",
    "same": "identical at one concurrency (3) only",
    "differs": "NOT run-to-run identical",
    "declared": "identical at two concurrencies, but the padding is only declared",
    "unrecorded": "identical, but the observed peak in flight is not recorded in",
}


def test_the_near_misses_are_every_verdict_build_places_can_write() -> None:
    # NEAR_MISS claims to cover every verdict but the accepted one. build_places sets
    # repeat_verdict once per verdict, so a verdict added there without a case here is red.
    tree = ast.parse((ROOT / "scripts" / "compare_captures.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_places")
    assigned = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign | ast.AnnAssign)
        and any(
            isinstance(t, ast.Name) and t.id == "repeat_verdict"
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
    ]
    assert len(assigned) == len(NEAR_MISS) + 1


@pytest.mark.parametrize("repeat", list(NEAR_MISS))
def test_every_compare_captures_near_miss_verdict_withholds_the_scores(tmp_path, repeat) -> None:
    _, record, _ = places_from_captures(tmp_path, repeat)
    verdict = record["runs"]["bfloat16"]["repeat_verdict"]
    assert verdict.startswith(NEAR_MISS[repeat]) and verdict != sp.SAME_SHAPE_VERDICT
    run = sp.derive(record, NO_WORDS, draws=0)["runs"]["bfloat16"]
    assert run["repeat_verdict_accepted"] is False
    assert all(arm["flag"] is None for arm in run["arms"].values())


def test_a_places_record_built_from_a_stamped_capture_is_refused_and_so_is_the_capture(
    tmp_path,
) -> None:
    # The whole path: the capture probe stamps a capture its test seams took, build_places
    # carries the stamp into the places record, and the scorer refuses the record and, given as
    # --confidence, the capture itself.
    cc = _compare_captures()
    fixed_path = tmp_path / "fixed.json"
    fixed = cc_capture(cc, fixed_path, "fixed", 32, CC_FIXED, fake=True)
    ragged = cc_capture(cc, tmp_path / "ragged.json", "ragged", 32, CC_RAGGED)
    record = json.loads(
        json.dumps(cc.build_places(fixed, ragged, repeat=None, fixed_vs_repeat=None))
    )
    assert record["fake_pipeline"] is True
    with pytest.raises(ValueError, match=re.escape("record refused: fake_pipeline is True")):
        sp.derive(record, NO_WORDS, draws=0)
    del record["fake_pipeline"]
    with pytest.raises(ValueError, match=re.escape("has fake_pipeline True")):
        sp.load_confidence_capture(fixed_path, "a", record)


def test_a_compare_captures_records_matmul_is_labelled_declared_not_observed(tmp_path) -> None:
    # The server does not report its matmul precision: the capture declares it, the comparator
    # copies the declaration (it is not one of its OBSERVED settings), and the scorer says so.
    cc = _compare_captures()
    assert not {"matmul", "matmul_precision"} & set(cc.OBSERVED)
    fixed = cc_capture(cc, tmp_path / "fixed.json", "fixed", 32, CC_FIXED, matmul="highest")
    ragged = cc_capture(cc, tmp_path / "ragged.json", "ragged", 32, CC_RAGGED, matmul="highest")
    record = json.loads(
        json.dumps(cc.build_places(fixed, ragged, repeat=None, fixed_vs_repeat=None))
    )
    assert record["matmul_precision"] == "highest"
    diffs = sp.setting_differences(record, SERVING)
    line = "matmul_precision (declared, not observed): record 'highest', serving 'high'"
    assert [d for d in diffs if d.startswith("matmul_precision")] == [line]


def test_a_compare_captures_record_without_its_timing_unit_is_refused(tmp_path) -> None:
    _, record, _ = places_from_captures(tmp_path, "other")
    del record["timing_unit"]
    with pytest.raises(ValueError, match="must say timing_unit 'ms'; it says None"):
        sp.derive(record, NO_WORDS, draws=0)


# --- a case copied from the smoke record ---------------------------------------------------


def test_im_against_i_am_is_two_contraction_form_errors() -> None:
    # Copied from probe-output/step1/smoke-bf16-1120.json (sha256 e99215c3...c50c9ec), recording
    # 7902-96591-0000, ragged arm, side a. That record came from a probe build older than the
    # timing guard (verbatim_commit 6b7c38f, no timing_channel key); only this text is used.
    reference = "i am from the cutter lying off the coast"
    text = "I'm from the cutter lying off the coast."
    r = score((reference, text, text))
    assert r["corpus"]["word_errors_a"] == 2
    assert r["corpus"]["contraction_form_word_errors_a"] == 2


# --- contract C8: a capture taken with word confidence on, on places taken with it off ------

#: CC_FIXED with one word's start 1 ms later; and with one word changed.
CC_SHIFTED = {
    **CC_FIXED,
    "r2": (CC_FIXED["r2"][0], [*CC_FIXED["r2"][1][:3], ("hum.", 241, 320, 0.2)]),
}
CC_CHANGED = {
    **CC_FIXED,
    "r2": ("A dog ran home.", [*CC_FIXED["r2"][1][:3], ("home.", 240, 320, 0.2)]),
}


def off_places(tmp_path: Path, *, repeat: bool = True, **fixed: Any) -> tuple[Any, dict, Path]:
    """Places, as --places-out writes them, from captures taken with word confidence off: the
    fixed capture at concurrency 32 (side a, built with ``fixed``), the ragged one, and, with
    ``repeat``, a second fixed one at concurrency 1, so that the repeat verdict is accepted."""
    cc = _compare_captures()
    fixed_path = tmp_path / "fixed-off.json"
    first = cc_capture(cc, fixed_path, "fixed", 32, CC_FIXED, confidence="off", **fixed)
    ragged = cc_capture(cc, tmp_path / "ragged-off.json", "ragged", 32, CC_RAGGED, confidence="off")
    second = same = None
    if repeat:
        second = cc_capture(
            cc, tmp_path / "repeat-off.json", "fixed", 1, CC_FIXED, confidence="off"
        )
        same = cc.identity(first.record, second.record)
    record = cc.build_places(first, ragged, repeat=second, fixed_vs_repeat=same)
    return cc, json.loads(json.dumps(record)), fixed_path


def on_capture(
    cc: Any, tmp_path: Path, answers: dict = CC_FIXED, padding: str = "fixed", **kw: Any
) -> Path:
    """A fixed capture at concurrency 32 taken with word confidence on."""
    path = tmp_path / "fixed-on.json"
    cc_capture(cc, path, padding, 32, answers, **kw)
    return path


def answers_digest(cc: Any, answers: dict) -> str:
    recordings = {
        rid: {"n": n, "text": text, "words": [list(w[:3]) for w in words]}
        for n, (rid, (text, words)) in enumerate(answers.items())
    }
    return cc.digest_of({"recordings": recordings})


def test_an_on_capture_of_the_same_answers_scores_the_places_of_the_off_captures(
    tmp_path,
) -> None:
    cc, record, fixed_path = off_places(tmp_path)
    on = on_capture(cc, tmp_path)
    off_loaded, on_loaded = cc.load_capture(fixed_path), cc.load_capture(on)
    assert off_loaded.sha256 != on_loaded.sha256
    # The two differ in the word confidence, as configured and as observed, and in nothing else
    # the comparator reads: that is what C8 leaves out.
    s_off, s_on = cc.settings_of(off_loaded.record), cc.settings_of(on_loaded.record)
    assert {k for k in s_off if s_off[k] != s_on[k]} == {
        "word_confidence",
        "decoder_step_confidence",
    }
    given = sp.load_confidence_capture(on, "a", record)
    assert given["identity"]["verdict"] == "IDENTICAL IN TEXT AND TIMINGS"
    assert given["identity"]["against"] == {"path": str(fixed_path), "sha256": off_loaded.sha256}
    assert given["identity"]["finals_digest"] == off_loaded.record["finals_digest"]
    out = sp.derive(record, NO_WORDS, draws=0, confidence_at=("bfloat16", cc.ARM), confidence=given)
    assert out["runs"]["bfloat16"]["repeat_verdict_accepted"] is True
    c = out["runs"]["bfloat16"]["arms"][cc.ARM]["confidence"]["a"]
    assert (c["status"], c["k_words"], c["flagged_words"]) == ("scored", 1, {"r2": [[3, 4]]})
    assert c["identity"] == given["identity"]
    # The side's own capture is taken byte for byte, and says so; it carries no confidence.
    same = sp.load_confidence_capture(fixed_path, "a", record)
    assert same["identity"] == {"rule": "the same bytes as the places record's a_fixed capture"}
    assert same["unavailable"].startswith("no word entry")


#: One defect each, against the accepted case above: (the off side-a capture's arguments, the
#: on capture's arguments, what the refusal says). Built by ``c8_case``.
C8_DEFECTS: dict[str, tuple[Callable[[Any], dict], Callable[[Any], dict], str]] = {
    "a word 1 ms later": (lambda cc: {}, lambda cc: {"answers": CC_SHIFTED}, "digests differ"),
    "a word changed": (lambda cc: {}, lambda cc: {"answers": CC_CHANGED}, "digests differ"),
    "the on digest is not its own": (
        lambda cc: {},
        lambda cc: {"answers": CC_SHIFTED, "digest": answers_digest(cc, CC_FIXED)},
        "{on}: the stored finals_digest is not the digest of its own recordings",
    ),
    "the off digest is not its own": (
        lambda cc: {"digest": answers_digest(cc, CC_SHIFTED)},
        lambda cc: {"answers": CC_SHIFTED},
        "{off}: the stored finals_digest is not the digest of its own recordings",
    ),
    "another order": (
        lambda cc: {},
        lambda cc: {"tweak": swap_r0_r1},
        "the captures do not hold the same recordings in the same order",
    ),
    "other audio": (
        lambda cc: {},
        lambda cc: {"tweak": lambda r: r["recordings"]["r1"].update(pcm_sha256="other")},
        "1 recording(s) were sent different audio; first r1",
    ),
    "another chunk": (
        lambda cc: {},
        lambda cc: {"tweak": lambda r: r["server"]["chunk_ms"].update(reported_by_server=1120)},
        "the captures differ in chunk_ms: 160 and 1120",
    ),
    "other server code": (
        lambda cc: {},
        lambda cc: {"tweak": other_server_code},
        "the servers ran different code: server.code.verbatim_path",
    ),
    "another padding": (
        lambda cc: {},
        lambda cc: {"padding": "ragged"},
        "its padding is 'ragged' and the side a capture's is 'fixed'",
    ),
}


def swap_r0_r1(record: dict) -> None:
    r = record["recordings"]
    r["r0"]["n"], r["r1"]["n"] = r["r1"]["n"], r["r0"]["n"]


def other_server_code(record: dict) -> None:
    record["server"]["code"]["reported_by_server"]["verbatim_path"] = "elsewhere/src/verbatim"


@pytest.mark.parametrize("defect", list(C8_DEFECTS))
def test_an_on_capture_that_is_not_the_same_answers_is_refused(tmp_path, defect) -> None:
    off_kw, on_kw, says = C8_DEFECTS[defect]
    cc = _compare_captures()
    kw = off_kw(cc)
    cc, record, fixed_path = off_places(tmp_path, repeat="digest" not in kw, **kw)
    on = on_capture(cc, tmp_path, **on_kw(cc))
    with pytest.raises(sp.Refused) as info:
        sp.load_confidence_capture(on, "a", record)
    message = str(info.value)
    assert message.startswith(f"confidence refused: {on} (sha256 ")
    assert "it is not shown to be the same answers as that capture (contract C8): " in message
    assert says.format(on=on, off=fixed_path) in message


def test_the_side_capture_is_read_where_confidence_against_says_and_must_be_the_named_one(
    tmp_path,
) -> None:
    cc, record, fixed_path = off_places(tmp_path)
    on = on_capture(cc, tmp_path)
    moved = fixed_path.rename(tmp_path / "moved.json")
    with pytest.raises(sp.Refused, match=re.escape(f"cannot be read at {fixed_path}")):
        sp.load_confidence_capture(on, "a", record)
    given = sp.load_confidence_capture(on, "a", record, against=moved)
    assert given["identity"]["against"]["path"] == str(moved)
    other = tmp_path / "ragged-off.json"
    with pytest.raises(sp.Refused) as info:
        sp.load_confidence_capture(on, "a", record, against=other)
    assert str(info.value).startswith(f"confidence refused: {other} (sha256 ")
    assert "is not the side a capture the places record names" in str(info.value)
    del record["captures"]["a_fixed"]["path"]
    with pytest.raises(sp.Refused, match="the record gives no path to that capture"):
        sp.load_confidence_capture(on, "a", record)


def test_an_on_capture_the_comparator_cannot_read_is_refused_not_raised(tmp_path) -> None:
    cc, record, _ = off_places(tmp_path)
    on = on_capture(
        cc, tmp_path, tweak=lambda r: r["recordings"]["r0"].pop("text"), digest="0" * 64
    )
    with pytest.raises(sp.Refused, match=re.escape("the two cannot be compared as captures")):
        sp.load_confidence_capture(on, "a", record)


def test_c8_is_refused_when_the_comparator_cannot_be_imported(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, sp._COMPARATOR, raising=False)
    monkeypatch.setitem(sys.modules, "verbatim_bench.canonical", None)
    with pytest.raises(sp.Refused, match="cannot be imported here"):
        sp.comparator()
    assert sp._COMPARATOR not in sys.modules


def test_main_scores_off_places_with_an_on_capture_and_refuses_a_shifted_one(
    tmp_path, capsys
) -> None:
    cc, record, fixed_path = off_places(tmp_path)
    places = tmp_path / "places.json"
    places.write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "words.txt").write_text("the\n", encoding="utf-8")
    on = on_capture(cc, tmp_path)
    out = derived(tmp_path, places, "--confidence", str(on))
    c = out["runs"]["bfloat16"]["arms"][cc.ARM]["confidence"]["a"]
    assert (c["status"], c["identity"]["verdict"]) == ("scored", "IDENTICAL IN TEXT AND TIMINGS")
    moved = fixed_path.rename(tmp_path / "moved.json")
    against = ["--confidence", str(on), "--confidence-against", str(moved)]
    assert scorer(places, *against, "--out", str(tmp_path / "moved-derived.json")) == 0
    shifted = tmp_path / "shifted-on.json"
    cc_capture(cc, shifted, "fixed", 32, CC_SHIFTED)
    against = ["--confidence", str(shifted), "--confidence-against", str(moved)]
    assert scorer(places, *against, "--out", str(tmp_path / "shifted-derived.json")) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"step1_places: confidence refused: {shifted} (sha256 ")
    assert "digests differ" in err and err.count("\n") == 1
    assert not (tmp_path / "shifted-derived.json").exists()
