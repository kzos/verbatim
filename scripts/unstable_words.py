#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Which words change with the batch, and was either version right?

Reads the stock-pipeline divergence records (a recording transcribed alone, and in slot 0 of
a batch of 32) and writes one derived record with every number the white paper quotes about
the words that changed. Nothing here touches a GPU; it re-derives from the JSON the probes
wrote, so anyone can check it.

    python3 scripts/unstable_words.py > rows/exploratory/unstable-words-2026-09-24.json

Rules, stated because every number below depends on them:

* Tokens are the lowercased transcript split on whitespace. Neither side has punctuation.
* A **place** is one non-equal block of `difflib.SequenceMatcher(alone, in_batch)` with
  `autojunk=False`. A place can be one word for one word, several words, or a word present in
  one version only. Places are what earlier drafts called "words".
* A side of a place is **right** when, aligned to the reference by the same matcher, every
  word it has there is an exact match and no reference word is missing between them. A side
  that is empty at a place is right when the other side's words there are all insertions
  against the reference and no reference word is missing at that point. Both sides can be
  wrong; both cannot be right, because they differ.
* **Word error rate** is word-level Levenshtein distance over the reference word count.
* A word is **out of dictionary** when its lowercased form is not in the word list given by
  `--wordlist` (default `/usr/share/dict/american-english`, the `wamerican` package). The list
  is American English, so British spellings count as absent; its SHA-256 is recorded.
  "Reference words at a place" are the reference words either side is aligned to there
  (a substitution block maps to its whole reference block), plus, for a side that is empty
  there, the reference words it lacks at that point.
* A place **involves a negation or a number** when the negation words, or the number words, in
  its two sides differ (word sets below).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "rows" / "exploratory"

B300 = "stock-divergence-b300-2026-09-11.json"
A6000_BF16 = "stock-divergence-a6000-bf16-2026-09-10.json"
A6000_FP32 = "stock-divergence-a6000-fp32-2026-09-10.json"

NEGATIONS = frozenset(
    [
        "not",
        "no",
        "nor",
        "never",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        "nought",
        "naught",
        "nowt",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "ain't",
        "warn't",
        "shan't",
        "shouldn't",
        "couldn't",
        "mustn't",
        "hasn't",
        "haven't",
        "hadn't",
        "needn't",
    ]
)
NUMBERS = frozenset(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "once",
        "twice",
        "thrice",
        "half",
    ]
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class Aligned:
    """One hypothesis aligned to the reference."""

    def __init__(self, hyp: list[str], ref: list[str]) -> None:
        self.correct: set[int] = set()
        self.to_ref: dict[int, set[int]] = {i: set() for i in range(len(hyp))}
        #: hypothesis position -> reference words missing immediately before it
        self.missing_before: dict[int, set[int]] = {}
        ops = difflib.SequenceMatcher(None, hyp, ref, autojunk=False).get_opcodes()
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                for k in range(i2 - i1):
                    self.correct.add(i1 + k)
                    self.to_ref[i1 + k] = {j1 + k}
            elif tag == "replace":
                for i in range(i1, i2):
                    self.to_ref[i] = set(range(j1, j2))
            elif tag == "insert":  # reference words the hypothesis lacks
                self.missing_before.setdefault(i1, set()).update(range(j1, j2))

    def right(self, i1: int, i2: int) -> bool:
        if not all(i in self.correct for i in range(i1, i2)):
            return False
        return not any(i in self.missing_before for i in range(i1 + 1, i2))

    def refs(self, i1: int, i2: int) -> set[int]:
        """Reference words this side is aligned to at a place; for an empty side, the
        reference words it lacks at that point."""
        if i1 == i2:
            return set(self.missing_before.get(i1, set()))
        out: set[int] = set()
        for i in range(i1, i2):
            out |= self.to_ref[i]
        for i in range(i1 + 1, i2):
            out |= self.missing_before.get(i, set())
        return out


def sign_test(a: int, b: int) -> float:
    """Two-sided exact binomial test of a against b at p = 1/2."""
    n, k = a + b, min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def analyse(divergences: list[dict[str, Any]], words: frozenset[str]) -> dict[str, Any]:
    places = alone_right = batch_right = neither = both = 0
    one_for_one = multi_word = one_side_empty = 0
    ref_total = ref_oov = at_place = at_place_oov = 0
    errors_alone = errors_batch = 0
    meaning: list[dict[str, str]] = []
    for d in divergences:
        ref, alone, batch = (d[k].lower().split() for k in ("reference", "alone", "in_batch"))
        ref_total += len(ref)
        ref_oov += sum(w not in words for w in ref)
        errors_alone += edit_distance(alone, ref)
        errors_batch += edit_distance(batch, ref)
        a, b = Aligned(alone, ref), Aligned(batch, ref)
        touched: set[int] = set()
        ops = difflib.SequenceMatcher(None, alone, batch, autojunk=False).get_opcodes()
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            places += 1
            if i2 - i1 == 1 and j2 - j1 == 1:
                one_for_one += 1
            elif i1 == i2 or j1 == j2:
                one_side_empty += 1
            else:
                multi_word += 1
            if i1 == i2:
                ra = all(j not in b.correct for j in range(j1, j2)) and i1 not in a.missing_before
            else:
                ra = a.right(i1, i2)
            if j1 == j2:
                rb = all(i not in a.correct for i in range(i1, i2)) and j1 not in b.missing_before
            else:
                rb = b.right(j1, j2)
            alone_right += ra and not rb
            batch_right += rb and not ra
            neither += not ra and not rb
            both += ra and rb
            touched |= a.refs(i1, i2) | b.refs(j1, j2)
            sa, sb = set(alone[i1:i2]), set(batch[j1:j2])
            if (sa & NEGATIONS) != (sb & NEGATIONS) or (sa & NUMBERS) != (sb & NUMBERS):
                meaning.append(
                    {
                        "librispeech_id": d["librispeech_id"],
                        "alone": " ".join(alone[i1:i2]),
                        "in_batch": " ".join(batch[j1:j2]),
                        "reference": " ".join(
                            ref[j] for j in sorted(a.refs(i1, i2) | b.refs(j1, j2))
                        ),
                    }
                )
        at_place += len(touched)
        at_place_oov += sum(ref[j] not in words for j in touched)
    return {
        "recordings_changed": len(divergences),
        "places": places,
        "place_shapes": {
            "one_for_one": one_for_one,
            "multi_word": multi_word,
            "one_side_empty": one_side_empty,
        },
        "alone_right": alone_right,
        "batch_right": batch_right,
        "neither_right": neither,
        "both_right": both,
        "alone_vs_batch_sign_test_p": round(sign_test(alone_right, batch_right), 4),
        "reference_words": ref_total,
        "word_errors_alone": errors_alone,
        "word_errors_in_batch": errors_batch,
        "wer_alone": round(errors_alone / ref_total, 5),
        "wer_in_batch": round(errors_batch / ref_total, 5),
        "net_extra_errors_in_batch": errors_batch - errors_alone,
        "out_of_dictionary": {
            "reference_words": ref_total,
            "reference_words_absent": ref_oov,
            "rate_all_reference_words": round(ref_oov / ref_total, 5),
            "reference_words_at_places": at_place,
            "reference_words_at_places_absent": at_place_oov,
            "rate_at_places": round(at_place_oov / at_place, 5) if at_place else None,
        },
        "negation_or_number_places": meaning,
    }


def ids(divergences: list[dict[str, Any]]) -> set[str]:
    return {d["librispeech_id"] for d in divergences}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wordlist", type=Path, default=Path("/usr/share/dict/american-english"))
    args = parser.parse_args()
    words = frozenset(
        w.strip().lower()
        for w in args.wordlist.read_text(encoding="utf-8").splitlines()
        if w.strip()
    )

    b300 = json.loads((ROWS / B300).read_text(encoding="utf-8"))
    a_bf16 = json.loads((ROWS / A6000_BF16).read_text(encoding="utf-8"))
    a_fp32 = json.loads((ROWS / A6000_FP32).read_text(encoding="utf-8"))

    b_eq = b300["runs"]["bfloat16"]["equalised"]["divergences"]
    b_rg = b300["runs"]["bfloat16"]["ragged"]["divergences"]
    a_eq = a_bf16["arms"]["controlled"]["divergences"]
    a_rg = a_bf16["arms"]["uncontrolled"]["divergences"]

    out = {
        "question": (
            "At the places where a recording's transcript alone differs from its transcript"
            " in a batch of 32, was either version right?"
        ),
        "not_a_row": (
            "Derived from stock-pipeline probes at att_context [70,13] (1,120 ms chunks) and"
            " batch 32, not the server's [70,1] / bucket-128 setting; the server's own output"
            " at these places was not scored."
        ),
        "inputs": {name: sha256(ROWS / name) for name in (B300, A6000_BF16, A6000_FP32)},
        "wordlist": {
            "path": str(args.wordlist),
            "sha256": sha256(args.wordlist),
            "entries": len(words),
        },
        "rules": "see the module docstring of scripts/unstable_words.py",
        "recordings_checked": b300["runs"]["bfloat16"]["equalised"]["checked"],
        "changed_recordings": {
            "b300_bf16_equalised": len(b_eq),
            "b300_bf16_ragged": len(b_rg),
            "b300_fp32_equalised": len(b300["runs"]["float32"]["equalised"]["divergences"]),
            "b300_fp32_ragged": len(b300["runs"]["float32"]["ragged"]["divergences"]),
            "a6000_bf16_equalised": len(a_eq),
            "a6000_bf16_varying": len(a_rg),
            "a6000_fp32_equalised": len(a_fp32["arms"]["controlled"]["divergences"]),
            "a6000_fp32_varying": len(a_fp32["arms"]["uncontrolled"]["divergences"]),
        },
        "unstable_in_both_arms": {
            "b300_bf16": len(ids(b_eq) & ids(b_rg)),
            "b300_bf16_union": len(ids(b_eq) | ids(b_rg)),
            "a6000_bf16": len(ids(a_eq) & ids(a_rg)),
            "a6000_bf16_union": len(ids(a_eq) | ids(a_rg)),
        },
        "arms": {
            "b300_bf16_equalised": analyse(b_eq, words),
            "b300_bf16_ragged": analyse(b_rg, words),
            "a6000_bf16_equalised": analyse(a_eq, words),
            "a6000_bf16_varying": analyse(a_rg, words),
        },
    }
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
