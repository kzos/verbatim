#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Step 1: where do two transcripts of a recording differ, was either right, and does that flag
find errors better than the model's own confidence at the same number of flagged words?

Reads one record and writes one derived JSON. Nothing here touches a GPU.

    python3 scripts/step1_places.py RECORD.json [--out DERIVED.json] [--wordlist P]
        [--confidence CAPTURE.json [--confidence-at DTYPE/ARM/SIDE]
            [--confidence-against SIDE_CAPTURE.json]]
        [--draws N] [--seed S] [--serving-bucket N] [--serving-execution MODE]
        [--serving-decoder-graphs on|off] ...

A refused input is not scored: the reason is printed as one line on stderr and the exit status
is 1. ``--out`` is written new: it is refused when it, or its temporary file ``OUT.tmp``, exists
or when it is one of the inputs, and the file is linked into place so that a file appearing there
meanwhile is not written over either. Without ``--out`` the derived JSON goes to stdout.

Accepted input
==============

Two record shapes, told apart by the top-level ``source`` key. Both use the nesting that
``probes/stock_divergence.py`` writes (``runs[dtype].arms[arm]``), so the divergences handed to
``scripts/unstable_words.py``'s ``analyse`` have one shape whichever record they came from. A
record with ``"fake_pipeline"`` present and not ``false`` is refused: a test double wrote it (the
stock probe's CPU fake) or went into it. ``probes/server_frozen_answers.py`` stamps a capture
taken through its test seams the same way, ``scripts/compare_captures.py``'s ``build_places``
carries the stamp into a places record built from such a capture, and a ``--confidence``
capture so stamped is refused too.

1. A stock probe record: no ``source`` key; what ``probes/stock_divergence.py`` writes::

     {"model", "model_revision", "chunk_ms", "batch", ...: provenance, copied to the output
          and compared with the serving setting;
      "att_context_size", "matmul_precision": what the probe asked the pipeline for, copied;
          compared, labelled "requested, not observed" (``att_context_size`` only when the
          record has no reading of it, below);
      "att_context_size_observed": optional, the attention context read off the built encoder
          (the probe refuses to run when it is not the one asked for); where the record or a
          run has it, it is compared, per run, instead of the request;
      "use_cuda_graphs": optional bool, whether the pipeline spec asked for CUDA graphs in the
          ENCODER step (``NeMoPipelineSpec.use_cuda_graphs``). Compared with the serving
          ``execution`` ("graph path" or "eager", how the server's encoder step runs) and
          labelled "requested, not observed";
      "decoder_graphs_observed": optional bool, whether the built RNNT DECODER captures CUDA
          graphs (its ``cuda_graphs_mode``), read from the built pipeline. Compared with the
          serving decoder graphs (the server's ``--decoder-graphs``). A different setting from
          ``execution``: neither stands in for the other;
      "decoder_graphs_mode_observed": optional, copied;
      "word_confidence": optional str, the mode asked for; compared, labelled "requested, not
          observed";
      "word_confidence_observed": optional {"mode_requested", "decoder_step_confidence",
          "nonzero_conf_words_on_guard_recording"}; its ``decoder_step_confidence`` (read from
          the built decoder) is compared with the serving word confidence (on unless "off");
      "timing_unit": absent (or "s"): timings are seconds;
      "references": {rid: reference text},
      "runs": {dtype: {"repeat_verdict": str,
                       "repeat": {"alone_identical", "batch_identical", "checked"},
                       "positive_control": str, absent until the run finishes,
                       "decoder_graphs_observed", "word_confidence_observed",
                       "att_context_size_observed": optional, this dtype's readings, used in
                           preference to the top-level ones (a top-level reading used for a
                           run is labelled as top-level),
                       "arms": {"ragged" | "equalised" | "fixed": ARM}}}}

2. A server-captures places record: what ``scripts/compare_captures.py --places-out`` writes
   (its ``build_places``) from a fixed-padding and a ragged-padding capture of the same
   recordings by ``probes/server_frozen_answers.py``::

     {"source": "scripts/compare_captures.py",
      "model", "model_revision", "chunk_ms",
      "matmul_precision": what the capture DECLARED (``--matmul``), checked against the
          installed source's default and nothing more: the server does not report it and the
          comparator does not read it as observed. Compared, labelled "declared, not observed";
      "att_context_size": what the server's ``/readyz`` observed on its built encoder;
      "execution": "eager" or "graph path", the encoder step, as ``/readyz`` reports it;
      "decoder_graphs": bool, the decoder's CUDA graphs, as the capture stamps it;
      "word_confidence": the mode, as the capture stamps it;
      "bucket" and "batch": the server's bucket; refused when both are present and differ;
      "captures": {"a_fixed": {"path", "sha256", ...}, "b_ragged": {"path", "sha256", ...}};
      "timing_unit": required, exactly "ms"; any other value, or none, is refused;
      "references": {rid: reference text},
      "runs": {dtype: {
          "repeat_verdict": scores are given only when this is exactly "run-to-run identical
              in the same shape", which the writer may say only after two fixed captures at
              the same bucket and different concurrency gave identical finals digests;
          "repeat": {"recordings", "identical", "digests_match"}: those two captures compared;
          "positive_control": starts with "present" when the ragged capture differs somewhere;
          "arms": {"server_fixed_vs_ragged": ARM}}}}

   Side ``a`` is the fixed capture, the answer served; side ``b`` is the ragged capture.

ARM, in both shapes::

     {"checked": int, "text_divergent": int, "timing_only_divergent": int,
      "transcripts": {rid: {"a": text} when the two texts are equal,
                           {"a": text, "b": text} when they differ},
      "divergences": [{"n": int, "librispeech_id": rid, "text_differs": bool,
                       "a": text, "b": text,
                       "a_timings": [[word, start, end, ...], ...], "b_timings": [...]}],
      "every_recording": optional EVERY, and with it
      "confidence_only_divergent": int}

   A timing entry may carry more than three elements (the stock probe appends the segment's
   confidence); only the first three are compared or measured.

EVERY (what ``probes/stock_divergence.py`` adds under each arm)::

     {rid: {"a_text": str, "b_text": str,
            "a_words": [[word, start_s, end_s, conf], ...], "b_words": [...]}}

   for every rid in ``transcripts`` and no other. Checked before use (``check_every_recording``):
   the texts are the transcripts'; every word entry has four elements; a recording differs in
   text or in (word, start, end) exactly when it is one of the divergences, and then its words'
   (word, start, end) are that divergence's timings'; ``confidence_only_divergent`` is the number
   of recordings that agree in text and in (word, start, end) and not in conf. Those are not
   text or timing divergences. An arm with ``confidence_only_divergent`` and no
   ``every_recording`` is refused: nothing could check the count. An arm with neither reads
   ``confidence_only_divergent`` null.

Where per-word confidence comes from
====================================

* A stock record: ``every_recording``, for both sides. An arm without it gives, for each side,
  "confidence unavailable", and its recall is withheld (below). The confidences that the
  divergences' timings carry cover divergent recordings only and are not used; the output says
  so. ``--confidence`` is refused for a stock record.
* A server-captures record: ``--confidence CAPTURE``, a ``probes/server_frozen_answers.py``
  capture whose ``recordings[rid]["words"]`` entries carry a fourth element, the word
  confidence, when the wire carried one. It is attached at ``--confidence-at DTYPE/ARM/SIDE``
  (default ``DTYPE/server_fixed_vs_ragged/a``, the served side, when the record has one run).
  When the places record names its captures, the file must be that side's capture byte for byte
  (same SHA-256), or the same answers as it by contract C8, judged by
  ``scripts/compare_captures.py``'s ``confidence_equivalence`` with that capture as the one
  taken with word confidence off (``same_answers``): each capture's finals digest, recomputed,
  is its stored one (text and (word, start_ms, end_ms); a confidence is not in it); the two
  digests are equal; the two hold the same recordings in the same order with the same
  ``pcm_sha256``; and every setting the comparator compares but the word confidence is equal.
  The scorer also asks that its padding be that capture's: the confidence is attached to one
  side, and the side is a padding. That is how a capture taken with word confidence on scores
  the places of captures taken with it off. The side's capture is read from the path the places
  record gives, or from ``--confidence-against``, and must have the SHA-256 the record gives.
  The output's ``identity`` says which rule accepted the file. A capture in which no word entry
  has a fourth element gives "confidence unavailable"; one in which some do and some do not, or
  in which an entry has neither three nor four elements, is refused when it is loaded
  (``load_confidence_capture``). The confidence is the fourth element. The other side, and every
  other run and arm, reads "confidence unavailable".

Normalised, a side's confidence words must spell that side's normalised transcript, recording
by recording, or the input is refused. Nothing is invented.

Rules, stated because every number depends on them
===================================================

* **Normalisation.** The model writes capitals and punctuation; LibriSpeech references are
  lowercase with apostrophes only. Every transcript, reference and confidence word goes through
  ``normalise``: the typographic apostrophe U+2019 becomes ``'``, the text is lowercased, and
  every run of characters that is not a word character, an apostrophe or a space becomes one
  space. That is the probe's own ``words()`` rule except for U+2019, which the probe does not
  map (it splits "I\\u2019m" into "i m"). Hyphens split words. Contractions are not expanded.
* **Places, right/wrong by analyse, word error counts, out-of-dictionary figures,
  negation/number places** come from ``scripts/unstable_words.py``'s ``analyse`` unchanged,
  applied to the normalised text of every recording whose two normalised transcripts differ.
  ``analyse`` names the sides ``alone`` and ``in_batch``; here they mean side ``a`` and side
  ``b`` (see ``sides``).
* A recording is **punctuation-or-case-only** when its raw transcripts differ but the normalised
  ones are equal. It gives no place. When no recording of an arm differs after normalisation,
  ``places`` is null (``analyse`` divides by the changed recordings' reference words).
* **Contractions.** ``expansions`` maps a word to its possible expansions: the irregular forms
  (``IRREGULAR``) and the suffix rules ``n't`` -> ``not``, ``'m`` -> ``am``, ``'re`` -> ``are``,
  ``'ve`` -> ``have``, ``'ll`` -> ``will``/``shall``, ``'d`` -> ``would``/``had``/``did``,
  ``'s`` -> ``is``/``has``. The **contraction-tolerant distance** is word Levenshtein distance in
  which a word also matches one of its expansions at no cost. A place **differs only by a
  contraction** when its sides are at contraction-tolerant distance 0. **Contraction-form word
  errors** are a side's word errors minus its contraction-tolerant distance to the reference.
* **Whole-corpus WER** of a side: word Levenshtein distance to the reference summed over every
  checked recording, over the reference words. Side ``b`` of a recording without a ``b`` is ``a``.
* A **flag** marks spans of one side's words. The **place flag** on side ``s`` marks, at each
  place, that side's words there (an empty span where the side has none). The **confidence
  flag** on side ``s`` marks the K lowest-confidence words of side ``s`` over every checked
  recording, K being the words the place flag marks on that same side ``s``; ties are broken by
  a seeded shuffle and reported, and every other way of breaking them is scored too
  (``tie_range``: each figure's lowest and highest over every choice of the tied words at the
  cutoff, without nulls; not enumerated past ``TIE_CHOICES_MAX`` choices, and then it says so);
  consecutive flagged words of a recording form one span. The
  place flag can also mark an empty span (a gap); the confidence flag cannot. When the place
  flag marks 0 words on a side, that side has nothing to compare.
* **Not right**, for every span of every flag, the place flag included (``span_not_right``): a
  non-empty span is not right when ``not Aligned(side, reference).right(i1, i2)``, the rule
  ``analyse`` applies to a non-empty side of a place; an empty span is not right when a
  reference word is missing at that point. **Precision** of a flag on side ``s``: its spans that
  are not right, over its spans. ``analyse``'s own count for the places, the places minus the
  side's ``right`` count minus ``both_right``, is kept as ``not_right_by_analyse``. The two
  differ only at empty spans, where ``analyse`` also calls a gap not right when the other
  side's words there are correct, so ``not_right_by_analyse >= not_right``.
* **both_right** is the number of places where neither side's span is not right by the same
  rule (``both_right_by_analyse`` is ``analyse``'s). When it is 0, at least one side is not right
  at every place, so ``precision_a + precision_b >= 1`` by construction.
* **Word errors removable** by a flag: the side's word errors minus the word edit distance to the
  reference once each flagged span is replaced by a wildcard that matches any run of reference
  words (including none) at no cost. It is what a perfect correction of every flagged span would
  remove, and so counts an error next to a span whenever the best alignment lets the span absorb
  it: an upper bound. **Recall**: that over all the side's word errors across the arm. For a
  stock record, recall (and each null's ``recall_mean``) is given only when the arm carries
  ``every_recording``; without it recall is withheld and the output says why.
* **Nulls.** For each flag and side, ``--draws`` seeded draws of each of two nulls, scored with
  the same wildcard and not-right rules: (a) **same recordings**: each flagged recording keeps
  its span lengths, placed at uniformly random non-overlapping positions on the same side of the
  same recording; (b) **random recordings**: each flagged recording's span lengths are placed the
  same way in a recording drawn uniformly, with replacement, from the checked recordings with
  enough words to hold them. p-values are one-sided, ``(1 + draws >= observed) / (draws + 1)``.
  ``observed_over_null_mean`` is the flag's removable errors over the null's mean.
  ``reference_words_at_places`` is descriptive only: the wildcard can absorb errors at reference
  words no place touches, so it is not a baseline for recall; the nulls are.
* **Timing shift** of a timing-only divergence (raw text equal, timings differ): the two lists
  are paired index by index when they have the same length and the same word texts; only a
  timing entry's first three elements (word, start, end) are read; each time is
  rounded to whole milliseconds; the recording's shift is the largest absolute start or end
  difference over its pairs. Unpairable recordings are listed, not guessed. Recordings whose
  text differs are not measured. Quantiles are nearest-rank.
* **Out of dictionary**: the word list from ``--wordlist``, lowercased; its SHA-256 is recorded.
* **Controls.** Scores are withheld unless the run's ``repeat_verdict`` is exactly
  "run-to-run identical in the same shape" AND the run's own ``repeat`` counts show it: for a
  stock run, ``checked`` > 0 and ``alone_identical`` = ``batch_identical`` = ``checked``; for a
  server-captures run, ``recordings`` > 0, ``identical`` = ``recordings`` and ``digests_match``
  exactly ``true``. Every count must be a whole number: a bool or a float is not one, although
  ``True == 1`` and ``1.0 == 1``. A ``repeat`` that is not a mapping, or is absent, shows
  nothing, so the scores are withheld then too. An arm with no divergence is uninterpretable
  unless the run's ``positive_control`` starts with "present" (its first seven characters,
  exactly; "NOT present" does not) AND a two-shape arm of the run (``ragged`` or
  ``equalised``; for a server-captures record, its one arm) has a divergence in the record.
  That control is a shape change, so for the stock ``fixed`` arm, which varies only the
  neighbours' content at one shape, it shows the harness sees a shape change, not a content
  change. A server-captures arm with no divergence is its own control, so it is always
  uninterpretable.
* **Serving setting.** Every way the record's setting differs from the one given on the
  command line is listed (``setting_differences``). For a stock record, a setting the probe
  only asked for is labelled "requested, not observed": ``use_cuda_graphs``,
  ``word_confidence``, ``matmul_precision``, and ``att_context_size`` when the record carries
  no ``att_context_size_observed``. A reading off the built pipeline
  (``decoder_graphs_observed``, ``word_confidence_observed``'s ``decoder_step_confidence``,
  ``att_context_size_observed``) is compared per run and labelled "observed in the DTYPE run",
  or "observed, the record's top-level reading" when the run has none of its own.
  ``chunk_ms``, the run's dtype and ``batch`` are compared as the record states them,
  unlabelled. A server-captures record's settings are the captures' stamps (see its shape
  above) and are compared unlabelled, but for ``matmul_precision``: the server does not report
  it, so the record holds only what the capture declared, and it is labelled "declared, not
  observed". A setting not given reads "<placeholder: not given>".

Every count is checked against the record before it is used (``check_arm``,
``check_every_recording``); a record whose counters disagree with its own transcripts is refused,
not scored.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("unstable_words", _HERE / "unstable_words.py")
assert _spec is not None and _spec.loader is not None
uw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uw)

DEFAULT_WORDLIST = Path("/usr/share/dict/american-english")
SAME_SHAPE_VERDICT = "run-to-run identical in the same shape"
CAPTURES = "scripts/compare_captures.py"
CAPTURES_ARM = "server_fixed_vs_ragged"
TIMING_UNITS = {"s": 1000.0, "ms": 1.0}
#: The key a writer adds when a test double stood in for the model, the server, the host or the
#: corpus, and what it means. Present with any value but ``false``, the record (or the
#: ``--confidence`` capture) is refused. ``probes/stock_divergence.py`` and
#: ``probes/server_frozen_answers.py`` write it, and ``scripts/compare_captures.py``'s
#: ``build_places`` carries a capture's into the places record.
FAKE_MARKERS = {
    "fake_pipeline": "a test double, not the model on a GPU or a running server, produced it",
}
DEFAULT_DRAWS = 1000
#: The most ways of breaking a tie at the confidence cutoff ``tie_range`` scores one by one.
TIE_CHOICES_MAX = 1000
#: The figures ``tie_range`` gives the lowest and highest of.
TIE_FIGURES = ("spans", "not_right", "precision", "word_errors_removable", "recall")
DEFAULT_SEED = 20260924
#: The places record's ``captures`` key of each side's capture.
CAPTURE_OF_SIDE = {"a": "a_fixed", "b": "b_ragged"}
EXIT_REFUSED = 1


class Refused(ValueError):
    """The input is refused, not scored; ``main`` prints the reason and exits ``EXIT_REFUSED``."""


_APOSTROPHE = str.maketrans({"\u2019": "'"})
_NOT_WORD = re.compile(r"[^\w' ]+")


def normalise(text: str) -> str:
    """Lowercase; punctuation except apostrophes becomes a space; whitespace collapsed."""
    return " ".join(_NOT_WORD.sub(" ", text.translate(_APOSTROPHE).lower()).split())


IRREGULAR: dict[str, tuple[tuple[str, ...], ...]] = {
    "won't": (("will", "not"),),
    "can't": (("can", "not"), ("cannot",)),
    "shan't": (("shall", "not"),),
    "ain't": (("am", "not"), ("is", "not"), ("are", "not"), ("has", "not"), ("have", "not")),
    "let's": (("let", "us"),),
    "'tis": (("it", "is"),),
    "'twas": (("it", "was"),),
    "'em": (("them",),),
    "o'er": (("over",),),
    "e'er": (("ever",),),
    "ne'er": (("never",),),
}
_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("n't", ("not",)),
    ("'m", ("am",)),
    ("'re", ("are",)),
    ("'ve", ("have",)),
    ("'ll", ("will", "shall")),
    ("'d", ("would", "had", "did")),
    ("'s", ("is", "has")),
)


def expansions(word: str) -> tuple[tuple[str, ...], ...]:
    """The word sequences a contraction may stand for; empty for a word that is not one."""
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suffix, fulls in _SUFFIXES:
        stem = word[: -len(suffix)]
        if word.endswith(suffix) and stem.isalpha():
            return tuple((stem, full) for full in fulls)
    return ()


def contraction_distance(hyp: list[str], ref: list[str]) -> int:
    """Word Levenshtein distance in which a word also matches one of its expansions for free."""
    hx = [expansions(w) for w in hyp]
    rx = [expansions(w) for w in ref]
    if not any(hx) and not any(rx):
        return uw.edit_distance(hyp, ref)
    n, m = len(hyp), len(ref)
    dist = [[n + m + 1] * (m + 1) for _ in range(n + 1)]
    dist[0][0] = 0

    def relax(i: int, j: int, d: int) -> None:
        if d < dist[i][j]:
            dist[i][j] = d

    for i in range(n + 1):
        for j in range(m + 1):
            d = dist[i][j]
            if i < n and j < m:
                relax(i + 1, j + 1, d + (hyp[i] != ref[j]))
            if i < n:
                relax(i + 1, j, d + 1)
                for e in hx[i]:
                    if tuple(ref[j : j + len(e)]) == e:
                        relax(i + 1, j + len(e), d)
            if j < m:
                relax(i, j + 1, d + 1)
                for e in rx[j]:
                    if tuple(hyp[i : i + len(e)]) == e:
                        relax(i + len(e), j + 1, d)
    return dist[n][m]


WILD = object()


def masked(hyp: list[str], spans: list[tuple[int, int]]) -> list[Any]:
    """``hyp`` with each span [i1, i2) replaced by one wildcard (an empty span inserts one)."""
    out: list[Any] = []
    pos = 0
    for i1, i2 in sorted(spans):
        out.extend(hyp[pos:i1])
        out.append(WILD)
        pos = i2
    out.extend(hyp[pos:])
    return out


def wildcard_distance(items: list[Any], ref: list[str]) -> int:
    """Word Levenshtein distance where a ``WILD`` item absorbs any run of reference words free."""
    prev = list(range(len(ref) + 1))
    for x in items:
        if x is WILD:
            cur = [prev[0]]
            for j in range(1, len(ref) + 1):
                cur.append(min(prev[j], cur[j - 1]))
        else:
            cur = [prev[0] + 1]
            for j, y in enumerate(ref, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def to_ms(value: float, unit: str = "s") -> int:
    return round(float(value) * TIMING_UNITS[unit])


def largest_shift_ms(a_timings: list, b_timings: list, unit: str = "s") -> tuple[int, int] | None:
    """(largest start/end shift in ms, segments shifted), or None when the lists do not pair."""
    if len(a_timings) != len(b_timings):
        return None
    if any(x[0] != y[0] for x, y in zip(a_timings, b_timings, strict=True)):
        return None
    largest = shifted = 0
    for x, y in zip(a_timings, b_timings, strict=True):
        start = abs(to_ms(x[1], unit) - to_ms(y[1], unit))
        end = abs(to_ms(x[2], unit) - to_ms(y[2], unit))
        s = max(start, end)
        largest = max(largest, s)
        shifted += s > 0
    return largest, shifted


def nearest_rank(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def rate(num: float, den: float) -> float | None:
    return round(num / den, 5) if den else None


SIDES: dict[str, dict[str, str]] = {
    "ragged": {
        "a": "the recording alone, at its own length",
        "b": "slot 0 of a batch of real recordings, every row at its own length",
    },
    "equalised": {
        "a": "the recording alone, zero-padded to the batch's common length",
        "b": "slot 0 of the batch of real recordings, every row zero-padded to one common length",
    },
    "fixed": {
        "a": "slot 0 of a batch whose other rows are silence, every row at the common length",
        "b": "slot 0 of the batch of real recordings at the common length (the equalised arm's b)",
    },
    CAPTURES_ARM: {
        "a": "the served answer: the server capture with --padding fixed",
        "b": "a server capture of the same recordings with --padding ragged",
    },
}

SECOND_DECODE = (
    "A place exists only where two decodes of the same audio in two different shapes differ."
    " The server decodes each stream once, so serving a place flag costs at least one extra"
    " decode per stream; the confidence flag needs none."
)
SHAPE_AXIS = (
    "The probe's positive control is a shape change: it is present when a two-shape arm (ragged"
    " or equalised) diverged. It shows the harness can see a shape difference. No control on"
    " the run exercises the neighbour-content axis, the only thing the stock fixed arm varies."
)
STOCK_SESSIONS = (
    "sessions: the stock probe starts and ends every row of a batch together, with silence or"
    " real recordings as neighbours; the server admits sessions at different times, keeps pad"
    " streams alive, and splits ending rows into eager side batches"
)


# --- the record is checked before it is scored ------------------------------------------------


def comparable(timings: list) -> list[tuple]:
    """(word, start, end) of each timing entry: what the probe compares; a stored confidence,
    the fourth element, is not."""
    return [tuple(t[:3]) for t in timings]


def check_arm(arm: dict[str, Any], references: dict[str, str]) -> None:
    """Refuse an arm whose counters disagree with its own stored transcripts."""
    t = arm["transcripts"]
    divs = arm["divergences"]
    problems = []
    if len(t) != arm["checked"]:
        problems.append(f"checked {arm['checked']} but {len(t)} transcripts")
    with_b = sum("b" in v for v in t.values())
    if with_b != arm["text_divergent"]:
        problems.append(f"text_divergent {arm['text_divergent']} but {with_b} transcripts store b")
    timing_div = sum(not d["text_differs"] for d in divs)
    if timing_div != arm["timing_only_divergent"]:
        problems.append(
            f"timing_only_divergent {arm['timing_only_divergent']}"
            f" but {timing_div} divergences have text_differs false"
        )
    seen = Counter(d["librispeech_id"] for d in divs)
    for rid, k in seen.items():
        if k > 1:
            problems.append(f"divergence {rid} appears {k} times")
    for d in divs:
        rid = d["librispeech_id"]
        stored = t.get(rid)
        if stored is None:
            problems.append(f"divergence {rid} has no transcript")
            continue
        if d["a"] != stored["a"] or d["b"] != stored.get("b", stored["a"]):
            problems.append(f"divergence {rid} text disagrees with its transcript")
        if d["text_differs"] != (d["a"] != d["b"]):
            problems.append(f"divergence {rid} text_differs flag disagrees with its texts")
        if not d["text_differs"] and comparable(d["a_timings"]) == comparable(d["b_timings"]):
            problems.append(f"divergence {rid} is timing-only but its timings are equal")
    text_ids = {d["librispeech_id"] for d in divs if d["text_differs"]}
    for rid, stored in t.items():
        if "b" in stored and stored["b"] == stored["a"]:
            problems.append(f"transcript {rid} stores a b equal to its a")
        elif "b" in stored and rid not in text_ids:
            problems.append(f"transcript {rid} stores a b but has no text divergence")
    missing = [rid for rid in t if rid not in references]
    if missing:
        problems.append(f"{len(missing)} transcripts without a reference, e.g. {missing[0]}")
    if problems:
        raise Refused("record refused: " + "; ".join(problems))


def word_entries(words: Any, where: str) -> list[tuple]:
    """``[[word, start_s, end_s, conf], ...]`` as tuples; anything else is refused."""
    if not isinstance(words, list):
        raise Refused(f"record refused: {where} is not a list")
    for i, w in enumerate(words):
        if not isinstance(w, list | tuple) or len(w) != 4:
            raise Refused(f"record refused: {where} entry {i} is not [word, start_s, end_s, conf]")
    return [tuple(w) for w in words]


def check_every_recording(arm: dict[str, Any]) -> None:
    """Refuse an arm whose ``every_recording`` disagrees with its transcripts, its divergences
    or its ``confidence_only_divergent``. An arm with neither passes. An arm with one and not the
    other is refused: the count is of ``every_recording``'s recordings, so without them nothing
    can check it."""
    every = arm.get("every_recording")
    if every is None:
        if "confidence_only_divergent" in arm:
            raise Refused(
                f"record refused: confidence_only_divergent {arm['confidence_only_divergent']!r}"
                " without every_recording, the recordings it counts; it cannot be checked"
            )
        return
    if not isinstance(every, dict):
        raise Refused("record refused: every_recording is not a mapping of recording id")
    t = arm["transcripts"]
    divs = {d["librispeech_id"]: d for d in arm["divergences"]}
    problems = []
    missing = [rid for rid in t if rid not in every]
    if missing:
        problems.append(
            f"every_recording lacks {len(missing)} checked recordings, e.g. {missing[0]}"
        )
    extra = [rid for rid in every if rid not in t]
    if extra:
        problems.append(f"every_recording holds {len(extra)} unchecked recordings, e.g. {extra[0]}")
    confidence_only = 0
    for rid, e in every.items():
        stored = t.get(rid)
        if stored is None:
            continue
        a_text, b_text = e.get("a_text"), e.get("b_text")
        if a_text != stored["a"] or b_text != stored.get("b", stored["a"]):
            problems.append(f"every_recording {rid} text disagrees with its transcript")
        aw = word_entries(e.get("a_words"), f"every_recording {rid} a_words")
        bw = word_entries(e.get("b_words"), f"every_recording {rid} b_words")
        differs = a_text != b_text or comparable(aw) != comparable(bw)
        d = divs.get(rid)
        if differs and d is None:
            problems.append(f"every_recording {rid} differs but no divergence lists it")
        elif d is not None and not differs:
            problems.append(f"every_recording {rid} agrees but a divergence lists it")
        elif d is not None and (
            comparable(aw) != comparable(d["a_timings"])
            or comparable(bw) != comparable(d["b_timings"])
        ):
            problems.append(f"every_recording {rid} words disagree with its divergence's timings")
        confidence_only += not differs and aw != bw
    if "confidence_only_divergent" not in arm:
        problems.append("every_recording without confidence_only_divergent")
    elif arm["confidence_only_divergent"] != confidence_only:
        problems.append(
            f"confidence_only_divergent {arm['confidence_only_divergent']} but"
            f" {confidence_only} recordings differ only in confidence"
        )
    if problems:
        raise Refused("record refused: " + "; ".join(problems))


# --- flags, their scores, and the nulls they are judged against ------------------------------


class Rec:
    """One side of one recording, normalised, with its word errors against the reference."""

    __slots__ = ("_aligned", "errors", "hyp", "ref")

    def __init__(self, hyp: list[str], ref: list[str]) -> None:
        self.hyp, self.ref = hyp, ref
        self.errors = uw.edit_distance(hyp, ref)
        self._aligned: Any = None

    @property
    def aligned(self) -> Any:
        if self._aligned is None:
            self._aligned = uw.Aligned(self.hyp, self.ref)
        return self._aligned

    def removable(self, spans: list[tuple[int, int]]) -> int:
        return self.errors - wildcard_distance(masked(self.hyp, spans), self.ref)

    def not_right(self, spans: list[tuple[int, int]]) -> int:
        return sum(span_not_right(self.aligned, i1, i2) for i1, i2 in spans)


def span_not_right(aligned: Any, i1: int, i2: int) -> bool:
    """The not-right rule of every flag, the place flag included: ``analyse``'s rule for a
    non-empty side of a place; for an empty span, a reference word missing at that point."""
    if i1 == i2:
        return i1 in aligned.missing_before
    return not aligned.right(i1, i2)


def place_randomly(n: int, lengths: list[int], rng: random.Random) -> list[tuple[int, int]]:
    """Non-overlapping spans of the given lengths at uniformly random positions in ``n`` words."""
    order = list(lengths)
    rng.shuffle(order)
    free = n - sum(order)
    if free < 0:
        raise ValueError(f"spans of {sum(order)} words do not fit in {n}")
    cuts = sorted(rng.sample(range(free + len(order)), len(order)))
    spans = []
    used = 0
    for k, (cut, length) in enumerate(zip(cuts, order, strict=True)):
        start = cut - k + used
        spans.append((start, start + length))
        used += length
    return spans


def p_value(null: list[int], observed: int) -> float:
    return round((1 + sum(x >= observed for x in null)) / (len(null) + 1), 5)


def nulls(
    spans_by_rid: dict[str, list[tuple[int, int]]],
    recs: dict[str, Rec],
    observed_removable: int,
    observed_not_right: int,
    total_errors: int | None,
    draws: int,
    seed: str,
) -> dict[str, Any] | None:
    """Both nulls of one flag; ``total_errors`` None withholds ``recall_mean``."""
    if not spans_by_rid or draws <= 0:
        return None
    units = sum(len(s) for s in spans_by_rid.values())
    by_len = sorted(recs, key=lambda rid: len(recs[rid].hyp))
    lens = [len(recs[rid].hyp) for rid in by_len]
    out: dict[str, Any] = {"draws": draws, "seed": seed}
    for name in ("same_recordings", "random_recordings"):
        rng = random.Random(f"{seed}/{name}")
        removable: list[int] = []
        not_right: list[int] = []
        for _ in range(draws):
            e = w = 0
            for rid, spans in spans_by_rid.items():
                lengths = [i2 - i1 for i1, i2 in spans]
                if name == "same_recordings":
                    target = recs[rid]
                else:
                    lo = bisect.bisect_left(lens, sum(lengths))
                    target = recs[by_len[rng.randrange(lo, len(by_len))]]
                placed = place_randomly(len(target.hyp), lengths, rng)
                e += target.removable(placed)
                w += target.not_right(placed)
            removable.append(e)
            not_right.append(w)
        mean_e = sum(removable) / draws
        mean_w = sum(not_right) / draws
        out[name] = {
            "recordings_drawn_from": len(spans_by_rid) if name == "same_recordings" else len(recs),
            "word_errors_removable_mean": round(mean_e, 5),
            "recall_mean": None if total_errors is None else rate(mean_e, total_errors),
            "observed_over_null_mean": rate(observed_removable, mean_e),
            "p_value_word_errors_removable": p_value(removable, observed_removable),
            "not_right_mean": round(mean_w, 5),
            "not_right_rate_mean": rate(mean_w, units),
            "p_value_not_right": p_value(not_right, observed_not_right),
        }
    return out


def score_flag(
    spans_by_rid: dict[str, list[tuple[int, int]]],
    recs: dict[str, Rec],
    draws: int,
    seed: str,
    recall_withheld: str | None = None,
) -> dict[str, Any]:
    """Precision, recall and both nulls of one flag on one side, by the rules above; every flag
    goes through here, so every flag is judged by ``span_not_right`` and the same wildcard."""
    units = sum(len(s) for s in spans_by_rid.values())
    not_right = sum(recs[rid].not_right(spans) for rid, spans in spans_by_rid.items())
    removable = sum(recs[rid].removable(spans) for rid, spans in spans_by_rid.items())
    total = sum(r.errors for r in recs.values())
    out = {
        "spans": units,
        "empty_spans": sum(i1 == i2 for s in spans_by_rid.values() for i1, i2 in s),
        "words_flagged": sum(i2 - i1 for s in spans_by_rid.values() for i1, i2 in s),
        "recordings_flagged": len(spans_by_rid),
        "not_right": not_right,
        "precision": rate(not_right, units),
        "word_errors_removable": removable,
        "corpus_word_errors": total,
        "recall": None if recall_withheld else rate(removable, total),
    }
    if recall_withheld:
        out["recall_withheld"] = recall_withheld
    out["null"] = nulls(
        spans_by_rid,
        recs,
        removable,
        not_right,
        None if recall_withheld else total,
        draws,
        seed,
    )
    return out


def confidence_tokens(
    entries: Any, rid: str, tokens: Callable[[str], list[str]]
) -> tuple[list[str], list[float]]:
    """A side's confidence entries as normalised tokens, each with its word's confidence, the
    fourth element of ``[word, start, end, confidence]``; any other length is refused."""
    if not isinstance(entries, list):
        raise Refused(f"confidence refused: the words of {rid} are not a list")
    toks: list[str] = []
    confs: list[float] = []
    for i, e in enumerate(entries):
        if not isinstance(e, list | tuple) or len(e) != 4:
            size = len(e) if isinstance(e, list | tuple) else type(e).__name__
            raise Refused(
                f"confidence refused: entry {i} of {rid} has {size} elements; expected"
                " [word, start, end, confidence]"
            )
        c = e[3]
        if isinstance(c, bool) or not isinstance(c, int | float) or not math.isfinite(c):
            raise Refused(f"confidence refused: word {i} of {rid} has confidence {c!r}")
        for tok in tokens(str(e[0])):
            toks.append(tok)
            confs.append(float(c))
    return toks, confs


def runs_of(positions: list[int]) -> list[tuple[int, int]]:
    """Consecutive positions as [i1, i2) spans."""
    spans: list[tuple[int, int]] = []
    for p in sorted(positions):
        if spans and spans[-1][1] == p:
            spans[-1] = (spans[-1][0], p + 1)
        else:
            spans.append((p, p + 1))
    return spans


def tie_range(
    ranked: list[tuple[Any, ...]],
    k: int,
    recs: dict[str, Rec],
    recall_withheld: str | None = None,
) -> dict[str, Any]:
    """Every way of breaking the tie at the cutoff, scored: the lowest and highest of each of
    ``TIE_FIGURES``. ``ranked`` is the confidence flag's ranking (confidence first, the seeded
    tie-break draw second, recording and position last); the flag takes its first ``k``. Every
    word below the cutoff is flagged whichever way the tie is broken; the words at the cutoff
    fill the rest, in each of their combinations. Scored without nulls."""
    flagged = ranked[:k]
    cutoff = flagged[-1][0]
    below = [(rid, pos) for c, *_, rid, pos in flagged if c < cutoff]
    tied = [(rid, pos) for c, *_, rid, pos in ranked if c == cutoff]
    chosen = len(flagged) - len(below)
    choices = math.comb(len(tied), chosen)
    out: dict[str, Any] = {"tied": len(tied), "chosen": chosen, "choices": choices}
    if choices > TIE_CHOICES_MAX:
        out["range"] = None
        out["not_enumerated"] = f"{choices} choices, more than the {TIE_CHOICES_MAX} scored"
        return out
    seen: dict[str, list[Any]] = {key: [] for key in TIE_FIGURES}
    for pick in itertools.combinations(tied, chosen):
        positions: dict[str, list[int]] = {}
        for rid, pos in [*below, *pick]:
            positions.setdefault(rid, []).append(pos)
        spans_by_rid = {rid: runs_of(ps) for rid, ps in positions.items()}
        score = score_flag(spans_by_rid, recs, 0, "", recall_withheld)
        for key in TIE_FIGURES:
            seen[key].append(score[key])
    out["range"] = {
        key: None if None in values else [min(values), max(values)] for key, values in seen.items()
    }
    return out


def confidence_flag(
    side: str,
    confidence: dict[str, Any] | None,
    recs: dict[str, Rec],
    places: dict[str, Any],
    tokens: Callable[[str], list[str]],
    draws: int,
    seed: str,
    unavailable: str,
    recall_withheld: str | None = None,
) -> dict[str, Any]:
    """The confidence flag on one side, K being the words the place flag marks on that side."""
    if confidence is None or confidence.get("unavailable"):
        reason = unavailable if confidence is None else confidence["unavailable"]
        return {"status": "confidence unavailable", "reason": f"{reason}; nothing was ranked"}
    words = confidence.get("words")
    if not isinstance(words, dict):
        raise Refused("confidence refused: words is not a mapping of recording id to words")
    per_token: dict[str, list[float]] = {}
    for rid, rec in recs.items():
        if rid not in words:
            raise Refused(f"confidence refused: no words for transcript {rid}")
        toks, confs = confidence_tokens(words[rid], rid, tokens)
        if toks != rec.hyp:
            raise Refused(f"confidence refused: the words of {rid} do not spell side {side}")
        per_token[rid] = confs
    head = {
        "side": side,
        "source": confidence.get("source"),
        "entries_not_in_this_arm": sum(rid not in recs for rid in words),
    }
    if "identity" in confidence:
        head["identity"] = confidence["identity"]
    k = places["words_flagged"]
    values = {c for confs in per_token.values() for c in confs}
    if len(values) <= 1:
        return {
            **head,
            "status": "confidence constant",
            "value": next(iter(values)) if values else None,
            "reason": f"every word on side {side} has the same confidence, so a ranking would be"
            " the tie-break alone; not scored",
        }
    if k == 0:
        return {
            **head,
            "status": "nothing to compare",
            "reason": f"the place flag marks 0 words on side {side}",
        }
    rng = random.Random(f"{seed}/ties")
    ranked = sorted(
        (c, rng.random(), rid, pos)
        for rid, confs in per_token.items()
        for pos, c in enumerate(confs)
    )
    flagged = ranked[:k]
    cutoff = flagged[-1][0]
    positions: dict[str, list[int]] = {}
    for _, _, rid, pos in flagged:
        positions.setdefault(rid, []).append(pos)
    spans_by_rid = {rid: runs_of(ps) for rid, ps in positions.items()}
    score = score_flag(spans_by_rid, recs, draws, seed, recall_withheld)

    def brief(s: dict[str, Any]) -> dict[str, Any]:
        n = s["null"] or {}
        same = n.get("same_recordings") or {}
        rand = n.get("random_recordings") or {}
        return {
            "words_flagged": s["words_flagged"],
            "spans": s["spans"],
            "precision": s["precision"],
            "recall": s["recall"],
            "not_right_rate_same_recordings_null": same.get("not_right_rate_mean"),
            "not_right_rate_random_recordings_null": rand.get("not_right_rate_mean"),
            "p_value_removable_random_recordings": rand.get("p_value_word_errors_removable"),
        }

    return {
        **head,
        "status": "scored",
        "k_words": k,
        "k_rule": f"K = the words the place flag marks on side {side}",
        "cutoff_confidence": cutoff,
        "tied_at_cutoff": {
            "flagged": sum(c == cutoff for c, *_ in flagged),
            "all": sum(c == cutoff for c, *_ in ranked),
        },
        "tie_break": f"seeded shuffle, seed {seed}/ties",
        "tie_range": tie_range(ranked, k, recs, recall_withheld),
        "flag": score,
        "flagged_words": {rid: [list(span) for span in s] for rid, s in spans_by_rid.items()},
        "side_by_side": {"places": brief(places), "confidence": brief(score)},
        "place_flag_empty_spans": places["empty_spans"],
    }


# --- one arm ----------------------------------------------------------------------------------


def score_arm(
    name: str,
    arm: dict[str, Any],
    references: dict[str, str],
    words: frozenset[str],
    normalise: Callable[[str], str] = normalise,
    *,
    timing_unit: str = "s",
    draws: int = DEFAULT_DRAWS,
    seed: str = str(DEFAULT_SEED),
    confidence: dict[str, Any] | None = None,
    captures: bool = False,
) -> dict[str, Any]:
    check_arm(arm, references)
    check_every_recording(arm)
    if timing_unit not in TIMING_UNITS:
        raise Refused(f"record refused: timing_unit {timing_unit!r} is not one of 's', 'ms'")
    every = arm.get("every_recording")
    by_side: dict[str, dict[str, Any] | None] = {"a": None, "b": None}
    if every is not None:
        for s in "ab":
            by_side[s] = {
                "side": s,
                "source": "every_recording",
                "words": {rid: e[f"{s}_words"] for rid, e in every.items()},
            }
    if confidence is not None:
        side = confidence.get("side")
        if side not in ("a", "b"):
            raise Refused(f"confidence refused: side {side!r} is not 'a' or 'b'")
        if every is not None:
            raise Refused(
                "record refused: two confidence inputs for this arm, every_recording and"
                " --confidence"
            )
        by_side[side] = confidence
    recall_withheld = None
    if captures:
        unavailable = {s: f"no confidence input covers side {s}" for s in "ab"}
    else:
        unavailable = {
            s: "the record has no every_recording for this arm, so no per-word confidence for"
            " every checked recording"
            for s in "ab"
        }
        if every is None:
            recall_withheld = (
                "the record has no every_recording for this arm; contract C2 computes recall"
                " against every word error of the side only from every_recording"
            )
    if any(len(t) > 3 for d in arm["divergences"] for t in d["a_timings"] + d["b_timings"]):
        for s in "ab":
            unavailable[s] += (
                ". The divergences' timings carry a confidence element, but only for divergent"
                " recordings; ranking those alone would show the confidence flag where the places"
                " are"
            )

    def tokens(text: str) -> list[str]:
        # exactly the tokens analyse() will see for this text
        return normalise(text).lower().split()

    label = f"{seed}/{name}"
    ref_words = ref_oov = 0
    tolerant = {"a": 0, "b": 0}
    digit_tokens = {"a": 0, "b": 0}
    raw_divergent = punctuation_only = 0
    recs: dict[str, dict[str, Rec]] = {"a": {}, "b": {}}
    spans: dict[str, dict[str, list[tuple[int, int]]]] = {"a": {}, "b": {}}
    both_right = 0
    divergent: list[dict[str, str]] = []
    contraction_places: list[dict[str, str]] = []
    for rid, stored in arm["transcripts"].items():
        a_raw = stored["a"]
        b_raw = stored.get("b", a_raw)
        ref = tokens(references[rid])
        side = {"a": tokens(a_raw), "b": tokens(b_raw)}
        ref_words += len(ref)
        ref_oov += sum(w not in words for w in ref)
        for s, hyp in side.items():
            recs[s][rid] = Rec(hyp, ref)
            tolerant[s] += contraction_distance(hyp, ref)
            digit_tokens[s] += sum(any(c.isdigit() for c in w) for w in hyp)
        raw_divergent += a_raw != b_raw
        a, b = side["a"], side["b"]
        if a == b:
            punctuation_only += a_raw != b_raw
            continue
        divergent.append(
            {
                "librispeech_id": rid,
                "reference": " ".join(ref),
                "alone": " ".join(a),
                "in_batch": " ".join(b),
            }
        )
        # The same matcher call analyse() makes on the same tokens: these are its places.
        ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        places = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in ops if tag != "equal"]
        spans["a"][rid] = [(i1, i2) for i1, i2, _, _ in places]
        spans["b"][rid] = [(j1, j2) for _, _, j1, j2 in places]
        for i1, i2, j1, j2 in places:
            a_not_right = span_not_right(recs["a"][rid].aligned, i1, i2)
            b_not_right = span_not_right(recs["b"][rid].aligned, j1, j2)
            both_right += not a_not_right and not b_not_right
            if contraction_distance(a[i1:i2], b[j1:j2]) == 0:
                contraction_places.append(
                    {"librispeech_id": rid, "a": " ".join(a[i1:i2]), "b": " ".join(b[j1:j2])}
                )

    # analyse() divides by the changed recordings' reference words, so it cannot run on none.
    analysis = uw.analyse(divergent, words) if divergent else None
    place_count = sum(len(s) for s in spans["a"].values())
    by_analyse = {"a": 0, "b": 0}
    both_right_by_analyse = 0
    at_place_ref = at_place_absent = 0
    if analysis is not None:
        both_right_by_analyse = analysis["both_right"]
        by_analyse = {
            "a": place_count - analysis["alone_right"] - both_right_by_analyse,
            "b": place_count - analysis["batch_right"] - both_right_by_analyse,
        }
        at_place_ref = analysis["out_of_dictionary"]["reference_words_at_places"]
        at_place_absent = analysis["out_of_dictionary"]["reference_words_at_places_absent"]

    place_scores = {}
    for s in "ab":
        place_scores[s] = score_flag(
            spans[s], recs[s], draws, f"{label}/places/{s}", recall_withheld
        )
        place_scores[s]["not_right_by_analyse"] = by_analyse[s]
    if not place_count:
        bound = "no places on this arm"
    elif both_right == 0:
        bound = (
            "both_right is 0 on this arm, so at every place at least one side is not right:"
            " precision_a + precision_b >= 1 holds by construction. A precision near 0.5 for"
            " either side is what place selection alone guarantees; judge each precision"
            " against its nulls' not-right rate, not against 0.5."
        )
    else:
        bound = (
            f"both_right is {both_right} on this arm, so precision_a + precision_b >= 1 does"
            " not hold by construction here."
        )

    timing = [d for d in arm["divergences"] if not d["text_differs"]]
    shifts: list[int] = []
    shifted_segments = total_segments = 0
    unpairable: list[str] = []
    for d in timing:
        got = largest_shift_ms(d["a_timings"], d["b_timings"], timing_unit)
        if got is None:
            unpairable.append(d["librispeech_id"])
            continue
        shifts.append(got[0])
        shifted_segments += got[1]
        total_segments += len(d["a_timings"])
    shifts.sort()

    confidence_scores = {
        s: confidence_flag(
            s,
            by_side[s],
            recs[s],
            place_scores[s],
            tokens,
            draws,
            f"{label}/confidence/{s}",
            unavailable[s],
            recall_withheld,
        )
        for s in "ab"
    }
    withheld = [f"recall: {recall_withheld}"] if recall_withheld else []
    withheld += [
        f"confidence flag, side {s}: {c['status']}: {c['reason']}"
        for s, c in confidence_scores.items()
        if c["status"] != "scored"
    ]
    errors = {s: sum(r.errors for r in recs[s].values()) for s in "ab"}
    checked = arm["checked"]
    return {
        "sides": SIDES.get(name, {"a": "unknown arm; see the probe", "b": "unknown arm"}),
        "analyse_labels": {
            "alone": "side a",
            "in_batch": "side b",
            "batch": "side b",
            "out_of_dictionary.rate_at_places": "the share of the reference words touched by any"
            " place that are absent from the word list, whether or not either side was wrong",
        },
        "checked": checked,
        "text_divergent": arm["text_divergent"],
        "text_divergent_rate": rate(arm["text_divergent"], checked),
        "timing_only_divergent": arm["timing_only_divergent"],
        "timing_only_divergent_rate": rate(arm["timing_only_divergent"], checked),
        "confidence_only_divergent": arm.get("confidence_only_divergent"),
        "withheld": withheld,
        "normalisation": {
            "text_divergent_raw": raw_divergent,
            "text_divergent_after_normalisation": len(divergent),
            "punctuation_or_case_only": punctuation_only,
            "places_differing_only_by_a_contraction": len(contraction_places),
            "contraction_only_places": contraction_places,
            "tokens_with_digits": digit_tokens,
        },
        "places": analysis,
        "places_note": (
            None
            if analysis is not None
            else "no recording's transcripts differ after normalisation: no places, analyse not run"
        ),
        "flag": {
            "places": place_count,
            "both_right": both_right,
            "both_right_by_analyse": both_right_by_analyse,
            "not_right_rule": "span_not_right for every flag; not_right_by_analyse is analyse's"
            " count, which also calls a gap not right when the other side's words there are"
            " correct",
            "precision_bound": bound,
            "cost": SECOND_DECODE,
            "reference_words_at_places": at_place_ref,
            "reference_words_at_places_note": "descriptive only, not a baseline for recall: the"
            " wildcard can absorb errors at reference words no place touches; see the nulls",
            "a": place_scores["a"],
            "b": place_scores["b"],
        },
        "confidence": confidence_scores,
        "out_of_dictionary_at_places": {
            "label": "reference words at places absent from the word list, whether or not"
            " either side was wrong there",
            "reference_words_at_places": at_place_ref,
            "absent_from_word_list": at_place_absent,
            "share_absent": rate(at_place_absent, at_place_ref),
            "corpus_share_absent": rate(ref_oov, ref_words),
        },
        "corpus": {
            "recordings": checked,
            "reference_words": ref_words,
            "word_errors_a": errors["a"],
            "word_errors_b": errors["b"],
            "wer_a": rate(errors["a"], ref_words),
            "wer_b": rate(errors["b"], ref_words),
            "contraction_form_word_errors_a": errors["a"] - tolerant["a"],
            "contraction_form_word_errors_b": errors["b"] - tolerant["b"],
            "wer_a_contractions_tolerated": rate(tolerant["a"], ref_words),
            "wer_b_contractions_tolerated": rate(tolerant["b"], ref_words),
            "reference_words_out_of_dictionary": ref_oov,
            "out_of_dictionary_rate": rate(ref_oov, ref_words),
        },
        "timing_only_shift_ms": {
            "recordings": len(timing),
            "paired": len(shifts),
            "unpairable": unpairable,
            "largest_shift_counts": {str(k): v for k, v in sorted(Counter(shifts).items())},
            "min": shifts[0] if shifts else None,
            "median": nearest_rank(shifts, 0.5),
            "p90": nearest_rank(shifts, 0.9),
            "max": shifts[-1] if shifts else None,
            "segments_shifted": shifted_segments,
            "segments_in_paired_recordings": total_segments,
        },
    }


# --- controls, the serving setting, and the record as a whole ---------------------------------


def _count(value: Any) -> int | None:
    """``value`` when it is a whole number (not a bool, not a float); else None."""
    return value if type(value) is int else None


def repeat_contradicts(repeat: Any, captures: bool) -> str | None:
    """None when the run's own repeat counts show every repeat identical; else what they show.
    The verdict string is the writer's reading of these counts, so it is checked against them.
    Every count, the denominator and each numerator, must be a whole number: a bool or a float
    equal to the denominator (``True == 1``, ``1.0 == 1``) is not a count, so it shows nothing."""
    if not isinstance(repeat, dict):
        return f"the run's repeat counts are {repeat!r}"
    if captures:
        n = _count(repeat.get("recordings"))
        ok = n is not None and n > 0 and _count(repeat.get("identical")) == n
        ok = ok and repeat.get("digests_match") is True
        keys = ("recordings", "identical", "digests_match")
    else:
        n = _count(repeat.get("checked"))
        ok = n is not None and n > 0
        ok = ok and _count(repeat.get("alone_identical")) == n
        ok = ok and _count(repeat.get("batch_identical")) == n
        keys = ("checked", "alone_identical", "batch_identical")
    if ok:
        return None
    return "the run's repeat counts read " + ", ".join(f"{k} {repeat.get(k)!r}" for k in keys)


def diverged(arm: dict[str, Any]) -> bool:
    return bool(arm["text_divergent"] or arm["timing_only_divergent"])


def control_seen(run: dict[str, Any]) -> bool:
    """Whether the record itself shows the positive control: a two-shape arm (for a
    server-captures record, its one arm) with a divergence. The string alone is not enough."""
    return any(
        diverged(arm)
        for name, arm in run["arms"].items()
        if name in ("ragged", "equalised", CAPTURES_ARM)
    )


def zero_reading(name: str, arm: dict[str, Any], positive_control: Any, seen: bool) -> str | None:
    """What a zero in this arm can mean, given the run's positive control as it reads and as
    the run's arms show it (``seen``); None if the arm is not a zero."""
    if diverged(arm):
        return None
    says = isinstance(positive_control, str) and positive_control.startswith("present")
    if not (says and seen):
        if says:
            control = (
                f"reads {positive_control!r}, but no two-shape arm of this run has a divergence"
                " in the record"
            )
        elif positive_control is None:
            control = "is not in the record (the run has not finished)"
        else:
            control = f"reads {positive_control!r}"
        return (
            f"uninterpretable: no divergence, and the positive control {control}; a zero here"
            " cannot be told from a harness that sees nothing"
        )
    if name == "fixed":
        return (
            "zero at one shape. The positive control is present but on the shape axis; nothing"
            " on this run shows the harness can see a change of neighbour content, the only"
            " thing this arm varies."
        )
    return "zero, with the positive control present on the shape axis"


def setting_differences(record: dict[str, Any], serving: dict[str, Any]) -> list[str]:
    """Every way the record's setting differs from the serving setting, read from the record."""
    diffs: list[str] = []
    captures = record.get("source") == CAPTURES

    def compare(label: str, have: Any, want: Any) -> None:
        if want is None:
            said = "not recorded in the record" if have is None else f"record {have!r}"
            diffs.append(f"{label}: {said}, serving <placeholder: not given>")
        elif have is None:
            diffs.append(f"{label}: not recorded in the record, serving {want!r}")
        elif have != want:
            diffs.append(f"{label}: record {have!r}, serving {want!r}")

    runs = record.get("runs", {})

    def readings(key: str) -> list[tuple[str, Any, str]]:
        """(dtype, value, what the value is) of a reading the probe takes off the built
        pipeline: the run's own when it carries the key; else the record's top-level one, which
        is the last dtype built, labelled so; else None, "not recorded"."""
        out = []
        for dtype, run in runs.items():
            if key not in run and key in record:
                where = f"observed, the record's top-level reading; the {dtype} run has none"
                out.append((dtype, record[key], where))
            else:
                out.append((dtype, run.get(key), f"observed in the {dtype} run"))
        return out

    compare("chunk_ms", record.get("chunk_ms"), serving["chunk_ms"])
    if captures:
        # The capture stamps att_context as the server's /readyz observed it on the built
        # encoder. The server does not report matmul: the capture declares it and checks the
        # declaration against the installed default only, so it is labelled as declared.
        compare("att_context_size", record.get("att_context_size"), serving["att_context_size"])
        compare(
            "matmul_precision (declared, not observed)",
            record.get("matmul_precision"),
            serving["matmul_precision"],
        )
    else:
        # A stock record's att_context_size and matmul_precision are what the probe asked the
        # pipeline for. The encoder's attention context read off the built pipeline, where the
        # record has it, is compared instead of the request; nothing reads the matmul back.
        want = serving["att_context_size"]
        if "att_context_size_observed" in record or any(
            "att_context_size_observed" in run for run in runs.values()
        ):
            for _, value, where in readings("att_context_size_observed"):
                compare(f"att_context_size ({where})", value, want)
        else:
            compare(
                "att_context_size (requested, not observed)", record.get("att_context_size"), want
            )
        compare(
            "matmul_precision (requested, not observed)",
            record.get("matmul_precision"),
            serving["matmul_precision"],
        )
    for dtype in runs:
        compare("dtype", dtype, serving["dtype"])
    if captures:
        compare("word_confidence", record.get("word_confidence"), serving["word_confidence"])
        compare("bucket", record.get("bucket", record.get("batch")), serving["bucket"])
        # The encoder step and the decoder's CUDA graphs are two settings; each is compared.
        compare("execution", record.get("execution"), serving["execution"])
        compare("decoder_graphs", on_off(record.get("decoder_graphs")), serving["decoder_graphs"])
        return diffs
    compare(
        "word_confidence (requested, not observed)",
        record.get("word_confidence"),
        serving["word_confidence"],
    )
    mode = serving["word_confidence"]
    step_wanted = None if mode is None else on_off(mode != "off")
    for _, observed, where in readings("word_confidence_observed"):
        step = observed.get("decoder_step_confidence") if isinstance(observed, dict) else None
        compare(f"decoder_step_confidence ({where})", on_off(step), step_wanted)
    want = serving["bucket"] if serving["bucket"] is not None else "<placeholder: not given>"
    diffs.append(
        f"batch: record {record.get('batch')!r} rows in the stock pipeline, serving bucket {want}"
    )
    # The encoder step: only the spec's request is in a stock record, and it is labelled so.
    graphs = record.get("use_cuda_graphs")
    have = None if graphs is None else ("graph path" if graphs else "eager")
    compare("execution (requested, not observed)", have, serving["execution"])
    # The decoder's CUDA graphs, read from the built pipeline: a separate setting.
    for _, observed, where in readings("decoder_graphs_observed"):
        compare(f"decoder_graphs ({where})", on_off(observed), serving["decoder_graphs"])
    diffs.append(STOCK_SESSIONS)
    return diffs


def on_off(value: Any) -> Any:
    """A recorded bool as the serving flag's "on" or "off"; anything else unchanged."""
    if isinstance(value, bool):
        return "on" if value else "off"
    return value


def load_record(path: Path) -> tuple[dict, str]:
    """Parse a record; return it and the SHA-256 of the exact bytes parsed. The probe writes
    through a temporary file and ``os.replace``, so a read sees a whole file or the old one."""
    raw = path.read_bytes()
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except json.JSONDecodeError as exc:
        raise Refused(f"{path}: not valid JSON: {exc}") from exc


def load_words(path: Path) -> frozenset[str]:
    return frozenset(
        w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()
    )


_COMPARATOR = "step1_places_compare_captures"


def comparator() -> Any:
    """``scripts/compare_captures.py``, loaded once; contract C8 is judged by its functions. It
    needs ``verbatim_bench`` (``bench/src``) importable; without it a C8 check is refused."""
    if _COMPARATOR in sys.modules:
        return sys.modules[_COMPARATOR]
    spec = importlib.util.spec_from_file_location(_COMPARATOR, _HERE / "compare_captures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_COMPARATOR] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        del sys.modules[_COMPARATOR]
        raise Refused(
            f"confidence refused: contract C8 is judged by scripts/compare_captures.py, which"
            f" cannot be imported here ({exc}); put bench/src on PYTHONPATH"
        ) from exc
    return module


def same_answers(
    path: Path,
    digest: str,
    capture: dict[str, Any],
    side: str,
    named: dict[str, Any],
    against: Path | None,
) -> dict[str, Any]:
    """The identity of a ``--confidence`` capture whose bytes are not the side's capture the
    places record names. Accepted only when contract C8 holds against that capture, judged by
    ``scripts/compare_captures.py``'s ``confidence_equivalence`` (the side's capture taken with
    word confidence off, this one with it on), and when this one's padding is that capture's:
    the confidence is attached to one side, and the side is a padding. The side's capture is
    read from ``against``, or else from the path the record gives, and must be the bytes the
    record names. Refused otherwise, with every reason."""
    cc = comparator()
    judge = cc.confidence_equivalence
    not_named = (
        f"confidence refused: {path} (sha256 {digest}) is not the side {side} capture the places"
        f" record was built from (sha256 {named['sha256']})"
    )
    where = against if against is not None else named.get("path")
    if where is None:
        raise Refused(
            f"{not_named}, and the record gives no path to that capture to judge contract C8"
            " against; give --confidence-against CAPTURE"
        )
    try:
        theirs = cc.load_capture(Path(where))
    except (OSError, ValueError) as exc:
        raise Refused(
            f"{not_named}, and that capture cannot be read at {where} to judge contract C8"
            f" against ({exc}); give --confidence-against CAPTURE"
        ) from exc
    if theirs.sha256 != named["sha256"]:
        raise Refused(
            f"confidence refused: {where} (sha256 {theirs.sha256}) is not the side {side} capture"
            f" the places record names (sha256 {named['sha256']}), so contract C8 cannot be"
            " judged against it"
        )
    ours = cc.Loaded(str(path), digest, capture)
    try:
        problems = judge(theirs, ours)
        if not problems:
            # confidence_equivalence found both captures usable, so neither padding contradicts
            # itself and padding_of returns.
            p_theirs, p_ours = cc.padding_of(theirs.record)[0], cc.padding_of(ours.record)[0]
            if p_theirs != p_ours:
                problems.append(
                    f"its padding is {p_ours!r} and the side {side} capture's is {p_theirs!r}:"
                    " its confidences are of another shape"
                )
    except (KeyError, TypeError, AttributeError) as exc:
        raise Refused(f"{not_named}, and the two cannot be compared as captures ({exc!r})") from exc
    if problems:
        raise Refused(
            f"{not_named}, and it is not shown to be the same answers as that capture (contract"
            " C8): " + "; ".join(problems)
        )
    return {
        "rule": "contract C8, judged by scripts/compare_captures.py's confidence_equivalence:"
        " both digests recomputed and equal, the same recordings in the same order with the same"
        " audio, every setting but the word confidence equal; and the same padding",
        "verdict": cc.CONFIDENCE_EQUIVALENT,
        "against": {"path": str(where), "sha256": theirs.sha256},
        "finals_digest": capture.get("finals_digest"),
    }


def load_confidence_capture(
    path: Path, side: str, record: dict[str, Any], against: Path | None = None
) -> dict[str, Any]:
    """A ``probes/server_frozen_answers.py`` capture's words, as a CONFIDENCE object. A capture
    stamped as taken through test doubles is refused, as a record is (``FAKE_MARKERS``). When the
    places record names the side's capture, the file must be it byte for byte, or be the same
    answers by contract C8 (``same_answers``); ``identity`` says which."""
    capture, digest = load_record(path)
    if not isinstance(capture, dict):
        raise Refused(f"{path}: not a JSON object; not a server capture")
    for marker in FAKE_MARKERS:
        fake = capture.get(marker, False)
        if fake is not False:
            raise Refused(
                f"confidence refused: {path} has {marker} {fake!r}; {FAKE_MARKERS[marker]}, so its"
                " confidences measure nothing"
            )
    key = CAPTURE_OF_SIDE[side]
    named = (record.get("captures") or {}).get(key) or {}
    if named.get("sha256") is None:
        identity = None
    elif named["sha256"] == digest:
        identity = {"rule": f"the same bytes as the places record's {key} capture"}
    else:
        identity = same_answers(path, digest, capture, side, named, against)
    recordings = capture.get("recordings")
    if not isinstance(recordings, dict):
        raise Refused(f"{path}: no recordings mapping; not a server capture")
    words = {rid: entry.get("words") for rid, entry in recordings.items()}
    head = {"side": side, "source": f"{path} sha256 {digest}", "identity": identity}
    # The capture appends the confidence only where the wire carried one, so a capture is read
    # whole: every entry with it, or none. Anything between is refused here, not scored.
    sizes: Counter[Any] = Counter(
        len(e) if isinstance(e, list) else type(e).__name__
        for ws in words.values()
        if isinstance(ws, list)
        for e in ws
    )
    odd = sorted(str(k) for k in sizes if k not in (3, 4))
    if odd:
        raise Refused(
            f"confidence refused: {path} has word entries of {', '.join(odd)} elements; expected"
            " [word, start, end] or [word, start, end, confidence]"
        )
    if not sizes[4]:
        return {
            **head,
            "words": None,
            "unavailable": f"no word entry in {path} carries a confidence",
        }
    if sizes[3]:
        raise Refused(
            f"confidence refused: {path} is mixed: {sizes[4]} word entries carry a confidence and"
            f" {sizes[3]} do not"
        )
    return {**head, "words": words}


COPIED = (
    "source",
    "model",
    "model_revision",
    "machine",
    "capability",
    "torch",
    "nemo",
    "nemo_commit",
    "verbatim_commit",
    "tracked_files_modified",
    "att_context_size",
    "att_context_size_observed",
    "chunk_ms",
    "batch",
    "bucket",
    "execution",
    "use_cuda_graphs",
    "decoder_graphs",
    "decoder_graphs_observed",
    "decoder_graphs_mode_observed",
    "word_confidence",
    "word_confidence_observed",
    "matmul_precision",
    "timing_unit",
    "timing_tuple",
    "gpu_uuid",
    "captures",
    "targets",
    "started",
    "finished",
)


def derive(
    record: dict[str, Any],
    words: frozenset[str],
    normalise: Callable[[str], str] = normalise,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    confidence_at: tuple[str, str] | None = None,
    confidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for marker, meaning in FAKE_MARKERS.items():
        fake = record.get(marker, False)
        if fake is not False:
            raise Refused(
                f"record refused: {marker} is {fake!r}; {meaning}, so it measures nothing"
            )
    captures = record.get("source") == CAPTURES
    if record.get("source") not in (None, CAPTURES):
        raise Refused(f"record refused: source {record.get('source')!r} is not known")
    unit = record.get("timing_unit")
    if captures:
        if unit != "ms":
            raise Refused(
                f"record refused: a server-captures record must say timing_unit 'ms'; it says"
                f" {unit!r}"
            )
        if "bucket" in record and "batch" in record and record["bucket"] != record["batch"]:
            raise Refused(
                f"record refused: bucket {record['bucket']!r} and batch {record['batch']!r}"
                " disagree"
            )
    elif unit not in (None, "s"):
        raise Refused(
            f"record refused: a stock record's timings are seconds; it says timing_unit {unit!r}"
        )
    timing_unit = "ms" if captures else "s"
    if confidence is not None and not captures:
        raise Refused(
            "confidence refused: --confidence reads a server capture; a stock record carries its"
            " confidence in every_recording"
        )
    if confidence_at is not None and confidence_at[1] not in record["runs"].get(
        confidence_at[0], {}
    ).get("arms", {}):
        raise Refused(f"confidence refused: the record has no arm {'/'.join(confidence_at)}")
    runs: dict[str, Any] = {}
    for dtype, run in record["runs"].items():
        verdict = run.get("repeat_verdict")
        pc = run.get("positive_control")
        seen = control_seen(run)
        withheld = None
        if verdict != SAME_SHAPE_VERDICT:
            withheld = (
                f"repeat_verdict is {verdict!r}, not {SAME_SHAPE_VERDICT!r}: the run is not"
                " shown to be run-to-run deterministic, so every score below would include noise"
            )
        elif (against := repeat_contradicts(run.get("repeat"), captures)) is not None:
            withheld = (
                f"repeat_verdict is {SAME_SHAPE_VERDICT!r}, but {against}: the verdict is not"
                " what the run's own repeat counts show, so every score below is withheld"
            )
        arms: dict[str, Any] = {}
        for name, arm in run["arms"].items():
            if captures and name != CAPTURES_ARM:
                raise Refused(
                    f"record refused: server-captures arm {name!r}; expected {CAPTURES_ARM!r}"
                )
            given = confidence if confidence_at == (dtype, name) else None
            if withheld is not None:
                check_arm(arm, record["references"])
                check_every_recording(arm)
                scored: dict[str, Any] = {
                    "sides": SIDES.get(name),
                    "checked": arm["checked"],
                    "text_divergent": arm["text_divergent"],
                    "timing_only_divergent": arm["timing_only_divergent"],
                    "scores_withheld": withheld,
                    "places": None,
                    "flag": None,
                    "confidence": None,
                }
            else:
                scored = score_arm(
                    name,
                    arm,
                    record["references"],
                    words,
                    normalise,
                    timing_unit=timing_unit,
                    draws=draws,
                    seed=f"{seed}/{dtype}",
                    confidence=given,
                    captures=captures,
                )
            scored["served_side"] = "a" if captures else None
            scored["zero_reading"] = zero_reading(name, arm, pc, seen)
            arms[name] = scored
        runs[dtype] = {
            "finished": run.get("finished"),
            "repeat": run.get("repeat"),
            "repeat_verdict": verdict,
            "repeat_verdict_accepted": withheld is None,
            "positive_control": pc,
            "positive_control_seen_in_arms": seen,
            "positive_control_axis": SHAPE_AXIS,
            "decoder_graphs_observed": run.get("decoder_graphs_observed"),
            "word_confidence_observed": run.get("word_confidence_observed"),
            "att_context_size_observed": run.get("att_context_size_observed"),
            "arms": arms,
        }
    return {
        "record_settings": {k: record.get(k) for k in COPIED},
        "complete": "finished" in record,
        "runs": runs,
    }


def parse_confidence_at(text: str | None, record: dict[str, Any]) -> tuple[str, str, str]:
    if record.get("source") != CAPTURES:
        raise Refused(
            "--confidence reads a server capture and goes with a server-captures record; a stock"
            " record carries its confidence in every_recording"
        )
    if text is None:
        runs = list(record.get("runs", {}))
        if len(runs) != 1:
            raise Refused(
                "--confidence-at DTYPE/ARM/SIDE is required when the record has more than one run"
            )
        return runs[0], CAPTURES_ARM, "a"
    parts = text.split("/")
    if len(parts) != 3 or parts[2] not in ("a", "b"):
        raise Refused(f"--confidence-at {text!r} is not DTYPE/ARM/SIDE with SIDE a or b")
    return parts[0], parts[1], parts[2]


def temporary_of(out: Path) -> Path:
    return out.with_name(out.name + ".tmp")


def refuse_out(out: Path, inputs: list[Path]) -> None:
    """Refuse an ``--out`` that is one of the inputs, or that (or its temporary file) exists:
    a derived file is written new, never over anything."""
    target = out.resolve()
    for given in inputs:
        if given.resolve() == target:
            raise Refused(f"--out {out} is the input {given}; an input is never written over")
    for path in (out, temporary_of(out)):
        if os.path.lexists(path):
            raise Refused(f"--out: {path} exists; it is not written over")


def write_new(out: Path, text: str) -> None:
    """Write ``text`` to ``out`` only where nothing is: through a temporary file created
    exclusively, then linked into place, which fails rather than replace a file that appeared
    after ``refuse_out`` looked."""
    tmp = temporary_of(out)
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            fh.write(text)
    except FileExistsError as exc:
        raise Refused(f"--out: {tmp} appeared while scoring; it is not written over") from exc
    try:
        os.link(tmp, out)
    except FileExistsError as exc:
        raise Refused(f"--out: {out} appeared while scoring; it is not written over") from exc
    finally:
        tmp.unlink()


def main(argv: list[str] | None = None) -> int:
    """Score one record; print a refusal as one line on stderr and return ``EXIT_REFUSED``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--wordlist", type=Path, default=DEFAULT_WORDLIST)
    parser.add_argument(
        "--out", type=Path, default=None, help="written new: refused when it exists or is an input"
    )
    parser.add_argument("--confidence", type=Path, default=None)
    parser.add_argument("--confidence-at", default=None, metavar="DTYPE/ARM/SIDE")
    parser.add_argument(
        "--confidence-against",
        type=Path,
        default=None,
        metavar="CAPTURE",
        help="the places record's capture of the --confidence side, when it is no longer at the"
        " path the record gives; it must have the sha256 the record gives",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    serving = parser.add_argument_group("the serving setting the record is compared with")
    serving.add_argument("--serving-chunk-ms", type=int, default=160)
    serving.add_argument("--serving-att-context", default="70,1")
    serving.add_argument("--serving-dtype", default="bfloat16")
    serving.add_argument("--serving-matmul", default="high")
    serving.add_argument("--serving-bucket", type=int, default=None)
    serving.add_argument(
        "--serving-execution",
        choices=("eager", "graph path"),
        default=None,
        help="how the server's encoder step runs (its /readyz 'execution')",
    )
    serving.add_argument(
        "--serving-decoder-graphs",
        choices=("on", "off"),
        default=None,
        help="whether the server runs with --decoder-graphs (the RNNT decoder's CUDA graphs)",
    )
    serving.add_argument("--serving-word-confidence", default="off")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (Refused, OSError) as exc:
        print(f"step1_places: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def run(args: argparse.Namespace) -> int:
    if args.out is not None:
        inputs = [args.record, args.wordlist, args.confidence, args.confidence_against]
        refuse_out(args.out, [p for p in inputs if p is not None])
    record, digest = load_record(args.record)
    words = load_words(args.wordlist)
    confidence_at = confidence = None
    if args.confidence is not None:
        dtype, arm, side = parse_confidence_at(args.confidence_at, record)
        confidence_at = (dtype, arm)
        confidence = load_confidence_capture(
            args.confidence, side, record, against=args.confidence_against
        )
    setting = {
        "chunk_ms": args.serving_chunk_ms,
        "att_context_size": [int(x) for x in args.serving_att_context.split(",")],
        "dtype": args.serving_dtype,
        "matmul_precision": args.serving_matmul,
        "bucket": args.serving_bucket,
        "execution": args.serving_execution,
        "decoder_graphs": args.serving_decoder_graphs,
        "word_confidence": args.serving_word_confidence,
    }
    out = {
        "question": (
            "Where do two transcripts of a recording differ, was either right, what share of"
            " all word errors fall at those places, and does that flag beat the model's own"
            " confidence at the same number of flagged words?"
        ),
        "not_a_row": {
            "setting_differences": setting_differences(record, setting),
            "serving_setting_compared": setting,
            "second_decode": SECOND_DECODE,
        },
        "input": {"path": str(args.record), "sha256": digest},
        "wordlist": {
            "path": str(args.wordlist),
            "sha256": uw.sha256(args.wordlist),
            "entries": len(words),
        },
        "nulls": {"draws": args.draws, "seed": args.seed},
        "rules": "see the module docstring of scripts/step1_places.py",
        **derive(
            record,
            words,
            draws=args.draws,
            seed=args.seed,
            confidence_at=confidence_at,
            confidence=confidence,
        ),
    }
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        write_new(args.out, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
