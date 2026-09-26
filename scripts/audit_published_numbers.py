#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Every number this project publishes in prose, checked against the record it cites.

Run it before any claim goes outward:

    PYTHONPATH=src:bench/src python3 scripts/audit_published_numbers.py

Why it exists. Prose and records drift in one direction only: a measurement gets
superseded and the sentence that quoted it does not. This project has caught that late
four times — a README dividing NVIDIA's ceiling by a withdrawn stream count, a decision
record asserting two numbers "remain what they are" after one had moved, a phrase-book
default set from two readings and a gap, a banner warning of a failure the server's own
evidence contradicted. Each was found by a person re-reading, which is not a method.

What it is not. It does not parse the prose; nothing here can tell that a sentence was
edited. It pins the numbers a human transcribed into documents against the JSON they came
from, so that when a record is replaced the audit fails and names the sentence to fix.
Adding a claim here is part of publishing it.

The white paper from Draft 3 on. The paper keeps a numbers mapping beside it (numbers.tsv:
section, the string as printed, its source, the value there). Every row of it is either a
claim in `draft3` below, keyed exactly as the mapping keys it (section, string, and the
RFC 6901 pointer when the source is the summary), or an entry in `NOT_FROM_A_RECORD` with
the reason no record can hold it (a date, a citation, a pending value, a sentence with no
number). A claim reads the summary rows/exploratory/step1-summary-2026-09-26.json, which
scripts/step1_summary.py derives from the raw records, or the record Draft 2 used. It
compares exactly unless the paper rounds, and then within half the last printed place, with
the rounding stated in the claim; a tolerance without one is refused. Where the paper
states a number in words ("two commits", "about a third"), the claim states the reading it
applies. A claim whose string does not print the value it checks fails as well, so the
string and the value cannot drift apart inside this file. Given the mapping,

    PYTHONPATH=src:bench/src python3 scripts/audit_published_numbers.py --numbers numbers.tsv

also fails on a row that nothing here checks or lists, and on a claim the mapping does not
list, so the audit states what the paper prints now. `--summary` reads another copy of the
summary, which is how a changed number is shown to fail.

Exit 0 when every claim matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from verbatim_bench import constants

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "rows" / "exploratory"
#: Draft 3's one source for the Nemotron figures, derived by scripts/step1_summary.py.
SUMMARY = "step1-summary-2026-09-26.json"

#: Floats agree within this, for binary representation only (0.05145 * 100 is
#: 5.1450000000000005). It is not a rounding allowance: a claim states its rounding.
REPRESENTATION = 1e-9

#: A number as the paper prints one: thousands commas, an optional decimal part, and not the
#: tail of a word or a version ("bfloat16", "a207de6", "2.13.0" give nothing past "2.13").
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

_PARSE = object()  # the published value is the one number the string prints
_AT = object()  # the record is the value at the claim's pointer


def _printed_numbers(printed: str) -> list[float]:
    return [float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(printed)]


def _parse(printed: str) -> int | float:
    found = _NUMBER.findall(printed)
    if len(found) != 1:
        raise ValueError(f"{printed!r} does not print exactly one number; state the value")
    text = found[0].replace(",", "")
    return float(text) if "." in text else int(text)


def _numbers_in(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        return [n for item in value for n in _numbers_in(item)]
    return []


def _shows(printed: str, published: Any) -> bool:
    """Whether the string prints the value checked: every number in it, or the text itself."""
    if isinstance(published, str):
        return published in printed
    shown = set(_printed_numbers(printed))
    return all(n in shown for n in _numbers_in(published))


def _agrees(record: Any, published: Any, tol: float) -> bool:
    if isinstance(record, bool) or isinstance(published, bool):
        return type(record) is type(published) and record == published
    if isinstance(record, (int, float)) and isinstance(published, (int, float)):
        return abs(record - published) <= tol + REPRESENTATION
    if isinstance(record, (list, tuple)) and isinstance(published, (list, tuple)):
        return len(record) == len(published) and all(
            _agrees(r, p, tol) for r, p in zip(record, published, strict=True)
        )
    if isinstance(record, dict) and isinstance(published, dict):
        return record.keys() == published.keys() and all(
            _agrees(record[k], published[k], tol) for k in record
        )
    return record == published


def ptr(*keys: str) -> str:
    """The RFC 6901 pointer to `keys` ("/" in a key is "~1", "~" is "~0")."""
    return "".join("/" + str(k).replace("~", "~0").replace("/", "~1") for k in keys)


def nearest_fraction(x: float, denominators: tuple[int, ...]) -> tuple[int, int]:
    """The fraction p/q nearest `x`, q in `denominators` and 0 < p < q: how prose rounds a
    share into words ("two thirds", "a third")."""
    candidates = {Fraction(p, q) for q in denominators for p in range(1, q)}
    best = min(candidates, key=lambda f: abs(float(f) - x))
    return best.numerator, best.denominator


class Audit:
    def __init__(self, summary: Path | None = None) -> None:
        self.matched = 0
        self.failures: list[str] = []
        #: (section, string as printed, pointer or "") of every Draft 3 claim, as the paper's
        #: numbers mapping keys its rows.
        self.printed_keys: Counter[tuple[str, str, str]] = Counter()
        self.summary_path = summary if summary is not None else ROWS / SUMMARY
        self._summary: Any = None

    def load(self, name: str) -> Any:
        return json.loads((ROWS / name).read_text(encoding="utf-8"))

    @property
    def summary(self) -> Any:
        if self._summary is None:
            self._summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return self._summary

    def at(self, pointer: str) -> Any:
        """The summary's value at an RFC 6901 pointer."""
        node = self.summary
        for part in pointer.split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            node = node[int(part)] if isinstance(node, list) else node[part]
        return node

    def claim(self, where: str, record: Any, published: Any, tol: float = 0.0) -> None:
        """`record` is read from the JSON; `published` is what a document says."""
        if _agrees(record, published, tol):
            self.matched += 1
        else:
            self.failures.append(f"{where}: record={record!r} published={published!r}")

    def printed(
        self,
        section: str,
        printed: str,
        source: str,
        published: Any = _PARSE,
        *,
        record: Any = _AT,
        tol: float = 0.0,
        rounds: str | None = None,
        rule: str | None = None,
    ) -> None:
        """One number the white paper prints, keyed as its numbers mapping keys the row.

        `source` is the pointer into the summary the row names, or "" when the row's source
        is a Draft 2 record or a derivation. `published` defaults to the one number `printed`
        shows; `record` to the summary's value at `source`. `tol` is allowed only with
        `rounds`, the rounding the paper applied, and is half its last place. `rule` states
        how words are read as a value; without one, the string must print the value checked.
        """
        if (tol > 0) != (rounds is not None):
            raise ValueError(f"{section} {printed!r}: a tolerance is the rounding it states")
        if published is _PARSE:
            published = _parse(printed)
        if record is _AT:
            try:
                record = self.at(source)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                record = f"<nothing at {source}: {exc!r}>"
        where = f"paper {section}: {printed!r}"
        if rounds is not None:
            where += f" (rounded {rounds})"
        if rule is not None:
            where += f" [{rule}]"
        self.printed_keys[(section, printed, source)] += 1
        if rule is None and not _shows(printed, published):
            self.failures.append(f"{where}: checks {published!r}, which the string does not print")
            return
        self.claim(where, record, published, tol)


#: Below this, more than half of what the phrase list puts into a transcript is wrong. No
#: recall figure makes that a usable setting, so the knee is the best weight that stays
#: above it rather than the weight with the highest recall outright.
KNEE_MIN_PRECISION = 0.5


def knee(rare: dict[str, Any]) -> float:
    """The shipped weight a sweep supports: highest recall among the weights whose precision
    has not collapsed.

    This previously maximised recall and broke ties on false accepts. That was harmless on the
    2026-09-15 rarity row, where the highest-recall weight was also the sane one, and wrong in
    general: on the 2026-09-20 unseen-name row the highest recall belongs to 2.0, which buys
    0.046 recall for 12.8x the false accepts and drops precision from 0.772 to 0.220. A rule
    that picks that weight is not describing a knee.
    """
    usable = [
        arm
        for arm in rare["arms"]
        if arm["boost"] is not None and arm["terms"]["precision"] >= KNEE_MIN_PRECISION
    ]
    if not usable:
        raise SystemExit("no weight in this sweep keeps precision above the floor")
    best = max(usable, key=lambda arm: (arm["terms"]["recall"], -arm["terms"]["false_accepts"]))
    return best["boost"]


def per_level_divergences(document: dict[str, Any]) -> dict[str, int]:
    return {
        slot: len(
            {
                d["stream_id"]
                for d in document["divergences"]
                if d["against"] == "1" and d["level"] == slot
            }
        )
        for slot in ("32a", "32b", "max")
    }


def best_per_seed(document: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for rung in document["rungs"]:
        if rung.get("passed"):
            seed = rung["seed"] % 100
            out[seed] = max(out.get(seed, 0), rung["n"])
    return out


def audit(summary: Path | None = None) -> Audit:
    a = Audit(summary)

    # --- README and DR-0016: the capacity correction -------------------------------
    fixed = a.load("ladder-b300-fixed-dr0016-2026-09-15.json")
    ragged = a.load("ladder-b300-ragged-dr0016-2026-09-15.json")
    a.claim("DR-0016 fixed S", fixed["s"], 124)
    a.claim("DR-0016 fixed criterion", fixed["ending_criterion"], "integrity:refused")
    a.claim("DR-0016 ragged S", ragged["s"], 77)
    a.claim("DR-0016 ragged criterion", ragged["ending_criterion"], "integrity:refused")
    top = sorted((r for r in fixed["rungs"] if r["passed"]), key=lambda r: -r["n"])[:4]
    p95s = [round(r["p95_ms"], 1) for r in top]
    a.claim("README p95 floor at the top rungs", min(p95s), 256.7, 0.05)
    a.claim("README p95 ceiling at the top rungs", max(p95s), 285.3, 0.05)
    at_124 = next(r for r in fixed["rungs"] if r["n"] == 124 and r["passed"])
    a.claim("README WER at 124", round(at_124["wer_vs_batch1"], 4), 0.0136, 0.0002)

    # --- README: the ratio that replaced the withdrawn one -------------------------
    ceiling = a.load("nemo-ceiling-b300-bf16-2026-09-13.json")["arms"]["batch128_graphed"]
    a.claim("README graphed ceiling", ceiling["rtfx_median"], 143.75)
    a.claim("README ceiling's highest run", ceiling["rtfx_max"], 148.97)
    a.claim("README bar at the median", round(0.8 * ceiling["rtfx_median"], 1), 115.0, 0.05)
    a.claim("README bar at the ceiling's max", round(0.8 * ceiling["rtfx_max"], 1), 119.2, 0.05)
    a.claim("README ratio", round(124 / ceiling["rtfx_median"], 3), 0.863, 0.0005)

    # --- DR-0015 and the phrase book: the weight curve -----------------------------
    rare = a.load("rare-terms-b300-2026-09-15.json")
    arms = {("bare" if arm["boost"] is None else arm["boost"]): arm for arm in rare["arms"]}
    published_curve = [
        ("bare", 0.911, 3, 0.0739),
        (1.0, 0.945, 17, 0.0737),
        (2.0, 0.942, 105, 0.0911),
        (4.0, 0.921, 591, 0.2543),
        (10.0, 0.438, 1867, 0.8175),
    ]
    for weight, recall, false_accepts, wer in published_curve:
        arm = arms[weight]
        a.claim(f"DR-0015 w={weight} recall", round(arm["terms"]["recall"], 3), recall, 0.0005)
        a.claim(f"DR-0015 w={weight} false accepts", arm["terms"]["false_accepts"], false_accepts)
        a.claim(f"DR-0015 w={weight} WER", round(arm["wer"], 4), wer, 0.00005)
    a.claim("DR-0015 bare missed", arms["bare"]["terms"]["misses"], 26)
    a.claim("DR-0015 missed at the knee", arms[1.0]["terms"]["misses"], 16)
    a.claim(
        "DR-0015 occurrences recovered",
        arms[1.0]["terms"]["hits"] - arms["bare"]["terms"]["hits"],
        10,
    )
    a.claim("DR-0015 terms in the set", len(rare["term_set"]["terms"]), 256)

    # The shipped default must be the knee the sweep found, not a number someone liked.
    book = json.loads((ROOT / "corpora" / "phrasebooks" / "domains-v1.json").read_text())
    a.claim("shipped phrase-book weight is the measured knee", book["boost"], knee(rare))

    # --- DR-0015: the invariance arms ----------------------------------------------
    a.claim(
        "DR-0015 fixed+biasing verdict",
        a.load("invariance-biasing-b300-2026-09-14.json")["verdict"],
        "invariant",
    )
    a.claim(
        "DR-0015 ragged+biasing divergences",
        per_level_divergences(a.load("invariance-biasing-ragged-b300-2026-09-14.json")),
        {"32a": 117, "32b": 117, "max": 89},
    )

    # --- DR-0015: what biasing buys on the population it is aimed at (full corpus) ---
    unseen = a.load("rare-terms-unseen-full-b300-2026-09-20.json")
    a.claim("DR-0015 unseen run is usable", unseen["usable"], True)
    a.claim("DR-0015 unseen list reached the decoder", unseen["reached_the_decoder"], True)
    a.claim("DR-0015 unseen corpus size", unseen["corpus"]["utterances"], 2939)
    a.claim("DR-0015 unseen rule", unseen["term_set"]["rule"]["kind"], "unseen")
    a.claim("DR-0015 unseen term count", len(unseen["term_set"]["terms"]), 256)
    unseen_arms = {
        ("bare" if arm["boost"] is None else arm["boost"]): arm for arm in unseen["arms"]
    }
    for weight, recall, hits, misses, false_accepts, precision, wer in (
        ("bare", 0.458, 140, 166, 12, 0.921, 0.0710),
        (1.0, 0.709, 217, 89, 64, 0.772, 0.0719),
        (2.0, 0.755, 231, 75, 820, 0.220, 0.0929),
        (4.0, 0.703, 215, 91, 5887, 0.035, 0.2710),
    ):
        terms = unseen_arms[weight]["terms"]
        a.claim(f"DR-0015 unseen w={weight} recall", round(terms["recall"], 3), recall, 0.0005)
        a.claim(f"DR-0015 unseen w={weight} hits", terms["hits"], hits)
        a.claim(f"DR-0015 unseen w={weight} misses", terms["misses"], misses)
        a.claim(f"DR-0015 unseen w={weight} false accepts", terms["false_accepts"], false_accepts)
        a.claim(
            f"DR-0015 unseen w={weight} precision", round(terms["precision"], 3), precision, 0.0005
        )
        a.claim(
            f"DR-0015 unseen w={weight} WER", round(unseen_arms[weight]["wer"], 4), wer, 0.00005
        )

    # The headline: the model gets fewer than half of these right unaided, and the knee
    # recovers 46% of the gap at 0.68 false accepts per occurrence recovered.
    bare_t, knee_t = unseen_arms["bare"]["terms"], unseen_arms[1.0]["terms"]
    recovered = knee_t["hits"] - bare_t["hits"]
    extra_false = knee_t["false_accepts"] - bare_t["false_accepts"]
    a.claim("DR-0015 unseen occurrences recovered at the knee", recovered, 77)
    a.claim(
        "DR-0015 unseen share of the gap closed",
        round(recovered / bare_t["misses"] * 100),
        46,
    )
    a.claim(
        "DR-0015 unseen false accepts per occurrence recovered",
        round(extra_false / recovered, 2),
        0.68,
        0.005,
    )
    # And the knee rule must still choose the shipped weight on THIS row, where a
    # recall-maximising rule would have chosen 2.0.
    a.claim("DR-0015 the knee rule picks the shipped weight on the unseen row", knee(unseen), 1.0)
    a.claim(
        "DR-0015 a recall-maximising rule would have picked 2.0 here",
        max(
            (arm for arm in unseen["arms"] if arm["boost"] is not None),
            key=lambda arm: arm["terms"]["recall"],
        )["boost"],
        2.0,
    )
    # The residual is substitutions, so the beam-search trigger is NOT fired.
    for weight, deletions, substitutions in (("bare", 1, 165), (1.0, 1, 88)):
        misses = unseen_arms[weight]["misses_by_kind"]
        a.claim(f"DR-0015 unseen w={weight} residual deletions", misses["deletion"], deletions)
        a.claim(
            f"DR-0015 unseen w={weight} residual substitutions",
            misses["substitution"],
            substitutions,
        )

    # --- DR-0017: the bucket is the capacity knob, and 128 is its optimum ----------
    b256 = a.load("ladder-b300-fixed-bucket256-2026-09-15.json")
    a.claim("DR-0017 bucket-256 S", b256["s"], 19)
    a.claim("DR-0017 bucket-256 criterion", b256["ending_criterion"], "latency")
    a.claim("DR-0017 bucket-256 per seed", best_per_seed(b256), {14: 46, 15: 46, 16: 19})
    a.claim("DR-0017 bucket-128 per seed", best_per_seed(fixed), {14: 124, 15: 126, 16: 124})
    # Capacity is bounded above by the bucket, so the optimum is where they meet. At 128
    # the server reaches 97% of its bucket; at 256, 18%. That is the whole argument.
    a.claim("DR-0017 bucket 128 is at its fixed point", round(124 / 128, 2), 0.97, 0.005)
    ragged_repeat = a.load("ladder-b300-ragged-repeat-2026-09-15.json")
    a.claim("DR-0017 the ragged arm did not reproduce", ragged_repeat["s"], 0)

    # --- DR-0017: where the step's time goes, PROFILED (the derived figure is withdrawn)
    phase = a.load("step-phase-speech-b300-2026-09-16.json")
    by_batch = {arm["batch"]: arm for arm in phase["arms"]}
    for batch, step, enc, dec, rest in (
        (32, 26.2, 14.5, 7.0, 4.7),
        (128, 40.2, 15.1, 12.3, 12.8),
        (256, 56.2, 14.6, 18.0, 23.6),
    ):
        arm = by_batch[batch]
        a.claim(f"DR-0017 step at batch {batch}", round(arm["step_median_ms"], 1), step, 0.05)
        a.claim(
            f"DR-0017 encoder at batch {batch}", round(arm["encoder"]["median_ms"], 1), enc, 0.05
        )
        a.claim(
            f"DR-0017 decoder at batch {batch}", round(arm["decoder"]["median_ms"], 1), dec, 0.05
        )
        a.claim(f"DR-0017 rest at batch {batch}", round(arm["rest_ms"], 1), rest, 0.05)

    span = 256 - 32

    def marginal(key: str) -> float:
        lo, hi = by_batch[32], by_batch[256]
        if key in ("encoder", "decoder"):
            return (hi[key]["median_ms"] - lo[key]["median_ms"]) / span
        return (hi[key] - lo[key]) / span

    a.claim("DR-0017 marginal step", round(marginal("step_median_ms"), 3), 0.134, 0.0005)
    # The finding: the encoder does not scale with occupancy at all.
    a.claim("DR-0017 marginal encoder", round(marginal("encoder"), 3), 0.001, 0.0005)
    a.claim("DR-0017 marginal decoder", round(marginal("decoder"), 3), 0.049, 0.0005)
    a.claim("DR-0017 marginal rest", round(marginal("rest_ms"), 3), 0.084, 0.0005)
    # And the step cost does not explain the 124 cap: 40 ms against a 112 ms budget.
    a.claim(
        "DR-0017 bucket-128 step is well inside the budget",
        round(by_batch[128]["step_median_ms"], 1) < 112.0,
        True,
    )

    # --- DR-0017: CUDA graphs buy fixed cost, in the decoder as in the encoder -----
    dg = a.load("step-phase-speech-decgraph-b300-2026-09-16.json")
    dg_batch = {arm["batch"]: arm for arm in dg["arms"]}
    a.claim("DR-0017 decoder-graph run declares the flag", dg["decoder_graphs"], True)
    for batch, off, on in ((32, 7.0, 3.7), (128, 12.3, 9.5), (256, 18.0, 16.4)):
        a.claim(
            f"DR-0017 decoder off at batch {batch}",
            round(by_batch[batch]["decoder"]["median_ms"], 1),
            off,
            0.05,
        )
        a.claim(
            f"DR-0017 decoder on at batch {batch}",
            round(dg_batch[batch]["decoder"]["median_ms"], 1),
            on,
            0.05,
        )
    # The finding: the saving is roughly constant, so it is launch overhead and not per-row.
    savings = [
        by_batch[b]["decoder"]["median_ms"] - dg_batch[b]["decoder"]["median_ms"]
        for b in (32, 128, 256)
    ]
    a.claim("DR-0017 the decoder-graph saving does not grow with batch", max(savings) < 4.0, True)
    a.claim(
        "DR-0017 invariance survives decoder graphs",
        a.load("invariance-decgraph-b300-2026-09-16.json")["verdict"],
        "invariant",
    )

    # --- DR-0017: where a FINAL's latency goes, and the gap that section left open --
    decomp = a.load("latency-decomposition-b300-2026-09-20.json")
    arms = {arm["bucket"]: arm for arm in decomp["arms"]}
    for bucket, total, wait, step, edge, overhead in (
        (
            256,
            (167.5, 257.5, 478.4),
            (81.2, 155.9, 244.2),
            (53.6, 62.3, 376.9),
            (22.9, 28.2, 31.1),
            (2.0, 14.5, 243.7),
        ),
        (
            128,
            (160.7, 237.7, 360.1),
            (82.5, 156.3, 258.4),
            (46.1, 55.6, 59.4),
            (26.0, 33.0, 50.2),
            (2.2, 3.0, 55.5),
        ),
    ):
        terms = arms[bucket]["terms"]
        for name, published in (
            ("total_ms", total),
            ("wait_ms", wait),
            ("step_ms", step),
            ("edge_ms", edge),
            ("overhead_ms", overhead),
        ):
            for slot, value in zip(("p50", "p95", "p99"), published, strict=True):
                a.claim(
                    f"DR-0017 {name} {slot} at bucket {bucket}",
                    round(terms[name][slot], 1),
                    value,
                    0.05,
                )
        # The join is only trustworthy if nothing fell out of it. A row with unjoined or
        # impossible sessions is a row whose decomposition was computed on a subset it
        # did not name, which is how this probe's first run produced a 91-second wait.
        arm = arms[bucket]
        a.claim(f"DR-0017 bucket {bucket} joined every session", arm["sessions_unjoined"], 0)
        a.claim(f"DR-0017 bucket {bucket} no impossible wait", arm["sessions_impossible_wait"], 0)
        # And the five terms must close on the total, or one of them is absorbing the others.
        a.claim(
            f"DR-0017 bucket {bucket} residual closes",
            round(terms["rest_ms"]["p95"], 1),
            0.0,
            0.05,
        )

    # The finding: at p95 the grid wait is most of the latency, and it does not move with
    # the bucket. That is DR-0012's phase offset, recovered by a different instrument.
    at256 = arms[256]["at_p95_session"]
    a.claim(
        "DR-0017 the wait is 69% of the p95 latency at bucket 256",
        round(at256["wait_ms"] / at256["total_ms"] * 100),
        69,
    )
    a.claim(
        "DR-0017 the wait does not depend on the bucket",
        abs(arms[256]["terms"]["wait_ms"]["p95"] - arms[128]["terms"]["wait_ms"]["p95"]) < 2.0,
        True,
    )
    # And what separates the buckets is the step's TAIL, not its median.
    for bucket, ratio in ((256, 7.0), (128, 1.3)):
        terms = arms[bucket]["terms"]["step_ms"]
        a.claim(
            f"DR-0017 step p99/p50 at bucket {bucket}",
            round(terms["p99"] / terms["p50"], 1),
            ratio,
            0.05,
        )

    # --- DR-0016 and the README: the A6000 is NOT withdrawn ------------------------
    a6000 = a.load("ladder-a6000-bf16-eager-2026-09-13-all-criteria.json")
    a.claim(
        "A6000 failing rungs failed on latency, so DR-0016 does not reach them",
        {r.get("first_failing_criterion") for r in a6000["rungs"] if not r["passed"]},
        {"latency"},
    )
    a.claim("A6000 certification", a6000["s"], 0)

    # --- The white paper and the plan: the stock-pipeline probe --------------------
    # Counts are read from the probes' own records, not from the derived one, so a
    # stale derived file cannot vouch for them.
    stock_b300 = a.load("stock-divergence-b300-2026-09-11.json")["runs"]
    stock_a6000 = a.load("stock-divergence-a6000-bf16-2026-09-10.json")["arms"]
    stock_a6000_fp32 = a.load("stock-divergence-a6000-fp32-2026-09-10.json")["arms"]
    for where, record, published in (
        ("A6000 bf16 varying", stock_a6000["uncontrolled"], 287),
        ("A6000 bf16 equal-length", stock_a6000["controlled"], 328),
        ("A6000 fp32 varying", stock_a6000_fp32["uncontrolled"], 6),
        ("A6000 fp32 equal-length", stock_a6000_fp32["controlled"], 6),
        ("B300 bf16 varying", stock_b300["bfloat16"]["ragged"], 264),
        ("B300 bf16 equal-length", stock_b300["bfloat16"]["equalised"], 295),
        ("B300 fp32 varying", stock_b300["float32"]["ragged"], 1),
        ("B300 fp32 equal-length", stock_b300["float32"]["equalised"], 1),
    ):
        a.claim(f"paper §5.1 {where} changed", len(record["divergences"]), published)
        a.claim(f"paper §5.1 {where} checked", record["checked"], 2939)
    # Draft 2's abstract rates ("about 9% to 11%") stand in Draft 3's abstract, second
    # paragraph, and are checked there with the rest of Draft 3's numbers (`draft3`).

    # --- ...and the words that changed (scripts/unstable_words.py) ------------------
    words = a.load("unstable-words-2026-09-24.json")
    word_arms = words["arms"]
    eq = word_arms["b300_bf16_equalised"]
    a.claim("paper §5.4 places in the B300 equal-length arm", eq["places"], 328)
    a.claim("paper §5.4 alone right", eq["alone_right"], 119)
    a.claim("paper §5.4 batch right", eq["batch_right"], 96)
    a.claim("paper §5.4 neither right", eq["neither_right"], 113)
    a.claim("paper §5.4 both right is impossible", eq["both_right"], 0)
    a.claim("paper §5.4 sign test", eq["alone_vs_batch_sign_test_p"], 0.13, 0.005)
    a.claim("paper §5.4 WER alone %", round(eq["wer_alone"] * 100, 2), 11.45, 0.005)
    a.claim("paper §5.4 WER in batch %", round(eq["wer_in_batch"] * 100, 2), 12.14, 0.005)
    for arm, net in (
        ("b300_bf16_equalised", 43),
        ("b300_bf16_ragged", 29),
        ("a6000_bf16_equalised", -6),
        ("a6000_bf16_varying", -3),
    ):
        a.claim(
            f"paper §5.4 net extra errors in batch, {arm}",
            word_arms[arm]["net_extra_errors_in_batch"],
            net,
        )
    oov = eq["out_of_dictionary"]
    a.claim(
        "paper §5.4 absent from the word list, all reference words %",
        round(oov["rate_all_reference_words"] * 100, 1),
        3.3,
        0.05,
    )
    a.claim(
        "paper §5.4 absent from the word list, at the places %",
        round(oov["rate_at_places"] * 100, 1),
        19.6,
        0.05,
    )
    ratios = [
        arm["out_of_dictionary"]["rate_at_places"]
        / arm["out_of_dictionary"]["rate_all_reference_words"]
        for arm in word_arms.values()
    ]
    # Draft 2 also said "about six times" in its abstract; Draft 3 says it in §5.4 only.
    a.claim(
        "plan and paper §5.4: 'about six times', 'between 5.9 and 6.4', lowest arm",
        round(min(ratios), 1),
        5.9,
        0.05,
    )
    a.claim(
        "plan and paper §5.4: 'about six times', 'between 5.9 and 6.4', highest arm",
        round(max(ratios), 1),
        6.4,
        0.05,
    )
    # Draft 2's abstract gave the B300 shares (29%, 36%, 34%); Draft 3 removed them from the
    # abstract, and §5.4 and Figure 3 still print them. The "plan:" claims below bound all
    # four arms, which is not the same statement, so the B300 arm is pinned here as printed,
    # each to a whole percent.
    for key, share in (("alone_right", 36), ("batch_right", 29), ("neither_right", 34)):
        a.claim(
            f"paper §5.4 and Figure 3, B300 equal-length: {key} {share}% (to a whole percent)",
            eq[key] / eq["places"] * 100,
            share,
            0.5,
        )
    shares = {
        key: [arm[key] / arm["places"] for arm in word_arms.values()]
        for key in ("alone_right", "batch_right", "neither_right")
    }
    a6000_eq = word_arms["a6000_bf16_equalised"]
    a.claim("plan: alone right, low end %", round(min(shares["alone_right"]) * 100), 29)
    a.claim("plan: alone right, high end %", round(max(shares["alone_right"]) * 100), 36)
    a.claim("plan: batch right on the B300 %", round(eq["batch_right"] / eq["places"] * 100), 29)
    a.claim(
        "plan: batch right on the A6000 %",
        round(a6000_eq["batch_right"] / a6000_eq["places"] * 100),
        33,
    )
    a.claim("plan: neither right, low end %", round(min(shares["neither_right"]) * 100), 34)
    a.claim("plan: neither right, high end %", round(max(shares["neither_right"]) * 100), 38)
    a.claim(
        "plan: which version does better is within chance on every arm",
        all(arm["alone_vs_batch_sign_test_p"] > 0.05 for arm in word_arms.values()),
        True,
    )
    a.claim(
        "paper §5.4 unstable in both B300 arms", words["unstable_in_both_arms"]["b300_bf16"], 233
    )
    # "79% of the 295 and 88% of the 264": those recordings over each B300 arm's changed ones.
    for arm, changed, share in (("equalised", 295, 79), ("ragged", 264, 88)):
        a.claim(
            f"paper §5.4 unstable in both B300 arms, {share}% of the {changed} "
            "(to a whole percent)",
            words["unstable_in_both_arms"]["b300_bf16"]
            / len(stock_b300["bfloat16"][arm]["divergences"])
            * 100,
            share,
            0.5,
        )
    a.claim(
        "paper §5.4 places involving a negation or a number",
        len(eq["negation_or_number_places"]),
        6,
    )
    a.claim(
        "paper §5.4 place shapes",
        eq["place_shapes"],
        {"one_for_one": 223, "multi_word": 61, "one_side_empty": 44},
    )
    a.claim("paper §5.4 absent words overall", oov["reference_words_absent"], 208)
    a.claim("paper §5.4 reference words overall", oov["reference_words"], 6236)
    a.claim("paper §5.4 absent words at the places", oov["reference_words_at_places_absent"], 85)
    a.claim("paper §5.4 reference words at the places", oov["reference_words_at_places"], 434)
    a.claim("paper §5.4 A6000 places", a6000_eq["places"], 350)
    a.claim("paper §5.4 A6000 batch right", a6000_eq["batch_right"], 114)
    a.claim("paper §5.4 A6000 alone right", a6000_eq["alone_right"], 105)

    # --- The white paper, Table 1: every server run and its control -----------------
    # Streams differing from concurrency 1, (at the highest level, at any level).
    control_arm = a.load("invariance-control-arm-b300-2026-09-14.json")
    a.claim(
        "paper Table 1 constant-occupancy runs ran at bucket 38",
        control_arm["shared"]["bucket"],
        38,
    )
    a.claim(
        "paper Table 1 constant control, bucket 38",
        per_level_divergences(control_arm["raw"]["ragged"]),
        {"32a": 118, "32b": 118, "max": 66},
    )
    a.claim(
        "paper Table 1 constant control at any level",
        control_arm["arms"]["ragged"]["streams_differing_from_concurrency_1"],
        128,
    )
    a.claim(
        "paper Table 1 constant fixed, bucket 38", control_arm["arms"]["fixed"]["divergences"], 0
    )
    churn = a.load("invariance-churn-b300-2026-09-14.json")
    a.claim("paper Table 1 churn runs at bucket 128", churn["shared"]["bucket"], 128)
    a.claim(
        "paper Table 1 churn control",
        churn["arms"]["ragged_churned"]["streams_differing_per_comparison"],
        {"32a_vs_1": 118, "32b_vs_1": 118, "max_vs_1": 88},
    )
    a.claim(
        "paper §5.2 highest level, constant",
        churn["arms"]["ragged_constant"]["max_level_concurrency"],
        38,
    )
    a.claim(
        "paper §5.2 highest level, churned",
        churn["arms"]["ragged_churned"]["max_level_concurrency"],
        42,
    )
    # The churned union is stated only in the record's own reading; the audit pins that
    # reading rather than a count the record does not carry.
    a.claim(
        "paper Table 1 churn control at any level (128 -> 120)",
        "(128 -> 120)" in churn["reads"][2],
        True,
    )
    a.claim("paper Table 1 churn fixed", churn["arms"]["fixed_churned"]["verdict"], "invariant")
    for where, name, levels, any_level in (
        ("vocabulary 2.0", "invariance-biasing-ragged-b300-2026-09-14.json", (117, 117, 89), 120),
        (
            "vocabulary 1.0",
            "invariance-biasing-ragged-w1-b300-2026-09-15.json",
            (114, 114, 84),
            117,
        ),
    ):
        document = a.load(name)
        a.claim(
            f"paper Table 1 {where} control",
            per_level_divergences(document),
            dict(zip(("32a", "32b", "max"), levels, strict=True)),
        )
        a.claim(
            f"paper Table 1 {where} control at any level",
            len({d["stream_id"] for d in document["divergences"] if d["against"] == "1"}),
            any_level,
        )
    digests = {}
    for where, name in (
        ("constant, bucket 128", "invariance-b300-bf16-2026-09-13.json"),
        ("vocabulary 2.0", "invariance-biasing-b300-2026-09-14.json"),
        ("vocabulary 1.0", "invariance-biasing-w1-b300-2026-09-15.json"),
        ("decoder graphs", "invariance-decgraph-b300-2026-09-16.json"),
    ):
        document = a.load(name)
        a.claim(
            f"paper Table 1 fixed {where}",
            (document["verdict"], len(document["divergences"])),
            ("invariant", 0),
        )
        level_digests = {level["digest"] for level in document["levels"]}
        a.claim(f"paper Table 1 fixed {where}: one digest across levels", len(level_digests), 1)
        digests[where] = level_digests.pop()
    a.claim(
        "paper §5.2 the frozen output depends on the bucket",
        digests["constant, bucket 128"] != control_arm["raw"]["fixed"]["levels"][0]["digest"],
        True,
    )
    a.claim(
        "paper §5.2 the frozen output depends on decoder graphs",
        digests["constant, bucket 128"] != digests["decoder graphs"],
        True,
    )
    # Table 1's caption: zero of 256 bounds the per-stream rate below about 1.2% at 95%
    # confidence, the one-sided bound 1 - 0.05 ** (1 / n) for n streams, to one decimal place.
    n_streams = churn["shared"]["streams_per_level"]
    a.claim("paper Table 1 caption: streams per run", n_streams, 256)
    a.claim(
        "paper Table 1 caption: the 95% bound on the rate, 'about 1.2%' (to one decimal place)",
        (1 - 0.05 ** (1 / n_streams)) * 100,
        1.2,
        0.05,
    )
    # §5.2's two examples of the unpadded control's divergence, alone against the highest
    # level (38 streams, "paper §5.2 highest level, constant" above).
    top = [
        d
        for d in control_arm["raw"]["ragged"]["divergences"]
        if d["level"] == "max" and d["against"] == "1"
    ]
    a.claim(
        "paper §5.2 one stream read 'they' alone and 'there' at 38 streams",
        any(d["kind"] == "text" and (d["left"], d["right"]) == ("they", "there") for d in top),
        True,
    )
    a.claim(
        "paper §5.2 a word moved from 720-800 ms to 800-880 ms",
        any(
            d["kind"] == "words"
            and re.fullmatch(r"\('[\w']+', 720, 800\)", d["left"]) is not None
            and re.fullmatch(r"\('[\w']+', 800, 880\)", d["right"]) is not None
            for d in top
        ),
        True,
    )
    # §4.2: the churned runs' highest level admits on a triangle wave "every 20 seconds".
    churned = [
        level["churn_period_s"]
        for name in (
            "invariance-biasing-b300-2026-09-14.json",
            "invariance-biasing-w1-b300-2026-09-15.json",
            "invariance-biasing-ragged-b300-2026-09-14.json",
            "invariance-biasing-ragged-w1-b300-2026-09-15.json",
            "invariance-decgraph-b300-2026-09-16.json",
        )
        for level in a.load(name)["levels"]
        if level["slot"] == "max"
    ]
    a.claim(
        "paper §4.2 the churn period, seconds",
        sorted({churn["shared"]["churn_period_s"], *churned}),
        [20.0],
    )
    a.claim(
        "paper §5.2 the churned fixed run froze the same output as the constant one",
        digests["constant, bucket 128"].startswith(
            churn["arms"]["fixed_churned"]["digests"]["max"]
        ),
        True,
    )

    # --- The white paper, §5.3: the reference point and what padding costs ----------
    ref = a.load("nemo-ceiling-b300-bf16-2026-09-13.json")
    a.claim("paper §5.3 reference chunk", ref["chunk_ms"], 160)
    eager, graphed = ref["arms"]["batch128_eager"], ref["arms"]["batch128_graphed"]
    a.claim("paper §5.3 reference eager", eager["rtfx_median"], 149.48)
    a.claim(
        "paper §5.3 reference runs", (len(eager["rtfx_runs"]), len(graphed["rtfx_runs"])), (5, 5)
    )
    a.claim("paper §5.3 eager spread %", round(eager["spread_pct_of_median"]), 16)
    a.claim("paper §5.3 graphed spread %", round(graphed["spread_pct_of_median"]), 10)
    a.claim("paper §5.3 share of eager", round(124 / eager["rtfx_median"], 2), 0.83)
    a.claim("paper §5.3 share of graphed", round(124 / graphed["rtfx_median"], 2), 0.86)
    a.claim("paper §5.3 unpadded seeds", best_per_seed(ragged), {14: 77, 15: 126, 16: 96})
    a.claim(
        "paper Table 2 refusals began at 126 streams",
        min(r["n"] for r in fixed["rungs"] if r.get("sessions_refused")),
        126,
    )
    price = a.load("invariance-price-b300-2026-09-14.json")["result"]
    a.claim(
        "paper §5.3 matched pair, padded against unpadded",
        (price["sustained_streams_fixed"], price["sustained_streams_ragged"]),
        (42, 59),
    )
    noise = {arm["batch"]: arm for arm in a.load("step-phase-noise-b300-2026-09-16.json")["arms"]}
    a.claim("paper §5.3 256-row step, noise", round(noise[256]["step_median_ms"], 1), 45.7, 0.05)

    # --- The white paper, Figure 2: the two sessions drawn ---------------------------
    for bucket, streams, parts in (
        (128, 124, (237.7, 146.6, 58.4, 30.2, 2.4)),
        (256, 46, (257.5, 178.3, 53.2, 24.2, 1.8)),
    ):
        arm = arms[bucket]
        a.claim(f"paper Figure 2 streams at bucket {bucket}", arm["streams"], streams)
        session = arm["at_p95_session"]
        for key, value in zip(
            ("total_ms", "wait_ms", "step_ms", "edge_ms", "overhead_ms"), parts, strict=True
        ):
            a.claim(f"paper Figure 2 bucket {bucket} {key}", round(session[key], 1), value, 0.05)

    # --- The white paper, Draft 3 ------------------------------------------------------
    draft3(a)
    return a


#: The digest the six padded Nemotron captures share (the summary's finals_digests key).
FROZEN_DIGEST = "45e1ddba293689ca005cdf2b7d84e8ffd8125a4e0d929f6c0d0ae74c53bd5a7f"

#: Rows of the paper's numbers mapping that no record can hold, with the reason. They are
#: listed rather than checked; `--numbers` fails on a row that is neither this nor a claim.
NOT_FROM_A_RECORD: dict[tuple[str, str], str] = {
    ("header", "Draft 3"): "the draft's own number",
    ("header", "26 September 2026"): "the draft's date",
    ("§1 p4 (changed)", "September 2025"): "the date of citation [1]",
    ("§1 p5 (new)", "Draft 3"): "the draft's own number",
    ("§2 (changed)", "2025; June 2025; 100%; 2.13.0; 1.10.0"): "dates, a quotation and "
    "release numbers from citations [5]-[7]",
    (
        "§6.6 p2",
        "(no number) 'On the A6000 it stopped there each time'",
    ): "no number; the runbook's logs, which are not among the records",
    ("§2 LLMs (changed)", "September 2025"): "the date of citation [1]",
    ("§2 LLMs (changed)", "August 2026"): "the date of citation [4]",
    ("§2 LLMs (changed)", "15\u2013324%"): "a figure citation [4] reports (an en dash)",
    ("§9 p1 (changed)", "16-frame"): "citation [9]'s reproduction, as the pull request states it",
    ("§9 p1 (changed)", "20-frame"): "citation [9]'s reproduction, as the pull request states it",
    ("§9 p1 (changed)", "0.428"): "citation [9]: 'Maximum absolute difference 0.428'",
    ("§9 p1 (changed)", "September 2026"): "the date citation [8] was merged",
    (
        "§9 crash",
        "(no number) 'one LibriSpeech test-other recording', "
        "'He has one son, and 'tis the finest boy'",
    ): "no number; a quotation of the recording's text",
    ("Reproducibility", "<commit>"): "pending: the commit that will carry the records",
    ("References [1]", "10 September 2025"): "citation",
    ("References [2]", "22 September 2025"): "citation",
    ("References [14]", "SLT 2022"): "citation",
    ("References [14]", "arXiv:2212.08703"): "citation",
    ("footer", "Draft 3"): "the draft's own number",
}

WHOLE = "to a whole percent"
ONE_PLACE = "to one decimal place of a percent"
MS = "to a whole millisecond"
TWO_PLACES = "to two decimal places"


def draft3(a: Audit) -> None:
    """The white paper, Draft 3: each number its numbers mapping lists, keyed as the mapping
    keys it. Sections are the mapping's; the pointers are into the summary."""
    P, at, S = a.printed, a.at, a.summary
    SB = ("stock", "nemotron-bf16-1120", "runs", "bfloat16", "arms")
    SF = ("stock", "nemotron-fp32-1120", "runs", "float32", "arms")
    SH = ("stock", "hybrid-bf16-1120", "runs", "bfloat16", "arms")
    S16 = ("stock", "nemotron-bf16-160-high-n1024", "runs", "bfloat16", "arms")
    SC = ("stock_160ms_scored", "runs", "bfloat16", "arms", "ragged")
    RVF = ("server", "ragged_vs_fixed")
    RVR = ("server", "ragged_vs_ragged", "a207de6")
    CAP = ("server", "captures")
    PB = ("flips_vs_confidence", "modes", "paper-best")
    NS = ("flips_vs_confidence", "modes", "nemo-shipped")
    PA = ("flips_vs_confidence", "places_analysis")
    FROZ = ("server", "frozen")
    C22_32, C22_8, C22_R = (
        "server-22e8406-fixed-c32.json.gz",
        "server-22e8406-fixed-c8.json.gz",
        "server-22e8406-ragged-c32.json.gz",
    )
    CA2_32, CA2_8, CA2_PB, CA2_NS, CA2_R, CA2_RR = (
        "server-a207de6-fixed-c32.json.gz",
        "server-a207de6-fixed-c8.json.gz",
        "server-a207de6-fixed-paperbest-c32.json.gz",
        "server-a207de6-fixed-nemoshipped-c32.json.gz",
        "server-a207de6-ragged-c32.json.gz",
        "server-a207de6-ragged-c32-repeat.json.gz",
    )
    MODEL = ptr("server", "setting", "model")
    model_name = at(MODEL).split("/")[-1]
    captures = S["server"]["captures"]
    frozen = S["server"]["frozen"]
    padded = at(ptr("server", "finals_digests", FROZEN_DIGEST))
    unpadded = [name for name, capture in captures.items() if capture["padding"] == "ragged"]
    stock_runs = {
        "nemotron-bf16-1120": "bfloat16",
        "nemotron-fp32-1120": "float32",
        "hybrid-bf16-1120": "bfloat16",
        "nemotron-bf16-160-high-n1024": "bfloat16",
    }
    table3_runs = ("nemotron-bf16-1120", "nemotron-fp32-1120", "hybrid-bf16-1120")

    def stock_arms(run: str) -> dict[str, Any]:
        return S["stock"][run]["runs"][stock_runs[run]]["arms"]

    def pct(pointer: str) -> float:
        return at(pointer) * 100

    def percent(section: str, text: str, pointer: str, *, rounded: bool) -> None:
        """A percentage as printed, of the fraction at `pointer`. `rounded` says whether the
        paper printed fewer places than the summary holds; if so the rounding is to the
        places printed, and the tolerance half the last of them."""
        places = len(text.rstrip("%").partition(".")[2])
        P(
            section,
            text,
            pointer,
            _parse(text),
            record=pct(pointer),
            tol=0.5 * 10**-places if rounded else 0.0,
            rounds=(
                f"to {places} decimal place{'s' * (places > 1)} of a percent" if places else WHOLE
            )
            if rounded
            else None,
        )

    def millis(section: str, text: str, pointer: str) -> None:
        """Milliseconds as printed, rounded to whole ones from the summary's float."""
        P(section, text, pointer, tol=0.5, rounds=MS)

    def both_commits(*keys: str) -> tuple[Any, Any]:
        return tuple(at(ptr(*RVF, commit, *keys)) for commit in ("22e8406", "a207de6"))

    def concurrencies(names: list[str]) -> list[int]:
        return sorted({captures[name]["concurrency_configured"] for name in names})

    def confidence_on_is_the_padded_answer(pointer: str) -> bool:
        # Read through the row's own pointer for one mode, and the same fields for the other.
        verdicts = [at(pointer)] + [
            mode["verdict"]
            for name, mode in S["server"]["confidence_equivalence"].items()
            if ptr("server", "confidence_equivalence", name, "verdict") != pointer
        ]
        on = [mode["on"] for mode in S["server"]["confidence_equivalence"].values()]
        return (
            len(verdicts) == 2
            and all(v == "IDENTICAL IN TEXT AND TIMINGS" for v in verdicts)
            and all(captures[name]["finals_digest"] == FROZEN_DIGEST for name in on)
        )

    # --- Draft 2's records, for Draft 2 numbers standing in changed sentences --------
    a6000_bf16 = a.load("stock-divergence-a6000-bf16-2026-09-10.json")
    a6000_fp32 = a.load("stock-divergence-a6000-fp32-2026-09-10.json")
    b300 = a.load("stock-divergence-b300-2026-09-11.json")
    bf16_arms = [
        a6000_bf16["arms"]["uncontrolled"],
        a6000_bf16["arms"]["controlled"],
        b300["runs"]["bfloat16"]["ragged"],
        b300["runs"]["bfloat16"]["equalised"],
    ]
    fp32_a6000 = [a6000_fp32["arms"]["uncontrolled"], a6000_fp32["arms"]["controlled"]]
    fp32_b300 = [b300["runs"]["float32"]["ragged"], b300["runs"]["float32"]["equalised"]]
    bf16_changed = [len(arm["divergences"]) for arm in bf16_arms]  # 287, 328, 264, 295
    bf16_pct = [len(arm["divergences"]) / arm["checked"] * 100 for arm in bf16_arms]

    control_arm = a.load("invariance-control-arm-b300-2026-09-14.json")
    churn = a.load("invariance-churn-b300-2026-09-14.json")
    gates = {
        name: a.load(name)
        for name in (
            "invariance-b300-bf16-2026-09-13.json",
            "invariance-biasing-b300-2026-09-14.json",
            "invariance-biasing-w1-b300-2026-09-15.json",
            "invariance-decgraph-b300-2026-09-16.json",
            "invariance-biasing-ragged-b300-2026-09-14.json",
            "invariance-biasing-ragged-w1-b300-2026-09-15.json",
        )
    }
    ragged_w2 = gates["invariance-biasing-ragged-b300-2026-09-14.json"]
    ragged_w1 = gates["invariance-biasing-ragged-w1-b300-2026-09-15.json"]
    # Table 1's six padded rows, streams changed in each.
    fixed_rows = [
        len(gates["invariance-b300-bf16-2026-09-13.json"]["divergences"]),
        control_arm["arms"]["fixed"]["divergences"],
        sum(churn["arms"]["fixed_churned"]["streams_differing_per_comparison"].values()),
        len(gates["invariance-biasing-b300-2026-09-14.json"]["divergences"]),
        len(gates["invariance-biasing-w1-b300-2026-09-15.json"]["divergences"]),
        len(gates["invariance-decgraph-b300-2026-09-16.json"]["divergences"]),
    ]
    # Table 1's four control rows, at the highest level and at any level. The churned run's
    # any-level count exists only in its record's own reading, as in Table 1's claim above.
    control_highest = [
        per_level_divergences(control_arm["raw"]["ragged"])["max"],
        churn["arms"]["ragged_churned"]["streams_differing_per_comparison"]["max_vs_1"],
        per_level_divergences(ragged_w2)["max"],
        per_level_divergences(ragged_w1)["max"],
    ]
    churn_union = re.search(r"\(128 -> (\d+)\)", churn["reads"][2])
    control_any = [
        control_arm["arms"]["ragged"]["streams_differing_from_concurrency_1"],
        int(churn_union.group(1)) if churn_union else None,
        len({d["stream_id"] for d in ragged_w2["divergences"] if d["against"] == "1"}),
        len({d["stream_id"] for d in ragged_w1["divergences"] if d["against"] == "1"}),
    ]
    streams_per_level = sorted(
        {level["streams"] for gate in gates.values() for level in gate["levels"]}
        | {level["streams"] for level in control_arm["arms"]["fixed"]["levels"].values()}
        | {level["streams"] for level in control_arm["arms"]["ragged"]["levels"].values()}
        | {churn["shared"]["streams_per_level"]}
    )
    levels = (
        [level["concurrency"] for gate in gates.values() for level in gate["levels"]]
        + [level["concurrency"] for level in control_arm["arms"]["fixed"]["levels"].values()]
        + [arm["max_level_concurrency"] for arm in churn["arms"].values()]
    )
    ladder = a.load("ladder-b300-fixed-dr0016-2026-09-15.json")
    ladder_256 = a.load("ladder-b300-fixed-bucket256-2026-09-15.json")
    # The budget as the harness sets it (one chunk plus X_MS), and as the ladders' own
    # rungs show it: every pass at or under it, every latency failure over it.
    budget = ladder["config"]["server"]["chunk_ms"] + constants.X_MS
    rungs = ladder["rungs"] + ladder_256["rungs"]
    budget_observed = all(r["p95_ms"] <= budget for r in rungs if r.get("passed")) and all(
        r["p95_ms"] > budget for r in rungs if r.get("first_failing_criterion") == "latency"
    )
    words = a.load("unstable-words-2026-09-24.json")["arms"]
    place_shares = [
        arm[key] / arm["places"] * 100
        for arm in words.values()
        for key in ("alone_right", "batch_right")
    ]

    # --- Abstract ------------------------------------------------------------------------
    sec = "abstract p1"
    P(sec, "nemotron-speech-streaming-en-0.6b", MODEL, model_name, record=model_name)
    P(sec, "2,939", ptr(*FROZ, "a207de6", "fixed_c32_vs_fixed_c8", "identical"))
    P(sec, "2,939", ptr(*FROZ, "22e8406", "fixed_c32_vs_fixed_c8", "identical"))
    P(sec, "32", ptr(*CAP, C22_32, "observed_peak_in_flight"))
    P(sec, "8", ptr(*CAP, C22_8, "observed_peak_in_flight"))
    P(
        sec,
        "two separate runs of the server",
        ptr(*FROZ),
        (2, True),
        record=(len(at(ptr(*FROZ))), all(v["frozen"] for v in frozen.values())),
        rule="'two': the commits the summary judges frozen, a run of the server at each, all of "
        "them frozen",
    )
    confidence_verdict = ptr("server", "confidence_equivalence", "paper-best", "verdict")
    P(
        sec,
        "word confidence switched on",
        confidence_verdict,
        True,
        record=confidence_on_is_the_padded_answer(confidence_verdict),
        rule="both confidence modes identical to confidence off, with the padded digest",
    )
    P(sec, "147", ptr(*RVR, "word_level"))
    P(sec, "133", ptr(*RVR, "punctuation_only"))
    P(sec, "935", ptr(*RVR, "timing_only"))

    sec = "abstract p2"
    P(sec, "264 to 328", "", (264, 328), record=(min(bf16_changed), max(bf16_changed)))
    P(
        sec,
        "2,939",
        "",
        [2939],
        record=sorted({arm["checked"] for arm in bf16_arms + fp32_a6000 + fp32_b300}),
    )
    P(
        sec,
        "about 9% to 11%",
        "",
        (9, 11),
        record=(min(bf16_pct), max(bf16_pct)),
        tol=0.5,
        rounds=WHOLE,
    )
    P(sec, "32", "", [32], record=sorted({a6000_bf16["batch"], a6000_fp32["batch"], b300["batch"]}))
    P(
        sec,
        "0.2%",
        "",
        [0.2, 0.2],
        record=[len(arm["divergences"]) / arm["checked"] * 100 for arm in fp32_a6000],
        tol=0.05,
        rounds=ONE_PLACE,
    )
    P(
        sec,
        "0.03%",
        "",
        [0.03, 0.03],
        record=[len(arm["divergences"]) / arm["checked"] * 100 for arm in fp32_b300],
        tol=0.005,
        rounds="to two decimal places of a percent",
    )
    P(sec, "112", ptr(*SB, "ragged", "word_level"))
    P(sec, "119", ptr(*SB, "equalised", "word_level"))
    P(sec, "297", ptr(*SH, "ragged", "word_level"))
    P(sec, "313", ptr(*SH, "equalised", "word_level"))
    P(
        sec,
        "none at float32",
        ptr(*SF, "ragged", "word_level"),
        (0, 0),
        record=(at(ptr(*SF, "ragged", "word_level")), at(ptr(*SF, "equalised", "word_level"))),
        rule="'none': no word-level change in either unpadded arm",
    )
    P(sec, "0 of 256", "", (0, [256]), record=(max(fixed_rows), streams_per_level))
    P(sec, "1 to 42", "", (1, 42), record=(min(levels), max(levels)))
    # Draft 2 printed "66 to 88" here, which left out Table 1's own 89 (the weight-2.0
    # vocabulary control) while "117 to 128" beside it counts that row. Draft 3 prints 89.
    P(
        sec,
        "66 to 89 of 256",
        "",
        (66, 89, [256]),
        record=(min(control_highest), max(control_highest), streams_per_level),
    )
    P(sec, "117 to 128", "", (117, 128), record=(min(control_any), max(control_any)))
    seeds = best_per_seed(ladder).values()
    P(sec, "124 to 126", "", (124, 126), record=(min(seeds), max(seeds)))
    P(sec, "310 ms", "", (310, True), record=(budget, budget_observed))
    P(sec, "128-row bucket", ptr("server", "setting", "bucket"))
    P(
        sec,
        "8 streams",
        ptr(*CAP, C22_8, "ticks_over_budget_during_run"),
        (8, 8, True),
        record=(
            captures[C22_8]["concurrency_configured"],
            captures[CA2_8]["concurrency_configured"],
            at(ptr(*CAP, C22_8, "ticks_over_budget_during_run")) > 0
            and captures[CA2_8]["ticks_over_budget_during_run"] > 0,
        ),
    )

    sec = "abstract p3"
    flips_places = ptr("flips_vs_confidence", "places")
    places_commit = at(flips_places).removeprefix("places-").removesuffix(".json.gz")
    P(sec, "a207de6", flips_places, "a207de6", record=places_commit)
    P(sec, "62", ptr(*PA, "alone_right"))
    P(sec, "189", ptr(*PA, "places"))
    P(sec, "54", ptr(*PA, "batch_right"))
    P(sec, "73", ptr(*PA, "neither_right"))
    percent(sec, "84%", ptr(*PB, "confidence", "precision"), rounded=True)
    percent(sec, "66%", ptr(*PB, "flips", "precision"), rounded=True)
    ns_conf, pb_conf, flips = (
        at(ptr(*NS, "confidence")),
        at(ptr(*PB, "confidence")),
        at(ptr(*PB, "flips")),
    )
    P(
        sec,
        "did about as well",
        ptr(*NS, "confidence", "precision"),
        True,
        record=all(
            abs(ns_conf[k] - flips[k]) < abs(pb_conf[k] - ns_conf[k])
            for k in ("precision", "word_errors_removable")
        ),
        rule="NeMo-shipped nearer the flips than paper-best, in precision and in errors removable",
    )

    # --- §1 --------------------------------------------------------------------------
    sec = "§1 p5 (new)"
    P(sec, "nemotron-speech-streaming-en-0.6b", MODEL, model_name, record=model_name)
    P(sec, "2,939", ptr(*SB, "ragged", "checked"))
    P(sec, "1,120 ms", ptr("stock", "nemotron-bf16-1120", "settings", "chunk_ms"))
    P(sec, "1,024", ptr(*S16, "ragged", "checked"))
    P(sec, "160 ms", ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "chunk_ms"))
    sec = "§1 contributions"
    P(
        sec,
        "two precisions on two GPU generations",
        "",
        (2, 2),
        record=(
            len({a6000_bf16["compute_dtype"], a6000_fp32["compute_dtype"], *b300["runs"]}),
            len({a6000_bf16["machine"], a6000_fp32["machine"], b300["machine"]}),
        ),
        rule="'two', 'two': the dtypes and the machines of Draft 2's stock records",
    )
    P(
        sec,
        "six padded captures",
        ptr("server", "finals_digests", FROZEN_DIGEST),
        (6, ["fixed"]),
        record=(len(padded), sorted({captures[name]["padding"] for name in padded})),
        rule="'six': the captures with the padded digest, all of them padded",
    )
    P(
        sec,
        "two concurrencies",
        ptr(*CAP, CA2_8, "concurrency_configured"),
        2,
        record=len(concurrencies(padded)),
        rule="'two': the concurrencies among the six padded captures",
    )
    P(
        sec,
        "two commits",
        ptr(*FROZ),
        (2, True),
        record=(len(at(ptr(*FROZ))), all(v["frozen"] for v in frozen.values())),
        rule="'two': the commits the summary judges frozen, all of them frozen",
    )
    P(
        sec,
        "three confidence settings",
        ptr(*CAP, CA2_PB, "word_confidence"),
        3,
        record=len({captures[name]["word_confidence"] for name in padded}),
        rule="'three': the word-confidence settings among the six padded captures",
    )
    P(sec, "147", ptr(*RVR, "word_level"))
    P(sec, "2,939", ptr(*RVR, "recordings"))
    P(sec, "935", ptr(*RVR, "timing_only"))

    # --- §4.4 ------------------------------------------------------------------------
    sec = "§4.4"
    P(sec, "3.1.0+cf724ac33", ptr("server", "setting", "nemo_version"), "3.1.0+cf724ac33")
    P(sec, "2.11.0+cu128", ptr("server", "setting", "torch_version"), "2.11.0+cu128")
    P(
        sec,
        "ebe59e5a",
        ptr("server", "setting", "revision"),
        "ebe59e5a",
        record=at(ptr("server", "setting", "revision"))[:8],
    )
    batch = ptr("stock", "nemotron-bf16-1120", "settings", "batch")
    P(sec, "32", batch)
    P(sec, "32", batch)  # the fixed arm's equal-length batch of 32
    P(sec, "31", batch, 31, record=at(batch) - 1, rule="the neighbours: the batch less the target")
    P(sec, "1,120 ms", ptr("stock", "nemotron-bf16-1120", "settings", "chunk_ms"))
    P(sec, "[70,13]", ptr("stock", "nemotron-bf16-1120", "settings", "att_context_size"), [70, 13])
    P(sec, "highest", ptr("stock", "nemotron-bf16-1120", "settings", "matmul_precision"), "highest")
    P(sec, "160 ms", ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "chunk_ms"))
    P(
        sec,
        "[70,1]",
        ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "att_context_size"),
        [70, 1],
    )
    P(
        sec,
        "high",
        ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "matmul_precision"),
        "high",
    )
    P(sec, "1,024", ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "targets"))
    P(
        sec,
        "8",
        ptr("stock", "nemotron-bf16-1120", "runs", "bfloat16", "repeat", "checked"),
        [8],
        record=sorted(
            {S["stock"][run]["runs"][dt]["repeat"]["checked"] for run, dt in stock_runs.items()}
        ),
    )
    P(
        sec,
        "every repeat did",
        ptr("stock", "nemotron-bf16-1120", "runs", "bfloat16", "repeat_verdict"),
        True,
        record=all(
            S["stock"][run]["runs"][dt]["repeat_verdict"]
            == "run-to-run identical in the same shape"
            and len(set(S["stock"][run]["runs"][dt]["repeat"].values())) == 1
            for run, dt in stock_runs.items()
        ),
        rule="every stock run: alone and batch repeats identical on every recording repeated",
    )
    P(
        sec,
        "2,939",
        ptr(*CAP, CA2_32, "recordings"),
        [2939],
        record=sorted({capture["recordings"] for capture in captures.values()}),
    )
    P(sec, "32", ptr(*CAP, CA2_32, "concurrency_configured"))
    P(sec, "8", ptr(*CAP, CA2_8, "concurrency_configured"))
    P(sec, "160 ms", ptr("server", "setting", "chunk_ms"))
    P(sec, "128-row bucket", ptr("server", "setting", "bucket"))
    P(
        sec,
        "22e8406",
        ptr(*CAP, C22_32, "commit"),
        "22e8406",
        record=at(ptr(*CAP, C22_32, "commit"))[:7],
    )
    P(
        sec,
        "a207de6",
        ptr(*CAP, CA2_32, "commit"),
        "a207de6",
        record=at(ptr(*CAP, CA2_32, "commit"))[:7],
    )
    P(
        sec,
        "twice more",
        ptr(*CAP, CA2_PB, "word_confidence"),
        2,
        record=sum(
            1
            for name in padded
            if captures[name]["commit"].startswith("a207de6")
            and captures[name]["word_confidence"] != "off"
        ),
        rule="'twice': the padded a207de6 captures with word confidence on",
    )
    P(
        sec,
        "32 streams ... at each commit, and again at a207de6",
        ptr("server", "finals_digests"),
        ([32], ["22e8406", "a207de6", "a207de6"]),
        record=(
            concurrencies(unpadded),
            sorted(captures[name]["commit"][:7] for name in unpadded),
        ),
    )

    # --- §5.1, bullet 3 (its last sentence changed) ------------------------------------
    sec = "§5.1 bullet 3 (changed)"
    P(
        sec,
        "32; 287; 328; 264; 295; 1; 32; 32",
        "",
        (32, 287, 328, 264, 295, 32, 32),
        record=(a6000_bf16["batch"], *bf16_changed, b300["batch"], b300["batch"]),
        rule="the 1 is the lone decode, a batch of one by the probe's design; each 32 is the "
        "batch both GPUs' records ran",
    )
    P(
        sec,
        "changed no recording",
        ptr(*SB, "fixed", "text_divergent"),
        0,
        record=max(
            arms["fixed"][k]
            for arms in map(stock_arms, stock_runs)
            for k in ("text_divergent", "timing_only")
        ),
        rule="'no': every stock run's fixed arm, in text and in timings",
    )

    # --- §6 --------------------------------------------------------------------------
    sec = "§6"
    P(sec, "nemotron-speech-streaming-en-0.6b", MODEL, model_name, record=model_name)
    size = re.search(r"-(\d+(?:\.\d+)?)b$", at(MODEL))
    P(sec, "0.6B parameters", MODEL, 0.6, record=float(size.group(1)) if size else None)
    P(
        sec,
        "writes punctuation and capitals",
        ptr(*SB, "ragged", "punctuation_only"),
        True,
        record=at(ptr(*SB, "ragged", "punctuation_only")) > 0,
        rule="some change is in punctuation or capitals alone",
    )

    # --- §6.1 and Table 3 ------------------------------------------------------------
    P("§6.1 heading", "1,120 ms", ptr("stock", "nemotron-bf16-1120", "settings", "chunk_ms"))
    table3 = {
        ("ragged", "varying"): (("112", "99", "1,038"), ("0", "0", "2"), ("297", "0", "1,139")),
        ("equalised", "equal-length"): (
            ("119", "110", "1,103"),
            ("0", "0", "2"),
            ("313", "0", "1,208"),
        ),
        ("fixed", "fixed"): (("0", "0", "0"), ("0", "0", "0"), ("0", "0", "0")),
    }
    names = ("Nemotron bf16", "Nemotron fp32", "older model bf16")
    for (arm, label), rows in table3.items():
        for run, name, cells in zip(table3_runs, names, rows, strict=True):
            base = ("stock", run, "runs", stock_runs[run], "arms", arm)
            for key, cell in zip(
                ("word_level", "punctuation_only", "timing_only"), cells, strict=True
            ):
                P(f"Table 3 {name} {label}", cell, ptr(*base, key))
    sec = "Table 3 caption"
    P(sec, "2,939", ptr(*SB, "ragged", "checked"))
    P(
        sec,
        "[70,13]",
        ptr("stock", "nemotron-bf16-1120", "settings", "att_context_size"),
        [[70, 13]],
        record=[
            list(x)
            for x in {tuple(S["stock"][run]["settings"]["att_context_size"]) for run in table3_runs}
        ],
    )
    P(
        sec,
        "1,120 ms",
        ptr("stock", "hybrid-bf16-1120", "settings", "chunk_ms"),
        [1120],
        record=sorted({S["stock"][run]["settings"]["chunk_ms"] for run in table3_runs}),
    )
    P(
        sec,
        "32",
        ptr("stock", "hybrid-bf16-1120", "settings", "batch"),
        [32],
        record=sorted({S["stock"][run]["settings"]["batch"] for run in table3_runs}),
    )
    for text, pointer, key in (
        (
            "highest",
            ptr("stock", "nemotron-fp32-1120", "settings", "matmul_precision"),
            "matmul_precision",
        ),
        ("3.1.0+cf724ac33", ptr("stock", "hybrid-bf16-1120", "settings", "nemo"), "nemo"),
        ("2.11.0+cu128", ptr("stock", "hybrid-bf16-1120", "settings", "torch"), "torch"),
    ):
        values = {S["stock"][run]["settings"][key] for run in table3_runs}
        P(sec, text, pointer, text, record=values.pop() if len(values) == 1 else sorted(values))
    P(
        sec,
        "ae981433",
        ptr("stock", "hybrid-bf16-1120", "settings", "model_revision"),
        "ae981433",
        record=at(ptr("stock", "hybrid-bf16-1120", "settings", "model_revision"))[:8],
    )
    P(
        sec,
        "none of its changes was in punctuation or capitals alone",
        ptr(*SH, "ragged", "punctuation_only"),
        0,
        record=max(arm["punctuation_only"] for arm in stock_arms("hybrid-bf16-1120").values()),
        rule="'none': the older model's punctuation-only count, every arm",
    )
    P(
        sec,
        "8",
        ptr("stock", "hybrid-bf16-1120", "runs", "bfloat16", "repeat", "alone_identical"),
        [8],
        record=sorted(
            {
                v
                for run in table3_runs
                for v in S["stock"][run]["runs"][stock_runs[run]]["repeat"].values()
            }
        ),
    )

    sec = "§6.1 p1"
    P(sec, "112", ptr(*SB, "ragged", "word_level"))
    P(sec, "119", ptr(*SB, "equalised", "word_level"))
    percent(sec, "3.8%", ptr(*SB, "ragged", "word_level_rate"), rounded=True)
    percent(sec, "4.0%", ptr(*SB, "equalised", "word_level_rate"), rounded=True)
    P(sec, "211", ptr(*SB, "ragged", "text_divergent"))
    P(sec, "229", ptr(*SB, "equalised", "text_divergent"))
    P(sec, "1,038", ptr(*SB, "ragged", "timing_only"))
    P(sec, "1,103", ptr(*SB, "equalised", "timing_only"))

    def both_arms(base: tuple[str, ...], *keys: str) -> tuple[Any, Any]:
        return tuple(at(ptr(*base, arm, *keys)) for arm in ("ragged", "equalised"))

    P(
        sec,
        "80 ms",
        ptr(*SB, "ragged", "timing_shift_ms", "median"),
        (80, 80),
        record=both_arms(SB, "timing_shift_ms", "median"),
    )
    p90 = ptr(*SB, "ragged", "timing_shift_ms", "p90")
    P(
        sec,
        "90th percentile",
        p90,
        90,
        record=90 if isinstance(at(p90), int) and p90.endswith("/p90") else None,
        rule="the quantile the figure is: the summary's p90 field",
    )
    P(sec, "320 ms", p90, (320, 320), record=both_arms(SB, "timing_shift_ms", "p90"))
    P(
        sec,
        "720 ms",
        ptr(*SB, "equalised", "timing_shift_ms", "max"),
        (720, 720),
        record=both_arms(SB, "timing_shift_ms", "max"),
    )
    percent(sec, "5.145%", ptr(*SB, "ragged", "wer", "wer_a"), rounded=False)
    percent(sec, "5.154%", ptr(*SB, "ragged", "wer", "wer_b"), rounded=False)
    sec = "§6.1 p2"
    P(sec, "2", ptr(*SF, "ragged", "timing_only"), (2, 2), record=both_arms(SF, "timing_only"))
    P(
        sec,
        "80 ms",
        ptr(*SF, "ragged", "timing_shift_ms", "max"),
        (80, 80, 80, 80),
        record=both_arms(SF, "timing_shift_ms", "min") + both_arms(SF, "timing_shift_ms", "max"),
    )
    P(
        sec,
        "no recording's text changed",
        ptr(*SF, "ragged", "text_divergent"),
        0,
        record=max(both_arms(SF, "text_divergent")),
        rule="'no': neither unpadded float32 arm",
    )

    # --- §6.2 ------------------------------------------------------------------------
    sec = "§6.2"
    P(sec, "297", ptr(*SH, "ragged", "word_level"))
    P(sec, "313", ptr(*SH, "equalised", "word_level"))
    P(sec, "287", "", record=bf16_changed[0])
    P(sec, "328", "", record=bf16_changed[1])
    P(sec, "matmul precision high", "", "high", record=a6000_bf16["matmul_precision"])
    P(sec, "1,139", ptr(*SH, "ragged", "timing_only"))
    P(sec, "1,208", ptr(*SH, "equalised", "timing_only"))
    P(sec, "3,360 ms", ptr(*SH, "equalised", "timing_shift_ms", "max"))
    percent(sec, "5.1%", ptr(*SB, "ragged", "wer", "wer_a"), rounded=True)
    percent(sec, "6.4%", ptr(*SH, "ragged", "wer", "wer_a"), rounded=True)

    # --- §6.3 ------------------------------------------------------------------------
    chunk_160 = ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "chunk_ms")
    P("§6.3 heading", "160 ms", chunk_160)
    sec = "§6.3 p1"
    P(sec, "160 ms", ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "chunk_ms"))
    P(
        sec,
        "[70,1]",
        ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "att_context_size"),
        [70, 1],
    )
    P(
        sec,
        "high",
        ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "matmul_precision"),
        "high",
    )
    P(
        sec,
        "1,024",
        ptr(*S16, "ragged", "checked"),
        [1024],
        record=sorted(
            {arm["checked"] for arm in stock_arms("nemotron-bf16-160-high-n1024").values()}
        ),
    )
    P(sec, "51", ptr(*S16, "ragged", "word_level"), (51, 51), record=both_arms(S16, "word_level"))
    P(sec, "49", ptr(*S16, "ragged", "punctuation_only"))
    P(sec, "44", ptr(*S16, "equalised", "punctuation_only"))
    P(sec, "375", ptr(*S16, "ragged", "timing_only"))
    P(sec, "369", ptr(*S16, "equalised", "timing_only"))
    P(
        sec,
        "The fixed arm again changed nothing",
        ptr(*S16, "fixed", "text_divergent"),
        (0, 0),
        record=(at(ptr(*S16, "fixed", "text_divergent")), at(ptr(*S16, "fixed", "timing_only"))),
        rule="'nothing': the 160 ms fixed arm, in text and in timings",
    )
    sec = "§6.3 p2"
    flag = (*SC, "flag_a")
    P(sec, "40", ptr(*flag, "not_right"))
    P(
        sec,
        "54",
        ptr(*flag, "spans"),
        (54, 54),
        record=(at(ptr(*flag, "spans")), at(ptr(*SC, "places"))),
    )
    percent(sec, "74%", ptr(*flag, "precision"), rounded=True)
    null = ptr(*flag, "null", "random_recordings", "not_right_rate_mean")
    percent(sec, "7.1%", null, rounded=True)
    P(sec, "1,000", ptr(*flag, "null", "draws"))
    p_value = ptr(*flag, "null", "random_recordings", "p_value_not_right")
    draws = at(ptr(*flag, "null", "draws"))
    P(
        sec,
        "no draw of 1,000 reached 40",
        p_value,
        (0, 1000, 40),
        record=(round(at(p_value) * (draws + 1) - 1), draws, at(ptr(*flag, "not_right"))),
        rule="'no draw': p = (1 + draws reaching it) / (draws + 1) leaves none",
    )
    P(
        sec,
        "word confidence off",
        ptr(*SC, "confidence", "a"),
        True,
        record=at(ptr(*SC, "confidence", "a")) == "confidence constant"
        and S["stock"]["nemotron-bf16-160-high-n1024"]["settings"]["word_confidence"] == "off",
        rule="the 160 ms run's settings, and its confidence constant on the side scored",
    )

    # --- §6.4 ------------------------------------------------------------------------
    sec = "§6.4"
    P(sec, "64", ptr("replay", "targets"))
    replayed = [
        run for run, entry in S["stock"].items() if entry["record"] == at(ptr("replay", "replays"))
    ]
    P(
        sec,
        "1,120 ms bfloat16 run",
        ptr("replay", "replays"),
        (1120, ["bfloat16"]),
        record=(
            S["stock"][replayed[0]]["settings"]["chunk_ms"] if len(replayed) == 1 else None,
            sorted(S["stock"][replayed[0]]["runs"]) if len(replayed) == 1 else None,
        ),
    )
    P(
        sec,
        "reproduced it wherever the earlier record allows a comparison",
        ptr("replay", "reproduced"),
        (True, []),
        record=(at(ptr("replay", "reproduced")), at(ptr("replay", "differences"))),
        rule="the summary's replay verdict, with no difference listed: compare_stock_replay "
        "compares every target's texts, and word timings where the earlier record keeps them",
    )
    P(sec, "2", ptr("replay", "arms", "bfloat16/ragged", "text_divergent"))
    P(sec, "16", ptr("replay", "arms", "bfloat16/ragged", "timing_only_divergent"))
    P(sec, "4", ptr("replay", "arms", "bfloat16/equalised", "text_divergent"))
    P(sec, "19", ptr("replay", "arms", "bfloat16/equalised", "timing_only_divergent"))
    P(
        sec,
        "none in the fixed arm",
        ptr("replay", "arms", "bfloat16/fixed", "text_divergent"),
        (0, 0),
        record=tuple(
            at(ptr("replay", "arms", "bfloat16/fixed", k))
            for k in ("text_divergent", "timing_only_divergent")
        ),
        rule="'none': the replay's fixed arm, in text and in timings",
    )

    # --- Figure 4 --------------------------------------------------------------------
    sec = "Figure 4"
    P(sec, "2,939", ptr(*FROZ, "a207de6", "fixed_c32_vs_fixed_c8", "recordings"))
    P(sec, "32", ptr(*CAP, CA2_32, "observed_peak_in_flight"))
    P(sec, "8", ptr(*CAP, CA2_8, "observed_peak_in_flight"))
    for name in (C22_R, CA2_R, CA2_RR):
        P(sec, "32", ptr(*CAP, name, "observed_peak_in_flight"))  # an unpadded bar's label
    P(
        sec,
        "2,939 identical",
        ptr(*FROZ, "22e8406", "fixed_c32_vs_fixed_c8", "identical"),
        (2939, 2939),
        record=tuple(
            frozen[c]["fixed_c32_vs_fixed_c8"]["identical"] for c in ("22e8406", "a207de6")
        ),
    )
    for commit, counts in (
        ("22e8406", ("1,538", "1,071", "330")),
        ("a207de6", ("1,562", "1,028", "349")),
    ):
        places = at(ptr(*RVF, commit, "places"))
        P(
            sec,
            commit,
            ptr(*RVF, commit, "places"),
            commit,
            record=places.removeprefix("places-").removesuffix(".json.gz"),
        )
        for key, cell in zip(("identical", "timing_only", "text_differs"), counts, strict=True):
            P(sec, cell, ptr(*RVF, commit, key))
    for key, cell in zip(
        ("identical", "timing_only", "text_differs"), ("1,724", "935", "280"), strict=True
    ):
        P(sec, cell, ptr(*RVR, key))
    P(sec, "160 ms", ptr("server", "setting", "chunk_ms"))
    P(sec, "bucket 128", ptr("server", "setting", "bucket"))
    P(
        sec,
        "six padded captures share one digest",
        ptr("server", "finals_digests", FROZEN_DIGEST),
        (6, 1),
        record=(len(padded), len({captures[name]["finals_digest"] for name in padded})),
        rule="'six', 'one': the captures under the padded digest, and their digests",
    )
    digests = at(ptr("server", "finals_digests"))
    P(
        sec,
        "each of the three unpadded captures has its own",
        ptr("server", "finals_digests"),
        (3, True),
        record=(
            len(unpadded),
            all(
                digests.get(captures[name]["finals_digest"]) == [name]
                and captures[name]["finals_digest"] != FROZEN_DIGEST
                for name in unpadded
            ),
        ),
        rule="'three': the unpadded captures, each alone under a digest not the padded one",
    )
    P(sec, "166", ptr(*RVF, "22e8406", "word_level"))
    P(sec, "181", ptr(*RVF, "a207de6", "word_level"))
    P(sec, "147", ptr(*RVR, "word_level"))
    bars = (("22e8406",), ("a207de6",))
    P(
        sec,
        "flex widths (2939; 1538/1071/330; 1562/1028/349; 1724/935/280)",
        "",
        ([2939], [1538, 1071, 330], [1562, 1028, 349], [1724, 935, 280]),
        record=(
            [frozen["22e8406"]["fixed_c32_vs_fixed_c8"]["identical"]],
            *(
                [at(ptr(*RVF, *bar, k)) for k in ("identical", "timing_only", "text_differs")]
                for bar in bars
            ),
            [at(ptr(*RVR, k)) for k in ("identical", "timing_only", "text_differs")],
        ),
    )

    # --- §6.5 ------------------------------------------------------------------------
    sec = "§6.5 p1"
    P(sec, "32", ptr(*CAP, C22_32, "observed_peak_in_flight"))
    P(sec, "8", ptr(*CAP, C22_8, "observed_peak_in_flight"))
    P(sec, "2,939", ptr(*FROZ, "22e8406", "fixed_c32_vs_fixed_c8", "identical"))
    P(
        sec,
        "The two commits gave the same digest",
        ptr(*FROZ, "a207de6", "finals_digest"),
        (2, True),
        record=(
            len(frozen),
            at(ptr(*FROZ, "a207de6", "finals_digest")) == frozen["22e8406"]["finals_digest"],
        ),
        rule="'two': the frozen commits, and their digests equal",
    )
    identical_on = ptr("server", "confidence_equivalence", "nemo-shipped", "identical")
    P(
        sec,
        "two more captures ... one in each of its modes",
        identical_on,
        (2, True),
        record=(
            len(S["server"]["confidence_equivalence"]),
            at(identical_on)
            and all(m["identical"] for m in S["server"]["confidence_equivalence"].values()),
        ),
        rule="'two', 'one in each': one capture per confidence mode, each identical",
    )
    P(
        sec,
        "six captures",
        ptr("server", "finals_digests", FROZEN_DIGEST),
        6,
        record=len(padded),
        rule="'six': the captures under the padded digest",
    )
    zero = ptr(*FROZ, "a207de6", "fixed_c32_vs_fixed_c8", "text_differs")
    P(
        sec,
        "A zero in 2,939",
        zero,
        (0, 2939),
        record=(
            max(
                frozen[c]["fixed_c32_vs_fixed_c8"][k]
                for c in frozen
                for k in ("text_differs", "timing_only")
            ),
            at(ptr(*FROZ, "a207de6", "fixed_c32_vs_fixed_c8", "recordings")),
        ),
        rule="'a zero': no change in text or timings, at either commit",
    )
    sec = "§6.5 p2"
    P(sec, "32", ptr(*CAP, C22_R, "observed_peak_in_flight"))
    P(sec, "330", ptr(*RVF, "22e8406", "text_differs"))
    P(sec, "166", ptr(*RVF, "22e8406", "word_level"))
    P(sec, "1,071", ptr(*RVF, "22e8406", "timing_only"))
    P(sec, "349", ptr(*RVF, "a207de6", "text_differs"))
    P(sec, "181", ptr(*RVF, "a207de6", "word_level"))
    P(sec, "1,028", ptr(*RVF, "a207de6", "timing_only"))
    P(
        sec,
        "80 ms",
        ptr(*RVF, "22e8406", "timing_shift_ms", "median"),
        (80, 80),
        record=both_commits("timing_shift_ms", "median"),
    )
    p90 = ptr(*RVF, "22e8406", "timing_shift_ms", "p90")
    P(
        sec,
        "90th percentile",
        p90,
        90,
        record=90 if isinstance(at(p90), int) and p90.endswith("/p90") else None,
        rule="the quantile the figure is: the summary's p90 field",
    )
    P(sec, "320 ms", p90, (320, 320), record=both_commits("timing_shift_ms", "p90"))
    P(
        sec,
        "640 ms",
        ptr(*RVF, "22e8406", "timing_shift_ms", "max"),
        (640, 640),
        record=both_commits("timing_shift_ms", "max"),
    )
    P(sec, "280", ptr(*RVR, "text_differs"))
    P(sec, "147", ptr(*RVR, "word_level"))
    P(sec, "935", ptr(*RVR, "timing_only"))
    P(sec, "1,724", ptr(*RVR, "identical"))
    sec = "§6.5 p3"
    P(
        sec,
        "6.152%",
        ptr(*RVF, "22e8406", "wer", "wer_a"),
        (6.152, 6.152),
        record=tuple(x * 100 for x in both_commits("wer", "wer_a")),
    )
    percent(sec, "6.161%", ptr(*RVF, "22e8406", "wer", "wer_b"), rounded=False)
    percent(sec, "6.159%", ptr(*RVF, "a207de6", "wer", "wer_b"), rounded=False)
    sec = "§6.5 p4"
    P(sec, "a207de6", flips_places, "a207de6", record=places_commit)
    P(sec, "189", ptr(*PA, "places"))
    P(sec, "181", ptr(*PA, "recordings_changed"))
    P(sec, "62", ptr(*PA, "alone_right"))
    P(sec, "54", ptr(*PA, "batch_right"))
    P(sec, "73", ptr(*PA, "neither_right"))
    for text, key in (("33%", "alone_right"), ("29%", "batch_right"), ("39%", "neither_right")):
        P(
            sec,
            text,
            "",
            _parse(text),
            record=at(ptr(*PA, key)) / at(ptr(*PA, "places")) * 100,
            tol=0.5,
            rounds=WHOLE,
        )
    P(sec, "0.52", ptr(*PA, "alone_vs_batch_sign_test_p"), 0.52, tol=0.005, rounds=TWO_PLACES)
    P(sec, "111", ptr(*PA, "place_shapes", "one_for_one"))
    P(sec, "68", ptr(*PA, "place_shapes", "multi_word"))
    P(sec, "10", ptr(*PA, "place_shapes", "one_side_empty"))
    P(sec, "189", ptr(*PA, "places"))  # "Of the 189 places"

    # --- §6.6 ------------------------------------------------------------------------
    sec = "§6.6 p1"
    P(sec, "128-row bucket", ptr("server", "setting", "bucket"))
    P(sec, "32", ptr(*CAP, C22_32, "concurrency_configured"))
    p95 = ptr(*CAP, C22_32, "p95_tick_ms_after")
    P(
        sec,
        "95th-percentile",
        p95,
        95,
        record=95 if isinstance(at(p95), float) and "/p95_" in p95 else None,
        rule="the quantile the figure is: the summary's p95_tick_ms_after field",
    )
    millis(sec, "187 ms", p95)
    millis(sec, "183 ms", ptr(*CAP, CA2_32, "p95_tick_ms_after"))
    P(sec, "160 ms", ptr("server", "setting", "chunk_ms"))
    P(sec, "160 ms", ptr("server", "setting", "chunk_ms"))  # "within 160 ms"
    P(sec, "2,534", ptr(*CAP, C22_32, "ticks_over_budget_during_run"))
    P(sec, "4,008", ptr(*CAP, C22_32, "ticks_during_run"))
    P(sec, "2,452", ptr(*CAP, CA2_32, "ticks_over_budget_during_run"))
    P(sec, "4,002", ptr(*CAP, CA2_32, "ticks_during_run"))
    P(sec, "8", ptr(*CAP, C22_8, "concurrency_configured"))
    millis(sec, "136 ms", ptr(*CAP, C22_8, "p95_tick_ms_after"))
    millis(sec, "113 ms", ptr(*CAP, CA2_8, "p95_tick_ms_after"))
    P(sec, "2,419", ptr(*CAP, C22_8, "ticks_over_budget_during_run"))
    P(sec, "2,053", ptr(*CAP, CA2_8, "ticks_over_budget_during_run"))
    for name in (C22_32, C22_8, C22_R, CA2_32, CA2_8, CA2_PB, CA2_NS, CA2_R, CA2_RR):
        P(
            sec,
            "No stream was refused in any capture",
            ptr(*CAP, name, "refused_during_run"),
            0,
            rule="'No stream': zero refused in this capture",
        )
    for name in (C22_32, C22_8, C22_R, CA2_32, CA2_8, CA2_PB, CA2_NS, CA2_R, CA2_RR):
        P(
            "§6.6 p2",
            "none refused a stream",
            ptr(*CAP, name, "refused_during_run"),
            0,
            rule="'none': zero refused in this capture",
        )

    # --- §7 and Table 4 --------------------------------------------------------------
    sec = "§7 p2"
    P(sec, "189", ptr("flips_vs_confidence", "places_count"))
    P(
        sec,
        "240",
        ptr(*PB, "k_words"),
        [240],
        record=sorted(
            {
                at(ptr(*PB, "k_words")),
                at(ptr(*NS, "k_words")),
                at(ptr(*PB, "flips", "words_flagged")),
            }
        ),
    )
    P(
        sec,
        "240 served words with the lowest word confidence",
        ptr(*PB, "confidence", "words_flagged"),
        [240],
        record=sorted(
            {
                at(ptr(*PB, "confidence", "words_flagged")),
                at(ptr(*NS, "confidence", "words_flagged")),
            }
        ),
    )
    same = ptr(*PB, "same_answers_as_served")
    P(
        sec,
        "Both confidence captures returned exactly the served answer",
        same,
        True,
        record=all(
            at(ptr(*mode, "same_answers_as_served")) == "IDENTICAL IN TEXT AND TIMINGS"
            and at(ptr(*mode, "finals_digest")) == FROZEN_DIGEST
            for mode in (PB, NS)
        )
        and at(same) == "IDENTICAL IN TEXT AND TIMINGS",
        rule="both modes' answers identical to the served capture, under the padded digest",
    )
    P("Table 4", "240", ptr(*NS, "k_words"))
    table4 = (
        ("flips", ptr(*PB, "flips"), ("189", "125", "66%", "174", "5.4%", "8.0%")),
        ("paper-best", ptr(*PB, "confidence"), ("226", "190", "84%", "221", "6.9%", "7.0%")),
        ("nemo-shipped", ptr(*NS, "confidence"), ("208", "145", "70%", "178", "5.5%", "7.5%")),
    )
    for name, base, (spans, wrong, precision, removable, recall, random) in table4:
        sec = f"Table 4 {name}"
        P(sec, spans, base + "/spans")
        P(sec, wrong, base + "/not_right")
        percent(sec, precision, base + "/precision", rounded=True)
        P(sec, removable, base + "/word_errors_removable")
        percent(sec, recall, base + "/recall", rounded=True)
        percent(sec, random, base + "/null/random_recordings/not_right_rate_mean", rounded=True)
    P(
        "Table 4",
        "flips = the same row in both modes",
        ptr(*NS, "flips", "not_right"),
        True,
        record=at(ptr(*NS, "flips")) == at(ptr(*PB, "flips")),
        rule="the flips block is the same in both modes, so one row stands for both",
    )
    sec = "Table 4 caption"
    P(sec, "240", ptr(*PB, "flips", "words_flagged"))
    blocks = (ptr(*PB, "flips"), ptr(*PB, "confidence"), ptr(*NS, "confidence"))
    P(
        sec,
        "3,220",
        ptr(*PB, "flips", "corpus_word_errors"),
        [3220],
        record=sorted({at(b + "/corpus_word_errors") for b in blocks}),
    )
    P(
        sec,
        "1,000",
        ptr(*PB, "flips", "null", "draws"),
        [1000],
        record=sorted({at(b + "/null/draws") for b in blocks}),
    )
    for b in blocks:
        P(sec, "0.001", b + "/null/random_recordings/p_value_not_right")
    P(
        sec,
        "a207de6",
        ptr("flips_vs_confidence", "places"),
        "a207de6",
        record=at(ptr("flips_vs_confidence", "places"))
        .removeprefix("places-")
        .removesuffix(".json.gz"),
    )
    P(sec, "160 ms", ptr("server", "setting", "chunk_ms"))
    P(sec, "bucket 128", ptr("server", "setting", "bucket"))
    P(sec, "189", ptr(*PB, "flips", "spans"))
    P(sec, "4", ptr(*PB, "flips", "empty_spans"))
    P(sec, "64", ptr("flips_vs_confidence", "served_right", "span_rule"))
    P(sec, "189", ptr(*PB, "flips", "spans"))
    by_analyse = ptr("flips_vs_confidence", "served_right", "by_analyse")
    P(
        sec,
        "62",
        by_analyse,
        (62, 62),
        record=(at(by_analyse), at(ptr(*PA, "alone_right")) + at(ptr(*PA, "both_right"))),
        rule="the flips' spans right by analyse's rule, which are §6.5's served-right places",
    )
    ties = (*PB, "tie_range")
    P(
        sec,
        "six words tie at the cutoff confidence",
        ptr(*ties, "tied"),
        (6, 1, [1, 6]),
        record=(
            at(ptr(*ties, "tied")),
            at(ptr(*ties, "chosen")),
            [at(ptr(*PB, "tied_at_cutoff", k)) for k in ("flagged", "all")],
        ),
        rule="'six': the words at the cutoff, one of them flagged, as the scorer reported the tie",
    )
    P(
        sec,
        "all six picks",
        ptr(*ties, "choices"),
        6,
        rule="'six': every way of breaking the tie, each scored",
    )
    P(sec, "225 to 226", ptr(*ties, "range", "spans"), [225, 226])
    P(sec, "189 to 190", ptr(*ties, "range", "not_right"), [189, 190])
    P(sec, "219 to 221", ptr(*ties, "range", "word_errors_removable"), [219, 221])
    recall_range = ptr(*ties, "range", "recall")
    P(
        sec,
        "6.8% to 6.9%",
        recall_range,
        [6.8, 6.9],
        record=[x * 100 for x in at(recall_range)],
        tol=0.05,
        rounds=ONE_PLACE,
    )
    precision_range = ptr(*ties, "range", "precision")
    P(
        sec,
        "84% for every pick",
        precision_range,
        [84, 84],
        record=[x * 100 for x in at(precision_range)],
        tol=0.5,
        rounds=WHOLE,
    )
    P(sec, "1,000", ptr(*PB, "flips", "null", "draws"))  # "the smallest 1,000 draws allow"
    sec = "§7 p3"
    P(
        sec,
        "Two thirds",
        ptr(*PB, "flips", "precision"),
        (2, 3),
        record=nearest_fraction(at(ptr(*PB, "flips", "precision")), (2, 3, 4)),
        rule="the nearest of halves, thirds and quarters",
    )
    percent(
        sec,
        "8.0%",
        ptr(*PB, "flips", "null", "random_recordings", "not_right_rate_mean"),
        rounded=True,
    )
    percent(
        sec,
        "17%",
        ptr(*PB, "flips", "null", "same_recordings", "not_right_rate_mean"),
        rounded=True,
    )
    percent(sec, "84%", ptr(*PB, "confidence", "precision"), rounded=True)
    percent(sec, "66%", ptr(*PB, "flips", "precision"), rounded=True)
    P(sec, "219 to 221", ptr(*ties, "range", "word_errors_removable"), [219, 221])
    P(
        sec,
        "six words tied at its cutoff",
        ptr(*ties, "tied"),
        6,
        rule="'six': the words at the cutoff, as the scorer reported the tie",
    )
    P(sec, "174", ptr(*PB, "flips", "word_errors_removable"))
    percent(sec, "70%", ptr(*NS, "confidence", "precision"), rounded=True)
    P(sec, "178", ptr(*NS, "confidence", "word_errors_removable"))
    source = ptr("inputs", "sha256_raw", "stock-nemotron-bf16-1120.json.gz")
    P(
        "§7 p4 (Jev)",
        "1,120 ms stock probe",
        source,
        1120,
        record=S["stock"]["nemotron-bf16-1120"]["settings"]["chunk_ms"]
        if re.fullmatch(r"[0-9a-f]{64}", at(source))
        and S["stock"]["nemotron-bf16-1120"]["record"] == "stock-nemotron-bf16-1120.json.gz"
        else None,
    )

    # --- §8 --------------------------------------------------------------------------
    sec = "§8 subtitles"
    rates = [*both_commits("timing_only_rate"), at(ptr(*RVR, "timing_only_rate"))]
    P(
        sec,
        "about a third",
        ptr(*RVF, "22e8406", "timing_only_rate"),
        [(1, 3)],
        record=sorted({nearest_fraction(r, (2, 3, 4)) for r in rates}),
        rule="each unpadded comparison's timing-only share, to halves, thirds or quarters",
    )
    P(
        sec,
        "80",
        ptr(*RVF, "22e8406", "timing_shift_ms", "min"),
        (80, 80),
        record=both_commits("timing_shift_ms", "min"),
    )
    P(
        sec,
        "640 ms",
        ptr(*RVF, "22e8406", "timing_shift_ms", "max"),
        (640, 640),
        record=both_commits("timing_shift_ms", "max"),
    )
    sec = "§8 consistency (changed)"
    P(sec, "1 and 42", "", (1, 42), record=(min(levels), max(levels)))
    P(
        sec,
        "8 and 32",
        ptr(*CAP, C22_8, "concurrency_configured"),
        [8, 32],
        record=concurrencies(padded),
    )
    P(sec, "124-stream capacity", "", record=ladder["s"])
    P(
        sec,
        "one time in three",
        "",
        (29, 36),
        record=(min(place_shares), max(place_shares)),
        tol=0.5,
        rounds=WHOLE,
        rule="each version's share right, every bf16 arm, within the 29% to 36% §5.4 prints",
    )
    P(sec, "62", ptr(*PA, "alone_right"))
    P(sec, "189", ptr(*PA, "places"))
    P(
        "§8 why (changed)",
        "one recording in ten",
        "",
        (9, 11),
        record=(min(bf16_pct), max(bf16_pct)),
        tol=0.5,
        rounds=WHOLE,
        rule="the bf16 rates, within the 9% to 11% the abstract prints",
    )
    sec = "§8 alternatives (changed)"
    P(
        sec,
        "1 recording in 2,939",
        "",
        (1, 2939),
        record=(len(fp32_b300[0]["divergences"]), fp32_b300[0]["checked"]),
    )
    P(
        sec,
        "no recording's words changed",
        ptr(*SF, "ragged", "word_level"),
        0,
        record=max(both_arms(SF, "word_level")),
        rule="'no': neither unpadded float32 arm, at word level",
    )
    P(sec, "2", ptr(*SF, "equalised", "timing_only"), (2, 2), record=both_arms(SF, "timing_only"))

    # --- Limitations -----------------------------------------------------------------
    sec = "Limitations"
    P(sec, "256; 42", "", ([256], 42), record=(streams_per_level, max(levels)))
    P(
        sec,
        "2,939",
        ptr(*CAP, CA2_32, "recordings"),
        [2939],
        record=sorted({capture["recordings"] for capture in captures.values()}),
    )
    P(
        sec,
        "8 and 32",
        ptr(*CAP, CA2_32, "concurrency_configured"),
        [8, 32],
        record=concurrencies(list(captures)),
    )
    P(
        sec,
        "one bucket",
        ptr("server", "setting", "bucket"),
        128,
        rule="'one': the summary's one server setting, 128 rows",
    )
    P(
        sec,
        "two of the unpadded server at the second",
        ptr(*RVR, "captures"),
        (2, ["a207de6"]),
        record=(
            len(at(ptr(*RVR, "captures"))),
            sorted({captures[name]["commit"][:7] for name in at(ptr(*RVR, "captures"))}),
        ),
        rule="'two': the unpadded captures compared with each other, both at a207de6",
    )
    P(
        sec,
        "[70,13]; 32; [70,1]; 128-row; batch of 1",
        "",
        ([70, 13], 32, [70, 1], 128),
        record=(
            b300["att_context_size"],
            b300["batch"],
            control_arm["shared"]["att_context"],
            churn["shared"]["bucket"],
        ),
        rule="the 1 is the lone decode, a batch of one by the probe's design",
    )
    P(sec, "240", ptr(*PB, "k_words"))

    # --- §9, §10, Reproducibility, References ----------------------------------------
    P("§9 crash", "160 ms", ptr("stock", "nemotron-bf16-160-high-n1024", "settings", "chunk_ms"))
    sec = "§10"
    P(sec, "2,939", ptr(*RVR, "recordings"))
    P(
        sec,
        "six padded captures",
        ptr("server", "finals_digests", FROZEN_DIGEST),
        6,
        record=len(padded),
        rule="'six': the captures under the padded digest",
    )
    P(
        sec,
        "two concurrencies, two commits and three confidence settings",
        ptr(*FROZ),
        (2, 2, 3),
        record=(
            len(concurrencies(padded)),
            len(at(ptr(*FROZ))),
            len({captures[name]["word_confidence"] for name in padded}),
        ),
        rule="as §1 contributions",
    )
    P(sec, "147", ptr(*RVR, "word_level"))
    P(sec, "935", ptr(*RVR, "timing_only"))
    P(sec, "42", "", record=max(levels))
    sec = "Reproducibility"
    P(sec, "16", ptr("inputs", "sha256_raw"), record=len(at(ptr("inputs", "sha256_raw"))))
    directory = at(ptr("inputs", "directory"))
    P(
        sec,
        "step1-a6000-2026-09-26",
        ptr("inputs", "directory"),
        "step1-a6000-2026-09-26",
        record=directory if (ROWS / directory).is_dir() else None,
    )
    P(
        sec,
        "step1-summary-2026-09-26.json",
        "",
        SUMMARY,
        record=SUMMARY if (ROWS / SUMMARY).is_file() else None,
    )
    P("References [15]", "nemotron-speech-streaming-en-0.6b", MODEL, model_name, record=model_name)


def numbers_mapping(path: Path) -> Counter[tuple[str, str, str]]:
    """The paper's numbers mapping, keyed as `Audit.printed` keys its claims."""
    rows: Counter[tuple[str, str, str]] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")
        if cells[:2] == ["section", "printed"]:
            continue
        section, printed, source = cells[0], cells[1], cells[2]
        pointer = source.split()[1] if source.startswith("summary /") else ""
        rows[(section, printed, pointer)] += 1
    return rows


def compare_mapping(a: Audit, rows: Counter[tuple[str, str, str]]) -> list[str]:
    """Rows nothing here checks or lists, and claims the mapping does not list."""
    ours = a.printed_keys + Counter({(s, p, ""): 1 for s, p in NOT_FROM_A_RECORD})
    problems = []
    for (section, printed, pointer), n in sorted((rows - ours).items()):
        problems.append(
            f"numbers.tsv {section}: {printed!r} ({pointer or 'no pointer'}) x{n}: "
            "no claim here checks it and it is not listed as not from a record"
        )
    for (section, printed, pointer), n in sorted((ours - rows).items()):
        problems.append(
            f"claim {section}: {printed!r} ({pointer or 'no pointer'}) x{n}: "
            "the paper's numbers mapping does not list it"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, help=f"read this copy of {SUMMARY}")
    parser.add_argument("--numbers", type=Path, help="the white paper's numbers.tsv to compare")
    args = parser.parse_args(argv)

    a = audit(args.summary)
    print(f"{a.matched} published claims matched their record")
    drafted = sum(a.printed_keys.values())
    print(
        f"of them {drafted} are Draft 3's printed numbers; "
        f"{len(NOT_FROM_A_RECORD)} of its strings are from no record and are listed, not checked"
    )
    failures = list(a.failures)
    if args.numbers is not None:
        rows = numbers_mapping(args.numbers)
        problems = compare_mapping(a, rows)
        failures += problems
        if not problems:
            print(f"numbers.tsv: all {sum(rows.values())} rows are claims here or listed")
    if failures:
        print(f"\n*** {len(failures)} MISMATCH(ES) — fix the prose or the audit:")
        for failure in failures:
            print(f"    {failure}")
        return 1
    print("no mismatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
