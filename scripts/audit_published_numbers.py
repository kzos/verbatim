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

Exit 0 when every claim matches, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "rows" / "exploratory"


class Audit:
    def __init__(self) -> None:
        self.matched = 0
        self.failures: list[str] = []

    def load(self, name: str) -> Any:
        return json.loads((ROWS / name).read_text(encoding="utf-8"))

    def claim(self, where: str, record: Any, published: Any, tol: float = 0.0) -> None:
        """`record` is read from the JSON; `published` is what a document says."""
        if isinstance(record, (int, float)) and isinstance(published, (int, float)):
            good = abs(record - published) <= tol
        else:
            good = record == published
        if good:
            self.matched += 1
        else:
            self.failures.append(f"{where}: record={record!r} published={published!r}")


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


def main() -> int:
    a = Audit()

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

    def best_per_seed(document):
        out = {}
        for rung in document["rungs"]:
            if rung.get("passed"):
                seed = rung["seed"] % 100
                out[seed] = max(out.get(seed, 0), rung["n"])
        return out

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
    bf16_rates = [
        len(r["divergences"]) / r["checked"]
        for r in (
            stock_a6000["uncontrolled"],
            stock_a6000["controlled"],
            stock_b300["bfloat16"]["ragged"],
            stock_b300["bfloat16"]["equalised"],
        )
    ]
    a.claim("paper abstract: lowest bf16 rate, 'about 9%'", round(min(bf16_rates) * 100), 9)
    a.claim("paper abstract: highest bf16 rate, '11%'", round(max(bf16_rates) * 100), 11)

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
    a.claim("plan and paper: 'about six times', lowest arm", round(min(ratios), 1), 5.9, 0.05)
    a.claim("plan and paper: 'about six times', highest arm", round(max(ratios), 1), 6.4, 0.05)
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
    a.claim(
        "paper §5.4 places involving a negation or a number",
        len(eq["negation_or_number_places"]),
        6,
    )

    print(f"{a.matched} published claims matched their record")
    if a.failures:
        print(f"\n*** {len(a.failures)} MISMATCH(ES) — fix the prose or the audit:")
        for failure in a.failures:
            print(f"    {failure}")
        return 1
    print("no mismatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
