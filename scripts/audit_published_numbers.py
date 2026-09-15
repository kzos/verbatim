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
    best = max(
        (arm for arm in rare["arms"] if arm["boost"] is not None),
        key=lambda arm: (arm["terms"]["recall"], -arm["terms"]["false_accepts"]),
    )
    a.claim("shipped phrase-book weight is the measured knee", book["boost"], best["boost"])

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

    # --- DR-0016 and the README: the A6000 is NOT withdrawn ------------------------
    a6000 = a.load("ladder-a6000-bf16-eager-2026-09-13-all-criteria.json")
    a.claim(
        "A6000 failing rungs failed on latency, so DR-0016 does not reach them",
        {r.get("first_failing_criterion") for r in a6000["rungs"] if not r["passed"]},
        {"latency"},
    )
    a.claim("A6000 certification", a6000["s"], 0)

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
