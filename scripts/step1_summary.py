#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Step 1 on one RTX A6000: the figures its raw records give, derived with the repo's own tools.

    python3 scripts/step1_summary.py [--records DIR] [--out NEW.json | --check SUMMARY.json]
        [--jobs N]

Reads ONLY the gzipped records in ``rows/exploratory/step1-a6000-2026-09-26/`` (``--records``)
and their ``MANIFEST.json``, and writes one small JSON with sorted keys:
``rows/exploratory/step1-summary-2026-09-26.json`` is this script's output. ``--check`` compares
the output with a file byte for byte (exit 1 when they differ); ``--out`` writes a new file and
refuses one that exists; with neither, the JSON goes to stdout. Nothing here touches a GPU.

Refused (exit 2), before anything is derived:

* the directory's ``*.json.gz`` files are not exactly the ones the manifest lists;
* a file's sha256 is not the manifest's ``sha256_gz``, or the sha256 of its decompressed bytes
  is not the manifest's ``sha256_raw``;
* the commit the manifest gives is not the one the record stamps (``commit_stamp`` names the
  field);
* a places record does not name, by sha256, the captures here it was built from;
* a record holds a machine path (``scripts/scrub_record_paths.py``'s ``machine_paths``: an
  absolute path other than a placeholder's tail or the kernel's ``/proc/``, ``/dev/``,
  ``/sys/``). The records are committed with the paths their tools stamped replaced by
  placeholders; the manifest keeps each record's sha256 as its tool wrote it.

Every figure comes from the repo's own tools, imported, never restated:

* **Stock arms** (``probes/stock_divergence.py`` records): ``checked``; ``text_divergent`` and
  ``timing_only`` (text equal, word timings not), recounted from the stored divergences and
  refused unless the recount is the probe's own counter; ``word_level``, the text-divergent
  recordings whose ``words()`` differ (the probe's normalisation, imported from the probe:
  lowercased, and every run of characters other than word characters, apostrophes and spaces
  made a space, so "B.C." is two words and "BC" one); ``punctuation_only``, the text-divergent
  recordings whose words agree. Rates are ``scripts/step1_places.py``'s ``rate``, over
  ``checked``.
* **Word error rate per side and the timing-shift distribution** of every arm and of each
  server fixed-against-ragged comparison: ``scripts/step1_places.py``'s ``derive`` on the
  record, which checks every counter against its transcripts first. It is given an empty word
  list, so no out-of-dictionary figure is derived here: those need a word list that is not in
  these records, and are left out.
* **Scores with nulls** (the 160 ms stock record, and the flips against the model's confidence):
  ``derive`` with ``step1_places``' defaults, 1,000 draws and seed 20260924, the setting the
  scored files of 2026-09-26 used. The confidence is attached by ``step1_places``'
  ``load_confidence_capture``, which accepts a confidence capture only when contract C8 holds
  against the served capture the places record names; side ``a`` is the served (fixed) answer.
  The flips are the place flag on side ``a``; the confidence flag marks the same number K of
  side ``a``'s lowest-confidence words. Where words tie at the cutoff, ``step1_places`` breaks
  the tie by a seeded shuffle and also scores every other way of breaking it (``tie_range``),
  which is kept. ``served_right`` counts the flips' spans that are right by the flag's rule
  (``span_not_right``) and by ``analyse``'s, which differ at empty spans.
* **Server**: ``scripts/compare_captures.py``: ``compare`` (fixed c32 against fixed c8 with the
  ragged capture as ``--control`` for FROZEN; fixed against ragged with the second fixed
  capture as ``--repeat``; ragged against ragged), ``confidence_equivalence`` (each confidence-on
  capture against the fixed capture taken with it off). The fixed-against-ragged counts are
  refused unless they are the places record's own. ``word_level`` as for the stock arms.
* **Ticks**: each capture's ``server.load`` as the capture recorded it (the tick p95 the server
  reported before and after the run, and the ticks over budget, late and in all during it). The
  tick budget is not stamped in the captures, so nothing here says what it was.
* **Replay**: ``scripts/compare_stock_replay.py``'s ``differences`` between the bfloat16
  1,120 ms stock record and the replay of its first targets.

The output carries no machine path (refused if it would), and names each record by its file
name here. ``--jobs`` runs the scoring in that many processes (spawned; 1 runs it in this one).
Each scoring job reads its own records and seeds its own draws, so the output does not depend on
``--jobs``: the committed summary was checked with 1 and with 3.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RECORDS = ROOT / "rows" / "exploratory" / "step1-a6000-2026-09-26"
SUMMARY = ROOT / "rows" / "exploratory" / "step1-summary-2026-09-26.json"
MANIFEST = "MANIFEST.json"


def _load(name: str, path: Path) -> Any:
    """A module from a file, registered first (a dataclass resolves its module by name)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The scorer; ``unstable_words`` (its ``analyse``) comes with it, and ``compare_captures`` is
#: the instance the scorer judges contract C8 with.
sp = _load("step1_summary_step1_places", SCRIPTS / "step1_places.py")
scrubber = _load("step1_summary_scrub_record_paths", SCRIPTS / "scrub_record_paths.py")
cc = sp.comparator()
csr = _load("step1_summary_compare_stock_replay", SCRIPTS / "compare_stock_replay.py")
probe = _load("step1_summary_stock_divergence", ROOT / "probes" / "stock_divergence.py")

EXIT_OK, EXIT_DIFFERS, EXIT_REFUSED = 0, 1, 2
#: step1_places' defaults, the setting of the scored files these figures are checked against.
DRAWS, SEED = sp.DEFAULT_DRAWS, sp.DEFAULT_SEED

STOCK = {
    "nemotron-bf16-1120": "stock-nemotron-bf16-1120.json.gz",
    "nemotron-fp32-1120": "stock-nemotron-fp32-1120.json.gz",
    "hybrid-bf16-1120": "stock-hybrid-bf16-1120.json.gz",
    "nemotron-bf16-160-high-n1024": "stock-nemotron-bf16-160-high-n1024.json.gz",
}
#: The stock record scored with nulls: the only one that stores every recording's words.
SCORED_STOCK = "nemotron-bf16-160-high-n1024"
REPLAY = "replay-22e8406-nemotron-bf16-1120-n64.json.gz"
REPLAYED = STOCK["nemotron-bf16-1120"]
COMMITS = ("22e8406", "a207de6")
PLACES = {commit: f"places-{commit}.json.gz" for commit in COMMITS}
#: The places record's captures, by its ``captures`` key.
PLACES_CAPTURES = {"a_fixed": "fixed-c32", "b_ragged": "ragged-c32", "repeat_fixed": "fixed-c8"}
FLIPS_COMMIT = "a207de6"
CONFIDENCE = {
    "paper-best": "server-a207de6-fixed-paperbest-c32.json.gz",
    "nemo-shipped": "server-a207de6-fixed-nemoshipped-c32.json.gz",
}
RAGGED_REPEAT = "server-a207de6-ragged-c32-repeat.json.gz"
DTYPE = "bfloat16"


def capture(commit: str, label: str) -> str:
    return f"server-{commit}-{label}.json.gz"


CAPTURES = [capture(c, x) for c in COMMITS for x in ("fixed-c32", "fixed-c8", "ragged-c32")]
CAPTURES += [*CONFIDENCE.values(), RAGGED_REPEAT]

#: The settings a stock record stamps, copied when present.
STOCK_SETTINGS = (
    "model",
    "model_revision",
    "machine",
    "torch",
    "nemo",
    "verbatim_commit",
    "chunk_ms",
    "att_context_size",
    "att_context_size_observed",
    "batch",
    "matmul_precision",
    "targets",
    "word_confidence",
    "use_cuda_graphs",
)
#: What a capture's settings are compared on across all of them: every one but the padding and
#: the word confidence (and the decoder step confidence, which is the word confidence observed).
SHARED_SETTINGS = (
    "model",
    "revision",
    "chunk_ms",
    "dtype",
    "execution",
    "matmul",
    "bucket",
    "att_context",
    "pipeline",
    "biasing",
    "decoder_graphs",
    "nemo_version",
    "torch_version",
    "device_name",
)
FLAG_KEYS = (
    "spans",
    "empty_spans",
    "words_flagged",
    "recordings_flagged",
    "not_right",
    "not_right_by_analyse",
    "precision",
    "word_errors_removable",
    "corpus_word_errors",
    "recall",
    "recall_withheld",
    "null",
)
#: ``unstable_words.analyse``'s figures, but the out-of-dictionary ones (no word list here) and
#: the list of negation or number places (counted instead).
PLACE_KEYS = (
    "recordings_changed",
    "places",
    "place_shapes",
    "alone_right",
    "batch_right",
    "neither_right",
    "both_right",
    "alone_vs_batch_sign_test_p",
    "reference_words",
    "word_errors_alone",
    "word_errors_in_batch",
    "wer_alone",
    "wer_in_batch",
    "net_extra_errors_in_batch",
)
CORPUS_KEYS = ("reference_words", "word_errors_a", "word_errors_b", "wer_a", "wer_b")
LOAD_KEYS = (
    "ticks_during_run",
    "ticks_over_budget_during_run",
    "ticks_late_during_run",
    "admitted_during_run",
    "refused_during_run",
)

RULES = {
    "word_level": "a text-divergent recording whose two transcripts differ after the probe's "
    "words() (probes/stock_divergence.py: lowercased, every run of characters other than word "
    "characters, apostrophes and spaces made a space)",
    "punctuation_only": "a text-divergent recording whose two transcripts agree after words()",
    "timing_only": "the two texts are equal and the word timings (word, start, end) are not",
    "rates": "over the recordings checked, rounded to 5 decimals (scripts/step1_places.py rate)",
    "wer": "scripts/step1_places.py corpus figures: word Levenshtein distance to the reference, "
    "summed over every checked recording, over the reference words, after its normalise(); side "
    "a and side b as the arm names them",
    "places_analysis": "scripts/unstable_words.py analyse via scripts/step1_places.py, whose "
    "labels name side a 'alone' and side b 'in_batch' (or 'batch'); out-of-dictionary figures "
    "are not derived here (no word list in these records)",
    "flags": "scripts/step1_places.py: precision is the flag's spans that are not right over its "
    "spans; recall is the word errors a perfect correction of every flagged span would remove, "
    "over all the side's word errors; each null draws spans of the same lengths at random, "
    "1,000 draws, seed 20260924",
    "flips_vs_confidence": "the flips are the place flag on side a, the served (fixed) answer; "
    "the confidence flag marks side a's K lowest-confidence words, K being the words the flips "
    "mark",
    "tie_range": "scripts/step1_places.py tie_range: when words tie at the confidence cutoff, "
    "every choice of the tied words that fills K, scored without nulls; each figure's lowest and "
    "highest. The flag's own figures are the seeded choice's",
    "served_right": "the flips' spans that are right: by the flag's rule (span_not_right, a gap "
    "not right only when a reference word is missing there), spans minus not_right; by analyse's "
    "rule, which also calls a gap not right when side b's words there are right, spans minus "
    "not_right_by_analyse",
    "ticks": "server.load as each capture recorded it; the tick budget is not stamped in the "
    "captures",
}


class Refused(ValueError):
    """The records cannot be summarised; ``main`` prints why and exits ``EXIT_REFUSED``."""


# --- the records, as the manifest names them ------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stamp(record: Mapping[str, Any], dotted: str) -> Any:
    node: Any = record
    for key in dotted.split("."):
        node = node.get(key) if isinstance(node, Mapping) else None
    return node


def read_verified(directory: Path, name: str, entry: Mapping[str, Any]) -> tuple[bytes, Any]:
    """The raw bytes of ``name`` and the record they hold, refused unless the gz file and its
    decompressed bytes have the sha256 the manifest gives and the record stamps its commit."""
    packed = (directory / name).read_bytes()
    if sha256(packed) != entry["sha256_gz"]:
        raise Refused(
            f"{name}: its sha256 is {sha256(packed)}, the manifest says {entry['sha256_gz']}"
        )
    raw = gzip.decompress(packed)
    if sha256(raw) != entry["sha256_raw"]:
        raise Refused(
            f"{name}: the sha256 of its decompressed bytes is {sha256(raw)}, the manifest says "
            f"{entry['sha256_raw']}"
        )
    record = json.loads(raw)
    found = machine_paths(record)
    if found:
        raise Refused(f"{name}: holds a machine path: " + "; ".join(found[:3]))
    stamped = _stamp(record, entry["commit_stamp"])
    if stamped != entry["commit"]:
        raise Refused(
            f"{name}: {entry['commit_stamp']} is {stamped!r}, the manifest says {entry['commit']!r}"
        )
    return raw, record


@dataclass(frozen=True)
class Inputs:
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    records: dict[str, Any]


def load_inputs(directory: Path = RECORDS) -> Inputs:
    """Every record the manifest lists, verified; refused before anything is derived."""
    raw_manifest = (directory / MANIFEST).read_bytes()
    manifest = json.loads(raw_manifest)
    listed = set(manifest["records"])
    present = {p.name for p in directory.glob("*.json.gz")}
    if listed != present:
        raise Refused(
            f"{directory.name}: the manifest lists {sorted(listed - present)} that are not here "
            f"and not {sorted(present - listed)} that are"
        )
    records = {
        name: read_verified(directory, name, entry)[1]
        for name, entry in sorted(manifest["records"].items())
    }
    for commit in COMMITS:
        check_places_links(records[PLACES[commit]], commit, manifest["records"])
    return Inputs(directory, manifest, sha256(raw_manifest), records)


def check_places_links(
    places: Mapping[str, Any], commit: str, entries: Mapping[str, Mapping[str, Any]]
) -> None:
    """Refuse a places record that does not name, by sha256, the captures here of its commit."""
    for key, label in PLACES_CAPTURES.items():
        named = (places.get("captures") or {}).get(key) or {}
        want = entries[capture(commit, label)]["sha256_raw"]
        if named.get("sha256") != want:
            raise Refused(
                f"{PLACES[commit]}: its {key} capture has sha256 {named.get('sha256')!r}, not "
                f"{capture(commit, label)}'s {want}"
            )


# --- the scoring, which is slow, in worker processes ----------------------------------------


def _entries(inputs: Inputs, *names: str) -> dict[str, Any]:
    return {name: inputs.manifest["records"][name] for name in names}


def jobs_of(inputs: Inputs) -> list[tuple[str, str, str, dict[str, Any]]]:
    """(kind, key, directory, the manifest entries of the files it reads), slowest first."""
    d = str(inputs.directory)
    flips = PLACES[FLIPS_COMMIT]
    served = capture(FLIPS_COMMIT, "fixed-c32")
    jobs = [
        ("flips", mode, d, _entries(inputs, flips, served, name))
        for mode, name in CONFIDENCE.items()
    ]
    jobs.append(("stock", SCORED_STOCK, d, _entries(inputs, STOCK[SCORED_STOCK])))
    jobs += [
        ("stock", key, d, _entries(inputs, name))
        for key, name in STOCK.items()
        if key != SCORED_STOCK
    ]
    jobs += [("places", commit, d, _entries(inputs, PLACES[commit])) for commit in COMMITS]
    return jobs


def derive_job(job: tuple[str, str, str, dict[str, Any]]) -> tuple[tuple[str, str], Any]:
    """One ``step1_places.derive``; reads and verifies its own records."""
    kind, key, directory, entries = job
    where = Path(directory)
    got = {name: read_verified(where, name, entry) for name, entry in entries.items()}
    none: frozenset[str] = frozenset()
    if kind == "stock":
        record = got[STOCK[key]][1]
        draws = DRAWS if key == SCORED_STOCK else 0
        return (kind, key), sp.derive(record, none, draws=draws, seed=SEED)
    if kind == "places":
        return (kind, key), sp.derive(got[PLACES[key]][1], none, draws=0, seed=SEED)
    places = got[PLACES[FLIPS_COMMIT]][1]
    with tempfile.TemporaryDirectory(prefix="step1-summary-") as tmp:
        # load_confidence_capture reads files: the verified bytes, written as they are.
        on, off = Path(tmp) / "on.json", Path(tmp) / "off.json"
        on.write_bytes(got[CONFIDENCE[key]][0])
        off.write_bytes(got[capture(FLIPS_COMMIT, "fixed-c32")][0])
        confidence = sp.load_confidence_capture(on, "a", places, against=off)
        derived = sp.derive(
            places,
            none,
            draws=DRAWS,
            seed=SEED,
            confidence_at=(DTYPE, sp.CAPTURES_ARM),
            confidence=confidence,
        )
    return (kind, key), derived


def derive_all(inputs: Inputs, jobs: int = 1) -> dict[tuple[str, str], Any]:
    """Every ``derive`` the summary reads, in ``jobs`` processes (in this one when 1)."""
    work = jobs_of(inputs)
    if jobs <= 1:
        return dict(derive_job(job) for job in work)
    context = multiprocessing.get_context("spawn")
    if str(SCRIPTS) not in sys.path:
        # A spawned worker imports this module by name, from the parent's sys.path.
        sys.path.insert(0, str(SCRIPTS))
    with concurrent.futures.ProcessPoolExecutor(jobs, mp_context=context) as pool:
        return dict(pool.map(derive_job, work))


# --- the figures ------------------------------------------------------------------------------


def differs_in_words(a: str, b: str) -> bool:
    return probe.words(a) != probe.words(b)


def flag_brief(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: score[key] for key in FLAG_KEYS if key in score}


def places_brief(analysis: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    out = {key: analysis[key] for key in PLACE_KEYS}
    out["negation_or_number_places"] = len(analysis["negation_or_number_places"])
    return out


def shift_brief(shift: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in shift.items() if key != "unpairable"}
    out["unpairable"] = len(shift["unpairable"])
    return out


def stock_arm(arm: Mapping[str, Any], derived: Mapping[str, Any], where: str) -> dict[str, Any]:
    """One arm's counts, recounted from its divergences and checked against the probe's own
    counters and the scorer's reading of the arm."""
    checked = arm["checked"]
    text = [d for d in arm["divergences"] if d["text_differs"]]
    timing = [d for d in arm["divergences"] if not d["text_differs"]]
    counters = (arm["text_divergent"], arm["timing_only_divergent"])
    recount = (len(text), len(timing))
    scorer = (derived["text_divergent"], derived["timing_only_divergent"])
    if recount != counters or scorer != counters:
        raise Refused(
            f"{where}: recounted {recount[0]} text and {recount[1]} timing-only divergences; "
            f"the probe counted {counters}, the scorer read {scorer}"
        )
    word_level = sum(differs_in_words(d["a"], d["b"]) for d in text)
    rate = sp.rate
    return {
        "checked": checked,
        "text_divergent": len(text),
        "text_divergent_rate": rate(len(text), checked),
        "word_level": word_level,
        "word_level_rate": rate(word_level, checked),
        "punctuation_only": len(text) - word_level,
        "punctuation_only_rate": rate(len(text) - word_level, checked),
        "timing_only": len(timing),
        "timing_only_rate": rate(len(timing), checked),
        "confidence_only": arm.get("confidence_only_divergent"),
        "timing_shift_ms": shift_brief(derived["timing_only_shift_ms"]),
        "wer": {key: derived["corpus"][key] for key in CORPUS_KEYS},
        "zero_reading": derived["zero_reading"],
    }


def stock_section(name: str, record: Mapping[str, Any], derived: Mapping[str, Any]) -> dict:
    runs = {}
    for dtype, run in record["runs"].items():
        arms = {
            arm_name: stock_arm(arm, derived["runs"][dtype]["arms"][arm_name], f"{name} {arm_name}")
            for arm_name, arm in run["arms"].items()
        }
        runs[dtype] = {
            "repeat": run.get("repeat"),
            "repeat_verdict": run.get("repeat_verdict"),
            "positive_control": run.get("positive_control"),
            "arms": arms,
        }
    return {
        "record": name,
        "settings": {key: record[key] for key in STOCK_SETTINGS if key in record},
        "runs": runs,
    }


def scored_section(name: str, derived: Mapping[str, Any]) -> dict[str, Any]:
    """The 160 ms stock record's places, flags and nulls, as step1_places scores them."""
    runs = {}
    for dtype, run in derived["runs"].items():
        arms = {}
        for arm_name, arm in run["arms"].items():
            arms[arm_name] = {
                "places": arm["flag"]["places"],
                "both_right": arm["flag"]["both_right"],
                "places_analysis": places_brief(arm["places"]),
                "flag_a": flag_brief(arm["flag"]["a"]),
                "flag_b": flag_brief(arm["flag"]["b"]),
                "confidence": {s: arm["confidence"][s]["status"] for s in "ab"},
                "normalisation": {
                    key: arm["normalisation"][key]
                    for key in (
                        "text_divergent_raw",
                        "text_divergent_after_normalisation",
                        "punctuation_or_case_only",
                        "places_differing_only_by_a_contraction",
                    )
                },
            }
        runs[dtype] = {"repeat_verdict_accepted": run["repeat_verdict_accepted"], "arms": arms}
    return {"record": name, "nulls": {"draws": DRAWS, "seed": SEED}, "runs": runs}


def replay_section(inputs: Inputs) -> dict[str, Any]:
    stock, replay = inputs.records[REPLAYED], inputs.records[REPLAY]
    differences = csr.differences(stock, replay)
    return {
        "record": REPLAY,
        "replays": REPLAYED,
        "targets": replay["targets"],
        "reproduced": not differences,
        "differences": differences,
        "arms": {
            f"{dtype}/{name}": {
                key: arm[key] for key in ("checked", "text_divergent", "timing_only_divergent")
            }
            for dtype, run in replay["runs"].items()
            for name, arm in run["arms"].items()
        },
    }


def loaded(inputs: Inputs, name: str) -> Any:
    entry = inputs.manifest["records"][name]
    return cc.Loaded(name, entry["sha256_raw"], inputs.records[name])


def identity_counts(same: Mapping[str, Any], a: Mapping, b: Mapping) -> dict[str, Any]:
    """compare_captures' identity counts, with the word-level split of the text differences."""
    n = same["recordings"]
    word_level = sum(
        differs_in_words(a["recordings"][rid]["text"], b["recordings"][rid]["text"])
        for rid in same["text_differs_ids"]
    )
    return {
        "recordings": n,
        "identical": same["identical"],
        "text_differs": same["text_differs"],
        "text_differs_rate": sp.rate(same["text_differs"], n),
        "word_level": word_level,
        "word_level_rate": sp.rate(word_level, n),
        "punctuation_only": same["text_differs"] - word_level,
        "timing_only": same["timing_only"],
        "timing_only_rate": sp.rate(same["timing_only"], n),
        "confidence_only_differs": same["confidence_only_differs"],
        "digests_match": same["digests_match"],
    }


def common_setting(inputs: Inputs, names: Sequence[str]) -> dict[str, Any]:
    """The setting every capture shares; refused when any two differ in one of them."""
    settings = {name: cc.settings_of(inputs.records[name]) for name in names}
    first = settings[names[0]]
    for name, got in settings.items():
        apart = [key for key in SHARED_SETTINGS if got[key] != first[key]]
        if apart:
            raise Refused(f"{name} and {names[0]} differ in {', '.join(apart)}")
    return {key: first[key] for key in SHARED_SETTINGS}


def capture_facts(inputs: Inputs, name: str) -> dict[str, Any]:
    x = loaded(inputs, name)
    problems = cc.capture_problems(x)
    if problems:
        raise Refused("; ".join(problems))
    record = x.record
    load = record["server"]["load"]
    padding, how = cc.padding_of(record)
    return {
        "commit": record["client"]["verbatim_commit"],
        "padding": padding,
        "padding_known": how,
        "word_confidence": cc.settings_of(record)["word_confidence"],
        "concurrency_configured": cc.concurrency_of(record),
        "observed_peak_in_flight": cc.peak_of(record),
        "recordings": len(record["recordings"]),
        "finals_digest": record["finals_digest"],
        "started": record["started"],
        "wall_clock_s": record["wall_clock_s"],
        "p95_tick_ms_before": load["before"]["p95_tick_ms"],
        "p95_tick_ms_after": load["after"]["p95_tick_ms"],
        **{key: load[key] for key in LOAD_KEYS},
        "warnings": record["warnings"],
        "reading": record["reading"],
    }


def frozen_at(inputs: Inputs, commit: str) -> dict[str, Any]:
    """compare_captures' FROZEN reading of one commit's captures: fixed c32 against fixed c8,
    with the ragged capture as the control."""
    fixed, second, ragged = (
        loaded(inputs, capture(commit, label)) for label in ("fixed-c32", "fixed-c8", "ragged-c32")
    )
    frozen = cc.compare(fixed, second, control=ragged)
    return {
        "frozen": frozen["frozen"],
        "verdict": frozen["verdict"],
        "finals_digest": fixed.record["finals_digest"],
        "fixed_c32_vs_fixed_c8": identity_counts(frozen["identity"], fixed.record, second.record),
    }


def ragged_vs_fixed(
    inputs: Inputs, derived: Mapping[tuple[str, str], Any], commit: str
) -> dict[str, Any]:
    """One commit's fixed capture (served, side a) against its ragged capture (side b), with
    the second fixed capture as the repeat; refused unless the counts are the places record's."""
    fixed, second, ragged = (
        loaded(inputs, capture(commit, label)) for label in ("fixed-c32", "fixed-c8", "ragged-c32")
    )
    apart = cc.compare(fixed, ragged, repeat=second)
    same = apart["identity"]
    arm = inputs.records[PLACES[commit]]["runs"][DTYPE]["arms"][cc.ARM]
    counted = (same["text_differs"], same["timing_only"])
    stored = (arm["text_divergent"], arm["timing_only_divergent"])
    if counted != stored:
        raise Refused(
            f"{commit}: fixed against ragged differ on {counted[0]} texts and {counted[1]} "
            f"timings only; the places record says {stored[0]} and {stored[1]}"
        )
    scored = derived[("places", commit)]["runs"][DTYPE]["arms"][cc.ARM]
    return {
        **identity_counts(same, fixed.record, ragged.record),
        "verdict": apart["verdict"],
        "places": PLACES[commit],
        "sides": sp.SIDES[cc.ARM],
        "wer": {key: scored["corpus"][key] for key in CORPUS_KEYS},
        "timing_shift_ms": shift_brief(scored["timing_only_shift_ms"]),
    }


def server_section(inputs: Inputs, derived: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    records = inputs.records
    out: dict[str, Any] = {
        "setting": common_setting(inputs, CAPTURES),
        "captures": {name: capture_facts(inputs, name) for name in CAPTURES},
        "frozen_rule": cc.FROZEN_RULE,
        "frozen": {commit: frozen_at(inputs, commit) for commit in COMMITS},
        "ragged_vs_fixed": {commit: ragged_vs_fixed(inputs, derived, commit) for commit in COMMITS},
    }
    served = loaded(inputs, capture(FLIPS_COMMIT, "fixed-c32"))
    out["confidence_equivalence"] = {}
    for mode, name in CONFIDENCE.items():
        differences = cc.confidence_equivalence(served, loaded(inputs, name))
        out["confidence_equivalence"][mode] = {
            "on": name,
            "off": served.path,
            "identical": not differences,
            "verdict": cc.CONFIDENCE_EQUIVALENT if not differences else "NOT SHOWN IDENTICAL",
            "differences": differences,
        }
    first = loaded(inputs, capture(FLIPS_COMMIT, "ragged-c32"))
    again = loaded(inputs, RAGGED_REPEAT)
    twice = cc.compare(first, again)
    out["ragged_vs_ragged"] = {
        FLIPS_COMMIT: {
            **identity_counts(twice["identity"], first.record, again.record),
            "captures": [first.path, again.path],
            "verdict": twice["verdict"],
        }
    }
    digests: dict[str, list[str]] = {}
    for name in CAPTURES:
        digests.setdefault(records[name]["finals_digest"], []).append(name)
    out["finals_digests"] = {digest: sorted(names) for digest, names in digests.items()}
    return out


def flips_section(inputs: Inputs, derived: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    """The flips against the model's own confidence at the same number of words, per mode."""
    modes = {}
    for mode, name in CONFIDENCE.items():
        arm = derived[("flips", mode)]["runs"][DTYPE]["arms"][sp.CAPTURES_ARM]
        confidence = arm["confidence"]["a"]
        if confidence.get("status") != "scored":
            raise Refused(
                f"{name}: the confidence flag was not scored ({confidence.get('status')!r}: "
                f"{confidence.get('reason')!r})"
            )
        flips, ranked = arm["flag"]["a"], confidence["flag"]
        # The flips mark one span per place; the confidence flag marks K words, K the flips'.
        k = confidence["k_words"]
        shaped = flips["spans"] == arm["flag"]["places"]
        shaped = shaped and ranked["words_flagged"] == k == flips["words_flagged"]
        if not shaped:
            raise Refused(
                f"{name}: the flips are {flips['spans']} spans for {arm['flag']['places']} places "
                f"and {flips['words_flagged']} words; the confidence flag marks "
                f"{ranked['words_flagged']} words at K {confidence['k_words']}"
            )
        modes[mode] = {
            "confidence_capture": name,
            "same_answers_as_served": confidence["identity"]["verdict"],
            "finals_digest": confidence["identity"]["finals_digest"],
            "k_words": confidence["k_words"],
            "cutoff_confidence": confidence["cutoff_confidence"],
            "tied_at_cutoff": confidence["tied_at_cutoff"],
            "tie_range": confidence["tie_range"],
            "flips": flag_brief(flips),
            "confidence": flag_brief(ranked),
        }
    # The places, and so the place flags, are the same whichever capture gives the confidence.
    arm = derived[("flips", next(iter(CONFIDENCE)))]["runs"][DTYPE]["arms"][sp.CAPTURES_ARM]
    served = arm["flag"]["a"]
    return {
        "places": PLACES[FLIPS_COMMIT],
        "side": sp.SIDES[sp.CAPTURES_ARM]["a"],
        "nulls": {"draws": DRAWS, "seed": SEED},
        "places_count": arm["flag"]["places"],
        "both_right": arm["flag"]["both_right"],
        "places_analysis": places_brief(arm["places"]),
        "flips_side_b": flag_brief(arm["flag"]["b"]),
        "served_right": {
            "span_rule": served["spans"] - served["not_right"],
            "by_analyse": served["spans"] - served["not_right_by_analyse"],
        },
        "modes": modes,
    }


#: Every string in a JSON value that holds an absolute path, with where it is: the scrub's own
#: test, so a record the scrub passes and a summary this writes are held to one rule.
machine_paths = scrubber.machine_paths


def assemble(inputs: Inputs, derived: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    records = inputs.records
    summary = {
        "question": "Step 1 on nvidia/nemotron-speech-streaming-en-0.6b, on one RTX A6000: how "
        "often do its answers depend on the batch, does the server's fixed padding freeze them, "
        "and do the places where they flip find errors better than the model's own confidence?",
        "not_a_row": "Derived by scripts/step1_summary.py from the exploratory records in "
        f"rows/exploratory/{inputs.directory.name}/ (stock-pipeline probes and server captures on "
        "one card). Not a harness row: nothing here is a benchmark result or reads a gate.",
        "inputs": {
            "directory": inputs.directory.name,
            "manifest_sha256": inputs.manifest_sha256,
            "sha256_raw": {
                name: entry["sha256_raw"] for name, entry in inputs.manifest["records"].items()
            },
        },
        "rules": RULES,
        "stock": {
            key: stock_section(name, records[name], derived[("stock", key)])
            for key, name in STOCK.items()
        },
        "stock_160ms_scored": scored_section(STOCK[SCORED_STOCK], derived[("stock", SCORED_STOCK)]),
        "replay": replay_section(inputs),
        "server": server_section(inputs, derived),
        "flips_vs_confidence": flips_section(inputs, derived),
    }
    return summary


def render(summary: Mapping[str, Any]) -> str:
    """The summary as written: sorted keys, one space of indent; refused if it would carry a
    machine path."""
    found = machine_paths(summary)
    if found:
        raise Refused("the summary would carry a machine path: " + "; ".join(found))
    return json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def build(directory: Path = RECORDS, jobs: int = 1) -> str:
    inputs = load_inputs(directory)
    return render(assemble(inputs, derive_all(inputs, jobs)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=RECORDS)
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--out", type=Path, default=None, help="a new file; refused if it exists")
    where.add_argument("--check", type=Path, default=None, help="exit 1 unless the output is it")
    parser.add_argument("--jobs", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.out is not None and os.path.lexists(args.out):
        print(f"step1_summary: {args.out} exists; it is not written over", file=sys.stderr)
        return EXIT_REFUSED
    try:
        text = build(args.records, args.jobs)
    except (Refused, OSError, ValueError, KeyError) as exc:
        print(f"step1_summary: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.check is not None:
        same = args.check.read_bytes() == text.encode("utf-8")
        print(f"step1_summary: {'the same bytes as' if same else 'DIFFERS from'} {args.check}")
        return EXIT_OK if same else EXIT_DIFFERS
    if args.out is None:
        sys.stdout.write(text)
        return EXIT_OK
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
