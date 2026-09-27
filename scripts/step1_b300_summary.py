#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Step 1 on one B300 with a retest at 64, and the A6000's ragged pair at 8: what the records give.

    python3 scripts/step1_b300_summary.py [--b300 DIR] [--a6000-c8 DIR] [--a6000 DIR]
        [--a6000-gate-smoke DIR] [--out NEW.json | --check SUMMARY.json]

Reads ONLY the gzipped records in ``rows/exploratory/step1-b300-2026-09-26/`` (``--b300``),
``rows/exploratory/step1-a6000-c8-2026-09-26/`` (``--a6000-c8``),
``rows/exploratory/step1-a6000-2026-09-26/`` (``--a6000``) and
``rows/exploratory/step1-a6000-gate-smoke-2026-09-26/`` (``--a6000-gate-smoke``) and their
``MANIFEST.json`` files, and writes one small JSON with sorted keys:
``rows/exploratory/step1-b300-summary-2026-09-26.json`` is this script's output. ``--check``
compares the output with a file byte for byte (exit 1 when they differ); ``--out`` writes a new
file and refuses one that exists; with neither, the JSON goes to stdout. Nothing here touches a
GPU. The A6000 records of ``step1-a6000-2026-09-26/`` are ``scripts/step1_summary.py``'s; they
are read here only for the cards they name and for the server processes that ran the unpadded
pair at concurrency 32 (the ``a6000`` section).

Refused (exit 2), before anything is derived:

* a directory's ``*.json.gz`` files are not exactly the ones its manifest lists;
* a file's sha256 is not the manifest's ``sha256_gz``, or the sha256 of its decompressed bytes
  is not the manifest's ``sha256_raw``;
* a record holds a machine path (``scripts/scrub_record_paths.py``'s ``machine_paths``);
* the commit the manifest gives is not the one the record stamps (``commit_stamp`` names the
  field). A gate record whose ``commit_stamp`` is null stamps no commit; it is accepted only for
  the gate's kinds (``UNSTAMPED_KINDS``), and only when its manifest commit is the one the gate
  summary stamps for every arm;
* the places record does not name, by sha256, the three captures here it was built from, or an
  arm's record summary does not name, by sha256, that arm's invariance record and finals here.

Every figure comes from the repo's own tools, imported, never restated:

* **Captures** (``scripts/compare_captures.py``, through ``scripts/step1_summary.py``'s readers):
  each capture's file name says its padding, execution and configured concurrency, and each is
  refused unless the capture records that padding (``padding_of``), that execution
  (``settings_of``, as the server reported it: ``eager`` or ``graph path``) and that concurrency.
  FROZEN (``compare`` of fixed c32 against fixed c8 with the ragged capture as ``--control``, in
  eager execution; in graph execution there is no ragged capture, and the comparator's own
  ``pair_problems`` says why the eager one cannot stand in); fixed eager against fixed graphs
  (``identity``, after ``pair_problems`` finds nothing apart but the execution); fixed against
  ragged (``compare`` with the second fixed capture as ``--repeat``, refused unless the counts are
  the places record's own, and ``step1_places.derive`` on the places record for word error rate
  and the timing shifts); ragged against ragged, on the B300 at concurrency 32 and 64 and on
  the A6000 at concurrency 8, each refused unless the invariance gate's own ``diff_records``,
  run on every recording's two finals, gives the comparator's identical, text and timing-only
  counts. ``word_level`` splits the text differences by the stock probe's ``words()``, as
  ``step1_summary`` does. Each capture's tick statistics are its ``server.load``.
* **The refused runs** (the overload test's second run at concurrency 128, and the fixed run of
  the retest at 64): ``capture_problems`` must refuse each (it is not a capture), and its name
  must say the padding, execution and concurrency it records; its counts (``counts``, and
  ``server.load``'s admitted and refused during the run) are refused unless a recount from its
  recordings gives them.
* **The retest at concurrency 64**: its failed first attempt must be refused by
  ``capture_problems`` too, and its counts (sent, finals, partials, refusals, time-outs) are
  refused unless a recount from its recordings gives them and every recording is either sent or
  refused. Which server process ran each of the retest's records, and in what order, comes from
  the process id and start each record stamps, never printed; a process that had admitted no
  session before the run is one started for it.
* **The cards** (``card_uuids``): every GPU UUID any record names, under any key whose name holds
  ``uuid`` (NVIDIA's ``GPU-`` prefix dropped, so the torch spelling counts as the same card). On
  the B300 ``cards`` is the count over the B300 records; on the A6000 over the records of the three
  A6000 directories. Each section also says how many of its records name a card at all.
* **Server processes**: each ragged pair's ``server_processes`` is how many distinct process ids
  its two captures stamp (before and after the run), never printed.
* **The A6000's smoke-stopped runbook attempts** (``step1-a6000-gate-smoke-2026-09-26/``): for
  each bucket, the smoke check's budget, highest p95 tick and problems, refused unless the serve
  spec's argv and NeMo spec both give that bucket and the fixed padding, the smoke check's budget
  is the serve spec's, and the server's /readyz names the device. The runbook stops before the
  long gate when the smoke check lists a problem (``scripts/step1_gate_runbook.sh``
  ``run_smoke``).
* **The gate** (``verbatim_bench.invariance``): each arm's finals are rebuilt into level runs and
  ``assess`` gives the verdict, the digests and the divergences again; refused unless they are
  the invariance record's, the finals', the record summary's and the gate summary's. Per level,
  the streams that differ are split into text (and of those, word-level) and timing-only
  (``kind`` "words": the texts equal and the timings not), refused unless the parts sum to the
  streams. The smoke check's budget must be the one the arm's serve spec derived.

The output carries no machine path (refused if it would), and names each record by its file
name here.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ROWS = ROOT / "rows" / "exploratory"
B300_RECORDS = ROWS / "step1-b300-2026-09-26"
C8_RECORDS = ROWS / "step1-a6000-c8-2026-09-26"
A6000_RECORDS = ROWS / "step1-a6000-2026-09-26"
SMOKE_RECORDS = ROWS / "step1-a6000-gate-smoke-2026-09-26"
SUMMARY = ROWS / "step1-b300-summary-2026-09-26.json"
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


#: The A6000 summary's readers, and through it the comparator, the scorer and the probe's words().
s1 = _load("step1_b300_summary_step1_summary", SCRIPTS / "step1_summary.py")
cc, sp, probe = s1.cc, s1.sp, s1.probe
from verbatim_bench import invariance as inv  # noqa: E402  (bench/src, as compare_captures needs)
from verbatim_bench.canonical import FinalRecord  # noqa: E402

EXIT_OK, EXIT_DIFFERS, EXIT_REFUSED = 0, 1, 2
DTYPE = "bfloat16"

# --- the B300 records, by name ---------------------------------------------------------------

CAPTURE_COMMIT = "26eaebb"
#: The execution each label names, as the server reports it (``settings_of``'s ``execution``).
EXECUTIONS = {"eager": "eager", "graphs": "graph path"}


def capture(label: str) -> str:
    return f"server-{CAPTURE_COMMIT}-{label}.json.gz"


FIXED = {(e, c): capture(f"fixed-{e}-c{c}") for e in EXECUTIONS for c in (32, 8)}
RAGGED = capture("ragged-eager-c32")
RAGGED_REPEAT = capture("ragged-eager-c32-repeat")
PLACES = f"places-{CAPTURE_COMMIT}-eager.json.gz"
#: The places record's captures, by its ``captures`` key.
PLACES_CAPTURES = {
    "a_fixed": FIXED[("eager", 32)],
    "b_ragged": RAGGED,
    "repeat_fixed": FIXED[("eager", 8)],
}
OVERLOAD = "server-ba20c49-ragged-c128-overload-run1.json.gz"
REFUSED_RUN = "server-ba20c49-ragged-c128-overload-run2.FAILED.json.gz"
#: The retest at concurrency 64, at ba20c49: a first attempt at the ragged run that failed, then
#: two complete ragged runs one after the other on one server process, then the fixed run on
#: another, refused.
C64 = ("server-ba20c49-ragged-c64-run1.json.gz", "server-ba20c49-ragged-c64-run2.json.gz")
C64_FIXED_REFUSED = "server-ba20c49-fixed-c64.FAILED.json.gz"
C64_FIRST_ATTEMPT = "server-ba20c49-ragged-c64-first-attempt.FAILED.json.gz"
#: What each complete B300 capture's name says: (padding, execution, configured concurrency).
B300_CAPTURES = {
    **{name: ("fixed", EXECUTIONS[e], c) for (e, c), name in FIXED.items()},
    RAGGED: ("ragged", "eager", 32),
    RAGGED_REPEAT: ("ragged", "eager", 32),
    OVERLOAD: ("ragged", "eager", 128),
    C64[0]: ("ragged", "eager", 64),
    C64[1]: ("ragged", "eager", 64),
}
#: What each B300 record of a failed run says in its name; none is a capture.
B300_FAILED = {
    REFUSED_RUN: ("ragged", "eager", 128),
    C64_FIXED_REFUSED: ("fixed", "eager", 64),
    C64_FIRST_ATTEMPT: ("ragged", "eager", 64),
}

GATE_SUMMARY = "gate-ba20c49-summary.json.gz"
ARMS = ("fixed-churn", "fixed-const", "ragged-churn", "ragged-const")
CARD = "gate-ba20c49-card.json.gz"
MODEL = "gate-ba20c49-model.json.gz"
#: The gate's records that stamp no commit; the gate summary stamps it for every arm.
UNSTAMPED_KINDS = frozenset(
    {
        "gate-invariance",
        "gate-finals",
        "gate-record-summary",
        "gate-smoke-check",
        "gate-serve-spec",
        "gate-preflight",
        "gate-readyz",
    }
)


def gate(kind: str, arm: str) -> str:
    return f"gate-ba20c49-{kind}-{arm}.json.gz"


#: The A6000's two ragged captures at concurrency 8, one after the other on one server.
C8 = ("server-ba20c49-ragged-c8-run1.json.gz", "server-ba20c49-ragged-c8-run2.json.gz")
C8_SAYS = ("ragged", "eager", 8)
#: The A6000's unpadded pair at concurrency 32 (``step1_summary``'s ragged against ragged).
A6000_C32 = ("server-a207de6-ragged-c32.json.gz", "server-a207de6-ragged-c32-repeat.json.gz")
A6000_C32_SAYS = ("ragged", "eager", 32)
#: The A6000 runbook's two attempts, stopped by the smoke check before the long gate.
SMOKE_BUCKETS = (64, 32)


def smoke(kind: str, bucket: int) -> str:
    return f"gate-5eda079-b{bucket}-{kind}-fixed-churn.json.gz"


#: A card's UUID as NVIDIA and torch spell it (the ``GPU-`` prefix is NVIDIA's).
CARD_UUID = re.compile(r"(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")

#: The settings the captures of one card share; the execution is compared by name instead.
SHARED = tuple(key for key in s1.SHARED_SETTINGS if key != "execution")
SMOKE_KEYS = (
    "budget_ms",
    "max_p95_tick_ms",
    "max_consecutive_overruns",
    "max_degradation_level",
    "refused_total_after",
)
MONITOR_KEYS = (
    "max_p95_tick_ms",
    "max_consecutive_overruns",
    "max_degradation_level",
    "max_live",
    "max_refused_total",
)
REFUSED_COUNTS = (
    "recordings",
    "recordings_sent",
    "sessions_acknowledged",
    "recordings_with_terminal_final",
    "recordings_missing_final",
)
#: How the refusal names itself in a refused session's client error.
REFUSAL = "RESOURCE_EXHAUSTED"
#: How the client names a session it gave up on, having sent all its audio and had no final.
TIMED_OUT = "TimeoutError"
#: The failed attempt's counts as its record states them, each recounted from its recordings.
ATTEMPT_COUNTS = (
    "recordings",
    "recordings_sent",
    "sessions_acknowledged",
    "recordings_with_terminal_final",
    "recordings_missing_final",
    "recordings_without_partials",
)

RULES = {
    **{
        key: s1.RULES[key]
        for key in ("word_level", "punctuation_only", "timing_only", "rates", "wer")
    },
    "execution": "as the server reported it (compare_captures settings_of): 'eager' is the eager "
    "encoder step, 'graph path' the encoder step on the CUDA-graph path; decoder graphs off in "
    "every capture",
    "frozen": "compare_captures compare: FROZEN_RULE. With no ragged capture on the graph path, "
    "the graph-path pair is identical at two observed peaks but not called frozen; "
    "graphs_control_refused is why the comparator will not take the eager ragged capture as its "
    "control",
    "ticks": "server.load as each capture recorded it: the ticks during the run, over budget and "
    "late, and the p95 tick the server reported before and after. The captures do not stamp the "
    "tick budget; the gate's serve spec and smoke check stamp the budget of the gate's server",
    "refused_run": "the overload test's second run, which the server's admission control "
    "refused: compare_captures capture_problems refuses it, so it is no capture and nothing is "
    "compared with it; its counts are recounted from its recordings (sent, and a client error "
    "naming RESOURCE_EXHAUSTED). repeatability_measured is whether it is a complete capture",
    "gate": "verbatim_bench.invariance assess over each arm's finals, checked against the "
    "invariance record, the finals, the runbook's record summary and gate summary. Against '1' is "
    "batch dependence; 32b against 32a is run to run at one concurrency. text/timing_only are the "
    "divergence kinds 'text' and 'words'; word_level as above, on the two levels' final texts",
    "cross_restart": "for each padding, whether the churn arm's and the const arm's digests are "
    "equal at a level: two server processes, the same corpus. Levels 1, 32a and 32b are the same "
    "condition in both arms; at max the churn arm churned and the const arm did not",
    "ticks_over_budget_rate": "ticks over budget during the run over the ticks during the run, "
    "rounded to 5 decimals",
    "card": "card_uuids: every GPU UUID a record names under a key whose name holds 'uuid' (the "
    "'GPU-' prefix dropped); cards is how many distinct ones the section's records name, "
    "records_naming_a_card how many of its records name any, records how many it read. The B300 "
    "section's records are the B300 directory's; the A6000 section's are those of "
    "step1-a6000-2026-09-26, step1-a6000-c8-2026-09-26 and step1-a6000-gate-smoke-2026-09-26",
    "server_processes": "how many distinct server process ids the pair's two captures stamp "
    "(server.process before and after each run); the ids are compared, never printed",
    "sessions_timed": "the client's in-flight count times a session from its first audio frame "
    "to its last final, so it covers only the sessions that received a final: sessions_timed is "
    "how many recordings carry an in_flight_s, recounted, and observed_peak_in_flight is the "
    "most of those in flight at once",
    "a6000_gate_smoke": "the invariance runbook on the A6000 at buckets 64 and 32: each attempt's "
    "smoke check before its first arm (fixed-churn), its serve spec and the server's /readyz "
    "before the smoke. stopped_before_the_gate is whether the smoke check lists a problem, which "
    "makes scripts/step1_gate_runbook.sh stop before the long gate; bucket is the serve spec's "
    "argv and NeMo batch size, refused unless both agree with the record's name",
    "ragged_vs_ragged": "compare_captures compare of two complete ragged captures at one "
    "configured concurrency, refused unless the invariance gate's diff_records over every "
    "recording's two finals (text first, then the word timings) gives the same identical, text "
    "and timing-only counts",
    "retest_c64": "the retest at concurrency 64 at ba20c49, on the card of the other B300 "
    "records. first_attempt is the ragged run's first attempt, which failed: not a capture, "
    "nothing is compared with it, and its counts are recounted from its recordings (sent, a "
    "final that covers the audio, partials, a client error naming RESOURCE_EXHAUSTED or "
    "TimeoutError). fixed_refused is the fixed run, refused as run2_refused of the overload test "
    "is. timeline orders the four records by the start each stamps; server_process numbers the "
    "server processes in that order, by the process id each record stamps (not printed); "
    "admitted_before_run is the server's admitted_total before the run, 0 for a process that "
    "had served no session before it",
}


class Refused(ValueError):
    """The records cannot be summarised; ``main`` prints why and exits ``EXIT_REFUSED``."""


# --- the records, as the manifests name them --------------------------------------------------


def read_verified(directory: Path, name: str, entry: Mapping[str, Any]) -> tuple[bytes, Any]:
    """``step1_summary.read_verified``, and for a gate record that stamps no commit the same
    checks with none to read."""
    if entry["commit_stamp"] is not None:
        try:
            return s1.read_verified(directory, name, entry)
        except s1.Refused as exc:
            raise Refused(str(exc)) from None
    if entry.get("kind") not in UNSTAMPED_KINDS:
        raise Refused(f"{name}: a {entry.get('kind')!r} record must stamp its commit")
    packed = (directory / name).read_bytes()
    if s1.sha256(packed) != entry["sha256_gz"]:
        raise Refused(
            f"{name}: its sha256 is {s1.sha256(packed)}, the manifest says {entry['sha256_gz']}"
        )
    raw = gzip.decompress(packed)
    if s1.sha256(raw) != entry["sha256_raw"]:
        raise Refused(
            f"{name}: the sha256 of its decompressed bytes is {s1.sha256(raw)}, the manifest "
            f"says {entry['sha256_raw']}"
        )
    record = json.loads(raw)
    found = s1.machine_paths(record)
    if found:
        raise Refused(f"{name}: holds a machine path: " + "; ".join(found[:3]))
    return raw, record


def load_directory(directory: Path) -> Any:
    """Every record a manifest lists, verified, as ``step1_summary.Inputs``."""
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
    return s1.Inputs(directory, manifest, s1.sha256(raw_manifest), records)


def check_links(b300: Any) -> None:
    """The places record names its captures, each arm's record summary its invariance record and
    finals, by sha256; a record that stamps no commit has the one the gate summary stamps."""
    entries, records = b300.manifest["records"], b300.records
    places = records[PLACES].get("captures") or {}
    for key, name in PLACES_CAPTURES.items():
        named, want = (places.get(key) or {}).get("sha256"), entries[name]["sha256_raw"]
        if named != want:
            raise Refused(f"{PLACES}: its {key} capture has sha256 {named!r}, not {name}'s {want}")
    for arm in ARMS:
        summary = records[gate("record-summary", arm)]
        for field, kind in (("record_sha256", "invariance"), ("finals_sha256", "finals")):
            want = entries[gate(kind, arm)]["sha256_raw"]
            if summary.get(field) != want:
                raise Refused(
                    f"{gate('record-summary', arm)}: its {field} is {summary.get(field)!r}, not "
                    f"{gate(kind, arm)}'s {want}"
                )
    heads = {arm: s["git_head"] for arm, s in records[GATE_SUMMARY]["settings_by_arm"].items()}
    if set(heads) != set(ARMS) or len(set(heads.values())) != 1:
        raise Refused(f"{GATE_SUMMARY}: the arms' git_head are {heads}")
    head = next(iter(heads.values()))
    for name, entry in entries.items():
        if entry["commit_stamp"] is None and entry["commit"] != head:
            raise Refused(f"{name}: the manifest says {entry['commit']!r}; the gate stamps {head}")


def says(inputs: Any, name: str, want: tuple[str, str, int]) -> None:
    """Refuse a capture whose name says a padding, execution or concurrency it did not record."""
    record = inputs.records[name]
    got = (
        cc.padding_of(record)[0],
        cc.settings_of(record)["execution"],
        cc.concurrency_of(record),
    )
    if got != want:
        raise Refused(
            f"{name}: its name says padding, execution and concurrency {want}; it records {got}"
        )


def common_setting(inputs: Any, names: Sequence[str]) -> dict[str, Any]:
    """The setting every capture shares but the execution; refused when two differ in one."""
    settings = {name: cc.settings_of(inputs.records[name]) for name in names}
    first = settings[names[0]]
    for name, got in settings.items():
        apart = [key for key in SHARED if got[key] != first[key]]
        if apart:
            raise Refused(f"{name} and {names[0]} differ in {', '.join(apart)}")
    return {key: first[key] for key in SHARED}


def capture_facts(inputs: Any, name: str) -> dict[str, Any]:
    """``step1_summary.capture_facts``, and the share of the run's ticks that went over budget."""
    facts = s1.capture_facts(inputs, name)
    over, ticks = facts["ticks_over_budget_during_run"], facts["ticks_during_run"]
    return {**facts, "ticks_over_budget_rate": sp.rate(over, ticks)}


def compared(*captures: Any, **kw: Any) -> dict[str, Any]:
    try:
        return cc.compare(*captures, **kw)
    except cc.Refused as exc:
        raise Refused(f"compare_captures refused: {exc}") from None


def counts(same: Mapping[str, Any], a: Any, b: Any) -> dict[str, Any]:
    return s1.identity_counts(same, a.record, b.record)


# --- the B300 captures ------------------------------------------------------------------------


def frozen_section(b300: Any) -> dict[str, Any]:
    loaded = {name: s1.loaded(b300, name) for name in (*FIXED.values(), RAGGED)}
    out: dict[str, Any] = {}
    for execution in EXECUTIONS:
        c32, c8 = loaded[FIXED[(execution, 32)]], loaded[FIXED[(execution, 8)]]
        control = loaded[RAGGED] if execution == "eager" else None
        report = compared(c32, c8, control=control)
        section = {
            "captures": [c32.path, c8.path],
            "control": None if control is None else control.path,
            "frozen": report["frozen"],
            "verdict": report["verdict"],
            "finals_digest": c32.record["finals_digest"],
            "fixed_c32_vs_fixed_c8": counts(report["identity"], c32, c8),
        }
        if control is not None:
            section["fixed_c32_vs_control"] = counts(report["third_capture_identity"], c32, control)
        else:
            section["graphs_control_refused"] = cc.pair_problems(c32, loaded[RAGGED])
        out[execution] = section
    return out


def eager_vs_graphs(b300: Any) -> dict[str, Any]:
    """Fixed eager against fixed graphs at each concurrency: the comparator finds nothing apart
    but the execution, and then their answers are compared recording by recording."""
    out = {}
    for c in (32, 8):
        eager, graphs = (s1.loaded(b300, FIXED[(e, c)]) for e in EXECUTIONS)
        apart = cc.pair_problems(eager, graphs)
        eager_is, graphs_is = EXECUTIONS["eager"], EXECUTIONS["graphs"]
        only = [f"the captures differ in execution: {eager_is!r} and {graphs_is!r}"]
        if apart != only:
            raise Refused(f"{eager.path} and {graphs.path}: {apart}; expected only {only}")
        out[f"c{c}"] = {
            "captures": [eager.path, graphs.path],
            "comparator_finds_apart": apart,
            **counts(cc.identity(eager.record, graphs.record), eager, graphs),
        }
    return out


def ragged_vs_fixed(b300: Any) -> dict[str, Any]:
    """The fixed capture (served, side a) against the ragged one (side b), the second fixed
    capture as the repeat; refused unless the counts are the places record's own."""
    fixed, second, ragged = (
        s1.loaded(b300, name) for name in (FIXED[("eager", 32)], FIXED[("eager", 8)], RAGGED)
    )
    apart = compared(fixed, ragged, repeat=second)
    same = apart["identity"]
    arm = b300.records[PLACES]["runs"][DTYPE]["arms"][cc.ARM]
    counted = (same["text_differs"], same["timing_only"])
    stored = (arm["text_divergent"], arm["timing_only_divergent"])
    if counted != stored:
        raise Refused(
            f"fixed against ragged differ on {counted[0]} texts and {counted[1]} timings only; "
            f"the places record says {stored[0]} and {stored[1]}"
        )
    scored = sp.derive(b300.records[PLACES], frozenset(), draws=0, seed=sp.DEFAULT_SEED)
    scored = scored["runs"][DTYPE]["arms"][cc.ARM]
    return {
        **counts(same, fixed, ragged),
        "captures": [fixed.path, ragged.path],
        "repeat": second.path,
        "verdict": apart["verdict"],
        "places": PLACES,
        "sides": sp.SIDES[cc.ARM],
        "wer": {key: scored["corpus"][key] for key in s1.CORPUS_KEYS},
        "timing_shift_ms": s1.shift_brief(scored["timing_only_shift_ms"]),
    }


def final_of(rid: str, entry: Mapping[str, Any]) -> Any:
    """One recording's final as the gate holds a stream's: its text and word timings."""
    return FinalRecord(rid, entry["text"], tuple(tuple(w) for w in cc.triples(entry["words"])))


def recount_by_the_gate(a: Any, b: Any) -> dict[str, int]:
    """Every recording's two finals through the invariance gate's own ``diff_records`` (text
    first, then the word timings): how many are identical, differ in text, or in timings only."""
    kinds = {"identical": 0, "text": 0, "words": 0}
    for rid, x in cc.ordered(a.record):
        found = inv.diff_records(final_of(rid, x), final_of(rid, b.record["recordings"][rid]))
        kinds["identical" if found is None else found[0]] += 1
    return kinds


def ragged_vs_ragged(inputs: Any, first: str, again: str) -> dict[str, Any]:
    """Two ragged captures compared; refused unless the gate's recount gives the same counts."""
    a, b = s1.loaded(inputs, first), s1.loaded(inputs, again)
    twice = compared(a, b)
    same = twice["identity"]
    counted = {
        "identical": same["identical"],
        "text": same["text_differs"],
        "words": same["timing_only"],
    }
    recount = recount_by_the_gate(a, b)
    if counted != recount:
        raise Refused(
            f"{a.path} against {b.path}: the comparator counts {counted}; the gate's "
            f"diff_records gives {recount}"
        )
    return {
        **counts(same, a, b),
        "captures": [a.path, b.path],
        "server_processes": server_processes(inputs, [first, again]),
        "verdict": twice["verdict"],
    }


def digests_of(inputs: Any, names: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in names:
        out.setdefault(inputs.records[name]["finals_digest"], []).append(name)
    return {digest: sorted(group) for digest, group in out.items()}


def pid_of(record: Mapping[str, Any]) -> Any:
    """The server process id a record stamps (compared, never printed)."""
    return record["server"]["process"]["before"]["pid"]


def server_processes(inputs: Any, names: Sequence[str]) -> int:
    """How many server processes ran these captures: the distinct process ids they stamp, before
    and after each run (compared, never printed)."""
    pids = {
        inputs.records[name]["server"]["process"][when]["pid"]
        for name in names
        for when in ("before", "after")
    }
    if None in pids:
        raise Refused(f"{', '.join(names)}: a capture stamps no server process id")
    return len(pids)


def not_a_capture(inputs: Any, name: str) -> list[str]:
    """Why the comparator refuses a failed run's record; refused if it would take it."""
    problems = cc.capture_problems(s1.loaded(inputs, name))
    if not problems:
        raise Refused(f"{name}: compare_captures would take it as a capture")
    return problems


def refused_run(b300: Any, name: str = REFUSED_RUN) -> dict[str, Any]:
    """A run the server's admission control refused: not a capture; its counts, recounted."""
    record = b300.records[name]
    problems = not_a_capture(b300, name)
    recordings = record["recordings"].values()
    recount = {
        "recordings": len(record["recordings"]),
        "recordings_sent": sum(e.get("sent") is True for e in recordings),
        "recordings_with_terminal_final": sum(e.get("terminal_final") is True for e in recordings),
        "refused_by_the_server": sum(REFUSAL in (e.get("client_error") or "") for e in recordings),
    }
    stated = {key: record["counts"][key] for key in REFUSED_COUNTS}
    load = record["server"]["load"]
    agree = (
        recount["recordings"] == stated["recordings"]
        and recount["recordings_sent"] == stated["recordings_sent"] == load["admitted_during_run"]
        and recount["recordings_with_terminal_final"] == stated["recordings_with_terminal_final"]
        and recount["refused_by_the_server"]
        == stated["recordings_missing_final"]
        == load["refused_during_run"]
    )
    if not agree:
        raise Refused(f"{name}: recounted {recount}; it states {stated} and load {load}")
    return {
        "record": name,
        "status": record["status"],
        "success": record["success"],
        "comparator_refuses_it": problems,
        "counts": stated,
        "recount": recount,
        "p95_tick_ms_before": load["before"]["p95_tick_ms"],
        "p95_tick_ms_after": load["after"]["p95_tick_ms"],
        **{key: load[key] for key in s1.LOAD_KEYS},
        "last_refusal_reason": record["server"]["admission_after"]["last_refusal_reason"],
    }


def first_attempt(b300: Any) -> dict[str, Any]:
    """The retest's first attempt at the ragged run: not a capture; its counts, recounted. A
    recording was either sent or refused; a sent one either got a final that covers its audio
    or not, and the client either gave up on it (``TIMED_OUT``) or did not."""
    name = C64_FIRST_ATTEMPT
    record = b300.records[name]
    problems = not_a_capture(b300, name)
    recordings = list(record["recordings"].values())
    sent = [e for e in recordings if e.get("sent") is True]
    no_final = [e for e in sent if e.get("terminal_final") is not True]
    timed = sum(e.get("in_flight_s") is not None for e in recordings)
    timed_out = [e for e in recordings if (e.get("client_error") or "").startswith(TIMED_OUT)]
    recount = {
        "recordings": len(recordings),
        "recordings_sent": len(sent),
        "sessions_acknowledged": sum(bool(e.get("server_session_id")) for e in sent),
        "recordings_with_terminal_final": sum(e.get("terminal_final") is True for e in recordings),
        "recordings_missing_final": sum(e.get("terminal_final") is not True for e in recordings),
        "recordings_without_partials": sum(not e.get("partials") for e in recordings),
    }
    sent_split = {
        "with_terminal_final": len(sent) - len(no_final),
        "without_terminal_final": len(no_final),
        "without_terminal_final_or_partial": sum(not e.get("partials") for e in no_final),
        "with_a_final_not_covering_the_audio": sum(bool(e.get("finals")) for e in no_final),
        "client_timed_out": len(timed_out),
    }
    refused = sum(REFUSAL in (e.get("client_error") or "") for e in recordings)
    stated = {key: record["counts"][key] for key in ATTEMPT_COUNTS}
    load = record["server"]["load"]
    agree = (
        recount == stated
        and recount["recordings_sent"] == load["admitted_during_run"]
        and refused == load["refused_during_run"]
        and refused + recount["recordings_sent"] == recount["recordings"]
        and not any(REFUSAL in (e.get("client_error") or "") for e in sent)
        and all(any(e is x for x in no_final) for e in timed_out)
        and timed == record["client"]["concurrency"]["sessions_timed"]
    )
    if not agree:
        raise Refused(
            f"{name}: recounted {recount}, {refused} refused, sent {sent_split} and {timed} "
            f"timed; it states {stated}, load {load} and "
            f"{record['client']['concurrency']['sessions_timed']} timed"
        )
    return {
        "record": name,
        "status": record["status"],
        "success": record["success"],
        "comparator_refuses_it": problems,
        "counts": stated,
        "refused_by_the_server": refused,
        "sent": sent_split,
        "observed_peak_in_flight": cc.peak_of(record),
        "sessions_timed": timed,
        "p95_tick_ms_before": load["before"]["p95_tick_ms"],
        "p95_tick_ms_after": load["after"]["p95_tick_ms"],
        **{key: load[key] for key in s1.LOAD_KEYS},
        "live_after": load["after"]["live"],
        "last_refusal_reason": record["server"]["admission_after"]["last_refusal_reason"],
    }


def retest_c64(b300: Any) -> dict[str, Any]:
    """The retest at concurrency 64: the failed first attempt, the two ragged runs compared, and
    the fixed run, refused; in the order they ran, and which server process ran each."""
    names = [C64_FIRST_ATTEMPT, *C64, C64_FIXED_REFUSED]
    records = {name: b300.records[name] for name in names}
    when = {
        name: tuple(
            datetime.strptime(r[key], "%Y-%m-%dT%H:%M:%S%z") for key in ("started", "finished")
        )
        for name, r in records.items()
    }
    order = sorted(names, key=lambda name: when[name])
    processes: list[Any] = []
    for name in order:
        if pid_of(records[name]) not in processes:
            processes.append(pid_of(records[name]))
    timeline = [
        {
            "record": name,
            "started": records[name]["started"],
            "finished": records[name]["finished"],
            "server_process": processes.index(pid_of(records[name])) + 1,
            "admitted_before_run": records[name]["server"]["load"]["before"]["admitted_total"],
        }
        for name in order
    ]
    gap = when[C64[0]][0] - when[C64_FIRST_ATTEMPT][1]
    return {
        "timeline": timeline,
        "server_processes": len(processes),
        "seconds_from_first_attempt_end_to_run1_start": int(gap.total_seconds()),
        "first_attempt": first_attempt(b300),
        "ragged_vs_ragged": ragged_vs_ragged(b300, *C64),
        "fixed_refused": refused_run(b300, C64_FIXED_REFUSED),
        "finals_digests": digests_of(b300, list(C64)),
    }


# --- the gate -----------------------------------------------------------------------------------


def level_runs(finals: Mapping[str, Any]) -> list[Any]:
    """The finals file rebuilt into the gate's level runs."""
    return [
        inv.LevelRun(
            inv.Level(level["slot"], level["concurrency"], level["churn_period_s"]),
            finals={
                f["stream_id"]: FinalRecord(
                    f["stream_id"], f["text"], tuple(tuple(w) for w in f["words"])
                )
                for f in level["finals"]
            },
            errors=dict(level["errors"]),
        )
        for level in finals["levels"]
    ]


def pair_counts(
    divergences: Sequence[Any], texts: Mapping[str, Mapping[str, str]], level: str, against: str
) -> dict[str, int]:
    """The streams that differ between two levels, split by kind; refused unless the parts sum."""
    found = [d for d in divergences if d.level == level and d.against == against]
    text = [d for d in found if d.kind == "text"]
    word_level = sum(
        s1.differs_in_words(texts[against][d.stream_id], texts[level][d.stream_id]) for d in text
    )
    out = {
        "streams": len({d.stream_id for d in found}),
        "text": len(text),
        "word_level": word_level,
        "punctuation_only": len(text) - word_level,
        "timing_only": sum(d.kind == "words" for d in found),
        "missing": sum(d.kind == "missing" for d in found),
    }
    if out["text"] + out["timing_only"] + out["missing"] != out["streams"]:
        raise Refused(f"{level} against {against}: the kinds {out} do not sum to the streams")
    return out


def gate_arm(b300: Any, arm: str, summary_arm: Mapping[str, Any]) -> dict[str, Any]:
    records = b300.records
    record, finals = records[gate("invariance", arm)], records[gate("finals", arm)]
    checked = records[gate("record-summary", arm)]
    smoke, spec = records[gate("smoke-check", arm)], records[gate("serve-spec", arm)]
    report = inv.assess(level_runs(finals))
    digests = report.digests
    verdicts = {
        "the invariance record": record["verdict"],
        "the finals": finals["verdict"],
        "the record summary": checked["verdict"],
        "the gate summary": summary_arm["verdict"],
    }
    for where, verdict in verdicts.items():
        if verdict != report.verdict:
            raise Refused(f"{arm}: assess gives {report.verdict}; {where} says {verdict}")
    level_digests = {
        "the invariance record": {level["slot"]: level["digest"] for level in record["levels"]},
        "the finals": {level["slot"]: level["digest"] for level in finals["levels"]},
        "the record summary": checked["digests"],
    }
    for where, got in level_digests.items():
        if got != digests:
            raise Refused(f"{arm}: assess gives the digests {digests}; {where} says {got}")
    if summary_arm["distinct_digests"] != len(set(digests.values())):
        raise Refused(f"{arm}: the gate summary counts {summary_arm['distinct_digests']} digests")
    if [d.to_json_dict() for d in report.divergences] != record["divergences"]:
        raise Refused(f"{arm}: assess's divergences are not the invariance record's")
    if report.exit_code != record["exit_code"]:
        raise Refused(f"{arm}: assess exits {report.exit_code}, the record {record['exit_code']}")
    argv = spec["derived_from"]
    padding, occupancy = arm.split("-")
    churned = [lv["churn_period_s"] for lv in record["levels"] if lv["churn_period_s"] is not None]
    if (
        argv[argv.index("--padding") + 1] != padding
        or summary_arm["padding"] != padding
        or summary_arm["occupancy"] != occupancy
        or bool(churned) != (occupancy == "churn")
    ):
        raise Refused(f"{arm}: the serve spec, the gate summary or the levels say otherwise")
    if smoke["budget_ms"] != spec["budget_ms"]:
        raise Refused(f"{arm}: smoke budget {smoke['budget_ms']}, serve spec {spec['budget_ms']}")
    texts = {
        level["slot"]: {f["stream_id"]: f["text"] for f in level["finals"]}
        for level in finals["levels"]
    }
    pairs = {
        f"{level}_vs_{against}": pair_counts(report.divergences, texts, level, against)
        for level, against in inv.PAIRS
    }
    against_one = [pairs[f"{slot}_vs_1"]["streams"] for slot in inv.SLOTS if slot != "1"]
    readings = {
        "the gate summary": (
            summary_arm["differing_vs_1_32a_32b_max"],
            summary_arm["differing_32b_vs_32a"],
        ),
        "the record summary": (
            [checked["streams_differing"][f"{slot}_vs_1"] for slot in inv.SLOTS if slot != "1"],
            checked["streams_differing"]["32b_vs_32a"],
        ),
    }
    for where, reading in readings.items():
        if reading != (against_one, pairs["32b_vs_32a"]["streams"]):
            raise Refused(f"{arm}: the streams differing are {pairs}; {where} read {reading}")
    return {
        "padding": padding,
        "occupancy": occupancy,
        "execution": spec["execution"],
        "verdict": report.verdict,
        "exit_code": report.exit_code,
        "levels": {
            run.level.slot: {
                "concurrency": run.level.concurrency,
                "churn_period_s": run.level.churn_period_s,
                "streams": len(run.finals),
                "errors": len(run.errors),
                "digest": run.digest,
            }
            for run in report.runs
        },
        "distinct_digests": len(set(digests.values())),
        "differing": pairs,
        "distinct_streams_vs_1": len({d.stream_id for d in report.divergences if d.against == "1"}),
        "smoke": {key: smoke[key] for key in SMOKE_KEYS},
        "monitor": {key: summary_arm["monitor"][key] for key in MONITOR_KEYS},
    }


def gate_section(b300: Any) -> dict[str, Any]:
    summary = b300.records[GATE_SUMMARY]
    by_name = {entry["arm"]: entry for entry in summary["arms"]}
    if set(by_name) != set(ARMS):
        raise Refused(f"{GATE_SUMMARY}: its arms are {sorted(by_name)}")
    arms = {arm: gate_arm(b300, arm, by_name[arm]) for arm in ARMS}
    cross = {
        padding: {
            slot: arms[f"{padding}-churn"]["levels"][slot]["digest"]
            == arms[f"{padding}-const"]["levels"][slot]["digest"]
            for slot in inv.SLOTS
        }
        for padding in ("fixed", "ragged")
    }
    stored = summary["cross_restart_digest_agreement"]
    for padding, key in (
        ("fixed", "fixed-churn_vs_fixed-const"),
        ("ragged", "ragged-churn_vs_ragged-const"),
    ):
        if any(cross[padding][slot] is not value for slot, value in stored[key].items()):
            raise Refused(
                f"{GATE_SUMMARY}: {key} is {stored[key]}; the digests give {cross[padding]}"
            )
    corpus = b300.records[gate("invariance", ARMS[0])]["corpus"]
    settings = summary["settings_by_arm"]
    return {
        "commit": settings[ARMS[0]]["git_head"],
        "bucket": {arm: settings[arm]["bucket"] for arm in ARMS},
        "max_level": {arm: settings[arm]["max_level"] for arm in ARMS},
        "corpus": {"id": corpus["id"], "utterances": corpus["utterances"]},
        "runbook_exit_code": summary["exit_code"],
        "arms": arms,
        "cross_restart": cross,
    }


# --- the card, and the summary ---------------------------------------------------------------


def card_uuids(node: Any, key: str = "") -> set[str]:
    """Every GPU UUID ``node`` names under a key whose name holds ``uuid``, without NVIDIA's
    ``GPU-`` prefix (so ``GPU-b43f...`` and torch's ``b43f...`` are one card)."""
    if isinstance(node, Mapping):
        return {u for k, v in node.items() for u in card_uuids(v, str(k))}
    if isinstance(node, (list, tuple)):
        return {u for v in node for u in card_uuids(v, key)}
    if isinstance(node, str) and "uuid" in key.lower():
        return set(CARD_UUID.findall(node))
    return set()


def cards_named(*directories: Any) -> dict[str, Any]:
    """How many cards the records of these directories name, and how many of them name one."""
    named = {
        f"{d.directory.name}/{name}": card_uuids(record)
        for d in directories
        for name, record in d.records.items()
    }
    return {
        "cards": len(set().union(*named.values())),
        "records": len(named),
        "records_naming_a_card": sum(bool(uuids) for uuids in named.values()),
    }


def card_section(b300: Any) -> dict[str, Any]:
    """One card or several: every B300 record's card UUID, the gate's preflight's included."""
    card, model = b300.records[CARD], b300.records[MODEL]
    names = [*B300_CAPTURES, *B300_FAILED]
    uuids = {cc.settings_of(b300.records[n])["gpu_uuid"] for n in names}
    uuids |= {card["gpu_uuid"], card["gpu_uuid_reported"]}
    uuids |= {s["gpu_uuid"] for s in b300.records[GATE_SUMMARY]["settings_by_arm"].values()}
    if None in uuids:
        raise Refused("a B300 capture or the gate's preflight names no card")
    named = cards_named(b300)
    # the fields read by name, and every card any record names under a uuid key
    uuids = {u.removeprefix("GPU-") for u in uuids} | set().union(
        *(card_uuids(record) for record in b300.records.values())
    )
    revisions = {cc.settings_of(b300.records[n])["revision"] for n in names} | {model["revision"]}
    if len(revisions) != 1:
        raise Refused(f"the B300 records name model revisions {sorted(revisions)}")
    devices = {cc.settings_of(b300.records[n])["device_name"] for n in names}
    return {
        **named,
        "cards": len(uuids),
        "device_names": sorted(devices),
        "model": model["model"],
        "revision": model["revision"],
        "nemo_sha256": model["nemo_sha256"],
    }


def a6000_section(a6000: Any, c8: Any, gate_smoke: Any) -> dict[str, Any]:
    """The A6000's cards, over its three directories, and the server processes that ran its
    unpadded pair at concurrency 32."""
    captures = [n for n in a6000.records if n.startswith("server-")]
    devices = {cc.settings_of(a6000.records[n])["device_name"] for n in captures}
    devices |= {cc.settings_of(c8.records[n])["device_name"] for n in C8}
    devices |= {gate_smoke.records[smoke("readyz", b)]["device_name"] for b in SMOKE_BUCKETS}
    return {
        "card": {**cards_named(a6000, c8, gate_smoke), "device_names": sorted(devices)},
        "ragged_c32_pair": {
            "captures": list(A6000_C32),
            # says() refused each capture unless it recorded this concurrency
            "concurrency_configured": A6000_C32_SAYS[2],
            "server_processes": server_processes(a6000, A6000_C32),
        },
    }


def argv_value(argv: Sequence[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def gate_smoke_section(gate_smoke: Any) -> dict[str, Any]:
    """Each smoke-stopped attempt of the A6000 runbook: its bucket, budget, highest p95 tick and
    problems; refused unless the serve spec and the record's name agree on the bucket, the smoke
    check's budget is the serve spec's, and the padding is fixed."""
    out = {}
    for bucket in SMOKE_BUCKETS:
        check = gate_smoke.records[smoke("smoke-check", bucket)]
        spec = gate_smoke.records[smoke("serve-spec", bucket)]
        readyz = gate_smoke.records[smoke("readyz", bucket)]
        argv = spec["derived_from"]
        buckets = (argv_value(argv, "--bucket"), spec["spec"]["batch_size"])
        if buckets != (str(bucket), bucket):
            raise Refused(f"{smoke('serve-spec', bucket)}: its buckets are {buckets}, not {bucket}")
        if argv_value(argv, "--padding") != "fixed":
            raise Refused(f"{smoke('serve-spec', bucket)}: padding {argv_value(argv, '--padding')}")
        if check["budget_ms"] != spec["budget_ms"]:
            raise Refused(
                f"{smoke('smoke-check', bucket)}: budget {check['budget_ms']}, serve spec "
                f"{spec['budget_ms']}"
            )
        out[f"b{bucket}"] = {
            "bucket": bucket,
            "padding": "fixed",
            "execution": spec["execution"],
            "device_name": readyz["device_name"],
            "budget_ms": check["budget_ms"],
            "max_p95_tick_ms": check["max_p95_tick_ms"],
            "max_degradation_level": check["max_degradation_level"],
            "refused_total_after": check["refused_total_after"],
            "problems": check["problems"],
            "stopped_before_the_gate": bool(check["problems"]),
        }
    return out


def inputs_of(inputs: Any) -> dict[str, Any]:
    return {
        "directory": inputs.directory.name,
        "manifest_sha256": inputs.manifest_sha256,
        "sha256_raw": {name: e["sha256_raw"] for name, e in inputs.manifest["records"].items()},
    }


class Loaded(NamedTuple):
    """The four directories' records, verified."""

    b300: Any
    c8: Any
    a6000: Any
    gate_smoke: Any


def load_inputs(
    b300_dir: Path = B300_RECORDS,
    c8_dir: Path = C8_RECORDS,
    a6000_dir: Path = A6000_RECORDS,
    smoke_dir: Path = SMOKE_RECORDS,
) -> Loaded:
    b300, c8 = load_directory(b300_dir), load_directory(c8_dir)
    a6000, gate_smoke = load_directory(a6000_dir), load_directory(smoke_dir)
    check_links(b300)
    for name, want in {**B300_CAPTURES, **B300_FAILED}.items():
        says(b300, name, want)
    for name in C8:
        says(c8, name, C8_SAYS)
    for name in A6000_C32:
        says(a6000, name, A6000_C32_SAYS)
    return Loaded(b300, c8, a6000, gate_smoke)


def assemble(b300: Any, c8: Any, a6000: Any, gate_smoke: Any) -> dict[str, Any]:
    complete = list(B300_CAPTURES)
    at_26eaebb = [*FIXED.values(), RAGGED, RAGGED_REPEAT]
    run1_process = pid_of(b300.records[OVERLOAD])
    return {
        "question": "Step 1 on nvidia/nemotron-speech-streaming-en-0.6b on one B300 (with a "
        "retest at concurrency 64), and the A6000's unpadded server run twice at concurrency 8: "
        "does the fixed-shape server give one "
        "answer at every occupancy, in eager and graph execution, and does the unpadded server "
        "give the same answer twice?",
        "not_a_row": "Derived by scripts/step1_b300_summary.py from the exploratory records in "
        f"rows/exploratory/{b300.directory.name}/, {c8.directory.name}/, "
        f"{a6000.directory.name}/ and {gate_smoke.directory.name}/. Not a harness row: "
        "nothing here is a benchmark result or reads a gate.",
        "inputs": {
            "b300": inputs_of(b300),
            "a6000_c8": inputs_of(c8),
            "a6000": inputs_of(a6000),
            "a6000_gate_smoke": inputs_of(gate_smoke),
        },
        "rules": RULES,
        "b300": {
            "card": card_section(b300),
            "setting": common_setting(b300, complete),
            "captures": {name: capture_facts(b300, name) for name in complete},
            "frozen_rule": cc.FROZEN_RULE,
            "frozen": frozen_section(b300),
            "fixed_eager_vs_fixed_graphs": eager_vs_graphs(b300),
            "ragged_vs_fixed": ragged_vs_fixed(b300),
            "ragged_vs_ragged": ragged_vs_ragged(b300, RAGGED, RAGGED_REPEAT),
            "finals_digests": digests_of(b300, at_26eaebb),
            "overload": {
                "run1": OVERLOAD,
                "run2_refused": {
                    **refused_run(b300),
                    "same_server_process_as_run1": pid_of(b300.records[REFUSED_RUN])
                    == run1_process,
                },
                "repeatability_measured": b300.records[REFUSED_RUN]["success"] is True,
            },
            "retest_c64": retest_c64(b300),
            "gate": gate_section(b300),
        },
        "a6000_c8": {
            "setting": common_setting(c8, list(C8)),
            "captures": {name: capture_facts(c8, name) for name in C8},
            "ragged_vs_ragged": ragged_vs_ragged(c8, *C8),
            "finals_digests": digests_of(c8, list(C8)),
        },
        "a6000": a6000_section(a6000, c8, gate_smoke),
        "a6000_gate_smoke": gate_smoke_section(gate_smoke),
    }


def render(summary: Mapping[str, Any]) -> str:
    try:
        return s1.render(summary)
    except s1.Refused as exc:
        raise Refused(str(exc)) from None


def build(
    b300_dir: Path = B300_RECORDS,
    c8_dir: Path = C8_RECORDS,
    a6000_dir: Path = A6000_RECORDS,
    smoke_dir: Path = SMOKE_RECORDS,
) -> str:
    return render(assemble(*load_inputs(b300_dir, c8_dir, a6000_dir, smoke_dir)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--b300", type=Path, default=B300_RECORDS)
    parser.add_argument("--a6000-c8", type=Path, default=C8_RECORDS)
    parser.add_argument("--a6000", type=Path, default=A6000_RECORDS)
    parser.add_argument("--a6000-gate-smoke", type=Path, default=SMOKE_RECORDS)
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--out", type=Path, default=None, help="a new file; refused if it exists")
    where.add_argument("--check", type=Path, default=None, help="exit 1 unless the output is it")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.out is not None and os.path.lexists(args.out):
        print(f"step1_b300_summary: {args.out} exists; it is not written over", file=sys.stderr)
        return EXIT_REFUSED
    try:
        text = build(args.b300, args.a6000_c8, args.a6000, args.a6000_gate_smoke)
    except (Refused, OSError, ValueError, KeyError) as exc:
        print(f"step1_b300_summary: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.check is not None:
        same = args.check.read_bytes() == text.encode("utf-8")
        print(f"step1_b300_summary: {'the same bytes as' if same else 'DIFFERS from'} {args.check}")
        return EXIT_OK if same else EXIT_DIFFERS
    if args.out is None:
        sys.stdout.write(text)
        return EXIT_OK
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
