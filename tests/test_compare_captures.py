# SPDX-License-Identifier: Apache-2.0
"""``scripts/compare_captures.py``: identity on both channels, the frozen rule, the places.

The captures here are built in the shape ``probes/server_frozen_answers.py`` writes, with a
digest computed from their own recordings. ``tests/test_server_frozen_answers.py`` runs the same
comparison on captures the probe actually took from a server, so the shape is also checked
against the real writer.

The mutation-style test takes a pair that differs only in one word's timing and shows three
things: the comparator does not call it identical; the text channel alone would have; and a copy
of the comparator with its timing comparison and its digest cross-check removed does call it
identical, so the assertions are able to fail.

What the captures observed is in them as the probe writes it: the padding in the server's
command line and banner, the attention context ``/readyz`` read off the built encoder, the
revision in the hub cache the server's own environment names, the server process and its
starting tick, each session's time in flight on the wire and the peak those times give, the
client code that took it, the server code that served it (``/readyz`` ``code``, contract C7),
and whether the client's tree was clean. A capture's peak is what its three recordings can
reach at its concurrency, min(concurrency, 3), unless ``peak`` says otherwise (a server that
held sessions back), and its recordings' ``in_flight_s`` give exactly that peak.
``observed=False`` makes a capture whose padding is only declared; ``tree`` one whose
``tracked_files_modified`` is not false; ``fake`` one a test double took; ``confidence`` one
taken with word confidence on, every word carrying a confidence as its fourth element.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _load("compare_captures", "scripts/compare_captures.py")

MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
PIN = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
BUCKET = 8

FIXED = {
    "r0": ("I'm from the cutter.", [("I'm", 0, 160), ("from", 160, 320), ("the", 320, 400),
                                    ("cutter.", 400, 640)]),
    "r1": ("No, sir.", [("No,", 0, 160), ("sir.", 160, 320)]),
    "r2": ("", []),
}  # fmt: skip
REFERENCES = {"r0": "i'm from the cutter", "r1": "no sir", "r2": "oh"}
#: The ragged side: r1 differs in a word, r0 only in the end of its last word.
RAGGED = {
    **FIXED,
    "r0": (FIXED["r0"][0], [*FIXED["r0"][1][:3], ("cutter.", 400, 720)]),
    "r1": ("Now, sir.", [("Now,", 0, 160), ("sir.", 160, 320)]),
}


#: probes/server_frozen_answers.py's HUB_FROM_SERVER_ENVIRON, written out: the one hub_dir_source
#: under which the comparator takes the revision as observed of the server.
SERVER_HUB = "the server's environment (/proc/<pid>/environ), by huggingface_hub's rule"
#: The client code every capture here was taken with (``client.<key>``).
CODE = {"verbatim_commit": "c" * 40, "probe_sha256": "p" * 64, "bench_client_sha256": "b" * 64}
#: The server code every capture here was served by (``server.code.reported_by_server``,
#: contract C7): placeholders, not directories on any machine.
SERVER_CODE = {
    "verbatim_path": "/checkout/src/verbatim",
    "bench_path": "/checkout/bench/src/verbatim_bench",
}
#: The confidence a capture taken with word confidence on carries on every word.
CONFIDENCE = 0.5

_RUN = itertools.count()


def spans(recordings: int, peak: int) -> list[list[float]]:
    """``in_flight_s`` for ``recordings`` sessions whose most in flight at once is ``peak``:
    groups of ``peak`` that start within 0.125 s of each other and all end before the next group
    starts."""
    return [
        [group + j / (8 * peak), group + 0.875]
        for k in range(recordings)
        for group, j in [divmod(k, peak)]
    ]


def make(
    padding: str = "fixed",
    concurrency: int = 32,
    answers: dict[str, tuple[str, list[tuple[Any, ...]]]] | None = None,
    *,
    observed: bool = True,
    peak: int | None = None,
    tree: Any = False,
    fake: Any = None,
    confidence: str = "off",
) -> dict[str, Any]:
    """A capture in the probe's shape; each call is a separate run with its own timestamps.
    Its client peaked at min(concurrency, recordings) in flight unless ``peak`` says otherwise,
    and every recording's ``in_flight_s`` gives that peak; it ran from a clean tree unless
    ``tree`` says otherwise; ``fake`` is its ``fake_pipeline`` stamp, absent when None;
    ``confidence`` is its word confidence, and when it is not off every word carries
    ``CONFIDENCE`` as its fourth element, as the probe writes a "c" the wire carried."""
    answers = FIXED if answers is None else answers
    peak = min(concurrency, len(answers)) if peak is None else peak
    run = next(_RUN)
    on = confidence != "off"
    recordings = {
        rid: {
            "n": n,
            "text": text,
            "words": [[*w[:3], CONFIDENCE] if on else list(w) for w in words],
            "pcm_sha256": f"audio-of-{rid}",
            "reference": REFERENCES[rid],
            "in_flight_s": span,
        }
        for n, ((rid, (text, words)), span) in enumerate(
            zip(answers.items(), spans(len(answers), peak), strict=True)
        )
    }
    seen = padding if observed else None
    record: dict[str, Any] = {} if fake is None else {"fake_pipeline": fake}
    record |= {
        "record": cc.RECORD_KIND,
        "success": True,
        "status": "complete",
        "failures": [],
        "warnings": [],
        "started": f"run-{run:04d}-started",
        "finished": f"run-{run:04d}-finished",
        "server": {
            "model": {"reported_by_server": MODEL, "declared": MODEL},
            "model_revision": {
                "observed_in_hf_cache": PIN,
                "declared": PIN,
                "hub_dir_source": SERVER_HUB,
            },
            "chunk_ms": {"reported_by_server": 160, "declared": 160},
            "dtype": {"reported_by_server": "bfloat16", "declared": "bfloat16"},
            "execution": {
                "reported_by_server": "eager",
                "observed_in_cmdline": "eager",
                "declared": "eager",
            },
            "matmul_precision": {"declared": "high", "installed_default": "high"},
            "bucket": {
                "reported_by_server": BUCKET,
                "observed_in_cmdline": BUCKET,
                "declared": BUCKET,
            },
            "padding": {
                "declared": padding,
                "observed_in_cmdline": seen,
                "observed_in_server_log": seen,
            },
            "att_context": {
                "observed_in_readyz": [70, 1],
                "derived_from_cmdline": [70, 1],
                "observed_in_server_log": [70, 1],
                "declared": [70, 1],
            },
            "word_confidence": {
                "reported_by_server": confidence,
                "decoder_step_confidence": on,
                "observed_in_cmdline": confidence,
                "observed_in_server_log": confidence,
                "declared": confidence,
            },
            "decoder_graphs": {
                "reported_by_server": False,
                "observed_in_cmdline": False,
                "observed_in_server_log": False,
                "declared": False,
            },
            "gpu": {"before": {"uuid": "GPU-test-double"}},
            "code": {"reported_by_server": dict(SERVER_CODE)},
            "process": {"before": {"pid": 4242}},
            "readyz_before": {
                "pipeline": "cache_aware_rnnt",
                "biasing": False,
                "nemo_version": "n",
                "torch_version": "t",
                "device_name": "d",
                "tick_id": 1000 * (run + 1),
            },
        },
        "client": {
            "concurrency": {"configured": concurrency, "observed_peak_in_flight": peak},
            "seed": 1,
            **CODE,
            "tracked_files_modified": tree,
            "code_outside_checkout": [],
        },
        "recordings": recordings,
    }
    record["finals_digest"] = cc.digest_of(record)
    return record


def loaded(record: dict[str, Any], name: str = "capture.json") -> Any:
    """As ``load_capture`` would give it, with the SHA-256 of these bytes."""
    raw = json.dumps(record, sort_keys=True).encode("utf-8")
    return cc.Loaded(name, hashlib.sha256(raw).hexdigest(), record)


def _set(record: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    node = record
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return record


# --- the frozen rule ---


def test_identical_fixed_captures_without_a_control_are_not_called_frozen() -> None:
    report = cc.compare(loaded(make(concurrency=32)), loaded(make(concurrency=1)))
    assert report["identity"]["identical"] == 3 and report["identity"]["digests_match"]
    assert report["frozen"] is False
    assert report["verdict"].startswith("IDENTICAL, FROZEN NOT CLAIMED")


def test_frozen_needs_two_concurrencies_and_a_ragged_control_that_differs() -> None:
    report = cc.compare(
        loaded(make(concurrency=32)),
        loaded(make(concurrency=1)),
        control=loaded(make("ragged", 32, RAGGED), "ragged.json"),
    )
    assert report["frozen"] is True
    assert report["verdict"].startswith(
        "FROZEN: the fixed captures at concurrency 32 and 1 (observed peak in flight 3 and 1)"
    )
    assert report["third_capture_identity"]["text_differs"] == 1
    assert report["third_capture_identity"]["timing_only"] == 1


def test_a_ragged_control_that_matches_does_not_make_frozen() -> None:
    report = cc.compare(
        loaded(make(concurrency=32)),
        loaded(make(concurrency=1)),
        control=loaded(make("ragged", 32), "ragged.json"),
    )
    assert report["frozen"] is False
    assert report["verdict"].startswith("NOT SHOWN FROZEN")


def test_a_ragged_control_that_differs_only_in_word_timings_differs() -> None:
    """Identity is on both channels: a ragged capture whose text is the fixed one's word for
    word, and one word's end is not, has shown a batch effect."""
    control = make("ragged", 32, {**FIXED, "r0": RAGGED["r0"]})
    report = cc.compare(
        loaded(make(concurrency=32)), loaded(make(concurrency=1)), control=loaded(control)
    )
    assert report["third_capture_identity"]["text_differs"] == 0
    assert report["frozen"] is True
    assert report["verdict"].endswith(
        "and the ragged capture differs on 0 in text and 1 in word timings only"
    )


def test_a_match_at_one_concurrency_is_repeatability_not_frozen() -> None:
    report = cc.compare(
        loaded(make(concurrency=32)),
        loaded(make(concurrency=32)),
        control=loaded(make("ragged", 32, RAGGED), "ragged.json"),
    )
    assert report["frozen"] is False
    assert report["verdict"].startswith(
        "IDENTICAL AT ONE CONCURRENCY (3): the fixed captures at concurrency 32 and 32 "
        "(observed peak in flight 3 and 3) match on all 3 recordings, but both peaked at 3"
    )


def test_two_configured_concurrencies_that_peaked_alike_are_one_occupancy_not_frozen(
    tmp_path: Path,
) -> None:
    """Configured at 32 and 8, both observed at a peak of 3 in flight (three recordings never
    have more than three in flight), with a ragged control that differs. The two fixed captures
    ran at one occupancy: no FROZEN and no same-shape verdict, and each verdict names both the
    configured and the observed values."""
    c32 = loaded(make(concurrency=32, peak=3), "fixed-c32.json")
    c8 = loaded(make(concurrency=8, peak=3), "fixed-c8.json")
    control = loaded(make("ragged", 32, RAGGED, peak=3), "ragged.json")
    report = cc.compare(c32, c8, control=control)
    assert report["frozen"] is False
    assert report["verdict"] == (
        "IDENTICAL AT ONE CONCURRENCY (3): the fixed captures at concurrency 32 and 8 (observed "
        "peak in flight 3 and 3) match on all 3 recordings, but both peaked at 3 in flight: "
        "run-to-run repeatability, not batch independence; frozen not claimed"
    )
    assert report["captures"]["a"]["observed_peak_in_flight"] == 3
    assert report["captures"]["a"]["concurrency"] == 32

    out = tmp_path / "places.json"
    report = cc.compare(c32, loaded(make("ragged", 32, RAGGED, peak=3)), repeat=c8, places_out=out)
    assert report["frozen"] is False
    verdict = json.loads(out.read_text())["runs"]["bfloat16"]["repeat_verdict"]
    assert verdict == (
        "identical at one concurrency (3) only: both captures peaked at 3 in flight (configured "
        "32 and 8); not shown across occupancy"
    )
    derived = _step1_places().derive(json.loads(out.read_text()), frozenset(), draws=20)
    assert derived["runs"]["bfloat16"]["repeat_verdict_accepted"] is False


def test_the_observed_peaks_decide_not_the_configured_concurrency() -> None:
    """The other direction: configured alike, observed apart (a server that held sessions
    back), is two occupancies."""
    report = cc.compare(
        loaded(make(concurrency=32, peak=3)),
        loaded(make(concurrency=32, peak=1)),
        control=loaded(make("ragged", 32, RAGGED)),
    )
    assert report["frozen"] is True
    assert report["verdict"].startswith(
        "FROZEN: the fixed captures at concurrency 32 and 32 (observed peak in flight 3 and 1)"
    )


@pytest.mark.parametrize(
    "peak", [None, 0, True, "8", 8.0], ids=["missing", "zero", "bool", "string", "float"]
)
def test_a_peak_that_is_not_recorded_withholds_frozen_and_the_same_shape_verdict(
    tmp_path: Path, peak: Any
) -> None:
    """A capture that does not record a whole positive peak shows no occupancy at all."""
    unrecorded = _set(make(concurrency=1), "client.concurrency.observed_peak_in_flight", peak)
    # Two unrecorded peaks are not one occupancy either, though None equals None.
    both = cc.occupancy(loaded(unrecorded, "x.json"), loaded(copy.deepcopy(unrecorded), "y.json"))
    assert both.unrecorded == ("x.json", "y.json") and both.one is False
    report = cc.compare(
        loaded(make(concurrency=32)),
        loaded(unrecorded, "no-peak.json"),
        control=loaded(make("ragged", 32, RAGGED)),
    )
    assert report["frozen"] is False
    assert report["verdict"].startswith("IDENTICAL, OCCUPANCY NOT OBSERVED"), report["verdict"]
    assert "the observed peak in flight is not recorded in no-peak.json" in report["verdict"]

    out = tmp_path / "places.json"
    cc.compare(
        loaded(make("fixed", 32)),
        loaded(make("ragged", answers=RAGGED)),
        repeat=loaded(unrecorded, "no-peak.json"),
        places_out=out,
    )
    verdict = json.loads(out.read_text())["runs"]["bfloat16"]["repeat_verdict"]
    assert verdict == (
        "identical, but the observed peak in flight is not recorded in no-peak.json (configured "
        "32 and 1): not shown across occupancy"
    )


def test_a_text_difference_between_fixed_captures_is_not_frozen() -> None:
    changed = {**FIXED, "r1": RAGGED["r1"]}
    report = cc.compare(
        loaded(make(concurrency=32)),
        loaded(make(concurrency=1, answers=changed)),
        control=loaded(make("ragged", 32, RAGGED), "ragged.json"),
    )
    assert report["frozen"] is False
    assert report["identity"]["text_differs_ids"] == ["r1"]
    assert report["verdict"].startswith("NOT FROZEN: of 3 recordings, 1 in text and 0 in word")


# --- the timing channel: a timing-only difference is not identical ---


TIMING_ONLY = {**FIXED, "r0": RAGGED["r0"]}


def test_a_timing_only_difference_is_not_reported_as_identical() -> None:
    a, b = make(concurrency=32), make(concurrency=1, answers=TIMING_ONLY)
    report = cc.compare(loaded(a), loaded(b))
    same = report["identity"]
    assert same["identical"] == 2 and same["timing_only"] == 1 and same["text_differs"] == 0
    assert same["timing_only_ids"] == ["r0"]
    assert same["per_recording"]["r0"] == {"text": True, "timings": False}
    assert same["digests_match"] is False
    assert report["frozen"] is False
    assert report["verdict"].startswith("NOT FROZEN: of 3 recordings, 0 in text and 1 in word")


def _mutant(*edits: tuple[str, str]) -> types.ModuleType:
    """A copy of the comparator with source edits applied; each edit must match once."""
    source = (REPO / "scripts" / "compare_captures.py").read_text(encoding="utf-8")
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType("compare_captures_mutant")
    module.__dict__["__file__"] = "compare_captures_mutant"
    sys.modules[module.__name__] = module
    exec(compile(source, "compare_captures_mutant", "exec"), module.__dict__)
    return module


BLIND_TIMINGS = ('same_timings = triples(x["words"]) == triples(y["words"])', "same_timings = True")
NO_DIGEST_CHECK = ("if digests_match != (identical == len(per)):", "if False:")


def test_mutation_the_timing_only_case_is_discriminating() -> None:
    a, b = make(concurrency=32), make(concurrency=1, answers=TIMING_ONLY)
    # The text channel alone cannot see this difference ...
    assert all(a["recordings"][r]["text"] == b["recordings"][r]["text"] for r in FIXED)
    # ... a comparator blind to timings is stopped by the digest cross-check ...
    blind = _mutant(BLIND_TIMINGS)
    with pytest.raises(RuntimeError, match="the comparison and the digest disagree"):
        blind.compare(blind.Loaded("a", "x", a), blind.Loaded("b", "y", b))
    # ... and with the cross-check gone too, it calls the pair identical: exactly what the test
    # above asserts does not happen, so that test can fail.
    blinder = _mutant(BLIND_TIMINGS, NO_DIGEST_CHECK)
    report = blinder.compare(blinder.Loaded("a", "x", a), blinder.Loaded("b", "y", b))
    assert report["identity"]["identical"] == 3 and report["identity"]["timing_only"] == 0
    assert report["verdict"].startswith("IDENTICAL, FROZEN NOT CLAIMED")


# --- places between fixed and ragged ---


def _step1_places() -> Any:
    return _load("step1_places", "scripts/step1_places.py")


@pytest.mark.parametrize("ragged_first", [False, True], ids=["fixed-first", "ragged-first"])
def test_fixed_against_ragged_writes_places_step1_places_reads(
    tmp_path: Path, ragged_first: bool
) -> None:
    fixed = loaded(make("fixed", 32), "fixed.json")
    ragged = loaded(make("ragged", 32, RAGGED), "ragged.json")
    repeat = loaded(make("fixed", 1), "fixed-c1.json")
    out = tmp_path / "places.json"
    pair = (ragged, fixed) if ragged_first else (fixed, ragged)
    report = cc.compare(*pair, repeat=repeat, places_out=out)
    assert report["verdict"].startswith("POSITIVE CONTROL PRESENT")
    assert (
        "FROZEN: the fixed captures at concurrency 32 and 1 (observed peak in flight 3 and 1)"
        in report["verdict"]
    )
    assert report["frozen"] is True
    assert report["places_written"] == str(out)

    record = json.loads(out.read_text())
    assert record["source"] == "scripts/compare_captures.py" and record["timing_unit"] == "ms"
    run = record["runs"]["bfloat16"]
    assert run["repeat_verdict"] == "run-to-run identical in the same shape"
    assert run["positive_control"].startswith("present")
    arm = run["arms"][cc.ARM]
    assert arm["checked"] == 3
    assert arm["text_divergent"] == 1 and arm["timing_only_divergent"] == 1
    # Side a is the fixed (served) answer whichever order the captures came in.
    assert arm["transcripts"]["r1"] == {"a": "No, sir.", "b": "Now, sir."}
    assert arm["transcripts"]["r0"] == {"a": FIXED["r0"][0]}
    timing = next(d for d in arm["divergences"] if d["librispeech_id"] == "r0")
    assert timing["a_timings"][-1] == ["cutter.", 400, 640]
    assert timing["b_timings"][-1] == ["cutter.", 400, 720]
    assert record["captures"]["a_fixed"]["path"] == "fixed.json"
    assert record["captures"]["b_ragged"]["path"] == "ragged.json"
    assert record["captures"]["repeat_fixed"]["path"] == "fixed-c1.json"
    assert record["references"] == REFERENCES
    assert (record["chunk_ms"], record["bucket"], record["execution"]) == (160, BUCKET, "eager")
    assert record["batch"] == record["bucket"] == BUCKET  # contract C4 names both
    assert record["captures"]["repeat_fixed"]["observed_peak_in_flight"] == 1
    assert record["captures"]["a_fixed"]["observed_peak_in_flight"] == 3
    assert record["verbatim_commit"] == CODE["verbatim_commit"]
    assert {k: record["captures"]["b_ragged"][k] for k in CODE} == CODE
    assert record["captures"]["b_ragged"]["server_code"] == SERVER_CODE
    assert record["matmul_precision"] == "high" and record["att_context_size"] == [70, 1]
    assert record["model_revision"] == PIN
    # Each divergence names its recording's own place in the corpus order.
    assert {d["librispeech_id"]: d["n"] for d in arm["divergences"]} == {"r0": 0, "r1": 1}
    # The places span every input: the earliest start (the fixed capture, taken first here)
    # to the latest finish (the repeat, taken last).
    assert record["started"] == fixed.record["started"] < repeat.record["started"]
    assert record["finished"] == repeat.record["finished"] > fixed.record["finished"]
    assert record["runs"]["bfloat16"]["finished"] == repeat.record["finished"]
    assert [record["captures"][k]["padding_source"] for k in ("a_fixed", "b_ragged")] == [
        "observed in the server's command line"
    ] * 2

    # The scorer reads it unchanged, and scores it: the repeat verdict is the one it accepts.
    derived = _step1_places().derive(record, frozenset(), draws=20)
    scored_run = derived["runs"]["bfloat16"]
    assert scored_run["repeat_verdict_accepted"] is True
    scored = scored_run["arms"][cc.ARM]
    assert scored["served_side"] == "a"
    assert scored["flag"]["places"] == 1
    assert scored["places"]["alone_right"] == 1  # analyse's "alone" is side a, the fixed answer
    assert scored["places"]["batch_right"] == 0
    assert scored["timing_only_shift_ms"]["largest_shift_counts"] == {"80": 1}
    assert derived["record_settings"]["bucket"] == BUCKET


def test_without_a_repeat_the_places_say_so_and_the_scorer_withholds_scores(
    tmp_path: Path,
) -> None:
    out = tmp_path / "places.json"
    report = cc.compare(
        loaded(make("fixed")), loaded(make("ragged", answers=RAGGED)), places_out=out
    )
    assert report["frozen"] is False
    assert report["verdict"].endswith("FROZEN NOT ASSESSED: no second fixed capture was given")
    record = json.loads(out.read_text())
    assert record["runs"]["bfloat16"]["repeat_verdict"].startswith("not measured")
    derived = _step1_places().derive(record, frozenset(), draws=20)
    assert derived["runs"]["bfloat16"]["repeat_verdict_accepted"] is False
    assert derived["runs"]["bfloat16"]["arms"][cc.ARM]["flag"] is None


def test_a_repeat_that_differs_is_not_the_same_shape_verdict(tmp_path: Path) -> None:
    out = tmp_path / "places.json"
    report = cc.compare(
        loaded(make("fixed")),
        loaded(make("ragged", answers=RAGGED)),
        repeat=loaded(make("fixed", 1, answers=TIMING_ONLY)),
        places_out=out,
    )
    assert report["frozen"] is False
    assert "NOT FROZEN: of 3 recordings, 0 in text and 1 in word timings only" in report["verdict"]
    verdict = json.loads(out.read_text())["runs"]["bfloat16"]["repeat_verdict"]
    assert verdict == (
        "NOT run-to-run identical: two fixed captures differ on 0 in text and 1 in word timings "
        "only, at concurrency 32 and 1 (observed peak in flight 3 and 1)"
    )


def test_a_repeat_at_the_same_concurrency_is_not_the_same_shape_verdict(tmp_path: Path) -> None:
    """step1_places scores only on its same-shape string, which needs two concurrencies."""
    out = tmp_path / "places.json"
    cc.compare(
        loaded(make("fixed", 32)),
        loaded(make("ragged", answers=RAGGED)),
        repeat=loaded(make("fixed", 32)),
        places_out=out,
    )
    verdict = json.loads(out.read_text())["runs"]["bfloat16"]["repeat_verdict"]
    assert verdict == (
        "identical at one concurrency (3) only: both captures peaked at 3 in flight "
        "(configured 32 and 32); not shown across occupancy"
    )


def test_a_ragged_capture_identical_to_the_fixed_one_is_an_absent_control(tmp_path: Path) -> None:
    out = tmp_path / "places.json"
    report = cc.compare(
        loaded(make("fixed")), loaded(make("ragged")), repeat=loaded(make("fixed", 1)),
        places_out=out,
    )  # fmt: skip
    assert report["verdict"].startswith("POSITIVE CONTROL ABSENT")
    assert "NOT SHOWN FROZEN" in report["verdict"] and report["frozen"] is False
    record = json.loads(out.read_text())
    assert record["runs"]["bfloat16"]["positive_control"].startswith("ABSENT")
    assert record["runs"]["bfloat16"]["arms"][cc.ARM]["divergences"] == []


def test_places_are_refused_when_padding_is_only_declared(tmp_path: Path) -> None:
    fixed = loaded(make("fixed", observed=False))
    ragged = loaded(make("ragged", answers=RAGGED))
    with pytest.raises(cc.Refused, match="padding is only declared"):
        cc.compare(fixed, ragged, places_out=tmp_path / "places.json")
    assert not (tmp_path / "places.json").exists()
    # The repeat's padding counts too.
    observed = loaded(make("fixed"))
    unobserved_repeat = loaded(make("fixed", 1, observed=False))
    with pytest.raises(cc.Refused, match="padding is only declared"):
        cc.compare(observed, ragged, repeat=unobserved_repeat, places_out=tmp_path / "p.json")
    report = cc.compare(
        fixed,
        ragged,
        repeat=loaded(make("fixed", 1)),
        places_out=tmp_path / "places.json",
        allow_declared_padding=True,
    )
    assert report["places_written"] and report["frozen"] is False
    # Accepted for the places, which say how each padding is known, but never the same-shape
    # verdict the scorer scores on.
    captures = json.loads((tmp_path / "places.json").read_text())["captures"]
    assert captures["a_fixed"]["padding_source"] == "declared only"
    assert captures["b_ragged"]["padding_source"] == "observed in the server's command line"
    verdict = json.loads((tmp_path / "places.json").read_text())["runs"]["bfloat16"]
    assert verdict["repeat_verdict"].startswith(
        "identical at two concurrencies, but the padding is only declared in "
    )
    assert verdict["repeat_verdict"].endswith(
        ", at concurrency 32 and 1 (observed peak in flight 3 and 1): not the same-shape verdict"
    )
    assert verdict["repeat_verdict"] != cc.SAME_SHAPE_VERDICT


def test_frozen_is_not_claimed_on_padding_that_was_only_declared() -> None:
    """Two fixed captures at two concurrencies that match and a ragged one that differs, with
    every other condition met: FROZEN only when the padding of all three was observed."""
    for unobserved in ("a", "b", "control"):
        captures = {
            "a": make(concurrency=32, observed=unobserved != "a"),
            "b": make(concurrency=1, observed=unobserved != "b"),
            "control": make("ragged", 32, RAGGED, observed=unobserved != "control"),
        }
        report = cc.compare(
            loaded(captures["a"], "a.json"),
            loaded(captures["b"], "b.json"),
            control=loaded(captures["control"], "control.json"),
        )
        assert report["frozen"] is False, unobserved
        assert report["verdict"].startswith("FROZEN NOT CLAIMED"), report["verdict"]
        assert f"only declared in {unobserved}.json" in report["verdict"]
    # Observed through the command line alone, or the banner alone, is observed.
    for path in ("server.padding.observed_in_cmdline", "server.padding.observed_in_server_log"):
        report = cc.compare(
            loaded(_set(make(concurrency=32), path, None)),
            loaded(make(concurrency=1)),
            control=loaded(make("ragged", 32, RAGGED)),
        )
        assert report["frozen"] is True, path


def test_a_confidence_element_is_not_part_of_the_timing_channel() -> None:
    """A capture whose words carry a fourth element, the confidence, compares identical on
    both channels to one without it; the difference is counted apart."""
    with_confidence = {
        rid: (text, [(*w, 0.5) for w in words]) for rid, (text, words) in FIXED.items()
    }
    report = cc.compare(
        loaded(make(concurrency=32)), loaded(make(concurrency=1, answers=with_confidence))
    )
    assert report["identity"]["identical"] == 3 and report["identity"]["digests_match"]
    assert report["identity"]["confidence_only_differs"] == 2  # r2 has no words


# --- refusals ---


SETTING_PATHS = {
    "model": ("server.model.reported_by_server", "some/other-model"),
    "revision": ("server.model_revision.observed_in_hf_cache", "f" * 40),
    "chunk_ms": ("server.chunk_ms.reported_by_server", 1120),
    "dtype": ("server.dtype.reported_by_server", "float32"),
    "execution": ("server.execution.reported_by_server", "graph path"),
    "bucket": ("server.bucket.reported_by_server", BUCKET * 2),
    "pipeline": ("server.readyz_before.pipeline", "cache_aware_ctc"),
    "biasing": ("server.readyz_before.biasing", True),
    "word_confidence": ("server.word_confidence.reported_by_server", "paper-best"),
    "decoder_step_confidence": ("server.word_confidence.decoder_step_confidence", True),
    "decoder_graphs": ("server.decoder_graphs.reported_by_server", True),
    "nemo_version": ("server.readyz_before.nemo_version", "other"),
    "torch_version": ("server.readyz_before.torch_version", "other"),
    "device_name": ("server.readyz_before.device_name", "other"),
    "att_context": ("server.att_context.observed_in_readyz", [70, 13]),
}


def _setting(record: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """``path`` set to ``value``; for one of the observed settings, every stamp of the setting too
    (``EXPECTED_OBSERVED``), so the capture is consistent and differs only from another capture."""
    section, key, _ = path.split(".", 2)
    if section == "server" and key in EXPECTED_OBSERVED:
        source, others = EXPECTED_OBSERVED[key]
        for stamp in (source, *others):
            record["server"][key][stamp] = copy.deepcopy(value)
        return record
    return _set(record, path, value)


@pytest.mark.parametrize("setting", sorted(SETTING_PATHS))
def test_captures_that_differ_in_a_setting_are_refused(setting: str) -> None:
    path, value = SETTING_PATHS[setting]
    other = _setting(make(concurrency=8), path, value)
    assert cc.capture_problems(loaded(other)) == []  # consistent in itself
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make()), loaded(other))
    assert [r for r in refused.value.reasons if r.startswith(f"the captures differ in {setting}:")]


def test_captures_that_differ_in_matmul_are_refused() -> None:
    other = make(concurrency=8)
    other["server"]["matmul_precision"] = {"declared": "highest", "installed_default": "highest"}
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make()), loaded(other))
    assert [r for r in refused.value.reasons if r.startswith("the captures differ in matmul:")]


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        (
            "server.model_revision.observed_in_hf_cache",
            "model_revision was not observed (server.model_revision.observed_in_hf_cache is null)",
        ),
        (
            "server.att_context.observed_in_readyz",
            "att_context was not observed (server.att_context.observed_in_readyz is null)",
        ),
        ("server.gpu.before.uuid", "a card is not recorded in both captures"),
    ],
    ids=["revision", "att-context", "card"],
)
def test_a_setting_one_capture_did_not_record_is_refused(path: str, reason: str) -> None:
    """None against None would compare equal: a setting nobody observed is refused instead."""
    other = _set(make(concurrency=8), path, None)
    with pytest.raises(cc.Refused, match=re.escape(reason)):
        cc.compare(loaded(make()), loaded(other))
    both = _set(make(), path, None)
    with pytest.raises(cc.Refused, match=re.escape(reason)):
        cc.compare(loaded(both), loaded(_set(make(concurrency=8), path, None)))


def test_the_attention_context_compared_is_the_observed_one_never_the_derived_one() -> None:
    """A client derives the same context from the same model and chunk, so the derived value
    cannot tell two captures of one setting apart; only the observed one can. The derived value
    is a claim about the observed one: unrecorded, the observation alone is compared; recorded
    and different, the capture contradicts itself and is refused."""
    no_derived = _set(make(concurrency=8), "server.att_context.derived_from_cmdline", None)
    report = cc.compare(loaded(make()), loaded(no_derived))
    assert report["settings"]["att_context"] == [70, 1]
    derived_differs = _set(make(concurrency=8), "server.att_context.derived_from_cmdline", [70, 13])
    with pytest.raises(cc.Refused, match="att_context derived_from_cmdline"):
        cc.compare(loaded(make()), loaded(derived_differs))
    observed_differs = _setting(
        make(concurrency=8), "server.att_context.observed_in_readyz", [70, 13]
    )
    with pytest.raises(cc.Refused, match=r"the captures differ in att_context: \[70, 1\] and"):
        cc.compare(loaded(make()), loaded(observed_differs))


def test_captures_on_different_cards_are_refused() -> None:
    other = _set(make(concurrency=8), "server.gpu.before.uuid", "GPU-another")
    with pytest.raises(cc.Refused, match="different cards"):
        cc.compare(loaded(make()), loaded(other))


def test_the_same_capture_twice_is_refused() -> None:
    one = make()
    with pytest.raises(cc.Refused, match="are the same capture: the same bytes"):
        cc.compare(loaded(one, "a.json"), loaded(copy.deepcopy(one), "b.json"))
    # Re-serialised (so the bytes differ) is still the same run.
    edited = copy.deepcopy(one)
    edited["client"]["concurrency"]["configured"] = 8
    with pytest.raises(cc.Refused, match="are the same capture: the same start and finish"):
        cc.compare(loaded(one, "a.json"), loaded(edited, "b.json"))
    # Its start and finish edited too: the server process and the tick it started at remain.
    edited.update(started="edited-start", finished="edited-finish")
    with pytest.raises(cc.Refused, match="the same server process at the same starting tick"):
        cc.compare(loaded(one, "a.json"), loaded(edited, "b.json"))
    # Another capture of the same process starts at another tick, and is not the same capture.
    assert cc.same_capture(loaded(make()), loaded(make(concurrency=8))) is None


@pytest.mark.parametrize(
    "order",
    ["fixed-ragged-repeat", "ragged-fixed-repeat", "repeat-ragged-fixed", "fixed-repeat-control"],
)
def test_the_same_capture_is_refused_in_every_pair_and_order(order: str) -> None:
    """The fixed capture again, re-serialised at another stated concurrency, would make a
    FROZEN verdict (and a same-shape repeat verdict) out of one run. Every pair of inputs is
    checked, so no order of arguments lets it through."""
    one = make()
    again = copy.deepcopy(one)
    again["client"]["concurrency"]["configured"] = 8
    fixed, repeat = loaded(one, "fixed.json"), loaded(again, "again.json")
    ragged = loaded(make("ragged", answers=RAGGED), "ragged.json")
    calls = {
        "fixed-ragged-repeat": lambda: cc.compare(fixed, ragged, repeat=repeat),
        "ragged-fixed-repeat": lambda: cc.compare(ragged, fixed, repeat=repeat),
        "repeat-ragged-fixed": lambda: cc.compare(repeat, ragged, repeat=fixed),
        "fixed-repeat-control": lambda: cc.compare(fixed, repeat, control=ragged),
    }
    with pytest.raises(cc.Refused, match="are the same capture: the same start and finish"):
        calls[order]()


def _without_r2() -> dict[str, Any]:
    record = make(concurrency=8)
    del record["recordings"]["r2"]
    record["client"]["concurrency"]["observed_peak_in_flight"] = 2  # what the two left give
    record["finals_digest"] = cc.digest_of(record)
    return record


def _changed(key: str, value: Any) -> dict[str, Any]:
    record = make(concurrency=8)
    record["recordings"]["r1"][key] = value
    return record


def _tampered_digest() -> dict[str, Any]:
    record = make(concurrency=8)
    record["recordings"]["r1"]["text"] = "edited after the capture"
    return record


def _failed() -> dict[str, Any]:
    record = make(concurrency=8)
    record.update(success=False, status="FAILED", failures=["1 of 3 recording(s) failed"])
    return record


def _padding_contradiction() -> dict[str, Any]:
    return _set(make(concurrency=8), "server.padding.observed_in_server_log", "ragged")


def _cmdline_padding_contradiction() -> dict[str, Any]:
    return _set(make(concurrency=8), "server.padding.observed_in_cmdline", "ragged")


def _matmul_contradiction() -> dict[str, Any]:
    return _set(make(concurrency=8), "server.matmul_precision.declared", "highest")


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (_without_r2, "the captures do not hold the same recordings"),
        (lambda: _changed("pcm_sha256", "other-audio"), "1 recording(s) were sent different audio"),
        (lambda: _changed("reference", "other"), "1 recording(s) carry different references"),
        (_tampered_digest, "capture.json: the stored finals_digest is not the digest"),
        (_failed, "capture.json: not a complete capture"),
        (
            lambda: {**make(concurrency=8), "record": "vb-server-frozen-answers/1"},
            "capture.json: record kind",
        ),
        (_padding_contradiction, "capture.json: padding declared 'fixed' but the server log says"),
        (
            _cmdline_padding_contradiction,
            "capture.json: padding declared 'fixed' but the server's command line says",
        ),
        (_matmul_contradiction, "capture.json: matmul declared 'highest'"),
    ],
    ids=[
        "recording-set",
        "audio",
        "reference",
        "digest",
        "failed",
        "kind",
        "padding-log",
        "padding-cmdline",
        "matmul",
    ],
)
def test_captures_that_cannot_be_compared_are_refused(build: Any, reason: str) -> None:
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(build()))
    assert [r for r in refused.value.reasons if r.startswith(reason)], refused.value.reasons


def test_a_contradicted_padding_never_reaches_a_verdict() -> None:
    """``capture_problems`` refuses a capture whose padding an observation contradicts; if one
    ever got past it, ``padding_of`` stops rather than name a padding."""
    for build in (_padding_contradiction, _cmdline_padding_contradiction):
        with pytest.raises(RuntimeError, match="a contradicted capture reached a verdict"):
            cc.padding_of(build())
    assert cc.padding_of(make()) == ("fixed", "observed in the server's command line")


def test_the_third_capture_must_be_the_missing_padding(tmp_path: Path) -> None:
    fixed2 = loaded(make(concurrency=8))
    ragged = loaded(make("ragged", answers=RAGGED))
    with pytest.raises(cc.Refused, match="the control is fixed, not ragged"):
        cc.compare(loaded(make()), fixed2, control=loaded(make()))
    with pytest.raises(cc.Refused, match="--control goes with two fixed captures"):
        cc.compare(loaded(make()), ragged, control=loaded(make("ragged")))
    with pytest.raises(cc.Refused, match="the repeat is ragged, not fixed"):
        cc.compare(loaded(make()), ragged, repeat=loaded(make("ragged")))
    with pytest.raises(cc.Refused, match="--repeat goes with one fixed and one ragged"):
        cc.compare(loaded(make()), fixed2, repeat=loaded(make(concurrency=4)))
    with pytest.raises(cc.Refused, match="--control and --repeat do not go together"):
        cc.compare(loaded(make()), fixed2, control=ragged, repeat=loaded(make(concurrency=4)))
    with pytest.raises(cc.Refused, match="control: the captures differ in bucket"):
        other = _setting(make("ragged", answers=RAGGED), "server.bucket.reported_by_server", 16)
        cc.compare(loaded(make()), fixed2, control=loaded(other))
    with pytest.raises(cc.Refused, match="--places-out needs one fixed and one ragged"):
        cc.compare(loaded(make()), fixed2, places_out=tmp_path / "p.json")


# --- observed settings: never a declared value in place of an observed one ---


#: Where each ``cc.OBSERVED`` setting lands in the places record.
PLACES_KEY = {
    "model": "model",
    "model_revision": "model_revision",
    "chunk_ms": "chunk_ms",
    "bucket": "bucket",
    "execution": "execution",
    "att_context": "att_context_size",
}
#: What ``cc.OBSERVED`` must be, written out here rather than read from it: a stamp dropped from
#: the comparator's table turns ``test_the_observed_table_is_the_one_written_here`` red, and its
#: contradiction case below still runs, and goes red too.
EXPECTED_OBSERVED: dict[str, tuple[str, tuple[str, ...]]] = {
    "model": ("reported_by_server", ("declared",)),
    "model_revision": ("observed_in_hf_cache", ("declared",)),
    "chunk_ms": ("reported_by_server", ("declared",)),
    "dtype": ("reported_by_server", ("declared",)),
    "bucket": ("reported_by_server", ("observed_in_cmdline", "declared")),
    "execution": ("reported_by_server", ("observed_in_cmdline", "declared")),
    "att_context": (
        "observed_in_readyz",
        ("derived_from_cmdline", "observed_in_server_log", "declared"),
    ),
}
#: A value no fixture carries, per setting.
OTHER_VALUE = {
    "model": "some/other-model",
    "model_revision": "f" * 40,
    "chunk_ms": 1120,
    "dtype": "float32",
    "bucket": BUCKET * 2,
    "execution": "graph path",
    "att_context": [70, 13],
}
STAMPS = [(key, other) for key, (_, others) in EXPECTED_OBSERVED.items() for other in others]


def _places(tmp_path: Path, fixed: dict[str, Any], **kw: Any) -> dict[str, Any]:
    out = tmp_path / "places.json"
    cc.compare(
        loaded(fixed, "fixed.json"),
        loaded(make("ragged", answers=RAGGED), "ragged.json"),
        repeat=loaded(make("fixed", 1), "fixed-c1.json"),
        places_out=out,
        **kw,
    )
    return json.loads(out.read_text())


def test_the_observed_table_is_the_one_written_here() -> None:
    """The comparator's table, stamp for stamp: which stamp is the observation and which others
    are held to it. Weakening it (a stamp no longer held, an observation read from a declaration)
    is a change this test names."""
    assert cc.OBSERVED == EXPECTED_OBSERVED
    assert set(OTHER_VALUE) == set(EXPECTED_OBSERVED)
    assert set(PLACES_KEY) | {"dtype"} == set(EXPECTED_OBSERVED)  # dtype is the runs key
    for key, (source, others) in EXPECTED_OBSERVED.items():
        assert source != "declared" and "declared" in others, key


@pytest.mark.parametrize(
    ("key", "stamp", "value", "observed", "source"),
    [
        ("bucket", "observed_in_cmdline", 16, BUCKET, "reported_by_server"),
        ("execution", "observed_in_cmdline", "graph path", "eager", "reported_by_server"),
        ("att_context", "derived_from_cmdline", [70, 13], [70, 1], "observed_in_readyz"),
        ("att_context", "observed_in_server_log", [70, 13], [70, 1], "observed_in_readyz"),
    ],
    ids=["bucket-cmdline", "execution-cmdline", "att-context-derived", "att-context-banner"],
)
def test_a_command_line_or_banner_that_contradicts_the_server_is_refused(
    key: str, stamp: str, value: Any, observed: Any, source: str
) -> None:
    """Written case by case, not taken from any table: a server that reports one bucket,
    execution or attention context while its own command line or banner says another is not
    one setting, and the capture is refused, by ``compare`` and by ``build_places``."""
    bad = _set(make(concurrency=8), f"server.{key}.{stamp}", value)
    reason = f"b.json: {key} {stamp} {value!r} contradicts the observed {source} {observed!r}"
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(bad, "b.json"))
    assert refused.value.reasons == [reason]
    ragged = loaded(make("ragged", answers=RAGGED), "ragged.json")
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(loaded(bad, "b.json"), ragged, repeat=None, fixed_vs_repeat=None)
    assert refused.value.reasons == [reason]


@pytest.mark.parametrize(("key", "other"), STAMPS, ids=[f"{k}-{o}" for k, o in STAMPS])
def test_a_stamp_that_contradicts_the_observed_setting_is_refused(
    tmp_path: Path, key: str, other: str
) -> None:
    """The declaration (or the command line, or the banner) says one thing and the observation
    another: the capture is refused by ``compare`` and by ``build_places`` alike, so neither
    value is ever chosen between."""
    source = EXPECTED_OBSERVED[key][0]
    bad = _set(make(concurrency=8), f"server.{key}.{other}", OTHER_VALUE[key])
    observed = bad["server"][key][source]
    reason = (
        f"b.json: {key} {other} {OTHER_VALUE[key]!r} contradicts the observed {source} {observed!r}"
    )
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(bad, "b.json"))
    assert reason in refused.value.reasons, refused.value.reasons
    ragged = loaded(make("ragged", answers=RAGGED), "ragged.json")
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(loaded(bad, "b.json"), ragged, repeat=None, fixed_vs_repeat=None)
    assert refused.value.reasons == [reason]


@pytest.mark.parametrize("key", sorted(EXPECTED_OBSERVED))
def test_a_setting_that_was_not_observed_is_refused_though_it_was_declared(
    tmp_path: Path, key: str
) -> None:
    source = EXPECTED_OBSERVED[key][0]
    unobserved = _set(make(concurrency=8), f"server.{key}.{source}", None)
    assert unobserved["server"][key]["declared"] is not None
    reason = f"b.json: {key} was not observed (server.{key}.{source} is null)"
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(unobserved, "b.json"))
    assert reason in refused.value.reasons, refused.value.reasons
    ragged = loaded(make("ragged", answers=RAGGED), "ragged.json")
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(loaded(unobserved, "b.json"), ragged, repeat=None, fixed_vs_repeat=None)
    assert refused.value.reasons == [reason]


@pytest.mark.parametrize("key", sorted(EXPECTED_OBSERVED))
def test_the_places_carry_the_observed_value_whatever_the_other_stamps_leave_out(
    tmp_path: Path, key: str
) -> None:
    """Every other stamp of the setting unrecorded (as a capture whose command line and log
    could not be read, written by a probe that declares nothing): the places still carry the
    observed value. A writer that read the declaration, the command line or the banner instead
    would write null here."""
    fixed = make("fixed", 32)
    source, others = EXPECTED_OBSERVED[key]
    for other in others:
        fixed["server"][key][other] = None
    places = _places(tmp_path, fixed)
    want = fixed["server"][key][source]
    if key == "dtype":
        assert list(places["runs"]) == [want]
    else:
        assert places[PLACES_KEY[key]] == want is not None
    if key == "bucket":
        assert places["batch"] == want


# --- a capture a test double took ---


@pytest.mark.parametrize("fake", [True, 1, "true", None], ids=["true", "one", "string", "null"])
@pytest.mark.parametrize("where", ["a", "b", "control", "repeat"])
def test_a_capture_stamped_fake_is_refused_wherever_it_is_given(fake: Any, where: str) -> None:
    """``fake_pipeline`` present and anything but false, as step1_places reads it."""
    stamped = {"fake_pipeline": fake}
    if where in ("a", "b", "control"):
        captures = {
            "a": make(concurrency=32),
            "b": make(concurrency=8),
            "control": make("ragged", 32, RAGGED),
        }
        captures[where] = {**stamped, **captures[where]}
        call = lambda: cc.compare(  # noqa: E731
            loaded(captures["a"], "a.json"),
            loaded(captures["b"], "b.json"),
            control=loaded(captures["control"], "control.json"),
        )
    else:
        call = lambda: cc.compare(  # noqa: E731
            loaded(make(concurrency=32), "a.json"),
            loaded(make("ragged", answers=RAGGED), "b.json"),
            repeat=loaded({**stamped, **make(concurrency=8)}, "repeat.json"),
        )
    with pytest.raises(cc.Refused) as refused:
        call()
    assert refused.value.reasons == [
        f"{where}.json: fake_pipeline is {fake!r}: test doubles took this capture, so it "
        "measures nothing"
    ]
    # False is the one value that is not a stamp.
    assert cc.capture_problems(loaded(make(fake=False))) == []


def test_places_built_from_a_fake_capture_are_stamped_and_the_scorer_refuses_them() -> None:
    """``compare`` never gets this far with a fake capture; ``build_places`` called directly
    (as step1_places' own contract test does) passes the stamp on, first, and the scorer
    refuses the record."""
    for where in ("fixed", "ragged", "repeat"):
        captures = {
            "fixed": make("fixed", 32),
            "ragged": make("ragged", answers=RAGGED),
            "repeat": make("fixed", 8),
        }
        captures[where] = {"fake_pipeline": True, **captures[where]}
        fixed, ragged, repeat = (loaded(captures[k], f"{k}.json") for k in captures)
        record = cc.build_places(
            fixed, ragged, repeat=repeat, fixed_vs_repeat=cc.identity(fixed.record, repeat.record)
        )
        assert next(iter(record)) == "fake_pipeline" and record["fake_pipeline"] is True
        assert record["not_a_row"].startswith(f"FAKE: derived from captures test doubles took "
                                              f"({where}.json)")  # fmt: skip
        with pytest.raises(ValueError, match="fake_pipeline is True"):
            _step1_places().derive(json.loads(json.dumps(record)), frozenset(), draws=0)
    clean = cc.build_places(
        loaded(make("fixed", 32)), loaded(make("ragged", answers=RAGGED)), repeat=None,
        fixed_vs_repeat=None,
    )  # fmt: skip
    assert "fake_pipeline" not in clean


# --- a capture taken from a modified tree ---


#: One capture's client stamps, and what the comparator says of its tree.
UNCLEAN = {
    "modified": ({"tracked_files_modified": True}, "modified"),
    "null": ({"tracked_files_modified": None}, "not recorded"),
    "not-a-bool": ({"tracked_files_modified": "yes"}, "not recorded"),
    "outside": (
        {"tracked_files_modified": None, "code_outside_checkout": ["verbatim"]},
        "code imported from outside its checkout",
    ),
    # A clean flag does not make code imported from elsewhere the checkout's.
    "outside-flag-clean": (
        {"tracked_files_modified": False, "code_outside_checkout": ["verbatim_bench.client"]},
        "code imported from outside its checkout",
    ),
}


def _with_client(record: dict[str, Any], stamps: dict[str, Any]) -> dict[str, Any]:
    record["client"].update(copy.deepcopy(stamps))
    return record


@pytest.mark.parametrize("case", sorted(UNCLEAN))
@pytest.mark.parametrize(
    ("path", "where"),
    [
        ("control", "a"),
        ("control", "b"),
        ("control", "control"),
        ("repeat", "fixed"),
        ("repeat", "ragged"),
        ("repeat", "repeat"),
    ],
    ids=["control-a", "control-b", "control-control", "repeat-fixed", "repeat-ragged",
         "repeat-repeat"],
)  # fmt: skip
def test_frozen_is_withheld_when_a_capture_is_not_from_a_clean_tree(
    case: str, path: str, where: str
) -> None:
    """Every other condition of FROZEN met, on either path to it (two fixed captures and a
    ``--control``, or a fixed and a ragged capture and a ``--repeat``); one capture's client tree
    was modified, its state was not recorded, or its client imported code from outside its
    checkout. FROZEN only with --allow-modified-tree, and the verdict says so."""
    stamps, state = UNCLEAN[case]
    if path == "control":
        names = ("a", "b", "control")
        records = [make(concurrency=32), make(concurrency=1), make("ragged", 32, RAGGED)]
    else:
        names = ("fixed", "ragged", "repeat")
        records = [make(concurrency=32), make("ragged", 32, RAGGED), make(concurrency=1)]
    given = [
        loaded(_with_client(r, stamps) if n == where else r, f"{n}.json")
        for n, r in zip(names, records, strict=True)
    ]

    def run(**kw: Any) -> dict[str, Any]:
        return cc.compare(given[0], given[1], **{path: given[2]}, **kw)

    reading = (
        "FROZEN NOT CLAIMED: the fixed captures at concurrency 32 and 1 (observed peak in flight "
        "3 and 1) match on all 3 recordings and the ragged capture differs, but not every "
        f"capture was taken from a clean tree: {where}.json ({state}); a capture from such a tree "
        "may not be the commit it names (--allow-modified-tree to accept)"
    )
    report = run()
    assert report["frozen"] is False
    assert report["verdict"] == (
        reading
        if path == "control"
        else "POSITIVE CONTROL PRESENT: ragged padding differs from fixed on 1 in text and 1 in "
        f"word timings only, of 3 recordings; those are the places. {reading}"
    )
    assert report["trees_not_clean"] == [f"{where}.json ({state})"]
    allowed = run(allow_modified_tree=True)
    assert allowed["frozen"] is True and allowed["allow_modified_tree"] is True
    assert (
        "FROZEN (--allow-modified-tree: accepted although not taken from a clean tree: "
        f"{where}.json ({state})): the fixed captures at concurrency 32 and 1 (observed peak in "
        "flight 3 and 1)"
    ) in allowed["verdict"]


def test_a_missing_tree_state_is_not_a_clean_tree(tmp_path: Path) -> None:
    missing = make(concurrency=1)
    del missing["client"]["tracked_files_modified"]
    report = cc.compare(
        loaded(make(concurrency=32)), loaded(missing, "b.json"),
        control=loaded(make("ragged", 32, RAGGED)),
    )  # fmt: skip
    assert report["frozen"] is False and "b.json (not recorded)" in report["verdict"]
    clean = cc.compare(
        loaded(make(concurrency=32)), loaded(make(concurrency=1)),
        control=loaded(make("ragged", 32, RAGGED)),
    )  # fmt: skip
    assert clean["frozen"] is True and clean["verdict"].startswith("FROZEN: ")
    assert clean["trees_not_clean"] == []


@pytest.mark.parametrize(
    "outside", ["absent", None, "verbatim", {}], ids=["absent", "null", "a-string", "a-mapping"]
)
def test_a_code_origin_that_was_not_recorded_is_not_a_clean_tree(outside: Any) -> None:
    """Three captures that meet every other condition of FROZEN, each with a clean
    ``tracked_files_modified`` and a ``code_outside_checkout`` that is not a list (absent, as a
    capture from a probe that never observed where its code came from): none of them recorded
    that its client's code was its checkout's, so none is a clean tree."""
    records = {
        "a": make(concurrency=32),
        "b": make(concurrency=1),
        "control": make("ragged", 32, RAGGED),
    }
    for record in records.values():
        if outside == "absent":
            del record["client"]["code_outside_checkout"]
        else:
            record["client"]["code_outside_checkout"] = outside
        assert cc.tree_state(record) == "not recorded"
    report = cc.compare(
        loaded(records["a"], "a.json"),
        loaded(records["b"], "b.json"),
        control=loaded(records["control"], "control.json"),
    )
    assert report["frozen"] is False
    assert report["trees_not_clean"] == [
        "a.json (not recorded)",
        "b.json (not recorded)",
        "control.json (not recorded)",
    ]
    assert report["verdict"].startswith("FROZEN NOT CLAIMED")
    # A recorded empty list, with a clean flag, is the one clean tree.
    assert cc.tree_state(make()) is None


@pytest.mark.parametrize(
    ("cases", "carried"),
    [
        ((None, None, None), False),
        ((None, "modified", None), True),
        ((None, None, "null"), None),
        (("null", "modified", None), True),
        ((None, "outside-flag-clean", None), None),
    ],
    ids=["clean", "one-modified", "one-unrecorded", "modified-beats-unrecorded", "one-outside"],
)
def test_the_places_carry_each_captures_tree_state(
    tmp_path: Path, cases: tuple[str | None, str | None, str | None], carried: Any
) -> None:
    records = [make("fixed", 32), make("ragged", answers=RAGGED), make("fixed", 1)]
    for record, case in zip(records, cases, strict=True):
        if case is not None:
            _with_client(record, UNCLEAN[case][0])
    out = tmp_path / "places.json"
    report = cc.compare(
        loaded(records[0], "fixed.json"),
        loaded(records[1], "ragged.json"),
        repeat=loaded(records[2], "repeat.json"),
        places_out=out,
    )
    places = json.loads(out.read_text())
    assert places["tracked_files_modified"] is carried
    captures = places["captures"]
    each = [captures[k]["tracked_files_modified"] for k in ("a_fixed", "b_ragged", "repeat_fixed")]
    assert each == [r["client"]["tracked_files_modified"] for r in records]
    outside = [
        captures[k]["code_outside_checkout"] for k in ("a_fixed", "b_ragged", "repeat_fixed")
    ]
    assert outside == [r["client"]["code_outside_checkout"] for r in records]
    assert report["frozen"] is (cases == (None, None, None))


# --- the peak in flight is recounted from the recordings, never taken on trust ---


@pytest.mark.parametrize(
    ("intervals", "peak"),
    [
        ([(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)], 2),
        ([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], 1),
        ([(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)], 3),
        ([(2.0, 3.0), (0.0, 9.0), (1.0, 2.5)], 3),
        ([(0.0, 1.0), None, (0.5, 0.75)], 2),
        ([], 0),
    ],
    ids=["overlap", "one-after-another", "all-at-once", "out-of-order", "untimed", "none"],
)
def test_the_peak_is_recounted_by_the_probes_rule(intervals: list[Any], peak: int) -> None:
    """The cases the probe's own rule is tested on (tests/test_server_frozen_answers.py holds
    the two rules equal on them): a session whose last final arrived at the instant another's
    first audio frame went out is not counted with it; sessions that start at one instant are."""
    assert cc.recounted_peak(intervals) == peak


SEQUENTIAL = [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]
_MISMATCH = "b.json: observed_peak_in_flight {} is not the {} its recordings' in_flight_s give"


def _stamped(
    stamped: Any = 3, spans_: list[Any] | None = None, configured: Any = 32
) -> dict[str, Any]:
    """A capture of three recordings whose client stamped ``stamped`` as its peak, configured
    at ``configured``, with ``spans_`` as its recordings' in_flight_s (all at once when None;
    ``"absent"`` as an entry leaves that recording without the key)."""
    record = make(concurrency=32)
    record["client"]["concurrency"].update(configured=configured, observed_peak_in_flight=stamped)
    for entry, span in zip(record["recordings"].values(), spans_ or [], strict=False):
        if span == "absent":
            del entry["in_flight_s"]
        else:
            entry["in_flight_s"] = span
    return record


@pytest.mark.parametrize(
    ("record", "reasons"),
    [
        (_stamped(3, SEQUENTIAL), [_MISMATCH.format(3, 1)]),
        (
            _stamped(99),
            [
                _MISMATCH.format(99, 3),
                "b.json: observed_peak_in_flight 99 is above the 3 recording(s) it holds",
                "b.json: observed_peak_in_flight 99 is above the configured concurrency 32",
            ],
        ),
        (
            _stamped(4),
            [
                _MISMATCH.format(4, 3),
                "b.json: observed_peak_in_flight 4 is above the 3 recording(s) it holds",
            ],
        ),
        (
            _stamped(3, configured=2),
            ["b.json: observed_peak_in_flight 3 is above the configured concurrency 2"],
        ),
        (
            _stamped(3, configured=None),
            ["b.json: the configured concurrency None is not a whole number"],
        ),
        (_stamped(3, ["absent", "absent", "absent"]), [_MISMATCH.format(3, 0)]),
        (_stamped(1, [None, None, None]), [_MISMATCH.format(1, 0)]),
        (
            _stamped(3, [[0.0, 1.0], ["a", 1.0], [True, 2.0]]),
            ["b.json: in_flight_s of 2 recording(s) is neither null nor two numbers; first r1"],
        ),
        (
            _stamped(3, [[0.0], [0.0, 1.0], [0.0, 1.0]]),
            ["b.json: in_flight_s of 1 recording(s) is neither null nor two numbers; first r0"],
        ),
        (
            _stamped(3, [[0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 1.0]]),
            ["b.json: in_flight_s of 1 recording(s) is neither null nor two numbers; first r1"],
        ),
    ],
    ids=[
        "not-what-the-recordings-give",
        "above-everything",
        "above-the-recordings",
        "above-the-configured",
        "configured-unrecorded",
        "no-in-flight-stamps",
        "untimed-sessions",
        "malformed",
        "one-number",
        "three-numbers",
    ],
)
def test_a_stamped_peak_its_recordings_do_not_give_is_refused(
    record: dict[str, Any], reasons: list[str]
) -> None:
    """A peak in flight is an occupancy only when the capture's own recordings give it: it is
    recounted from their in_flight_s, and it cannot exceed the recordings held or the
    concurrency configured. Refused by ``compare`` (so FROZEN never reads it) and by
    ``build_places`` (so the same-shape verdict never does)."""
    record = copy.deepcopy(record)
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(record, "b.json"))
    assert refused.value.reasons == reasons
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(
            loaded(record, "b.json"),
            loaded(make("ragged", answers=RAGGED), "ragged.json"),
            repeat=None,
            fixed_vs_repeat=None,
        )
    assert refused.value.reasons == reasons


def test_a_peak_stamped_above_what_the_wire_carried_no_longer_makes_frozen() -> None:
    """Before the recount, a fixed capture of three recordings stamped 99 (configured 32), or
    one stamped 8 whose sessions ran one after another, gave FROZEN against a capture that
    peaked at 1. Each is refused now; the same capture stamped with what its recordings give is
    one occupancy with the other, and FROZEN is withheld."""
    for record in (_stamped(99), _stamped(8, SEQUENTIAL)):
        with pytest.raises(cc.Refused):
            cc.compare(
                loaded(record, "b.json"),
                loaded(make(concurrency=1), "c1.json"),
                control=loaded(make("ragged", 32, RAGGED)),
            )
    honest = _stamped(1, SEQUENTIAL)
    report = cc.compare(
        loaded(honest, "b.json"),
        loaded(make(concurrency=1), "c1.json"),
        control=loaded(make("ragged", 32, RAGGED)),
    )
    assert report["frozen"] is False
    assert report["verdict"].startswith("IDENTICAL AT ONE CONCURRENCY (1)")


def test_a_peak_that_is_not_recorded_is_not_recounted() -> None:
    """Nothing reads an unrecorded peak as an occupancy (it withholds FROZEN and the same-shape
    verdict), so its recordings' stamps are not held to it."""
    record = _stamped(None, [["a"], ["b"], ["c"]])
    assert cc.capture_problems(loaded(record)) == []


# --- the client code that took each capture ---


#: What ``cc.CLIENT_CODE`` must be, written out here.
EXPECTED_CLIENT_CODE = ("verbatim_commit", "probe_sha256", "bench_client_sha256")


def test_the_client_code_compared_is_the_one_written_here() -> None:
    assert cc.CLIENT_CODE == EXPECTED_CLIENT_CODE


@pytest.mark.parametrize("key", EXPECTED_CLIENT_CODE)
def test_captures_taken_by_different_client_code_are_refused(key: str) -> None:
    """A ragged control, or a second fixed capture, taken by other code could differ, or match,
    because the code changed and not the padding or the occupancy: refused in every pair, by
    ``compare`` and by ``build_places``. A stamp one capture does not record is refused too."""
    other_value = "d" * 40
    other = _set(make(concurrency=1), f"client.{key}", other_value)
    differ = (
        f"the captures were taken by different client code: client.{key} {CODE[key]!r} and "
        f"{other_value!r}"
    )
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(other, "b.json"))
    assert refused.value.reasons == [differ]
    ragged = _set(make("ragged", 32, RAGGED), f"client.{key}", other_value)
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make()), loaded(make(concurrency=1)), control=loaded(ragged))
    assert refused.value.reasons == [f"a/control: {differ}", f"b/control: {differ}"]
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(
            loaded(make(), "fixed.json"),
            loaded(make("ragged", answers=RAGGED), "ragged.json"),
            repeat=loaded(other, "repeat.json"),
            fixed_vs_repeat=None,
        )
    assert refused.value.reasons == [
        f"fixed.json and repeat.json: {differ}",
        f"ragged.json and repeat.json: {differ}",
    ]
    unrecorded = _set(make(concurrency=1), f"client.{key}", None)
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make()), loaded(unrecorded))
    assert refused.value.reasons == [
        f"client.{key} is not recorded in both captures ({CODE[key]!r}, None)"
    ]


def test_three_captures_from_three_commits_are_not_frozen() -> None:
    """Fixed at concurrency 32 from one commit, fixed at 1 from another, ragged from a third,
    every tree clean: before, FROZEN. Refused now, for every pair."""
    commits = {"a": "c" * 40, "b": "d" * 40, "control": "e" * 40}
    records = {
        "a": make(concurrency=32),
        "b": make(concurrency=1),
        "control": make("ragged", 32, RAGGED),
    }
    for name, record in records.items():
        record["client"]["verbatim_commit"] = commits[name]
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(records["a"]), loaded(records["b"]), control=loaded(records["control"]))
    assert [r.split(": the captures were taken")[0] for r in refused.value.reasons] == [
        "the captures were taken by different client code: client.verbatim_commit "
        f"{commits['a']!r} and {commits['b']!r}",
        "a/control",
        "b/control",
    ]


# --- the server code that served each capture (contract C7) ---


def test_the_server_code_compared_is_the_one_written_here() -> None:
    assert cc.SERVER_CODE == ("verbatim_path", "bench_path")


def _code_absent() -> dict[str, Any]:
    record = make(concurrency=1)
    del record["server"]["code"]
    return record


_ELSEWHERE = "/elsewhere/src/verbatim"


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (
            lambda: _set(make(concurrency=1), "server.code.reported_by_server.verbatim_path",
                         _ELSEWHERE),
            "the servers ran different code: server.code.verbatim_path "
            f"{SERVER_CODE['verbatim_path']!r} and {_ELSEWHERE!r}",
        ),
        (
            lambda: _set(make(concurrency=1), "server.code.reported_by_server.bench_path", None),
            "the servers ran different code: server.code.bench_path "
            f"{SERVER_CODE['bench_path']!r} and None",
        ),
        (
            lambda: _set(make(concurrency=1), "server.code.reported_by_server.verbatim_path",
                         None),
            "server.code.verbatim_path is not recorded in both captures "
            f"({SERVER_CODE['verbatim_path']!r}, None): which code the server ran was not "
            "observed",
        ),
        (
            _code_absent,
            "server.code.verbatim_path is not recorded in both captures "
            f"({SERVER_CODE['verbatim_path']!r}, None): which code the server ran was not "
            "observed",
        ),
    ],
    ids=["verbatim-path", "bench-path", "verbatim-path-null", "no-code-object"],
)  # fmt: skip
def test_captures_served_by_different_server_code_are_refused(build: Any, reason: str) -> None:
    """A server started without its checkout first on the path answers from whatever
    ``verbatim`` the environment imports: a ragged control, or a second fixed capture, served
    by other code could differ, or match, because the code changed. Refused in every pair, by
    ``compare`` and by ``build_places``; a ``verbatim_path`` one capture does not record too."""
    other = build()
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(other, "b.json"))
    assert refused.value.reasons == [reason]
    ragged = copy.deepcopy(other)
    ragged["server"]["padding"] = {
        "declared": "ragged", "observed_in_cmdline": "ragged", "observed_in_server_log": "ragged",
    }  # fmt: skip
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make()), loaded(make(concurrency=1)), control=loaded(ragged))
    assert refused.value.reasons == [f"a/control: {reason}", f"b/control: {reason}"]
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(
            loaded(make(), "fixed.json"),
            loaded(make("ragged", answers=RAGGED), "ragged.json"),
            repeat=loaded(other, "repeat.json"),
            fixed_vs_repeat=None,
        )
    assert refused.value.reasons == [
        f"fixed.json and repeat.json: {reason}",
        f"ragged.json and repeat.json: {reason}",
    ]


def test_a_server_that_could_not_import_verbatim_bench_matches_another_that_could_not() -> None:
    """``bench_path`` is null where the server could not import verbatim_bench; two such
    captures are one server code, as far as it goes."""
    a = _set(make(), "server.code.reported_by_server.bench_path", None)
    b = _set(make(concurrency=1), "server.code.reported_by_server.bench_path", None)
    assert cc.pair_problems(loaded(a), loaded(b)) == []


# --- the model revision is observed of the server only in the server's own cache ---


def test_the_revision_is_observed_only_in_the_cache_the_servers_environment_names() -> None:
    assert cc.OBSERVED_WHERE == {"model_revision": ("hub_dir_source", SERVER_HUB)}


@pytest.mark.parametrize(
    "source",
    [
        "this client's own huggingface_hub cache (the server's environment was not read)",
        "--hf-hub-cache (the server's environment was not read)",
        None,
    ],
    ids=["client-cache", "flag", "unrecorded"],
)
def test_a_revision_read_from_another_cache_is_not_observed(source: Any) -> None:
    """The probe falls back to --hf-hub-cache, or to its own cache, when the server's
    environment cannot be read, and says so in hub_dir_source. A revision read there is not an
    observation of the server: refused by ``compare`` and by ``build_places``."""
    other = _set(make(concurrency=1), "server.model_revision.hub_dir_source", source)
    reason = (
        "b.json: model_revision was not observed of the server: "
        f"server.model_revision.hub_dir_source is {source!r}, not {SERVER_HUB!r}"
    )
    with pytest.raises(cc.Refused) as refused:
        cc.compare(loaded(make(), "a.json"), loaded(other, "b.json"))
    assert refused.value.reasons == [reason]
    with pytest.raises(cc.Refused) as refused:
        cc.compare(
            loaded(make(), "a.json"),
            loaded(other, "b.json"),
            control=loaded(make("ragged", 32, RAGGED)),
        )
    assert refused.value.reasons == [reason]
    with pytest.raises(cc.Refused) as refused:
        cc.build_places(
            loaded(other, "b.json"),
            loaded(make("ragged", answers=RAGGED), "ragged.json"),
            repeat=None,
            fixed_vs_repeat=None,
        )
    assert refused.value.reasons == [reason]


# --- the command ---


def test_the_command_prints_a_verdict_and_writes_the_report_and_places(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = {}
    for name, record in (
        ("fixed", make("fixed")),
        ("ragged", make("ragged", answers=RAGGED)),
        ("fixed-c1", make("fixed", 1)),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(record), encoding="utf-8")
    report, places = tmp_path / "report.json", tmp_path / "places.json"
    argv = [str(paths["fixed"]), str(paths["ragged"]), "--repeat", str(paths["fixed-c1"]),
            "--out", str(report), "--places-out", str(places)]  # fmt: skip
    assert cc.main(argv) == cc.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "recordings 3: identical 1, text differs 1, timing only 1; digests differ"
    assert lines[-1].startswith("VERDICT: POSITIVE CONTROL PRESENT")
    body = json.loads(report.read_text())
    assert body["captures"]["a"]["sha256"] == cc.load_capture(paths["fixed"]).sha256
    arm = json.loads(places.read_text())["runs"]["bfloat16"]["arms"][cc.ARM]
    assert arm["checked"] == 3


def test_the_command_refuses_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(make()), encoding="utf-8")
    b.write_text(
        json.dumps(_setting(make(), "server.bucket.reported_by_server", 16)), encoding="utf-8"
    )
    assert cc.main([str(a), str(b)]) == cc.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "[refused] the captures differ in bucket: 8 and 16" in err


def test_the_fixtures_are_what_they_claim() -> None:
    """The builder above must itself produce a comparable, consistent pair, or every refusal
    test could be passing on a broken fixture rather than on the one field it changes."""
    assert cc.pair_problems(loaded(make()), loaded(make(concurrency=8))) == []
    for record in (make(), make(concurrency=1), make("ragged", answers=RAGGED), make(peak=2)):
        assert cc.capture_problems(loaded(record)) == []
    # Each capture's peak is what its recordings give, and what its three recordings can reach.
    assert [cc.peak_of(make(concurrency=c)) for c in (32, 8, 3, 2, 1)] == [3, 3, 3, 2, 1]
    assert cc.peak_of(make(concurrency=32, peak=1)) == 1


def test_the_command_claims_frozen_on_a_modified_tree_only_when_told_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = {}
    for name, record in (
        ("fixed-c32", make(concurrency=32, tree=True)),
        ("fixed-c1", make(concurrency=1)),
        ("ragged", make("ragged", 32, RAGGED)),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(record), encoding="utf-8")
    argv = [str(paths["fixed-c32"]), str(paths["fixed-c1"]), "--control", str(paths["ragged"])]
    assert cc.main(argv) == cc.EXIT_OK
    verdict = capsys.readouterr().out.splitlines()[-1]
    assert verdict.startswith("VERDICT: FROZEN NOT CLAIMED") and "(modified)" in verdict
    assert cc.main([*argv, "--allow-modified-tree"]) == cc.EXIT_OK
    verdict = capsys.readouterr().out.splitlines()[-1]
    assert verdict.startswith(
        f"VERDICT: FROZEN (--allow-modified-tree: accepted although not taken from a clean tree: "
        f"{paths['fixed-c32']} (modified))"
    )
    # The other path to FROZEN: a fixed and a ragged capture and --repeat.
    argv = [str(paths["fixed-c32"]), str(paths["ragged"]), "--repeat", str(paths["fixed-c1"])]
    assert cc.main(argv) == cc.EXIT_OK
    verdict = capsys.readouterr().out.splitlines()[-1]
    assert (
        verdict.startswith("VERDICT: POSITIVE CONTROL PRESENT") and "FROZEN NOT CLAIMED" in verdict
    )
    assert cc.main([*argv, "--allow-modified-tree"]) == cc.EXIT_OK
    verdict = capsys.readouterr().out.splitlines()[-1]
    assert (
        f"FROZEN (--allow-modified-tree: accepted although not taken from a clean tree: "
        f"{paths['fixed-c32']} (modified))"
    ) in verdict


# --- confidence equivalence (contract C8) ---


ON = "nemo-shipped"
#: A 1 ms later start of r0's first word: the text and every other timing are the same.
START_1MS = {**FIXED, "r0": (FIXED["r0"][0], [("I'm", 1, 160), *FIXED["r0"][1][1:]])}
#: r1's first word changed, in the text and in the word list.
CHANGED_WORD = {**FIXED, "r1": RAGGED["r1"]}
#: r1's first word changed in the word list alone: the text channel cannot see it.
CHANGED_WORD_ONLY = {**FIXED, "r1": (FIXED["r1"][0], RAGGED["r1"][1])}


def _on(answers: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    """A capture of the same setting taken with word confidence on."""
    return make(concurrency=32, answers=answers, confidence=ON, **kw)


def _write(tmp_path: Path, **records: dict[str, Any]) -> dict[str, Path]:
    paths = {}
    for name, record in records.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(record), encoding="utf-8")
    return paths


def test_a_confidence_on_capture_with_the_same_answers_is_identical_in_text_and_timings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The critic's case: ``compare`` refuses the pair, which differs in word confidence, and
    ``--confidence-equivalence`` is the command that says they are the same answers."""
    off, on = loaded(make(), "off.json"), loaded(_on(), "on.json")
    assert on.record["recordings"]["r0"]["words"][0] == ["I'm", 0, 160, CONFIDENCE]
    assert off.record["finals_digest"] == on.record["finals_digest"]  # confidence is not in it
    differ = [r for r in cc.pair_problems(off, on) if r.startswith("the captures differ in")]
    assert differ == [
        f"the captures differ in word_confidence: 'off' and {ON!r}",
        "the captures differ in decoder_step_confidence: False and True",
    ]
    assert cc.confidence_equivalence(off, on) == []

    paths = _write(tmp_path, off=make(), on=_on())
    report = tmp_path / "report.json"
    argv = ["--confidence-equivalence", str(paths["off"]), str(paths["on"]), "--out", str(report)]
    assert cc.main(argv) == cc.EXIT_OK
    assert capsys.readouterr().out.splitlines() == ["VERDICT: IDENTICAL IN TEXT AND TIMINGS"]
    body = json.loads(report.read_text())
    assert body["identical"] is True and body["differences"] == []
    assert body["off"]["path"] == str(paths["off"]) and body["on"]["path"] == str(paths["on"])
    with pytest.raises(cc.Refused, match="the captures differ in word_confidence"):
        cc.compare(cc.load_capture(paths["off"]), cc.load_capture(paths["on"]))


@pytest.mark.parametrize(
    ("answers", "detail"),
    [
        (START_1MS, "0 recording(s) differ in text and 1 in word timings only; first r0"),
        (CHANGED_WORD, "1 recording(s) differ in text and 0 in word timings only; first r1"),
        (CHANGED_WORD_ONLY, "0 recording(s) differ in text and 1 in word timings only; first r1"),
    ],
    ids=["start-1ms-later", "changed-word", "changed-word-in-the-timings"],
)
def test_a_confidence_on_capture_whose_answers_differ_is_not_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answers: dict[str, Any], detail: str
) -> None:
    """Each capture is consistent in itself (its stored digest is its own): only the equality
    of the two digests can see the difference, and it does."""
    off, on = loaded(make(), "off.json"), loaded(_on(answers), "on.json")
    assert cc.capture_problems(on) == []
    digests = (off.record["finals_digest"][:16], on.record["finals_digest"][:16])
    reason = f"the finals digests differ ({digests[0]} and {digests[1]}): {detail}"
    assert cc.confidence_equivalence(off, on) == [reason]

    paths = _write(tmp_path, off=off.record, on=on.record)
    argv = ["--confidence-equivalence", str(paths["off"]), str(paths["on"])]
    assert cc.main(argv) == cc.EXIT_DIFFERENT == 1
    assert capsys.readouterr().out.splitlines() == [f"VERDICT: NOT SHOWN IDENTICAL: {reason}"]


@pytest.mark.parametrize("side", ["off", "on"])
def test_a_capture_whose_stored_digest_is_not_its_own_is_not_identical(side: str) -> None:
    """(i): one capture's text edited after it was taken, its stored digest left as it was.
    The two stored digests are still equal, so (ii) alone would call the pair identical."""
    records = {"off": make(), "on": _on()}
    records[side]["recordings"]["r1"]["text"] = "edited after the capture"
    off, on = loaded(records["off"], "off.json"), loaded(records["on"], "on.json")
    assert off.record["finals_digest"] == on.record["finals_digest"]
    assert cc.confidence_equivalence(off, on) == [
        f"{side}.json: the stored finals_digest is not the digest of its own recordings"
    ]


def _swapped_order() -> dict[str, Any]:
    """r0 and r1 in each other's place in the corpus order; the digest, which sorts by id,
    is unchanged."""
    record = _on()
    r0, r1 = record["recordings"]["r0"], record["recordings"]["r1"]
    r0["n"], r1["n"] = 1, 0
    return record


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (_swapped_order, "the captures do not hold the same recordings in the same order"),
        (
            lambda: _set(_on(), "recordings.r1.pcm_sha256", "other-audio"),
            "1 recording(s) were sent different audio; first r1",
        ),
        (
            lambda: _set(_on(), "recordings.r1.reference", "other"),
            "1 recording(s) carry different references; first r1",
        ),
        (
            lambda: _set(_on(), "server.code.reported_by_server.verbatim_path", _ELSEWHERE),
            "the servers ran different code: server.code.verbatim_path "
            f"{SERVER_CODE['verbatim_path']!r} and {_ELSEWHERE!r}",
        ),
        (
            lambda: _set(_on(), "client.probe_sha256", "d" * 64),
            "the captures were taken by different client code: client.probe_sha256 "
            f"{CODE['probe_sha256']!r} and {'d' * 64!r}",
        ),
        (
            lambda: _set(_on(), "server.gpu.before.uuid", "GPU-another"),
            "the captures ran on different cards: GPU-test-double and GPU-another",
        ),
    ],
    ids=["order", "audio", "reference", "server-code", "client-code", "card"],
)
def test_a_confidence_on_capture_of_other_recordings_or_code_is_not_identical(
    build: Any, reason: str
) -> None:
    """(iii): the digests are equal, and still the pair is not the same answers to the same
    question."""
    off, on = loaded(make(), "off.json"), loaded(build(), "on.json")
    assert off.record["finals_digest"] == on.record["finals_digest"]
    assert cc.capture_problems(on) == []
    assert cc.confidence_equivalence(off, on) == [reason]


#: Every setting ``compare`` holds two captures to but the two C8 lets differ.
NOT_CONFIDENCE = sorted(set(SETTING_PATHS) - {"word_confidence", "decoder_step_confidence"})


@pytest.mark.parametrize("setting", NOT_CONFIDENCE)
def test_every_other_setting_must_match_for_confidence_equivalence(setting: str) -> None:
    path, value = SETTING_PATHS[setting]
    on = _setting(_on(), path, value)
    assert cc.capture_problems(loaded(on)) == []  # consistent in itself
    reasons = cc.confidence_equivalence(loaded(make(), "off.json"), loaded(on, "on.json"))
    assert len(reasons) == 1 and reasons[0].startswith(f"the captures differ in {setting}:"), (
        reasons
    )


def test_the_off_capture_must_be_off_and_the_on_capture_on() -> None:
    """Two off captures, or two on captures, are not the question C8 answers; neither is the
    pair given the wrong way round."""
    off_reason = (
        f"off.json: word confidence {ON!r}, not 'off': the first capture is the one taken with "
        "word confidence off"
    )
    on_reason = (
        "on.json: word confidence 'off': the second capture is the one taken with word "
        "confidence on"
    )
    both_off = cc.confidence_equivalence(loaded(make(), "off.json"), loaded(make(), "on.json"))
    assert both_off == [on_reason]
    both_on = cc.confidence_equivalence(loaded(_on(), "off.json"), loaded(_on(), "on.json"))
    assert both_on == [off_reason]
    swapped = cc.confidence_equivalence(loaded(_on(), "off.json"), loaded(make(), "on.json"))
    assert swapped == [off_reason, on_reason]
    unrecorded = _on()
    for stamp in unrecorded["server"]["word_confidence"]:
        unrecorded["server"]["word_confidence"][stamp] = None
    found = cc.confidence_equivalence(loaded(make(), "off.json"), loaded(unrecorded, "on.json"))
    assert found[0] == (
        "on.json: word confidence None: the second capture is the one taken with word confidence on"
    )


@pytest.mark.parametrize("side", ["off", "on"])
def test_a_capture_that_cannot_be_used_is_not_identical(side: str) -> None:
    """A test double's capture, or an incomplete one, is not an answer at all."""
    for record, reason in (
        (
            {"fake_pipeline": True},
            f"{side}.json: fake_pipeline is True: test doubles took this capture, so it "
            "measures nothing",
        ),
        (
            {"success": False, "failures": ["x"]},
            f"{side}.json: not a complete capture (success False)",
        ),
    ):
        records = {"off": make(), "on": _on()}
        records[side] = {**records[side], **record}
        found = cc.confidence_equivalence(
            loaded(records["off"], "off.json"), loaded(records["on"], "on.json")
        )
        assert found == [reason]


def test_confidence_equivalence_takes_no_other_capture_or_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write(tmp_path, off=make(), on=_on(), other=make(concurrency=8))
    c8 = ["--confidence-equivalence", str(paths["off"]), str(paths["on"])]
    for extra, named in (
        ([str(paths["other"])], "A"),
        (["--repeat", str(paths["other"])], "--repeat"),
        (["--places-out", str(tmp_path / "p.json")], "--places-out"),
        (["--allow-modified-tree"], "--allow-modified-tree"),
    ):
        assert cc.main([*c8, *extra]) == cc.EXIT_REFUSED
        err = capsys.readouterr().err
        assert f"[refused] --confidence-equivalence takes no {named}" in err, err
    assert cc.main([str(paths["off"])]) == cc.EXIT_REFUSED
    assert "[refused] two captures are needed" in capsys.readouterr().err
    assert not (tmp_path / "p.json").exists()


# --- outputs are written once, never over an input ---


def _inputs(tmp_path: Path) -> dict[str, Path]:
    return _write(
        tmp_path,
        fixed=make("fixed"),
        ragged=make("ragged", answers=RAGGED),
        repeat=make("fixed", 1),
    )


@pytest.mark.parametrize(
    "case",
    ["places-out-is-a", "out-is-the-repeat-by-dotdot", "out-is-a-link-to-b", "out-exists",
     "out-and-places-out-one-file"],
)  # fmt: skip
def test_an_output_that_exists_or_is_an_input_is_refused_before_anything_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """A capture is hours of server time: ``--out`` or ``--places-out`` naming one of the
    inputs, however spelled, would have written the report over it. Refused, with every
    reason, before any input is read; nothing is written anywhere."""
    paths = _inputs(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}
    earlier = tmp_path / "earlier-report.json"
    earlier.write_text("an earlier report\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    link = tmp_path / "link.json"
    link.symlink_to(paths["ragged"])
    fresh = tmp_path / "fresh.json"
    base = [str(paths["fixed"]), str(paths["ragged"]), "--repeat", str(paths["repeat"])]
    dotdot = tmp_path / "sub" / ".." / "repeat.json"
    extra, reasons = {
        "places-out-is-a": (
            ["--places-out", str(paths["fixed"])],
            [
                f"--places-out {paths['fixed']} is the input {paths['fixed']}: an input is never "
                "written over",
                f"--places-out {paths['fixed']} exists: an output is written once, pick a new path",
            ],
        ),
        "out-is-the-repeat-by-dotdot": (
            ["--out", str(dotdot)],
            [
                f"--out {dotdot} is the input {paths['repeat']}: an input is never written over",
                f"--out {dotdot} exists: an output is written once, pick a new path",
            ],
        ),
        "out-is-a-link-to-b": (
            ["--out", str(link)],
            [
                f"--out {link} is the input {paths['ragged']}: an input is never written over",
                f"--out {link} exists: an output is written once, pick a new path",
            ],
        ),
        "out-exists": (
            ["--out", str(earlier)],
            [f"--out {earlier} exists: an output is written once, pick a new path"],
        ),
        "out-and-places-out-one-file": (
            ["--out", str(fresh), "--places-out", str(fresh)],
            [
                f"--out {fresh} and --places-out {fresh} are one file: each output needs its "
                "own path"
            ],
        ),
    }[case]
    assert cc.main([*base, *extra]) == cc.EXIT_REFUSED
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing was compared
    assert captured.err.splitlines() == [f"[refused] {r}" for r in reasons]
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert earlier.read_text(encoding="utf-8") == "an earlier report\n"
    assert not fresh.exists() and not list(tmp_path.glob("*.tmp"))


def test_confidence_equivalence_never_writes_over_a_capture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write(tmp_path, off=make(), on=_on())
    before = paths["on"].read_bytes()
    argv = ["--confidence-equivalence", str(paths["off"]), str(paths["on"]), "--out"]
    assert cc.main([*argv, str(paths["on"])]) == cc.EXIT_REFUSED
    assert capsys.readouterr().err.splitlines() == [
        f"[refused] --out {paths['on']} is the input {paths['on']}: an input is never written over",
        f"[refused] --out {paths['on']} exists: an output is written once, pick a new path",
    ]
    assert paths["on"].read_bytes() == before


@pytest.mark.parametrize("links", [True, False], ids=["hard-link", "no-hard-links"])
def test_an_output_that_appears_before_the_write_is_not_written_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, links: bool
) -> None:
    """``main`` refuses an existing path first; a file that appears there afterwards (another
    run given the same path) is left as it is, and no temporary file stays behind."""
    if not links:

        def no_links(*_: Any) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(cc.os, "link", no_links)
    out = tmp_path / "report.json"
    cc._write_json({"n": 1}, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"n": 1}
    with pytest.raises(cc.Refused) as refused:
        cc._write_json({"n": 2}, out)
    assert refused.value.reasons == [f"{out} exists: nothing was written over it"]
    assert json.loads(out.read_text(encoding="utf-8")) == {"n": 1}
    assert not list(tmp_path.glob("*.tmp"))
    # compare() called as a library, with places that appeared at the path: refused likewise.
    places = tmp_path / "places.json"
    places.write_text("earlier places\n", encoding="utf-8")
    with pytest.raises(cc.Refused, match="exists: nothing was written over it"):
        cc.compare(loaded(make("fixed")), loaded(make("ragged", answers=RAGGED)), places_out=places)
    assert places.read_text(encoding="utf-8") == "earlier places\n"
