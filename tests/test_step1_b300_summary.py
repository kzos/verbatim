# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``scripts/step1_b300_summary.py``: the committed summary is what the committed records give.

The records are committed (``rows/exploratory/step1-b300-2026-09-26/`` and
``step1-a6000-c8-2026-09-26/``), so nothing here is skipped.

* The summary reproduces, byte for byte, from the committed gz files.
* A finals record with one word changed is refused at each layer in turn: on the manifest's
  sha256 of the file; with that made to match, on the sha256 of its decompressed bytes; with
  that made to match, on the arm's record summary, which names the finals by sha256; and with
  that link rewritten too, on the digests, because ``assess`` over the changed finals is not the
  invariance record's verdict. A commit the gate does not stamp, a capture entry that stamps
  none, a places record that does not name its captures, a record holding a machine path, a
  refused run made to look complete, and counts its recordings do not give are refused too.
* The retest at concurrency 64: its second ragged run with one word changed is refused on the
  manifest's sha256 of the file, then of its decompressed bytes, then by the comparator (the
  stored digest is not its recordings'); forged consistently, digest and all, it gives a
  comparison that is not the committed one. The fixed run made to look complete, the fixed
  run's or the first attempt's counts changed (among them the claim that 2,927 recordings got
  neither a final nor a partial), a time-out on a recording that got its final, and a failed
  record filed under another's name are refused.
* Mutations, each a copy of the script with a source edit: dropping the timing-only count (in
  fixed against ragged, and in the gate's split by kind) is refused by the recount, and with
  that check removed gives a section that is not the committed one; swapping the eager and graph
  captures is refused on the places record's sha256 links, with that check removed on what the
  captures record, and with that removed too by the comparator itself; reading the gate
  summary's arms by position instead of by name is refused on the verdict, and with every check
  it trips removed still gives a gate section that is not the committed one; reading the verdict
  the wrong way round gives a summary that is not the committed one. At concurrency 64:
  dropping the ragged pair's timing-only count is refused by the invariance gate's recount, and
  with that check removed gives a retest section that is not the committed one; comparing a run
  with itself is refused by the comparator; counting the first attempt's recordings without a
  partial among the sent ones only is refused by the recount.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "step1_b300_summary.py"
B300 = ROOT / "rows" / "exploratory" / "step1-b300-2026-09-26"
C8 = ROOT / "rows" / "exploratory" / "step1-a6000-c8-2026-09-26"
A6000 = ROOT / "rows" / "exploratory" / "step1-a6000-2026-09-26"
SMOKE = ROOT / "rows" / "exploratory" / "step1-a6000-gate-smoke-2026-09-26"
SUMMARY = ROOT / "rows" / "exploratory" / "step1-b300-summary-2026-09-26.json"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
s2 = importlib.import_module("step1_b300_summary")

FINALS = s2.gate("finals", "fixed-churn")
CHECKED = s2.gate("record-summary", "fixed-churn")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """A temporary directory removed when its test ends; pytest's own is kept for three runs."""
    with tempfile.TemporaryDirectory(prefix="step1-b300-summary-test-") as name:
        yield Path(name)


@pytest.fixture(scope="module")
def built() -> Any:
    return s2.load_inputs(B300, C8)


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


# --- the committed summary is the records' ------------------------------------------------


def test_the_summary_reproduces_byte_for_byte_from_the_committed_records(built) -> None:
    assert s2.render(s2.assemble(*built)).encode("utf-8") == SUMMARY.read_bytes()


def test_the_summary_names_its_inputs_by_the_manifests(committed) -> None:
    for key, directory in (
        ("b300", B300),
        ("a6000_c8", C8),
        ("a6000", A6000),
        ("a6000_gate_smoke", SMOKE),
    ):
        manifest = (directory / s2.MANIFEST).read_bytes()
        named = committed["inputs"][key]
        assert named["manifest_sha256"] == sha(manifest)
        entries = json.loads(manifest)["records"]
        assert named["sha256_raw"] == {n: e["sha256_raw"] for n, e in entries.items()}


def test_the_headline_figures_are_the_ones_the_records_give(committed) -> None:
    """The figures the paper quotes, read here so a change to one is seen by name."""
    b300 = committed["b300"]
    assert b300["card"]["cards"] == 1
    assert b300["frozen"]["eager"]["frozen"] is True
    assert b300["frozen"]["graphs"]["frozen"] is False
    fixed = [b300["captures"][n]["finals_digest"] for n in s2.FIXED.values()]
    assert len(set(fixed)) == 1 and fixed[0].startswith("5a672de4fe0cb924")
    rvf, rvr = b300["ragged_vs_fixed"], b300["ragged_vs_ragged"]
    assert (rvf["text_differs"], rvf["word_level"], rvf["timing_only"]) == (266, 128, 815)
    assert (rvr["text_differs"], rvr["word_level"], rvr["timing_only"]) == (1, 0, 2)
    c8 = committed["a6000_c8"]["ragged_vs_ragged"]
    assert (c8["text_differs"], c8["word_level"], c8["timing_only"]) == (204, 104, 691)
    arms = b300["gate"]["arms"]
    assert [arms[a]["verdict"] for a in s2.ARMS] == ["invariant"] * 2 + ["divergent"] * 2
    assert [
        arms["ragged-churn"]["differing"][f"{s}_vs_1"]["streams"] for s in ("32a", "32b", "max")
    ] == [78, 78, 65]
    assert [
        arms["ragged-const"]["differing"][f"{s}_vs_1"]["streams"] for s in ("32a", "32b", "max")
    ] == [78, 78, 36]
    refused = b300["overload"]["run2_refused"]
    assert (refused["admitted_during_run"], refused["refused_during_run"]) == (388, 2551)
    assert b300["overload"]["repeatability_measured"] is False
    retest = b300["retest_c64"]
    c64 = retest["ragged_vs_ragged"]
    assert (c64["text_differs"], c64["word_level"], c64["timing_only"]) == (4, 0, 4)
    assert c64["captures"] == list(s2.C64) and c64["digests_match"] is False
    ticks = [b300["captures"][n] for n in s2.C64]
    assert [(t["ticks_over_budget_during_run"], t["ticks_during_run"]) for t in ticks] == [
        (285, 2016),
        (248, 2001),
    ]
    assert [t["ticks_late_during_run"] for t in ticks] == [97, 33]
    assert [t["refused_during_run"] for t in ticks] == [0, 0]
    fixed = retest["fixed_refused"]
    assert (fixed["admitted_during_run"], fixed["refused_during_run"]) == (350, 2589)
    assert fixed["ticks_over_budget_during_run"] == 136
    first = retest["first_attempt"]
    assert (first["admitted_during_run"], first["refused_during_run"]) == (128, 2811)
    assert first["counts"]["recordings_missing_final"] == 2927
    assert first["counts"]["recordings_without_partials"] == 2907
    assert (first["ticks_late_during_run"], first["ticks_during_run"]) == (335, 339)
    assert [e["server_process"] for e in retest["timeline"]] == [1, 2, 2, 3]
    assert [e["record"] for e in retest["timeline"]] == [
        s2.C64_FIRST_ATTEMPT,
        *s2.C64,
        s2.C64_FIXED_REFUSED,
    ]
    assert b300["card"]["cards"] == 1
    assert first["sessions_timed"] == 14 and first["observed_peak_in_flight"] == 14
    assert (b300["card"]["records_naming_a_card"], b300["card"]["records"]) == (15, 36)
    a6000 = committed["a6000"]
    assert a6000["card"] == {
        "cards": 1,
        "device_names": ["NVIDIA RTX A6000"],
        "records": 24,
        "records_naming_a_card": 18,
    }
    processes = [
        a6000["ragged_c32_pair"]["server_processes"],
        c8["server_processes"],
        rvr["server_processes"],
        c64["server_processes"],
    ]
    assert processes == [2, 1, 1, 1]
    stopped = committed["a6000_gate_smoke"]
    assert [(v["bucket"], round(v["max_p95_tick_ms"], 2)) for v in stopped.values()] == [
        (32, 123.96),
        (64, 156.75),
    ]
    assert all(v["stopped_before_the_gate"] and v["budget_ms"] == 112.0 for v in stopped.values())


# --- refused before anything is derived ---------------------------------------------------


def _copy(tmp_path: Path) -> tuple[Path, Path]:
    b300, c8 = tmp_path / B300.name, tmp_path / C8.name
    shutil.copytree(B300, b300)
    shutil.copytree(C8, c8)
    return b300, c8


def _edit_manifest(directory: Path, name: str, **fields: Any) -> None:
    path = directory / s2.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["records"][name].update(fields)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _write(directory: Path, name: str, record: Any) -> tuple[bytes, bytes]:
    """A record written in place of ``name``, gzipped; (gz, raw). The manifest is not touched."""
    raw = (json.dumps(record, indent=1) + "\n").encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    (directory / name).write_bytes(packed)
    return packed, raw


def _read(directory: Path, name: str) -> Any:
    return json.loads(gzip.decompress((directory / name).read_bytes()))


def test_a_finals_record_with_one_word_changed_is_refused_at_every_layer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b300, c8 = _copy(tmp_path)
    finals = _read(b300, FINALS)
    final = finals["levels"][1]["finals"][0]  # level 32a, the first stream
    words = final["text"].split()
    final["text"] = " ".join(["Tampered", *words[1:]])
    packed, raw = _write(b300, FINALS, finals)
    # 1. the file's sha256 is not the manifest's ...
    with pytest.raises(s2.Refused, match=f"{FINALS}: its sha256 is {sha(packed)}, the manifest"):
        s2.load_inputs(b300, c8)
    out = tmp_path / "summary.json"
    assert s2.main(["--b300", str(b300), "--a6000-c8", str(c8), "--out", str(out)]) == 2
    assert "the manifest says" in capsys.readouterr().err
    assert not out.exists()
    # 2. ... and with it made to match, the decompressed bytes' sha256 is not ...
    _edit_manifest(b300, FINALS, sha256_gz=sha(packed), bytes_gz=len(packed))
    with pytest.raises(s2.Refused, match=f"decompressed bytes is {sha(raw)}, the manifest"):
        s2.load_inputs(b300, c8)
    # 3. ... and with that made to match, the arm's record summary names other finals ...
    _edit_manifest(b300, FINALS, sha256_raw=sha(raw), bytes_raw=len(raw))
    with pytest.raises(s2.Refused, match=f"{CHECKED}: its finals_sha256 is '[0-9a-f]+', not"):
        s2.load_inputs(b300, c8)
    # 4. ... and with that link rewritten too, assess over the changed finals is not the verdict
    # the invariance record, the finals and both summaries give.
    checked = _read(b300, CHECKED)
    checked["finals_sha256"] = sha(raw)
    packed2, raw2 = _write(b300, CHECKED, checked)
    _edit_manifest(
        b300,
        CHECKED,
        sha256_gz=sha(packed2),
        bytes_gz=len(packed2),
        sha256_raw=sha(raw2),
        bytes_raw=len(raw2),
    )
    inputs = s2.load_inputs(b300, c8)
    with pytest.raises(
        s2.Refused,
        match="fixed-churn: assess gives divergent; the invariance record says invariant",
    ):
        s2.gate_section(inputs[0])


def test_an_unstamped_record_whose_commit_is_not_the_gates_is_refused(tmp_path: Path) -> None:
    b300, c8 = _copy(tmp_path)
    _edit_manifest(b300, s2.CARD, commit="26eaebb128cd00ea7b38c1245df4ac95b54cee6f")
    with pytest.raises(s2.Refused, match=f"{s2.CARD}: the manifest says '26eaebb1.*the gate"):
        s2.load_inputs(b300, c8)


def test_a_capture_that_stamps_no_commit_is_refused(tmp_path: Path) -> None:
    b300, c8 = _copy(tmp_path)
    _edit_manifest(b300, s2.RAGGED, commit_stamp=None)
    with pytest.raises(s2.Refused, match=f"{s2.RAGGED}: a 'capture' record must stamp its commit"):
        s2.load_inputs(b300, c8)


def test_a_commit_the_capture_does_not_stamp_is_refused(tmp_path: Path) -> None:
    b300, c8 = _copy(tmp_path)
    _edit_manifest(c8, s2.C8[0], commit="26eaebb128cd00ea7b38c1245df4ac95b54cee6f")
    with pytest.raises(s2.Refused, match=f"{s2.C8[0]}: client.verbatim_commit is 'ba20c498"):
        s2.load_inputs(b300, c8)


def test_a_places_record_that_does_not_name_its_captures_is_refused(tmp_path: Path) -> None:
    b300, c8 = _copy(tmp_path)
    entries = json.loads((b300 / s2.MANIFEST).read_text(encoding="utf-8"))["records"]
    other = entries[s2.RAGGED_REPEAT]["sha256_raw"]
    places = _read(b300, s2.PLACES)
    places["captures"]["b_ragged"]["sha256"] = other
    packed, raw = _write(b300, s2.PLACES, places)
    _edit_manifest(
        b300,
        s2.PLACES,
        sha256_gz=sha(packed),
        bytes_gz=len(packed),
        sha256_raw=sha(raw),
        bytes_raw=len(raw),
    )
    with pytest.raises(s2.Refused, match=f"{s2.PLACES}: its b_ragged capture has sha256 '{other}'"):
        s2.load_inputs(b300, c8)


def test_a_record_holding_a_machine_path_is_refused_though_the_manifest_agrees(
    tmp_path: Path,
) -> None:
    b300, c8 = _copy(tmp_path)
    model = _read(b300, s2.MODEL)
    assert model["hf_hub_cache"] == "<workdir>/hf/hub"
    model["hf_hub_cache"] = "/srv/someone/hf/hub"
    packed, raw = _write(b300, s2.MODEL, model)
    _edit_manifest(
        b300,
        s2.MODEL,
        sha256_gz=sha(packed),
        bytes_gz=len(packed),
        sha256_raw=sha(raw),
        bytes_raw=len(raw),
    )
    with pytest.raises(s2.Refused, match=f"{s2.MODEL}: holds a machine path: /hf_hub_cache"):
        s2.load_inputs(b300, c8)


def test_a_file_the_manifest_does_not_list_is_refused(tmp_path: Path) -> None:
    b300, c8 = _copy(tmp_path)
    shutil.copy(b300 / s2.CARD, b300 / "extra.json.gz")
    with pytest.raises(s2.Refused, match=r"not \['extra.json.gz'\] that are"):
        s2.load_inputs(b300, c8)


def _with(inputs: Any, name: str, record: Any) -> Any:
    return s2.s1.Inputs(
        inputs.directory, inputs.manifest, inputs.manifest_sha256, {**inputs.records, name: record}
    )


def test_a_refused_run_made_to_look_complete_is_refused(built) -> None:
    b300 = built.b300
    record = b300.records[s2.REFUSED_RUN]
    complete = {**record, "success": True, "failures": [], "status": "complete"}
    with pytest.raises(s2.Refused, match="compare_captures would take it as a capture"):
        s2.refused_run(_with(b300, s2.REFUSED_RUN, complete))


def test_refusal_counts_the_recordings_do_not_give_are_refused(built) -> None:
    b300 = built.b300
    record = b300.records[s2.REFUSED_RUN]
    load = {**record["server"]["load"], "refused_during_run": 2550}
    changed = {**record, "server": {**record["server"], "load": load}}
    with pytest.raises(s2.Refused, match=r"recounted .*'refused_by_the_server': 2551"):
        s2.refused_run(_with(b300, s2.REFUSED_RUN, changed))


def test_a_capture_whose_name_says_another_execution_is_refused(built) -> None:
    b300 = built.b300
    eager, graphs = s2.FIXED[("eager", 32)], s2.FIXED[("graphs", 32)]
    swapped = _with(b300, eager, b300.records[graphs])
    with pytest.raises(s2.Refused, match=r"execution and concurrency .*'eager'.*'graph path'"):
        s2.says(swapped, eager, s2.B300_CAPTURES[eager])


# --- the cards, the server processes and the A6000's smoke-stopped runbook ------------------

OTHER_CARD = "GPU-00000000-1111-2222-3333-444444444444"


def test_a_second_a6000_card_in_any_record_is_counted(built) -> None:
    """``cards`` counts identities, not device names: a c8 capture that names another A6000
    makes it 2, though every device name is still NVIDIA RTX A6000."""
    loaded = built
    assert s2.a6000_section(loaded.a6000, loaded.c8, loaded.gate_smoke)["card"]["cards"] == 1
    record = loaded.c8.records[s2.C8[1]]
    gpu = record["server"]["gpu"]
    other = {
        **record,
        "server": {
            **record["server"],
            "gpu": {**gpu, "after": {**gpu["after"], "uuid": OTHER_CARD}},
        },
    }
    section = s2.a6000_section(loaded.a6000, _with(loaded.c8, s2.C8[1], other), loaded.gate_smoke)
    assert section["card"]["cards"] == 2
    assert section["card"]["device_names"] == ["NVIDIA RTX A6000"]


def test_a_second_b300_card_in_a_gate_record_is_counted(built) -> None:
    b300 = built.b300
    assert s2.card_section(b300)["cards"] == 1
    finals = {**b300.records[FINALS], "gpu_uuid": OTHER_CARD}
    section = s2.card_section(_with(b300, FINALS, finals))
    assert (section["cards"], section["records_naming_a_card"]) == (2, 16)


def test_the_torch_spelling_of_a_card_is_the_same_card_and_other_uuids_are_not_cards() -> None:
    same = {"gpu_uuid": "GPU-b43f9262-f250-444a-bfbf-461dd3500f1e"}
    torch = {"gpu_uuid_torch": "b43f9262-f250-444a-bfbf-461dd3500f1e"}
    session = {"server_session_id": "00000000-1111-2222-3333-444444444444"}
    assert s2.card_uuids([same, torch, session]) == {"b43f9262-f250-444a-bfbf-461dd3500f1e"}
    assert s2.card_uuids({"driver_and_uuid": f"550.144.03, {OTHER_CARD}"}) == {
        OTHER_CARD.removeprefix("GPU-")
    }


def test_a_pair_on_one_process_counts_one_and_a_missing_pid_is_refused(built) -> None:
    a6000 = built.a6000
    assert s2.server_processes(a6000, s2.A6000_C32) == 2
    first, again = (a6000.records[n] for n in s2.A6000_C32)
    moved = {**again, "server": {**again["server"], "process": first["server"]["process"]}}
    assert s2.server_processes(_with(a6000, s2.A6000_C32[1], moved), s2.A6000_C32) == 1
    process = {**first["server"]["process"], "after": {"pid": None}}
    blank = {**first, "server": {**first["server"], "process": process}}
    with pytest.raises(s2.Refused, match="stamps no server process id"):
        s2.server_processes(_with(a6000, s2.A6000_C32[0], blank), s2.A6000_C32)


def test_first_attempt_sessions_timed_the_recordings_do_not_give_are_refused(built) -> None:
    b300 = built.b300
    record = b300.records[s2.C64_FIRST_ATTEMPT]
    concurrency = {**record["client"]["concurrency"], "sessions_timed": 128}
    changed = {**record, "client": {**record["client"], "concurrency": concurrency}}
    with pytest.raises(s2.Refused, match=r"14 timed; it states .* 128 timed"):
        s2.first_attempt(_with(b300, s2.C64_FIRST_ATTEMPT, changed))


@pytest.mark.parametrize(
    ("kind", "edit", "refusal"),
    [
        ("serve-spec", lambda r: {**r, "spec": {**r["spec"], "batch_size": 128}}, "buckets are"),
        (
            "serve-spec",
            lambda r: {
                **r,
                "derived_from": [*r["derived_from"][:7], "ragged", *r["derived_from"][8:]],
            },
            "padding ragged",
        ),
        ("smoke-check", lambda r: {**r, "budget_ms": 310.0}, "budget 310.0, serve spec 112.0"),
    ],
)
def test_a_smoke_attempt_whose_records_disagree_is_refused(built, kind, edit, refusal) -> None:
    smoke = built.gate_smoke
    name = s2.smoke(kind, 64)
    with pytest.raises(s2.Refused, match=refusal):
        s2.gate_smoke_section(_with(smoke, name, edit(smoke.records[name])))


def test_a_smoke_check_without_a_problem_is_not_a_stop(built) -> None:
    smoke = built.gate_smoke
    name = s2.smoke("smoke-check", 32)
    passed = {**smoke.records[name], "problems": []}
    section = s2.gate_smoke_section(_with(smoke, name, passed))
    assert section["b32"]["stopped_before_the_gate"] is False
    assert section["b64"]["stopped_before_the_gate"] is True


def test_a_smoke_file_the_manifest_does_not_list_is_refused(tmp_path: Path) -> None:
    smoke = tmp_path / SMOKE.name
    shutil.copytree(SMOKE, smoke)
    shutil.copy(smoke / s2.smoke("readyz", 64), smoke / "extra.json.gz")
    with pytest.raises(s2.Refused, match=r"not \['extra.json.gz'\] that are"):
        s2.load_inputs(B300, C8, A6000, smoke)


# --- the retest at concurrency 64 -----------------------------------------------------------


def _rewrite(directory: Path, name: str, record: Any) -> None:
    """A record written in place of ``name`` with its manifest entry made to match."""
    packed, raw = _write(directory, name, record)
    _edit_manifest(
        directory,
        name,
        sha256_gz=sha(packed),
        bytes_gz=len(packed),
        sha256_raw=sha(raw),
        bytes_raw=len(raw),
    )


def test_a_c64_run_with_one_word_changed_is_refused_at_every_layer(
    tmp_path: Path, committed
) -> None:
    b300, c8 = _copy(tmp_path)
    run2 = _read(b300, s2.C64[1])
    rid = "7902-96591-0000"  # identical in the two runs
    assert run2["recordings"][rid]["text"] == _read(b300, s2.C64[0])["recordings"][rid]["text"]
    words = run2["recordings"][rid]["text"].split()
    run2["recordings"][rid]["text"] = " ".join(["Tampered", *words[1:]])
    packed, raw = _write(b300, s2.C64[1], run2)
    # 1. the file's sha256 is not the manifest's ...
    with pytest.raises(s2.Refused, match=f"{s2.C64[1]}: its sha256 is {sha(packed)}, the"):
        s2.load_inputs(b300, c8)
    # 2. ... and with it made to match, the decompressed bytes' sha256 is not ...
    _edit_manifest(b300, s2.C64[1], sha256_gz=sha(packed), bytes_gz=len(packed))
    with pytest.raises(s2.Refused, match=f"decompressed bytes is {sha(raw)}, the manifest"):
        s2.load_inputs(b300, c8)
    # 3. ... and with that made to match, the comparator refuses a capture whose stored digest
    # is not the digest of its own recordings ...
    _edit_manifest(b300, s2.C64[1], sha256_raw=sha(raw), bytes_raw=len(raw))
    inputs = s2.load_inputs(b300, c8).b300
    with pytest.raises(s2.Refused, match=r"c64-run2\.json\.gz: the stored finals_digest is not"):
        s2.retest_c64(inputs)
    # 4. ... and forged consistently, digest and all, it is a comparison, but not the
    # committed one, and the committed summary names other bytes.
    run2["finals_digest"] = s2.cc.digest_of(run2)
    _rewrite(b300, s2.C64[1], run2)
    inputs = s2.load_inputs(b300, c8).b300
    got = s2.retest_c64(inputs)["ragged_vs_ragged"]
    want = committed["b300"]["retest_c64"]["ragged_vs_ragged"]
    assert (got["text_differs"], got["word_level"]) == (5, 1)
    assert (want["text_differs"], want["word_level"]) == (4, 0)
    assert s2.inputs_of(inputs)["sha256_raw"] != committed["inputs"]["b300"]["sha256_raw"]


def test_the_fixed_c64_run_made_to_look_complete_is_refused(built) -> None:
    b300 = built.b300
    record = b300.records[s2.C64_FIXED_REFUSED]
    complete = {**record, "success": True, "failures": [], "status": "complete"}
    with pytest.raises(s2.Refused, match="compare_captures would take it as a capture"):
        s2.refused_run(_with(b300, s2.C64_FIXED_REFUSED, complete), s2.C64_FIXED_REFUSED)


def test_fixed_c64_refusal_counts_the_recordings_do_not_give_are_refused(built) -> None:
    b300 = built.b300
    assert s2.refused_run(b300, s2.C64_FIXED_REFUSED)["refused_during_run"] == 2589
    record = b300.records[s2.C64_FIXED_REFUSED]
    load = {**record["server"]["load"], "refused_during_run": 2588}
    changed = {**record, "server": {**record["server"], "load": load}}
    with pytest.raises(s2.Refused, match=r"fixed-c64.FAILED.json.gz: recounted .*: 2589"):
        s2.refused_run(_with(b300, s2.C64_FIXED_REFUSED, changed), s2.C64_FIXED_REFUSED)


@pytest.mark.parametrize(
    ("where", "field", "value"),
    [
        # The brief's reading: 2,927 recordings with neither a final nor a partial. The record
        # says 2,927 without a final and 2,907 without a partial; the recount agrees with it.
        ("counts", "recordings_without_partials", 2927),
        ("counts", "recordings_with_terminal_final", 13),
        ("load", "admitted_during_run", 129),
        ("load", "refused_during_run", 2810),
    ],
)
def test_first_attempt_counts_the_recordings_do_not_give_are_refused(
    built, where: str, field: str, value: int
) -> None:
    b300 = built.b300
    record = b300.records[s2.C64_FIRST_ATTEMPT]
    assert s2.first_attempt(b300)["counts"]["recordings_without_partials"] == 2907
    if where == "counts":
        changed = {**record, "counts": {**record["counts"], field: value}}
    else:
        load = {**record["server"]["load"], field: value}
        changed = {**record, "server": {**record["server"], "load": load}}
    with pytest.raises(s2.Refused, match=r"first-attempt.FAILED.json.gz: recounted"):
        s2.first_attempt(_with(b300, s2.C64_FIRST_ATTEMPT, changed))


def test_a_first_attempt_recording_timed_out_despite_its_final_is_refused(built) -> None:
    """Only the check that every time-out is a sent recording without a final can see this:
    the stated counts and the server's load are left as they are."""
    b300 = built.b300
    record = b300.records[s2.C64_FIRST_ATTEMPT]
    rid, entry = next((k, e) for k, e in record["recordings"].items() if e["terminal_final"])
    timed_out = {**entry, "client_error": "TimeoutError: no final received after end"}
    changed = {**record, "recordings": {**record["recordings"], rid: timed_out}}
    with pytest.raises(s2.Refused, match=r"'client_timed_out': 115\}"):
        s2.first_attempt(_with(b300, s2.C64_FIRST_ATTEMPT, changed))


def test_the_first_attempt_made_to_look_complete_is_refused(built) -> None:
    b300 = built.b300
    record = b300.records[s2.C64_FIRST_ATTEMPT]
    complete = {**record, "success": True, "failures": [], "status": "complete"}
    with pytest.raises(s2.Refused, match="compare_captures would take it as a capture"):
        s2.first_attempt(_with(b300, s2.C64_FIRST_ATTEMPT, complete))


def test_a_failed_record_filed_under_another_name_is_refused(built) -> None:
    b300 = built.b300
    swapped = _with(b300, s2.C64_FIXED_REFUSED, b300.records[s2.C64_FIRST_ATTEMPT])
    with pytest.raises(s2.Refused, match=r"fixed-c64.FAILED.json.gz: its name says .*'fixed'"):
        s2.says(swapped, s2.C64_FIXED_REFUSED, s2.B300_FAILED[s2.C64_FIXED_REFUSED])


# --- mutations ------------------------------------------------------------------------------


def _mutant(*edits: tuple[str, str]) -> types.ModuleType:
    """A copy of the script with source edits applied; each edit must match once."""
    source = SCRIPT.read_text(encoding="utf-8")
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType("step1_b300_summary_mutant")
    module.__dict__["__file__"] = str(SCRIPT)
    sys.modules[module.__name__] = module
    exec(compile(source, "step1_b300_summary_mutant", "exec"), module.__dict__)
    return module


NO_TIMING = ('    same = apart["identity"]', '    same = {**apart["identity"], "timing_only": 0}')
NO_RECOUNT = ("    if counted != stored:", "    if False:")
NO_GATE_TIMING = (
    '        "timing_only": sum(d.kind == "words" for d in found),',
    '        "timing_only": 0,',
)
NO_GATE_SUM = (
    '    if out["text"] + out["timing_only"] + out["missing"] != out["streams"]:',
    "    if False:",
)
SWAP_EXECUTION = (
    'FIXED = {(e, c): capture(f"fixed-{e}-c{c}") for e in EXECUTIONS for c in (32, 8)}',
    "FIXED = {(e, c): capture(f\"fixed-{dict(eager='graphs', graphs='eager')[e]}-c{c}\")"
    " for e in EXECUTIONS for c in (32, 8)}",
)
NO_SAYS = ("    if got != want:", "    if False:")
NO_PLACES_LINK = ("        if named != want:", "        if False:")
BY_POSITION = (
    '    by_name = {entry["arm"]: entry for entry in summary["arms"]}',
    '    by_name = dict(zip(ARMS, summary["arms"]))',
)
NO_VERDICT_CHECK = ("        if verdict != report.verdict:", "        if False:")
NO_DISTINCT_CHECK = (
    '    if summary_arm["distinct_digests"] != len(set(digests.values())):',
    "    if False:",
)
NO_PADDING_CHECK = (
    "    if (\n        argv[argv.index",
    "    if False and (\n        argv[argv.index",
)
NO_READING_CHECK = (
    '        if reading != (against_one, pairs["32b_vs_32a"]["streams"]):',
    "        if False:",
)
WRONG_WAY_ROUND = (
    '        "verdict": report.verdict,',
    '        "verdict": "invariant" if report.equal is False else "divergent",',
)


def test_mutation_fixed_against_ragged_that_stops_counting_timing_only_goes_red(
    built, committed
) -> None:
    b300 = built.b300
    want = committed["b300"]["ragged_vs_fixed"]
    assert s2.ragged_vs_fixed(b300) == want
    blind = _mutant(NO_TIMING)
    with pytest.raises(blind.Refused, match="differ on 266 texts and 0 timings only; the places"):
        blind.ragged_vs_fixed(b300)
    blinder = _mutant(NO_TIMING, NO_RECOUNT)
    got = blinder.ragged_vs_fixed(b300)
    assert got["timing_only"] == 0 and want["timing_only"] == 815
    assert got != want


def test_mutation_a_gate_that_stops_counting_timing_only_goes_red(built, committed) -> None:
    b300 = built.b300
    want = committed["b300"]["gate"]
    assert s2.gate_section(b300) == want
    blind = _mutant(NO_GATE_TIMING)
    with pytest.raises(
        blind.Refused, match=r"32a against 1: the kinds .* do not sum to the streams"
    ):
        blind.gate_section(b300)
    blinder = _mutant(NO_GATE_TIMING, NO_GATE_SUM)
    got = blinder.gate_section(b300)
    assert got["arms"]["ragged-churn"]["differing"]["32a_vs_1"]["timing_only"] == 0
    assert want["arms"]["ragged-churn"]["differing"]["32a_vs_1"]["timing_only"] == 57
    assert got != want


def test_mutation_swapping_eager_and_graphs_goes_red() -> None:
    # The places record names the eager captures by sha256, so the swap is refused there ...
    swapped = _mutant(SWAP_EXECUTION)
    with pytest.raises(swapped.Refused, match=r"its a_fixed capture has sha256 '[0-9a-f]+', not "):
        swapped.load_inputs(B300, C8)
    # ... with that check removed, on what each capture records against what its name says ...
    unlinked = _mutant(SWAP_EXECUTION, NO_PLACES_LINK)
    with pytest.raises(unlinked.Refused, match=r"fixed-graphs-c32\.json\.gz: its name says"):
        unlinked.load_inputs(B300, C8)
    # ... and with that removed too, by the comparator itself: the eager ragged capture is no
    # control for graph-path captures, and eager against graphs is not what the labels say.
    unchecked = _mutant(SWAP_EXECUTION, NO_PLACES_LINK, NO_SAYS)
    b300 = unchecked.load_inputs(B300, C8).b300
    with pytest.raises(unchecked.Refused, match="differ in execution: 'graph path' and 'eager'"):
        unchecked.frozen_section(b300)
    with pytest.raises(unchecked.Refused, match="expected only"):
        unchecked.eager_vs_graphs(b300)


def test_mutation_reading_the_gate_summary_by_position_goes_red(built, committed) -> None:
    b300 = built.b300
    misread = _mutant(BY_POSITION)
    with pytest.raises(
        misread.Refused,
        match="fixed-const: assess gives invariant; the gate summary says divergent",
    ):
        misread.gate_section(b300)
    blind = _mutant(
        BY_POSITION, NO_VERDICT_CHECK, NO_DISTINCT_CHECK, NO_PADDING_CHECK, NO_READING_CHECK
    )
    got = blind.gate_section(b300)
    assert got != committed["b300"]["gate"]


def test_mutation_reading_the_verdict_the_wrong_way_round_goes_red(built) -> None:
    wrong = _mutant(WRONG_WAY_ROUND)
    text = wrong.render(wrong.assemble(*built))
    assert text.encode("utf-8") != SUMMARY.read_bytes()
    arms = json.loads(text)["b300"]["gate"]["arms"]
    assert arms["fixed-churn"]["verdict"] == "divergent"


NO_C64_TIMING = (
    '    same = twice["identity"]',
    '    same = {**twice["identity"], "timing_only": 0}',
)
NO_GATE_RECOUNT = ("    if counted != recount:", "    if False:")
C64_AGAINST_ITSELF = (
    '        "ragged_vs_ragged": ragged_vs_ragged(b300, *C64),',
    '        "ragged_vs_ragged": ragged_vs_ragged(b300, C64[0], C64[0]),',
)
PARTIALS_AMONG_SENT = (
    '        "recordings_without_partials": sum(not e.get("partials") for e in recordings),',
    '        "recordings_without_partials": sum(not e.get("partials") for e in sent),',
)


def test_mutation_a_c64_comparison_that_stops_counting_timing_only_goes_red(
    built, committed
) -> None:
    b300 = built.b300
    want = committed["b300"]["retest_c64"]
    assert s2.retest_c64(b300) == want
    blind = _mutant(NO_C64_TIMING)
    with pytest.raises(
        blind.Refused,
        match=r"c64-run1.json.gz against .*c64-run2.json.gz: the comparator counts "
        r"\{'identical': 2931, 'text': 4, 'words': 0\}; the gate's diff_records gives "
        r"\{'identical': 2931, 'text': 4, 'words': 4\}",
    ):
        blind.retest_c64(b300)
    blinder = _mutant(NO_C64_TIMING, NO_GATE_RECOUNT)
    got = blinder.retest_c64(b300)
    assert got["ragged_vs_ragged"]["timing_only"] == 0
    assert want["ragged_vs_ragged"]["timing_only"] == 4
    assert got != want


def test_mutation_comparing_a_c64_run_with_itself_goes_red(built) -> None:
    b300 = built.b300
    itself = _mutant(C64_AGAINST_ITSELF)
    with pytest.raises(itself.Refused, match="are the same capture: the same bytes"):
        itself.retest_c64(b300)


def test_mutation_counting_partials_among_the_sent_only_goes_red(built) -> None:
    b300 = built.b300
    narrow = _mutant(PARTIALS_AMONG_SENT)
    with pytest.raises(narrow.Refused, match=r"'recordings_without_partials': 96\}"):
        narrow.first_attempt(b300)
