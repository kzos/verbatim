# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``scripts/step1_summary.py``: the committed summary is what the committed records give.

The records are committed (``rows/exploratory/step1-a6000-2026-09-26/``), so nothing here is
skipped. Scoring them takes minutes (1,000 null draws for each flag), so it runs once for this
module, in worker processes, and every test that needs it reads that one result.

* The summary reproduces, byte for byte, from the committed gz files.
* A gz file with one word changed is refused on the manifest's sha256: of the file, and, when
  the manifest's file sha256 is changed to match, of its decompressed bytes. A commit the
  record does not stamp, a file the manifest does not list, a places record that does not
  name the captures here, and a record holding a machine path are refused too.
* Mutations, each a copy of the script with a source edit: one that stops counting timing-only
  differences (in a stock arm, and in the server's fixed against ragged comparison) is refused
  by the recount against the records' own counters, and with that check removed as well gives
  a section that is not the committed one, so the byte-for-byte test goes red. Likewise one
  that swaps the flips and the confidence flag.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "step1_summary.py"
RECORDS = ROOT / "rows" / "exploratory" / "step1-a6000-2026-09-26"
SUMMARY = ROOT / "rows" / "exploratory" / "step1-summary-2026-09-26.json"

# Imported by name from scripts/, the way a spawned scoring worker imports it.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
s1 = importlib.import_module("step1_summary")

#: The smallest record, and one word in it (a transcript's) that the tampering changes.
TAMPERED = "replay-22e8406-nemotron-bf16-1120-n64.json.gz"
WORD, OTHER = b"cutter", b"cotter"


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """A temporary directory removed when its test ends; pytest's own is kept for three runs."""
    with tempfile.TemporaryDirectory(prefix="step1-summary-test-") as name:
        yield Path(name)


@pytest.fixture(scope="module")
def built() -> tuple[Any, dict[tuple[str, str], Any]]:
    inputs = s1.load_inputs(RECORDS)
    return inputs, s1.derive_all(inputs, jobs=min(3, os.cpu_count() or 1))


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


# --- the committed summary is the records' ------------------------------------------------


def test_the_summary_reproduces_byte_for_byte_from_the_committed_records(built) -> None:
    inputs, derived = built
    assert s1.render(s1.assemble(inputs, derived)).encode("utf-8") == SUMMARY.read_bytes()


def test_the_summary_names_its_inputs_by_the_manifest(committed) -> None:
    manifest = (RECORDS / s1.MANIFEST).read_bytes()
    assert committed["inputs"]["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    entries = json.loads(manifest)["records"]
    assert committed["inputs"]["sha256_raw"] == {n: e["sha256_raw"] for n, e in entries.items()}


# --- refused before anything is derived ---------------------------------------------------


def _copy(tmp_path: Path) -> Path:
    dest = tmp_path / RECORDS.name
    shutil.copytree(RECORDS, dest)
    return dest


def _tamper(directory: Path) -> tuple[bytes, bytes]:
    """Change one word of one record, recompressed the way the manifest says; (gz, raw)."""
    path = directory / TAMPERED
    raw = gzip.decompress(path.read_bytes())
    assert raw.count(WORD) >= 1
    changed = raw.replace(WORD, OTHER, 1)
    json.loads(changed)  # still a record, one word apart
    packed = gzip.compress(changed, compresslevel=9, mtime=0)
    path.write_bytes(packed)
    return packed, changed


def _edit_manifest(directory: Path, name: str, **fields: Any) -> None:
    path = directory / s1.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["records"][name].update(fields)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def test_a_record_with_one_word_changed_is_refused_on_the_manifests_sha256(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _copy(tmp_path)
    packed, _ = _tamper(directory)
    with pytest.raises(s1.Refused, match=f"{TAMPERED}: its sha256 is {sha(packed)}, the manifest"):
        s1.load_inputs(directory)
    # ... and the script says so and exits 2, writing nothing.
    out = tmp_path / "summary.json"
    assert s1.main(["--records", str(directory), "--out", str(out), "--jobs", "1"]) == 2
    assert "the manifest says" in capsys.readouterr().err
    assert not out.exists()


def test_a_changed_word_under_a_matching_file_sha256_is_refused_on_the_raw_sha256(
    tmp_path: Path,
) -> None:
    directory = _copy(tmp_path)
    packed, changed = _tamper(directory)
    _edit_manifest(directory, TAMPERED, sha256_gz=sha(packed), bytes_gz=len(packed))
    with pytest.raises(
        s1.Refused, match=f"the sha256 of its decompressed bytes is {sha(changed)}, the manifest"
    ):
        s1.load_inputs(directory)


def test_a_commit_the_record_does_not_stamp_is_refused(tmp_path: Path) -> None:
    directory = _copy(tmp_path)
    _edit_manifest(directory, TAMPERED, commit="a207de6db3fb2601b225425f515e4b0ab6f6f50e")
    with pytest.raises(s1.Refused, match=f"{TAMPERED}: verbatim_commit is '22e84061"):
        s1.load_inputs(directory)


def test_a_file_the_manifest_does_not_list_is_refused(tmp_path: Path) -> None:
    directory = _copy(tmp_path)
    shutil.copy(directory / TAMPERED, directory / "extra.json.gz")
    with pytest.raises(s1.Refused, match=r"not \['extra.json.gz'\] that are"):
        s1.load_inputs(directory)


def test_a_places_record_that_does_not_name_the_captures_here_is_refused(tmp_path: Path) -> None:
    directory = _copy(tmp_path)
    entries = json.loads((directory / s1.MANIFEST).read_text(encoding="utf-8"))["records"]
    ragged, other = (
        entries[n]["sha256_raw"] for n in (s1.capture("a207de6", "ragged-c32"), s1.RAGGED_REPEAT)
    )
    name = s1.PLACES["a207de6"]
    raw = gzip.decompress((directory / name).read_bytes())
    # The places record names its ragged capture by sha256; here it names the other one, and
    # the manifest is made to agree with the file, so only the link can refuse it.
    assert raw.count(ragged.encode()) == 1
    changed = raw.replace(ragged.encode(), other.encode())
    packed = gzip.compress(changed, compresslevel=9, mtime=0)
    (directory / name).write_bytes(packed)
    _edit_manifest(
        directory,
        name,
        sha256_gz=sha(packed),
        bytes_gz=len(packed),
        sha256_raw=sha(changed),
        bytes_raw=len(changed),
    )
    with pytest.raises(s1.Refused, match=f"{name}: its b_ragged capture has sha256 '{other}'"):
        s1.load_inputs(directory)


def test_a_record_holding_a_machine_path_is_refused_though_the_manifest_agrees(
    tmp_path: Path,
) -> None:
    directory = _copy(tmp_path)
    raw = gzip.decompress((directory / TAMPERED).read_bytes())
    # The scrub left a placeholder where the replay's checkout was; put a path back in its place
    # and make the manifest agree with the file, so only the path can refuse it.
    assert raw.count(b'"<checkout>"') == 1
    changed = raw.replace(b'"<checkout>"', b'"/srv/someone/verbatim"')
    packed = gzip.compress(changed, compresslevel=9, mtime=0)
    (directory / TAMPERED).write_bytes(packed)
    _edit_manifest(
        directory,
        TAMPERED,
        sha256_gz=sha(packed),
        bytes_gz=len(packed),
        sha256_raw=sha(changed),
        bytes_raw=len(changed),
    )
    with pytest.raises(s1.Refused, match=f"{TAMPERED}: holds a machine path: /code/checkout"):
        s1.load_inputs(directory)


def test_a_capture_at_another_setting_is_refused() -> None:
    names = [s1.capture("a207de6", "fixed-c32"), s1.capture("a207de6", "fixed-c8")]
    records = {n: json.loads(gzip.decompress((RECORDS / n).read_bytes())) for n in names}
    inputs = s1.Inputs(RECORDS, {}, "", records)
    assert s1.common_setting(inputs, names)["bucket"] == 128
    other = records[names[1]]
    server = {**other["server"], "bucket": {**other["server"]["bucket"], "reported_by_server": 64}}
    records[names[1]] = {**other, "server": server}
    with pytest.raises(s1.Refused, match="differ in bucket"):
        s1.common_setting(inputs, names)


def test_a_machine_path_is_never_written() -> None:
    assert s1.machine_paths({"a": ["scripts/step1_places.py", "20260924/bfloat16/x/a"]}) == []
    with pytest.raises(s1.Refused, match="machine path"):
        s1.render({"a": {"b": ["fine", "read from /srv/data/run/fixed.json"]}})


def test_an_unscored_confidence_flag_is_refused() -> None:
    arm = {"confidence": {"a": {"status": "confidence unavailable", "reason": "no words"}}}
    derived = {
        ("flips", m): {"runs": {"bfloat16": {"arms": {s1.sp.CAPTURES_ARM: arm}}}}
        for m in s1.CONFIDENCE
    }
    with pytest.raises(s1.Refused, match="the confidence flag was not scored"):
        s1.flips_section(None, derived)


# --- mutations: a summary that stops counting, or swaps the flags, goes red -----------------


def _mutant(*edits: tuple[str, str]) -> types.ModuleType:
    """A copy of the script with source edits applied; each edit must match once."""
    source = SCRIPT.read_text(encoding="utf-8")
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType("step1_summary_mutant")
    module.__dict__["__file__"] = str(SCRIPT)
    sys.modules[module.__name__] = module
    exec(compile(source, "step1_summary_mutant", "exec"), module.__dict__)
    return module


NO_STOCK_TIMING = (
    '    timing = [d for d in arm["divergences"] if not d["text_differs"]]',
    "    timing = []",
)
NO_STOCK_RECOUNT = ("    if recount != counters or scorer != counters:", "    if False:")
NO_SERVER_TIMING = (
    '    same = apart["identity"]',
    '    same = {**apart["identity"], "timing_only": 0}',
)
NO_SERVER_RECOUNT = ("    if counted != stored:", "    if False:")
SWAPPED = (
    '        flips, ranked = arm["flag"]["a"], confidence["flag"]',
    '        ranked, flips = arm["flag"]["a"], confidence["flag"]',
)
NO_SHAPE_CHECK = ("        if not shaped:", "        if False:")


def test_mutation_a_stock_arm_that_stops_counting_timing_only_goes_red(built, committed) -> None:
    inputs, derived = built
    key = "nemotron-bf16-1120"
    record, scored = inputs.records[s1.STOCK[key]], derived[("stock", key)]
    assert s1.stock_section(s1.STOCK[key], record, scored) == committed["stock"][key]
    blind = _mutant(NO_STOCK_TIMING)
    with pytest.raises(blind.Refused, match="recounted 211 text and 0 timing-only divergences"):
        blind.stock_section(s1.STOCK[key], record, scored)
    blinder = _mutant(NO_STOCK_TIMING, NO_STOCK_RECOUNT)
    section = blinder.stock_section(s1.STOCK[key], record, scored)
    ragged = section["runs"]["bfloat16"]["arms"]["ragged"]
    assert ragged["timing_only"] == 0
    assert committed["stock"][key]["runs"]["bfloat16"]["arms"]["ragged"]["timing_only"] == 1038
    assert section != committed["stock"][key]


def test_mutation_a_server_comparison_that_stops_counting_timing_only_goes_red(
    built, committed
) -> None:
    inputs, derived = built
    want = committed["server"]["ragged_vs_fixed"]["a207de6"]
    assert s1.ragged_vs_fixed(inputs, derived, "a207de6") == want
    blind = _mutant(NO_SERVER_TIMING)
    with pytest.raises(blind.Refused, match="differ on 349 texts and 0 timings only; the places"):
        blind.ragged_vs_fixed(inputs, derived, "a207de6")
    blinder = _mutant(NO_SERVER_TIMING, NO_SERVER_RECOUNT)
    got = blinder.ragged_vs_fixed(inputs, derived, "a207de6")
    assert got["timing_only"] == 0 and want["timing_only"] == 1028
    assert got != want


def test_mutation_swapping_the_flips_and_the_confidence_flag_goes_red(built, committed) -> None:
    inputs, derived = built
    want = committed["flips_vs_confidence"]
    assert s1.flips_section(inputs, derived) == want
    swapped = _mutant(SWAPPED)
    with pytest.raises(swapped.Refused, match="the flips are 226 spans for 189 places"):
        swapped.flips_section(inputs, derived)
    unchecked = _mutant(SWAPPED, NO_SHAPE_CHECK)
    got = unchecked.flips_section(inputs, derived)
    mode = got["modes"]["paper-best"]
    assert mode["flips"] == want["modes"]["paper-best"]["confidence"]
    assert mode["confidence"] == want["modes"]["paper-best"]["flips"]
    assert got != want


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
