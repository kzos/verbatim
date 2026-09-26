# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``scripts/scrub_record_paths.py``: each guard it adds is shown able to fail.

* On a directory built here (a capture holding paths, a places record naming the capture by
  sha256, a stock record with none), the scrub replaces every given prefix by its placeholder,
  rewrites the link to the capture's new sha256, leaves the stock record byte for byte, keeps
  each rewritten record's sha256 as written, and a second run changes nothing.
* A path whose prefix was not given is refused, and nothing is written.
* A prefix is replaced only as a whole path or the head of one, never inside a longer name.
* A gz file whose sha256 is not the manifest's is refused.
* ``machine_paths`` finds a path wherever it stands in a string (a PYTHONPATH's second entry
  included) and passes a placeholder's tail, a URL, a model id and the kernel's ``/proc/``.
* The committed step-1 records hold no machine path; every one the scrub rewrote says so in the
  manifest, and the four that held none are as written.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "rows" / "exploratory" / "step1-a6000-2026-09-26"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
scrub = importlib.import_module("scrub_record_paths")

pytestmark = pytest.mark.skipif(shutil.which("gzip") is None, reason="gzip is not on PATH")

CHECKOUT = "/work/alice/checkouts/verbatim-x"
HOME = "/work/alice"
PREFIXES = [f"checkout={CHECKOUT}", f"home={HOME}"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put(directory: Path, name: str, record: dict) -> dict:
    raw = (json.dumps(record, indent=1) + "\n").encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    (directory / name).write_bytes(packed)
    return {
        "sha256_raw": sha(raw),
        "bytes_raw": len(raw),
        "sha256_gz": sha(packed),
        "bytes_gz": len(packed),
        "what": name,
    }


def build(directory: Path, extra_path: str | None = None) -> None:
    capture = {
        "client": {"checkout": CHECKOUT, "imported_from": f"{CHECKOUT}/src/verbatim/__init__.py"},
        "environ": {"HOME": HOME, "PYTHONPATH": f"{CHECKOUT}/src:{CHECKOUT}/bench/src"},
        "hub": f"{HOME}/.cache/huggingface/hub",
        "how": "read from /proc/<pid>/environ",
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "url": "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
        "neighbour": f"{CHECKOUT}-other/src",
        "words": ["it", "is"],
    }
    if extra_path is not None:
        capture["log"] = extra_path
    entries = {"capture.json.gz": put(directory, "capture.json.gz", capture)}
    places = {"captures": {"a": {"sha256": entries["capture.json.gz"]["sha256_raw"]}}}
    entries["places.json.gz"] = put(directory, "places.json.gz", places)
    entries["stock.json.gz"] = put(directory, "stock.json.gz", {"words": ["no", "path"]})
    manifest = {"gzip": "gzip -9 -n", "records": entries}
    (directory / scrub.MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def read(directory: Path, name: str) -> tuple[bytes, dict]:
    raw = gzip.decompress((directory / name).read_bytes())
    return raw, json.loads(raw)


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_the_scrub_replaces_relinks_keeps_as_written_and_settles(tmp_path: Path) -> None:
    build(tmp_path)
    before = json.loads((tmp_path / scrub.MANIFEST).read_text(encoding="utf-8"))["records"]
    stock_before = (tmp_path / "stock.json.gz").read_bytes()
    assert scrub.main([str(tmp_path), *(f"--prefix={p}" for p in PREFIXES)]) == 0
    manifest = json.loads((tmp_path / scrub.MANIFEST).read_text(encoding="utf-8"))
    raw, capture = read(tmp_path, "capture.json.gz")
    assert capture["client"]["checkout"] == "<checkout>"
    assert capture["environ"] == {
        "HOME": "<home>",
        "PYTHONPATH": "<checkout>/src:<checkout>/bench/src",
    }
    assert capture["hub"] == "<home>/.cache/huggingface/hub"
    # Only whole paths: the longer sibling keeps its own name after the home placeholder.
    assert capture["neighbour"] == "<home>/checkouts/verbatim-x-other/src"
    entry = manifest["records"]["capture.json.gz"]
    assert entry["sha256_raw"] == sha(raw)
    assert entry["sha256_gz"] == sha((tmp_path / "capture.json.gz").read_bytes())
    assert entry["sha256_as_written"] == before["capture.json.gz"]["sha256_raw"]
    assert entry["scrub"] == {"placeholders": {"checkout": 4, "home": 3}, "links_rewritten": 0}
    # The places record names the scrubbed capture, and says it was rewritten.
    _, places = read(tmp_path, "places.json.gz")
    assert places["captures"]["a"]["sha256"] == entry["sha256_raw"]
    assert manifest["records"]["places.json.gz"]["scrub"]["links_rewritten"] == 1
    # The stock record held nothing to replace: as written, and its entry untouched.
    assert (tmp_path / "stock.json.gz").read_bytes() == stock_before
    assert manifest["records"]["stock.json.gz"] == before["stock.json.gz"]
    assert "<checkout>" in manifest["scrubbed"] and CHECKOUT not in json.dumps(manifest)
    # A second run finds nothing to do.
    settled = snapshot(tmp_path)
    assert scrub.main([str(tmp_path), *(f"--prefix={p}" for p in PREFIXES)]) == 0
    assert snapshot(tmp_path) == settled


def test_a_path_whose_prefix_was_not_given_is_refused_and_nothing_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build(tmp_path, extra_path="/srv/runs/step1/server.log")
    before = snapshot(tmp_path)
    assert scrub.main([str(tmp_path), *(f"--prefix={p}" for p in PREFIXES)]) == 2
    err = capsys.readouterr().err
    assert "capture.json.gz still holds a machine path: /log: '/srv/runs/step1/server.log'" in err
    assert snapshot(tmp_path) == before


def test_a_record_that_is_not_the_manifests_is_refused(tmp_path: Path) -> None:
    build(tmp_path)
    raw, _ = read(tmp_path, "stock.json.gz")
    (tmp_path / "stock.json.gz").write_bytes(gzip.compress(raw.replace(b"no", b"on"), mtime=0))
    with pytest.raises(scrub.Refused, match=r"stock\.json\.gz: its sha256 is"):
        scrub.scrub(tmp_path, scrub.parse_prefixes(PREFIXES))


@pytest.mark.parametrize(
    "given",
    ["checkout", "nobody=/x/y", "home=relative/path", "home=/x/y/", f"home={HOME}"],
)
def test_a_prefix_that_is_not_a_placeholder_and_an_absolute_path_is_refused(given: str) -> None:
    extra = [f"home={HOME}"] if given == f"home={HOME}" else []
    with pytest.raises(scrub.Refused):
        scrub.parse_prefixes([given, *extra])


def test_machine_paths_finds_a_path_anywhere_and_passes_what_is_not_one() -> None:
    fine = {
        "a": "<checkout>/src/verbatim",
        "b": "PYTHONPATH <checkout>/src:<checkout>/bench/src",
        "c": "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
        "d": "nvidia/nemotron-speech-streaming-en-0.6b",
        "e": "read from /proc/<pid>/environ",
        "f": ["run-22e8406/server-fixed-c32.json", "20260924/bfloat16/x/a"],
    }
    assert scrub.machine_paths(fine) == []
    for text in (
        "/srv/data/run/fixed.json",
        "PYTHONPATH <checkout>/src:/srv/other/src",
        "2928912, /opt/env/bin/python, 7664",
        "HOME=/root/x",
    ):
        assert scrub.machine_paths({"k": [text]}) == [f"/k[0]: {text!r}"]


def test_the_committed_records_hold_no_machine_path() -> None:
    manifest = json.loads((RECORDS / scrub.MANIFEST).read_text(encoding="utf-8"))
    assert scrub.machine_paths(manifest) == []
    scrubbed = 0
    for name, entry in manifest["records"].items():
        raw = gzip.decompress((RECORDS / name).read_bytes())
        assert sha(raw) == entry["sha256_raw"], name
        assert scrub.machine_paths(json.loads(raw)) == [], name
        if "scrub" in entry:
            scrubbed += 1
            assert entry["sha256_as_written"] != entry["sha256_raw"], name
            assert sum(entry["scrub"]["placeholders"].values()) > 0, name
        else:
            assert name.startswith("stock-"), name
    assert scrubbed == 12
