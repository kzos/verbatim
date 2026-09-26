# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""`verbatim-bench invariance --finals-out`: the transcripts behind the digests.

The record keeps each level's digest and the first differing word of each stream. The
finals file keeps what was digested: per level and stream, the final text and the word
timings. The property pinned here is that each level's list re-digests, through the
gate's own canonicaliser, to the digest the record carries for that level, so the file
is the run the record describes and not a summary of it. CPU only: the command test
serves the fake pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from verbatim_bench.canonical import FinalRecord, finals_digest
from verbatim_bench.cli import _build_parser, main
from verbatim_bench.invariance import (
    EXIT_INVARIANT,
    FINALS_RECORD,
    SLOTS,
    Level,
    LevelRun,
    assess,
)

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
CHUNK_MS = 80


def _records(level: dict) -> list[FinalRecord]:
    return [
        FinalRecord(
            stream_id=entry["stream_id"],
            text=entry["text"],
            words=tuple((word, start, end) for word, start, end in entry["words"]),
        )
        for entry in level["finals"]
    ]


def _runs() -> list[LevelRun]:
    """Four levels over three streams, inserted out of order so the file's order is its
    own sort and not the order the sessions finished in."""
    finals = {
        "b-stream": FinalRecord("b-stream", "the cat sat", (("the", 0, 80), ("cat", 80, 160))),
        "B-stream": FinalRecord("B-stream", "hello", (("hello", 0, 160),)),
        "é-stream": FinalRecord("é-stream", "café au lait", (("café", 0, 80),)),
    }
    moved = dict(finals)
    moved["b-stream"] = FinalRecord("b-stream", "the cat sat", (("the", 0, 80), ("cat", 80, 240)))
    return [
        LevelRun(Level("1", 1), dict(finals)),
        LevelRun(Level("32a", 2), dict(finals)),
        LevelRun(
            Level("32b", 2),
            {k: v for k, v in finals.items() if k != "é-stream"},
            errors={"é-stream": "closed 1011"},
        ),
        LevelRun(Level("max", 3, churn_period_s=2.0), moved),
    ]


def test_each_level_re_digests_to_the_digest_its_run_carries() -> None:
    report = assess(_runs(), endpoint="ws://x/v1/stream", chunk_ms=160, corpus={"id": "c"})
    doc = json.loads(json.dumps(report.finals_json_dict()))  # what the file holds
    assert doc["record"] == FINALS_RECORD
    assert [level["slot"] for level in doc["levels"]] == list(SLOTS)
    for run, level in zip(report.runs, doc["levels"], strict=True):
        assert finals_digest(_records(level)) == finals_digest(run.finals.values())
        assert level["digest"] == run.digest
    # The timing-only change at max is in the file, word for word.
    max_level = doc["levels"][3]
    assert max_level["churn_period_s"] == 2.0
    moved = next(entry for entry in max_level["finals"] if entry["stream_id"] == "b-stream")
    assert moved["words"] == [["the", 0, 80], ["cat", 80, 240]]


def test_streams_are_in_byte_order_and_an_errored_stream_is_listed_as_an_error() -> None:
    report = assess(_runs())
    doc = report.finals_json_dict()
    ids = [entry["stream_id"] for entry in doc["levels"][0]["finals"]]
    assert ids == sorted(ids, key=lambda s: s.encode("utf-8"))
    assert ids == ["B-stream", "b-stream", "é-stream"]
    errored = doc["levels"][2]
    assert errored["digest"] is None
    assert errored["errors"] == {"é-stream": "closed 1011"}
    assert "é-stream" not in [entry["stream_id"] for entry in errored["finals"]]


def test_the_option_is_spelled_exactly_and_is_off_by_default() -> None:
    parser = _build_parser()
    sub = next(a for a in parser._subparsers._group_actions).choices["invariance"]
    assert "--finals-out" in sub._option_string_actions
    args = parser.parse_args(["invariance", "--endpoint", "ws://x/v1/stream", "--synthetic", "8"])
    assert args.finals_out is None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """This test's own directory, removed with everything in it when the test ends
    (pytest's tmp_path keeps the directories of its last runs)."""
    path = Path(tempfile.mkdtemp(prefix="finals-out-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_the_command_writes_the_finals_its_record_digested(scratch_dir: Path) -> None:
    """End to end over the served fake: the file is written, holds text and timings for
    every stream at every level, and re-digests to the record's four digests."""
    ws_port = _free_port()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO / 'bench' / 'src'}",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "verbatim",
            "serve",
            "stub",
            "--pipeline",
            "fake",
            "--chunk",
            f"{CHUNK_MS}ms",
            "--bucket",
            "32",
            "--host",
            "127.0.0.1",
            "--grpc-port",
            "0",
            "--ws-port",
            str(ws_port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    record = scratch_dir / "invariance.json"
    finals = scratch_dir / "nested" / "finals.json"
    try:
        assert proc.stdout is not None
        deadline = time.monotonic() + 30.0
        banner: list[str] = []
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            banner.append(line.rstrip())
            if "ready" in line:
                break
        assert any("ready" in line for line in banner), banner[-5:]
        code = main(
            [
                "invariance",
                "--endpoint",
                f"ws://127.0.0.1:{ws_port}/v1/stream",
                "--synthetic",
                "32",
                "--synthetic-s",
                str(2 * CHUNK_MS / 1000),
                "--chunk",
                f"{CHUNK_MS}ms",
                "--max",
                "32",
                "--out",
                str(record),
                "--finals-out",
                str(finals),
            ]
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert code == EXIT_INVARIANT
    rec = json.loads(record.read_text())
    doc = json.loads(finals.read_text())
    assert doc["record"] == FINALS_RECORD
    assert doc["corpus"] == rec["corpus"] and doc["verdict"] == rec["verdict"] == "invariant"
    assert [level["slot"] for level in doc["levels"]] == [lv["slot"] for lv in rec["levels"]]
    for level, rec_level in zip(doc["levels"], rec["levels"], strict=True):
        assert len(level["finals"]) == rec_level["streams"] == 32
        assert all(entry["text"] and entry["words"] for entry in level["finals"])
        assert finals_digest(_records(level)) == rec_level["digest"] == level["digest"]


def test_the_command_test_leaves_no_temporary_directory_behind(scratch_dir: Path) -> None:
    """The command test run in a child pytest whose TMPDIR is an empty directory of its
    own: that directory is empty again when the child exits."""
    tmpdir = scratch_dir / "tmpdir"
    tmpdir.mkdir()
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{Path(__file__).resolve()}::test_the_command_writes_the_finals_its_record_digested",
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "TMPDIR": str(tmpdir),
            "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO / 'bench' / 'src'}",
        },
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO),
    )
    assert done.returncode == 0 and "1 passed" in done.stdout, done.stdout[-2000:] + done.stderr
    assert list(tmpdir.iterdir()) == []
