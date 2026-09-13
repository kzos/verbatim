# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The batch-invariance gate against an in-process server over the CPU fake.

The fake is invariant by construction, so a gate that had only ever seen it could pass
for free. `batch_leak` makes the fake's output depend on how many live rows share its
batch, one channel at a time, and the tests here pin that the gate goes red on each:
the text, then the timings with the text unchanged. A gate that cannot fail proves
nothing, and this file is where it is shown that this one can.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from verbatim_bench.canonical import FinalRecord
from verbatim_bench.cli import main
from verbatim_bench.client import ChunkMode as BenchChunk
from verbatim_bench.invariance import (
    DEFAULT_MAX_CONCURRENCY,
    EXIT_DIVERGENT,
    EXIT_INVARIANT,
    EXIT_NO_VERDICT,
    SLOTS,
    Clip,
    GateRefusal,
    Level,
    LevelRun,
    assess,
    check_levels,
    default_levels,
    diff_records,
    run_gate,
    synthetic_corpus,
)

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import Engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.ws.server import WsServer, WsServerConfig

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[2]
CHUNK_MS = 80
BUCKET = 8
#: Four slots, small concurrencies: the slots name the row fields, the numbers are what ran.
LEVELS = (Level("1", 1), Level("32a", 4), Level("32b", 4), Level("max", BUCKET))
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@asynccontextmanager
async def _server(batch_leak: Sequence[str] = ()) -> AsyncIterator[str]:
    """A real engine on the real clock over the hash-mode fake, on an ephemeral port."""
    chunk = ChunkMode(CHUNK_MS)
    config = EngineConfig(chunk=chunk, buckets=(BUCKET,), idle_timeout_s=None)
    pipeline = FakePipelineAdapter(chunk, buckets=(BUCKET,), batch_leak=batch_leak)
    engine = Engine(config, pipeline)
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        yield server.endpoint


def _clips(n: int = BUCKET, chunks: int = 2) -> list[Clip]:
    return synthetic_corpus(n, duration_s=chunks * CHUNK_MS / 1000, seed=7)


async def _gate(endpoint: str):
    return await run_gate(endpoint, _clips(), LEVELS, chunk=BenchChunk(CHUNK_MS))


# --- the gate against the fake ---


@pytest.mark.asyncio
async def test_the_invariant_fake_is_reported_invariant_with_both_channels_compared() -> None:
    async with _server() as endpoint:
        report = await _gate(endpoint)
    assert report.verdict == "invariant"
    assert report.exit_code == EXIT_INVARIANT
    assert report.equal is True
    assert report.divergences == []
    digests = report.digests
    assert set(digests) == set(SLOTS)
    assert len({digests[slot] for slot in SLOTS}) == 1
    assert all(HEX64.match(digests[slot]) for slot in SLOTS)
    # Not vacuous: every stream at every level produced text and word timings, and the
    # timings are the fake's chunk-indexed integers, so both channels were compared.
    for run in report.runs:
        assert run.errors == {}
        assert len(run.finals) == BUCKET
        for record in run.finals.values():
            assert record.text
            assert record.words
            assert all(
                (s, e) == (j * CHUNK_MS, (j + 1) * CHUNK_MS)
                for j, (_, s, e) in enumerate(record.words)
            )
    assert report.timings_present is True
    assert "FINAL: invariant at 1 / 32a / 32b / max" in report.render()


@pytest.mark.asyncio
async def test_a_text_leak_turns_the_gate_red_and_names_the_streams() -> None:
    async with _server(batch_leak=("text",)) as endpoint:
        report = await _gate(endpoint)
    assert report.verdict == "divergent"
    assert report.exit_code == EXIT_DIVERGENT
    assert report.equal is False
    assert report.digests["1"] != report.digests["max"]
    assert report.divergences
    assert {d.kind for d in report.divergences} == {"text"}
    against_alone = [d for d in report.divergences if d.against == "1"]
    assert against_alone
    for d in against_alone:
        assert d.stream_id.startswith("synthetic-")
        assert d.index is not None
        assert d.left != d.right
        assert d.classification == "batch dependence"
    text = report.render()
    assert "*** DIVERGENCE #1 stream synthetic-" in text
    assert "FINAL: divergent," in text


@pytest.mark.asyncio
async def test_a_timing_leak_turns_the_gate_red_with_the_text_unchanged() -> None:
    """The timing channel is compared on its own: a server whose words shift by a
    millisecond with the batch is caught even though every transcript is identical."""
    async with _server(batch_leak=("timing",)) as endpoint:
        report = await _gate(endpoint)
    assert report.verdict == "divergent"
    assert report.equal is False
    assert {d.kind for d in report.divergences} == {"words"}
    by_slot = {run.level.slot: run for run in report.runs}
    for slot in SLOTS:
        for stream_id, record in by_slot[slot].finals.items():
            assert record.text == by_slot["1"].finals[stream_id].text
    first = next(d for d in report.divergences if d.against == "1")
    assert first.left != first.right
    assert first.left.startswith("('w") and first.right.startswith("('w")


# --- the pure parts ---


def _run(slot: str, concurrency: int, finals: dict[str, FinalRecord], **kw: object) -> LevelRun:
    return LevelRun(Level(slot, concurrency), finals, **kw)  # type: ignore[arg-type]


def _same_runs(text: str = "one two", words: tuple = (("one", 0, 80), ("two", 80, 160))):
    record = FinalRecord("s", text, words)
    return [
        _run("1", 1, {"s": record}),
        _run("32a", 32, {"s": record}),
        _run("32b", 32, {"s": record}),
        _run("max", 128, {"s": record}),
    ]


def test_default_levels_are_1_32a_32b_max_with_the_documented_maximum() -> None:
    levels = default_levels()
    assert [(x.slot, x.concurrency) for x in levels] == [
        ("1", 1),
        ("32a", 32),
        ("32b", 32),
        ("max", DEFAULT_MAX_CONCURRENCY),
    ]
    assert DEFAULT_MAX_CONCURRENCY == 128
    assert default_levels(112)[-1].concurrency == 112
    with pytest.raises(GateRefusal):
        default_levels(0)


def test_a_corpus_smaller_than_the_highest_level_is_refused_before_a_byte_is_sent() -> None:
    with pytest.raises(GateRefusal, match="cannot reach concurrency 128"):
        check_levels(default_levels(), corpus_size=127)
    check_levels(default_levels(), corpus_size=128)
    with pytest.raises(GateRefusal, match="slots"):
        check_levels((Level("1", 1), Level("max", 8)), corpus_size=8)


def test_a_server_that_transcribes_nothing_is_vacuous_not_invariant() -> None:
    report = assess(_same_runs(text="", words=()))
    assert report.verdict == "vacuous"
    assert report.exit_code == EXIT_NO_VERDICT
    assert report.equal is True  # the digests are equal; that is exactly why it is refused
    assert "proves nothing" in report.render()


def test_an_errored_level_has_no_digest_and_the_run_has_no_verdict() -> None:
    runs = _same_runs()
    runs[2] = _run("32b", 32, {}, errors={"s": "ServerError 8: at capacity"})
    report = assess(runs)
    assert report.verdict == "incomplete"
    assert report.exit_code == EXIT_NO_VERDICT
    assert report.digests["32b"] is None
    assert report.equal is None
    assert report.divergences == []
    assert "level 32b had 1 errored sessions (first: ServerError 8: at capacity)" in report.render()
    block = report.to_json_dict()["invariance"]
    assert block == {
        "hash_1": report.digests["1"],
        "hash_32a": report.digests["32a"],
        "hash_32b": None,
        "hash_max": report.digests["max"],
        "equal": None,
    }


def test_a_divergence_reports_the_first_differing_word_on_both_sides() -> None:
    left = FinalRecord("s", "the quick brown fox", (("the", 0, 80), ("quick", 80, 160)))
    assert diff_records(left, left) is None
    right = FinalRecord("s", "the quick brawn fox", left.words)
    assert diff_records(left, right) == ("text", 2, "brown", "brawn")
    shorter = FinalRecord("s", "the quick", left.words)
    assert diff_records(left, shorter) == ("text", 2, "brown", "<end>")
    spaced = FinalRecord("s", "the  quick brown fox", left.words)
    assert diff_records(left, spaced) == ("text", None, repr(left.text), repr(spaced.text))
    shifted = FinalRecord("s", left.text, (("the", 0, 80), ("quick", 80, 161)))
    assert diff_records(left, shifted) == (
        "words",
        1,
        "('quick', 80, 160)",
        "('quick', 80, 161)",
    )
    fewer = FinalRecord("s", left.text, (("the", 0, 80),))
    assert diff_records(left, fewer) == ("words", 1, "('quick', 80, 160)", "<end>")


def test_run_to_run_difference_is_named_apart_from_batch_dependence() -> None:
    runs = _same_runs()
    other = FinalRecord("s", "one three", (("one", 0, 80), ("three", 80, 160)))
    runs[2] = _run("32b", 32, {"s": other})
    report = assess(runs)
    assert report.verdict == "divergent"
    pairs = {(d.level, d.against, d.classification) for d in report.divergences}
    assert pairs == {
        ("32b", "1", "batch dependence"),
        ("32b", "32a", "run-to-run at one concurrency"),
    }
    assert "0 differ between 32a and 32b" not in report.render()
    assert "1 differ between 32a and 32b" in report.render()


def test_a_stream_missing_from_one_level_is_a_divergence_not_a_pass() -> None:
    runs = _same_runs()
    runs[3] = _run("max", 128, {})
    report = assess(runs)
    assert report.verdict == "divergent"
    assert [(d.kind, d.level, d.left, d.right) for d in report.divergences] == [
        ("missing", "max", "<present>", "<absent>")
    ]


def test_the_record_carries_the_row_block_and_every_level() -> None:
    report = assess(
        _same_runs(), endpoint="ws://x/v1/stream", chunk_ms=160, corpus={"kind": "synthetic"}
    )
    doc = report.to_json_dict()
    assert doc["record"] == "vb-invariance/1"
    assert doc["invariance"]["equal"] is True
    assert set(doc["invariance"]) == {"hash_1", "hash_32a", "hash_32b", "hash_max", "equal"}
    assert [x["slot"] for x in doc["levels"]] == list(SLOTS)
    assert [x["concurrency"] for x in doc["levels"]] == [1, 32, 32, 128]
    assert doc["verdict"] == "invariant"
    assert doc["exit_code"] == 0
    json.dumps(doc)


def test_the_synthetic_corpus_is_reproducible_and_sized_to_the_request() -> None:
    a = synthetic_corpus(3, duration_s=0.1, seed=1)
    b = synthetic_corpus(3, duration_s=0.1, seed=1)
    assert [c.pcm for c in a] == [c.pcm for c in b]
    assert len({c.pcm for c in a}) == 3
    assert all(len(c.pcm) == 2 * 1600 for c in a)
    assert [c.stream_id for c in a] == ["synthetic-0000", "synthetic-0001", "synthetic-0002"]
    assert synthetic_corpus(3, duration_s=0.1, seed=2)[0].pcm != a[0].pcm
    with pytest.raises(GateRefusal):
        synthetic_corpus(0)


# --- the command, against `verbatim serve --pipeline fake` ---


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_command_runs_the_gate_against_a_served_fake(tmp_path: Path) -> None:
    """What CI runs, at 1 / 32a / 32b / max with max = the served bucket: exit 0, the
    four digests equal, the record written. One-chunk clips keep the sequential level
    short; the cost is the corpus's duration at concurrency 1."""
    ws_port = _free_port()
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO / 'bench' / 'src'}"}
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
        out = tmp_path / "invariance.json"
        code = main(
            [
                "invariance",
                "--endpoint",
                f"ws://127.0.0.1:{ws_port}/v1/stream",
                "--synthetic",
                "32",
                "--synthetic-s",
                str(CHUNK_MS / 1000),
                "--chunk",
                f"{CHUNK_MS}ms",
                "--max",
                "32",
                "--out",
                str(out),
            ]
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert code == EXIT_INVARIANT
    doc = json.loads(out.read_text())
    block = doc["invariance"]
    assert block["equal"] is True
    assert len({block[f"hash_{slot}"] for slot in SLOTS}) == 1
    assert all(x["errors"] == 0 and x["streams"] == 32 for x in doc["levels"])
    assert doc["timings_present"] is True


def test_the_command_refuses_a_corpus_that_cannot_reach_its_max(
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["invariance", "--endpoint", "ws://127.0.0.1:1/v1/stream", "--synthetic", "8"])
    assert code == EXIT_NO_VERDICT
    assert (
        "refused: a corpus of 8 utterances cannot reach concurrency 128" in capsys.readouterr().out
    )
