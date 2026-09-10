# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for chunk modes, start-offset planning, paced replay, and results."""

from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from verbatim_bench import constants
from verbatim_bench.cli import main
from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import load_manifest, read_pcm16
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.pace import LoadSpec, plan_start_offsets, run_load
from verbatim_bench.results import RunResult, percentile, write_results

pytestmark = pytest.mark.cpu

RATE_HZ = 16000


def make_wav(path: Path, seconds: float) -> None:
    n = int(seconds * RATE_HZ)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(n) / RATE_HZ).astype(np.float32)
    sf.write(str(path), tone, RATE_HZ)


def make_manifest(path: Path, wav_paths: list[Path], stream_ids: list[str]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for wav, stream_id in zip(wav_paths, stream_ids, strict=True):
            handle.write(
                json.dumps(
                    {
                        "audio_filepath": wav.name,
                        "duration": 1.0,
                        "text": f"reference for {stream_id}",
                        "stream_id": stream_id,
                    }
                )
                + "\n"
            )
    return path


def make_spec(endpoint: str, manifest: Path, **overrides) -> LoadSpec:
    kwargs: dict = {
        "endpoint": endpoint,
        "manifest": manifest,
        "sessions": 4,
        "chunk": ChunkMode.parse(160),
        "seed": 11,
        "ramp_s": 0.05,
    }
    kwargs.update(overrides)
    return LoadSpec(**kwargs)


def test_chunk_mode_parsing() -> None:
    for spec in (160, "160", "160ms"):
        mode = ChunkMode.parse(spec)
        assert mode.samples == 2560
        assert mode.bytes == 5120
    with pytest.raises(ValueError):
        ChunkMode.parse("100ms")


def test_x_ms_defaults_to_the_frozen_constant(tmp_path: Path) -> None:
    assert make_spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl").x_ms == constants.X_MS


async def test_x_ms_is_read_from_the_spec_not_a_literal(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.2)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        result = await run_load(make_spec(server.endpoint, manifest, sessions=1, x_ms=50))
    output = write_results(result, tmp_path / "out")
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["config"]["x_ms"] == 50


@pytest.mark.parametrize("bad", ["1_60", "+160", "0160"])
def test_chunk_mode_rejects_non_canonical_spellings(bad: str) -> None:
    with pytest.raises(ValueError):
        ChunkMode.parse(bad)


def test_plan_start_offsets_is_deterministic(tmp_path: Path) -> None:
    base = make_spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl")
    first = plan_start_offsets(base)
    assert plan_start_offsets(base) == first
    altered = replace(base, seed=base.seed + 1)
    assert plan_start_offsets(altered) != first


def test_plan_start_offsets_uniform_spreads_over_the_ramp(tmp_path: Path) -> None:
    seed = 7
    spec = make_spec(
        "ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=8, ramp_s=4.0, seed=seed
    )
    offsets = plan_start_offsets(spec)
    rng = random.Random(seed)
    period_s = 160 / 1000.0
    phases = [rng.random() * period_s for _ in range(8)]
    for i, (offset, phase) in enumerate(zip(offsets, phases, strict=True)):
        assert offset - phase == pytest.approx(i * 4.0 / 7)
    assert all(0.0 <= offset < 4.0 + period_s for offset in offsets)


def test_plan_start_offsets_bursty_fits_the_window(tmp_path: Path) -> None:
    spec = make_spec(
        "ws://127.0.0.1:1/v1/stream",
        tmp_path / "m.jsonl",
        sessions=8,
        profile="bursty",
        burst_window_s=2.0,
    )
    period_s = 160 / 1000.0
    offsets = plan_start_offsets(spec)
    assert len(offsets) == 8
    assert all(0.0 <= offset < 2.0 + period_s for offset in offsets)


async def test_run_load_against_null_server(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        result = await run_load(make_spec(server.endpoint, manifest))
    assert result.to_json_dict()["result"]["sessions_completed"] == 4
    assert result.to_json_dict()["result"]["sessions_failed"] == 0
    for session in result.sessions:
        assert session.first_partial_ms is not None
        assert session.final_ms is not None
        assert len(session.partial_ms) >= 5
    assert result.to_json_dict()["result"]["chunks_sent"] == 4 * 7


async def test_run_load_is_paced(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        spec = make_spec(server.endpoint, manifest, sessions=1, ramp_s=0.0)
        started = time.monotonic()
        result = await run_load(spec)
        elapsed = time.monotonic() - started
    assert result.to_json_dict()["result"]["sessions_failed"] == 0
    assert elapsed >= 0.8, f"finished in {elapsed:.3f}s: faster than real time, not paced"
    assert elapsed <= 3.0, f"finished in {elapsed:.3f}s: pathologically slow"


async def test_session_error_is_recorded_not_raised(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.2)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    utterances = load_manifest(manifest)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
    session = await run_session(
        f"ws://127.0.0.1:{closed_port}/v1/stream",
        session_id="s0000",
        utterance=utterances[0],
        pcm=read_pcm16(wav),
        chunk=ChunkMode.parse(160),
        start_delay_s=0.0,
    )
    assert session.error is not None
    assert session.chunks == 0 and session.audio_s == 0.0
    run_result = RunResult(
        spec_dict={
            "arm": "test",
            "chunk_ms": 160,
            "seed": 0,
            "sessions": 1,
            "profile": "uniform",
            "endpoint": f"ws://127.0.0.1:{closed_port}/v1/stream",
            "words": False,
        },
        sessions=[session],
    )
    doc = run_result.to_json_dict()
    failed = doc["sessions"][0]
    assert failed["first_partial_ms"] is None
    assert failed["final_ms"] is None


async def test_final_latency_excludes_the_trailing_chunk_period(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    utterances = load_manifest(manifest)
    async with NullServer(NullServerConfig()) as server:
        session = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterances[0],
            pcm=read_pcm16(wav),
            chunk=ChunkMode.parse(160),
            start_delay_s=0.0,
        )
    assert session.error is None
    assert session.final_ms is not None
    assert session.final_ms < 80.0


async def test_server_error_frame_is_recorded_with_its_code(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    utterances = load_manifest(manifest)
    async with NullServer(NullServerConfig(fail_after_chunks=2)) as server:
        session = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterances[0],
            pcm=read_pcm16(wav),
            chunk=ChunkMode.parse(160),
            start_delay_s=0.0,
        )
    assert session.partials_received == 2
    assert len(session.partial_ms) > 0
    assert session.error is not None
    assert session.error.startswith("ServerError null_injected_failure")


async def test_cli_run_end_to_end(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.5)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        argv = [
            "run",
            "--endpoint",
            server.endpoint,
            "--manifest",
            str(manifest),
            "--sessions",
            "2",
            "--chunk",
            "160ms",
            "--ramp-s",
            "0.0",
            "--out",
            str(tmp_path / "cli-out"),
        ]
        rc = await asyncio.get_running_loop().run_in_executor(None, main, argv)
    assert rc == 0
    out_path = tmp_path / "cli-out" / "results.json"
    assert out_path.exists()
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["schema"] == "vb-results/1"
    assert doc["result"]["sessions_completed"] == 2
    # The usage-error path is synchronous but main() still funnels through
    # asyncio.run, which cannot nest inside this test's running loop (and cli.py
    # is out of scope to change), so call it directly on a loop-free thread
    # with no executor.
    outcome: dict[str, int] = {}

    def _usage_error() -> None:
        outcome["rc"] = main(
            [
                "run",
                "--endpoint",
                "ws://127.0.0.1:1/v1/stream",
                "--manifest",
                str(manifest),
                "--chunk",
                "100ms",
            ]
        )

    worker = threading.Thread(target=_usage_error)
    worker.start()
    worker.join()
    assert outcome["rc"] == 1


async def test_results_json_shape(tmp_path: Path) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.5)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        result = await run_load(make_spec(server.endpoint, manifest, sessions=2, ramp_s=0.0))
    out = write_results(result, tmp_path / "out")
    assert out.name == "results.json"
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "vb-results/1"
    assert doc["config"]["paced_real_time"] is True
    assert doc["config"]["corpus_id"].startswith("sha256:")
    for name in ("first_partial", "partial", "final"):
        block = doc["result"]["latency_ms"][name]
        assert block["p50"] <= block["p95"] <= block["p99"]
        assert block["n"] > 0
    assert len(doc["sessions"]) == 2
    assert_has_paths(doc, _expected_key_paths())


def test_percentile_is_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 95) == 0.0
    assert percentile([7], 99) == 7


async def test_more_sessions_than_manifest_entries_wraps(tmp_path: Path) -> None:
    wavs = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for wav in wavs:
        make_wav(wav, 0.2)
    manifest = make_manifest(tmp_path / "m.jsonl", wavs, ["e0", "e1"])
    async with NullServer(NullServerConfig()) as server:
        result = await run_load(
            make_spec(server.endpoint, manifest, sessions=5, ramp_s=0.0, seed=3)
        )
    assert [s.stream_id for s in result.sessions] == ["e0", "e1", "e0", "e1", "e0"]


def assert_has_paths(doc: dict, paths: list[str]) -> None:
    missing = [path for path in paths if _missing(doc, path.split("."))]
    assert not missing, f"missing key paths in results JSON: {missing}"


def _missing(node, parts: list[str]) -> bool:
    if not parts:
        return False
    if isinstance(node, dict):
        return parts[0] not in node or _missing(node[parts[0]], parts[1:])
    if isinstance(node, list):
        return len(node) == 0 or any(_missing(item, parts) for item in node)
    return len(parts) > 0


def _expected_key_paths() -> list[str]:
    paths = [
        "schema",
        "harness.version",
        "harness.git",
        "harness.client_floor_p95_ms",
        "arm.name",
        "arm.version",
        "arm.container_digest",
        "arm.surface",
        "checkpoint.hf_id",
        "checkpoint.revision",
        "checkpoint.file_sha256",
        "env.die",
        "env.sm",
        "env.vram_gb",
        "env.driver",
        "env.cuda",
        "env.nemo",
        "env.torch",
        "env.cpu_sets",
        "config.chunk_ms",
        "config.frame_ms",
        "config.corpus_id",
        "config.manifest",
        "config.seed",
        "config.sessions",
        "config.pacing_profile",
        "config.paced_real_time",
        "config.x_ms",
        "config.endpoint",
        "config.words",
        "result.streams",
        "result.latency_ms.first_partial.p50",
        "result.latency_ms.first_partial.p95",
        "result.latency_ms.first_partial.p99",
        "result.latency_ms.first_partial.n",
        "result.latency_ms.partial.p50",
        "result.latency_ms.partial.p95",
        "result.latency_ms.partial.p99",
        "result.latency_ms.partial.n",
        "result.latency_ms.final.p50",
        "result.latency_ms.final.p95",
        "result.latency_ms.final.p99",
        "result.latency_ms.final.n",
        "result.sessions_started",
        "result.sessions_completed",
        "result.sessions_failed",
        "result.chunks_sent",
        "result.partials_received",
        "result.finals_received",
        "result.wall_clock_s",
        "result.pacing_slip_ms.p50",
        "result.pacing_slip_ms.p95",
        "result.pacing_slip_ms.max",
        "date",
    ]
    session_keys = [
        "session_id",
        "stream_id",
        "server_session_id",
        "started_at_s",
        "chunks",
        "audio_s",
        "first_partial_ms",
        "final_ms",
        "partial_ms",
        "final_text",
        "reference_text",
        "error",
    ]
    paths += [f"sessions.{key}" for key in session_keys]
    return paths


def test_load_spec_carries_framing_and_replacement_fields(tmp_path: Path) -> None:
    spec = make_spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl")
    assert spec.frame_ms == constants.FRAME_MS
    assert spec.replace_gap_s == 0.0
    assert spec.window_s is None


def test_burst_is_an_accepted_alias_for_bursty(tmp_path: Path) -> None:
    base = make_spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=4)
    bursty = replace(base, profile="bursty")
    burst = replace(base, profile="burst")
    assert plan_start_offsets(bursty) == plan_start_offsets(replace(bursty, seed=bursty.seed))
    assert len(plan_start_offsets(burst)) == 4
    assert len(plan_start_offsets(bursty)) == 4


def test_diurnal_profile_is_declared_vocabulary(tmp_path: Path) -> None:
    spec = make_spec(
        "ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=3, profile="diurnal"
    )
    assert len(plan_start_offsets(spec)) == 3


async def test_run_load_returns_per_session_pacing_slip_samples(tmp_path: Path) -> None:
    from verbatim_bench.pace import pacing_slip_samples

    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.5)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        result = await run_load(make_spec(server.endpoint, manifest, sessions=2, ramp_s=0.0))
    slips = pacing_slip_samples(result)
    assert len(slips) == 2
    assert all(isinstance(samples, list) for samples in slips)
    assert all(len(samples) > 0 for samples in slips)


async def test_closed_loop_replacement_holds_concurrency_through_the_window(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 0.2)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    async with NullServer(NullServerConfig()) as server:
        spec = make_spec(server.endpoint, manifest, sessions=2, ramp_s=0.0)
        looped = replace(spec, window_s=0.6, replace_gap_s=0.0)
        result = await run_load(looped)
    assert len(result.sessions) > 2
