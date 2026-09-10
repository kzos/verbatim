# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Batch-invariance property tests against the CPU fake pipeline.

It does NOT prove the real system is batch-invariant -- a fake pipeline cannot --
it proves the *scheduler* does not introduce a dependence on batch composition,
which is the part that is testable without a GPU.
"""

from __future__ import annotations

import random

import numpy as np

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.core.types import PcmFrame, StepResult
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.scheduler.buckets import BucketScheduler
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.tick import TickLoop

CHUNK = ChunkMode(160)
N = CHUNK.samples


def _loop(bucket: int = 8, edge_batch: int = 8) -> tuple[TickLoop, FakePipelineAdapter]:
    config = EngineConfig(
        chunk=CHUNK,
        buckets=(bucket,),
        edge_batch=edge_batch,
        calibrated_ceiling=bucket,
        drain_margin=bucket,
    )
    pipeline = FakePipelineAdapter(CHUNK, buckets=(bucket,))
    return TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock()), pipeline


def _seeded_audio(seed: int, chunks: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, size=chunks * N).astype(np.float32)


def _join(
    loop: TickLoop, session_id: int, audio: np.ndarray, *, ring_seconds: float = 8.0
) -> Session:
    session = Session(session_id, CHUNK, ring_seconds=ring_seconds)
    session.configure()
    decision = loop.admit_session(session)
    assert decision.admitted
    assert session.ring.write(audio) == len(audio)
    session.begin_draining()
    return session


def _for_stream(results: list[StepResult], stream_id: int) -> list[StepResult]:
    return [r for r in results if r.stream_id == stream_id]


def _without_tick(results: list[StepResult]) -> list[tuple]:
    return [(r.partial_text, r.final_text, r.audio_processed_s, r.eager) for r in results]


def test_a_sessions_transcript_is_identical_at_every_concurrency() -> None:
    tracked = _seeded_audio(7, 10)
    alone_loop, _ = _loop(bucket=32)
    _join(alone_loop, 1, tracked)
    alone = _for_stream(alone_loop.run_for(12), 1)

    busy_loop, _ = _loop(bucket=32)
    _join(busy_loop, 1, tracked)
    for sid in range(2, 33):
        _join(busy_loop, sid, _seeded_audio(1000 + sid, 10))
    busy = _for_stream(busy_loop.run_for(12), 1)

    other_loop, _ = _loop(bucket=32)
    _join(other_loop, 1, tracked)
    for sid in range(2, 33):
        _join(other_loop, sid, _seeded_audio(2000 + sid, 10))
    others = _for_stream(other_loop.run_for(12), 1)

    assert alone == busy == others


def test_transcript_is_independent_of_row_position() -> None:
    audio = {sid: _seeded_audio(seed, 4) for sid, seed in [(1, 21), (2, 22), (3, 23)]}
    tracked_id = 2

    def _run(order: list[int]) -> list[tuple]:
        config = EngineConfig(chunk=CHUNK, buckets=(8,), calibrated_ceiling=8)
        pipeline = FakePipelineAdapter(CHUNK, buckets=(8,))
        loop = TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock())
        for sid in order:
            session = Session(sid, CHUNK, ring_seconds=8.0)
            session.configure()
            assert loop.admit_session(session).admitted
            assert session.ring.write(audio[sid]) == len(audio[sid])
            session.begin_draining()
        return _without_tick(_for_stream(loop.run_for(6), tracked_id))

    assert _run([1, 2, 3]) == _run([3, 1, 2])

    # The permutation must reach the scheduler: live rows leave plan() ascending
    # no matter what order the frames arrived in.
    scheduler = BucketScheduler(EngineConfig(chunk=CHUNK, buckets=(8,), calibrated_ceiling=8))

    def _frames(order: list[int]) -> list[PcmFrame]:
        return [
            PcmFrame(
                stream_id=sid,
                samples=audio[sid][:N],
                is_first=True,
                is_last=False,
                valid_samples=N,
            )
            for sid in order
        ]

    for order in ([1, 2, 3], [3, 1, 2]):
        plan = scheduler.plan(_frames(order), [])
        assert [f.stream_id for f in plan.steady if not f.is_pad] == [1, 2, 3]


def test_transcript_is_independent_of_join_phase() -> None:
    tracked = _seeded_audio(31, 8)
    early_loop, _ = _loop()
    _join(early_loop, 1, tracked)
    _join(early_loop, 2, _seeded_audio(32, 8))
    early_results = _for_stream(early_loop.run_for(10), 1)
    early = _without_tick(early_results)

    late_loop, _ = _loop()
    _join(late_loop, 9, _seeded_audio(33, 12))
    late_loop.run_for(7)
    _join(late_loop, 1, tracked)
    _join(late_loop, 3, _seeded_audio(34, 8))
    late_results = _for_stream(late_loop.run_for(10), 1)
    late = _without_tick(late_results)
    assert early == late
    # The same frames, only later: every tick id shifted by the join delay.
    assert [r.tick_id for r in late_results] == [t + 7 for t in [r.tick_id for r in early_results]]


def test_transcript_is_independent_of_a_starved_tick() -> None:
    audio = _seeded_audio(41, 10)
    ref_loop, _ = _loop()
    _join(ref_loop, 1, audio)
    ref = _without_tick(_for_stream(ref_loop.run_for(12), 1))

    gap_loop, _ = _loop()
    session = Session(1, CHUNK, ring_seconds=8.0)
    session.configure()
    assert gap_loop.admit_session(session).admitted
    chunks = [audio[i * N : (i + 1) * N] for i in range(10)]
    tick = 0
    fed = 0
    gapped_results: list[StepResult] = []
    while fed < 10:
        if tick not in (4, 5, 6):
            assert session.ring.write(chunks[fed]) == N
            fed += 1
        gapped_results.extend(gap_loop.run_tick())
        tick += 1
    session.begin_draining()
    gapped_results.extend(gap_loop.run_for(3))
    gapped = _without_tick(_for_stream(gapped_results, 1))
    assert gapped == ref
    assert any(stat.starved >= 1 for stat in gap_loop.stats)


def test_a_short_final_chunk_matches_a_padded_one() -> None:
    head = _seeded_audio(51, 3)
    tail = _seeded_audio(52, 1)[:100]
    short_loop, _ = _loop()
    _join(short_loop, 1, np.concatenate([head, tail]))
    short = _for_stream(short_loop.run_for(6), 1)

    padded_loop, _ = _loop()
    padding = np.zeros(N - 100, dtype=np.float32)
    _join(padded_loop, 1, np.concatenate([head, tail, padding]))
    padded = _for_stream(padded_loop.run_for(7), 1)

    short_partials = [r.partial_text for r in short]
    padded_partials = [r.partial_text for r in padded]
    assert short_partials == padded_partials[: len(short_partials)]
    assert short[-1].final_text == padded[-1].final_text


def test_property_over_seeded_configurations() -> None:
    for seed in range(1000, 1020):
        rng = random.Random(seed)
        bucket = rng.choice([4, 8, 16])
        edge_batch = rng.choice([2, 4, 8])
        n_sessions = rng.randint(2, bucket)
        join_at = sorted(rng.randint(0, 6) for _ in range(n_sessions))
        lengths = [rng.randint(3, 12) for _ in range(n_sessions)]
        tracked_idx = rng.randrange(n_sessions)
        horizon = max(j + ln for j, ln in zip(join_at, lengths, strict=True)) + 3
        audios = [_seeded_audio(seed * 7919 + i, lengths[i]) for i in range(n_sessions)]

        loop, _ = _loop(bucket=bucket, edge_batch=edge_batch)
        results: list[StepResult] = []
        for tick in range(horizon):
            for i in range(n_sessions):
                if join_at[i] == tick:
                    _join(loop, i + 1, audios[i])
            results.extend(loop.run_tick())
        full_tracked = _without_tick(_for_stream(results, tracked_idx + 1))

        solo_loop, _ = _loop(bucket=bucket, edge_batch=edge_batch)
        _join(solo_loop, 1, audios[tracked_idx])
        solo = _without_tick(_for_stream(solo_loop.run_for(horizon), 1))
        assert full_tracked == solo, f"batch dependence at seed={seed}"
