# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for the two surviving latency definitions."""

from __future__ import annotations

from pathlib import Path

import pytest
from verbatim_bench import constants
from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.latency import (
    ChunkSend,
    PartialEvent,
    ReferenceWord,
    final_latency_ms,
    first_partial_latency_ms,
    retraction_rate,
    unmatched_chunks,
    watermark_latencies,
    word_emission_latencies,
)
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.results import percentile

pytestmark = pytest.mark.cpu


def _sends(n: int, period_s: float = 0.16) -> list[ChunkSend]:
    return [
        ChunkSend(index=i, t_send_s=i * period_s, audio_end_s=(i + 1) * period_s) for i in range(n)
    ]


def test_watermark_latency_matches_the_first_sufficient_watermark() -> None:
    sends = _sends(3)
    partials = [
        PartialEvent(t_recv_s=0.05, watermark_s=0.16, text="a"),
        PartialEvent(t_recv_s=0.25, watermark_s=0.32, text="a b"),
        PartialEvent(t_recv_s=0.40, watermark_s=0.48, text="a b c"),
    ]
    samples = watermark_latencies(sends, partials)
    assert samples == pytest.approx([50.0, 90.0, 80.0])


def test_watermark_latency_ignores_an_older_chunks_partial() -> None:
    sends = _sends(2)
    partials = [
        PartialEvent(t_recv_s=0.20, watermark_s=0.08, text="old"),
        PartialEvent(t_recv_s=0.30, watermark_s=0.32, text="new"),
    ]
    samples = watermark_latencies(sends, partials)
    assert len(samples) == 2
    assert samples[0] == pytest.approx((0.30 - 0.0) * 1000.0)
    assert samples[1] == pytest.approx((0.30 - 0.16) * 1000.0)


def test_watermark_latency_grows_with_the_backlog() -> None:
    sends = _sends(3)
    one_behind = [
        PartialEvent(t_recv_s=0.32, watermark_s=0.16, text="a"),
        PartialEvent(t_recv_s=0.48, watermark_s=0.32, text="a b"),
        PartialEvent(t_recv_s=0.64, watermark_s=0.48, text="a b c"),
    ]
    two_behind = [
        PartialEvent(t_recv_s=0.48, watermark_s=0.16, text="a"),
        PartialEvent(t_recv_s=0.64, watermark_s=0.32, text="a b"),
        PartialEvent(t_recv_s=0.80, watermark_s=0.48, text="a b c"),
    ]
    three_behind = [
        PartialEvent(t_recv_s=0.64, watermark_s=0.16, text="a"),
        PartialEvent(t_recv_s=0.80, watermark_s=0.32, text="a b"),
        PartialEvent(t_recv_s=0.96, watermark_s=0.48, text="a b c"),
    ]
    first = watermark_latencies(sends, one_behind)[0]
    second = watermark_latencies(sends, two_behind)[0]
    third = watermark_latencies(sends, three_behind)[0]
    assert first < second < third


def test_unmatched_chunks_are_counted_not_dropped() -> None:
    sends = _sends(4)
    partials = [
        PartialEvent(t_recv_s=0.05, watermark_s=0.16, text="a"),
        PartialEvent(t_recv_s=0.25, watermark_s=0.32, text="a b"),
    ]
    assert unmatched_chunks(sends, partials) == 2
    assert len(watermark_latencies(sends, partials)) == 2


def test_word_emission_latency_uses_first_stable_appearance() -> None:
    words = [
        ReferenceWord(position=0, word="hello", audio_end_s=0.10),
        ReferenceWord(position=1, word="world", audio_end_s=0.20),
    ]
    partials = [
        PartialEvent(t_recv_s=1.00, watermark_s=None, text="hello"),
        PartialEvent(t_recv_s=1.10, watermark_s=None, text="hello world"),
        PartialEvent(t_recv_s=1.20, watermark_s=None, text="hello world"),
    ]
    samples = word_emission_latencies(words, partials, t_audio_start_s=0.0)
    assert samples == pytest.approx([900.0, 900.0])


def test_a_retracted_word_does_not_count_as_appearing() -> None:
    words = [ReferenceWord(position=1, word="world", audio_end_s=0.20)]
    partials = [
        PartialEvent(t_recv_s=1.00, watermark_s=None, text="hello world"),
        PartialEvent(t_recv_s=1.10, watermark_s=None, text="hello"),
        PartialEvent(t_recv_s=1.20, watermark_s=None, text="hello there"),
    ]
    assert word_emission_latencies(words, partials, t_audio_start_s=0.0) == []


def test_word_emission_latency_is_unbounded_under_overload() -> None:
    words = [ReferenceWord(position=0, word="hello", audio_end_s=0.10)]
    early = [PartialEvent(t_recv_s=1.00, watermark_s=None, text="hello")]
    late = [PartialEvent(t_recv_s=10.00, watermark_s=None, text="hello")]
    early_sample = word_emission_latencies(words, early, t_audio_start_s=0.0)[0]
    late_sample = word_emission_latencies(words, late, t_audio_start_s=0.0)[0]
    assert late_sample > early_sample
    assert late_sample == pytest.approx(9900.0)


def test_retraction_rate_is_per_hundred_final_words() -> None:
    partials = [
        PartialEvent(t_recv_s=0.1, watermark_s=None, text="hello world"),
        PartialEvent(t_recv_s=0.2, watermark_s=None, text="hello"),
        PartialEvent(t_recv_s=0.3, watermark_s=None, text="hello there"),
    ]
    rate = retraction_rate(partials, "hello there")
    assert rate == pytest.approx(50.0)
    assert retraction_rate(partials, "") == 0.0


def test_percentiles_come_from_the_results_module() -> None:
    source = Path(__file__).resolve().parents[2]
    latency_path = source / "bench" / "src" / "verbatim_bench" / "latency.py"
    text = latency_path.read_text(encoding="utf-8")
    assert "def percentile" not in text
    assert "results" in text or "percentile" not in text
    assert percentile([1, 2, 3, 4], 95) == 4


async def test_client_against_falling_behind_null_server_reports_growing_p95() -> None:
    pcm = bytes(int(3.0 * 16000 * 2))
    utterance = Utterance(
        stream_id="falling-behind",
        audio_path=Path("falling.wav"),
        duration_s=3.0,
        text="falling behind",
    )
    async with NullServer(NullServerConfig(partial_delay_ms=320)) as server:
        session = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterance,
            pcm=pcm,
            chunk=ChunkMode.parse("160ms"),
            start_delay_s=0.0,
        )
    assert session.error is None
    p95 = percentile(session.partial_ms, 95)
    assert p95 > 160 + constants.X_MS, f"p95={p95:.1f} did not grow past the threshold"


async def test_client_against_a_healthy_null_server_reports_below_one_period() -> None:
    pcm = bytes(int(1.0 * 16000 * 2))
    utterance = Utterance(
        stream_id="healthy",
        audio_path=Path("healthy.wav"),
        duration_s=1.0,
        text="healthy",
    )
    async with NullServer(NullServerConfig()) as server:
        session = await run_session(
            server.endpoint,
            session_id="s0000",
            utterance=utterance,
            pcm=pcm,
            chunk=ChunkMode.parse("160ms"),
            start_delay_s=0.0,
        )
    assert session.error is None
    assert percentile(session.partial_ms, 95) < 160.0


def test_no_partial_ms_sample_is_produced_from_an_unmatched_chunk() -> None:
    sends = _sends(3)
    partials = [PartialEvent(t_recv_s=0.05, watermark_s=0.16, text="a")]
    samples = watermark_latencies(sends, partials)
    assert len(samples) == 1
    assert unmatched_chunks(sends, partials) == 2
    assert first_partial_latency_ms(partials, t_first_sample_s=0.0) == pytest.approx(50.0)
    assert final_latency_ms(t_recv_final_s=1.0, t_last_sample_s=0.32) == pytest.approx(680.0)
