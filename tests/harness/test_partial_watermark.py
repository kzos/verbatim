# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Watermark matching tests for per-chunk partial latency."""

from __future__ import annotations

from pathlib import Path

import pytest
from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.results import percentile

pytestmark = pytest.mark.cpu


def _utterance() -> Utterance:
    return Utterance(
        stream_id="watermark-0",
        audio_path=Path("watermark.wav"),
        duration_s=1.6,
        text="watermark test",
    )


async def _run_null_session(partial_delay_ms: float):
    async with NullServer(NullServerConfig(partial_delay_ms=partial_delay_ms)) as server:
        return await run_session(
            server.endpoint,
            session_id="watermark-session",
            utterance=_utterance(),
            pcm=bytes(51200),
            chunk=ChunkMode.parse("80ms"),
            start_delay_s=0.0,
        )


def test_watermark_matcher_pairs_a_chunk_with_the_partial_that_covers_it() -> None:
    """Watermarks preserve honest backlog latency; arrival-order matching would return
    [160, 80, 0, 80, 0] for the overload vector instead of
    [160.0, 240.0, 320.0, 400.0, 480.0].
    """
    from verbatim_bench.client import match_partials_by_watermark

    send_times = [0.0, 0.08, 0.16, 0.24]
    sent_audio_s = [0.08, 0.16, 0.24, 0.32]
    one_per_chunk = [(0.01, 0.08), (0.09, 0.16), (0.17, 0.24), (0.25, 0.32)]
    assert match_partials_by_watermark(
        send_times,
        sent_audio_s,
        one_per_chunk,
    ) == pytest.approx([10.0, 10.0, 10.0, 10.0])
    assert match_partials_by_watermark(
        send_times,
        sent_audio_s,
        [(0.01, 0.08), (0.30, 0.32)],
    ) == pytest.approx([10.0, 220.0, 140.0, 60.0])
    assert match_partials_by_watermark(
        send_times,
        sent_audio_s,
        [(0.005, 0.04), *one_per_chunk],
    ) == pytest.approx([10.0] * 4)
    assert match_partials_by_watermark(
        send_times,
        sent_audio_s,
        [(0.005, None), *one_per_chunk],
    ) == pytest.approx([10.0] * 4)
    stopped = match_partials_by_watermark(
        send_times,
        sent_audio_s,
        [(0.01, 0.08), (0.09, 0.16)],
    )
    assert stopped == pytest.approx([10.0, 10.0])
    assert len(stopped) < len(send_times)

    b = 0.1234565
    assert match_partials_by_watermark([0.0], [b], [(0.01, round(b, 6))]) == pytest.approx([10.0])
    assert match_partials_by_watermark([0.0], [b], [(0.01, b - 1 / 16000)]) == []
    assert match_partials_by_watermark([1.0], [0.08], [(0.5, 0.08)]) == [0.0]

    overload = match_partials_by_watermark(
        [0.0, 0.08, 0.16, 0.24, 0.32],
        [0.08, 0.16, 0.24, 0.32, 0.40],
        [(0.16, 0.08), (0.32, 0.16), (0.48, 0.24), (0.64, 0.32), (0.80, 0.40)],
    )
    assert overload == pytest.approx([160.0, 240.0, 320.0, 400.0, 480.0])


async def test_partial_p95_reflects_the_backlog_when_the_server_falls_behind() -> None:
    session = await _run_null_session(160.0)
    p95 = percentile(session.partial_ms, 95)
    final_ms = session.final_ms if session.final_ms is not None else float("nan")
    observed = f"p95={p95:.1f} final_ms={final_ms:.1f}"

    assert session.error is None, observed
    assert session.chunks == 20, observed
    assert session.partials_received == 20, observed
    assert len(session.partial_ms) == 20, observed
    assert session.final_ms is not None and session.final_ms > 1000.0, observed
    assert p95 > 800.0, observed
    assert p95 > 0.5 * final_ms, observed
    assert session.partial_ms[-1] > 1000.0, observed


async def test_partial_p95_is_unchanged_when_the_server_keeps_up() -> None:
    session = await _run_null_session(40.0)

    assert session.error is None
    assert session.chunks == session.partials_received == len(session.partial_ms) == 20
    assert all(30.0 <= value < 80.0 for value in session.partial_ms)
