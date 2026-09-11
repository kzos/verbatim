# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The measurement window: the ramp, the warm-up that opens it, and what it contains.

Everything here runs against the null server. No GPU, no model, and every duration is
injected, so the suite exercises a three-phase rung without spending the frozen four
minutes on one.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest
from test_pace import make_manifest, make_spec, make_wav
from verbatim_bench import constants
from verbatim_bench.client import (
    WatermarkMatcher,
    match_partials_by_watermark,
    match_partials_with_recv,
)
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.pace import (
    executed_canonical_window,
    run_load,
    window_partial_samples,
)
from verbatim_bench.results import RunResult

pytestmark = pytest.mark.cpu

#: A warm-up and window short enough for a unit test, with the same shape as the frozen
#: one: readings long enough to hold several chunks, a floor, and a cap well above it.
FAST_WINDOW: dict[str, float] = {
    "ramp_s": 0.1,
    "warm_up_s": 0.5,
    "warm_up_reading_s": 0.5,
    "warm_up_cap_s": 6.0,
    "window_s": 1.0,
}


def _all_samples(result: RunResult) -> list[tuple[float, float]]:
    return [
        (latency, recv)
        for session in result.sessions
        for latency, recv in zip(session.partial_ms, session.partial_recv_s, strict=True)
    ]


async def _windowed_run(tmp_path: Path, **overrides: object) -> RunResult:
    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    config = overrides.pop("config", NullServerConfig(partial_delay_ms=20.0))
    assert isinstance(config, NullServerConfig)
    async with NullServer(config) as server:
        spec = make_spec(server.endpoint, manifest, sessions=2, **FAST_WINDOW)  # type: ignore[arg-type]
        return await run_load(replace(spec, **overrides))  # type: ignore[arg-type]


def test_the_streaming_matcher_reproduces_the_batch_match() -> None:
    """One definition of the watermark match, reachable two ways.

    The load generator reads latency while sessions are still open, which the batch
    function cannot do. If the streaming form disagreed with it, the number a window
    reports would not be the number the methodology defines.
    """
    rng = random.Random(20260911)
    for _ in range(50):
        chunks = rng.randint(1, 12)
        send_times = [round(0.08 * i, 6) for i in range(chunks)]
        sent_audio_s = [round(0.08 * (i + 1), 6) for i in range(chunks)]
        partials: list[tuple[float, float | None]] = []
        for _ in range(rng.randint(0, 16)):
            watermark = rng.choice([None, round(rng.random() * 0.08 * chunks, 6)])
            partials.append((round(rng.random() * 2.0, 6), watermark))
        partials.sort(key=lambda item: item[0])
        streamed = match_partials_with_recv(send_times, sent_audio_s, partials)
        assert [match.latency_ms for match in streamed] == match_partials_by_watermark(
            send_times, sent_audio_s, partials
        )
        # A streamed sample carries the moment it was matched, which is what lets the
        # generator decide which phase it belongs to.
        assert all(match.recv_s in {t for t, _ in partials} for match in streamed)


def test_a_partial_cannot_release_a_chunk_that_has_not_been_offered() -> None:
    matcher = WatermarkMatcher()
    assert matcher.offer_partial(1.0, 10.0) == []
    matcher.offer_chunk(0.0, 0.08)
    released = matcher.offer_partial(1.0, 10.0)
    assert [match.latency_ms for match in released] == pytest.approx([1000.0])
    assert matcher.offer_partial(2.0, 10.0) == []


async def test_the_window_opens_after_the_ramp_and_the_warm_up(tmp_path: Path) -> None:
    result = await _windowed_run(tmp_path)
    assert result.warm_up_converged is True
    assert result.all_live_at_s is not None
    assert result.window_open_s is not None and result.window_close_s is not None
    # All streams live, then the warm-up at N, then the window. In that order.
    assert result.wall_start_s <= result.all_live_at_s <= result.window_open_s
    assert result.window_open_s < result.window_close_s
    warm_up = result.warm_up_length_s
    window = result.window_length_s
    assert warm_up is not None and window is not None
    assert warm_up >= FAST_WINDOW["warm_up_s"]
    assert warm_up <= FAST_WINDOW["warm_up_cap_s"]
    assert window >= FAST_WINDOW["window_s"]
    # The run is longer than the window it ran, because the window is not the whole run.
    assert result.wall_clock_s >= warm_up + window
    # Two consecutive readings agreed, which is what opened it.
    numeric = [reading for reading in result.warm_up_readings if reading is not None]
    assert len(numeric) >= 2


async def test_samples_taken_before_the_window_opens_are_not_in_it(tmp_path: Path) -> None:
    result = await _windowed_run(tmp_path)
    assert result.window_open_s is not None and result.window_close_s is not None
    every = _all_samples(result)
    warm_up_samples = [recv for _, recv in every if recv < result.window_open_s]
    assert warm_up_samples, "the warm-up produced no samples, so nothing is being excluded"
    windowed = window_partial_samples(result)
    inside = [
        latency for latency, recv in every if result.window_open_s <= recv <= result.window_close_s
    ]
    assert sorted(windowed) == sorted(inside)
    assert len(windowed) < len(every)


async def test_closed_loop_replacement_runs_the_whole_window(tmp_path: Path) -> None:
    """The window is a duration of load, not one pass of the corpus down each slot."""
    result = await _windowed_run(tmp_path)
    # Two slots, one-second sessions, a warm-up and a one-second window: a generator that
    # ran each slot once would stop at two sessions and about a second of wall clock.
    assert len(result.sessions) > 2
    assert result.wall_clock_s > 2.0


async def test_a_warm_up_that_never_converges_opens_no_window(tmp_path: Path) -> None:
    result = await _windowed_run(
        tmp_path,
        config=NullServerConfig(partial_delay_ms=20.0, partial_delay_growth_ms_per_s=300.0),
        warm_up_cap_s=2.0,
    )
    assert result.warm_up_converged is False
    assert result.window_open_s is None and result.window_close_s is None
    assert window_partial_samples(result) == []
    assert result.warm_up_length_s is not None
    # It gave up at the cap rather than running on.
    assert result.warm_up_length_s <= 2.0 + FAST_WINDOW["warm_up_reading_s"]
    assert _all_samples(result), "the server did answer; it simply never settled"


def _frozen_spec_dict() -> dict[str, object]:
    """The spec a rung built from untouched command-line defaults asks for."""
    return {
        "window_s": float(constants.WINDOW_S),
        "warm_up_s": float(constants.WARM_UP_S),
        "warm_up_reading_s": float(constants.WARM_UP_READING_S),
        "warm_up_convergence": float(constants.WARM_UP_CONVERGENCE),
        "warm_up_cap_s": float(constants.WARM_UP_CAP_S),
    }


def _executed(**overrides: object) -> RunResult:
    open_s = 1000.0 + constants.WARM_UP_S
    fields: dict[str, object] = {
        "spec_dict": _frozen_spec_dict(),
        "all_live_at_s": 1000.0,
        "warm_up_end_s": open_s,
        "window_open_s": open_s,
        "window_close_s": open_s + constants.WINDOW_S,
        "warm_up_converged": True,
    }
    fields.update(overrides)
    return RunResult(**fields)  # type: ignore[arg-type]


def test_canonical_window_is_false_when_no_window_ran_despite_frozen_arguments() -> None:
    """The flag this replaces could not fail; this one fails on exactly that run.

    A rung built from untouched defaults that then ran no window at all is the run every
    ladder on 2026-09-11 produced, and every one of them was stamped canonical.
    """
    asked_for_the_frozen_window = _executed(
        all_live_at_s=None,
        warm_up_end_s=None,
        window_open_s=None,
        window_close_s=None,
        warm_up_converged=None,
    )
    assert asked_for_the_frozen_window.spec_dict == _frozen_spec_dict()
    assert executed_canonical_window(asked_for_the_frozen_window) is False


def test_canonical_window_is_true_only_for_a_window_that_ran_to_length() -> None:
    assert executed_canonical_window(_executed()) is True
    short_window = _executed(window_close_s=1000.0 + constants.WARM_UP_S + constants.WINDOW_S - 1)
    assert executed_canonical_window(short_window) is False
    short_warm_up = _executed(all_live_at_s=1000.0 + 1)
    assert executed_canonical_window(short_warm_up) is False
    unsettled = _executed(warm_up_converged=False)
    assert executed_canonical_window(unsettled) is False


def test_canonical_window_is_false_when_a_protocol_duration_was_overridden() -> None:
    for name in (
        "window_s",
        "warm_up_s",
        "warm_up_reading_s",
        "warm_up_convergence",
        "warm_up_cap_s",
    ):
        spec = _frozen_spec_dict()
        spec[name] = float(spec[name]) / 2.0  # type: ignore[arg-type]
        assert executed_canonical_window(_executed(spec_dict=spec)) is False, name


def test_a_warm_up_that_ran_past_the_cap_is_not_canonical() -> None:
    late = _executed(all_live_at_s=1000.0 + constants.WARM_UP_S - constants.WARM_UP_CAP_S - 1)
    assert executed_canonical_window(late) is False
