# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The polyphase resampler: rate, level, timeline and continuity across chunk boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.audio.resample import SUPPORTED_RATES, Resampler

pytestmark = pytest.mark.cpu


def _tone(rate: int, seconds: float = 1.0, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def _peak_hz(y: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(y))
    return float(np.fft.rfftfreq(len(y), 1 / rate)[int(spectrum.argmax())])


@pytest.mark.parametrize("in_rate", [r for r in SUPPORTED_RATES if r != 16000])
def test_a_tone_keeps_its_frequency_and_level(in_rate: int) -> None:
    x = _tone(in_rate)
    r = Resampler(in_rate, 16000)
    y = np.concatenate([r.process(x), r.flush()])
    assert abs(len(y) - len(x) * 16000 / in_rate) <= 1
    assert _peak_hz(y, 16000) == pytest.approx(440.0, abs=1.5)
    body = y[1600:-1600]
    assert float(np.sqrt(np.mean(body**2))) == pytest.approx(np.sqrt(0.5), rel=0.02)


@pytest.mark.parametrize("in_rate", [8000, 44100, 11025, 48000])
def test_chunk_boundaries_are_invisible_to_the_filter(in_rate: int) -> None:
    x = _tone(in_rate, seconds=0.5)
    one = Resampler(in_rate, 16000)
    whole = np.concatenate([one.process(x), one.flush()])
    chunked = Resampler(in_rate, 16000)
    cuts = np.sort(np.random.default_rng(1).integers(1, len(x), 40))
    parts = [chunked.process(part) for part in np.split(x, cuts)] + [chunked.flush()]
    assert np.allclose(np.concatenate(parts), whole, atol=1e-3)


def test_output_timeline_has_no_filter_delay() -> None:
    """Output k sits at input time k * M / L: an impulse at input sample 400 of an
    8 kHz stream lands at output sample 800 of the 16 kHz stream."""
    r = Resampler(8000, 16000)
    x = np.zeros(800, dtype=np.float32)
    x[400] = 1.0
    y = np.concatenate([r.process(x), r.flush()])
    assert int(np.argmax(np.abs(y))) == 800


def test_the_same_rate_is_a_passthrough() -> None:
    r = Resampler(16000, 16000)
    x = _tone(16000, 0.1)
    assert np.array_equal(r.process(x), x)
    assert r.flush().size == 0


def test_silence_stays_silent_and_process_after_flush_is_refused() -> None:
    r = Resampler(48000, 16000)
    assert np.all(r.process(np.zeros(4800, dtype=np.float32)) == 0)
    r.flush()
    with pytest.raises(ValueError, match="flush"):
        r.process(np.zeros(10, dtype=np.float32))


def test_empty_input_produces_nothing_and_keeps_the_state() -> None:
    r = Resampler(8000, 16000)
    assert r.process(np.zeros(0, dtype=np.float32)).size == 0
    assert len(r.process(_tone(8000, 0.1))) > 0
