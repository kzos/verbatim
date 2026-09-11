# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The per-session wire decoder: whatever the wire carries, the ring gets PCM16LE at 16 kHz."""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.audio.decoder import WireDecoder, wire_decoder

pytestmark = pytest.mark.cpu


def test_the_default_path_has_no_decoder() -> None:
    assert wire_decoder("LINEAR_PCM", 16000) is None
    assert wire_decoder("MULAW", 16000) is not None
    assert wire_decoder("LINEAR_PCM", 8000) is not None


def test_mulaw_at_16k_is_the_table_lookup_with_no_resample() -> None:
    d = WireDecoder("MULAW", 16000)
    out = np.frombuffer(d.decode(bytes([0x00, 0xFF, 0x80])), dtype="<i2")
    assert out.tolist() == [-32124, 0, 32124]
    assert d.flush() == b""


def test_mulaw_at_8k_doubles_the_sample_count_across_messages() -> None:
    d = WireDecoder("MULAW", 8000)
    total = 0
    for size in (1, 7, 100, 333, 2000, 5559):  # 8000 bytes: one second
        total += len(d.decode(b"\xff" * size))
    total += len(d.flush())
    assert abs(total // 2 - 16000) <= 1


def test_linear_pcm_at_8k_carries_an_odd_byte_and_matches_the_one_shot_decode() -> None:
    tone = (np.sin(np.linspace(0, 20 * np.pi, 800)) * 20000).astype("<i2").tobytes()  # 0.1 s
    one = WireDecoder("LINEAR_PCM", 8000)
    reference = np.frombuffer(one.decode(tone) + one.flush(), dtype="<i2")
    chunked = WireDecoder("LINEAR_PCM", 8000)
    pieces = [chunked.decode(tone[i : i + 333]) for i in range(0, len(tone), 333)]
    out = np.frombuffer(b"".join(pieces) + chunked.flush(), dtype="<i2")
    assert len(reference) == 1600
    assert np.allclose(out.astype(np.int32), reference.astype(np.int32), atol=8)


def test_alaw_at_48k_downsamples() -> None:
    d = WireDecoder("ALAW", 48000)
    out = d.decode(b"\x55" * 4800) + d.flush()  # 0.1 s
    assert abs(len(out) // 2 - 1600) <= 1


def test_unknown_encoding_and_unserved_rate_are_refused() -> None:
    with pytest.raises(ValueError, match="encoding"):
        WireDecoder("FLAC", 16000)
    with pytest.raises(ValueError, match="rate"):
        WireDecoder("MULAW", 8001)
