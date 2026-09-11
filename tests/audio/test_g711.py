# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""G.711 expansion: the tables are the CCITT reference, checked against ``audioop``
where it still exists and against fixed reference points everywhere."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from verbatim.audio.g711 import ALAW_TO_PCM16, MULAW_TO_PCM16, decode_alaw, decode_mulaw

pytestmark = pytest.mark.cpu


def test_mulaw_reference_points() -> None:
    assert MULAW_TO_PCM16[0x00] == -32124
    assert MULAW_TO_PCM16[0x80] == 32124
    assert MULAW_TO_PCM16[0xFF] == 0
    assert MULAW_TO_PCM16[0x7F] == 0


def test_alaw_reference_points() -> None:
    assert ALAW_TO_PCM16[0x55] == -8
    assert ALAW_TO_PCM16[0xD5] == 8
    assert ALAW_TO_PCM16[0x2A] == -32256
    assert ALAW_TO_PCM16[0xAA] == 32256


def test_tables_are_odd_symmetric_in_the_sign_bit() -> None:
    codes = np.arange(256, dtype=np.uint8)
    assert np.array_equal(MULAW_TO_PCM16[codes ^ 0x80], -MULAW_TO_PCM16[codes].astype(np.int32))
    assert np.array_equal(ALAW_TO_PCM16[codes ^ 0x80], -ALAW_TO_PCM16[codes].astype(np.int32))


def test_magnitude_grows_with_the_code_within_a_sign() -> None:
    # mu-law bytes 0x00..0x7F are the negative half, most negative first.
    negative = MULAW_TO_PCM16[np.arange(0x00, 0x80)].astype(np.int32)
    assert negative[0] == -32124 and negative[-1] == 0
    assert np.all(np.diff(negative) >= 0)
    # A-law toggles the even bits (0x55); undoing that, codes 0..0x7F are ascending
    # magnitudes of the negative half.
    codes = (np.arange(0x00, 0x80) ^ 0x55).astype(np.uint8)
    magnitudes = -ALAW_TO_PCM16[codes].astype(np.int32)
    assert magnitudes[0] == 8 and magnitudes[-1] == 32256
    assert np.all(np.diff(magnitudes) >= 0)


def test_tables_match_audioop_where_it_still_exists() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        audioop = pytest.importorskip("audioop")  # removed in 3.13; the points above still guard
    codes = bytes(range(256))
    assert np.array_equal(MULAW_TO_PCM16, np.frombuffer(audioop.ulaw2lin(codes, 2), dtype="<i2"))
    assert np.array_equal(ALAW_TO_PCM16, np.frombuffer(audioop.alaw2lin(codes, 2), dtype="<i2"))


def test_decode_is_one_int16_sample_per_byte_by_lookup() -> None:
    payload = bytes([0x00, 0xFF, 0x7F, 0x80, 0x55])
    out = decode_mulaw(payload)
    assert out.dtype == np.int16
    assert out.tolist() == [-32124, 0, 0, 32124, int(MULAW_TO_PCM16[0x55])]
    assert decode_alaw(b"\x55\xd5").tolist() == [-8, 8]
    assert decode_mulaw(b"").shape == (0,)
