# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``MULAW`` / ``ALAW`` via a 256-entry int16 lookup table. No dependency.

``audioop`` was removed from the standard library in 3.13 (PEP 594) and its PyPI
replacement requires >= 3.13, so depending on it would split the dependency graph
across the supported Python range in order to obtain a table lookup. Telephony
operators are a named constituency; G.711 must work on every supported Python.

The two tables are built once, at import, from the CCITT G.711 expansion formulas
in their reference form (Sun's ``g711.c``, the same code ``audioop`` carried), so a
byte maps to exactly the 16-bit sample ``audioop.ulaw2lin`` / ``alaw2lin`` gave.
Decoding a message is one fancy-index into the table: no loop, no branch, and a
cost that does not depend on the audio.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = ["ALAW_TO_PCM16", "MULAW_TO_PCM16", "decode_alaw", "decode_mulaw"]

_BIAS: Final = 0x84
_QUANT_MASK: Final = 0x0F
_SEG_MASK: Final = 0x70
_SEG_SHIFT: Final = 4
_SIGN_BIT: Final = 0x80


def _mulaw_to_linear(byte: int) -> int:
    """One mu-law byte to its 16-bit linear sample (CCITT G.711, reference form)."""
    u = ~byte & 0xFF
    t = ((u & _QUANT_MASK) << 3) + _BIAS
    t <<= (u & _SEG_MASK) >> _SEG_SHIFT
    return (_BIAS - t) if (u & _SIGN_BIT) else (t - _BIAS)


def _alaw_to_linear(byte: int) -> int:
    """One A-law byte to its 16-bit linear sample (CCITT G.711, reference form)."""
    a = byte ^ 0x55
    t = (a & _QUANT_MASK) << 4
    seg = (a & _SEG_MASK) >> _SEG_SHIFT
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (a & _SIGN_BIT) else -t


#: mu-law byte -> int16 sample. Byte 0xFF is zero; 0x00 is the most negative (-32124).
MULAW_TO_PCM16: Final = np.array([_mulaw_to_linear(b) for b in range(256)], dtype=np.int16)
#: A-law byte -> int16 sample. Byte 0x55 is -8, the smallest magnitude; 0xD5 is +8.
ALAW_TO_PCM16: Final = np.array([_alaw_to_linear(b) for b in range(256)], dtype=np.int16)
MULAW_TO_PCM16.setflags(write=False)
ALAW_TO_PCM16.setflags(write=False)


def decode_mulaw(payload: bytes) -> np.ndarray:
    """``MULAW`` bytes -> int16 samples, one sample per byte, by table lookup."""
    return MULAW_TO_PCM16[np.frombuffer(payload, dtype=np.uint8)]


def decode_alaw(payload: bytes) -> np.ndarray:
    """``ALAW`` bytes -> int16 samples, one sample per byte, by table lookup."""
    return ALAW_TO_PCM16[np.frombuffer(payload, dtype=np.uint8)]
