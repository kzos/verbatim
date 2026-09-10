# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``LINEAR_PCM`` decode: PCM16LE mono bytes to float32 in ``[-1, 1]``.

Decoding happens per message, before the ring, so a message with an odd byte
length is split across two conversions with a one-byte carry: without it an
odd-length message would drop or misalign a sample, making chunk boundaries a
function of packetisation.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This is plain sample arithmetic with no model behind it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decode_pcm16"]


def decode_pcm16(payload: bytes, carry: bytes = b"") -> tuple[np.ndarray, bytes]:
    """PCM16LE mono -> float32 in [-1, 1], scaled by 1/32768.

    `carry` is the odd trailing byte left over from the previous message; it is prefixed
    here. Returns the samples and the new carry (b"" or one byte). Without the carry an
    odd-length message split across two conversions would drop or misalign a sample, which
    would make chunk boundaries a function of packetisation -- the exact dependence this
    task exists to remove.
    """
    combined = carry + bytes(payload)
    complete = (len(combined) // 2) * 2
    raw = np.frombuffer(combined[:complete], dtype="<i2")
    samples = raw.astype(np.float32) * (1.0 / 32768.0)
    return samples, combined[complete:]
