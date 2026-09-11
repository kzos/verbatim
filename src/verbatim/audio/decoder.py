# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""One session's wire audio to the PCM16LE 16 kHz bytes the session takes.

The engine and its ring know exactly one format: PCM16LE mono at 16 kHz, the
format both target plugins hard-code. A session whose wire carries G.711 or
another rate gets one ``WireDecoder`` in its transport, built at open and called
from a worker thread on every message, so the ring only ever sees the one
format and the event loop never runs a resample. The default path has no
decoder at all (``wire_decoder`` returns None) and costs one buffer view.

A decoder is stateful and belongs to exactly one session: it carries an odd
trailing byte of LINEAR_PCM to the next message, as ``feed`` does for the
default path, and it carries the resampler's history across messages, so
chunk boundaries stay a function of the audio and not of packetisation.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from verbatim.audio.g711 import decode_alaw, decode_mulaw
from verbatim.audio.resample import SUPPORTED_RATES, Resampler

__all__ = ["ENGINE_RATE_HZ", "WIRE_ENCODINGS", "WireDecoder", "wire_decoder"]

#: The one rate the ring and the recognizer see.
ENGINE_RATE_HZ: Final = 16000
#: Wire encodings served. FLAC and OGGOPUS are codecs this server does not carry.
WIRE_ENCODINGS: Final = ("LINEAR_PCM", "MULAW", "ALAW")

_SCALE_IN: Final = 1.0 / 32768.0
_SCALE_OUT: Final = 32767.0


def _to_pcm16(samples: np.ndarray) -> bytes:
    if len(samples) == 0:
        return b""
    clipped = np.clip(np.rint(samples * _SCALE_OUT), -32768.0, 32767.0)
    return clipped.astype("<i2").tobytes()


class WireDecoder:
    """Wire bytes in, PCM16LE 16 kHz bytes out; one instance per session."""

    def __init__(self, encoding: str, sample_rate_hz: int) -> None:
        if encoding not in WIRE_ENCODINGS:
            names = ", ".join(WIRE_ENCODINGS)
            raise ValueError(f"unknown wire encoding {encoding!r}: expected one of {names}")
        if sample_rate_hz not in SUPPORTED_RATES:
            rates = ", ".join(str(r) for r in SUPPORTED_RATES)
            raise ValueError(f"unsupported wire rate {sample_rate_hz!r}: expected one of {rates}")
        self._encoding = encoding
        self._sample_rate_hz = sample_rate_hz
        self._carry = b""
        self._resampler = (
            Resampler(sample_rate_hz, ENGINE_RATE_HZ) if sample_rate_hz != ENGINE_RATE_HZ else None
        )

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    def decode(self, payload: bytes) -> bytes:
        """Any size in, including an odd byte count for LINEAR_PCM (the odd byte is
        carried to the next call); the PCM16LE 16 kHz bytes that audio became, out."""
        if self._encoding == "LINEAR_PCM":
            combined = self._carry + bytes(payload)
            complete = (len(combined) // 2) * 2
            self._carry = combined[complete:]
            pcm = np.frombuffer(combined[:complete], dtype="<i2")
        elif self._encoding == "MULAW":
            pcm = decode_mulaw(payload)
        else:
            pcm = decode_alaw(payload)
        if self._resampler is None:
            return pcm.astype("<i2", copy=False).tobytes()
        return _to_pcm16(self._resampler.process(pcm.astype(np.float32) * _SCALE_IN))

    def flush(self) -> bytes:
        """The resampler's tail at the end of the stream; empty at 16 kHz."""
        if self._resampler is None:
            return b""
        return _to_pcm16(self._resampler.flush())


def wire_decoder(encoding: str, sample_rate_hz: int) -> WireDecoder | None:
    """The decoder a session needs, or None on the default path (LINEAR_PCM at 16 kHz),
    where the bytes on the wire are the bytes the ring takes."""
    if encoding == "LINEAR_PCM" and sample_rate_hz == ENGINE_RATE_HZ:
        return None
    return WireDecoder(encoding, sample_rate_hz)
