# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Stateful chunked resampling for the rare non-16 kHz stream.

A polyphase FIR in numpy: upsample by ``L``, low-pass, decimate by ``M``, with
``L/M`` the reduced ratio of the two rates, computed one output sample at a time
from the polyphase branch that sample falls on, so a chunk boundary is invisible
to the filter. Each session keeps one instance; it carries the filter's history
across calls, and the concatenation of the chunked outputs is the one-shot output
to floating-point rounding.

The prototype filter is a Kaiser-windowed sinc, ``TAPS_PER_PHASE`` taps per
branch, cut at half the lower of the two rates, with a DC gain of exactly ``L``.
It is designed once per session, at open; the per-call cost is one gather and one
multiply-add per output sample per tap.

Neither ``soxr`` nor ``scipy`` is a dependency of this package, so neither is used
here: the numpy path is the only one, and it never runs on the GPU. A per-session
variable-shape operation on the tick path is precisely what costs CUDA graphs.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

__all__ = ["SUPPORTED_RATES", "TAPS_PER_PHASE", "Resampler"]

#: Wire sample rates served. The polyphase branch count is the reduced upsampling
#: factor, which stays under a thousand for every one of these; an arbitrary rate
#: (8001 Hz) would need tens of thousands, so rates are a list, not a range.
SUPPORTED_RATES: Final = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
TAPS_PER_PHASE: Final = 32
_KAISER_BETA: Final = 8.0


class Resampler:
    """Rate conversion with state. ``process`` returns the outputs that can be computed
    from the audio seen so far; ``flush`` returns the rest at the end of the stream, so
    that ``N`` input samples give exactly ``ceil(N * L / M)`` outputs, the samples whose
    time falls within the input's duration. Output ``k`` sits at input time ``k * M / L``
    exactly: there is no filter delay on the timeline, and the watermark counts the same
    seconds at either rate."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError(f"rates must be positive, got {in_rate!r} -> {out_rate!r}")
        self._in_rate = in_rate
        self._out_rate = out_rate
        g = math.gcd(in_rate, out_rate)
        self._up = out_rate // g  # L
        self._down = in_rate // g  # M
        self._identity = self._up == 1 and self._down == 1
        q = TAPS_PER_PHASE
        self._taps = q
        n = q * self._up
        # Cut at half the lower rate, in cycles per sample of the upsampled signal.
        cutoff = 0.5 * min(1.0 / self._up, 1.0 / self._down)
        centre = (n - 1) / 2.0
        window = np.kaiser(n, _KAISER_BETA)
        prototype = 2.0 * cutoff * np.sinc(2.0 * cutoff * (np.arange(n) - centre)) * window
        prototype *= self._up / prototype.sum()
        # Branch p holds taps p, p+L, p+2L, ...: shape (L, q).
        self._branches = prototype.reshape(q, self._up).T.astype(np.float32).copy()
        self._centre = (n - 1) // 2
        self._history = np.zeros(q - 1, dtype=np.float32)
        self._consumed = 0  # input samples ever consumed
        self._produced = 0  # output samples ever produced
        self._flushed = False

    @property
    def in_rate(self) -> int:
        return self._in_rate

    @property
    def out_rate(self) -> int:
        return self._out_rate

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Consume float32 samples at the input rate; return the float32 output samples
        that can be computed so far, at the output rate."""
        if self._flushed:
            raise ValueError("process called after flush: the stream has ended")
        x = np.asarray(samples, dtype=np.float32).ravel()
        if self._identity:
            return x.copy()
        return self._advance(x)

    def flush(self) -> np.ndarray:
        """The output samples at the tail of the stream, whose time falls within the audio
        already consumed but whose filter window reached past its end."""
        if self._flushed:
            return np.zeros(0, dtype=np.float32)
        self._flushed = True
        if self._identity or self._consumed == 0:
            return np.zeros(0, dtype=np.float32)
        # The stream covers the duration consumed / in_rate: every output whose time
        # falls inside it, which is ceil(consumed * L / M) samples in all.
        total = -(-self._consumed * self._up // self._down)
        padding = np.zeros(self._taps + self._centre // self._up + 2, dtype=np.float32)
        tail = self._advance(padding)
        keep = max(0, total - (self._produced - len(tail)))
        return tail[:keep]

    def _advance(self, x: np.ndarray) -> np.ndarray:
        q = self._taps
        up, down, centre = self._up, self._down, self._centre
        buffer = np.concatenate((self._history, x))  # buffer[i] = input[consumed - (q-1) + i]
        latest = self._consumed + len(x) - 1  # absolute index of the newest input sample
        # Output k reads the upsampled signal at position k*M + centre, which is input
        # index (k*M + centre) // L on branch (k*M + centre) % L. Produce every k whose
        # newest input has arrived.
        first = self._produced
        last = (latest * up - centre) // down if latest * up >= centre else -1
        if last < first:
            self._history = buffer[-(q - 1) :] if q > 1 else buffer[:0]
            self._consumed += len(x)
            return np.zeros(0, dtype=np.float32)
        ks = np.arange(first, last + 1, dtype=np.int64)
        positions = ks * down + centre
        newest = positions // up  # absolute input index of the newest tap
        branch = positions % up
        local = newest - (self._consumed - (q - 1))  # index into buffer
        gather = local[:, None] - np.arange(q, dtype=np.int64)[None, :]
        out = np.einsum("kq,kq->k", buffer[gather], self._branches[branch]).astype(np.float32)
        self._produced = int(last) + 1
        self._history = buffer[-(q - 1) :] if q > 1 else buffer[:0]
        self._consumed += len(x)
        return out
