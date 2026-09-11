# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Protocol surfaces and the doors into the server: the engine handles.

The ABCs exist so an operator can carry a private protocol adapter without
forking, and so the fake pipeline plus the conformance suite can test protocols
with no GPU.

``EngineHandle`` and ``SessionHandle`` are the one way a transport runs a session:
a synchronous, non-blocking ``feed`` plus a ``results`` async iterator fed from one
engine-owned queue. There is no other. The synchronous per-session recognizer the
transports once called on the event loop is gone, because a blocking recognizer on
the loop is head-of-line blocking for every other session, and because a tick
scheduler cannot batch what it is not handed.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final

from verbatim.audio.decoder import WIRE_ENCODINGS
from verbatim.audio.resample import SUPPORTED_RATES
from verbatim.core.errors import InvalidArgument
from verbatim.core.types import Word

__all__ = [
    "SAMPLE_RATE_HZ",
    "VALID_CHUNK_MS",
    "EngineHandle",
    "Hypothesis",
    "SessionHandle",
    "SessionOptions",
    "Word",
]

SAMPLE_RATE_HZ: Final = 16000
VALID_CHUNK_MS: Final = (80, 160, 560, 1120)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One recognizer output. `is_final` distinguishes a partial from an utterance final."""

    text: str
    is_final: bool
    audio_processed_s: float
    words: tuple[Word, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class SessionOptions:
    """Per-session numeric configuration. A transcript is only comparable with these."""

    chunk_ms: int = 160
    language_code: str = "en-US"
    sample_rate_hz: int = SAMPLE_RATE_HZ
    interim_results: bool = True
    word_timestamps: bool = False
    model: str = ""
    #: End-of-utterance silence in milliseconds for this session, or None for the
    #: engine default. Carried into NeMo's per-stream request options by the adapter.
    stop_history_eou_ms: int | None = None
    #: How the audio arrives on the wire. The recognizer always sees PCM16 at
    #: `sample_rate_hz`; a G.711 encoding or another rate is expanded and resampled
    #: by the transport, in a worker thread, before `feed`. Part of comparability:
    #: G.711 and a resample change the audio a transcript was made from.
    wire_encoding: str = "LINEAR_PCM"
    wire_sample_rate_hz: int = SAMPLE_RATE_HZ

    def __post_init__(self) -> None:
        """Raise InvalidArgument for a chunk_ms outside VALID_CHUNK_MS, a wire encoding
        or wire rate not served, naming the field and listing the valid values. A
        silent substitution would produce a transcript nobody can attribute to a
        configuration."""
        if self.chunk_ms not in VALID_CHUNK_MS:
            valid = ", ".join(str(v) for v in VALID_CHUNK_MS)
            raise InvalidArgument(f"invalid chunk_ms {self.chunk_ms!r}: must be one of {valid}")
        if self.wire_encoding not in WIRE_ENCODINGS:
            names = ", ".join(WIRE_ENCODINGS)
            raise InvalidArgument(
                f"invalid wire_encoding {self.wire_encoding!r}: must be one of {names}"
            )
        if (
            isinstance(self.wire_sample_rate_hz, bool)
            or self.wire_sample_rate_hz not in SUPPORTED_RATES
        ):
            rates = ", ".join(str(r) for r in SUPPORTED_RATES)
            raise InvalidArgument(
                f"invalid wire_sample_rate_hz {self.wire_sample_rate_hz!r}: must be one of {rates}"
            )
        if self.stop_history_eou_ms is not None and (
            isinstance(self.stop_history_eou_ms, bool) or self.stop_history_eou_ms < 0
        ):
            raise InvalidArgument(
                f"invalid stop_history_eou_ms {self.stop_history_eou_ms!r}: must be >= 0"
            )

    @property
    def chunk_samples(self) -> int:
        """Samples per chunk: chunk_ms * sample_rate_hz // 1000."""
        return self.chunk_ms * self.sample_rate_hz // 1000

    @property
    def chunk_bytes(self) -> int:
        """Bytes per chunk of PCM16LE mono: chunk_samples * 2."""
        return self.chunk_samples * 2


class SessionHandle(abc.ABC):
    """One live session, from a transport's point of view. The transport never sees a
    thread, a ring, a tick or a Session object."""

    @property
    @abc.abstractmethod
    def options(self) -> SessionOptions: ...

    @abc.abstractmethod
    def feed(self, pcm: bytes) -> int:
        """Hand PCM16LE bytes to the session. NON-BLOCKING and any size, including a size
        that is not a multiple of a chunk and including an odd byte count.

        Returns the number of BYTES this session took ownership of. A trailing odd byte
        held as carry IS owned and IS counted here, so a one-byte message returns 1: a
        return of 0 would tell the caller to re-offer a byte the session already holds,
        and re-offering it would count it twice into audio_processed_s.

        The return value is short ONLY when the session's ring is full. The caller MUST
        retain exactly the bytes that were not counted, and stop reading its socket until
        they have been accepted. Audio is never dropped server-side.
        """

    @abc.abstractmethod
    def end(self) -> None:
        """Half-close: no more audio. The remaining buffered audio flushes as normal
        chunks and the last partial chunk goes out as `is_last`."""

    @abc.abstractmethod
    def abort(self) -> None:
        """The client is gone. Runs the same drain path with an all-zero final chunk, so
        the slot bookkeeping has no special case. Buffered bytes are discarded and never
        counted into audio_processed_s."""

    @abc.abstractmethod
    def results(self) -> AsyncIterator[Hypothesis]:
        """Every result for this session, in order, forever until the session ends.

        Ends (StopAsyncIteration) after the `is_last` result, or after `abort()`. Raises
        the VerbatimError that killed the session, if one did. There is deliberately no
        callback and no blocking 'get one result' call: a transport that could pull a
        result synchronously would put the tick thread's latency on the client's send
        cadence, which is the defect this interface exists to prevent.
        """


class EngineHandle(abc.ABC):
    """The only door into the engine. A protocol adapter depends on this and on nothing
    under the scheduler package or the concrete engine module."""

    @property
    @abc.abstractmethod
    def chunk_ms(self) -> int:
        """The single chunk mode this engine serves."""

    @abc.abstractmethod
    def open_session(self, options: SessionOptions) -> SessionHandle:
        """Admit and register a session. Synchronous, O(1), safe to call from the asyncio
        loop. Raises ResourceExhausted (with retry_after_ms) when refused; nothing is
        reserved in that case."""

    @abc.abstractmethod
    async def wait_for_ticks(self, n: int) -> None:
        """Return after `n` further ticks have completed. A transport whose `feed` was
        accepted short waits one tick here before offering the remainder again; it
        reads nothing from its socket meanwhile, which is the back-pressure."""
