# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Protocol surfaces and the doors into the server: ``Recognizer`` and the engine handles.

The ABCs exist so an operator can carry a private protocol adapter without
forking, and so the fake pipeline plus the conformance suite can test protocols
with no GPU.

This module is home to two interfaces, side by side. ``Recognizer`` is what every
protocol surface talks to today: one instance per session. It is deliberately
synchronous and deliberately tiny, because the real adapter over NeMo's
cache-aware pipeline is driven from a dedicated tick thread, not from the event
loop; a transport that needs to call a blocking recognizer dispatches it to a
worker. ``EngineHandle`` / ``SessionHandle`` are the asynchronous result contract
that replaces it: a non-blocking ``feed`` plus a ``results`` async iterator fed
from one engine-owned queue.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Final

from verbatim.core.errors import InvalidArgument
from verbatim.core.types import Word

__all__ = [
    "SAMPLE_RATE_HZ",
    "VALID_CHUNK_MS",
    "EngineHandle",
    "Hypothesis",
    "Recognizer",
    "RecognizerFactory",
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

    def __post_init__(self) -> None:
        """Raise InvalidArgument for a chunk_ms outside VALID_CHUNK_MS, naming the
        field and listing the valid values. A silent substitution would produce a
        transcript nobody can attribute to a configuration."""
        if self.chunk_ms not in VALID_CHUNK_MS:
            valid = ", ".join(str(v) for v in VALID_CHUNK_MS)
            raise InvalidArgument(f"invalid chunk_ms {self.chunk_ms!r}: must be one of {valid}")

    @property
    def chunk_samples(self) -> int:
        """Samples per chunk: chunk_ms * sample_rate_hz // 1000."""
        return self.chunk_ms * self.sample_rate_hz // 1000

    @property
    def chunk_bytes(self) -> int:
        """Bytes per chunk of PCM16LE mono: chunk_samples * 2."""
        return self.chunk_samples * 2


class Recognizer(abc.ABC):
    """What every protocol surface talks to. One instance per session.

    Deliberately synchronous and deliberately tiny. The real adapter over NeMo's
    cache-aware pipeline is a later task and is driven from a dedicated tick thread,
    not from the event loop; a transport that needs to call a blocking recognizer
    dispatches it to a worker. Implementations here must be non-blocking.
    """

    @property
    @abc.abstractmethod
    def options(self) -> SessionOptions: ...

    @abc.abstractmethod
    def add_chunk(self, pcm: bytes) -> list[Hypothesis]:
        """Consume exactly `options.chunk_bytes` of PCM16LE and return what is ready.

        Returns zero or more hypotheses in emission order. A tick that produces
        nothing returns an empty list -- that is normal, not an error. Raises
        InvalidArgument if `len(pcm) != options.chunk_bytes`: the caller owns
        chunking, and a recognizer that silently accepted a short buffer would make
        chunk boundaries depend on the transport.
        """

    @abc.abstractmethod
    def finalize(self) -> list[Hypothesis]:
        """Flush. Returns the remaining hypotheses, the last of which has is_final=True.

        Called once, after the last `add_chunk`. Calling `add_chunk` afterwards raises
        InvalidArgument.
        """


RecognizerFactory = Callable[[SessionOptions], Recognizer]


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
