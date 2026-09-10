# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``Session``: id, chunk mode, options, state machine, deadlines.

The state machine (02_architecture.md §1.8)::

    CONNECTING --config--> CONFIGURED --admit--> ADMITTED --first chunk--> RUNNING <-> STARVED
         |                     | reject                                      |
         v                     v                                             | half-close / EOS
      CLOSED               REJECTED                                          v
                     (RESOURCE_EXHAUSTED)                        DRAINING --is_last--> CLOSED

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This state machine is clock-free bookkeeping: frames are cut from the
ring's own sample counter, so a session's frame sequence is a pure function of its
audio. This module must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from verbatim.config import SAMPLE_RATE_HZ, ChunkMode
from verbatim.core.errors import VerbatimError
from verbatim.core.ring import RingBuffer
from verbatim.core.types import PcmFrame
from verbatim.protocols.base import SessionOptions

__all__ = ["IllegalTransition", "Session", "SessionState"]


class SessionState(StrEnum):
    """Every state a session can be in. Terminal states have no outgoing edges."""

    CONNECTING = "CONNECTING"
    CONFIGURED = "CONFIGURED"
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    STARVED = "STARVED"
    DRAINING = "DRAINING"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


_LIVE_STATES = frozenset(
    {
        SessionState.ADMITTED,
        SessionState.RUNNING,
        SessionState.STARVED,
        SessionState.DRAINING,
    }
)

_ALLOWED: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CONNECTING: frozenset({SessionState.CONFIGURED, SessionState.CLOSED}),
    SessionState.CONFIGURED: frozenset({SessionState.ADMITTED, SessionState.REJECTED}),
    SessionState.ADMITTED: frozenset(
        {SessionState.RUNNING, SessionState.STARVED, SessionState.DRAINING}
    ),
    SessionState.RUNNING: frozenset({SessionState.STARVED, SessionState.DRAINING}),
    SessionState.STARVED: frozenset({SessionState.RUNNING, SessionState.DRAINING}),
    SessionState.DRAINING: frozenset({SessionState.CLOSED}),
    SessionState.CLOSED: frozenset(),
    SessionState.REJECTED: frozenset(),
}


class IllegalTransition(VerbatimError):
    """A session state transition the state machine does not allow."""


class Session:
    """One stream's lifecycle and its audio. Clock-free: chunk boundaries come from
    the ring's sample counter, never from a wall clock, so a STARVED tick or an
    overrunning tick delays audio instead of inserting or dropping samples.

    This is the one and only cut site: after the transports move onto the engine,
    the only place audio is cut into chunks is `next_frame` / `_drain_frame` over
    the ring, and the only place padding is invented is `RingBuffer.drain`. The
    caller owns chunking: `feed` (via the engine) accepts any byte size and the
    ring cuts the frames, so a session's frame sequence is a pure function of its
    audio."""

    def __init__(
        self,
        session_id: int,
        chunk: ChunkMode,
        ring_seconds: float = 3.0,
        *,
        options: SessionOptions | None = None,
    ) -> None:
        if session_id < 0:
            raise ValueError(f"session_id must be >= 0, got {session_id!r}")
        if ring_seconds <= 0:
            raise ValueError(f"ring_seconds must be positive, got {ring_seconds!r}")
        self._session_id = session_id
        self._chunk = chunk
        self._options = options
        self._state = SessionState.CONNECTING
        self._ring = RingBuffer(int(SAMPLE_RATE_HZ * ring_seconds))
        self._chunks_emitted = 0
        self._audio_processed_s = 0.0
        self._aborted = False
        self._final_emitted = False

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def chunk(self) -> ChunkMode:
        return self._chunk

    @property
    def options(self) -> SessionOptions | None:
        """The request options, set by the engine at construction; None off the engine path."""
        return self._options

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def ring(self) -> RingBuffer:
        return self._ring

    @property
    def chunks_emitted(self) -> int:
        return self._chunks_emitted

    @property
    def audio_processed_s(self) -> float:
        """Audio seconds counted from valid (non-padding) samples only."""
        return self._audio_processed_s

    def _move(self, target: SessionState) -> None:
        if target is self._state:
            return
        if target not in _ALLOWED[self._state]:
            raise IllegalTransition(
                f"illegal session transition {self._state.value} -> {target.value}: "
                f"session {self._session_id}"
            )
        self._state = target

    def configure(self) -> None:
        """CONNECTING -> CONFIGURED: options built from the first request, nothing reserved."""
        self._move(SessionState.CONFIGURED)

    def admit(self) -> None:
        """CONFIGURED -> ADMITTED: a slot is reserved; the first frame has not been sent."""
        self._move(SessionState.ADMITTED)

    def reject(self) -> None:
        """CONFIGURED -> REJECTED: admission refused, and no slot was reserved."""
        self._move(SessionState.REJECTED)

    def begin_draining(self, *, aborted: bool = False) -> None:
        """A live session starts flushing: remaining audio goes out as normal chunks and
        the last partial chunk as ``is_last``. An abort (client cancel, transport error)
        takes the same path with an all-zero final chunk -- there is no separate teardown."""
        if self._state not in _LIVE_STATES:
            raise IllegalTransition(
                f"illegal session transition {self._state.value} -> "
                f"{SessionState.DRAINING.value}: session {self._session_id} is not live"
            )
        if aborted:
            self._aborted = True
        if self._state is not SessionState.DRAINING:
            self._move(SessionState.DRAINING)

    def close(self) -> None:
        """DRAINING -> CLOSED once the ``is_last`` step ran (or CONNECTING -> CLOSED
        before anything was reserved)."""
        self._move(SessionState.CLOSED)

    def _account(self, valid_samples: int) -> None:
        self._chunks_emitted += 1
        self._audio_processed_s += valid_samples / SAMPLE_RATE_HZ

    def next_frame(self) -> PcmFrame | None:
        """One chunk, or None when starved. Sets is_first on the first ever frame and
        is_last on the drain frame; transitions RUNNING <-> STARVED as a side effect."""
        n = self._chunk.samples
        if self._state is SessionState.DRAINING:
            return self._drain_frame(n)
        if self._state not in (SessionState.ADMITTED, SessionState.RUNNING, SessionState.STARVED):
            return None
        samples = self._ring.pop(n)
        if samples is None:
            self._move(SessionState.STARVED)
            return None
        is_first = self._chunks_emitted == 0
        self._account(n)
        self._move(SessionState.RUNNING)
        return PcmFrame(
            stream_id=self._session_id,
            samples=samples,
            is_first=is_first,
            is_last=False,
            valid_samples=n,
        )

    def _drain_frame(self, n: int) -> PcmFrame | None:
        if self._final_emitted:
            return None
        is_first = self._chunks_emitted == 0
        if self._aborted:
            self._final_emitted = True
            self._account(0)
            return PcmFrame(
                stream_id=self._session_id,
                samples=np.zeros(n, dtype=np.float32),
                is_first=is_first,
                is_last=True,
                valid_samples=0,
            )
        if self._ring.available >= n:
            samples = self._ring.pop(n)
            assert samples is not None
            self._account(n)
            return PcmFrame(
                stream_id=self._session_id,
                samples=samples,
                is_first=is_first,
                is_last=False,
                valid_samples=n,
            )
        padded, valid = self._ring.drain(n)
        self._final_emitted = True
        self._account(valid)
        return PcmFrame(
            stream_id=self._session_id,
            samples=padded,
            is_first=is_first,
            is_last=True,
            valid_samples=valid,
        )
