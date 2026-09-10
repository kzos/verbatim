# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The live session table: join, leave, iterate. The only shared mutable state.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This registry is plain bookkeeping with no model behind it. This
module must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

from verbatim.core.errors import InvalidArgument
from verbatim.core.session import Session, SessionState

__all__ = ["SessionRegistry"]

_LIVE_STATES = frozenset(
    {
        SessionState.ADMITTED,
        SessionState.RUNNING,
        SessionState.STARVED,
        SessionState.DRAINING,
    }
)


class SessionRegistry:
    """The live session table. The only shared mutable state; every mutation goes through here."""

    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: int) -> bool:
        return session_id in self._sessions

    def add(self, session: Session) -> None:
        """Register a session. A duplicate id is a caller bug, raised loudly."""
        if session.session_id in self._sessions:
            raise InvalidArgument(
                f"duplicate session_id {session.session_id!r}: already registered"
            )
        self._sessions[session.session_id] = session

    def remove(self, session_id: int) -> None:
        """Unregister a session. Unknown ids raise KeyError: silently dropping a close
        would leak the slot it held."""
        del self._sessions[session_id]

    def get(self, session_id: int) -> Session:
        """Fetch a session by id, raising KeyError when unknown."""
        return self._sessions[session_id]

    def by_state(self, *states: SessionState) -> list[Session]:
        """Sessions currently in any of the given states, in ascending session id."""
        wanted = frozenset(states)
        return sorted(
            (s for s in self._sessions.values() if s.state in wanted),
            key=lambda s: s.session_id,
        )

    @property
    def live(self) -> int:
        """Sessions holding a slot: ADMITTED | RUNNING | STARVED | DRAINING."""
        return sum(1 for s in self._sessions.values() if s.state in _LIVE_STATES)
