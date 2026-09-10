# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Session state machine acceptance tests: legal paths, illegal transitions, frame flags."""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.config import SAMPLE_RATE_HZ, ChunkMode
from verbatim.core.session import IllegalTransition, Session, SessionState

CHUNK = ChunkMode(160)
N = CHUNK.samples


def _live_session(session_id: int = 1) -> Session:
    session = Session(session_id, CHUNK)
    session.configure()
    session.admit()
    return session


def test_happy_path_transitions() -> None:
    session = Session(1, CHUNK)
    assert session.state is SessionState.CONNECTING
    session.configure()
    assert session.state is SessionState.CONFIGURED
    session.admit()
    assert session.state is SessionState.ADMITTED
    session.ring.write(np.zeros(2 * N, dtype=np.float32))
    frame = session.next_frame()
    assert frame is not None
    assert session.state is SessionState.RUNNING
    session.begin_draining()
    assert session.state is SessionState.DRAINING
    session.close()
    assert session.state is SessionState.CLOSED


def test_reject_path() -> None:
    session = Session(1, CHUNK)
    session.configure()
    session.reject()
    assert session.state is SessionState.REJECTED


def test_illegal_transition_raises_and_names_both_states() -> None:
    session = Session(1, CHUNK)
    with pytest.raises(IllegalTransition, match=r"CONNECTING.*ADMITTED"):
        session.admit()
    live = _live_session()
    with pytest.raises(IllegalTransition, match=r"RUNNING.*CLOSED"):
        live.ring.write(np.zeros(N, dtype=np.float32))
        live.next_frame()
        live.close()


def test_first_frame_sets_is_first_exactly_once() -> None:
    session = _live_session()
    session.ring.write(np.zeros(3 * N, dtype=np.float32))
    flags = []
    for _ in range(3):
        frame = session.next_frame()
        assert frame is not None
        flags.append(frame.is_first)
    assert flags == [True, False, False]


def test_starved_when_less_than_a_chunk_is_buffered() -> None:
    session = _live_session()
    assert session.next_frame() is None
    assert session.state is SessionState.STARVED
    # No synthetic audio was inserted: the ring is still empty.
    assert session.ring.available == 0
    audio = np.arange(N, dtype=np.float32)
    session.ring.write(audio)
    frame = session.next_frame()
    assert frame is not None
    # The next call returns the audio that was already there.
    assert list(frame.samples) == list(audio)
    assert session.state is SessionState.RUNNING


def test_drain_frame_sets_is_last_and_valid_samples() -> None:
    session = _live_session()
    session.ring.write(np.ones(100, dtype=np.float32))
    session.begin_draining()
    frame = session.next_frame()
    assert frame is not None
    assert frame.is_last is True
    assert frame.valid_samples == 100
    assert len(frame.samples) == N
    assert list(frame.samples[:100]) == [1.0] * 100
    assert list(frame.samples[100:]) == [0.0] * (N - 100)


def test_abort_goes_through_draining_with_a_zero_chunk() -> None:
    session = _live_session()
    session.ring.write(np.zeros(N, dtype=np.float32))
    assert session.next_frame() is not None
    session.begin_draining(aborted=True)
    # No separate teardown path: an abort is DRAINING like any other end.
    assert session.state is SessionState.DRAINING
    frame = session.next_frame()
    assert frame is not None
    assert frame.is_last is True
    assert len(frame.samples) == N
    assert bool((frame.samples == 0).all())
    session.close()
    assert session.state is SessionState.CLOSED


def test_audio_processed_excludes_padding() -> None:
    session = _live_session()
    session.ring.write(np.zeros(2 * N + 100, dtype=np.float32))
    session.begin_draining()
    assert session.next_frame() is not None
    assert session.next_frame() is not None
    last = session.next_frame()
    assert last is not None and last.is_last
    assert session.audio_processed_s == pytest.approx((2 * N + 100) / SAMPLE_RATE_HZ)


def test_abort_does_not_count_silence_as_processed_audio() -> None:
    session = _live_session()
    session.ring.write(np.zeros(N, dtype=np.float32))
    assert session.next_frame() is not None
    session.begin_draining(aborted=True)
    frame = session.next_frame()
    assert frame is not None
    assert frame.valid_samples == 0
    assert session.audio_processed_s == pytest.approx(N / SAMPLE_RATE_HZ)
