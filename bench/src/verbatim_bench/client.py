# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The WebSocket session client: paced chunk-sized PCM replay with client-side timing.

All timestamps come from one monotonic clock in this process; the server is never
asked for its own latency figures. Receiving runs concurrently with sending so the
client measures the server, not its own blocking.

Partial samples cover only chunks acknowledged by a qualifying server watermark; when fewer samples
exist than chunks sent, the reported percentile is a lower bound. They are matched as the partials
arrive rather than after the session closes, and each one keeps the moment it was matched in
`partial_recv_s`, so the load generator can attribute it to the measurement phase it fell in without
waiting for a 180-second session to end.

A stream carries as many `final` frames as the server found endpoints in it, so the
reader reads to the close of the socket rather than stopping at the first one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from urllib.parse import urlencode

import websockets

from verbatim_bench import constants
from verbatim_bench.corpus import Utterance

_SAMPLE_RATE_HZ = 16000
_WATERMARK_TOL_S = 0.5 / _SAMPLE_RATE_HZ
_FINAL_WAIT_S = 15.0

_CHUNK_MODES_MS = (80, 160, 560, 1120)


def _watermark_of(event: Mapping[str, Any]) -> float | None:
    """Return a usable cumulative-audio watermark from a partial event."""
    value = event.get("audio_s")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


class WatermarkMatch(NamedTuple):
    """One matched chunk: its latency, and the wall-clock moment it was matched at."""

    latency_ms: float
    recv_s: float


class WatermarkMatcher:
    """The streaming form of the watermark match, fed in arrival order.

    Chunks are offered as they are sent and partials as they arrive; each partial
    releases every still-pending chunk its watermark covers. Running a whole session
    through it reproduces `match_partials_by_watermark` exactly, and that function is
    written in terms of this class so there is only one definition.

    The streaming form exists because the load generator has to read latency while the
    sessions are still open: a measurement window opens on a warm-up that has settled,
    and nothing has settled yet at the moment the first session happens to end.
    """

    __slots__ = ("_index", "_pending")

    def __init__(self) -> None:
        self._pending: list[tuple[float, float]] = []
        self._index = 0

    def offer_chunk(self, t_send: float, audio_s: float) -> None:
        """Record that a measurement chunk ending at `audio_s` was sent at `t_send`."""
        self._pending.append((t_send, audio_s))

    def offer_partial(self, t_recv: float, watermark: float | None) -> list[WatermarkMatch]:
        """Release every pending chunk this partial's watermark covers, oldest first."""
        if watermark is None:
            return []
        released: list[WatermarkMatch] = []
        while self._index < len(self._pending):
            t_send, audio_s = self._pending[self._index]
            if watermark < audio_s - _WATERMARK_TOL_S:
                break
            released.append(WatermarkMatch(max(0.0, (t_recv - t_send) * 1000.0), t_recv))
            self._index += 1
        return released


def match_partials_with_recv(
    send_times: Sequence[float],
    sent_audio_s: Sequence[float],
    partials: Sequence[tuple[float, float | None]],
) -> list[WatermarkMatch]:
    """`match_partials_by_watermark`, keeping the receive time of each matched sample."""
    matcher = WatermarkMatcher()
    for t_send, audio_s in zip(send_times, sent_audio_s, strict=True):
        matcher.offer_chunk(t_send, audio_s)
    matched: list[WatermarkMatch] = []
    for t_recv, watermark in partials:
        matched.extend(matcher.offer_partial(t_recv, watermark))
    return matched


def match_partials_by_watermark(
    send_times: Sequence[float],
    sent_audio_s: Sequence[float],
    partials: Sequence[tuple[float, float | None]],
) -> list[float]:
    """Match each chunk to the first received partial that covers its audio.

    Watermarks make backlog visible instead of assigning a partial to the next
    send merely because it arrived after that send.
    """
    matched = match_partials_with_recv(send_times, sent_audio_s, partials)
    return [match.latency_ms for match in matched]


def join_final_texts(events: Sequence[Mapping[str, Any]]) -> str:
    """Join the text of every final frame, in arrival order, into one utterance.

    The server marks a hypothesis final at every endpoint it detects, so an
    utterance containing an internal silence arrives as several segments. The
    transcript of the stream is their concatenation. Segments are joined with a
    single space and an empty or whitespace-only segment contributes nothing, so
    the result never carries a doubled or leading separator. A frame whose
    ``text`` is missing or not a string still counts as a final; it just has no
    text to contribute.
    """
    parts = [
        text.strip()
        for text in (event.get("text") for event in events)
        if isinstance(text, str) and text.strip()
    ]
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ChunkMode:
    ms: int

    @property
    def samples(self) -> int:
        return self.ms * _SAMPLE_RATE_HZ // 1000

    @property
    def bytes(self) -> int:
        return self.samples * 2

    @classmethod
    def parse(cls, spec: str | int) -> ChunkMode:
        """Accepts 160, "160", "160ms". Rejects anything not in {80, 160, 560, 1120}."""
        text = str(spec).strip().lower()
        if text.endswith("ms"):
            text = text[:-2].strip()
        try:
            ms = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid chunk mode {spec!r}: expected {_CHUNK_MODES_MS}") from exc
        if ms not in _CHUNK_MODES_MS or text != str(ms):
            raise ValueError(f"invalid chunk mode {spec!r}: expected {_CHUNK_MODES_MS}")
        return cls(ms=ms)


@dataclass
class SessionResult:
    session_id: str
    stream_id: str
    server_session_id: str | None = None
    started_at_s: float = 0.0
    chunks: int = 0
    audio_s: float = 0.0
    first_partial_ms: float | None = None
    # Milliseconds from the last sent chunk to the LAST final of the stream, which
    # is the one that completes it. A stream endpointed several times carries an
    # earlier final per segment; those are counted in `finals_received` and their
    # text is joined into `final_text`, but they do not set this.
    final_ms: float | None = None
    partial_ms: list[float] = field(default_factory=list)
    partial_recv_s: list[float] = field(default_factory=list)
    pacing_slip_ms: list[float] = field(default_factory=list)
    partials_received: int = 0
    # Every final frame the stream carried, not merely whether one arrived.
    finals_received: int = 0
    # The whole utterance: the text of every final joined in arrival order.
    final_text: str = ""
    reference_text: str = ""
    error: str | None = None
    frame_ms: int | None = None


def _split_chunks(pcm: bytes, frame_bytes: int) -> list[bytes]:
    if len(pcm) == 0:
        return []
    return [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]


async def run_session(
    endpoint: str,
    *,
    session_id: str,
    utterance: Utterance,
    pcm: bytes,
    chunk: ChunkMode,
    start_delay_s: float,
    words: bool = False,
    lang: str = "en-US",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    frame_ms: int | None = None,
    frame_seed: int = 0,
    on_open: Callable[[], None] | None = None,
    on_sample: Callable[[float, float], None] | None = None,
) -> SessionResult:
    """Open one WebSocket session and replay `pcm` at real-time pace.

    `on_open` fires once, when the server has acknowledged the session, which is the
    only moment at which this stream is demonstrably live on the server rather than
    merely scheduled. `on_sample` fires for every matched chunk as it is matched, with
    its latency and its receive time, so a caller can read latency while the session is
    still open. Both are called from this coroutine and must not block.

    Transport framing (`frame_ms`) is decoupled from measurement chunking
    (`chunk.ms`): canonical runs send 20 ms frames with seeded jitter, on the wire:
    frame i is sent at `t0 + i * frame_period + jitter_i` and graded against that
    same deadline, so the jitter shapes what the server sees and the slip measures
    only the generator's own lateness. The per-chunk watermark samples in
    `partial_ms` stay per chunk. `clock` and `sleep` are test-harness inputs. When `frame_ms`
    is None the client sends chunk-sized frames, which is recorded as
    non-canonical framing. The true short tail is sent unpadded and the
    server pads it, so client and server agree. Then sends `{"type": "end"}`
    and reads until the peer closes or the wait expires, collecting every
    `final` the stream carried rather than stopping at the first.

    Never raises for a server-side or transport error: it records it in
    `SessionResult.error` and returns. A run must survive a server that drops.
    """
    result = SessionResult(session_id=session_id, stream_id=utterance.stream_id)
    result.reference_text = utterance.text
    result.frame_ms = frame_ms if frame_ms is not None else chunk.ms
    try:
        return await _run_session_inner(
            endpoint,
            result=result,
            utterance=utterance,
            pcm=pcm,
            chunk=chunk,
            start_delay_s=start_delay_s,
            words=words,
            lang=lang,
            clock=clock,
            sleep=sleep,
            frame_ms=frame_ms,
            frame_seed=frame_seed,
            on_open=on_open,
            on_sample=on_sample,
        )
    except Exception as exc:  # transport errors are data, not crashes
        if result.error is None:
            result.error = f"{type(exc).__name__}: {exc}"
        return result


async def _run_session_inner(
    endpoint: str,
    *,
    result: SessionResult,
    utterance: Utterance,
    pcm: bytes,
    chunk: ChunkMode,
    start_delay_s: float,
    words: bool,
    lang: str,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    frame_ms: int | None = None,
    frame_seed: int = 0,
    on_open: Callable[[], None] | None = None,
    on_sample: Callable[[float, float], None] | None = None,
) -> SessionResult:
    if start_delay_s > 0:
        await sleep(start_delay_s)
    query = urlencode({"chunk_ms": chunk.ms, "lang": lang, "words": "1" if words else "0"})
    url = f"{endpoint}?{query}"
    effective_frame_ms = frame_ms if frame_ms is not None else chunk.ms
    if effective_frame_ms <= 0:
        raise ValueError(f"invalid frame_ms {effective_frame_ms!r}")
    frame_bytes = effective_frame_ms * _SAMPLE_RATE_HZ // 1000 * 2
    wire_frames = _split_chunks(pcm, frame_bytes)
    chunk_bytes = chunk.bytes
    # Group wire frames back into measurement chunks so `partial_ms` stays per
    # chunk while the transport uses canonical 20 ms framing.
    chunk_groups: list[list[bytes]] = []
    pending_group: list[bytes] = []
    pending_bytes = 0
    for wire in wire_frames:
        pending_group.append(wire)
        pending_bytes += len(wire)
        if pending_bytes >= chunk_bytes:
            chunk_groups.append(pending_group)
            pending_group = []
            pending_bytes = 0
    if pending_group:
        chunk_groups.append(pending_group)
    if not chunk_groups and len(pcm) == 0:
        chunk_groups = []
    frame_period_s = effective_frame_ms / 1000.0
    jitter_rng = random.Random(frame_seed)
    jitter_s = constants.FRAME_JITTER_MS / 1000.0

    def frame_deadline(index: int) -> float:
        """When frame `index` is due: the nominal grid plus this frame's seeded jitter,
        drawn once, here, for canonical framing only. The sender sleeps to this deadline
        and is graded against it, so the jitter is on the wire and the slip is the
        generator's own lateness. Frame 0 is due at t0. One draw per frame, in frame
        order, so a seed reproduces the schedule."""
        if index == 0:
            return t0
        jitter = jitter_rng.uniform(-jitter_s, jitter_s) if effective_frame_ms != chunk.ms else 0.0
        return t0 + index * frame_period_s + jitter

    send_times: list[float] = []
    matcher = WatermarkMatcher()
    partial_events: list[tuple[float, dict[str, Any]]] = []
    final_events: list[tuple[float, dict[str, Any]]] = []
    failure: str | None = None

    async with websockets.connect(url, max_size=None) as ws:
        try:
            first_raw = await asyncio.wait_for(ws.recv(), timeout=_FINAL_WAIT_S)
        except TimeoutError as exc:
            result.error = f"TimeoutError: no session message: {exc}"
            return result
        first = json.loads(first_raw) if isinstance(first_raw, str) else {}
        if not isinstance(first, dict) or first.get("type") != "session":
            result.error = f"ProtocolError: first frame was not a session message: {first!r:.120}"
            return result
        server_id = first.get("id")
        result.server_session_id = server_id if isinstance(server_id, str) else None
        result.started_at_s = clock()
        t0 = result.started_at_s
        if on_open is not None:
            on_open()

        async def reader() -> None:
            """Read to the close of the socket, keeping every frame the stream sent.

            A final is not the end of the stream: the server marks one per endpoint
            it detects, and partials keep coming after it. Returning on the first
            would leave the later segments out of the transcript and the later
            chunks without a watermark to match. Only an error frame, which the
            server does not continue past, ends the read early.
            """
            nonlocal failure
            try:
                async for message in ws:
                    now = clock()
                    if not isinstance(message, str):
                        continue
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    kind = event.get("type")
                    if kind == "partial":
                        partial_events.append((now, event))
                        for match in matcher.offer_partial(now, _watermark_of(event)):
                            result.partial_ms.append(match.latency_ms)
                            result.partial_recv_s.append(match.recv_s)
                            if on_sample is not None:
                                on_sample(match.latency_ms, match.recv_s)
                    elif kind == "final":
                        final_events.append((now, event))
                    elif kind == "error":
                        failure = f"ServerError {event.get('code')}: {event.get('message')}"
                        return
            except Exception as exc:  # a dropped socket is a datum, not a crash
                if not final_events and failure is None:
                    failure = f"{type(exc).__name__}: {exc}"

        reader_task = asyncio.create_task(reader())
        try:
            frame_index = 0
            deadline = frame_deadline(0)
            for group in chunk_groups:
                for position, wire in enumerate(group):
                    last_in_chunk = position == len(group) - 1
                    now = clock()
                    slip_ms = max(0.0, (now - deadline) * 1000.0)
                    result.pacing_slip_ms.append(slip_ms)
                    try:
                        await ws.send(wire)
                    except Exception as exc:
                        if failure is None:
                            failure = f"{type(exc).__name__}: {exc}"
                        break
                    frame_index += 1
                    result.audio_s += len(wire) / 2 / _SAMPLE_RATE_HZ
                    if last_in_chunk:
                        t_chunk = clock()
                        send_times.append(t_chunk)
                        result.chunks += 1
                        matcher.offer_chunk(t_chunk, result.audio_s)
                    if not last_in_chunk or group is not chunk_groups[-1]:
                        deadline = frame_deadline(frame_index)
                        delay = deadline - clock()
                        if delay > 0:
                            await sleep(delay)
                else:
                    continue
                break
            t_end_audio = send_times[-1] if send_times else clock()
            try:
                await ws.send(json.dumps({"type": "end"}))
            except Exception as exc:
                if failure is None:
                    failure = f"{type(exc).__name__}: {exc}"
            try:
                await asyncio.wait_for(reader_task, timeout=_FINAL_WAIT_S)
            except TimeoutError:
                # The wait bounds a server that never closes. Whatever arrived
                # before it expired is kept; only a stream that produced no final
                # at all is a failure.
                if not final_events and failure is None:
                    failure = "TimeoutError: no final received after end"
        finally:
            if not reader_task.done():
                reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reader_task

    result.partials_received = len(partial_events)
    result.finals_received = len(final_events)
    result.final_text = join_final_texts([event for _, event in final_events])
    t_first_audio = send_times[0] if send_times else result.started_at_s
    first_non_empty = next(
        (t for t, e in partial_events if isinstance(e.get("text"), str) and e["text"] != ""),
        None,
    )
    if first_non_empty is not None:
        result.first_partial_ms = (first_non_empty - t_first_audio) * 1000.0
    if final_events:
        # To the last final: an endpointed stream finalises each segment as it
        # ends, and only the last of them completes the stream.
        result.final_ms = (final_events[-1][0] - t_end_audio) * 1000.0
    if failure is not None and not final_events:
        result.error = failure
    return result
