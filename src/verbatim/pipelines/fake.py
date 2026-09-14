# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Deterministic CPU stub adapter. Makes the protocol and scheduler CI GPU-free.

Returns scripted transcripts with no torch, no CUDA and no NeMo, so the whole
protocol conformance suite and the whole scheduler test suite run on a free
GitHub CPU runner.

This is what makes the entire protocol conformance suite runnable on a free CPU
runner. It is a test double that ships in the package on purpose: the protocol
surfaces must be testable by anyone, on any machine, forever.

The NeMo adapter is ``verbatim.pipelines.cache_aware_rnnt``; this fake exists so
the scheduler and both protocol surfaces are testable without a GPU. This module
must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import ClassVar, Final

import numpy as np

from verbatim.config import ChunkMode
from verbatim.core.errors import InvalidArgument
from verbatim.core.types import PcmFrame, StepResult
from verbatim.pipelines.base import (
    UPSTREAM_WARMUP_STEPS,
    GraphCapability,
    GraphKey,
    GraphPathUnavailable,
    PipelineAdapter,
)
from verbatim.protocols.base import Hypothesis, SessionOptions, Word

__all__ = ["BATCH_LEAKS", "FakePipelineAdapter", "ScriptSource", "ScriptedTranscript"]

#: The two channels the invariance gate compares, each of which the fake can be made to
#: leak batch composition into. A test-harness input for the gate's own tests: a gate
#: that has only ever seen an invariant fake cannot be shown to fail, and a gate that
#: cannot fail proves nothing. Nothing in the package sets this.
BATCH_LEAKS: Final = frozenset({"text", "timing"})

#: How a scripted adapter picks a transcript for a stream. A test-harness input.
ScriptSource = Callable[[SessionOptions], Sequence[str] | None]


class ScriptedTranscript:
    """A deterministic scripted transcript source. No model, no torch, no GPU.

    Returns one partial per chunk, each the running prefix of `script` word by word,
    and one final on `finalize()` carrying the whole script. Word timings are derived
    from the chunk index, so they are exact integers and reproducible. It is the
    per-stream script behind `FakePipelineAdapter`'s scripted mode, and nothing on
    the wire calls it directly: every session runs through the engine.

    It ships in the package on purpose: the protocol surfaces must be testable by
    anyone, on any machine, forever, and this is what they are tested against.
    """

    DEFAULT_SCRIPT: ClassVar[tuple[str, ...]] = (
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
    )

    def __init__(
        self,
        options: SessionOptions,
        *,
        script: Sequence[str] | None = None,
        partial_every: int = 1,
    ) -> None:
        self._options = options
        self._script = tuple(script) if script is not None else self.DEFAULT_SCRIPT
        if partial_every < 1:
            raise InvalidArgument(
                f"invalid partial_every {partial_every!r}: must be a positive integer"
            )
        self._partial_every = partial_every
        self._chunks = 0
        self._finalized = False

    @property
    def options(self) -> SessionOptions:
        return self._options

    def _audio_s(self) -> float:
        # Exact by construction: an integer number of chunks times chunk_ms / 1000.
        return self._chunks * self._options.chunk_ms / 1000

    def add_chunk(self, pcm: bytes) -> list[Hypothesis]:
        if self._finalized:
            raise InvalidArgument("add_chunk called after finalize: the session is closed")
        expected = self._options.chunk_bytes
        if len(pcm) != expected:
            raise InvalidArgument(f"invalid chunk size {len(pcm)} bytes: expected {expected} bytes")
        self._chunks += 1
        if self._chunks % self._partial_every != 0:
            return []
        text = " ".join(self._script[: self._chunks])
        return [
            Hypothesis(text=text, is_final=False, audio_processed_s=self._audio_s()),
        ]

    def finalize(self) -> list[Hypothesis]:
        if self._finalized:
            raise InvalidArgument("finalize called twice: the session is already closed")
        self._finalized = True
        chunk_ms = self._options.chunk_ms
        words = tuple(
            Word(word=w, start_ms=j * chunk_ms, end_ms=(j + 1) * chunk_ms)
            for j, w in enumerate(self._script)
        )
        return [
            Hypothesis(
                text=" ".join(self._script),
                is_final=True,
                audio_processed_s=self._audio_s(),
                words=words,
            ),
        ]


class FakePipelineAdapter(PipelineAdapter):
    """Deterministic CPU stub. No CUDA, no NeMo, no torch.

    Its output for a row is a pure function of that row's own samples and that stream's
    own history -- exactly the property the real runtime must have and the invariance
    gate measures. Here it holds by construction, which is what lets the scheduler's
    bookkeeping be tested for it on CPU. Hash mode also stamps chunk-indexed word timings
    on the final, so the gate's timing channel has something to compare.

    ``batch_leak`` breaks the property on purpose, one channel at a time, so the gate can
    be shown to go red: ``"text"`` mixes the number of real streams in the batch into
    every token, ``"timing"`` adds the number of companions, in milliseconds, to every
    word's end time while the text stays the same. It exists for the gate's tests and
    nothing else sets it.

    ``honour_phrases`` mixes a stream's OWN biasing digest into its OWN tokens, which is
    what a boosting tree does to a transcript: same audio, different vocabulary,
    different text. It is a per-row input and never a cross-row one, so the fake stays
    invariant by construction while the gate's phrase-list arm has something to detect.
    Without it the fake ignores a phrase list entirely, which is the other case worth
    testing: the gate must then report no verdict rather than a free pass.

    It also records every batch shape it was called with, so a test can assert the steady
    batch never changed shape without instrumenting the scheduler.

    It fakes capture and replay at the same seam the real adapter uses. ``graphs``
    sets what ``graph_capability()`` reports, so the CPU suite can drive all three
    states -- graphed, eager by choice, and asked-for-but-absent -- on a machine that
    has no graph path at all. A ``graph=True`` call with no available graph path raises
    instead of running, because a silent downgrade is the bug.

    Capture follows upstream's rule and not a convenient one: a shape is captured on the
    call whose count at that shape *exceeds* ``graph_warmup_steps``, and replayed after
    that. It used to capture on first sight, which is what let a warm-up one step short
    of upstream's threshold pass the whole CPU suite while capturing nothing on the
    B300 (docs/decisions/0011). A fake that is easier to satisfy than the thing it
    stands for is not a test double.

        The NeMo adapter is ``verbatim.pipelines.cache_aware_rnnt``; this fake exists so
    the scheduler is testable without a GPU, and must never be mistaken for it.
    """

    def __init__(
        self,
        chunk: ChunkMode,
        *,
        buckets: Sequence[int],
        step_ms: float = 1.0,
        edge_step_ms: float = 1.0,
        scripted: bool = False,
        script_for: ScriptSource | None = None,
        partial_every: int = 1,
        batch_leak: Sequence[str] = (),
        honour_phrases: bool = False,
        graphs: GraphCapability | None = None,
        graph_warmup_steps: int = UPSTREAM_WARMUP_STEPS,
    ) -> None:
        leak = frozenset(batch_leak)
        unknown = leak - BATCH_LEAKS
        if unknown:
            raise InvalidArgument(
                f"invalid batch_leak {sorted(unknown)!r}: must be a subset of {sorted(BATCH_LEAKS)}"
            )
        self._batch_leak = leak
        self._honour_phrases = bool(honour_phrases)
        #: A stream's own biasing digest, mixed into its own tokens and nothing else's.
        self._phrase_salt: dict[int, bytes] = {}
        #: Streams opened carrying a non-empty phrase list, counted for the life of the
        #: adapter. `_phrase_salt` is dropped on close, so a test that only read that
        #: could not tell a session that sent no list from one that has since ended --
        #: and "did this arm put phrases on the wire at all" is exactly what a harness
        #: with a deliberately unbiased control arm has to be able to check.
        self.phrase_sessions = 0
        self._chunk = chunk
        self._buckets = tuple(buckets)
        # What a tick "costs": an input to the scheduler's budget arithmetic,
        # never a measurement of any hardware.
        self._step_ms = step_ms
        self._edge_step_ms = edge_step_ms
        self._scripted = scripted
        self._script_for = script_for
        self._partial_every = partial_every
        self._graphs = (
            graphs
            if graphs is not None
            else GraphCapability.eager(
                "the CPU fake runs no graphs unless a test hands it a GraphCapability"
            )
        )
        if graph_warmup_steps < 1:
            raise InvalidArgument(f"graph_warmup_steps must be >= 1, got {graph_warmup_steps!r}")
        self._graph_warmup_steps = graph_warmup_steps
        self._key_counts: dict[GraphKey, int] = {}
        self._texts: dict[int, list[str]] = {}
        self._stubs: dict[int, ScriptedTranscript] = {}
        self._last_partial: dict[int, str] = {}
        self._shapes: list[tuple[int, bool]] = []
        self._calls_seen: list[tuple[int, bool, bool]] = []
        self._captures: list[GraphKey] = []
        self._replays: list[GraphKey] = []
        self._calls = 0
        self._eager_calls = 0

    @property
    def chunk(self) -> ChunkMode:
        return self._chunk

    def supported_buckets(self) -> tuple[int, ...]:
        return self._buckets

    def graph_capability(self) -> GraphCapability:
        return self._graphs

    def retained_graphs(self) -> int:
        """Distinct shapes captured so far -- the fake's stand-in for the count NeMo
        keeps in its private ``_graphs`` dict."""
        return len(set(self._captures))

    @property
    def captures(self) -> list[GraphKey]:
        """Shapes captured, in order: the first ``graph=True`` call at each key."""
        return list(self._captures)

    @property
    def replays(self) -> list[GraphKey]:
        """Every later ``graph=True`` call at an already-captured key, in order."""
        return list(self._replays)

    @property
    def step_ms(self) -> float:
        """Steady-step cost fed to the admission budget as a plain number."""
        return self._step_ms

    @property
    def edge_step_ms(self) -> float:
        """Per-edge-batch cost fed to the admission budget as a plain number."""
        return self._edge_step_ms

    @property
    def shapes_seen(self) -> list[tuple[int, bool]]:
        """Every batch stepped: ``(batch size, keep_all_outputs)``, in call order."""
        return list(self._shapes)

    @property
    def calls_seen(self) -> list[tuple[int, bool, bool]]:
        """Every batch stepped: ``(batch size, keep_all_outputs, graph)``, in call
        order. The third element is what the scheduler asked for, which is the whole
        question a test about the graph path is asking."""
        return list(self._calls_seen)

    @property
    def steps(self) -> int:
        """Total ``transcribe_step`` calls."""
        return self._calls

    @property
    def eager_steps(self) -> int:
        """Calls with ``keep_all_outputs=True`` (the edge path)."""
        return self._eager_calls

    def open_stream(self, stream_id: int, options: SessionOptions | None) -> None:
        """Hash mode ignores this (and the None options every tests/scheduler/ Session
        carries) unless ``honour_phrases`` is set, in which case it keeps this stream's
        own biasing digest. Scripted mode builds one ScriptedTranscript per stream;
        without options no script can be chosen, so that is an InvalidArgument, said
        loudly."""
        if options is not None and options.phrases:
            self.phrase_sessions += 1
            if self._honour_phrases:
                self._phrase_salt[stream_id] = options.biasing_digest.encode()
        if not self._scripted:
            return
        if options is None:
            raise InvalidArgument(
                "scripted FakePipelineAdapter.open_stream requires SessionOptions "
                f"(stream {stream_id}): a script cannot be chosen without them"
            )
        script = self._script_for(options) if self._script_for is not None else None
        self._stubs[stream_id] = ScriptedTranscript(
            options, script=script, partial_every=self._partial_every
        )

    def close_stream(self, stream_id: int) -> None:
        """Drop per-stream state after the tick loop has emitted its last row."""
        self._stubs.pop(stream_id, None)
        self._last_partial.pop(stream_id, None)
        self._texts.pop(stream_id, None)
        self._phrase_salt.pop(stream_id, None)

    @staticmethod
    def _token(frame: PcmFrame, *, salt: bytes = b"", phrase_salt: bytes = b"") -> str:
        """One deterministic word per row: a stable hash of the row's own sample bytes.

        Independent of the batch it was computed in and of the row's position in it.
        A frame with no valid samples carries no audio and contributes no token, so a
        padded tail never changes the running text. `salt` is empty except under the
        text leak, where it carries the batch composition the token must not depend on.
        """
        if frame.valid_samples <= 0:
            return ""
        digest = hashlib.sha256(
            np.ascontiguousarray(frame.samples, dtype=np.float32).tobytes() + salt + phrase_salt
        ).hexdigest()[:8]
        return f"w{digest}"

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool, graph: bool
    ) -> list[StepResult]:
        self._calls += 1
        if keep_all_outputs:
            self._eager_calls += 1
        self._shapes.append((len(frames), keep_all_outputs))
        self._calls_seen.append((len(frames), keep_all_outputs, graph))
        if graph:
            if not self._graphs.available:
                raise GraphPathUnavailable(
                    f"a batch of {len(frames)} rows was handed to the graph path and this "
                    f"adapter has none: {self._graphs.reason}"
                )
            key = GraphKey(
                chunk_ms=self._chunk.ms,
                batch_size=len(frames),
                keep_all_outputs=keep_all_outputs,
            )
            if key in self._captures:
                self._replays.append(key)
            else:
                # Upstream's rule, verbatim: the count has to exceed the warm-up steps,
                # so a warm-up of exactly that many leaves nothing captured.
                self._key_counts[key] = self._key_counts.get(key, 0) + 1
                if self._key_counts[key] > self._graph_warmup_steps:
                    self._captures.append(key)
        if self._scripted:
            return [self._scripted_row(frame) for frame in frames]
        # The batch composition a row's output must not depend on: how many real streams
        # share the batch. Pad rows carry samples too, so they are told apart by their
        # negative stream id, not by their valid count. Only a leak reads this.
        real_rows = sum(1 for frame in frames if frame.stream_id >= 0)
        salt = real_rows.to_bytes(2, "little") if "text" in self._batch_leak else b""
        timing_shift_ms = real_rows - 1 if "timing" in self._batch_leak else 0
        chunk_ms = self._chunk.ms
        results: list[StepResult] = []
        for frame in frames:
            history = self._texts.setdefault(frame.stream_id, [])
            token = self._token(
                frame, salt=salt, phrase_salt=self._phrase_salt.get(frame.stream_id, b"")
            )
            if token:
                history.append(token)
            partial = " ".join(history)
            words: tuple[Word, ...] = ()
            if frame.is_last:
                # Chunk-indexed, exact integers: token j spans chunk j.
                words = tuple(
                    Word(word=w, start_ms=j * chunk_ms, end_ms=(j + 1) * chunk_ms + timing_shift_ms)
                    for j, w in enumerate(history)
                )
            results.append(
                StepResult(
                    stream_id=frame.stream_id,
                    tick_id=self._calls,
                    partial_text=partial,
                    final_text=partial if frame.is_last else None,
                    # Left at the default: the tick loop stamps the session's own
                    # audio clock onto every row, and an adapter that stamped its own
                    # could not tell real bytes from invented padding.
                    audio_processed_s=0.0,
                    eager=keep_all_outputs,
                    words=words,
                )
            )
        return results

    def _scripted_row(self, frame: PcmFrame) -> StepResult:
        """One row through the per-stream ScriptedTranscript. Pad rows have no script
        and no transcript; their results are dropped by the tick loop either way."""
        stub = self._stubs.get(frame.stream_id)
        if stub is None:
            return StepResult(
                stream_id=frame.stream_id,
                tick_id=self._calls,
                partial_text="",
                final_text=None,
                audio_processed_s=0.0,
                eager=False,
            )
        partial = self._last_partial.get(frame.stream_id, "")
        if frame.valid_samples > 0:
            for hypo in stub.add_chunk(b"\x00" * stub.options.chunk_bytes):
                partial = hypo.text
            self._last_partial[frame.stream_id] = partial
        final_text: str | None = None
        words: tuple[Word, ...] = ()
        if frame.is_last:
            # A zero-valid drain frame calls only finalize(): the tail shorter than a
            # chunk still gets its scripted final, exactly as the script source does.
            final = stub.finalize()[0]
            final_text = final.text
            words = final.words
        return StepResult(
            stream_id=frame.stream_id,
            tick_id=self._calls,
            partial_text=partial,
            final_text=final_text,
            # The stub's own audio_processed_s is discarded here: the tick loop stamps
            # the session's clock onto every row.
            audio_processed_s=0.0,
            eager=False,
            words=words,
        )
