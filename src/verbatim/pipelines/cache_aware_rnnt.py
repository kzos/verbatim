# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Adapter onto NeMo's ``CacheAwareRNNTPipeline``. The MVP path.

What the adapter owns, and what it leaves to NeMo:

- NeMo's ``BasePipeline.transcribe_step`` creates a stream's state on ``is_first``
  and deletes it on ``is_last``; its bufferer and context manager allocate a slot on
  a stream's first frame and free it on the last. ``open_stream`` calls
  ``init_state`` directly, so a stream's state exists before its first frame ever
  reaches a batch, and the first frame is still flagged ``is_first`` so NeMo's own
  bookkeeping agrees. A frame for a stream that was never opened is refused before
  NeMo sees it; the tick loop guarantees it never sends one.
- Pad rows are streams to NeMo too. In the steady batch they are opened once and
  never end, so they hold their slots for the life of the process, which is what
  keeps the steady shape fixed. In an edge batch they are one-shot: ``is_first`` and
  ``is_last`` on the same frame, so NeMo puts them in the same final sub-batch as
  the real finals and that sub-batch keeps the edge shape too.
- Endpointing is NeMo's. ``stop_history_eou`` rides on ``ASRRequestOptions`` per
  stream, and ``TranscribeStepOutput.final_transcript`` is non-empty on the step
  where NeMo detected an end of utterance. That becomes ``StepResult.final_text``,
  so a final reaches the wire without a half-close.
- The adapter never sets ``audio_processed_s``, ``valid_samples`` or ``is_last`` on
  a ``StepResult``; the tick loop stamps all three from the session and the frame.
- ``step_ms`` and ``edge_step_ms`` are the wall time of the last steady and edge
  step, measured, so the admission controller's budget arithmetic runs on real
  numbers. The tick loop reads them after the step.
- The record of which streams NeMo holds is written after the step, from what NeMo
  did rather than what it was asked. NeMo raises from the encoder, after its
  bufferer and context manager allocated and after the bufferer freed a final
  frame's slot, but before the context manager freed its slot or the state was
  deleted. A step that raises therefore leaves every stream in the batch holding
  something, and ``close_stream`` must release it; bookkeeping written before the
  step said the final stream was gone and stranded its context slot.

NeMo is reached through ``NeMoBoundary``, one object and four callables, so the CPU
suite can stand a fake at exactly NeMo's seam (``verbatim.pipelines.nemo_fake``) and
this module imports neither ``torch`` nor ``nemo`` at import time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from verbatim.config import SAMPLE_RATE_HZ, ChunkMode
from verbatim.core.errors import InvalidArgument
from verbatim.core.types import PcmFrame, StepResult, Word
from verbatim.pipelines.base import PipelineAdapter
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.graph_budget import ConfigError

__all__ = ["CacheAwarePipelineLike", "CacheAwareRNNTAdapter", "NeMoBoundary"]


class CacheAwarePipelineLike(Protocol):
    """The part of NeMo's ``CacheAwareRNNTPipeline`` the adapter touches.

    ``verbatim.pipelines.nemo_fake.FakeCacheAwarePipeline`` implements the same
    surface without NeMo.
    """

    chunk_size_in_secs: float
    sample_rate: int
    num_slots: int

    def transcribe_step(self, requests: list[Any]) -> list[Any]: ...
    def init_state(self, stream_id: int, options: Any) -> Any: ...
    def delete_state(self, stream_id: int) -> None: ...
    def get_state(self, stream_id: int) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class NeMoBoundary:
    """Everything the adapter needs from NeMo, injected so the seam can be faked.

    ``make_frame`` builds a ``Frame`` from the keywords ``samples``, ``stream_id``,
    ``is_first``, ``is_last``, ``length`` and ``options``. ``make_options`` builds an
    ``ASRRequestOptions`` from ``stop_history_eou`` and ``language_code``.
    ``to_samples`` turns a float32 numpy chunk into what ``Frame.samples`` expects,
    a CPU ``torch.Tensor`` for NeMo. ``release_stream`` frees the bufferer and
    context slots of a stream that was stepped but never sent an ``is_last`` frame,
    which is what a failed live session looks like; NeMo frees those slots only on
    ``is_last``, so without it a failed stream's slot would leak.
    """

    pipeline: CacheAwarePipelineLike
    make_frame: Callable[..., Any]
    make_options: Callable[..., Any]
    to_samples: Callable[[np.ndarray], Any]
    release_stream: Callable[[int], None] | None = None

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> NeMoBoundary:
        """Bind a built ``CacheAwareRNNTPipeline``.

        Imports torch and NeMo here and nowhere else on this package's import path,
        so the CPU suite never needs either.
        """
        import torch
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import (
            ASRRequestOptions,
        )

        def to_samples(samples: np.ndarray) -> Any:
            return torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))

        def release_stream(stream_id: int) -> None:
            # Two slot tables, keyed differently, and one name that means two things:
            # the bufferer's `free_slots` is a method taking SLOT ids, while the context
            # manager's `free_slots` is its Queue of free slots. The context manager is
            # released through `reset_slots`, which takes STREAM ids and raises KeyError
            # for a stream it does not map, hence the membership guard.
            bufferer = pipeline.bufferer
            slot = bufferer.streamidx2slotidx.get(stream_id)
            if slot is not None:
                bufferer.free_slots([slot])
            context = pipeline.context_manager
            if stream_id in getattr(context, "streamidx2slotidx", {}):
                context.reset_slots([stream_id], [True])

        return cls(
            pipeline=pipeline,
            make_frame=Frame,
            make_options=ASRRequestOptions,
            to_samples=to_samples,
            release_stream=release_stream,
        )


def _words_of(segments: Sequence[Any] | None) -> tuple[Word, ...]:
    """NeMo ``TextSegment`` records (seconds, float confidence) to integer-millisecond words."""
    words: list[Word] = []
    for segment in segments or ():
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        words.append(
            Word(
                word=text,
                start_ms=round(float(segment.start) * 1000),
                end_ms=round(float(segment.end) * 1000),
                confidence=float(segment.conf),
            )
        )
    return tuple(words)


class CacheAwareRNNTAdapter(PipelineAdapter):
    """Verbatim's ``PipelineAdapter`` over NeMo's cache-aware RNNT pipeline, one per
    chunk mode. See the module docstring for what it owns."""

    def __init__(
        self,
        chunk: ChunkMode,
        boundary: NeMoBoundary,
        *,
        buckets: Sequence[int],
        required_slots: int,
        stop_history_eou_ms: int = 800,
        language_code: str | None = None,
    ) -> None:
        pipeline = boundary.pipeline
        if abs(float(pipeline.chunk_size_in_secs) - chunk.period_s) > 1e-6:
            raise ConfigError(
                f"pipeline chunk size {float(pipeline.chunk_size_in_secs)!r}s does not match "
                f"the {chunk.ms} ms chunk mode ({chunk.period_s!r}s)"
            )
        if int(pipeline.sample_rate) != SAMPLE_RATE_HZ:
            raise ConfigError(
                f"pipeline sample rate {int(pipeline.sample_rate)} is not {SAMPLE_RATE_HZ}"
            )
        if int(pipeline.num_slots) < required_slots:
            raise ConfigError(
                f"pipeline has {int(pipeline.num_slots)} slots but the scheduler needs "
                f"{required_slots} (bucket, pad rows, edge batch and drain margin): NeMo "
                f"raises 'No free slots available' the first tick that runs short"
            )
        if stop_history_eou_ms < 0:
            raise ConfigError(f"stop_history_eou_ms must be >= 0, got {stop_history_eou_ms!r}")
        self._chunk = chunk
        self._boundary = boundary
        self._buckets = tuple(buckets)
        self._stop_history_eou_ms = int(stop_history_eou_ms)
        self._language_code = language_code
        # Real streams opened by open_stream, with the request options NeMo was given.
        self._opened: dict[int, Any] = {}
        # Streams, real or pad, that NeMo has seen at least once and not yet ended.
        self._started: set[int] = set()
        self._step_ms = 0.0
        self._edge_step_ms = 0.0

    @property
    def chunk(self) -> ChunkMode:
        return self._chunk

    def supported_buckets(self) -> tuple[int, ...]:
        return self._buckets

    @property
    def step_ms(self) -> float:
        """Wall time of the last steady step in milliseconds. Measured, not modelled."""
        return self._step_ms

    @property
    def edge_step_ms(self) -> float:
        """Wall time of the last edge step in milliseconds. Measured, not modelled."""
        return self._edge_step_ms

    @property
    def open_streams(self) -> frozenset[int]:
        """Real streams opened and not yet closed."""
        return frozenset(self._opened)

    def _request_options(self, options: SessionOptions | None) -> Any:
        eou = self._stop_history_eou_ms
        if options is not None and options.stop_history_eou_ms is not None:
            eou = options.stop_history_eou_ms
        language = self._language_code
        if language is None and options is not None:
            language = options.language_code
        return self._boundary.make_options(stop_history_eou=eou, language_code=language)

    def open_stream(self, stream_id: int, options: SessionOptions | None) -> None:
        """Create the stream's NeMo state now, on the tick thread, before its first frame.

        NeMo's ``init_state`` runs ``create_state``, which is where a bad per-stream
        option fails (a prompt-enabled model with a language code it does not know,
        for instance). Raising here fails this one session; raising inside a batch
        step would fail every session in it.
        """
        if stream_id < 0:
            raise InvalidArgument(f"pad row {stream_id} is opened by the adapter, not the loop")
        if stream_id in self._opened:
            raise InvalidArgument(f"stream {stream_id} is already open")
        request_options = self._request_options(options)
        self._boundary.pipeline.init_state(stream_id, request_options)
        self._opened[stream_id] = request_options

    def close_stream(self, stream_id: int) -> None:
        """Drop the stream. NeMo already deleted the state and freed the slots if the
        stream's ``is_last`` frame went through; otherwise release them here."""
        self._opened.pop(stream_id, None)
        if stream_id in self._started:
            self._started.discard(stream_id)
            if self._boundary.release_stream is not None:
                self._boundary.release_stream(stream_id)
        self._boundary.pipeline.delete_state(stream_id)

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        if not frames:
            return []
        requests = []
        ending: list[tuple[int, bool]] = []
        for frame in frames:
            stream_id = frame.stream_id
            if stream_id < 0:
                is_first = stream_id not in self._started
                # Edge path: NeMo peels finals into their own sub-batch, so a pad row
                # only keeps that sub-batch's shape if it is a final too. One-shot.
                is_last = keep_all_outputs
                if is_first:
                    self._boundary.pipeline.init_state(stream_id, self._request_options(None))
                request_options = self._request_options(None) if is_first else None
            else:
                request_options = self._opened.get(stream_id)
                if request_options is None:
                    raise InvalidArgument(
                        f"stream {stream_id} reached transcribe_step without open_stream; "
                        f"NeMo has no state for it and would dereference None"
                    )
                is_first = stream_id not in self._started
                is_last = frame.is_last
                if not is_first:
                    request_options = None
            requests.append(
                self._boundary.make_frame(
                    samples=self._boundary.to_samples(frame.samples),
                    stream_id=stream_id,
                    is_first=is_first,
                    is_last=is_last,
                    length=int(frame.valid_samples),
                    options=request_options,
                )
            )
            ending.append((stream_id, is_last))

        started_at = time.perf_counter()
        try:
            outputs = self._boundary.pipeline.transcribe_step(requests)
        except BaseException:
            # NeMo raised between allocating and freeing: every stream in the batch
            # may now hold a slot or a state, the final ones included. Record them
            # all as held so close_stream releases whatever is left.
            for stream_id, _ in ending:
                self._started.add(stream_id)
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        for stream_id, is_last in ending:
            if is_last:
                self._started.discard(stream_id)  # NeMo deleted the state, freed the slots
            else:
                self._started.add(stream_id)
        if keep_all_outputs:
            self._edge_step_ms = elapsed_ms
        else:
            self._step_ms = elapsed_ms

        if len(outputs) != len(frames):
            raise RuntimeError(f"pipeline returned {len(outputs)} outputs for {len(frames)} frames")
        results: list[StepResult] = []
        for frame, output in zip(frames, outputs, strict=True):
            if int(output.stream_id) != frame.stream_id:
                raise RuntimeError(
                    f"pipeline output for stream {output.stream_id} arrived in the row "
                    f"of stream {frame.stream_id}"
                )
            results.append(self._to_result(frame, output, eager=keep_all_outputs))
        return results

    @staticmethod
    def _to_result(frame: PcmFrame, output: Any, *, eager: bool) -> StepResult:
        """One NeMo step output to one ``StepResult``; the clock fields stay at their
        defaults for the tick loop to stamp."""
        final_transcript = str(output.final_transcript or "").strip()
        # NeMo's final_transcript is non-empty on the step it detected an end of
        # utterance. A last frame always ends with a final, empty if nothing was said.
        final_text = final_transcript if (final_transcript or frame.is_last) else None
        words: tuple[Word, ...] = ()
        confidence = 1.0
        if final_text is not None:
            words = _words_of(getattr(output, "final_segments", None))
            if words:
                confidence = sum(word.confidence for word in words) / len(words)
        return StepResult(
            stream_id=frame.stream_id,
            tick_id=0,
            partial_text=str(output.partial_transcript or "").strip(),
            final_text=final_text,
            audio_processed_s=0.0,
            eager=eager,
            words=words,
            confidence=confidence,
        )
