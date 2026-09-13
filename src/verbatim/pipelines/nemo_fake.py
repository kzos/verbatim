# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""A fake of NeMo's cache-aware pipelines at NeMo's own seam. No torch, no NeMo.

It stands where ``CacheAwareRNNTPipeline`` and ``CacheAwareCTCPipeline`` stand, so the
adapter's translation of frames, options and step outputs is exercised for real by the
CPU suite rather than restated. Both real pipelines inherit ``BasePipeline`` unchanged,
so one body is faithful to both; ``FakeCacheAwareRNNTPipeline`` and
``FakeCacheAwareCTCPipeline`` add the two places the seam can tell them apart.
What the body copies from NeMo, read from ``nemo.collections.asr.inference``:

- ``transcribe_step(requests)`` creates a stream's state on ``is_first`` and deletes
  it on ``is_last``; a request for a stream with no state dereferences ``None``, as
  NeMo does, so the never-opened hazard is real here too.
- A slot is taken on a stream's first sight and freed on ``is_last``; one more
  stream than ``num_slots`` raises ``RuntimeError("No free slots available")``.
- ``right padding`` is ``size - valid_size``: only the valid samples of a frame count.
- Endpointing: ``stop_history_eou`` milliseconds of silence after speech is an end
  of utterance, and a last frame always is. On that step ``final_transcript`` and
  ``final_segments`` carry the utterance and ``partial_transcript`` resets.

Recognition is scripted from the audio itself: every non-silent frame contributes
one word derived from a hash of its valid samples, so a transcript is a pure
function of the stream's own audio and never of its neighbours. That is the
property the real pipeline must have; here it holds by construction.

This is a test double that ships in the package on purpose, like the CPU fake
adapter: the adapter must be testable by anyone, on any machine, without a GPU.

``FakeCacheAwarePipeline`` itself declares neither pipeline's greedy decoder, because
that is also a real case the adapters must accept: a third party's pipeline, or a
double, that says nothing about which of the two it is. The crossed-wire guard only
fires on a pipeline that positively declares the sibling's.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verbatim.pipelines.cache_aware import NeMoBoundary

__all__ = [
    "FakeCacheAwareCTCPipeline",
    "FakeCacheAwarePipeline",
    "FakeCacheAwareRNNTPipeline",
    "FakeFrame",
    "FakeRequestOptions",
    "FakeStepOutput",
    "FakeTextSegment",
    "boundary_for",
]


@dataclass(frozen=True, slots=True)
class FakeFrame:
    """Shaped like ``nemo...framing.request.Frame``; samples stay a numpy array."""

    samples: np.ndarray
    stream_id: int
    is_first: bool = False
    is_last: bool = False
    length: int = -1
    vad_segments: Any = None
    options: Any = None

    @property
    def size(self) -> int:
        return int(np.asarray(self.samples).shape[0])

    @property
    def valid_size(self) -> int:
        return self.size if self.length == -1 else self.length


@dataclass(slots=True)
class FakeRequestOptions:
    """Shaped like ``ASRRequestOptions``: ``None`` means the pipeline default."""

    enable_itn: bool | None = None
    stop_history_eou: int | None = None
    asr_output_granularity: Any = None
    language_code: str | None = None


@dataclass(slots=True)
class FakeTextSegment:
    """Shaped like ``TextSegment``: seconds, float confidence."""

    text: str
    start: float
    end: float
    conf: float


@dataclass(slots=True)
class FakeStepOutput:
    """Shaped like ``TranscribeStepOutput``."""

    stream_id: int
    final_transcript: str = ""
    final_segments: list[FakeTextSegment] = field(default_factory=list)
    partial_transcript: str = ""
    current_step_transcript: str = ""


@dataclass(slots=True)
class _State:
    options: FakeRequestOptions
    words: list[tuple[str, float, float]] = field(default_factory=list)
    frames_seen: int = 0
    silence_ms: float = 0.0


class FakeCacheAwarePipeline:
    """The fake pipeline. Records every request it was handed in ``seen``."""

    #: What NeMo dereferences on a state that was never created. The RNNT pipeline
    #: reaches ``state.get_previous_hypothesis()`` before the encoder step; the CTC
    #: one reaches ``state.label_buffer`` in ``decode_log_probs``, after it. Either
    #: way it is an ``AttributeError`` on ``None``, and either way the adapter
    #: refuses the frame before NeMo sees it.
    missing_state_attribute = "get_previous_hypothesis"

    def __init__(
        self,
        chunk_ms: int = 160,
        *,
        sample_rate: int = 16000,
        num_slots: int = 64,
        stop_history_eou: int = 800,
    ) -> None:
        self.chunk_size_in_secs = chunk_ms / 1000
        self.sample_rate = sample_rate
        self.num_slots = num_slots
        self.stop_history_eou_in_milliseconds = stop_history_eou
        self._state_pool: dict[int, _State] = {}
        self._slots: set[int] = set()
        self.seen: list[FakeFrame] = []
        self.released: list[int] = []
        self.step_calls = 0
        #: Set to an exception to raise it where NeMo's encoder raises: after the
        #: bufferer and context manager allocated, before either cleaned up.
        self.raise_in_encoder: BaseException | None = None
        #: Shapes this fake has "captured", standing in for the private ``_graphs``
        #: dict upstream's ``CudaGraphsStreamingEncoderStep`` keeps. It follows
        #: upstream's rule -- a shape is captured on the call whose count at that
        #: shape exceeds ``graph_warmup_steps`` -- so the count a boundary reads back
        #: here moves when and only when upstream's would.
        self.graph_warmup_steps = 3
        self._graph_key_counts: dict[tuple[int, bool], int] = {}
        self._graphs: set[tuple[int, bool]] = set()

    def retained_graphs(self) -> int:
        """How many shapes this fake is holding, as NeMo's own step would report."""
        return len(self._graphs)

    def note_graph_step(self, batch_size: int, keep_all_outputs: bool) -> None:
        """Record a step upstream would have considered for capture."""
        key = (batch_size, keep_all_outputs)
        if key in self._graphs or keep_all_outputs:
            return
        self._graph_key_counts[key] = self._graph_key_counts.get(key, 0) + 1
        if self._graph_key_counts[key] > self.graph_warmup_steps:
            self._graphs.add(key)

    # --- the state pool, as BasePipeline has it ---

    def init_state(self, stream_id: int, options: Any) -> _State:
        if stream_id not in self._state_pool:
            stop = self.stop_history_eou_in_milliseconds
            language = None
            if options is not None:
                if options.stop_history_eou is not None:
                    stop = int(options.stop_history_eou)
                language = options.language_code
            self._state_pool[stream_id] = _State(
                FakeRequestOptions(stop_history_eou=stop, language_code=language)
            )
        return self._state_pool[stream_id]

    def get_state(self, stream_id: int) -> _State | None:
        return self._state_pool.get(stream_id)

    def delete_state(self, stream_id: int) -> None:
        self._state_pool.pop(stream_id, None)

    @property
    def live_slots(self) -> int:
        return len(self._slots)

    def release_stream(self, stream_id: int) -> None:
        """What ``NeMoBoundary.release_stream`` does against NeMo's bufferer and context."""
        self._slots.discard(stream_id)
        self.released.append(stream_id)

    # --- the step ---

    @staticmethod
    def _word_of(valid: np.ndarray) -> str:
        digest = hashlib.sha256(np.ascontiguousarray(valid, dtype=np.float32).tobytes())
        return "w" + digest.hexdigest()[:6]

    def transcribe_step(self, requests: list[FakeFrame]) -> list[FakeStepOutput]:
        if len(requests) == 0:
            raise IndexError("list index out of range")  # NeMo indexes requests[0]
        self.step_calls += 1
        # Upstream considers every step for capture; a final sub-batch is never captured.
        self.note_graph_step(len(requests), any(r.is_last for r in requests))
        states: list[_State | None] = []
        for request in requests:
            if request.is_first:
                self.init_state(request.stream_id, request.options)
            states.append(self.get_state(request.stream_id))
        for request in requests:
            if request.stream_id not in self._slots:
                if len(self._slots) >= self.num_slots:
                    raise RuntimeError("No free slots available")
                self._slots.add(request.stream_id)
        if self.raise_in_encoder is not None:
            # NeMo's encoder step: after allocation, before any of the cleanup below.
            raise self.raise_in_encoder

        outputs: list[FakeStepOutput] = []
        chunk_ms = self.chunk_size_in_secs * 1000.0
        for request, state in zip(requests, states, strict=True):
            self.seen.append(request)
            if state is None:
                raise AttributeError(
                    f"'NoneType' object has no attribute '{self.missing_state_attribute}'"
                )
            valid = np.asarray(request.samples, dtype=np.float32)[: request.valid_size]
            start = state.frames_seen * self.chunk_size_in_secs
            state.frames_seen += 1
            if valid.size and bool(np.any(valid != 0)):
                state.words.append((self._word_of(valid), start, start + self.chunk_size_in_secs))
                state.silence_ms = 0.0
            else:
                state.silence_ms += chunk_ms
            stop = state.options.stop_history_eou or 0
            eou = request.is_last or (bool(state.words) and stop > 0 and state.silence_ms >= stop)
            final_transcript = ""
            final_segments: list[FakeTextSegment] = []
            if eou and state.words:
                final_transcript = " ".join(word for word, _, _ in state.words)
                final_segments = [
                    FakeTextSegment(word, begin, end, 0.9) for word, begin, end in state.words
                ]
                state.words = []
                state.silence_ms = 0.0
            outputs.append(
                FakeStepOutput(
                    stream_id=request.stream_id,
                    final_transcript=final_transcript,
                    final_segments=final_segments,
                    partial_transcript=" ".join(word for word, _, _ in state.words),
                    current_step_transcript=state.words[-1][0] if state.words else "",
                )
            )
        for request in requests:
            if request.is_last:
                self.delete_state(request.stream_id)
                self._slots.discard(request.stream_id)
        return outputs


class FakeCacheAwareRNNTPipeline(FakeCacheAwarePipeline):
    """The body, declaring itself RNNT the way NeMo's does.

    ``CacheAwareRNNTPipeline.init_greedy_rnnt_decoder`` sets ``greedy_rnnt_decoder``;
    that attribute is the only thing at this seam that says which of the two
    pipelines a boundary is bound to, so the fake carries it rather than a flag of
    Verbatim's invention.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.greedy_rnnt_decoder = object()


class FakeCacheAwareCTCPipeline(FakeCacheAwarePipeline):
    """The body, declaring itself CTC the way NeMo's does.

    ``CacheAwareCTCPipeline.init_greedy_ctc_decoder`` sets ``greedy_ctc_decoder``, and
    its ``create_state`` never resolves a prompt index, so a ``language_code`` is
    recorded on the options and read by nothing -- which is what the body already
    does, so nothing here has to undo a prompt path that does not exist.
    """

    missing_state_attribute = "label_buffer"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.greedy_ctc_decoder = object()


def boundary_for(
    pipeline: FakeCacheAwarePipeline,
    *,
    graph_step_available: bool = False,
    step_attached: bool | None = None,
) -> NeMoBoundary:
    """The adapter's boundary bound to the fake: numpy samples pass through untouched.

    ``graph_step_available`` stands in for the runtime probe of the installed *package*.
    False is what every released NeMo wheel and this machine report; true is the
    source-built track of NeMo PR #15863, which the CPU suite has to be able to express
    precisely because it is the arm that cannot be run here.

    ``step_attached`` is the separate fact: whether *this pipeline object* has a graphed
    step on it, which is what a retained-graph count can be read from. It defaults to
    following ``graph_step_available``, because a package that has the class normally
    attaches it. Passing ``step_attached=False`` with ``graph_step_available=True`` is
    the crossed case -- the package can graph and this object cannot -- which is the CTC
    pipeline's situation if upstream's CTC branch never calls the switch.
    """
    attached = graph_step_available if step_attached is None else step_attached
    return NeMoBoundary(
        pipeline=pipeline,
        make_frame=FakeFrame,
        make_options=FakeRequestOptions,
        to_samples=lambda samples: np.asarray(samples, dtype=np.float32),
        release_stream=pipeline.release_stream,
        graph_step_available=graph_step_available,
        retained_graphs=pipeline.retained_graphs if attached else None,
    )
