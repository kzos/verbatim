# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""NeMo's cache-aware seam, and the stream lifecycle both cache-aware adapters drive.

Why one body serves RNNT and CTC, and where they genuinely diverge:

- NeMo puts the whole per-stream lifecycle in ``BasePipeline``, which both
  ``CacheAwareRNNTPipeline`` and ``CacheAwareCTCPipeline`` inherit unchanged:
  ``transcribe_step`` creates a stream's state on ``is_first``, deletes it on
  ``is_last`` and builds one ``TranscribeStepOutput`` per request; the bufferer and
  the ``CacheAwareContextManager`` are the same two objects, allocated by the same
  two ``init_*`` calls; and each pipeline's ``transcribe_step_for_frames`` peels
  ``is_last`` rows into their own ``keep_all_outputs=True`` sub-batch the same way.
  At this seam the two are one API, so the adapter that drives it is one body.
- What differs is entirely *behind* the seam and never crosses it. RNNT carries a
  per-stream decoder state -- ``previous_hypotheses`` in and out of every
  ``stream_step``, reset on ``is_last`` -- plus per-stream biasing and, for a
  prompt-conditioned checkpoint, a per-stream prompt vector built from the request's
  ``language_code``. CTC carries none of the three: its ``stream_step`` takes
  features and context and returns log probabilities, its ``create_state`` never
  resolves a prompt index, and ``CacheAwarePipelineBuilder`` builds its
  ``CTCDecodingConfig`` itself rather than reading ``asr.decoding``. So this module
  models no decoder state at all. There is no per-stream decoder abstraction here to
  make the two look alike, because at this seam nothing decoder-shaped is exchanged.

What every adapter over this seam owns:

- ``open_stream`` calls ``init_state`` directly, so a stream's state exists before
  its first frame ever reaches a batch, and the first frame is still flagged
  ``is_first`` so NeMo's own bookkeeping agrees. A frame for a stream that was never
  opened is refused before NeMo sees it; the tick loop guarantees it never sends one.
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
- A session's biasing phrases, when the server serves them, become one
  ``BiasingRequestItemConfig`` on its ``ASRRequestOptions``, built in ``open_stream``
  and released in ``close_stream``. Every way this can fail quietly is refused
  instead: a pipeline that cannot bias, a decoder built without the arena, a boundary
  that cannot release an entry, and a session that sends phrases to a server without
  biasing. An unbiased transcript is indistinguishable from a biased one, so nothing
  here is allowed to fall back.
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
from typing import Any, ClassVar, Final, Protocol

import numpy as np

from verbatim.config import SAMPLE_RATE_HZ, ChunkMode
from verbatim.core.errors import InvalidArgument
from verbatim.core.types import PcmFrame, StepResult, Word
from verbatim.pipelines.base import (
    GraphCapability,
    GraphPathUnavailable,
    PipelineAdapter,
)
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.graph_budget import ConfigError

__all__ = ["CacheAwareAdapter", "CacheAwarePipelineLike", "NeMoBoundary"]


class CacheAwarePipelineLike(Protocol):
    """The part of NeMo's cache-aware pipelines the adapter touches.

    Every member of it is declared on ``BasePipeline`` or set by the ``init_*`` calls
    both cache-aware pipelines make, so one Protocol covers RNNT and CTC.
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

    One boundary type covers both cache-aware pipelines: ``Frame``,
    ``ASRRequestOptions``, the bufferer and the context manager are the same classes
    on either, so ``from_pipeline`` binds an RNNT or a CTC pipeline unchanged. Which
    of the two it is, is the adapter's business, not the boundary's.

    ``graph_step_available`` is the runtime fact behind ``graph_capability()``:
    whether the installed NeMo carries both halves of PR #15863. It is probed once,
    where the boundary is bound, rather than assumed from a version number -- see
    ``docs/decisions/0002-the-graph-path-is-not-in-a-released-wheel.md``, where a
    version number would have got it wrong twice. The probe reads the cache-aware
    RNNT wrapper for either pipeline: it answers which NeMo track is installed, and
    where PR #15863 puts the switch for the CTC wrapper cannot be read from an
    installed wheel here.

    ``retained_graphs`` reads how many graphs NeMo is holding, so warm-up can check
    that a capture happened rather than assume it. It reads a private attribute of
    upstream's ``CudaGraphsStreamingEncoderStep`` because upstream exposes no public
    count, and it is bound to None when the attribute is not where this expects --
    a boundary that cannot tell says so, and is not read as zero.

    ``make_biasing`` builds one session's ``BiasingRequestItemConfig`` from its phrase
    list; ``release_biasing`` drops a stream's entry from the decoder's biasing arena.
    Both are None when the installed NeMo has no per-stream biasing, and an adapter
    asked to serve biasing over such a boundary refuses to start rather than accept
    phrase lists it cannot honour. ``release_biasing`` is separate from
    ``release_stream`` because NeMo releases the arena entry only on an ``is_last``
    frame inside ``transcribe_step``: a session that aborts without a final leaks its
    entry for the life of the process, and ``delete_state`` does not free it.
    """

    pipeline: CacheAwarePipelineLike
    make_frame: Callable[..., Any]
    make_options: Callable[..., Any]
    to_samples: Callable[[np.ndarray], Any]
    release_stream: Callable[[int], None] | None = None
    graph_step_available: bool = False
    retained_graphs: Callable[[], int] | None = None
    make_biasing: Callable[..., Any] | None = None
    release_biasing: Callable[[int], None] | None = None

    @classmethod
    def from_pipeline(
        cls, pipeline: Any, *, graph_step_available: bool | None = None
    ) -> NeMoBoundary:
        """Bind a built ``CacheAwareRNNTPipeline`` or ``CacheAwareCTCPipeline``.

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

        if graph_step_available is None:
            from verbatim.pipelines.nemo_runtime import graph_step_present

            graph_step_available = graph_step_present()
        retained_graphs = _retained_graphs_reader(pipeline)
        make_biasing, release_biasing = _biasing_seam(pipeline)
        return cls(
            pipeline=pipeline,
            make_frame=Frame,
            make_options=ASRRequestOptions,
            to_samples=to_samples,
            release_stream=release_stream,
            graph_step_available=graph_step_available,
            retained_graphs=retained_graphs,
            make_biasing=make_biasing,
            release_biasing=release_biasing,
        )


#: Where ``StreamingEncoder.set_streaming_cuda_graphs`` attaches the graphed step, read
#: from NeMo's own source: ``streaming.py`` sets ``_stream_step_cuda_graphs`` on the
#: encoder, and the cache-aware inference wrapper reaches the encoder through
#: ``asr_model.encoder``. Both nestings are tried because the wrapper's own attribute is
#: itself called ``asr_model`` and which one a pipeline hands back has changed upstream
#: before.
_GRAPH_STEP_PATHS = (
    ("asr_model", "asr_model", "encoder", "_stream_step_cuda_graphs"),
    ("asr_model", "encoder", "_stream_step_cuda_graphs"),
    ("encoder", "_stream_step_cuda_graphs"),
)


def _retained_graphs_reader(pipeline: Any) -> Callable[[], int] | None:
    """A reader for the number of CUDA graphs NeMo is holding, or None if it is not
    where this expects.

    Upstream keeps them in a private ``_graphs`` dict and exposes no count. Reading a
    private attribute is the cost of being able to check a capture happened; the
    alternative measured worse -- a warm-up that recorded a capture NeMo had not made
    (docs/decisions/0011).
    """
    for path in _GRAPH_STEP_PATHS:
        step: Any = pipeline
        for part in path:
            step = getattr(step, part, None)
            if step is None:
                break
        if step is None:
            continue
        graphs = getattr(step, "_graphs", None)
        if graphs is None:
            continue

        def reader(step: Any = step) -> int:
            return len(step._graphs)

        return reader
    return None


#: How a session's phrases are tokenised into its boosting tree. Case-insensitive, so a
#: client sending "Metformin" also boosts "metformin"; the transcript's own casing is the
#: model's. Set per request because NeMo's ``asr.per_stream_biasing_defaults`` block is
#: read only by the offline manifest path and never by the pipeline builder.
BIASING_BPE_MODE: Final = "case_insensitive"
#: The tokenizer language for phrases, used only by an aggregate (multilingual) tokenizer.
BIASING_SOURCE_LANG: Final = "en"


def _biasing_seam(
    pipeline: Any,
) -> tuple[Callable[..., Any] | None, Callable[[int], None] | None]:
    """``(make_biasing, release_biasing)``, or ``(None, None)`` when this pipeline has no
    per-stream biasing arena.

    Both are bound only when the *built decoder* actually carries the biasing
    multi-model -- not when the installed package merely has the classes. A pipeline
    built with ``enable_per_stream_biasing`` false has no arena, and NeMo logs a warning
    and decodes unbiased for any stream that carries a request. Returning None here is
    what lets the adapter refuse to start instead.

    ``cache_key`` is deliberately never set on the config this builds. NeMo's phrase-tree
    cache is a module-level dictionary with no eviction and no tenant scope; a shared key
    would hand one caller's vocabulary to another, which for the buyer this server is for
    is a disclosure, not an optimisation. The tree is rebuilt per session, and what that
    costs a tick is measured rather than assumed.
    """
    try:
        from nemo.collections.asr.inference.utils.per_stream_biasing import (
            release_auto_managed_stream_biasing,
        )
        from nemo.collections.asr.parts.context_biasing.biasing_multi_model import (
            BiasingRequestItemConfig,
        )
        from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
            BoostingTreeModelConfig,
            PhraseItem,
        )
    except Exception:
        return None, None

    computer = getattr(pipeline, "decoding_computer", None)
    if computer is None or not getattr(computer, "per_stream_biasing_enabled", False):
        return None, None
    if getattr(computer, "biasing_multi_model", None) is None:
        return None, None

    def make_biasing(
        phrases: Sequence[tuple[str, float | None]], alpha: float | None = None
    ) -> Any:
        items = [PhraseItem(phrase=text, alpha=boost) for text, boost in phrases]
        return BiasingRequestItemConfig(
            boosting_model_cfg=BoostingTreeModelConfig(
                key_phrase_items_list=items,
                source_lang=BIASING_SOURCE_LANG,
                bpe_mode=BIASING_BPE_MODE,
            ),
            boosting_model_alpha=1.0 if alpha is None else float(alpha),
            cache_key=None,
            auto_manage_multi_model=True,
        )

    def release_biasing(stream_id: int) -> None:
        state = pipeline.get_state(stream_id)
        if state is None or not state.has_biasing_request():
            return
        release_auto_managed_stream_biasing(state, computer.biasing_multi_model)

    return make_biasing, release_biasing


def _words_of(segments: Sequence[Any] | None) -> tuple[Word, ...]:
    """NeMo ``TextSegment`` records (seconds, float confidence) to integer-millisecond words.

    Both pipelines fill ``final_segments`` through the same ``BPEDecoder``, so one
    conversion serves both.
    """
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


class CacheAwareAdapter(PipelineAdapter):
    """The lifecycle every cache-aware adapter drives, one per chunk mode.

    Subclasses name the pipeline they belong to and nothing else; see the module
    docstring for why there is nothing else for them to carry.
    """

    #: What this pipeline is called in an operator-facing message.
    kind: ClassVar[str] = "cache-aware"
    #: The registered pipeline name that builds this adapter.
    registry_name: ClassVar[str] = ""
    #: NeMo's attribute for this pipeline's greedy decoder, set by its own
    #: ``init_greedy_*_decoder``, and the sibling pipeline's. A boundary carrying the
    #: sibling's decoder and not this one is a crossed wire: it would run, transcribe
    #: and reach a published row under the wrong name, so it is refused instead. A
    #: pipeline declaring neither -- a third party's, or a test double -- is accepted,
    #: because there is then nothing to contradict.
    decoder_attribute: ClassVar[str] = ""
    sibling_decoder_attribute: ClassVar[str] = ""
    sibling_registry_name: ClassVar[str] = ""
    #: Whether this pipeline can carry a per-session boosting tree at all. NeMo reaches
    #: biasing through the RNNT decoding computer and the cache-aware CTC pipeline has no
    #: equivalent, so the CTC adapter sets this false and refuses rather than accepting
    #: phrase lists and transcribing every session unbiased.
    supports_biasing: ClassVar[bool] = True

    def __init__(
        self,
        chunk: ChunkMode,
        boundary: NeMoBoundary,
        *,
        buckets: Sequence[int],
        required_slots: int,
        stop_history_eou_ms: int = 800,
        language_code: str | None = None,
        use_cuda_graphs: bool = False,
        biasing: bool = False,
    ) -> None:
        pipeline = boundary.pipeline
        self._refuse_a_crossed_wire(pipeline)
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
        if biasing:
            self._refuse_unservable_biasing(boundary)
        self._biasing = bool(biasing)
        self._chunk = chunk
        self._boundary = boundary
        self._buckets = tuple(buckets)
        self._stop_history_eou_ms = int(stop_history_eou_ms)
        self._language_code = language_code
        self._use_cuda_graphs = bool(use_cuda_graphs)
        # Real streams opened by open_stream, with the request options NeMo was given.
        self._opened: dict[int, Any] = {}
        # Streams, real or pad, that NeMo has seen at least once and not yet ended.
        self._started: set[int] = set()
        self._step_ms = 0.0
        self._edge_step_ms = 0.0

    @classmethod
    def _refuse_unservable_biasing(cls, boundary: NeMoBoundary) -> None:
        """Refuse to start a biasing server that could not actually bias.

        Three ways it fails, all of them silent if unchecked: the pipeline has no biasing
        path at all (CTC); NeMo's decoder was built without the biasing multi-model, in
        which case it logs a warning per step and decodes unbiased; or the arena entry a
        session takes could not be released, which leaks one entry per aborted session for
        the life of the process. In each case the server would accept a phrase list, return
        a transcript, and give the client no way to tell it was never biased.
        """
        if not cls.supports_biasing:
            raise ConfigError(
                f"per-stream biasing was asked for and {cls.registry_name} cannot serve it: "
                "NeMo reaches biasing through the RNNT decoding computer and the cache-aware "
                f"CTC pipeline has no equivalent. Serve --pipeline {cls.sibling_registry_name} "
                "or start without --biasing"
            )
        if boundary.make_biasing is None:
            raise ConfigError(
                "per-stream biasing was asked for and this built pipeline has no biasing "
                "arena: NeMo's decoding computer reports per_stream_biasing_enabled false, "
                "so it was built without asr.decoding.greedy.enable_per_stream_biasing. It "
                "would log a warning per step and decode every session unbiased"
            )
        if boundary.release_biasing is None:
            raise ConfigError(
                "per-stream biasing was asked for and this boundary cannot release a "
                "stream's arena entry: NeMo frees one only on an is_last frame, so every "
                "session that aborts would leak an entry for the life of the process"
            )

    @classmethod
    def _refuse_a_crossed_wire(cls, pipeline: Any) -> None:
        """Refuse a boundary bound to the sibling cache-aware pipeline."""
        if not cls.sibling_decoder_attribute:
            return
        if hasattr(pipeline, cls.sibling_decoder_attribute) and not hasattr(
            pipeline, cls.decoder_attribute
        ):
            raise ConfigError(
                f"the bound pipeline has {cls.sibling_decoder_attribute!r} and no "
                f"{cls.decoder_attribute!r}: it is a {cls.sibling_registry_name} pipeline, not "
                f"a {cls.registry_name} one. Build it with the matching decoding type or serve "
                f"it with --pipeline {cls.sibling_registry_name}; running it here would "
                f"transcribe and put the wrong pipeline's name on the row"
            )

    @property
    def chunk(self) -> ChunkMode:
        return self._chunk

    def supported_buckets(self) -> tuple[int, ...]:
        return self._buckets

    def retained_graphs(self) -> int | None:
        """How many CUDA graphs NeMo is holding, or None if the boundary cannot tell."""
        if self._boundary.retained_graphs is None:
            return None
        return int(self._boundary.retained_graphs())

    def graph_capability(self) -> GraphCapability:
        """Three facts, and the narrowest of them wins: what the pipeline was built
        for, what the installed package carries, and what *this pipeline object*
        actually has attached.

        The third is the one that was missing. ``graph_step_available`` is a probe of
        the installed wheel -- does the module carry ``CudaGraphsStreamingEncoderStep``
        and does the cache-aware RNNT wrapper carry ``set_streaming_cuda_graphs`` -- and
        it answers for the package, not for the pipeline in front of us. The only
        observation of a real capture reads ``_graphs`` off this object. When the
        package says yes and the object has no graphed step, those two disagree, and a
        capability that reported the package's answer would let the run announce the
        graph path while every tick ran eager. The CTC pipeline is the concrete case:
        the probe reads the RNNT wrapper for both, because where PR #15863 puts the CTC
        switch cannot be read from an installed wheel here.

        So a pipeline that cannot be asked how many graphs it holds does not get to
        claim the graph path. See ``docs/decisions/0011``.
        """
        if not self._use_cuda_graphs:
            return GraphCapability.eager(
                "the pipeline was built with asr.use_cuda_graphs=false, so the encoder "
                "step runs eager and the row must say so"
            )
        if not self._boundary.graph_step_available:
            return GraphCapability.missing(
                "the installed NeMo lacks the graphed streaming encoder step "
                "(NeMo PR #15863: streaming_encoder_cuda_graphs."
                "CudaGraphsStreamingEncoderStep and set_streaming_cuda_graphs)"
            )
        if self._boundary.retained_graphs is None:
            return GraphCapability.missing(
                "the installed NeMo carries the graphed streaming encoder step but this "
                f"built {type(self._boundary.pipeline).__name__} has none attached: none "
                f"of {_GRAPH_STEP_PATHS} reaches a step holding graphs. The package's "
                "answer is not this pipeline's answer, and a capture here could never be "
                "observed, so the graph path is reported missing rather than claimed"
            )
        return GraphCapability.graphed()

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

    @property
    def biasing(self) -> bool:
        """Whether this adapter serves per-session phrase lists."""
        return self._biasing

    def _biasing_cfg(self, options: SessionOptions | None) -> Any | None:
        """This session's boosting-tree request, or None when it asked for no phrases.

        A session that asked for phrases on a server without biasing is refused here,
        on the tick thread, where ``open_stream`` turns it into one failed session
        rather than a wrong transcript. Transcribing it unbiased would return text the
        client cannot distinguish from a biased one, which is the whole failure mode
        this server exists to rule out.
        """
        if options is None or not options.phrases:
            return None
        if not self._biasing:
            raise InvalidArgument(
                f"this session sent {len(options.phrases)} biasing phrase(s) and this "
                "server was started without --biasing: it would transcribe the session "
                "unbiased and the transcript would not say so"
            )
        make_biasing = self._boundary.make_biasing
        if make_biasing is None:  # refused at construction; belt and braces
            raise InvalidArgument(
                "this session sent biasing phrases and this pipeline has no biasing arena"
            )
        return make_biasing(
            [(phrase.text, phrase.boost) for phrase in options.phrases],
            options.boost,
        )

    def _request_options(self, options: SessionOptions | None) -> Any:
        eou = self._stop_history_eou_ms
        if options is not None and options.stop_history_eou_ms is not None:
            eou = options.stop_history_eou_ms
        language = self._language_code
        if language is None and options is not None:
            language = options.language_code
        biasing_cfg = self._biasing_cfg(options)
        if biasing_cfg is None:
            return self._boundary.make_options(stop_history_eou=eou, language_code=language)
        return self._boundary.make_options(
            stop_history_eou=eou, language_code=language, biasing_cfg=biasing_cfg
        )

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
        stream's ``is_last`` frame went through; otherwise release them here.

        The biasing arena is the third thing NeMo frees only on ``is_last``, and
        ``delete_state`` does not free it: without this a session that aborts -- a
        dropped socket, an idle deadline, a failed step -- leaks its arena entry for
        the life of the process. It runs before ``delete_state`` because it reads the
        state that ``delete_state`` removes, and it is a no-op for a stream NeMo has
        already released.
        """
        self._opened.pop(stream_id, None)
        if self._biasing and self._boundary.release_biasing is not None:
            self._boundary.release_biasing(stream_id)
        if stream_id in self._started:
            self._started.discard(stream_id)
            if self._boundary.release_stream is not None:
                self._boundary.release_stream(stream_id)
        self._boundary.pipeline.delete_state(stream_id)

    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool, graph: bool
    ) -> list[StepResult]:
        if graph:
            capability = self.graph_capability()
            if not capability.available:
                # Never downgrade. NeMo would run this batch eager and nothing outside
                # would know, which is exactly how a row comes to look graphed without
                # being graphed.
                raise GraphPathUnavailable(
                    f"a batch of {len(frames)} rows was handed to the graph path and "
                    f"this pipeline has none: {capability.reason}"
                )
            if keep_all_outputs:
                # The peel, restated at the seam. NeMo's own `_can_use_graphs()` never
                # captures a keep_all_outputs=True call, so a final sub-batch claiming
                # the graph path is a caller bug, not a capture the scheduler declined.
                raise GraphPathUnavailable(
                    f"a final (keep_all_outputs=True) batch of {len(frames)} rows was "
                    "handed to the graph path; finals are peeled into an eager "
                    "side-batch and that shape is never captured"
                )
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
