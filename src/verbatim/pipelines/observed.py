# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""What a built pipeline is, read off the objects NeMo built. Read-only.

``/readyz`` reports three facts under ``observed`` and one configuration beside them,
and none of them is taken from a flag, a spec or a banner. A value derived from what
was asked for agrees with what was asked for by construction, so it can never report
the case it exists to catch: a checkpoint whose encoder kept another attention context,
a decoder that computes no step confidence under a configuration that asked for it, a
graph mode NeMo fell back from. Each one here is an attribute of the built object.

Every attribute path below was read from the installed NeMo (``nemo`` 3.1.0+cf724ac33,
paths relative to ``nemo/collections/asr/``):

- ``att_context_size``: ``BasePipeline.copy_asr_model_attributes`` sets
  ``pipeline.asr_model`` to the inference wrapper (``inference/pipelines/
  base_pipeline.py:309``); the wrapper's own ``asr_model`` is the loaded model
  (``inference/model_wrappers/asr_inference_wrapper.py:70``); the pipeline sets the
  encoder's context from ``streaming.att_context_size``
  (``inference/pipelines/cache_aware_rnnt_pipeline.py:118-119``, through
  ``cache_aware_asr_inference_wrapper.py:116-117``) and ``ConformerEncoder`` stores it
  as ``self.att_context_size`` even when it is not one the checkpoint lists, with only a
  warning (``modules/conformer_encoder.py:993-1008``). NeMo reads it back the same way,
  ``self.asr_model.encoder.att_context_size`` (``cache_aware_asr_inference_wrapper.py:107``).
- ``decoder_step_confidence``: the RNNT pipeline keeps the decoder's computer as
  ``pipeline.decoding_computer`` (``cache_aware_rnnt_pipeline.py:95-103``). It is built
  with ``preserve_step_confidence=preserve_frame_confidence``
  (``parts/submodules/rnnt_greedy_decoding.py:643-656``), stored as
  ``self.preserve_step_confidence`` (``parts/submodules/transducer_decoding/
  rnnt_label_looping.py:227``), and that attribute is what gates the per-step
  confidence computation (``rnnt_label_looping.py:283-287``). The CTC pipeline has no
  decoding computer (``inference/pipelines/cache_aware_ctc_pipeline.py`` never names
  one), so both decoder facts are None there, not False: nothing was read, and a CTC
  decoder computes its confidence from every step's log-probabilities regardless.
- ``decoder_graphs``: the same computer's ``cuda_graphs_mode`` (``parts/submodules/
  transducer_decoding/label_looping_base.py:106``), set by ``maybe_enable_cuda_graphs``
  from ``allow_cuda_graphs`` (``label_looping_base.py:125-151``), cleared by
  ``disable_cuda_graphs`` (``label_looping_base.py:153-160``), and read on every call
  to choose the graphed implementation (``label_looping_base.py:321``). Of NeMo's three
  modes (``label_looping_base.py:101-104``), ``full_graph`` and ``no_while_loops`` (the
  fallback on a driver without conditional nodes) replay captured graphs; ``no_graphs``
  runs the same stateful loop with no graph at all (``rnnt_label_looping.py:769-776``,
  ``:919-932``), so it is False, as is None. A mode this does not know is None.
- the configured word confidence: the wrapper keeps the decoding configuration it
  applied as ``decoding_cfg`` (``asr_inference_wrapper.py:66``, applied at ``:213-221``).
  ``CacheAwarePipelineBuilder.get_rnnt_decoding_cfg`` builds it from ``asr.decoding``
  and ``_apply_confidence_cfg`` (``inference/factory/cache_aware_pipeline_builder.py:
  55-65``, ``inference/factory/base_builder.py:77-105``), which reads
  ``preserve_frame_confidence`` from the greedy or the beam block and, when it is true,
  copies the top-level ``confidence`` block's aggregation and method into
  ``confidence_cfg``. That is classified against the configuration
  ``nemo_runtime.pipeline_config`` writes for each ``--word-confidence`` value.

A pipeline that is not NeMo's reports None for what it does not have: the CPU fake at
NeMo's seam has no encoder, no decoding computer and no decoding configuration. The
scripted fake adapter (``pipelines.fake``) is the one adapter configured "off" without a
NeMo pipeline to read, because it runs no decoder at all; any other adapter without one
is None.

``CacheAwareAdapter`` keeps its boundary as ``_boundary`` and exposes no accessor, so
``built_pipeline`` reads that attribute. It is this package's own adapter, and reading
it here keeps the adapter's surface unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Final

from verbatim.config import ChunkMode
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.pipelines.nemo_runtime import (
    WORD_CONFIDENCE_MODES,
    NeMoPipelineSpec,
    pipeline_config,
)

__all__ = [
    "ObservedFacts",
    "built_pipeline",
    "configured_word_confidence",
    "disagreement",
    "observe",
]

#: The encoder, from the pipeline. The first path is the one NeMo's own wrapper reads;
#: the others are tried for the reason ``cache_aware._GRAPH_STEP_PATHS`` tries them:
#: which object a pipeline hands back as ``asr_model`` has changed upstream before.
_ENCODER_PATHS: Final = (
    ("asr_model", "asr_model", "encoder"),
    ("asr_model", "encoder"),
    ("encoder",),
)
#: The method fields ``_apply_confidence_cfg`` copies from the ``confidence`` block.
_METHOD_KEYS: Final = ("name", "entropy_type", "alpha", "entropy_norm")
#: Whether the label-looping computer replays CUDA graphs under each of NeMo's
#: ``CudaGraphsMode`` values (``label_looping_base.py:101-104``).
_GRAPH_MODES: Final = {"full_graph": True, "no_while_loops": True, "no_graphs": False}
_ABSENT: Final = object()


@dataclass(frozen=True, slots=True)
class ObservedFacts:
    """Facts read off a built pipeline. None is "this pipeline has no such object"."""

    att_context_size: tuple[int, int] | None = None
    decoder_step_confidence: bool | None = None
    decoder_graphs: bool | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "att_context_size": (
                None if self.att_context_size is None else list(self.att_context_size)
            ),
            "decoder_step_confidence": self.decoder_step_confidence,
            "decoder_graphs": self.decoder_graphs,
        }


def built_pipeline(adapter: Any) -> Any | None:
    """The NeMo pipeline object behind a cache-aware adapter, or None for any other
    adapter (the scripted fake has none)."""
    boundary = getattr(adapter, "_boundary", None)
    return getattr(boundary, "pipeline", None)


def _walk(obj: Any, path: Sequence[str]) -> Any:
    for part in path:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _context_pair(value: Any) -> tuple[int, int] | None:
    """``[left, right]`` as two ints, or None for anything else."""
    if value is None or isinstance(value, str | bytes):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    if len(items) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in items):
        return None
    return (int(items[0]), int(items[1]))


def _graphs(mode: Any) -> bool | None:
    """Whether a computer whose ``cuda_graphs_mode`` is ``mode`` runs CUDA graphs. None
    for a computer without the attribute, and for a mode that is not one of NeMo's."""
    if mode is _ABSENT:
        return None
    if mode is None:
        return False  # label_looping_base.py:321: no mode runs torch_impl
    value = getattr(mode, "value", mode)  # NeMo's PrettyStrEnum, or its string
    return _GRAPH_MODES.get(value) if isinstance(value, str) else None


def observe(pipeline: Any) -> ObservedFacts:
    """Read the three facts off ``pipeline`` now, by plain attribute reads. A reading that
    is not where this expects, or is not the expected type, is None, which is "not
    observed", never a default."""
    if pipeline is None:
        return ObservedFacts()
    context: tuple[int, int] | None = None
    for path in _ENCODER_PATHS:
        encoder = _walk(pipeline, path)
        if encoder is None:
            continue
        context = _context_pair(getattr(encoder, "att_context_size", None))
        if context is not None:
            break
    # No decoding computer (NeMo's CTC pipeline) leaves both decoder facts None.
    computer = getattr(pipeline, "decoding_computer", None)
    step_confidence: bool | None = None
    graphs: bool | None = None
    if computer is not None:
        flag = getattr(computer, "preserve_step_confidence", None)
        step_confidence = flag if isinstance(flag, bool) else None
        graphs = _graphs(getattr(computer, "cuda_graphs_mode", _ABSENT))
    return ObservedFacts(
        att_context_size=context,
        decoder_step_confidence=step_confidence,
        decoder_graphs=graphs,
    )


@cache
def _written() -> tuple[tuple[str, bool, tuple[Any, ...]], ...]:
    """``(mode, preserve_frame_confidence, (aggregation, *method))`` for every value
    ``--word-confidence`` takes, from the configuration ``pipeline_config`` writes for
    it. The spec's other fields do not reach these keys."""
    rows = []
    for mode in WORD_CONFIDENCE_MODES:
        spec = NeMoPipelineSpec(
            model="observed",
            chunk=ChunkMode(160),
            att_context=(0, 1),
            num_slots=1,
            batch_size=1,
            word_confidence=mode,
        )
        config = pipeline_config(spec)
        block = config["confidence"]
        preserve = bool(config["asr"]["decoding"]["greedy"]["preserve_frame_confidence"])
        method = tuple(block["method_cfg"][key] for key in _METHOD_KEYS)
        rows.append((mode, preserve, (block["aggregation"], *method)))
    return tuple(rows)


def configured_word_confidence(adapter: Any) -> str | None:
    """The ``--word-confidence`` value the built pipeline was configured with.

    "off" for the scripted fake, ``FakePipelineAdapter``: it runs no decoder, so nothing
    there computes a confidence, and the CLI refuses any other value for it. For a NeMo
    pipeline, the mode whose configuration matches the decoding configuration NeMo
    applied. None for anything else: an adapter that is neither, a pipeline without a
    readable decoding configuration, or one that matches no mode. None is "not read",
    never a default; "off" is said only of an object known to compute nothing.
    """
    pipeline = built_pipeline(adapter)
    if pipeline is None:
        return "off" if isinstance(adapter, FakePipelineAdapter) else None
    decoding = getattr(getattr(pipeline, "asr_model", None), "decoding_cfg", None)
    if decoding is None:
        return None
    # base_builder.py:86-88: the greedy block's flag, or the beam block's.
    flags = [
        getattr(getattr(decoding, block, None), "preserve_frame_confidence", False)
        for block in ("greedy", "beam")
    ]
    if not all(isinstance(flag, bool) for flag in flags):
        return None
    preserve = any(flags)
    candidates = [row for row in _written() if row[1] is preserve]
    if not preserve:
        # NeMo returns before reading the confidence block when the flag is false
        # (base_builder.py:89-90), so there is nothing further to compare.
        return candidates[0][0] if len(candidates) == 1 else None
    confidence = getattr(decoding, "confidence_cfg", None)
    method = getattr(confidence, "method_cfg", None)
    reading = (
        getattr(confidence, "aggregation", None),
        *(getattr(method, key, None) for key in _METHOD_KEYS),
    )
    for mode, _, expected in candidates:
        if reading == expected:
            return mode
    return None


def disagreement(configured: str | None, observed: ObservedFacts) -> str | None:
    """Why the configured word confidence and the built decoder contradict each other,
    or None when they do not, or when either side could not be read.

    A configuration other than "off" over a decoder that computes no step confidence
    would put a 0.0 on every word that is not a measurement; "off" over a decoder that
    computes it runs the decode path of the other configuration under this one's name.
    """
    if configured is None or observed.decoder_step_confidence is None:
        return None
    expected = configured != "off"
    if expected == observed.decoder_step_confidence:
        return None
    computes = "computes" if observed.decoder_step_confidence else "computes no"
    return (
        f"word confidence is configured {configured!r} and the built RNNT decoder {computes} "
        "step confidence (decoding_computer.preserve_step_confidence is "
        f"{observed.decoder_step_confidence}): the label and the decoder disagree"
    )
