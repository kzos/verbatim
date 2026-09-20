# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Building NeMo's cache-aware pipelines for ``verbatim serve``, and the pre-flight
``verbatim doctor`` runs first.

Three things live here, and NeMo is imported by exactly two of them, at call time:

- ``pipeline_config`` turns a ``NeMoPipelineSpec`` into the configuration NeMo's
  ``PipelineBuilder`` reads. It is a plain dictionary with the keys of NeMo's own
  ``examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml`` (Speech
  ``353d190``), which the wheel does not ship, so the keys are carried here. Pure,
  and tested without NeMo. ``spec.decoding`` picks the branch
  ``CacheAwarePipelineBuilder.build`` takes: ``rnnt`` or ``ctc``.
- ``inspect_runtime`` reports which NeMo track is installed and whether it carries
  the graphed streaming encoder step of NeMo PR #15863: the module
  ``nemo.collections.asr.parts.submodules.streaming_encoder_cuda_graphs`` with
  ``CudaGraphsStreamingEncoderStep``, and ``set_streaming_cuda_graphs`` on the
  cache-aware inference wrapper, which the builder calls with ``asr.use_cuda_graphs``.
  Both were read from that PR's source; neither released wheel (2.7.3, 3.0.0) has
  them, and ``serve`` refuses the graph path rather than running eager unasked. The
  probe stays keyed on the cache-aware RNNT wrapper for both decoding types: it
  answers "which NeMo track is installed", and where that PR put the switch for the
  CTC wrapper is not something an installed wheel here can be read for.
- ``build_pipeline`` runs NeMo's builder and turns whatever it raises into a
  ``PipelineBuildError`` an operator can act on: the checkpoint, the chunk mode and
  its ``att_context_size``, the slots asked for, and NeMo's own words.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from verbatim.config import SAMPLE_RATE_HZ, VALID_CHUNK_MS, ChunkMode
from verbatim.scheduler.graph_budget import ConfigError

__all__ = [
    "KNOWN_LEFT_CONTEXT",
    "NeMoPipelineSpec",
    "PipelineBuildError",
    "RuntimeReport",
    "att_context_size",
    "build_pipeline",
    "graph_step_present",
    "inspect_runtime",
    "pipeline_config",
]

#: Left attention context per checkpoint family. The right context follows from the
#: chunk mode: ``(k + 1) * 80 ms``, so ``k = chunk_ms / 80 - 1``. A checkpoint not
#: listed here needs its left context passed explicitly; it is not guessed.
KNOWN_LEFT_CONTEXT: dict[str, int] = {
    "nvidia/nemotron-3.5-asr-streaming-0.6b": 56,
    "nvidia/nemotron-speech-streaming-en-0.6b": 70,
}

#: Where NeMo PR #15863 put the graphed step, and the method the builder calls.
GRAPH_STEP_MODULE = "nemo.collections.asr.parts.submodules.streaming_encoder_cuda_graphs"
GRAPH_STEP_CLASS = "CudaGraphsStreamingEncoderStep"
WRAPPER_MODULE = "nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper"
WRAPPER_CLASS = "CacheAwareRNNTInferenceWrapper"
GRAPH_SWITCH = "set_streaming_cuda_graphs"

Importer = Callable[[str], Any]


class PipelineBuildError(Exception):
    """NeMo refused to build the pipeline; the message says what was asked and why."""


def right_context(chunk: ChunkMode) -> int:
    """``k`` in ``att_context_size = [left, k]`` for a chunk of ``(k + 1) * 80 ms``."""
    if chunk.ms not in VALID_CHUNK_MS:  # ChunkMode already refuses these; belt and braces
        raise ConfigError(f"no attention context for a {chunk.ms} ms chunk")
    return chunk.ms // 80 - 1


def att_context_size(model: str, chunk: ChunkMode, *, left: int | None = None) -> list[int]:
    """``[left, right]`` for this checkpoint and chunk mode.

    ``left`` comes from the checkpoint family when the family is known; a checkpoint
    this module does not know needs it passed, because a wrong left context loads and
    runs and transcribes worse, silently.
    """
    if left is None:
        left = KNOWN_LEFT_CONTEXT.get(model)
        if left is None:
            known = ", ".join(f"{name} ({value})" for name, value in KNOWN_LEFT_CONTEXT.items())
            raise ConfigError(
                f"no known left attention context for checkpoint {model!r}: pass "
                f"--att-context-left; the known families are {known}"
            )
    if left < 0:
        raise ConfigError(f"the left attention context must be >= 0, got {left!r}")
    return [left, right_context(chunk)]


@dataclass(frozen=True, slots=True)
class NeMoPipelineSpec:
    """Everything ``serve`` decides about the NeMo pipeline, in one record."""

    model: str
    chunk: ChunkMode
    att_context: tuple[int, int]
    num_slots: int
    batch_size: int
    #: Which branch of ``CacheAwarePipelineBuilder.build`` to take: "rnnt" or "ctc".
    #: Not inferable from the checkpoint here, so the caller states it.
    decoding: str = "rnnt"
    stop_history_eou_ms: int = 800
    #: Whether NeMo's greedy batched decoder builds the per-stream biasing multi-model,
    #: which is what carries a session's boosting tree. Off by default: turning it on
    #: changes the decoder's arithmetic for *every* row, biased or not -- the fused path
    #: takes a second max over the vocabulary view and a ``torch.where`` -- so a run with
    #: it on is a different configuration, and the transcript digests of a run with it
    #: off do not carry over.
    enable_per_stream_biasing: bool = False
    use_cuda_graphs: bool = False
    compute_dtype: str = "bfloat16"
    device_id: int = 0
    matmul_precision: str = "high"
    log_level: int = 30

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("a checkpoint name or .nemo path is required")
        if self.num_slots < 1:
            raise ConfigError(f"num_slots must be >= 1, got {self.num_slots!r}")
        if not 1 <= self.batch_size <= self.num_slots:
            raise ConfigError(
                f"batch_size must be between 1 and num_slots ({self.num_slots}), "
                f"got {self.batch_size!r}"
            )
        if self.stop_history_eou_ms < 0:
            raise ConfigError(f"stop_history_eou_ms must be >= 0, got {self.stop_history_eou_ms!r}")
        if self.device_id < 0:
            raise ConfigError(f"device_id must be >= 0, got {self.device_id!r}")
        if self.compute_dtype not in ("bfloat16", "float16", "float32"):
            raise ConfigError(
                f"compute_dtype must be bfloat16, float16 or float32, got {self.compute_dtype!r}"
            )
        if self.decoding not in ("rnnt", "ctc"):
            raise ConfigError(
                f"decoding must be rnnt or ctc, got {self.decoding!r}: NeMo's "
                "CacheAwarePipelineBuilder has those two branches and no other"
            )
        if self.enable_per_stream_biasing and self.decoding != "rnnt":
            raise ConfigError(
                "enable_per_stream_biasing needs decoding='rnnt': biasing is reached "
                "through the RNNT decoding computer and the cache-aware CTC pipeline has "
                "no equivalent, so a CTC server would accept phrase lists and transcribe "
                "every session unbiased without saying so"
            )


#: The only RNNT strategy that carries per-stream biasing. ``RNNTDecoding`` passes
#: ``enable_per_stream_biasing`` to the decoder in its ``greedy_batch`` branch only;
#: plain ``greedy`` builds a ``GreedyRNNTInfer`` that never receives the flag and never
#: raises about it, so that combination transcribes every session unbiased and says
#: nothing. Read from NeMo's ``rnnt_decoding.py``, not assumed.
_BIASING_STRATEGY: Final = "greedy_batch"

#: ``asr.decoding`` for the RNNT branch: NeMo merges it over ``RNNTDecodingConfig``.
#:
#: ``boosting_tree`` and ``boosting_tree_alpha`` stay empty on purpose and are not the
#: per-session path. They configure ONE tree for the whole process, fused into every
#: stream's decode, so tenant A's names would be boosted inside tenant B's transcript.
#: A session's vocabulary arrives per request instead, on ``ASRRequestOptions.biasing_cfg``.
_RNNT_DECODING: dict[str, Any] = {
    "strategy": _BIASING_STRATEGY,
    "preserve_alignments": False,
    "fused_batch_size": -1,
    "greedy": {
        "use_cuda_graph_decoder": False,
        "enable_per_stream_biasing": False,
        # Pinned rather than left to NeMo's default: the label-looping decoder is the
        # one whose fused path was read for the invariance argument, and a default that
        # moved would move the decode with it.
        "loop_labels": True,
        "preserve_frame_confidence": False,
        "max_symbols": 10,
        "ngram_lm_model": None,
        "ngram_lm_alpha": 0.0,
        "boosting_tree": {
            "model_path": None,
            "key_phrases_file": None,
            "key_phrases_list": None,
            "key_phrase_items_list": None,
            "source_lang": "en",
        },
        "boosting_tree_alpha": 0.0,
    },
}


def pipeline_config(spec: NeMoPipelineSpec) -> dict[str, Any]:
    """The configuration NeMo's ``PipelineBuilder.build_pipeline`` reads, as a plain
    dictionary; ``build_pipeline`` wraps it in ``OmegaConf`` at the last moment.

    What differs from NeMo's example file, and why: ``chunk_size_in_secs`` is the
    chunk mode rather than ``null``, so NeMo's shift size is the tick's; ``num_slots``
    and ``batch_size`` are the scheduler's; per-stream biasing follows
    ``spec.enable_per_stream_biasing`` and is off unless an operator asked for it;
    ITN, translation and metrics are off (not this server's); output granularity is
    ``word`` so finals carry word timings; ``return_tail_result`` stays false, the
    published streaming defect is measured with it that way.

    NeMo's example file also carries ``asr.per_stream_biasing_defaults``. It is not
    written here, because nothing on this path reads it: only ``utils/manifest_io.py``
    and the offline example script do, and ``CacheAwarePipelineBuilder`` never does. A
    block that looks like it sets the default ``bpe_mode`` and alpha, and does not, is
    the same defect as a warm-up recording a capture it never made -- so the fields it
    would have set are set per request instead, where they take effect.

    The ``asr.decoding`` block follows ``spec.decoding``. It is the RNNT block NeMo's
    ``get_rnnt_decoding_cfg`` merges over ``RNNTDecodingConfig``, or, for CTC, the two
    fields ``get_ctc_decoding_cfg`` sets on a fresh ``CTCDecodingConfig``. That method
    takes no config argument and reads nothing from here, so the CTC block is a record
    of what NeMo will build rather than an instruction to it: writing the RNNT greedy
    knobs there would claim a configuration nothing applies. ``return_tail_result`` is
    likewise inert for CTC under ``use_cache: true``, because the wrapper only slices a
    tail when ``valid_out_len`` is set, and it is ``None`` whenever the cache is on.
    """
    left, right = spec.att_context
    if spec.decoding == "ctc":
        decoding: dict[str, Any] = {"strategy": "greedy", "preserve_alignments": False}
    else:
        # A copy: the caller gets a configuration it may edit, not this module's state.
        decoding = copy.deepcopy(_RNNT_DECODING)
        decoding["greedy"]["enable_per_stream_biasing"] = spec.enable_per_stream_biasing
        if spec.enable_per_stream_biasing and decoding["strategy"] != _BIASING_STRATEGY:
            raise ConfigError(
                f"per-stream biasing needs strategy {_BIASING_STRATEGY!r}, got "
                f"{decoding['strategy']!r}: NeMo hands the flag to the decoder on that "
                "branch only and raises nothing on the others, so the server would accept "
                "phrase lists and decode every session unbiased"
            )
    return {
        "asr": {
            "model_name": spec.model,
            "device": "cuda",
            "device_id": spec.device_id,
            "compute_dtype": spec.compute_dtype,
            "use_amp": False,
            "use_cuda_graphs": spec.use_cuda_graphs,
            "decoding": decoding,
        },
        "itn": {
            "input_case": "lower_cased",
            "whitelist": None,
            "overwrite_cache": False,
            "max_number_of_permutations_per_split": 729,
            "left_padding_size": 4,
            "batch_size": 32,
            "n_jobs": 16,
        },
        "confidence": {
            "exclude_blank": True,
            "aggregation": "mean",
            "method_cfg": {
                "name": "entropy",
                "entropy_type": "tsallis",
                "alpha": 0.5,
                "entropy_norm": "exp",
            },
        },
        "endpointing": {
            "stop_history_eou": spec.stop_history_eou_ms,
            "residue_tokens_at_end": 2,
        },
        "streaming": {
            "sample_rate": SAMPLE_RATE_HZ,
            "batch_size": spec.batch_size,
            "word_boundary_tolerance": 4,
            "att_context_size": [left, right],
            "use_cache": True,
            "use_feat_cache": True,
            "chunk_size_in_secs": spec.chunk.period_s,
            "request_type": "frame",
            "num_slots": spec.num_slots,
        },
        "matmul_precision": spec.matmul_precision,
        "log_level": spec.log_level,
        "pipeline_type": "cache_aware",
        "asr_decoding_type": spec.decoding,
        "audio_file": None,
        "output_filename": None,
        "output_dir": None,
        "enable_itn": False,
        "enable_nmt": False,
        "asr_output_granularity": "word",
        "cache_dir": None,
        "lang": None,
        "return_tail_result": False,
        "calculate_wer": False,
        "calculate_bleu": False,
        "warmup_steps": 0,
        "run_steps": 1,
    }


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    """What ``doctor`` found. ``graph_step`` is the whole of "which track": true only
    when both halves of NeMo PR #15863 are present."""

    nemo_version: str | None
    torch_version: str | None
    cuda_available: bool
    device_name: str | None
    inference_package: bool
    graph_step: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        """NeMo, its inference package, torch and a CUDA device: enough to run eager."""
        return (
            self.nemo_version is not None
            and self.inference_package
            and self.torch_version is not None
            and self.cuda_available
        )

    def lines(self) -> list[str]:
        """The report, one fact per line, in the order an operator reads them."""
        graph = (
            "present (NeMo PR #15863)"
            if self.graph_step
            else "ABSENT: serve refuses the graph path"
        )
        return [
            f"nemo             {self.nemo_version or 'not installed'}",
            f"nemo inference   {'importable' if self.inference_package else 'not importable'}",
            f"torch            {self.torch_version or 'not installed'}",
            f"cuda             {self.device_name if self.cuda_available else 'no device'}",
            f"graphed step     {graph}",
            *self.notes,
        ]


def graph_step_present(import_module: Importer = importlib.import_module) -> bool:
    """Whether the installed NeMo carries **both** halves of PR #15863.

    The module with ``CudaGraphsStreamingEncoderStep`` and the wrapper method the
    builder calls with ``asr.use_cuda_graphs``: one without the other is not the
    graph path. Read directly rather than inferred from a version number, because
    ``docs/decisions/0002`` records two released versions where the number said
    nothing useful. ``doctor`` reports it and ``NeMoBoundary`` binds it.
    """
    try:
        module = import_module(GRAPH_STEP_MODULE)
        wrapper = getattr(import_module(WRAPPER_MODULE), WRAPPER_CLASS)
    except Exception:
        return False
    return hasattr(module, GRAPH_STEP_CLASS) and hasattr(wrapper, GRAPH_SWITCH)


def inspect_runtime(import_module: Importer = importlib.import_module) -> RuntimeReport:
    """Probe the installed runtime. ``import_module`` is injectable so the report can be
    tested on a machine with no NeMo, and against the shape of the PR's own tree."""
    notes: list[str] = []
    nemo_version: str | None = None
    torch_version: str | None = None
    cuda_available = False
    device_name: str | None = None
    inference_package = False
    graph_step = False

    try:
        nemo = import_module("nemo")
        nemo_version = str(getattr(nemo, "__version__", "unknown"))
    except Exception as exc:  # ImportError, or a broken install raising anything
        notes.append(f"nemo: {type(exc).__name__}: {exc}")
    try:
        torch = import_module("torch")
        torch_version = str(getattr(torch, "__version__", "unknown"))
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_name = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        notes.append(f"torch: {type(exc).__name__}: {exc}")
    if nemo_version is not None:
        try:
            import_module("nemo.collections.asr.inference.factory.pipeline_builder")
            inference_package = True
        except Exception as exc:
            notes.append(f"nemo.collections.asr.inference: {type(exc).__name__}: {exc}")
    if inference_package:
        graph_step = graph_step_present(import_module)
        if not graph_step:
            notes.append(
                "the installed NeMo predates PR #15863: pass --eager to run the encoder "
                "step eager, and say so in the row"
            )
    return RuntimeReport(
        nemo_version=nemo_version,
        torch_version=torch_version,
        cuda_available=cuda_available,
        device_name=device_name,
        inference_package=inference_package,
        graph_step=graph_step,
        notes=tuple(notes),
    )


def build_pipeline(
    spec: NeMoPipelineSpec, import_module: Importer = importlib.import_module
) -> Any:
    """Run NeMo's builder for ``spec``. This is the model download and load.

    Anything NeMo raises becomes a ``PipelineBuildError`` whose message an operator
    can act on without a stack trace: what was asked (checkpoint, chunk mode and its
    attention context, slots and batch) and NeMo's own words. The trace is chained
    for ``--traceback``.
    """
    left, right = spec.att_context
    asked = (
        f"checkpoint {spec.model!r}, cache-aware {spec.decoding.upper()}, "
        f"chunk {spec.chunk.ms} ms "
        f"(att_context_size [{left}, {right}]), num_slots {spec.num_slots}, "
        f"batch_size {spec.batch_size}, {spec.compute_dtype} on cuda:{spec.device_id}, "
        f"{'CUDA graphs' if spec.use_cuda_graphs else 'eager encoder step'}, "
        f"per-stream biasing {'on' if spec.enable_per_stream_biasing else 'off'}"
    )
    try:
        omegaconf = import_module("omegaconf")
        builder_module = import_module("nemo.collections.asr.inference.factory.pipeline_builder")
    except Exception as exc:
        raise PipelineBuildError(
            f"cannot import NeMo's pipeline builder ({type(exc).__name__}: {exc}); "
            f"install the model runtime with pip install 'verbatim[nemo]'. Asked for {asked}"
        ) from exc
    cfg = omegaconf.OmegaConf.create(pipeline_config(spec))
    try:
        return builder_module.PipelineBuilder.build_pipeline(cfg)
    except Exception as exc:
        raise PipelineBuildError(
            f"NeMo could not build the pipeline for {asked}: {type(exc).__name__}: {exc}. "
            "Check that the checkpoint exists and loads (HF_HOME for the cache), that the "
            "chunk mode is one the checkpoint's att_context_size family supports, and that "
            "the GPU has the memory for num_slots"
        ) from exc
