"""Step 1: does the new workhorse's transcript depend on its batch neighbours?

Re-measures, on ``nvidia/nemotron-speech-streaming-en-0.6b``, what the ``stock-divergence-*.json``
records measured on the 114M hybrid: each LibriSpeech test-other recording transcribed ALONE and in
SLOT 0 OF A BATCH of ``BATCH``, through ``verbatim.pipelines.nemo_runtime.build_pipeline`` ->
``CacheAwareRNNTPipeline.transcribe_step`` -- the code path ``serve`` builds, not an example script.

Arms, each a comparison of two transcripts of the same recording:

  ragged     alone vs batch; rows keep their own lengths
  equalised  alone vs batch; every row zero-padded to one common length. Still a batch of 1 against
             a batch of BATCH, so two shapes: this is NOT the fixed-shape design.
  fixed      the recording in a batch of BATCH whose other rows are silence, against the same
             equal-length batch of real neighbours. One shape; only the neighbours' content
             differs. This is fixed-shape padding inside the stock pipeline, which the white paper
             did not test.

Compared on two channels: the final text, and the word timings in ``final_segments`` (start, end).
Timings are a channel the text can hide; on the old model they diverged more often than the text.
Each timing tuple also stores the segment's ``conf`` as a fourth element, which is NOT compared:
``comparable`` drops it, so the counts mean what they meant before it was stored. A recording whose
text and (text, start, end) timings agree but whose confidences differ is counted apart, in
``confidence_only_divergent``; it is not a text or timing divergence and is not in ``divergences``.
With WORD_CONFIDENCE=off (the default) NeMo computes no confidence and every conf is 0.0; any other
value (``nemo-shipped``, ``paper-best``) is passed to ``NeMoPipelineSpec.word_confidence`` and is a
different configuration.

Per arm the record keeps, besides the counts, the ``transcripts`` and ``divergences`` it always
kept, and:

  every_recording  {librispeech_id: {"a_text", "b_text", "a_words", "b_words", "a_nemo_text",
                   "b_nemo_text"}} for EVERY checked recording, divergent or not; each word is
                   [word, start_s, end_s, conf]. This is what lets a later reader rank every
                   word by confidence, or compare a run with confidence on against one with it
                   off, recording by recording. The ``*_nemo_text`` are NeMo's own
                   concatenation of the side's finals (below).

Observed, not requested. After the build the probe reads, off the decoding computer the decode
path actually calls (``pipeline.asr_model.asr_model.decoding.decoding.decoding_computer``, the
object ``CacheAwareRNNTPipeline.init_decoding_computer`` also finds), whether it keeps step
confidence and whether it runs CUDA graphs, and refuses to run when either differs from what the
spec asked for. It reads the attention context off the built encoder
(``pipeline.asr_model.asr_model.encoder.att_context_size``, the path NeMo's own wrapper reads and
``verbatim.pipelines.observed`` documents) and refuses to run when it is not the context the spec
asked for: NeMo stores a context the checkpoint does not list with only a warning. After the
guard recording it refuses again when the number of nonzero confidences there contradicts
WORD_CONFIDENCE (above 0 under off, 0 under nemo-shipped or paper-best). The readings are
stamped: ``word_confidence_observed``, ``decoder_graphs_observed`` and
``att_context_size_observed``, at the top level (the last dtype built) and in each run.
``att_context_size``, ``use_cuda_graphs`` and ``word_confidence`` are still stamped from the
request, as before: they are what was asked for.

Controls that let a zero mean something:
  * repeat   the first REPEAT recordings are transcribed twice in the SAME shape (alone twice, batch
             twice). If those differ, the pipeline is not run-to-run deterministic and every other
             count is noise; the record says so. REPEAT=0 checks nothing, and the record says that
             instead of calling the run identical.
  * positive the ragged and equalised arms are expected to diverge at bfloat16. If every arm
             reads zero, the harness cannot tell "invariant" from "blind", and the record says that
             instead of a clean result.

Provenance is stamped by the run, not typed afterwards: GPU, driver, torch and its CUDA, the NeMo
version, the model's resolved Hugging Face revision (the run refuses to start if it does not
start with MODEL_REVISION; the record keeps the resolved one), dtype, matmul precision,
att_context, chunk. The resolved revision is the Hugging Face API's current one for the model,
not a reading of the snapshot NeMo loaded. ``nemo_commit`` is what the environment says
(NEMO_COMMIT), "unstated" when it says nothing.

The card, observed twice. ``gpu_uuid`` is the UUID ``nvidia-smi -i`` prints for the first card
in CUDA_VISIBLE_DEVICES, and ``gpu_uuid_torch`` is ``torch.cuda.get_device_properties(0).uuid``,
the card the run computes on. The two orders can differ (``CUDA_DEVICE_ORDER``, stamped as
``cuda_device_order``), so the run refuses to start unless both are read and name the same card
once nvidia-smi's "GPU-" prefix and the case are set aside.

The code, observed. ``code.verbatim_path`` is the directory of the ``verbatim`` package this
process imported (``os.path.dirname(verbatim.__file__)``, resolved); the run refuses to start
unless it is ``src/verbatim`` of the checkout this file lives in, because an installed
``verbatim`` from another checkout would run code the record cannot name. ``code.probe_sha256``
is this file's sha256, read at the start. ``verbatim_commit`` is ``git rev-parse HEAD`` of that
checkout and ``tracked_files_modified`` whether ``git status`` lists a modified tracked file
(``code.tracked_changes`` lists them); both are null when the checkout is not the top of a git
work tree or git cannot say. What VERBATIM_COMMIT said is kept apart, in
``verbatim_commit_from_env`` ("unstated" when it says nothing).

A record is written once. The run refuses to start when OUT or OUT.tmp exists. Its first save
creates OUT and never replaces a file there: one that appears at OUT or OUT.tmp during the run
(another run given the same OUT) is left as it is and the run stops. Later saves replace the
run's own record.

The data. The pool is the first ``targets + BATCH`` recordings of the stream, or all of them
when the stream holds fewer (``pool`` stamps how many). Target i's neighbours are
``pool[(i + 1 + k) % len(pool)]``, so when the stream runs out the last targets' neighbours
wrap to the first recordings, as the stock run's last 31 did (test-other holds 2,939
recordings; that run asked for 2,939 + 32). The run refuses a pool smaller than the targets or
than BATCH: a target would be missing, or would be its own neighbour.

Importing this module loads no torch, no NeMo, no model and no data: the comparisons, the controls
and the per-recording entries are plain functions, tested on the CPU. Everything that touches the
GPU, the dataset or the model runs from ``main()``. ``main(backend=...)`` takes a stand-in for
all of that, for the CPU test only, and a record written that way is stamped
``"fake_pipeline": true`` whatever the stand-in says about itself.

This is not, line for line, the code that wrote the finished stock run's record: that record
was written at commit 6583a83, when the probe was one top-level script. The restructuring was
meant to keep the decode path, the spec, the arms, the neighbours, the padding, the
normalisation and the order of calls. On a fake pipeline, the CPU tests in
``tests/test_stock_divergence.py`` pin these: the calls to ``transcribe_step``, in order, with
their rows, frames, stream ids, zero padding and float32 samples, including the neighbours of
the last targets when the pool wraps; the recording's text, as the probe builds it from the
per-step finals; ``words()``; which side of each arm is ``a`` and which is ``b``; the divergence
index ``n``; both channels of the repeat control; the spec and the record's fields under
settings that are not the defaults; the guards. What they cannot pin is NeMo on a GPU. That the
two probes give the same answers there is shown only by replaying targets against that record
(``scripts/compare_stock_replay.py``).

The text is the probe's per-step space join, NOT NeMo's concatenation of its finals: each
nonempty step final is stripped and the finals are joined with one space, as the stock run did.
NeMo's ``TranscribeStepOutput.from_state`` gives a final a leading separator after the stream's
first request, but none when the step's first word continues the last word finalised
(``concat_with_space`` False). NeMo's own concatenation (``BasePipeline.run`` appends each
final, stripping the separator only while nothing has accumulated) then reads that word whole,
and the probe's text reads it split in two. The join is wrong in BOTH directions:

  * It can make a TEXT divergence NeMo does not see. The stock record shows this at n=327
    (3080-5032-0015). There, side a of the ragged and equalised arms reads "that the y are",
    with the segments "the" (15.6-15.68 s) and "y" (15.68-15.76 s) split exactly at the end of
    the 14th 1,120 ms chunk. Side b reads "that they are", and both arms count the recording as
    a TEXT divergence. If that split is NeMo's step boundary, as the timings say, NeMo's own
    text would read "they" on both sides.
  * It can HIDE one NeMo does see. When side a's step continues the last word and side b's
    starts a new one, with the same words and timings, NeMo's texts read "they" and "the y"
    while the probe reads "the y" on both sides and calls the pair identical. So no count of
    the probe's text or timing channel, the fixed arm's zero included, excludes such a
    difference.

So every recording's entry also keeps NeMo's own concatenation of each side's finals,
``a_nemo_text`` and ``b_nemo_text``, and each arm counts ``nemo_text_divergent`` (NeMo's two
texts differ) and, of those, ``nemo_text_divergent_hidden`` (the probe's two texts agree: the
join hid it). Neither is compared or classified: the text, the counts above them and
``divergences`` are the join's, because the stock run used it and the replay must reproduce
it. A record written before these fields cannot show the hiding direction at all. The tests
pin the join and NeMo's concatenation on NeMo-shaped finals: with the separator, without it,
and with empty steps; and, on NeMo's own ``StreamingState``, both directions.

Exploratory. NOT a harness row.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import verbatim
from verbatim.config import ChunkMode
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, att_context_size

#: This file, resolved: its sha256 is stamped, and its checkout is the one whose code must run.
PROBE_FILE = Path(__file__).resolve()
#: The separator NeMo's ``TranscribeStepOutput.from_state`` puts before a final, and the one
#: ``BasePipeline.run`` strips while nothing has accumulated.
NEMO_SEP = " "

#: The arms this probe knows how to build. Anything else in ARMS is refused, not silently empty.
KNOWN_ARMS = ("ragged", "equalised", "fixed")
#: What ``classify`` calls one (a, b) pair.
IDENTICAL, TEXT, TIMING, CONFIDENCE = "identical", "text", "timing", "confidence"
SAME_SHAPE_VERDICT = "run-to-run identical in the same shape"
NOT_DETERMINISTIC_VERDICT = "NOT run-to-run deterministic: divergence counts below include noise"
NOT_CHECKED_VERDICT = (
    "NOT checked: REPEAT=0, so run-to-run determinism is unknown and the counts below may"
    " include noise"
)
#: The chain of attributes from the built pipeline to the decoding computer the decode calls:
#: ``CacheAwareRNNTInferenceWrapper.execute_step`` calls
#: ``self.asr_model.decoding.rnnt_decoder_predictions_tensor``, which calls ``self.decoding``,
#: whose label-looping greedy decode calls ``self.decoding_computer``.
DECODER_PATH = ("asr_model", "asr_model", "decoding", "decoding", "decoding_computer")
#: The chain from the built pipeline to the encoder whose ``att_context_size`` the streaming
#: step uses: ``BasePipeline`` keeps the inference wrapper as ``asr_model``, the wrapper keeps
#: the loaded model as its own ``asr_model``, and NeMo's wrapper reads the context back as
#: ``self.asr_model.encoder.att_context_size`` (``cache_aware_asr_inference_wrapper.py:107``).
ENCODER_PATH = ("asr_model", "asr_model", "encoder")


# --- pure: what is compared, and how ----------------------------------------------------------


def words(text: str) -> list[str]:
    """Lowercase, punctuation stripped, apostrophes kept: comparable with LibriSpeech references."""
    return re.sub(r"[^\w' ]+", " ", text.lower()).split()


def comparable(result):
    """(text, timings) with each timing cut to (text, start, end): what the arms and the repeat
    control compare. The stored conf is the fourth element and is deliberately left out, and so
    is NeMo's own text, a side's third element when it has one."""
    text, timings = result[0], result[1]
    return text, [tuple(t[:3]) for t in timings]


def timing_of(seg: Any) -> tuple[str, float, float, float]:
    """One final segment as the stored timing tuple: (text, start_s, end_s, conf)."""
    return (
        seg.text.strip(),
        round(float(seg.start), 4),
        round(float(seg.end), 4),
        float(seg.conf),
    )


def classify(a_side: tuple[str, list], b_side: tuple[str, list]) -> str:
    """What one (a, b) pair of (text, timings) is.

    ``text``: the texts differ. ``timing``: the texts agree and the (text, start, end) timings do
    not. ``confidence``: text and (text, start, end) agree and the stored confidences do not.
    ``identical``: nothing differs.
    """
    if a_side[0] != b_side[0]:
        return TEXT
    if comparable(a_side)[1] != comparable(b_side)[1]:
        return TIMING
    if [tuple(t) for t in a_side[1]] != [tuple(t) for t in b_side[1]]:
        return CONFIDENCE
    return IDENTICAL


def same_in_one_shape(first: tuple[str, list], second: tuple[str, list]) -> bool:
    """The repeat control's comparison: text and (text, start, end) timings, conf left out."""
    return comparable(first) == comparable(second)


def repeat_verdict(rep: Mapping[str, int]) -> str:
    """The verdict the repeat control's counts support."""
    if rep["checked"] == 0:
        return NOT_CHECKED_VERDICT
    if rep["alone_identical"] != rep["checked"] or rep["batch_identical"] != rep["checked"]:
        return NOT_DETERMINISTIC_VERDICT
    return SAME_SHAPE_VERDICT


def new_arm() -> dict[str, Any]:
    """An arm's record before any recording is checked."""
    return {
        "checked": 0,
        "text_divergent": 0,
        "timing_only_divergent": 0,
        "divergences": [],
        "transcripts": {},
        "confidence_only_divergent": 0,
        "nemo_text_divergent": 0,
        "nemo_text_divergent_hidden": 0,
        "every_recording": {},
    }


def recording_entry(a_side: tuple[str, list, str], b_side: tuple[str, list, str]) -> dict[str, Any]:
    """One recording's entry in ``every_recording``: both texts, both word lists, and NeMo's
    own concatenation of each side's finals."""
    return {
        "a_text": a_side[0],
        "b_text": b_side[0],
        "a_words": [list(t) for t in a_side[1]],
        "b_words": [list(t) for t in b_side[1]],
        "a_nemo_text": a_side[2],
        "b_nemo_text": b_side[2],
    }


def record_pair(
    arm: dict[str, Any],
    n: int,
    rid: str,
    a_side: tuple[str, list, str],
    b_side: tuple[str, list, str],
) -> str:
    """Count one checked recording into its arm; return what ``classify`` called it.

    A side is (text, timings, nemo_text), as ``transcribe`` returns it. ``classify`` and every
    count but the two ``nemo_text_*`` read the text and timings only."""
    a_text, a_time, b_text, b_time = a_side[0], a_side[1], b_side[0], b_side[1]
    kind = classify(a_side, b_side)
    arm["checked"] += 1
    if a_side[2] != b_side[2]:
        arm["nemo_text_divergent"] += 1
        arm["nemo_text_divergent_hidden"] += a_text == b_text
    arm["transcripts"][rid] = {"a": a_text, "b": b_text} if kind == TEXT else {"a": a_text}
    if kind == TEXT:
        arm["text_divergent"] += 1
    elif kind == TIMING:
        arm["timing_only_divergent"] += 1
    elif kind == CONFIDENCE:
        arm["confidence_only_divergent"] += 1
    if kind in (TEXT, TIMING):
        arm["divergences"].append(
            {
                "n": n,
                "librispeech_id": rid,
                "text_differs": kind == TEXT,
                "a": a_text,
                "b": b_text,
                "a_timings": a_time,
                "b_timings": b_time,
            }
        )
    arm["every_recording"][rid] = recording_entry(a_side, b_side)
    return kind


def positive_control(arms_record: Mapping[str, Mapping[str, Any]], arms: Sequence[str]) -> str:
    """Whether a two-shape arm saw a text or timing divergence on this run."""
    counts = {
        arm: arms_record[arm]["text_divergent"] + arms_record[arm]["timing_only_divergent"]
        for arm in arms
    }
    # Text OR timing: a two-shape arm that moved only a timing still shows the harness can see.
    positive = [arm for arm in ("ragged", "equalised") if arm in arms and counts[arm] > 0]
    return (
        f"present: {', '.join(positive)} diverged, so the harness can see a difference on this run"
        if positive
        else "ABSENT: no two-shape arm diverged, so a zero in any arm cannot be told from a blind"
        " harness"
    )


def count_nonzero_conf(timings: Sequence[Sequence[Any]]) -> int:
    """How many stored timing tuples carry a confidence other than 0.0."""
    return sum(1 for t in timings if t[3] != 0.0)


# --- pure: what the built pipeline does, against what was asked ------------------------------


def observe_decoder(pipeline: Any) -> dict[str, Any]:
    """Read, off the BUILT pipeline, what its decoding computer does.

    The computer is reached by ``DECODER_PATH``, the chain the decode itself follows (NeMo
    cf724ac: ``cache_aware_rnnt_inference_wrapper.py:192`` -> ``rnnt_decoding.py:752`` ->
    ``rnnt_greedy_decoding.py:819``), which is also the lookup
    ``CacheAwareRNNTPipeline.init_decoding_computer`` makes
    (``cache_aware_rnnt_pipeline.py:95-103``).

    * ``decoder_step_confidence``: ``preserve_step_confidence``, set from the decoding config's
      ``preserve_frame_confidence`` in ``GreedyBatchedRNNTLabelLoopingComputer.__init__``
      (``rnnt_label_looping.py:230``) and read at every step (:434, :495, :517).
    * ``decoder_graphs``: ``cuda_graphs_mode is not None``. ``maybe_enable_cuda_graphs``
      (``label_looping_base.py:125-151``) leaves it None unless ``allow_cuda_graphs``
      (``rnnt_label_looping.py:267``, from ``use_cuda_graph_decoder``) is true, and ``__call__``
      takes the graph implementation exactly when it is not None and the encoder output is on
      CUDA (``label_looping_base.py:321``). ``decoder_graphs_mode`` is its value; a FULL_GRAPH
      mode may fall back to NO_WHILE_LOOPS at the first capture (``rnnt_label_looping.py:919-931``),
      which leaves ``decoder_graphs`` true.
    """
    computer = pipeline
    walked = "pipeline"
    for name in DECODER_PATH:
        computer = getattr(computer, name, None)
        walked += f".{name}"
        if computer is None:
            raise SystemExit(
                f"[guard] FAILED: the built pipeline has no {walked}; what the decoder computes"
                " cannot be observed, so the record could only repeat what was asked for"
            )
    step_confidence = getattr(computer, "preserve_step_confidence", None)
    if not isinstance(step_confidence, bool):
        raise SystemExit(
            f"[guard] FAILED: {walked}.preserve_step_confidence is {step_confidence!r}, not a bool;"
            " whether the decoder keeps confidence cannot be observed"
        )
    if not hasattr(computer, "cuda_graphs_mode"):
        raise SystemExit(
            f"[guard] FAILED: {walked} has no cuda_graphs_mode; whether the decoder runs CUDA"
            " graphs cannot be observed"
        )
    mode = computer.cuda_graphs_mode
    return {
        "decoder_step_confidence": step_confidence,
        "decoder_graphs": mode is not None,
        "decoder_graphs_mode": None if mode is None else str(getattr(mode, "value", mode)),
    }


def refuse_unrequested_decoder(
    word_confidence: str, decoder_graphs: bool, observed: Mapping[str, Any]
) -> None:
    """Stop when the built decoder does not do what the spec asked for."""
    wanted = word_confidence != "off"
    if observed["decoder_step_confidence"] is not wanted:
        raise SystemExit(
            f"[guard] FAILED: word_confidence {word_confidence!r} asks for step confidence"
            f" {wanted}, but the built decoder's preserve_step_confidence is"
            f" {observed['decoder_step_confidence']}"
        )
    if observed["decoder_graphs"] is not bool(decoder_graphs):
        raise SystemExit(
            f"[guard] FAILED: the spec asks for decoder graphs {bool(decoder_graphs)}, but the"
            f" built decoder's cuda_graphs_mode is {observed['decoder_graphs_mode']!r}"
        )


def observe_att_context(pipeline: Any) -> list[int]:
    """Read, off the BUILT pipeline's encoder, the attention context it runs with.

    ``CacheAwareRNNTPipeline.init_parameters`` hands ``streaming.att_context_size`` to the
    encoder's ``set_default_att_context_size`` (``cache_aware_rnnt_pipeline.py:118-119``), which
    stores it as ``self.att_context_size`` even when the checkpoint does not list it, with only
    a warning (``conformer_encoder.py:993-1008``). It may be an OmegaConf list; any sequence of
    two ints is read. Anything else stops the run: an unreadable context could only be
    replaced by the requested one, which is the reading this exists to check.
    """
    encoder = pipeline
    walked = "pipeline"
    for name in ENCODER_PATH:
        encoder = getattr(encoder, name, None)
        walked += f".{name}"
        if encoder is None:
            raise SystemExit(
                f"[guard] FAILED: the built pipeline has no {walked}; the attention context it"
                " runs with cannot be observed"
            )
    value = getattr(encoder, "att_context_size", None)
    try:
        items = None if isinstance(value, str | bytes) else list(value)
    except TypeError:
        items = None
    if (
        items is None
        or len(items) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in items)
    ):
        raise SystemExit(
            f"[guard] FAILED: {walked}.att_context_size is {value!r}, not two ints; the"
            " attention context the encoder runs with cannot be observed"
        )
    return items


def refuse_unrequested_att_context(requested: Sequence[int], observed: Sequence[int]) -> None:
    """Stop when the built encoder does not run the attention context the spec asked for."""
    if list(observed) != list(requested):
        raise SystemExit(
            f"[guard] FAILED: the spec asks for att_context_size {list(requested)}, but the built"
            f" encoder's att_context_size is {list(observed)}"
        )


def word_confidence_stamp(
    mode_requested: str, observed: Mapping[str, Any], nonzero: int
) -> dict[str, Any]:
    """The record's ``word_confidence_observed``: the mode asked for, beside what the built
    decoder keeps (``observed``, from ``observe_decoder``) and the guard recording's count."""
    return {
        "mode_requested": mode_requested,
        "decoder_step_confidence": observed["decoder_step_confidence"],
        "nonzero_conf_words_on_guard_recording": nonzero,
    }


def refuse_contradicting_confidence(word_confidence: str, nonzero: int) -> None:
    """Stop when the guard recording's confidences contradict the mode that was asked for."""
    if word_confidence == "off" and nonzero > 0:
        raise SystemExit(
            f"[guard] FAILED: word_confidence off, but {nonzero} words on the guard recording"
            " carry a nonzero confidence; off is not off"
        )
    if word_confidence != "off" and nonzero == 0:
        raise SystemExit(
            f"[guard] FAILED: word_confidence {word_confidence!r}, but every word on the guard"
            " recording carries confidence 0.0; the switch did not reach the words"
        )


# --- settings, the record, the stream ---------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What the run was asked for, read from the environment and argv."""

    out: str
    model: str
    model_revision: str
    batch: int
    n_targets: int
    chunk_ms: int
    dtypes: list[str]
    arms: list[str]
    matmul: str
    repeat: int
    left: int | None
    word_confidence: str


def settings_from(env: Mapping[str, str], argv: Sequence[str]) -> Settings:
    """The probe's environment variables and argument, with the defaults it always had."""
    settings = Settings(
        out=env["OUT"],
        model=env.get("MODEL", "nvidia/nemotron-speech-streaming-en-0.6b"),
        model_revision=env.get("MODEL_REVISION", ""),
        batch=int(env.get("BATCH", "32")),
        n_targets=int(argv[1]) if len(argv) > 1 else 2939,
        chunk_ms=int(env.get("CHUNK_MS", "1120")),
        dtypes=env.get("DTYPES", "bfloat16").split(","),
        arms=env.get("ARMS", "ragged,equalised,fixed").split(","),
        matmul=env.get("MATMUL", "highest"),
        repeat=int(env.get("REPEAT", "8")),
        left=int(env["ATT_LEFT"]) if env.get("ATT_LEFT") else None,
        word_confidence=env.get("WORD_CONFIDENCE", "off"),
    )
    unknown = [arm for arm in settings.arms if arm not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"[guard] unknown arms {unknown}; this probe builds {list(KNOWN_ARMS)}")
    return settings


def spec_for(
    settings: Settings, dtype: str, chunk: ChunkMode, att: Sequence[int]
) -> NeMoPipelineSpec:
    """The pipeline spec for one dtype."""
    return NeMoPipelineSpec(
        model=settings.model,
        chunk=chunk,
        att_context=tuple(att),
        num_slots=max(256, settings.batch * 4),
        batch_size=settings.batch,
        compute_dtype=dtype,
        matmul_precision=settings.matmul,
        word_confidence=settings.word_confidence,
    )


def refuse_existing_out(out: str) -> None:
    """Stop before anything runs when OUT or OUT.tmp is already there: a record is written
    once, and a mistyped OUT must not replace another run's record."""
    for path in (out, out + ".tmp"):
        if os.path.lexists(path):
            raise SystemExit(
                f"[guard] FAILED: {path} exists; a record is written once, pick a new OUT"
            )


def save(state: Mapping[str, Any], out: str, *, first: bool = False) -> None:
    """Write the record through ``OUT.tmp``: a reader sees the whole previous record or the
    whole new one, never a half-written file.

    The first save of a run (``first``) creates OUT.tmp and OUT and replaces neither: a file
    that appeared at either during the run is left as it is, and the run stops. Later saves
    replace the run's own record with ``os.replace``."""
    tmp = out + ".tmp"
    try:
        with open(tmp, "x" if first else "w") as f:
            json.dump(state, f, indent=1)
    except FileExistsError:
        raise SystemExit(
            f"[guard] FAILED: {tmp} appeared during the run; it is not written over"
        ) from None
    if not first:
        os.replace(tmp, out)
        return
    try:
        os.link(tmp, out)  # atomic, and unlike os.replace it never replaces
    except FileExistsError:
        os.unlink(tmp)
        raise SystemExit(
            f"[guard] FAILED: {out} appeared during the run; it is not written over"
        ) from None
    except OSError:
        # A filesystem without hard links: an exclusive create is not atomic, but it is
        # still never a replacement.
        try:
            with open(out, "x") as f:
                f.write(Path(tmp).read_text())
        except FileExistsError:
            os.unlink(tmp)
            raise SystemExit(
                f"[guard] FAILED: {out} appeared during the run; it is not written over"
            ) from None
    os.unlink(tmp)


def refuse_short_pool(pool_size: int, n_targets: int, batch: int) -> None:
    """Stop when the data holds fewer recordings than the targets (a target would be missing)
    or than BATCH (a target would be its own neighbour after the wrap)."""
    if pool_size < n_targets or pool_size < batch:
        raise SystemExit(
            f"[guard] FAILED: the data holds {pool_size} recordings; {n_targets} targets at"
            f" BATCH {batch} need at least {max(n_targets, batch)}"
        )


# --- which card, which code --------------------------------------------------------------------


def normalise_uuid(uuid: str) -> str:
    """A GPU UUID as both readers can print it: nvidia-smi's "GPU-" prefix off, lowercase."""
    text = uuid.strip().lower()
    return text[len("gpu-") :] if text.startswith("gpu-") else text


def smi_uuid(driver_and_uuid: str) -> str | None:
    """The UUID in nvidia-smi's one line ``<driver>, GPU-<uuid>``; None when it is not that
    (``driver_version``'s "unknown (...)", or more than one card: every line has its comma)."""
    fields = [f.strip() for f in driver_and_uuid.strip().split(",")]
    if len(fields) != 2:
        return None
    return fields[1] if re.fullmatch(r"(?:GPU|MIG)-[0-9A-Fa-f-]+", fields[1]) else None


def refuse_other_card(smi: str | None, torch_uuid: str | None) -> None:
    """Stop unless nvidia-smi's card and torch's ``cuda:0`` are read and are the same card."""
    if not smi or not torch_uuid:
        raise SystemExit(
            f"[guard] FAILED: the card's UUID reads {smi!r} from nvidia-smi and {torch_uuid!r}"
            " from torch; which card ran cannot be checked"
        )
    if normalise_uuid(smi) != normalise_uuid(torch_uuid):
        raise SystemExit(
            f"[guard] FAILED: nvidia-smi names card {smi}, but torch's cuda:0 is {torch_uuid};"
            " the record would name a card the run did not use (set CUDA_DEVICE_ORDER=PCI_BUS_ID)"
        )


def run_git(cwd: Path, *args: str) -> str | None:
    """``git -C cwd args``: its output, stripped, or None when git fails or is missing."""
    try:
        done = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def code_identity(
    probe_file: Path,
    verbatim_file: str,
    git: Callable[..., str | None] = run_git,
) -> dict[str, Any]:
    """What code this run is: read, not stated.

    ``verbatim_path`` is ``os.path.dirname(verbatim_file)``, resolved. ``checkout`` is the
    directory above the one ``probe_file`` lives in. The git state is read only when that
    directory is the top of a git work tree; otherwise a git above it would describe another
    tree, and it is null. ``--no-optional-locks`` keeps ``git status`` from writing the index
    of a checkout other sessions share."""
    probe = Path(probe_file).resolve()
    checkout = probe.parent.parent
    top = git(checkout, "rev-parse", "--show-toplevel")
    here = top is not None and top != "" and Path(top).resolve() == checkout
    head = git(checkout, "rev-parse", "HEAD") if here else None
    status = (
        git(checkout, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no")
        if head
        else None
    )
    changes = None if status is None else status.splitlines()
    return {
        "verbatim_path": str(Path(os.path.dirname(verbatim_file)).resolve()),
        "checkout": str(checkout),
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "verbatim_commit": head or None,
        "tracked_files_modified": None if changes is None else bool(changes),
        "tracked_changes": changes,
    }


def refuse_code_outside_checkout(code: Mapping[str, Any]) -> None:
    """Stop unless the ``verbatim`` this process imported is this checkout's ``src/verbatim``."""
    expected = str((Path(code["checkout"]) / "src" / "verbatim").resolve())
    if code["verbatim_path"] != expected:
        raise SystemExit(
            f"[guard] FAILED: verbatim is imported from {code['verbatim_path']}, not from this"
            f" probe's checkout ({expected}); the record could not name the code that ran. Put"
            " the checkout's src first on PYTHONPATH"
        )


def transcribe(backend: Any, pipeline: Any, n: int, nid: list[int], audios, pad_to=None):
    """Transcribe rows together; returns (text, timings, nemo_text) per row, in order.

    A row's text is its nonempty step finals, each stripped, joined with one space: the stock
    run's join, not NeMo's concatenation, which glues a final that carries no separator onto
    the one before (see the module docstring). ``nemo_text`` is NeMo's own: every final
    appended as it came, the separator stripped off only while nothing has accumulated, as
    ``BasePipeline.run`` builds a stream's text."""
    rows = []
    for a in audios:
        if pad_to is not None and len(a) < pad_to:
            p = np.zeros(pad_to, dtype=np.float32)
            p[: len(a)] = a
            a = p
        rows.append(np.asarray(a, dtype=np.float32))
    ids = []
    for _ in rows:
        ids.append(nid[0])
        nid[0] += 1
    steps = max(math.ceil(len(a) / n) for a in rows)
    texts = {s: [] for s in ids}
    timings = {s: [] for s in ids}
    nemo = dict.fromkeys(ids, "")
    opts = backend.request_options()
    for step in range(steps):
        frames = []
        for sid, a in zip(ids, rows, strict=True):
            total = math.ceil(len(a) / n)
            if step >= total:
                continue
            piece = a[step * n : step * n + n]
            valid = len(piece)
            if valid < n:
                b = np.zeros(n, dtype=np.float32)
                b[:valid] = piece
                piece = b
            frames.append(
                backend.frame(
                    samples=backend.samples(np.ascontiguousarray(piece)),
                    stream_id=sid,
                    is_first=(step == 0),
                    is_last=(step == total - 1),
                    length=valid,
                    options=opts if step == 0 else None,
                )
            )
        if not frames:
            break
        with backend.no_grad():
            outs = pipeline.transcribe_step(frames)
        for o in outs:
            sid = int(o.stream_id)
            raw = str(o.final_transcript or "")
            txt = raw.strip()
            if txt:
                texts[sid].append(txt)
            nemo[sid] += raw if nemo[sid] else raw.lstrip(NEMO_SEP)
            for seg in o.final_segments or []:
                timings[sid].append(timing_of(seg))
    return [(" ".join(texts[s]).strip(), timings[s], nemo[s]) for s in ids]


# --- the real thing: LibriSpeech, NeMo, torch, a GPU -------------------------------------------


def driver_version() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                visible,
                "--query-gpu=driver_version,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return out.stdout.strip()
    except Exception as exc:  # recorded, not swallowed: an unknown driver is still a fact
        return f"unknown ({exc!r})"


def resolved_revision(model: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model).sha


class NeMoBackend:
    """Everything the probe touches outside itself, for a real run. Built only by ``main()``
    when no backend is injected, so importing the module loads none of it."""

    def __init__(self) -> None:
        import soundfile
        import torch
        from datasets import Audio, load_dataset
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import (
            ASRRequestOptions,
        )

        self._sf = soundfile
        self._torch = torch
        self._audio = Audio
        self._load_dataset = load_dataset
        self.frame = Frame
        self.request_options = ASRRequestOptions

    def resolved_revision(self, model: str) -> str:
        return resolved_revision(model)

    def load(self, count: int) -> tuple[list[np.ndarray], list[dict[str, str]]]:
        ds = self._load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
        ds = ds.cast_column("audio", self._audio(decode=False))
        pool, meta = [], []
        for rec in ds:
            a = rec["audio"]
            if a.get("bytes"):
                raw = a["bytes"]
            else:
                with open(a["path"], "rb") as fh:
                    raw = fh.read()
            data, sr = self._sf.read(io.BytesIO(raw), dtype="float32")
            if sr != 16000:
                raise SystemExit(f"[guard] {rec['id']} is {sr} Hz, not 16000")
            pool.append(np.asarray(data, dtype=np.float32))
            meta.append({"id": rec["id"], "reference": rec["text"].lower()})
            if len(pool) >= count:
                break
        return pool, meta

    def provenance(self) -> dict[str, Any]:
        """The card, read twice: nvidia-smi's for the first card in CUDA_VISIBLE_DEVICES, and
        torch's ``cuda:0``, the card the run computes on. ``main`` refuses a mismatch."""
        torch = self._torch
        smi = driver_version()
        return {
            "machine": torch.cuda.get_device_name(0),
            "capability": "sm_" + "".join(map(str, torch.cuda.get_device_capability(0))),
            "driver_and_uuid": smi,
            "gpu_uuid": smi_uuid(smi),
            "gpu_uuid_torch": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }

    def build(self, spec: NeMoPipelineSpec) -> Any:
        from verbatim.pipelines.nemo_runtime import build_pipeline

        return build_pipeline(spec)

    def nemo_version(self) -> str:
        import nemo

        return nemo.__version__

    def samples(self, piece: np.ndarray) -> Any:
        return self._torch.from_numpy(piece)

    def no_grad(self) -> Any:
        return self._torch.inference_mode()

    def release(self) -> None:
        self._torch.cuda.empty_cache()


# --- the run ----------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    backend: Any = None,
) -> dict[str, Any]:
    """Run the probe and return the record it saved.

    ``backend`` is for the CPU test only. When it is given, the record is stamped
    ``"fake_pipeline": true`` and its ``not_a_row`` says so, whatever the backend is: the only
    unstamped path is the one that builds ``NeMoBackend`` here.
    """
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    fake = backend is not None
    settings = settings_from(env, argv)
    out = settings.out
    batch = settings.batch
    refuse_existing_out(out)
    code = code_identity(PROBE_FILE, verbatim.__file__, run_git)
    refuse_code_outside_checkout(code)
    if backend is None:
        warnings.filterwarnings("ignore")
        backend = NeMoBackend()

    revision = backend.resolved_revision(settings.model)
    if settings.model_revision and not revision.startswith(settings.model_revision):
        raise SystemExit(
            f"[guard] {settings.model} resolves to {revision},"
            f" not the pinned {settings.model_revision}"
        )

    chunk = ChunkMode(settings.chunk_ms)
    att = att_context_size(settings.model, chunk, left=settings.left)

    # The card before the data: a mismatch stops the run before anything is streamed.
    prov = backend.provenance()
    refuse_other_card(prov["gpu_uuid"], prov["gpu_uuid_torch"])

    print("[data] streaming LibriSpeech test-other", flush=True)
    pool, meta = backend.load(settings.n_targets + batch)
    print(f"[data] {len(pool)} utterances held", flush=True)
    refuse_short_pool(len(pool), settings.n_targets, batch)

    state: dict[str, Any] = {}
    written = [False]

    def publish() -> None:
        """The run's first save creates OUT; every later one replaces the run's own record."""
        save(state, out, first=not written[0])
        written[0] = True

    if fake:
        state["fake_pipeline"] = True
    state.update(
        {
            "question": (
                "Does a recording's transcript depend on its batch neighbours, on this checkpoint?"
            ),
            "not_a_row": (
                "FAKE PIPELINE: a CPU test double produced this record; it measures nothing."
                if fake
                else "Stock-pipeline probe (alone vs slot 0 of a batch); not the server, not a"
                " harness row."
            ),
            "model": settings.model,
            "model_revision": revision,
            "machine": prov["machine"],
            "capability": prov["capability"],
            "driver_and_uuid": prov["driver_and_uuid"],
            # Both readings; refuse_other_card stopped the run unless they name one card.
            "gpu_uuid": prov["gpu_uuid"],
            "gpu_uuid_torch": prov["gpu_uuid_torch"],
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": env.get("CUDA_DEVICE_ORDER"),
            "torch": prov["torch"],
            "torch_cuda": prov["torch_cuda"],
            "nemo_commit": env.get("NEMO_COMMIT", "unstated"),
            # Read off the checkout; what the environment said is kept apart.
            "verbatim_commit": code["verbatim_commit"],
            "tracked_files_modified": code["tracked_files_modified"],
            "verbatim_commit_from_env": env.get("VERBATIM_COMMIT", "unstated"),
            "code": code,
            "path": (
                "verbatim.pipelines.nemo_runtime.build_pipeline"
                " -> CacheAwareRNNTPipeline.transcribe_step"
            ),
            # Asked for, as the stock record has it; the built encoder's is *_observed.
            "att_context_size": list(att),
            "chunk_ms": settings.chunk_ms,
            "batch": batch,
            "matmul_precision": settings.matmul,
            "arms": settings.arms,
            "targets": settings.n_targets,
            "pool": len(pool),
            "neighbours": (
                "pool[(i + 1 + k) % len(pool)] for k in range(BATCH - 1), the next recordings"
                " in order"
            ),
            "compared_channels": ["final text", "word timings (final_segments start, end)"],
            "timing_tuple": "(text, start_s, end_s, conf); compared on (text, start_s, end_s) only",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "references": {m["id"]: m["reference"] for m in meta[: settings.n_targets]},
            "runs": {},
        }
    )

    for dtype in settings.dtypes:
        spec = spec_for(settings, dtype, chunk, att)
        print(f"\n[build] {dtype} matmul={settings.matmul} att={att}", flush=True)
        pipeline = backend.build(spec)
        state["nemo"] = backend.nemo_version()
        # What was asked for, read off the spec handed to build_pipeline ...
        state["use_cuda_graphs"] = spec.use_cuda_graphs
        state["word_confidence"] = spec.word_confidence
        # ... and what the built decoder does, read off it. A mismatch stops the run.
        observed = observe_decoder(pipeline)
        refuse_unrequested_decoder(spec.word_confidence, spec.use_cuda_graph_decoder, observed)
        att_observed = observe_att_context(pipeline)
        refuse_unrequested_att_context(spec.att_context, att_observed)
        state["att_context_size_observed"] = att_observed
        print(f"[build] encoder att_context_size {att_observed}", flush=True)
        N = round(float(pipeline.chunk_size_in_secs) * int(pipeline.sample_rate))
        print(f"[build] chunk {pipeline.chunk_size_in_secs}s = {N} samples", flush=True)
        nid = [1]

        def stream(audios, pad_to=None, _pipeline=pipeline, _n=N, _nid=nid):
            return transcribe(backend, _pipeline, _n, _nid, audios, pad_to)

        # Guard: the model transcribes this audio at all, judged on normalised words.
        probe = stream([pool[0], pool[1]])
        ref0, hyp0 = set(words(meta[0]["reference"])), set(words(probe[0][0]))
        ov = len(ref0 & hyp0) / max(1, len(ref0))
        print(
            f"  [guard] ref {meta[0]['reference'][:70]}\n  [guard] hyp {probe[0][0][:70]}\n"
            f"  [guard] overlap {ov:.2f}",
            flush=True,
        )
        if not all(side[0].strip() for side in probe) or ov < 0.5:
            raise SystemExit(
                f"[guard] FAILED: empty transcript or overlap {ov:.2f}; counts would be meaningless"
            )
        # Guard: the timing channel is live. An empty final_segments would make every timing
        # comparison trivially equal, a channel that cannot fail.
        seg0 = probe[0][1]
        if not seg0:
            raise SystemExit(
                "[guard] FAILED: final_segments empty; the timing channel would be blind"
            )
        nonzero = count_nonzero_conf(seg0)
        state["timing_channel"] = {
            "segments_on_guard_recording": len(seg0),
            "words_on_guard_recording": len(probe[0][0].split()),
            "sample": seg0[:3],
            # Observed, not intended. Off must read 0 and the other modes above 0; the guard
            # below stops the run otherwise.
            "nonzero_conf_segments_on_guard_recording": nonzero,
        }
        print(f"  [guard] timing channel live: {len(seg0)} segments, e.g. {seg0[:2]}", flush=True)
        refuse_contradicting_confidence(spec.word_confidence, nonzero)
        word_confidence_observed = word_confidence_stamp(spec.word_confidence, observed, nonzero)
        state["word_confidence_observed"] = word_confidence_observed
        state["decoder_graphs_observed"] = observed["decoder_graphs"]
        state["decoder_graphs_mode_observed"] = observed["decoder_graphs_mode"]
        print(
            f"  [guard] observed {word_confidence_observed}, decoder graphs {observed}", flush=True
        )

        run: dict[str, Any] = {"arms": {}, "repeat": {}}
        state["runs"][dtype] = run
        # The same observations per dtype; the top-level copies are the last dtype built.
        run["word_confidence_observed"] = dict(word_confidence_observed)
        run["decoder_graphs_observed"] = observed["decoder_graphs"]
        run["att_context_size_observed"] = list(att_observed)

        # Repeat control: same input, same shape, twice. Must be identical or the rest is noise.
        rep = {"alone_identical": 0, "batch_identical": 0, "checked": 0}
        for i in range(min(settings.repeat, settings.n_targets)):
            target = pool[i]
            nbrs = [pool[(i + 1 + k) % len(pool)] for k in range(batch - 1)]
            rep["alone_identical"] += same_in_one_shape(stream([target])[0], stream([target])[0])
            rep["batch_identical"] += same_in_one_shape(
                stream([target, *nbrs])[0], stream([target, *nbrs])[0]
            )
            rep["checked"] += 1
        run["repeat"] = rep
        print(f"  [repeat] {rep}", flush=True)
        run["repeat_verdict"] = repeat_verdict(rep)
        publish()

        for arm in settings.arms:
            run["arms"][arm] = new_arm()
        t0 = time.time()
        silence_cache: dict[int, np.ndarray] = {}
        for i in range(settings.n_targets):
            target = pool[i]
            nbrs = [pool[(i + 1 + k) % len(pool)] for k in range(batch - 1)]
            rid = meta[i]["id"]
            results: dict[str, tuple] = {}
            if "ragged" in settings.arms:
                results["ragged"] = (stream([target])[0], stream([target, *nbrs])[0])
            if "equalised" in settings.arms or "fixed" in settings.arms:
                common = max(len(a) for a in [target, *nbrs])
                batched_eq = stream([target, *nbrs], pad_to=common)[0]
                if "equalised" in settings.arms:
                    results["equalised"] = (stream([target], pad_to=common)[0], batched_eq)
                if "fixed" in settings.arms:
                    if common not in silence_cache:
                        silence_cache.clear()
                        silence_cache[common] = np.zeros(common, dtype=np.float32)
                    quiet = [silence_cache[common]] * (batch - 1)
                    results["fixed"] = (stream([target, *quiet], pad_to=common)[0], batched_eq)
            for arm, (a_side, b_side) in results.items():
                record_pair(run["arms"][arm], i, rid, a_side, b_side)
            if (i + 1) % 16 == 0 or i + 1 == settings.n_targets:
                el = time.time() - t0
                now = run["arms"]
                summary = ", ".join(
                    f"{arm} {now[arm]['text_divergent']}+{now[arm]['timing_only_divergent']}t"
                    f"+{now[arm]['confidence_only_divergent']}c"
                    for arm in settings.arms
                )
                print(
                    f"  {dtype} {i + 1}/{settings.n_targets} [{summary}] {el:.0f}s"
                    f" ({el / (i + 1):.1f}s/target)",
                    flush=True,
                )
                run["elapsed_s"] = round(el, 1)
                publish()

        run["positive_control"] = positive_control(run["arms"], settings.arms)
        run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        publish()
        for arm in settings.arms:
            r = run["arms"][arm]
            print(
                f"=== {dtype}/{arm}: text {r['text_divergent']},"
                f" timing-only {r['timing_only_divergent']},"
                f" confidence-only {r['confidence_only_divergent']} of {r['checked']};"
                f" NeMo's own text differs in {r['nemo_text_divergent']}"
                f" ({r['nemo_text_divergent_hidden']} with the probe's texts equal)",
                flush=True,
            )
        print(f"=== {dtype}: positive control {run['positive_control']}", flush=True)
        del pipeline, stream
        backend.release()

    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    publish()
    print("DONE", flush=True)
    return state


if __name__ == "__main__":
    main()
