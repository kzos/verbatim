# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The word-confidence switch: ``NeMoPipelineSpec.word_confidence`` and
``verbatim serve --word-confidence``.

What is held here, and how:

* "off" is the configuration as it was before the switch existed, byte for byte. The
  digests below were computed from ``pipeline_config`` at 6583a83, before the switch was
  added, for three specs that between them flip every other flag the config carries.
* Each other mode changes exactly the NeMo keys it names and no other key.
* The flag parses, reaches the spec, and is refused where it would mean nothing (CTC,
  the fake pipeline).
* A confidence NeMo puts on a segment reaches the Riva wire unchanged. 0.0 stays 0.0,
  which is what "off" sends today.
* The probe stores ``conf``, compares timings on (text, start, end) only, and refuses a
  built decoder or a guard count that contradicts the mode it was asked for.

The last group runs NeMo's own code on the CPU when NeMo is installed: its builder reading
this config into the decoder's confidence config, its pipeline aggregator, its streaming
decoder's 0.0 fill, and a toy label-looping RNNT decode. The toy decode checks that the
switch reaches NeMo's decoder. It proves nothing about transcript digests on a GPU.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from verbatim.cli import EXIT_CONFIG, EXIT_OK, Hooks, main
from verbatim.config import ChunkMode
from verbatim.core.types import PcmFrame, StepResult
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter
from verbatim.pipelines.nemo_fake import FakeCacheAwareRNNTPipeline, boundary_for
from verbatim.pipelines.nemo_runtime import (
    WORD_CONFIDENCE_MODES,
    NeMoPipelineSpec,
    PipelineBuildError,
    RuntimeReport,
    build_pipeline,
    pipeline_config,
)
from verbatim.protocols.base import SessionOptions
from verbatim.protocols.emit import emissions_for
from verbatim.protocols.riva.mapping import final_response
from verbatim.scheduler.graph_budget import ConfigError

ROOT = Path(__file__).resolve().parent.parent
ENGLISH = "nvidia/nemotron-speech-streaming-en-0.6b"
CHUNK = ChunkMode(160)
PFC = ("asr", "decoding", "greedy", "preserve_frame_confidence")

cpu = pytest.mark.cpu


def _spec(**overrides: Any) -> NeMoPipelineSpec:
    args: dict[str, Any] = {
        "model": ENGLISH,
        "chunk": CHUNK,
        "att_context": (70, 1),
        "num_slots": 31,
        "batch_size": 8,
    }
    args.update(overrides)
    return NeMoPipelineSpec(**args)


def _leaves(tree: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Every leaf of a nested dict by its key path, in insertion order."""
    if not isinstance(tree, dict):
        return {prefix: tree}
    out: dict[tuple[str, ...], Any] = {}
    for key, value in tree.items():
        out.update(_leaves(value, (*prefix, key)))
    return out


# --- "off" is today's configuration ------------------------------------------------------

#: sha256 of ``json.dumps(pipeline_config(spec))`` (insertion order, default separators),
#: computed at 6583a83, before ``word_confidence`` existed. ``rnnt_all_on`` flips every
#: other flag, so a mode leaking into a combination shows up here as well.
OFF_BEFORE_THE_SWITCH = {
    "rnnt_default": (
        "482b6f46854990d9155149637ebce2c71df1c1d26449111ffd0efd3eec8578d7",
        {},
    ),
    "ctc": (
        "e3077a67931955041e5c0e381ece8825ebebd59b936bcb4b2dc262e49bca0aee",
        {"decoding": "ctc"},
    ),
    "rnnt_all_on": (
        "0b200b6507f9d0c2aa638e0123cb2efe47b82703f82c43f1feb3412ea3925655",
        {
            "enable_per_stream_biasing": True,
            "use_cuda_graph_decoder": True,
            "use_cuda_graphs": True,
            "compute_dtype": "float32",
            "device_id": 3,
            "stop_history_eou_ms": 0,
        },
    ),
}


@cpu
@pytest.mark.parametrize("name", sorted(OFF_BEFORE_THE_SWITCH))
@pytest.mark.parametrize("explicit", [False, True])
def test_off_is_byte_identical_to_the_configuration_before_the_switch(
    name: str, explicit: bool
) -> None:
    digest, overrides = OFF_BEFORE_THE_SWITCH[name]
    if explicit:
        overrides = {**overrides, "word_confidence": "off"}
    text = json.dumps(pipeline_config(_spec(**overrides)))
    assert hashlib.sha256(text.encode()).hexdigest() == digest, text


@cpu
def test_off_is_the_default_and_the_modes_are_three() -> None:
    assert _spec().word_confidence == "off"
    assert WORD_CONFIDENCE_MODES == ("off", "nemo-shipped", "paper-best")


# --- each mode sets exactly the keys NeMo reads, to exactly the intended values ---------

#: The leaves each mode changes against "off". Nothing else may move.
INTENDED_CHANGES: dict[str, dict[tuple[str, ...], Any]] = {
    "nemo-shipped": {PFC: True},
    "paper-best": {
        PFC: True,
        ("confidence", "aggregation"): "min",
        ("confidence", "method_cfg", "alpha"): 0.33,
    },
}

#: The ``confidence`` block per mode, spelled out. nemo-shipped is NeMo's streaming YAML
#: (cache_aware_rnnt.yaml, ``confidence:``), paper-best is Laptev and Ginsburg 2022.
BLOCKS = {
    "off": {
        "exclude_blank": True,
        "aggregation": "mean",
        "method_cfg": {
            "name": "entropy",
            "entropy_type": "tsallis",
            "alpha": 0.5,
            "entropy_norm": "exp",
        },
    },
    "nemo-shipped": {
        "exclude_blank": True,
        "aggregation": "mean",
        "method_cfg": {
            "name": "entropy",
            "entropy_type": "tsallis",
            "alpha": 0.5,
            "entropy_norm": "exp",
        },
    },
    "paper-best": {
        "exclude_blank": True,
        "aggregation": "min",
        "method_cfg": {
            "name": "entropy",
            "entropy_type": "tsallis",
            "alpha": 0.33,
            "entropy_norm": "exp",
        },
    },
}
PRESERVE = {"off": False, "nemo-shipped": True, "paper-best": True}


@cpu
@pytest.mark.parametrize("mode", ["nemo-shipped", "paper-best"])
@pytest.mark.parametrize(
    "base",
    [{}, {"enable_per_stream_biasing": True, "use_cuda_graph_decoder": True}],
    ids=["plain", "biasing+decoder-graphs"],
)
def test_each_mode_changes_exactly_the_intended_nemo_keys(mode: str, base: dict[str, Any]) -> None:
    off = _leaves(pipeline_config(_spec(**base)))
    on = _leaves(pipeline_config(_spec(**base, word_confidence=mode)))
    assert list(on) == list(off), "a mode added, removed or reordered a key"
    changed = {path: value for path, value in on.items() if off[path] != value}
    assert changed == INTENDED_CHANGES[mode]


@cpu
@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_the_confidence_block_and_flag_are_the_ones_each_mode_names(mode: str) -> None:
    cfg = pipeline_config(_spec(word_confidence=mode))
    assert cfg["confidence"] == BLOCKS[mode]
    assert cfg["asr"]["decoding"]["greedy"]["preserve_frame_confidence"] is PRESERVE[mode]
    # NeMo raises on anything else: the streaming pipelines carry non-blank confidence only.
    assert cfg["confidence"]["exclude_blank"] is True
    # No beam block is written: the strategy is pinned, and the builder reads greedy's flag.
    assert "beam" not in cfg["asr"]["decoding"]
    assert cfg["asr"]["decoding"]["strategy"] == "greedy_batch"


@cpu
def test_the_config_is_a_copy_the_caller_may_edit() -> None:
    first = pipeline_config(_spec(word_confidence="paper-best"))
    first["confidence"]["aggregation"] = "max"
    first["confidence"]["method_cfg"]["alpha"] = 0.9
    again = pipeline_config(_spec(word_confidence="paper-best"))
    assert again["confidence"] == BLOCKS["paper-best"]


@cpu
def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ConfigError, match="word_confidence must be one of"):
        _spec(word_confidence="on")


@cpu
@pytest.mark.parametrize("mode", ["nemo-shipped", "paper-best"])
def test_ctc_refuses_every_mode_but_off(mode: str) -> None:
    with pytest.raises(ConfigError, match="needs decoding='rnnt'"):
        _spec(decoding="ctc", word_confidence=mode)
    assert _spec(decoding="ctc", word_confidence="off").word_confidence == "off"


@cpu
def test_a_refused_build_names_the_word_confidence_it_was_asked_for() -> None:
    def refuse(cfg: Any) -> Any:
        raise RuntimeError("no")

    modules = {
        "omegaconf": type("O", (), {"OmegaConf": type("C", (), {"create": staticmethod(dict)})}),
        "nemo.collections.asr.inference.factory.pipeline_builder": type(
            "M", (), {"PipelineBuilder": type("B", (), {"build_pipeline": staticmethod(refuse)})}
        ),
    }
    with pytest.raises(PipelineBuildError, match="word confidence paper-best"):
        build_pipeline(_spec(word_confidence="paper-best"), modules.__getitem__)


# --- the CLI flag -------------------------------------------------------------------------

RELEASED = RuntimeReport(
    nemo_version="3.1.0",
    torch_version="2.11.0",
    cuda_available=True,
    device_name="NVIDIA RTX A6000",
    inference_package=True,
    graph_step=False,
)
SERVE = ["serve", ENGLISH, "--chunk", "160ms", "--bucket", "8", "--eager"]


class _Seen:
    def __init__(self) -> None:
        import io

        self.specs: list[NeMoPipelineSpec] = []
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()


def _hooks(seen: _Seen) -> Hooks:
    async def no_server(*_: Any, **__: Any) -> None:
        return None

    def build(spec: NeMoPipelineSpec) -> Any:
        seen.specs.append(spec)
        return boundary_for(
            FakeCacheAwareRNNTPipeline(spec.chunk.ms, num_slots=max(64, spec.num_slots))
        )

    return Hooks(
        inspect_runtime=lambda: RELEASED,
        build_boundary=build,
        run_server=no_server,
        stdout=seen.stdout,
        stderr=seen.stderr,
    )


@cpu
def test_without_the_flag_the_spec_is_off_and_the_banner_says_nothing() -> None:
    seen = _Seen()
    assert main(SERVE, hooks=_hooks(seen)) == EXIT_OK
    assert [spec.word_confidence for spec in seen.specs] == ["off"]
    assert "confidence" not in seen.stdout.getvalue()


@cpu
@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_the_flag_parses_and_reaches_the_spec(mode: str) -> None:
    seen = _Seen()
    assert main([*SERVE, "--word-confidence", mode], hooks=_hooks(seen)) == EXIT_OK
    assert [spec.word_confidence for spec in seen.specs] == [mode]


@cpu
def test_the_banner_names_the_measure_and_calls_it_a_separate_configuration() -> None:
    seen = _Seen()
    assert main([*SERVE, "--word-confidence", "paper-best"], hooks=_hooks(seen)) == EXIT_OK
    out = seen.stdout.getvalue()
    assert (
        "word confidence ON, paper-best: tsallis entropy, exp normalisation, alpha 0.33, "
        "min over a word's tokens." in out
    )
    assert "SEPARATE CONFIGURATION" in out and "NOT proven" in out

    shipped = _Seen()
    assert main([*SERVE, "--word-confidence", "nemo-shipped"], hooks=_hooks(shipped)) == EXIT_OK
    assert "alpha 0.5, mean over a word's tokens." in shipped.stdout.getvalue()


@cpu
def test_an_unknown_value_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    seen = _Seen()
    assert main([*SERVE, "--word-confidence", "on"], hooks=_hooks(seen)) == EXIT_CONFIG
    assert "invalid choice: 'on'" in capsys.readouterr().err
    assert seen.specs == []


@cpu
def test_ctc_with_confidence_is_refused_before_any_model_loads() -> None:
    seen = _Seen()
    argv = [*SERVE, "--pipeline", "cache_aware_ctc", "--word-confidence", "paper-best"]
    assert main(argv, hooks=_hooks(seen)) == EXIT_CONFIG
    assert "needs decoding='rnnt'" in seen.stderr.getvalue()
    assert seen.specs == []


@cpu
def test_the_fake_pipeline_with_confidence_is_refused() -> None:
    seen = _Seen()
    argv = ["serve", ENGLISH, "--chunk", "160ms", "--bucket", "8", "--pipeline", "fake"]
    assert main([*argv, "--word-confidence", "nemo-shipped"], hooks=_hooks(seen)) == EXIT_CONFIG
    assert "needs a NeMo pipeline" in seen.stderr.getvalue()
    # Off is what the fake always served, with or without the flag spelled out.
    served = _Seen()
    assert main([*argv, "--word-confidence", "off"], hooks=_hooks(served)) == EXIT_OK
    assert "FAKE" in served.stdout.getvalue()


# --- the confidence reaches the wire ------------------------------------------------------


class _ScriptedConfidence(FakeCacheAwareRNNTPipeline):
    """The NeMo-shaped fake, with each final segment's ``conf`` taken from a script: what
    NeMo's ``BPEDecoder`` would have aggregated for that word."""

    def __init__(self, confs: list[float], **kwargs: Any) -> None:
        super().__init__(CHUNK.ms, **kwargs)
        self._confs = iter(confs)

    def transcribe_step(self, requests: list[Any]) -> list[Any]:
        outputs = super().transcribe_step(requests)
        for output in outputs:
            for segment in output.final_segments:
                segment.conf = next(self._confs)
        return outputs


def _two_word_final(confs: list[float]) -> StepResult:
    pipeline = _ScriptedConfidence(confs, num_slots=64)
    adapter = CacheAwareRNNTAdapter(CHUNK, boundary_for(pipeline), buckets=(4,), required_slots=8)
    adapter.open_stream(1, None)
    rng = np.random.default_rng(3)
    speech = [rng.uniform(-0.5, 0.5, size=CHUNK.samples).astype(np.float32) for _ in range(2)]
    first = PcmFrame(1, speech[0], is_first=True, is_last=False, valid_samples=CHUNK.samples)
    last = PcmFrame(1, speech[1], is_first=False, is_last=True, valid_samples=CHUNK.samples)
    adapter.transcribe_step([first], keep_all_outputs=False, graph=False)
    (result,) = adapter.transcribe_step([last], keep_all_outputs=True, graph=False)
    return result


@cpu
@pytest.mark.parametrize(
    ("confs", "mean"),
    [([0.0, 0.0], 0.0), ([0.25, 0.75], 0.5)],
    ids=["off-sends-zero", "on-sends-what-nemo-computed"],
)
def test_a_segments_confidence_reaches_the_riva_words_unchanged(
    confs: list[float], mean: float
) -> None:
    result = _two_word_final(confs)
    assert [word.confidence for word in result.words] == confs
    assert result.confidence == mean
    emission = emissions_for(result, SessionOptions())
    (final,) = [h for h in emission.hypotheses if h.is_final]
    response = final_response(final, request_id="r", word_timestamps=True)
    alternative = response.results[0].alternatives[0]
    # 0.0, 0.25, 0.5, 0.75 are exact in the proto's float32.
    assert [info.confidence for info in alternative.words] == confs
    assert alternative.confidence == mean


# --- the probe ----------------------------------------------------------------------------
#
# The probe is imported, not parsed: importing it loads no torch, NeMo, model or data. These
# are the confidence-specific checks; the arms, the controls and main() end to end are in
# tests/test_stock_divergence.py.

PROBE = ROOT / "probes" / "stock_divergence.py"


def _probe() -> Any:
    spec = importlib.util.spec_from_file_location("stock_divergence_confidence", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a dataclass resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


probe = _probe()


@cpu
def test_the_probe_compares_timings_without_the_stored_conf() -> None:
    a = ("hello world", [("hello", 0.0, 0.16, 0.0), ("world", 0.16, 0.32, 0.0)])
    b = ("hello world", [("hello", 0.0, 0.16, 0.91), ("world", 0.16, 0.32, 0.4)])
    assert probe.comparable(a) == probe.comparable(b)
    moved = ("hello world", [("hello", 0.0, 0.16, 0.0), ("world", 0.24, 0.32, 0.0)])
    assert probe.comparable(a) != probe.comparable(moved)
    assert probe.comparable(a) == ("hello world", [("hello", 0.0, 0.16), ("world", 0.16, 0.32)])
    # The arms and the repeat control see the conf-only pair as the same text and timings,
    # and the arms count it apart.
    assert probe.classify(a, b) == "confidence"
    assert probe.classify(a, moved) == "timing"
    assert probe.same_in_one_shape(a, b)
    assert not probe.same_in_one_shape(a, moved)


@cpu
def test_the_probe_stores_the_segments_conf() -> None:
    segment = type("Seg", (), {"text": " world ", "start": 0.16, "end": 0.32, "conf": 0.4})()
    assert probe.timing_of(segment) == ("world", 0.16, 0.32, 0.4)


@cpu
@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_word_confidence_reaches_the_probes_spec(mode: str) -> None:
    settings = probe.settings_from({"OUT": "unused", "WORD_CONFIDENCE": mode}, ["probe"])
    spec = probe.spec_for(settings, "bfloat16", ChunkMode(1120), [70, 13])
    assert spec.word_confidence == mode
    assert probe.settings_from({"OUT": "unused"}, ["probe"]).word_confidence == "off"


@cpu
@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_the_probe_refuses_a_decoder_or_a_count_that_contradicts_the_mode(mode: str) -> None:
    on = mode != "off"
    seen = {"decoder_step_confidence": on, "decoder_graphs": False, "decoder_graphs_mode": None}
    probe.refuse_unrequested_decoder(mode, False, seen)
    with pytest.raises(SystemExit, match="preserve_step_confidence"):
        probe.refuse_unrequested_decoder(mode, False, {**seen, "decoder_step_confidence": not on})
    probe.refuse_contradicting_confidence(mode, 3 if on else 0)
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED"):
        probe.refuse_contradicting_confidence(mode, 0 if on else 3)


# --- NeMo's own code, on the CPU, when NeMo is installed ----------------------------------
#
# Not marked cpu: that marker means "no NeMo". Nothing below touches a GPU: every tensor is
# created on the CPU, and ``_nemo`` refuses to run unless CUDA_VISIBLE_DEVICES is "".

EXPECTED_METHOD = {
    "nemo-shipped": ("mean", 0.5),
    "paper-best": ("min", 0.33),
}


def _nemo(
    find_spec: Any = importlib.util.find_spec, environ: Mapping[str, str] = os.environ
) -> Any:
    """NeMo's builder and OmegaConf, for the tests below.

    Skips only when NeMo is not installed at all. An installed NeMo that fails to import is a
    failure, not a skip: a quiet skip there would read as a pass. And these tests refuse to
    run where a GPU is visible: they build everything on the CPU, and on a shared box the
    cards belong to other jobs, so ``CUDA_VISIBLE_DEVICES`` must be set to the empty string.
    """
    if find_spec("nemo") is None:
        pytest.skip("NeMo is not installed")
    if environ.get("CUDA_VISIBLE_DEVICES") != "":
        pytest.fail(
            "these tests run NeMo on the CPU and must not see a GPU: run them with"
            ' CUDA_VISIBLE_DEVICES=""'
        )
    try:
        from nemo.collections.asr.inference.factory.cache_aware_pipeline_builder import (
            CacheAwarePipelineBuilder,
        )
        from omegaconf import OmegaConf
    except ImportError as exc:
        pytest.fail(f"NeMo is installed but its pipeline builder does not import: {exc!r}")
    return OmegaConf, CacheAwarePipelineBuilder


@cpu
def test_the_nemo_helper_skips_only_when_nemo_is_absent_and_fails_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = lambda name: object()  # noqa: E731
    with pytest.raises(BaseException) as absent:
        _nemo(find_spec=lambda name: None, environ={"CUDA_VISIBLE_DEVICES": ""})
    assert absent.type is pytest.skip.Exception
    with pytest.raises(BaseException) as gpu:
        _nemo(find_spec=installed, environ={"CUDA_VISIBLE_DEVICES": "3"})
    assert gpu.type is pytest.fail.Exception
    with pytest.raises(BaseException) as unset:
        _nemo(find_spec=installed, environ={})
    assert unset.type is pytest.fail.Exception
    # Installed but broken: a None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(
        sys.modules, "nemo.collections.asr.inference.factory.cache_aware_pipeline_builder", None
    )
    with pytest.raises(BaseException) as broken:
        _nemo(find_spec=installed, environ={"CUDA_VISIBLE_DEVICES": ""})
    assert broken.type is pytest.fail.Exception, "a broken NeMo must fail, not skip"
    assert "does not import" in str(broken.value)


@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_nemos_builder_reads_the_switch_into_the_decoders_confidence_config(mode: str) -> None:
    omegaconf, builder = _nemo()
    cfg = omegaconf.create(pipeline_config(_spec(word_confidence=mode)))
    confidence = builder.get_rnnt_decoding_cfg(cfg).confidence_cfg
    if mode == "off":
        # _apply_confidence_cfg returned early: nothing is preserved, nothing computed.
        assert not confidence.preserve_frame_confidence
        assert not confidence.preserve_word_confidence
        return
    aggregation, alpha = EXPECTED_METHOD[mode]
    assert confidence.preserve_frame_confidence is True
    assert confidence.preserve_token_confidence is True
    assert confidence.preserve_word_confidence is True
    assert confidence.exclude_blank is True
    assert confidence.aggregation == aggregation
    assert confidence.method_cfg.name == "entropy"
    assert confidence.method_cfg.entropy_type == "tsallis"
    assert confidence.method_cfg.entropy_norm == "exp"
    assert confidence.method_cfg.alpha == alpha


@pytest.mark.parametrize("mode", ["nemo-shipped", "paper-best"])
def test_nemos_pipeline_aggregates_words_with_the_modes_measure(mode: str) -> None:
    omegaconf, _ = _nemo()
    from nemo.collections.asr.inference.utils.pipeline_utils import get_confidence_utils

    cfg = omegaconf.create(pipeline_config(_spec(word_confidence=mode)))
    conf_func, aggregate = get_confidence_utils(cfg.confidence)
    aggregation, alpha = EXPECTED_METHOD[mode]
    assert aggregate([0.25, 0.75]) == {"mean": 0.5, "min": 0.25}[aggregation]
    assert conf_func.keywords == {"t": alpha}


def test_nemos_streaming_decoder_fills_zero_when_no_confidence_was_computed() -> None:
    _nemo()
    from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import (
        RNNTGreedyDecoder,
    )

    decoder = RNNTGreedyDecoder(vocabulary=["a", "b", "c"])
    off, _, _ = decoder(global_timestamps=[0, 1, 3], tokens=[0, 1, 2], length=4, confidences=None)
    assert off["confidences"] == [0.0, 0.0, 0.0]
    on, _, _ = decoder(
        global_timestamps=[0, 1, 3], tokens=[0, 1, 2], length=4, confidences=[0.25, 0.5, 0.75]
    )
    assert on["confidences"] == [0.25, 0.5, 0.75]


@pytest.fixture(scope="module")
def toy_decode() -> Any:
    """A seeded toy RNNT decoder and joint, decoded through NeMo's ``RNNTDecoding`` with
    the decoding config NeMo's own builder derives from ``pipeline_config``."""
    omegaconf, builder = _nemo()
    import torch
    from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding

    vocab = [chr(ord("a") + i) for i in range(7)]
    torch.manual_seed(0)
    decoder = RNNTDecoder(prednet={"pred_hidden": 16, "pred_rnn_layers": 1}, vocab_size=len(vocab))
    joint = RNNTJoint(
        jointnet={
            "encoder_hidden": 12,
            "pred_hidden": 16,
            "joint_hidden": 24,
            "activation": "relu",
        },
        num_classes=len(vocab),
    )
    decoder.eval()
    joint.eval()
    with torch.no_grad():  # a little more blank, so hypotheses are not every-frame max_symbols
        joint.joint_net[-1].bias[len(vocab)] += 0.4
    encoded = torch.randn(3, 12, 20)
    lengths = torch.tensor([20, 15, 9])

    def run(mode: str) -> tuple[Any, list[Any]]:
        cfg = omegaconf.create(pipeline_config(_spec(word_confidence=mode)))
        decoding = RNNTDecoding(
            builder.get_rnnt_decoding_cfg(cfg), decoder=decoder, joint=joint, vocabulary=vocab
        )
        with torch.inference_mode():
            hyps = decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=lengths, return_hypotheses=True
            )
        return decoding, hyps

    return {mode: run(mode) for mode in WORD_CONFIDENCE_MODES}


def test_toy_decode_off_computes_no_confidence(toy_decode: Any) -> None:
    decoding, hyps = toy_decode["off"]
    assert decoding.decoding.decoding_computer.preserve_step_confidence is False
    assert all(len(h.y_sequence) > 0 for h in hyps), "the toy decoded nothing; it proves nothing"
    assert all(h.non_blank_step_confidence_precomputed is None for h in hyps)


@pytest.mark.parametrize("mode", ["nemo-shipped", "paper-best"])
def test_toy_decode_on_computes_one_confidence_per_token_with_the_modes_alpha(
    toy_decode: Any, mode: str
) -> None:
    decoding, hyps = toy_decode[mode]
    computer = decoding.decoding.decoding_computer
    assert computer.preserve_step_confidence is True
    assert computer.alpha == EXPECTED_METHOD[mode][1]
    assert decoding.word_confidence_aggregation == EXPECTED_METHOD[mode][0]
    for h in hyps:
        confs = h.non_blank_step_confidence_precomputed
        assert confs is not None and len(confs) == len(h.y_sequence)
        assert all(0.0 <= c <= 1.0 for c in confs)
        assert any(c != 0.0 for c in confs)


def test_toy_decode_the_two_measures_differ_and_the_tokens_do_not(toy_decode: Any) -> None:
    """The alpha reached the decoder: the two measures give different numbers. On this toy,
    on the CPU, the tokens are the same in all three modes. That is a sanity check of the
    argmax-before-confidence reading, not the GPU proof the spec's docstring asks for."""
    _, shipped = toy_decode["nemo-shipped"]
    _, paper = toy_decode["paper-best"]
    assert [h.non_blank_step_confidence_precomputed for h in shipped] != [
        h.non_blank_step_confidence_precomputed for h in paper
    ]
    tokens = {mode: [h.y_sequence.tolist() for h in hyps] for mode, (_, hyps) in toy_decode.items()}
    assert tokens["off"] == tokens["nemo-shipped"] == tokens["paper-best"]


# --- the probe reads NeMo's own decoding computer ------------------------------------------


def _built(decoding: Any) -> Any:
    """Shaped like a built ``CacheAwareRNNTPipeline`` down to the model's ``decoding``."""
    return SimpleNamespace(asr_model=SimpleNamespace(asr_model=SimpleNamespace(decoding=decoding)))


@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_the_probe_observes_nemos_own_decoding_computer(toy_decode: Any, mode: str) -> None:
    from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import (
        CacheAwareRNNTPipeline,
    )

    decoding, _ = toy_decode[mode]
    built = _built(decoding)
    # NeMo's own lookup finds the object the probe walks to.
    CacheAwareRNNTPipeline.init_decoding_computer(built)
    assert built.decoding_computer is decoding.decoding.decoding_computer
    seen = probe.observe_decoder(built)
    assert seen == {
        "decoder_step_confidence": mode != "off",
        "decoder_graphs": False,
        "decoder_graphs_mode": None,
    }
    probe.refuse_unrequested_decoder(mode, False, seen)


def test_the_probe_sees_decoder_graphs_when_nemo_turns_them_on(toy_decode: Any) -> None:
    omegaconf, builder = _nemo()
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding

    greedy = toy_decode["off"][0].decoding
    cfg = omegaconf.create(pipeline_config(_spec(use_cuda_graph_decoder=True)))
    decoding = RNNTDecoding(
        builder.get_rnnt_decoding_cfg(cfg),
        decoder=greedy.decoder,
        joint=greedy.joint,
        vocabulary=[chr(ord("a") + i) for i in range(7)],
    )
    seen = probe.observe_decoder(_built(decoding))
    assert seen["decoder_graphs"] is True
    assert seen["decoder_graphs_mode"] in ("full_graph", "no_while_loops")
    probe.refuse_unrequested_decoder("off", True, seen)
    with pytest.raises(SystemExit, match="decoder graphs False"):
        probe.refuse_unrequested_decoder("off", False, seen)


def test_the_probes_real_backend_hands_nemo_its_own_frames() -> None:
    """``NeMoBackend`` on the CPU, through the probe's ``transcribe``: NeMo's ``Frame`` and
    ``ASRRequestOptions``, CPU tensors, inference mode, the chunking and the padding. Only
    the pipeline is a recorder; nothing is downloaded and no GPU is visible."""
    _nemo()
    import torch
    from nemo.collections.asr.inference.streaming.framing.request import Frame
    from nemo.collections.asr.inference.streaming.framing.request_options import (
        ASRRequestOptions,
    )

    backend = probe.NeMoBackend()
    steps: list[list[Any]] = []
    inference: list[bool] = []

    #: What each stream finalises at each step, NeMo-shaped (``TranscribeStepOutput.from_state``):
    #: a leading separator after the stream's first request, "" and no segments on a step that
    #: finalises nothing.
    script = {
        1: [
            ("I'm", [SimpleNamespace(text="I'm", start=0.08, end=0.4, conf=0.25)]),
            ("", []),
            (
                " from the",
                [
                    SimpleNamespace(text=" from", start=0.4, end=0.56, conf=0.5),
                    SimpleNamespace(text="the", start=0.56, end=0.64, conf=0.125),
                ],
            ),
        ],
        2: [("w", [SimpleNamespace(text="w", start=0.0, end=0.1, conf=0.25)])],
    }

    class Recorder:
        def transcribe_step(self, frames: list[Any]) -> list[Any]:
            steps.append(frames)
            inference.append(torch.is_inference_mode_enabled())
            outs = []
            for f in frames:
                text, segments = script[f.stream_id].pop(0)
                outs.append(
                    SimpleNamespace(
                        stream_id=f.stream_id, final_transcript=text, final_segments=segments
                    )
                )
            return outs

    nid = [1]
    rows = [np.arange(1, 41, dtype=np.float32), np.arange(1, 17, dtype=np.float32)]
    out = probe.transcribe(backend, Recorder(), 16, nid, rows)
    assert [[(f.stream_id, f.is_first, f.is_last, f.length) for f in step] for step in steps] == [
        [(1, True, False, 16), (2, True, True, 16)],
        [(1, False, False, 16)],
        [(1, False, True, 8)],
    ]
    frames = [f for step in steps for f in step]
    assert all(type(f) is Frame for f in frames)
    assert all(
        isinstance(f.samples, torch.Tensor) and f.samples.device.type == "cpu" for f in frames
    )
    assert all(f.samples.dtype == torch.float32 for f in frames), "NeMo is handed float32"
    assert isinstance(steps[0][0].options, ASRRequestOptions)
    assert steps[0][0].options is steps[0][1].options, "one options object per call"
    assert steps[1][0].options is None
    assert steps[2][0].samples.tolist() == [float(v) for v in range(33, 41)] + [0.0] * 8
    assert inference == [True, True, True]
    assert out == [
        (
            "I'm from the",
            [("I'm", 0.08, 0.4, 0.25), ("from", 0.4, 0.56, 0.5), ("the", 0.56, 0.64, 0.125)],
            "I'm from the",
        ),
        ("w", [("w", 0.0, 0.1, 0.25)], "w"),
    ]
    assert nid == [3]


def test_nemo_gives_a_continued_word_no_separator_and_the_probe_space_joins_it() -> None:
    """NeMo's own ``StreamingState`` and ``TranscribeStepOutput.from_state`` make the finals.
    A step whose first word continues the last word finalised is pushed with
    ``merge_first_word``, which leaves ``concat_with_space`` False (``state.py:350-354``), and
    its final then carries no separator (``base_pipeline.py:96-101``). NeMo's concatenation of
    the finals reads "they"; the probe's per-step space join reads "the y", as the stock record
    does at n=327 (the words and timings here are that record's). Pinned as it is: the replay
    must reproduce the stock run's text. The probe returns NeMo's own concatenation beside it,
    and that reads "they"."""
    final = _nemo_final()
    outs = [
        final(
            [("so", 13.84, 14.0), ("ill", 14.0, 14.08), ("that", 14.48, 14.56)],
            is_first=True,
            continues=False,
        ),
        final([("the", 15.6, 15.68)], is_first=False, continues=False),
        final(
            [("y", 15.68, 15.76), ("are", 15.76, 15.84), ("not", 15.84, 15.92)],
            is_first=False,
            continues=True,
        ),
    ]
    finals = [o.final_transcript for o in outs]
    assert finals == ["so ill that", " the", "y are not"]
    assert "".join(finals) == "so ill that they are not", "NeMo's own concatenation"

    class Replayer:
        def transcribe_step(self, frames: list[Any]) -> list[Any]:
            assert [f.stream_id for f in frames] == [1]
            return [outs.pop(0)]

    out = probe.transcribe(probe.NeMoBackend(), Replayer(), 16, [1], [np.ones(48, np.float32)])
    assert outs == [], "one final per step"
    assert out == [
        (
            "so ill that the y are not",
            [
                ("so", 13.84, 14.0, 0.0),
                ("ill", 14.0, 14.08, 0.0),
                ("that", 14.48, 14.56, 0.0),
                ("the", 15.6, 15.68, 0.0),
                ("y", 15.68, 15.76, 0.0),
                ("are", 15.76, 15.84, 0.0),
                ("not", 15.84, 15.92, 0.0),
            ],
            "so ill that they are not",
        )
    ]


def _nemo_final() -> Any:
    """A step's output as NeMo makes it: its own ``StreamingState.push_back_words`` sets
    ``concat_with_space`` and its own ``TranscribeStepOutput.from_state`` adds the separator."""
    _nemo()
    from nemo.collections.asr.inference.pipelines.base_pipeline import TranscribeStepOutput
    from nemo.collections.asr.inference.streaming.state.state import StreamingState
    from nemo.collections.asr.inference.utils.text_segment import TextSegment, Word

    def final(words: list[tuple[str, float, float]], *, is_first: bool, continues: bool) -> Any:
        state = StreamingState()
        state.push_back_words([Word(w, s, e, 0.0) for w, s, e in words], merge_first_word=continues)
        state.final_transcript = " ".join(w for w, _, _ in words)
        state.final_segments = [TextSegment(w, s, e, 0.0) for w, s, e in words]
        request = SimpleNamespace(is_first=is_first, stream_id=1)
        return TranscribeStepOutput.from_state(state, request)

    return final


def test_the_join_can_hide_a_word_boundary_nemos_own_text_shows() -> None:
    """The other direction. Side a's third step continues the last word, side b's starts a new
    one; the words and timings are the same. NeMo's own concatenations differ ("they" and "the
    y"), and the probe's space join reads "the y" on both sides, so ``classify`` calls the pair
    identical and no count the stock run kept can show it. The record keeps NeMo's texts and
    counts the pair as hidden by the join."""
    final = _nemo_final()
    steps = [
        [("so", 13.84, 14.0), ("that", 14.48, 14.56)],
        [("the", 15.6, 15.68)],
        [("y", 15.68, 15.76), ("are", 15.76, 15.84)],
    ]

    def side(continues: bool) -> tuple[Any, list[str]]:
        outs = [
            final(steps[0], is_first=True, continues=False),
            final(steps[1], is_first=False, continues=False),
            final(steps[2], is_first=False, continues=continues),
        ]
        finals = [o.final_transcript for o in outs]

        class Replayer:
            def transcribe_step(self, frames: list[Any]) -> list[Any]:
                return [outs.pop(0)]

        (row,) = probe.transcribe(
            probe.NeMoBackend(), Replayer(), 16, [1], [np.ones(48, np.float32)]
        )
        return row, finals

    a_side, a_finals = side(True)
    b_side, b_finals = side(False)
    assert a_finals == ["so that", " the", "y are"]
    assert b_finals == ["so that", " the", " y are"]
    assert (a_side[2], b_side[2]) == ("so that they are", "so that the y are"), "NeMo's own"
    assert a_side[0] == b_side[0] == "so that the y are", "the probe's join"
    assert a_side[1] == b_side[1]
    assert probe.classify(a_side, b_side) == "identical"
    arm = probe.new_arm()
    assert probe.record_pair(arm, 0, "rid-0", a_side, b_side) == "identical"
    assert (arm["text_divergent"], arm["timing_only_divergent"], arm["divergences"]) == (0, 0, [])
    assert (arm["nemo_text_divergent"], arm["nemo_text_divergent_hidden"]) == (1, 1)
    entry = arm["every_recording"]["rid-0"]
    assert (entry["a_nemo_text"], entry["b_nemo_text"]) == ("so that they are", "so that the y are")


def test_the_probe_reads_the_context_nemos_encoder_keeps() -> None:
    """A real ``ConformerEncoder``, tiny, on the CPU. NeMo's pipeline hands the encoder its
    ``streaming.att_context_size``, an OmegaConf list, through ``set_default_att_context_size``
    (``cache_aware_rnnt_pipeline.py:118-119``); the probe reads it back where the wrapper does.
    A context the encoder does not list is kept too, with only a warning, and the probe then
    refuses the run."""
    omegaconf, _ = _nemo()
    from nemo.collections.asr.modules import ConformerEncoder

    encoder = ConformerEncoder(
        feat_in=8,
        n_layers=1,
        d_model=8,
        n_heads=2,
        att_context_size=[[70, 13], [70, 1]],
        att_context_style="chunked_limited",
        conv_context_size="causal",
        subsampling_factor=8,
        subsampling_conv_channels=8,
    )
    built = SimpleNamespace(asr_model=SimpleNamespace(asr_model=SimpleNamespace(encoder=encoder)))
    assert probe.observe_att_context(built) == [70, 13], "the first listed context by default"
    encoder.set_default_att_context_size(att_context_size=omegaconf.create({"a": [70, 1]}).a)
    assert type(encoder.att_context_size).__name__ == "ListConfig"
    assert probe.observe_att_context(built) == [70, 1]
    probe.refuse_unrequested_att_context((70, 1), probe.observe_att_context(built))
    encoder.set_default_att_context_size(att_context_size=omegaconf.create({"a": [70, 6]}).a)
    assert probe.observe_att_context(built) == [70, 6], "NeMo kept a context it does not list"
    with pytest.raises(SystemExit, match=r"att_context_size \[70, 1\], but the built encoder"):
        probe.refuse_unrequested_att_context((70, 1), probe.observe_att_context(built))
    # A left context of its own, the look-ahead as asked: kept, read, and refused.
    encoder.set_default_att_context_size(att_context_size=omegaconf.create({"a": [56, 1]}).a)
    assert probe.observe_att_context(built) == [56, 1]
    with pytest.raises(SystemExit, match=r"\[70, 1\], but the built encoder's .* is \[56, 1\]"):
        probe.refuse_unrequested_att_context((70, 1), probe.observe_att_context(built))


@pytest.mark.parametrize("mode", ["nemo-shipped", "paper-best"])
def test_the_other_modes_measure_would_turn_the_nemo_checks_red(mode: str) -> None:
    """The aggregation/alpha mutation, kept in the suite: the block of ``mode`` with the other
    mode's aggregation and alpha is read by NeMo as a different measure, so the checks above
    can tell the two modes apart."""
    omegaconf, builder = _nemo()
    from nemo.collections.asr.inference.utils.pipeline_utils import get_confidence_utils

    other = {"nemo-shipped": "paper-best", "paper-best": "nemo-shipped"}[mode]
    cfg = pipeline_config(_spec(word_confidence=mode))
    cfg["confidence"]["aggregation"] = EXPECTED_METHOD[other][0]
    cfg["confidence"]["method_cfg"]["alpha"] = EXPECTED_METHOD[other][1]
    mutated = omegaconf.create(cfg)
    aggregation, alpha = EXPECTED_METHOD[mode]
    conf_func, aggregate = get_confidence_utils(mutated.confidence)
    assert aggregate([0.25, 0.75]) != {"mean": 0.5, "min": 0.25}[aggregation]
    assert conf_func.keywords != {"t": alpha}
    confidence = builder.get_rnnt_decoding_cfg(mutated).confidence_cfg
    assert (confidence.aggregation, confidence.method_cfg.alpha) != (aggregation, alpha)


# --- every caller builds the spec with keywords -------------------------------------------

#: model, chunk, att_context, num_slots, batch_size: the fields before the defaults.
POSITIONAL_FIELDS = 5


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _spec_calls(source: str, filename: str) -> tuple[int, list[int]]:
    """How many ``NeMoPipelineSpec(...)`` calls a source has, and the lines of those that
    pass a field past ``batch_size`` by position (or splat a sequence into it)."""
    calls, offenders = 0, []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Call) and _callee(node) == "NeMoPipelineSpec":
            calls += 1
            starred = any(isinstance(arg, ast.Starred) for arg in node.args)
            if starred or len(node.args) > POSITIONAL_FIELDS:
                offenders.append(node.lineno)
    return calls, offenders


@cpu
def test_every_nemo_pipeline_spec_in_the_repo_names_its_optional_fields() -> None:
    """``word_confidence`` was inserted among the defaulted fields of a dataclass that is not
    keyword-only, so a caller passing them by position would shift every later field without
    an error. None may: past ``batch_size``, every argument is a keyword."""
    # The scan itself can fail: one field too many by position, or a splat, is caught.
    assert _spec_calls("NeMoPipelineSpec(m, c, a, 8, 8, 'rnnt')", "sixth") == (1, [1])
    assert _spec_calls("x.NeMoPipelineSpec(*args)", "splat") == (1, [1])
    assert _spec_calls("NeMoPipelineSpec(m, c, a, 8, 8, decoding='rnnt')", "ok") == (1, [])
    calls, offenders = 0, []
    for folder in ("src", "bench", "probes", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            found, lines = _spec_calls(path.read_text(encoding="utf-8"), str(path))
            calls += found
            offenders += [f"{path.relative_to(ROOT)}:{line}" for line in lines]
    assert calls >= 3, f"found only {calls} NeMoPipelineSpec calls; the scan is not looking"
    assert offenders == []
