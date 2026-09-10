# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The NeMo runtime seam, tested without NeMo: the configuration `serve` hands to
NeMo's builder, the attention-context arithmetic, the pre-flight report against
both installed tracks' shapes and the PR's, and what a refused build says."""

from __future__ import annotations

import types
from typing import Any

import pytest

from verbatim.config import ChunkMode
from verbatim.pipelines.nemo_runtime import (
    GRAPH_STEP_MODULE,
    WRAPPER_MODULE,
    NeMoPipelineSpec,
    PipelineBuildError,
    att_context_size,
    build_pipeline,
    inspect_runtime,
    pipeline_config,
)
from verbatim.scheduler.graph_budget import ConfigError

pytestmark = pytest.mark.cpu

MULTILINGUAL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
ENGLISH = "nvidia/nemotron-speech-streaming-en-0.6b"

#: Top-level keys of NeMo's own example config that its builder and pipeline read at
#: construction (Speech 353d190, examples/asr/conf/asr_streaming_inference/
#: cache_aware_rnnt.yaml). `nmt` and `metrics` are read only when enabled or by the
#: evaluation script, and are left out on purpose.
REQUIRED_TOP_LEVEL = {
    "asr",
    "itn",
    "confidence",
    "endpointing",
    "streaming",
    "matmul_precision",
    "log_level",
    "pipeline_type",
    "asr_decoding_type",
    "enable_itn",
    "enable_nmt",
    "asr_output_granularity",
    "cache_dir",
    "lang",
    "return_tail_result",
}


def _spec(**overrides: Any) -> NeMoPipelineSpec:
    args: dict[str, Any] = {
        "model": MULTILINGUAL,
        "chunk": ChunkMode(160),
        "att_context": (56, 1),
        "num_slots": 79,
        "batch_size": 32,
    }
    args.update(overrides)
    return NeMoPipelineSpec(**args)


@pytest.mark.parametrize(
    ("model", "chunk_ms", "expected"),
    [
        (MULTILINGUAL, 80, [56, 0]),
        (MULTILINGUAL, 160, [56, 1]),
        (MULTILINGUAL, 560, [56, 6]),
        (MULTILINGUAL, 1120, [56, 13]),
        (ENGLISH, 160, [70, 1]),
        (ENGLISH, 1120, [70, 13]),
    ],
)
def test_att_context_follows_the_family_and_the_chunk(
    model: str, chunk_ms: int, expected: list[int]
) -> None:
    assert att_context_size(model, ChunkMode(chunk_ms)) == expected


def test_an_unknown_checkpoint_needs_its_left_context_passed() -> None:
    with pytest.raises(ConfigError, match="--att-context-left") as raised:
        att_context_size("someone/some-checkpoint", ChunkMode(160))
    assert "56" in str(raised.value)
    assert "70" in str(raised.value)
    assert att_context_size("someone/some-checkpoint", ChunkMode(560), left=64) == [64, 6]


def test_a_negative_left_context_is_refused() -> None:
    with pytest.raises(ConfigError, match=">= 0"):
        att_context_size(MULTILINGUAL, ChunkMode(160), left=-1)


def test_the_spec_refuses_a_batch_larger_than_the_slots_and_a_bad_dtype() -> None:
    with pytest.raises(ConfigError, match="batch_size"):
        _spec(num_slots=8, batch_size=9)
    with pytest.raises(ConfigError, match="compute_dtype"):
        _spec(compute_dtype="int8")
    with pytest.raises(ConfigError, match="checkpoint"):
        _spec(model="")


def test_pipeline_config_carries_the_keys_nemo_reads_and_the_scheduler_values() -> None:
    cfg = pipeline_config(_spec(stop_history_eou_ms=640, use_cuda_graphs=True))
    assert set(cfg) >= REQUIRED_TOP_LEVEL
    assert cfg["asr"]["model_name"] == MULTILINGUAL
    assert cfg["asr"]["device"] == "cuda"
    assert cfg["asr"]["use_cuda_graphs"] is True
    assert cfg["asr"]["use_amp"] is False
    assert cfg["asr"]["decoding"]["strategy"] == "greedy_batch"
    assert cfg["asr"]["decoding"]["greedy"]["enable_per_stream_biasing"] is False
    assert cfg["asr"]["decoding"]["greedy"]["use_cuda_graph_decoder"] is False
    assert cfg["streaming"]["chunk_size_in_secs"] == pytest.approx(0.16)
    assert cfg["streaming"]["att_context_size"] == [56, 1]
    assert cfg["streaming"]["num_slots"] == 79
    assert cfg["streaming"]["batch_size"] == 32
    assert cfg["streaming"]["sample_rate"] == 16000
    assert cfg["streaming"]["request_type"] == "frame"
    assert cfg["endpointing"]["stop_history_eou"] == 640
    assert cfg["pipeline_type"] == "cache_aware"
    assert cfg["asr_decoding_type"] == "rnnt"
    assert cfg["asr_output_granularity"] == "word"
    assert cfg["enable_itn"] is False
    assert cfg["enable_nmt"] is False
    assert cfg["return_tail_result"] is False


def test_eager_is_a_false_flag_not_a_missing_key() -> None:
    assert pipeline_config(_spec(use_cuda_graphs=False))["asr"]["use_cuda_graphs"] is False


# --- the pre-flight, against three shapes of installed runtime ---


def _importer(modules: dict[str, Any]) -> Any:
    def import_module(name: str) -> Any:
        try:
            return modules[name]
        except KeyError:
            raise ImportError(f"No module named {name!r}") from None

    return import_module


def _torch(cuda: bool, name: str = "NVIDIA RTX A6000") -> Any:
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda, get_device_name=lambda _index=0: name
    )
    return types.SimpleNamespace(__version__="2.11.0", cuda=cuda_ns)


def _released_track(version: str = "3.0.0", cuda: bool = True) -> dict[str, Any]:
    """NeMo 2.7.3 or 3.0.0: the inference package without PR #15863."""
    return {
        "nemo": types.SimpleNamespace(__version__=version),
        "torch": _torch(cuda),
        "nemo.collections.asr.inference.factory.pipeline_builder": types.SimpleNamespace(),
        WRAPPER_MODULE: types.SimpleNamespace(CacheAwareRNNTInferenceWrapper=type("W", (), {})),
    }


def _pr_track() -> dict[str, Any]:
    """The shape of Speech after PR #15863: the graphed step module and the switch."""
    modules = _released_track("3.1.0")
    wrapper = type("W", (), {"set_streaming_cuda_graphs": lambda self, enabled=True: None})
    modules[WRAPPER_MODULE] = types.SimpleNamespace(CacheAwareRNNTInferenceWrapper=wrapper)
    modules[GRAPH_STEP_MODULE] = types.SimpleNamespace(
        CudaGraphsStreamingEncoderStep=type("G", (), {})
    )
    return modules


def test_no_nemo_at_all_is_reported_and_not_usable() -> None:
    report = inspect_runtime(_importer({}))
    assert report.nemo_version is None
    assert report.torch_version is None
    assert report.cuda_available is False
    assert report.inference_package is False
    assert report.graph_step is False
    assert report.usable is False
    assert any("nemo" in note for note in report.notes)
    assert "not installed" in report.lines()[0]


def test_a_released_track_is_usable_but_has_no_graph_step() -> None:
    report = inspect_runtime(_importer(_released_track()))
    assert report.nemo_version == "3.0.0"
    assert report.usable is True
    assert report.graph_step is False
    assert any("--eager" in note for note in report.notes)
    assert any("ABSENT" in line for line in report.lines())


def test_the_pr_track_has_the_graph_step() -> None:
    report = inspect_runtime(_importer(_pr_track()))
    assert report.graph_step is True
    assert report.usable is True
    assert report.device_name == "NVIDIA RTX A6000"
    assert not any("--eager" in note for note in report.notes)


def test_half_of_the_pr_is_not_the_graph_step() -> None:
    """The module without the switch, or the switch without the module: neither is the
    graph path, because the builder needs both."""
    only_module = _released_track()
    only_module[GRAPH_STEP_MODULE] = types.SimpleNamespace(CudaGraphsStreamingEncoderStep=object)
    assert inspect_runtime(_importer(only_module)).graph_step is False
    only_switch = _pr_track()
    del only_switch[GRAPH_STEP_MODULE]
    assert inspect_runtime(_importer(only_switch)).graph_step is False


def test_nemo_without_a_cuda_device_is_not_usable() -> None:
    report = inspect_runtime(_importer(_released_track(cuda=False)))
    assert report.nemo_version == "3.0.0"
    assert report.cuda_available is False
    assert report.usable is False
    assert "no device" in report.lines()[3]


def test_a_broken_inference_package_is_a_note_not_a_crash() -> None:
    modules = _released_track()

    class _Broken:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("torch.cuda is broken")

    modules["torch"] = types.SimpleNamespace(__version__="2.11.0", cuda=_Broken())
    report = inspect_runtime(_importer(modules))
    assert report.torch_version == "2.11.0"
    assert report.cuda_available is False
    assert any("torch.cuda is broken" in note for note in report.notes)


# --- the guarded build ---


def _omegaconf() -> Any:
    return types.SimpleNamespace(OmegaConf=types.SimpleNamespace(create=lambda d: d))


def test_a_build_that_nemo_refuses_is_one_message_naming_what_was_asked() -> None:
    def build_pipeline_(cfg: Any) -> Any:
        raise RuntimeError("Failed to load model nvidia/nemotron-3.5-asr-streaming-0.6b")

    modules = {
        "omegaconf": _omegaconf(),
        "nemo.collections.asr.inference.factory.pipeline_builder": types.SimpleNamespace(
            PipelineBuilder=types.SimpleNamespace(build_pipeline=build_pipeline_)
        ),
    }
    with pytest.raises(PipelineBuildError) as raised:
        build_pipeline(_spec(), _importer(modules))
    message = str(raised.value)
    assert MULTILINGUAL in message
    assert "160 ms" in message
    assert "[56, 1]" in message
    assert "num_slots 79" in message
    assert "RuntimeError: Failed to load model" in message
    assert "HF_HOME" in message
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_missing_runtime_at_build_time_says_how_to_install_it() -> None:
    with pytest.raises(PipelineBuildError, match=r"verbatim\[nemo\]"):
        build_pipeline(_spec(), _importer({}))


def test_a_successful_build_hands_nemo_the_config_and_returns_its_pipeline() -> None:
    seen: list[Any] = []
    pipeline = object()

    def build_pipeline_(cfg: Any) -> Any:
        seen.append(cfg)
        return pipeline

    modules = {
        "omegaconf": _omegaconf(),
        "nemo.collections.asr.inference.factory.pipeline_builder": types.SimpleNamespace(
            PipelineBuilder=types.SimpleNamespace(build_pipeline=build_pipeline_)
        ),
    }
    assert build_pipeline(_spec(use_cuda_graphs=True), _importer(modules)) is pipeline
    assert seen[0]["asr"]["use_cuda_graphs"] is True
    assert seen[0]["streaming"]["att_context_size"] == [56, 1]
