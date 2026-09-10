# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The skeleton smoke test: everything that is supposed to be importable, is.

This is deliberately the whole of the initial test suite. It asserts two things:

1. every planned package and module exists and
   imports on a machine with no GPU, no CUDA and no NeMo installed;
2. the vendored Riva protos generate stubs that import, and the service surface
   they expose is the one ``02_architecture.md`` §2.1 says it is.

Real behaviour arrives one subsystem at a time. Each one brings its
own tests; none of them may need a GPU or NeMo either.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.cpu

SERVER_MODULES = [
    "verbatim",
    "verbatim.cli",
    "verbatim.config",
    "verbatim.core",
    "verbatim.core.session",
    "verbatim.core.ring",
    "verbatim.core.backpressure",
    "verbatim.core.registry",
    "verbatim.core.types",
    "verbatim.core.errors",
    "verbatim.scheduler",
    "verbatim.scheduler.tick",
    "verbatim.scheduler.buckets",
    "verbatim.scheduler.boundary",
    "verbatim.scheduler.graph_budget",
    "verbatim.scheduler.admission",
    "verbatim.scheduler.slots",
    "verbatim.scheduler.clock",
    "verbatim.pipelines",
    "verbatim.pipelines.base",
    "verbatim.pipelines.cache_aware_rnnt",
    "verbatim.pipelines.cache_aware_ctc",
    "verbatim.pipelines.nemo_compat",
    "verbatim.pipelines.fake",
    "verbatim.pipelines.registry",
    "verbatim.protocols",
    "verbatim.protocols.base",
    "verbatim.protocols.health",
    "verbatim.protocols.riva",
    "verbatim.protocols.riva.server",
    "verbatim.protocols.riva.mapping",
    "verbatim.protocols.riva.conformance",
    "verbatim.protocols.ws",
    "verbatim.protocols.ws.server",
    "verbatim.protocols.ws.frames",
    "verbatim.audio",
    "verbatim.audio.pool",
    "verbatim.audio.pcm",
    "verbatim.audio.g711",
    "verbatim.audio.resample",
    "verbatim.obs",
    "verbatim.obs.counters",
    "verbatim.obs.latency",
    "verbatim.obs.metrics",
]

BENCH_MODULES = [
    "verbatim_bench",
    "verbatim_bench.cli",
    "verbatim_bench.pace",
    "verbatim_bench.client",
    "verbatim_bench.wer",
    "verbatim_bench.invariance",
    "verbatim_bench.verify",
    "verbatim_bench.report",
    "verbatim_bench.arms",
]


@pytest.mark.parametrize("name", SERVER_MODULES)
def test_server_module_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no docstring; every placeholder must say what belongs in it"


@pytest.mark.parametrize("name", BENCH_MODULES)
def test_bench_module_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no docstring; every placeholder must say what belongs in it"


def test_version_is_a_string() -> None:
    import verbatim_bench

    import verbatim

    assert isinstance(verbatim.__version__, str)
    assert isinstance(verbatim_bench.__version__, str)


class TestGeneratedRivaStubs:
    """The vendored protos are only useful if the committed stubs import and are correct."""

    def test_stubs_import(self) -> None:
        from verbatim.protocols.riva._gen import (
            riva_asr_pb2,
            riva_asr_pb2_grpc,
            riva_audio_pb2,
            riva_common_pb2,
        )

        assert riva_asr_pb2.DESCRIPTOR.package == "nvidia.riva.asr"
        assert riva_audio_pb2.DESCRIPTOR.package == "nvidia.riva"
        assert riva_common_pb2.DESCRIPTOR.package == "nvidia.riva"
        assert hasattr(riva_asr_pb2_grpc, "RivaSpeechRecognitionServicer")
        assert hasattr(riva_asr_pb2_grpc, "RivaSpeechRecognitionStub")

    def test_service_has_the_three_rpcs(self) -> None:
        from verbatim.protocols.riva._gen import riva_asr_pb2

        service = riva_asr_pb2.DESCRIPTOR.services_by_name["RivaSpeechRecognition"]
        assert sorted(m.name for m in service.methods) == [
            "GetRivaSpeechRecognitionConfig",
            "Recognize",
            "StreamingRecognize",
        ]

    def test_streaming_request_shape(self) -> None:
        """The oneof, the runtime_config map and the request id, at their field numbers."""
        from verbatim.protocols.riva._gen import riva_asr_pb2

        fields = {
            f.name: f.number for f in riva_asr_pb2.StreamingRecognizeRequest.DESCRIPTOR.fields
        }
        assert fields == {
            "streaming_config": 1,
            "audio_content": 2,
            "runtime_config": 3,
            "id": 100,
        }
        oneof = riva_asr_pb2.StreamingRecognizeRequest.DESCRIPTOR.oneofs_by_name[
            "streaming_request"
        ]
        assert sorted(f.name for f in oneof.fields) == ["audio_content", "streaming_config"]

    def test_endpointing_config_has_the_six_fields_pipecat_sets(self) -> None:
        from verbatim.protocols.riva._gen import riva_asr_pb2

        assert sorted(f.name for f in riva_asr_pb2.EndpointingConfig.DESCRIPTOR.fields) == [
            "start_history",
            "start_threshold",
            "stop_history",
            "stop_history_eou",
            "stop_threshold",
            "stop_threshold_eou",
        ]

    def test_audio_encoding_values(self) -> None:
        from verbatim.protocols.riva._gen import riva_audio_pb2

        encoding = riva_audio_pb2.AudioEncoding
        assert encoding.Value("ENCODING_UNSPECIFIED") == 0
        assert encoding.Value("LINEAR_PCM") == 1
        assert encoding.Value("FLAC") == 2
        assert encoding.Value("MULAW") == 3
        assert encoding.Value("OGGOPUS") == 4
        assert encoding.Value("ALAW") == 20

    def test_a_streaming_config_round_trips(self) -> None:
        """What both target plugins actually send as their first message."""
        from verbatim.protocols.riva._gen import riva_asr_pb2, riva_audio_pb2

        request = riva_asr_pb2.StreamingRecognizeRequest(
            streaming_config=riva_asr_pb2.StreamingRecognitionConfig(
                config=riva_asr_pb2.RecognitionConfig(
                    encoding=riva_audio_pb2.LINEAR_PCM,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                    max_alternatives=1,
                    audio_channel_count=1,
                    enable_word_time_offsets=True,
                    enable_automatic_punctuation=True,
                    model="",
                ),
                interim_results=True,
            )
        )
        decoded = riva_asr_pb2.StreamingRecognizeRequest()
        decoded.ParseFromString(request.SerializeToString())
        assert decoded.WhichOneof("streaming_request") == "streaming_config"
        assert decoded.streaming_config.config.sample_rate_hertz == 16000
        assert decoded.streaming_config.interim_results is True
