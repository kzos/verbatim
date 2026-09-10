# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Mapping unit tests: RecognitionConfig -> SessionOptions, hypotheses -> protos.

No server here: options_from_config raises verbatim.core.errors, never grpc ones.
"""

from __future__ import annotations

import pytest

from verbatim.core.errors import InvalidArgument, NotFound, Unimplemented
from verbatim.protocols.base import Hypothesis, Word
from verbatim.protocols.riva._gen import riva_asr_pb2, riva_audio_pb2
from verbatim.protocols.riva.mapping import (
    DEFAULT_STOP_HISTORY_EOU_MS,
    config_response,
    final_response,
    options_from_config,
    partial_response,
)

pytestmark = pytest.mark.cpu

ENC = riva_audio_pb2.AudioEncoding

SERVED = ("verbatim-stub",)


def _config(**kwargs: object) -> riva_asr_pb2.RecognitionConfig:
    base: dict[str, object] = {
        "encoding": ENC.LINEAR_PCM,
        "sample_rate_hertz": 16000,
        "language_code": "en-US",
        "model": "",
        "max_alternatives": 1,
        "audio_channel_count": 1,
    }
    base.update(kwargs)
    return riva_asr_pb2.RecognitionConfig(**base)  # type: ignore[arg-type]


def _map(config: riva_asr_pb2.RecognitionConfig, **kwargs: object) -> object:
    params: dict[str, object] = {
        "interim_results": True,
        "request_id": "",
        "served_models": SERVED,
    }
    params.update(kwargs)
    return options_from_config(config, **params)  # type: ignore[arg-type]


def test_linear_pcm_16k_maps_cleanly() -> None:
    out = _map(_config())
    assert out.options.chunk_ms == 160
    assert out.options.sample_rate_hz == 16000
    assert out.ignored == ()


def test_encoding_unspecified_is_invalid_argument() -> None:
    with pytest.raises(InvalidArgument, match="encoding"):
        _map(_config(encoding=ENC.ENCODING_UNSPECIFIED))


@pytest.mark.parametrize("encoding", [ENC.FLAC, ENC.MULAW, ENC.ALAW, ENC.OGGOPUS])
def test_compressed_encodings_are_unimplemented(encoding: int) -> None:
    with pytest.raises(Unimplemented):
        _map(_config(encoding=encoding))


def test_zero_sample_rate_means_16k() -> None:
    out = _map(_config(sample_rate_hertz=0))
    assert out.options.sample_rate_hz == 16000


def test_other_sample_rates_are_unimplemented() -> None:
    with pytest.raises(Unimplemented, match="sample_rate_hertz"):
        _map(_config(sample_rate_hertz=8000))


def test_multichannel_is_unimplemented() -> None:
    with pytest.raises(Unimplemented, match="audio_channel_count"):
        _map(_config(audio_channel_count=2))


def test_separate_recognition_per_channel_is_unimplemented() -> None:
    with pytest.raises(Unimplemented, match="enable_separate_recognition_per_channel"):
        _map(_config(enable_separate_recognition_per_channel=True))


def test_empty_model_is_accepted() -> None:
    out = _map(_config(model=""))
    assert out.options.model == ""


def test_served_model_is_accepted() -> None:
    out = _map(_config(model="verbatim-stub"))
    assert out.options.model == "verbatim-stub"


def test_unknown_model_is_not_found() -> None:
    with pytest.raises(NotFound) as excinfo:
        _map(_config(model="nope"))
    assert "nope" in str(excinfo.value)
    assert "verbatim-stub" in str(excinfo.value)


def test_word_time_offsets_maps_to_word_timestamps() -> None:
    assert _map(_config(enable_word_time_offsets=True)).options.word_timestamps is True
    assert _map(_config(enable_word_time_offsets=False)).options.word_timestamps is False


def test_max_alternatives_gt_1_is_accepted_and_ignored() -> None:
    out = _map(_config(max_alternatives=3))
    assert "max_alternatives" in out.ignored


def test_profanity_filter_and_punctuation_and_verbatim_are_ignored() -> None:
    out = _map(
        _config(profanity_filter=True, enable_automatic_punctuation=True, verbatim_transcripts=True)
    )
    assert "profanity_filter" in out.ignored
    assert "enable_automatic_punctuation" in out.ignored
    assert "verbatim_transcripts" in out.ignored


def test_speech_contexts_are_accepted_and_ignored() -> None:
    config = _config()
    context = config.speech_contexts.add()
    context.phrases.append("hello world")
    context.boost = 2.0
    out = _map(config)
    assert "speech_contexts" in out.ignored


def test_endpointing_prefers_stop_history_eou() -> None:
    config = _config()
    config.endpointing_config.stop_history = 320
    config.endpointing_config.stop_history_eou = 500
    assert _map(config).stop_history_eou_ms == 500


def test_endpointing_falls_back_to_stop_history() -> None:
    config = _config()
    config.endpointing_config.stop_history = 320
    assert _map(config).stop_history_eou_ms == 320


def test_endpointing_defaults_when_absent() -> None:
    assert _map(_config()).stop_history_eou_ms == DEFAULT_STOP_HISTORY_EOU_MS


def test_endpointing_thresholds_are_ignored_not_fatal() -> None:
    config = _config()
    config.endpointing_config.start_history = 100
    config.endpointing_config.start_threshold = 0.5
    config.endpointing_config.stop_threshold = 0.7
    config.endpointing_config.stop_threshold_eou = 0.9
    out = _map(config)
    assert out.stop_history_eou_ms == DEFAULT_STOP_HISTORY_EOU_MS


def test_custom_configuration_chunk_ms_is_honoured() -> None:
    config = _config()
    config.custom_configuration["verbatim.chunk_ms"] = "560"
    assert _map(config).options.chunk_ms == 560


def test_custom_configuration_bad_chunk_ms_is_invalid_argument() -> None:
    config = _config()
    config.custom_configuration["verbatim.chunk_ms"] = "100"
    with pytest.raises(InvalidArgument, match="custom_configuration"):
        _map(config)


def test_custom_configuration_unknown_keys_are_ignored() -> None:
    config = _config()
    config.custom_configuration["some.unknown_key"] = "v"
    out = _map(config)
    assert any("custom_configuration" in field for field in out.ignored)


def test_diarization_true_is_unimplemented() -> None:
    config = _config()
    config.diarization_config.enable_speaker_diarization = True
    with pytest.raises(Unimplemented, match="diarization"):
        _map(config)


def test_diarization_false_is_fine() -> None:
    config = _config()
    config.diarization_config.enable_speaker_diarization = False
    _map(config)


def test_partial_response_shape() -> None:
    hypo = Hypothesis(text="the quick", is_final=False, audio_processed_s=0.32)
    response = partial_response(hypo, request_id="abc")
    assert len(response.results) == 1
    result = response.results[0]
    assert result.is_final is False
    assert result.stability == pytest.approx(0.5)
    assert result.channel_tag == 1
    assert result.audio_processed == pytest.approx(0.32)
    assert response.id.value == "abc"
    assert len(result.alternatives) == 1
    assert result.alternatives[0].transcript == "the quick"
    assert len(result.alternatives[0].words) == 0


def test_final_response_word_times_are_integer_ms() -> None:
    hypo = Hypothesis(
        text="the",
        is_final=True,
        audio_processed_s=0.16,
        words=(Word(word="the", start_ms=0, end_ms=160, confidence=0.9),),
    )
    response = final_response(hypo, request_id="", word_timestamps=True)
    words = response.results[0].alternatives[0].words
    assert len(words) == 1
    assert words[0].word == "the"
    assert words[0].start_time == 0
    assert words[0].end_time == 160
    assert isinstance(words[0].start_time, int)
    assert isinstance(words[0].end_time, int)


def test_final_response_omits_words_when_not_requested() -> None:
    hypo = Hypothesis(
        text="the",
        is_final=True,
        audio_processed_s=0.16,
        words=(Word(word="the", start_ms=0, end_ms=160),),
    )
    response = final_response(hypo, request_id="", word_timestamps=False)
    assert len(response.results[0].alternatives[0].words) == 0


def test_final_confidence_is_the_mean_of_word_confidences() -> None:
    hypo = Hypothesis(
        text="a b",
        is_final=True,
        audio_processed_s=0.32,
        words=(
            Word(word="a", start_ms=0, end_ms=160, confidence=0.5),
            Word(word="b", start_ms=160, end_ms=320, confidence=1.0),
        ),
    )
    response = final_response(hypo, request_id="", word_timestamps=True)
    assert response.results[0].alternatives[0].confidence == pytest.approx(0.75)
    bare = Hypothesis(text="a b", is_final=True, audio_processed_s=0.32)
    empty = final_response(bare, request_id="", word_timestamps=True)
    assert empty.results[0].alternatives[0].confidence == 0.0


def test_config_response_parameters() -> None:
    response = config_response([("verbatim-stub", 160, "en-US")])
    assert len(response.model_config) == 1
    entry = response.model_config[0]
    assert entry.parameters["type"] == "online"
    assert entry.parameters["streaming"] == "true"
    assert entry.parameters["language_code"] == "en-US"
    assert entry.parameters["chunk_ms"].isdigit()


def test_config_response_filters_by_model_name() -> None:
    served = [("a-stub", 160, "en-US"), ("b-stub", 560, "en-US")]
    filtered = config_response(served, model_name_filter="b-stub")
    assert [c.model_name for c in filtered.model_config] == ["b-stub"]
    with pytest.raises(NotFound):
        config_response(served, model_name_filter="nope")


def test_endpointing_silence_rides_on_the_session_options() -> None:
    config = riva_asr_pb2.RecognitionConfig(encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM)
    config.endpointing_config.stop_history_eou = 320
    session = options_from_config(
        config, interim_results=True, request_id="r", served_models=("verbatim-stub",)
    )
    assert session.stop_history_eou_ms == 320
    assert session.options.stop_history_eou_ms == 320
    default = options_from_config(
        riva_asr_pb2.RecognitionConfig(encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM),
        interim_results=True,
        request_id="r",
        served_models=("verbatim-stub",),
    )
    assert default.options.stop_history_eou_ms == DEFAULT_STOP_HISTORY_EOU_MS
