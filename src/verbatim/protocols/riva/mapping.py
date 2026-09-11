# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Riva field <-> Verbatim session options.

Unknown *fields* are ignored per proto3 semantics; unknown *values* are rejected
with ``INVALID_ARGUMENT`` naming the field, because a silent wrong answer is worse
than a refusal for the operators this serves.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from verbatim.audio.resample import SUPPORTED_RATES
from verbatim.core.errors import InvalidArgument, NotFound, Unimplemented
from verbatim.protocols.base import VALID_CHUNK_MS, Hypothesis, SessionOptions
from verbatim.protocols.riva._gen import riva_asr_pb2, riva_audio_pb2

__all__ = [
    "DEFAULT_STOP_HISTORY_EOU_MS",
    "RivaSessionConfig",
    "config_response",
    "final_response",
    "options_from_config",
    "partial_response",
]

logger = logging.getLogger(__name__)

DEFAULT_STOP_HISTORY_EOU_MS: Final = 800

_VALID_CHUNK_MS_STR: Final = ", ".join(str(v) for v in VALID_CHUNK_MS)

_G711_ENCODINGS: Final = {
    riva_audio_pb2.AudioEncoding.MULAW: "MULAW",
    riva_audio_pb2.AudioEncoding.ALAW: "ALAW",
}
_CODEC_ENCODINGS: Final = {
    riva_audio_pb2.AudioEncoding.FLAC: "FLAC",
    riva_audio_pb2.AudioEncoding.OGGOPUS: "OGGOPUS",
}
_SUPPORTED_RATES_STR: Final = ", ".join(str(r) for r in SUPPORTED_RATES)


@dataclass(frozen=True, slots=True)
class RivaSessionConfig:
    """What a validated RecognitionConfig turns into. `options` is what the recognizer sees."""

    options: SessionOptions
    request_id: str
    stop_history_eou_ms: int
    ignored: tuple[str, ...]  # fields accepted-and-ignored on this request, for logging/tests


def options_from_config(
    config: riva_asr_pb2.RecognitionConfig,
    *,
    interim_results: bool,
    request_id: str,
    served_models: Sequence[str],
    default_chunk_ms: int = 160,
) -> RivaSessionConfig:
    """Validate and map. Raises the errors from verbatim.core.errors, never grpc ones --
    the servicer owns the translation to a status code, so this function stays testable
    with no server."""
    ignored: list[str] = []

    def note(field: str, message: str) -> None:
        ignored.append(field)
        logger.info("session %r: ignoring %s: %s", request_id, field, message)

    encoding = config.encoding
    if encoding == riva_audio_pb2.AudioEncoding.LINEAR_PCM:
        wire_encoding = "LINEAR_PCM"
    elif encoding in _G711_ENCODINGS:
        wire_encoding = _G711_ENCODINGS[encoding]
    elif encoding in _CODEC_ENCODINGS:
        raise Unimplemented(
            f"unsupported encoding {_CODEC_ENCODINGS[encoding]} in field 'encoding': "
            "FLAC and OGGOPUS are codecs this server does not carry; "
            "LINEAR_PCM, MULAW and ALAW are served"
        )
    else:
        raise InvalidArgument(
            f"invalid encoding {encoding} in field 'encoding': "
            "expected LINEAR_PCM (1), MULAW (3) or ALAW (20)"
        )

    # 0 means 16000, the rate both target plugins send. Anything else that is served
    # is resampled to 16000 by the transport, in a worker thread, before the ring.
    sample_rate = config.sample_rate_hertz or 16000
    if sample_rate not in SUPPORTED_RATES:
        raise InvalidArgument(
            f"invalid sample_rate_hertz {sample_rate}: "
            f"must be one of {_SUPPORTED_RATES_STR} (0 means 16000)"
        )

    channels = config.audio_channel_count
    if channels not in (0, 1):
        if channels > 1:
            raise Unimplemented(
                f"unsupported audio_channel_count {channels}: "
                "only mono (0 or 1) is served in this task"
            )
        raise InvalidArgument(
            f"invalid audio_channel_count {channels}: expected 0 or 1 in this task"
        )
    if config.enable_separate_recognition_per_channel:
        raise Unimplemented(
            "unsupported enable_separate_recognition_per_channel=true: "
            "only mono (a single channel) is served in this task"
        )

    language_code = config.language_code or "en-US"

    model = config.model
    if model and model not in served_models:
        served = ", ".join(str(m) for m in served_models) or "<none>"
        raise NotFound(f"unknown model {model!r} in field 'model': served models are {served}")

    if config.max_alternatives > 1:
        note(
            "max_alternatives",
            f"got {config.max_alternatives}; serving top-1 only",
        )
    elif config.max_alternatives < 0:
        raise InvalidArgument(
            f"invalid max_alternatives {config.max_alternatives}: expected 0 or more"
        )

    if config.profanity_filter:
        note("profanity_filter", "no profanity filter stage exists; passing audio through")
    if config.enable_automatic_punctuation:
        note("enable_automatic_punctuation", "punctuation is the recognizer's own")
    if config.verbatim_transcripts:
        note("verbatim_transcripts", "transcripts pass through unmodified")

    if len(config.speech_contexts) > 0:
        note(
            "speech_contexts",
            f"got {len(config.speech_contexts)} context(s); per-stream biasing is a later task",
        )

    if config.HasField("diarization_config"):
        diar = config.diarization_config
        if diar.enable_speaker_diarization:
            raise Unimplemented(
                "unsupported diarization_config.enable_speaker_diarization=true: "
                "diarization is a year-2 session type; speaker_tag stays 0"
            )
        if diar.max_speaker_count:
            note(
                "diarization_config.max_speaker_count",
                "diarization is a year-2 session type",
            )

    stop_history_eou_ms = DEFAULT_STOP_HISTORY_EOU_MS
    if config.HasField("endpointing_config"):
        endpointing = config.endpointing_config
        if endpointing.stop_history_eou:
            stop_history_eou_ms = int(endpointing.stop_history_eou)
        elif endpointing.stop_history:
            stop_history_eou_ms = int(endpointing.stop_history)
        for field in (
            "start_history",
            "start_threshold",
            "stop_threshold",
            "stop_threshold_eou",
        ):
            if getattr(endpointing, field):
                note(f"endpointing_config.{field}", "there is no VAD stage")

    chunk_ms = default_chunk_ms
    custom = config.custom_configuration
    if "verbatim.chunk_ms" in custom:
        raw = custom["verbatim.chunk_ms"]
        try:
            chunk_ms = int(raw)
        except ValueError:
            raise InvalidArgument(
                f'invalid custom_configuration["verbatim.chunk_ms"]={raw!r}: '
                f"expected an integer, one of {_VALID_CHUNK_MS_STR}"
            ) from None
        if chunk_ms not in VALID_CHUNK_MS:
            raise InvalidArgument(
                f'invalid custom_configuration["verbatim.chunk_ms"]={raw!r}: '
                f"must be one of {_VALID_CHUNK_MS_STR}"
            )
    for key in custom:
        if key != "verbatim.chunk_ms":
            note(f"custom_configuration[{key!r}]", "unknown key")

    options = SessionOptions(
        chunk_ms=chunk_ms,
        language_code=language_code,
        sample_rate_hz=16000,
        interim_results=interim_results,
        word_timestamps=bool(config.enable_word_time_offsets),
        model=model,
        stop_history_eou_ms=stop_history_eou_ms,
        wire_encoding=wire_encoding,
        wire_sample_rate_hz=sample_rate,
    )
    return RivaSessionConfig(
        options=options,
        request_id=request_id,
        stop_history_eou_ms=stop_history_eou_ms,
        ignored=tuple(ignored),
    )


def partial_response(h: Hypothesis, *, request_id: str) -> riva_asr_pb2.StreamingRecognizeResponse:
    """One partial hypothesis becomes one response with one non-final result."""
    response = riva_asr_pb2.StreamingRecognizeResponse()
    response.id.value = request_id
    result = response.results.add()
    result.is_final = False
    result.stability = 0.5
    result.channel_tag = 1
    result.audio_processed = h.audio_processed_s
    result.alternatives.add().transcript = h.text
    return response


def final_response(
    h: Hypothesis, *, request_id: str, word_timestamps: bool
) -> riva_asr_pb2.StreamingRecognizeResponse:
    """One final hypothesis becomes one response with one final result."""
    response = riva_asr_pb2.StreamingRecognizeResponse()
    response.id.value = request_id
    result = response.results.add()
    result.is_final = True
    result.channel_tag = 1
    result.audio_processed = h.audio_processed_s
    alternative = result.alternatives.add()
    alternative.transcript = h.text
    if word_timestamps:
        for word in h.words:
            info = alternative.words.add()
            info.word = word.word
            info.start_time = int(word.start_ms)
            info.end_time = int(word.end_ms)
            info.confidence = word.confidence
    if h.words:
        alternative.confidence = sum(w.confidence for w in h.words) / len(h.words)
    else:
        alternative.confidence = 0.0
    return response


def config_response(
    served: Sequence[tuple[str, int, str]],  # (model_name, chunk_ms, language_code)
    *,
    model_name_filter: str = "",
) -> riva_asr_pb2.RivaSpeechRecognitionConfigResponse:
    """One Config per served mode; a non-empty filter selects a single model."""
    if model_name_filter:
        matched = [entry for entry in served if entry[0] == model_name_filter]
        if not matched:
            names = ", ".join(name for name, _, _ in served) or "<none>"
            raise NotFound(
                f"unknown model {model_name_filter!r} in field 'model_name': "
                f"served models are {names}"
            )
        served = matched
    response = riva_asr_pb2.RivaSpeechRecognitionConfigResponse()
    for model_name, chunk_ms, language_code in served:
        entry = response.model_config.add()
        entry.model_name = model_name
        entry.parameters["type"] = "online"
        entry.parameters["language_code"] = language_code
        entry.parameters["streaming"] = "true"
        entry.parameters["chunk_ms"] = str(chunk_ms)
    return response
