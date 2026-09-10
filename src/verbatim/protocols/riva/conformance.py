# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The honoured / accepted-and-ignored / rejected field table, as data.

``docs/protocols/riva.md`` is generated from this module and CI fails if the two
diverge, so the implemented subset is a reviewable artefact rather than folklore.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from verbatim.protocols.riva._gen import riva_asr_pb2

__all__ = [
    "CONFORMANCE",
    "FieldRule",
    "FieldStatus",
    "missing_fields",
    "rules_for",
]


class FieldStatus(StrEnum):
    """What the server does with a field: act on it, accept it silently, or refuse it."""

    HONOURED = "honoured"
    ACCEPTED_AND_IGNORED = "accepted-and-ignored"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class FieldRule:
    """One row of the subset table: a field, its disposition, and who exercises it."""

    message: str  # e.g. "RecognitionConfig"
    field: str  # e.g. "encoding"
    status: FieldStatus
    behaviour: str  # one sentence, the text the generated doc prints
    exercised_by: tuple[str, ...] = ()  # ("livekit", "pipecat", ...)


CONFORMANCE: Final[tuple[FieldRule, ...]] = (
    # --- RecognitionConfig ---
    FieldRule(
        "RecognitionConfig",
        "encoding",
        FieldStatus.REJECTED,
        "LINEAR_PCM is honoured; FLAC, MULAW, ALAW and OGGOPUS abort with "
        "UNIMPLEMENTED (decoders are a later task); ENCODING_UNSPECIFIED or any "
        "other value aborts with INVALID_ARGUMENT naming encoding.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "sample_rate_hertz",
        FieldStatus.REJECTED,
        "16000 (or 0, meaning 16000) is honoured; any other value aborts with "
        "UNIMPLEMENTED (the resampler is a later task).",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "language_code",
        FieldStatus.HONOURED,
        "Carried to SessionOptions.language_code; empty means the server default.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "max_alternatives",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "0 or 1 is honoured; a larger value is accepted and ignored, top-1 only, with a log line.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "profanity_filter",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored (no-op), logged once per session.",
        ("pipecat",),
    ),
    FieldRule(
        "RecognitionConfig",
        "speech_contexts",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored in this task (per-stream biasing is a later task), logged.",
        ("pipecat",),
    ),
    FieldRule(
        "RecognitionConfig",
        "audio_channel_count",
        FieldStatus.REJECTED,
        "0 or 1 is honoured; a larger value aborts with UNIMPLEMENTED.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "enable_word_time_offsets",
        FieldStatus.HONOURED,
        "Honoured as SessionOptions.word_timestamps; words appear on finals only.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "enable_automatic_punctuation",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored in this task.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "enable_separate_recognition_per_channel",
        FieldStatus.REJECTED,
        "False is honoured; true aborts with UNIMPLEMENTED.",
    ),
    FieldRule(
        "RecognitionConfig",
        "model",
        FieldStatus.REJECTED,
        "Empty or a served name/alias is honoured; anything else aborts with "
        "NOT_FOUND naming what was asked for and what is served.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "RecognitionConfig",
        "verbatim_transcripts",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored in this task.",
        ("pipecat",),
    ),
    FieldRule(
        "RecognitionConfig",
        "diarization_config",
        FieldStatus.REJECTED,
        "Passed through while disabled; enable_speaker_diarization=true aborts "
        "with UNIMPLEMENTED (diarization is a year-2 session type).",
    ),
    FieldRule(
        "RecognitionConfig",
        "custom_configuration",
        FieldStatus.HONOURED,
        "verbatim.chunk_ms is honoured (one of 80, 160, 560, 1120, else "
        "INVALID_ARGUMENT naming the field); every other key is ignored with a "
        "log line.",
        ("pipecat",),
    ),
    FieldRule(
        "RecognitionConfig",
        "endpointing_config",
        FieldStatus.HONOURED,
        "stop_history_eou is honoured if set, else stop_history, else the server "
        "default; the value rides on the session. Threshold and start_* fields "
        "are accepted and ignored (there is no VAD stage).",
        ("pipecat",),
    ),
    # --- StreamingRecognitionConfig ---
    FieldRule(
        "StreamingRecognitionConfig",
        "config",
        FieldStatus.HONOURED,
        "Carries the RecognitionConfig; required on the first stream message.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "StreamingRecognitionConfig",
        "interim_results",
        FieldStatus.HONOURED,
        "Honoured; false suppresses partials while finals still flow.",
        ("livekit", "pipecat"),
    ),
    # --- StreamingRecognizeRequest ---
    FieldRule(
        "StreamingRecognizeRequest",
        "streaming_config",
        FieldStatus.HONOURED,
        "Required on the first message; audio before config aborts with "
        "INVALID_ARGUMENT and a second config on the same stream aborts with "
        "INVALID_ARGUMENT.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "StreamingRecognizeRequest",
        "audio_content",
        FieldStatus.HONOURED,
        "Buffered PCM16 bytes; chunk boundaries are cut from the session sample "
        "counter, never from arrival timing.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "StreamingRecognizeRequest",
        "runtime_config",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Unknown keys are ignored; force_eou is accepted and ignored with a "
        "warning and never aborts the stream.",
        ("pipecat",),
    ),
    FieldRule(
        "StreamingRecognizeRequest",
        'runtime_config["force_eou"]',
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored with a warning: honouring it waits on an upstream "
        "NeMo force-EOU hook, so until that exists accepted-and-ignored is the "
        "contract and the stream is not aborted.",
    ),
    FieldRule(
        "StreamingRecognizeRequest",
        "id",
        FieldStatus.HONOURED,
        "Echoed on every response; empty string when the client sent none.",
        ("livekit", "pipecat"),
    ),
    # --- EndpointingConfig ---
    FieldRule(
        "EndpointingConfig",
        "stop_history_eou",
        FieldStatus.HONOURED,
        "Preferred end-of-utterance silence; honoured when set.",
        ("pipecat",),
    ),
    FieldRule(
        "EndpointingConfig",
        "stop_history",
        FieldStatus.HONOURED,
        "Used when stop_history_eou is unset (Pipecat sends 320).",
        ("pipecat",),
    ),
    FieldRule(
        "EndpointingConfig",
        "start_history",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored: there is no VAD stage.",
    ),
    FieldRule(
        "EndpointingConfig",
        "start_threshold",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored: there is no VAD stage.",
    ),
    FieldRule(
        "EndpointingConfig",
        "stop_threshold",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored: there is no VAD stage.",
    ),
    FieldRule(
        "EndpointingConfig",
        "stop_threshold_eou",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored: there is no VAD stage.",
    ),
    # --- SpeechContext ---
    FieldRule(
        "SpeechContext",
        "phrases",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored in this task (per-stream biasing is a later task).",
        ("pipecat",),
    ),
    FieldRule(
        "SpeechContext",
        "boost",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored in this task (per-stream biasing is a later task).",
        ("pipecat",),
    ),
    # --- SpeakerDiarizationConfig ---
    FieldRule(
        "SpeakerDiarizationConfig",
        "enable_speaker_diarization",
        FieldStatus.REJECTED,
        "False is fine; true aborts with UNIMPLEMENTED and speaker_tag stays 0 "
        "(diarization is a year-2 session type).",
    ),
    FieldRule(
        "SpeakerDiarizationConfig",
        "max_speaker_count",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Accepted and ignored: diarization is a year-2 session type.",
    ),
    # --- StreamingRecognizeResponse (server-emitted) ---
    FieldRule(
        "StreamingRecognizeResponse",
        "results",
        FieldStatus.HONOURED,
        "Exactly one StreamingRecognitionResult per response.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "StreamingRecognizeResponse",
        "id",
        FieldStatus.HONOURED,
        "Echo of the request id the client put on its config message.",
        ("livekit", "pipecat"),
    ),
    # --- StreamingRecognitionResult (server-emitted) ---
    FieldRule(
        "StreamingRecognitionResult",
        "alternatives",
        FieldStatus.HONOURED,
        "Exactly one alternative carrying the transcript.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "StreamingRecognitionResult",
        "is_final",
        FieldStatus.HONOURED,
        "False on partials, true on the utterance final.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "StreamingRecognitionResult",
        "stability",
        FieldStatus.HONOURED,
        "0.5 on partials; unset on finals.",
        ("livekit",),
    ),
    FieldRule(
        "StreamingRecognitionResult",
        "channel_tag",
        FieldStatus.HONOURED,
        "Always 1: mono only in this task.",
        ("livekit",),
    ),
    FieldRule(
        "StreamingRecognitionResult",
        "audio_processed",
        FieldStatus.HONOURED,
        "Seconds of audio consumed, stamped from the real bytes handed to the recognizer,"
        " padding excluded.",
        ("livekit",),
    ),
    FieldRule(
        "StreamingRecognitionResult",
        "pipeline_states",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Never populated in this task.",
    ),
    # --- SpeechRecognitionAlternative (server-emitted) ---
    FieldRule(
        "SpeechRecognitionAlternative",
        "transcript",
        FieldStatus.HONOURED,
        "The hypothesis text, on partials and finals alike.",
        ("livekit", "pipecat"),
    ),
    FieldRule(
        "SpeechRecognitionAlternative",
        "confidence",
        FieldStatus.HONOURED,
        "The mean of the word confidences on finals (0.0 when there are no "
        "words); unset on partials.",
        ("livekit",),
    ),
    FieldRule(
        "SpeechRecognitionAlternative",
        "words",
        FieldStatus.HONOURED,
        "Integer-millisecond WordInfos on finals when word timestamps were "
        "requested; never on partials.",
        ("livekit",),
    ),
    FieldRule(
        "SpeechRecognitionAlternative",
        "language_code",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Never populated in this task.",
    ),
    # --- WordInfo (server-emitted) ---
    FieldRule(
        "WordInfo",
        "start_time",
        FieldStatus.HONOURED,
        "Integer milliseconds (LiveKit divides by 1000).",
        ("livekit",),
    ),
    FieldRule(
        "WordInfo",
        "end_time",
        FieldStatus.HONOURED,
        "Integer milliseconds (LiveKit divides by 1000).",
        ("livekit",),
    ),
    FieldRule(
        "WordInfo",
        "word",
        FieldStatus.HONOURED,
        "The word text.",
        ("livekit",),
    ),
    FieldRule(
        "WordInfo",
        "confidence",
        FieldStatus.HONOURED,
        "The word confidence, passed through from the recognizer.",
        ("livekit",),
    ),
    FieldRule(
        "WordInfo",
        "speaker_tag",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Always 0: diarization is a year-2 session type.",
        ("livekit",),
    ),
    FieldRule(
        "WordInfo",
        "language_code",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Never populated in this task.",
    ),
    # --- PipelineStates (server-emitted, never populated) ---
    FieldRule(
        "PipelineStates",
        "vad_probabilities",
        FieldStatus.ACCEPTED_AND_IGNORED,
        "Never populated in this task: there is no VAD stage.",
    ),
    # --- RequestId ---
    FieldRule(
        "RequestId",
        "value",
        FieldStatus.HONOURED,
        "The opaque client id, echoed verbatim on every response.",
        ("livekit", "pipecat"),
    ),
)


def rules_for(message: str) -> tuple[FieldRule, ...]:
    """Return the conformance rows for one proto message name."""
    return tuple(rule for rule in CONFORMANCE if rule.message == message)


def _reachable_descriptors() -> dict[str, object]:
    """All message descriptors transitively reachable from StreamingRecognize I/O."""
    roots = (
        riva_asr_pb2.StreamingRecognizeRequest.DESCRIPTOR,
        riva_asr_pb2.StreamingRecognizeResponse.DESCRIPTOR,
    )
    seen: dict[str, object] = {}
    stack = list(roots)
    while stack:
        descriptor = stack.pop()
        if descriptor.full_name in seen:
            continue
        seen[descriptor.full_name] = descriptor
        for field in descriptor.fields:
            message_type = field.message_type
            if message_type is None or message_type.GetOptions().map_entry:
                continue
            stack.append(message_type)
    return seen


def missing_fields() -> dict[str, tuple[str, ...]]:
    """Fields present in the compiled descriptors that have no rule, per message.

    Walks the descriptors of every message reachable from StreamingRecognize and
    diffs them against CONFORMANCE. This is what makes the subset a reviewable
    artefact rather than folklore: a field added upstream shows up here, and the
    test below fails until someone writes a row for it.
    """
    covered: dict[str, set[str]] = {}
    for rule in CONFORMANCE:
        covered.setdefault(rule.message, set()).add(rule.field)
    missing: dict[str, tuple[str, ...]] = {}
    for descriptor in _reachable_descriptors().values():
        absent = tuple(
            field.name
            for field in descriptor.fields
            if field.name not in covered.get(descriptor.name, set())
        )
        if absent:
            missing[descriptor.name] = absent
    return missing
