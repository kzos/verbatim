from riva.proto import riva_audio_pb2 as _riva_audio_pb2
from riva.proto import riva_common_pb2 as _riva_common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RivaSpeechRecognitionConfigRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class RivaSpeechRecognitionConfigResponse(_message.Message):
    __slots__ = ("model_config",)
    class Config(_message.Message):
        __slots__ = ("model_name", "parameters")
        class ParametersEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
        PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        model_name: str
        parameters: _containers.ScalarMap[str, str]
        def __init__(self, model_name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...
    MODEL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    model_config: _containers.RepeatedCompositeFieldContainer[RivaSpeechRecognitionConfigResponse.Config]
    def __init__(self, model_config: _Optional[_Iterable[_Union[RivaSpeechRecognitionConfigResponse.Config, _Mapping]]] = ...) -> None: ...

class RecognizeRequest(_message.Message):
    __slots__ = ("config", "audio", "id")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    config: RecognitionConfig
    audio: bytes
    id: _riva_common_pb2.RequestId
    def __init__(self, config: _Optional[_Union[RecognitionConfig, _Mapping]] = ..., audio: _Optional[bytes] = ..., id: _Optional[_Union[_riva_common_pb2.RequestId, _Mapping]] = ...) -> None: ...

class StreamingRecognizeRequest(_message.Message):
    __slots__ = ("streaming_config", "audio_content", "runtime_config", "id")
    class RuntimeConfigEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STREAMING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CONTENT_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    streaming_config: StreamingRecognitionConfig
    audio_content: bytes
    runtime_config: _containers.ScalarMap[str, str]
    id: _riva_common_pb2.RequestId
    def __init__(self, streaming_config: _Optional[_Union[StreamingRecognitionConfig, _Mapping]] = ..., audio_content: _Optional[bytes] = ..., runtime_config: _Optional[_Mapping[str, str]] = ..., id: _Optional[_Union[_riva_common_pb2.RequestId, _Mapping]] = ...) -> None: ...

class EndpointingConfig(_message.Message):
    __slots__ = ("start_history", "start_threshold", "stop_history", "stop_threshold", "stop_history_eou", "stop_threshold_eou")
    START_HISTORY_FIELD_NUMBER: _ClassVar[int]
    START_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    STOP_HISTORY_FIELD_NUMBER: _ClassVar[int]
    STOP_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    STOP_HISTORY_EOU_FIELD_NUMBER: _ClassVar[int]
    STOP_THRESHOLD_EOU_FIELD_NUMBER: _ClassVar[int]
    start_history: int
    start_threshold: float
    stop_history: int
    stop_threshold: float
    stop_history_eou: int
    stop_threshold_eou: float
    def __init__(self, start_history: _Optional[int] = ..., start_threshold: _Optional[float] = ..., stop_history: _Optional[int] = ..., stop_threshold: _Optional[float] = ..., stop_history_eou: _Optional[int] = ..., stop_threshold_eou: _Optional[float] = ...) -> None: ...

class RecognitionConfig(_message.Message):
    __slots__ = ("encoding", "sample_rate_hertz", "language_code", "max_alternatives", "profanity_filter", "speech_contexts", "audio_channel_count", "enable_word_time_offsets", "enable_automatic_punctuation", "enable_separate_recognition_per_channel", "model", "verbatim_transcripts", "diarization_config", "custom_configuration", "endpointing_config")
    class CustomConfigurationEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_HERTZ_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    MAX_ALTERNATIVES_FIELD_NUMBER: _ClassVar[int]
    PROFANITY_FILTER_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXTS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHANNEL_COUNT_FIELD_NUMBER: _ClassVar[int]
    ENABLE_WORD_TIME_OFFSETS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_AUTOMATIC_PUNCTUATION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_SEPARATE_RECOGNITION_PER_CHANNEL_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VERBATIM_TRANSCRIPTS_FIELD_NUMBER: _ClassVar[int]
    DIARIZATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    ENDPOINTING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    encoding: _riva_audio_pb2.AudioEncoding
    sample_rate_hertz: int
    language_code: str
    max_alternatives: int
    profanity_filter: bool
    speech_contexts: _containers.RepeatedCompositeFieldContainer[SpeechContext]
    audio_channel_count: int
    enable_word_time_offsets: bool
    enable_automatic_punctuation: bool
    enable_separate_recognition_per_channel: bool
    model: str
    verbatim_transcripts: bool
    diarization_config: SpeakerDiarizationConfig
    custom_configuration: _containers.ScalarMap[str, str]
    endpointing_config: EndpointingConfig
    def __init__(self, encoding: _Optional[_Union[_riva_audio_pb2.AudioEncoding, str]] = ..., sample_rate_hertz: _Optional[int] = ..., language_code: _Optional[str] = ..., max_alternatives: _Optional[int] = ..., profanity_filter: _Optional[bool] = ..., speech_contexts: _Optional[_Iterable[_Union[SpeechContext, _Mapping]]] = ..., audio_channel_count: _Optional[int] = ..., enable_word_time_offsets: _Optional[bool] = ..., enable_automatic_punctuation: _Optional[bool] = ..., enable_separate_recognition_per_channel: _Optional[bool] = ..., model: _Optional[str] = ..., verbatim_transcripts: _Optional[bool] = ..., diarization_config: _Optional[_Union[SpeakerDiarizationConfig, _Mapping]] = ..., custom_configuration: _Optional[_Mapping[str, str]] = ..., endpointing_config: _Optional[_Union[EndpointingConfig, _Mapping]] = ...) -> None: ...

class StreamingRecognitionConfig(_message.Message):
    __slots__ = ("config", "interim_results")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    INTERIM_RESULTS_FIELD_NUMBER: _ClassVar[int]
    config: RecognitionConfig
    interim_results: bool
    def __init__(self, config: _Optional[_Union[RecognitionConfig, _Mapping]] = ..., interim_results: _Optional[bool] = ...) -> None: ...

class SpeakerDiarizationConfig(_message.Message):
    __slots__ = ("enable_speaker_diarization", "max_speaker_count")
    ENABLE_SPEAKER_DIARIZATION_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEAKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    enable_speaker_diarization: bool
    max_speaker_count: int
    def __init__(self, enable_speaker_diarization: _Optional[bool] = ..., max_speaker_count: _Optional[int] = ...) -> None: ...

class SpeechContext(_message.Message):
    __slots__ = ("phrases", "boost")
    PHRASES_FIELD_NUMBER: _ClassVar[int]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    phrases: _containers.RepeatedScalarFieldContainer[str]
    boost: float
    def __init__(self, phrases: _Optional[_Iterable[str]] = ..., boost: _Optional[float] = ...) -> None: ...

class RecognizeResponse(_message.Message):
    __slots__ = ("results", "id")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SpeechRecognitionResult]
    id: _riva_common_pb2.RequestId
    def __init__(self, results: _Optional[_Iterable[_Union[SpeechRecognitionResult, _Mapping]]] = ..., id: _Optional[_Union[_riva_common_pb2.RequestId, _Mapping]] = ...) -> None: ...

class SpeechRecognitionResult(_message.Message):
    __slots__ = ("alternatives", "channel_tag", "audio_processed")
    ALTERNATIVES_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_TAG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    alternatives: _containers.RepeatedCompositeFieldContainer[SpeechRecognitionAlternative]
    channel_tag: int
    audio_processed: float
    def __init__(self, alternatives: _Optional[_Iterable[_Union[SpeechRecognitionAlternative, _Mapping]]] = ..., channel_tag: _Optional[int] = ..., audio_processed: _Optional[float] = ...) -> None: ...

class SpeechRecognitionAlternative(_message.Message):
    __slots__ = ("transcript", "confidence", "words", "language_code")
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    transcript: str
    confidence: float
    words: _containers.RepeatedCompositeFieldContainer[WordInfo]
    language_code: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, transcript: _Optional[str] = ..., confidence: _Optional[float] = ..., words: _Optional[_Iterable[_Union[WordInfo, _Mapping]]] = ..., language_code: _Optional[_Iterable[str]] = ...) -> None: ...

class WordInfo(_message.Message):
    __slots__ = ("start_time", "end_time", "word", "confidence", "speaker_tag", "language_code")
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    WORD_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_TAG_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    start_time: int
    end_time: int
    word: str
    confidence: float
    speaker_tag: int
    language_code: str
    def __init__(self, start_time: _Optional[int] = ..., end_time: _Optional[int] = ..., word: _Optional[str] = ..., confidence: _Optional[float] = ..., speaker_tag: _Optional[int] = ..., language_code: _Optional[str] = ...) -> None: ...

class StreamingRecognizeResponse(_message.Message):
    __slots__ = ("results", "id")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[StreamingRecognitionResult]
    id: _riva_common_pb2.RequestId
    def __init__(self, results: _Optional[_Iterable[_Union[StreamingRecognitionResult, _Mapping]]] = ..., id: _Optional[_Union[_riva_common_pb2.RequestId, _Mapping]] = ...) -> None: ...

class PipelineStates(_message.Message):
    __slots__ = ("vad_probabilities",)
    VAD_PROBABILITIES_FIELD_NUMBER: _ClassVar[int]
    vad_probabilities: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vad_probabilities: _Optional[_Iterable[float]] = ...) -> None: ...

class StreamingRecognitionResult(_message.Message):
    __slots__ = ("alternatives", "is_final", "stability", "channel_tag", "audio_processed", "pipeline_states")
    ALTERNATIVES_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    STABILITY_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_TAG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_STATES_FIELD_NUMBER: _ClassVar[int]
    alternatives: _containers.RepeatedCompositeFieldContainer[SpeechRecognitionAlternative]
    is_final: bool
    stability: float
    channel_tag: int
    audio_processed: float
    pipeline_states: PipelineStates
    def __init__(self, alternatives: _Optional[_Iterable[_Union[SpeechRecognitionAlternative, _Mapping]]] = ..., is_final: _Optional[bool] = ..., stability: _Optional[float] = ..., channel_tag: _Optional[int] = ..., audio_processed: _Optional[float] = ..., pipeline_states: _Optional[_Union[PipelineStates, _Mapping]] = ...) -> None: ...
