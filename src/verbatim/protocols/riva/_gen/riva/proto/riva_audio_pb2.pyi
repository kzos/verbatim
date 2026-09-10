from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class AudioEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENCODING_UNSPECIFIED: _ClassVar[AudioEncoding]
    LINEAR_PCM: _ClassVar[AudioEncoding]
    FLAC: _ClassVar[AudioEncoding]
    MULAW: _ClassVar[AudioEncoding]
    OGGOPUS: _ClassVar[AudioEncoding]
    ALAW: _ClassVar[AudioEncoding]
ENCODING_UNSPECIFIED: AudioEncoding
LINEAR_PCM: AudioEncoding
FLAC: AudioEncoding
MULAW: AudioEncoding
OGGOPUS: AudioEncoding
ALAW: AudioEncoding
