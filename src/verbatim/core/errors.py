# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The error taxonomy every protocol adapter maps onto.

Riva/gRPC status codes are the reference mapping (02_architecture.md §2.1):
``INVALID_ARGUMENT`` for an unknown *value* of a known field, ``UNIMPLEMENTED``
for a year-1 out-of-scope feature, ``RESOURCE_EXHAUSTED`` for a refused
admission, ``NOT_FOUND`` for an unserved model name.

The names are the gRPC status names because the Riva surface is the production
one; the WebSocket adapter emits them as the ``code`` string, so a bug report from
the demo path names the same condition as a bug report from the gRPC path.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DeadlineExceeded",
    "ErrorCode",
    "InvalidArgument",
    "NotFound",
    "ResourceExhausted",
    "Unimplemented",
    "VerbatimError",
]


class ErrorCode(StrEnum):
    """The taxonomy every protocol adapter maps onto.

    The names are the gRPC status names because the Riva surface is the production
    one; the WebSocket adapter emits them as the `code` string, so a bug report from
    the demo path names the same condition as a bug report from the gRPC path.
    """

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    UNIMPLEMENTED = "UNIMPLEMENTED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    INTERNAL = "INTERNAL"


class VerbatimError(Exception):
    """Base error carrying a protocol-level code."""

    code: ErrorCode

    def __init__(self, message: str, code: ErrorCode = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.code = code


class InvalidArgument(VerbatimError):
    """An unknown value of a known field."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.INVALID_ARGUMENT)


class NotFound(VerbatimError):
    """An unserved model name or unknown resource."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.NOT_FOUND)


class Unimplemented(VerbatimError):
    """A year-1 out-of-scope feature."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.UNIMPLEMENTED)


class DeadlineExceeded(VerbatimError):
    """A session that sent no audio for longer than the engine's idle deadline."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.DEADLINE_EXCEEDED)


class ResourceExhausted(VerbatimError):
    """A refused admission."""

    retry_after_ms: int

    def __init__(self, message: str, retry_after_ms: int = 0) -> None:
        super().__init__(message, ErrorCode.RESOURCE_EXHAUSTED)
        self.retry_after_ms = retry_after_ms
