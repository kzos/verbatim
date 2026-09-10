# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The four-message demo protocol.

Server -> client: ``session`` on connect, then ``partial`` / ``final``, plus
``error``. Client -> server: binary PCM16 LE 16 kHz mono frames of any size, and a
text ``{"type": "end"}`` to drain.

JSON lives on this path and nowhere else. ``to_json`` produces compact,
deterministic JSON: keys in wire order, ``audio_s`` rounded to 6 decimal places
so the output is stable across platforms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import parse_qsl

from verbatim.core.errors import ErrorCode, InvalidArgument
from verbatim.protocols.base import VALID_CHUNK_MS, SessionOptions, Word

__all__ = [
    "ErrorFrame",
    "FinalFrame",
    "PartialFrame",
    "SessionFrame",
    "parse_client_text",
    "parse_query",
]

_VALID_CHUNK_MS_STR: Final = ", ".join(str(v) for v in VALID_CHUNK_MS)
# Plain ASCII digits only: `[0-9]` never matches full-width or other Unicode
# digits the way `\d` and `int()` do.
_CANONICAL_UINT_RE: Final = re.compile(r"[0-9]+")
# Query keys whose second occurrence is an error rather than an override.
_SINGLETON_PARAMS: Final = ("chunk_ms", "lang", "words", "interim_results")


@dataclass(frozen=True, slots=True)
class SessionFrame:
    """Sent once, immediately on connect, before any audio is accepted."""

    id: str
    chunk_ms: int
    invariance_class: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": "session",
                "id": self.id,
                "chunk_ms": self.chunk_ms,
                "invariance_class": self.invariance_class,
            },
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class PartialFrame:
    """An interim hypothesis plus the seconds of audio consumed so far."""

    text: str
    audio_s: float

    def to_json(self) -> str:
        return json.dumps(
            {"type": "partial", "text": self.text, "audio_s": round(float(self.audio_s), 6)},
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class FinalFrame:
    """The utterance final. `words=None` omits the key; it is only sent when asked for."""

    text: str
    audio_s: float
    words: tuple[Word, ...] | None = None  # None -> the key is omitted

    def to_json(self) -> str:
        obj: dict[str, object] = {"type": "final", "text": self.text}
        if self.words is not None:
            obj["words"] = [{"w": w.word, "s": w.start_ms, "e": w.end_ms} for w in self.words]
        obj["audio_s"] = round(float(self.audio_s), 6)
        return json.dumps(obj, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ErrorFrame:
    """A session-level failure. The client is always told why a session dies."""

    code: ErrorCode
    message: str

    def to_json(self) -> str:
        code = self.code.value if isinstance(self.code, ErrorCode) else str(self.code)
        return json.dumps(
            {"type": "error", "code": code, "message": self.message},
            separators=(",", ":"),
        )


def parse_client_text(payload: str) -> Literal["end"]:
    """Parse a client text frame. Only `{"type": "end"}` is valid.

    Raises InvalidArgument for anything else -- malformed JSON, a non-object, a
    missing or unknown `type`. The message names what was wrong.
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InvalidArgument(f"invalid client text frame: not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise InvalidArgument(
            f"invalid client text frame: expected a JSON object, got {type(obj).__name__}"
        )
    msg_type = obj.get("type")
    if msg_type == "end":
        return "end"
    if msg_type is None:
        raise InvalidArgument("invalid client text frame: missing 'type'")
    raise InvalidArgument(f"invalid client text frame: unknown type {msg_type!r}")


def _parse_bool_param(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes"):
        return True
    if lowered in ("0", "false", "no"):
        return False
    raise InvalidArgument(f"invalid query parameter {name}={value!r}: expected 0 or 1")


def parse_query(raw_query: str) -> SessionOptions:
    """Build SessionOptions from the endpoint query string.

    Unknown parameters are ignored (a client is allowed to send more than we read);
    a known parameter with an unusable value raises InvalidArgument naming it. This
    asymmetry is deliberate and is the same rule the gRPC surface uses.

    Known parameters: ``chunk_ms`` (one of 80/160/560/1120), ``lang`` (a language
    code such as ``en-US``), ``words`` (``1`` asks for word timings on finals),
    ``interim_results`` (``0`` suppresses partials; the final is still sent).
    """
    query = raw_query[1:] if raw_query.startswith("?") else raw_query
    pairs = parse_qsl(query, keep_blank_values=True)
    counts: dict[str, int] = {}
    for key, _ in pairs:
        counts[key] = counts.get(key, 0) + 1
    for name in _SINGLETON_PARAMS:
        if counts.get(name, 0) > 1:
            raise InvalidArgument(f"invalid query parameter {name}: duplicated parameter")
    params = dict(pairs)

    chunk_ms = 160
    if "chunk_ms" in params:
        raw = params["chunk_ms"]
        if _CANONICAL_UINT_RE.fullmatch(raw) is None or (len(raw) > 1 and raw.startswith("0")):
            raise InvalidArgument(
                f"invalid query parameter chunk_ms={raw!r}: "
                f"expected an integer, one of {_VALID_CHUNK_MS_STR}"
            )
        chunk_ms = int(raw)

    language_code = params.get("lang", "en-US")
    if "lang" in params and not language_code:
        raise InvalidArgument("invalid query parameter lang: expected a non-empty language code")

    word_timestamps = False
    if "words" in params:
        word_timestamps = _parse_bool_param("words", params["words"])

    interim_results = True
    if "interim_results" in params:
        interim_results = _parse_bool_param("interim_results", params["interim_results"])

    return SessionOptions(
        chunk_ms=chunk_ms,
        language_code=language_code,
        interim_results=interim_results,
        word_timestamps=word_timestamps,
    )
