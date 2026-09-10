# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Canonical digests for the invariance gate and the keyed checksum of a row.

The finals digest compares what the server emitted, byte for byte: each run's
finals are canonicalised and hashed with SHA-256, and the invariance verdict is
``True`` only if the hashes of all compared runs are equal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class FinalRecord:
    stream_id: str
    text: str
    words: tuple[tuple[str, int, int], ...] = ()  # (word, start_ms, end_ms)


def _check_word_times(words: Iterable[tuple[str, int, int]]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for entry in words:
        word, start_ms, end_ms = entry
        for value in (start_ms, end_ms):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "word timestamps must be integer milliseconds exactly as emitted, "
                    f"got {value!r}"
                )
        out.append([word, start_ms, end_ms])
    return out


def canonicalise_finals(finals: Iterable[FinalRecord]) -> str:
    """Deterministic UTF-8 serialisation, sorted by stream_id in byte order.

    No normalisation of any kind is applied to `text`: not case, not punctuation, not
    whitespace, not Unicode form. The digest exists to detect a difference the
    normaliser would hide.
    """
    ordered = sorted(finals, key=lambda record: record.stream_id.encode("utf-8"))
    payload = [
        {
            "stream_id": record.stream_id,
            "text": record.text,
            "words": _check_word_times(record.words),
        }
        for record in ordered
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def finals_digest(finals: Iterable[FinalRecord]) -> str:
    """SHA-256 hex of `canonicalise_finals`."""
    return hashlib.sha256(canonicalise_finals(finals).encode("utf-8")).hexdigest()


def partials_digest(partials: Mapping[str, Sequence[str]]) -> str:
    """Reported, never gated: partial cadence legitimately depends on arrival phase."""
    ordered_keys = sorted(partials, key=lambda key: key.encode("utf-8"))
    payload = [{"stream_id": key, "partials": list(partials[key])} for key in ordered_keys]
    serialised = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def finals_from_results(doc: Mapping[str, Any]) -> list[FinalRecord]:
    """Extract the finals of every session with no `error`. Sessions that errored are
    excluded; `errored_sessions(doc)` reports how many, so a digest over a partly
    failed run cannot pass for a clean one."""
    records: list[FinalRecord] = []
    sessions = doc.get("sessions", [])
    if not isinstance(sessions, list):
        return records
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        if session.get("error") is not None:
            continue
        stream_id = session.get("stream_id", "")
        text = session.get("final_text", "")
        raw_words = session.get("words") or []
        words: list[tuple[str, int, int]] = []
        if isinstance(raw_words, list):
            for entry in raw_words:
                if isinstance(entry, Mapping):
                    words.append((entry.get("w", ""), entry.get("s", 0), entry.get("e", 0)))
        records.append(
            FinalRecord(
                stream_id=stream_id if isinstance(stream_id, str) else "",
                text=text if isinstance(text, str) else "",
                words=tuple(words),
            )
        )
    return records


def errored_sessions(doc: Mapping[str, Any]) -> int:
    """Count sessions carrying an `error`, i.e. excluded from `finals_from_results`."""
    sessions = doc.get("sessions", [])
    if not isinstance(sessions, list):
        return 0
    return sum(
        1
        for session in sessions
        if isinstance(session, Mapping) and session.get("error") is not None
    )


CHECKSUM_KEY_ID: Final = "vb-results-1"
CHECKSUM_ALGORITHM: Final = "hmac-sha256"

_CHECKSUM_KEY: Final = b"vb-results-1-checksum-key"


def _checksum_payload(doc: Mapping[str, Any]) -> bytes:
    stripped = {key: value for key, value in doc.items() if key != "checksum"}
    return json.dumps(stripped, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def compute_checksum(doc: Mapping[str, Any]) -> str:
    """HMAC-SHA256 over the canonical serialisation of `doc` with `checksum` removed.

    This deters casual editing of a results file. It is NOT a security boundary: the
    key ships in this source file, and anyone who wants to forge a row can. Trust in a
    row comes from reproduction on someone else's hardware, not from this value.
    """
    return hmac.new(_CHECKSUM_KEY, _checksum_payload(doc), hashlib.sha256).hexdigest()


def stamp_checksum(doc: MutableMapping[str, Any]) -> dict[str, Any]:
    """Attach (or refresh) the `checksum` block of `doc` in place and return it."""
    doc["checksum"] = {
        "algorithm": CHECKSUM_ALGORITHM,
        "key_id": CHECKSUM_KEY_ID,
        "value": compute_checksum(doc),
    }
    return dict(doc)


def verify_checksum(doc: Mapping[str, Any]) -> bool:
    """Recompute the keyed checksum; `False` when absent or when it does not match."""
    block = doc.get("checksum")
    if not isinstance(block, Mapping):
        return False
    value = block.get("value")
    if not isinstance(value, str):
        return False
    return hmac.compare_digest(value, compute_checksum(doc))
