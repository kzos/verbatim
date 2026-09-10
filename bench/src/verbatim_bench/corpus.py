# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Manifest loading and PCM decoding for the paced load generator.

The manifest is NeMo-compatible JSONL read in file order: arrival order is part
of the fixture, so entries are never sorted. Audio must be exactly 16 kHz mono;
anything else fails loudly because resampling would change the bytes a row was
measured on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


class ManifestError(ValueError):
    """A malformed manifest line, missing key, duplicate id, or missing file."""


class AudioFormatError(ValueError):
    """Audio that is not exactly the expected rate and channel count."""


@dataclass(frozen=True, slots=True)
class Utterance:
    stream_id: str
    audio_path: Path
    duration_s: float
    text: str
    sha256: str | None = None


_REQUIRED_KEYS = ("audio_filepath", "duration", "text", "stream_id")


def load_manifest(path: Path) -> list[Utterance]:
    """Read a NeMo-compatible JSONL manifest. Order is preserved exactly; never sorted.

    Raises ManifestError on: a malformed line, a missing required key, a duplicate
    stream_id, or a relative audio_filepath that does not resolve.
    """
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    utterances: list[Utterance] = []
    seen: set[str] = set()
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{manifest_path}:{lineno}: malformed JSON: {exc}") from exc
        if not isinstance(entry, dict):
            raise ManifestError(f"{manifest_path}:{lineno}: entry is not a JSON object")
        for key in _REQUIRED_KEYS:
            if key not in entry:
                raise ManifestError(f"{manifest_path}:{lineno}: missing required key {key!r}")
        stream_id = entry["stream_id"]
        if not isinstance(stream_id, str) or not stream_id:
            raise ManifestError(f"{manifest_path}:{lineno}: invalid stream_id {stream_id!r}")
        if stream_id in seen:
            raise ManifestError(f"{manifest_path}:{lineno}: duplicate stream_id {stream_id!r}")
        seen.add(stream_id)
        audio_ref = entry["audio_filepath"]
        audio_path = Path(audio_ref)
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        if not audio_path.is_file():
            raise ManifestError(
                f"{manifest_path}:{lineno}: audio_filepath does not resolve: {audio_ref!r}"
            )
        try:
            duration_s = float(entry["duration"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"{manifest_path}:{lineno}: invalid duration {entry['duration']!r}"
            ) from exc
        text = entry["text"]
        if not isinstance(text, str):
            raise ManifestError(f"{manifest_path}:{lineno}: invalid text {text!r}")
        sha256 = entry.get("sha256")
        if sha256 is not None and not isinstance(sha256, str):
            raise ManifestError(f"{manifest_path}:{lineno}: invalid sha256 {sha256!r}")
        utterances.append(
            Utterance(
                stream_id=stream_id,
                audio_path=audio_path,
                duration_s=duration_s,
                text=text,
                sha256=sha256,
            )
        )
    return utterances


def read_pcm16(path: Path, *, expect_rate_hz: int = 16000) -> bytes:
    """Decode a WAV file to 16 kHz mono PCM16 little-endian bytes.

    Raises AudioFormatError -- never resamples and never downmixes -- if the file is
    not exactly `expect_rate_hz` mono. Resampling here would change the bytes a
    measurement was taken on.
    """
    file_path = Path(path)
    try:
        info = sf.info(str(file_path))
    except Exception as exc:
        raise AudioFormatError(f"cannot read audio {file_path}: {exc}") from exc
    if info.samplerate != expect_rate_hz:
        raise AudioFormatError(
            f"audio {file_path} has sample rate {info.samplerate} Hz, "
            f"expected {expect_rate_hz} Hz; refusing to resample"
        )
    if info.channels != 1:
        raise AudioFormatError(
            f"audio {file_path} has {info.channels} channels, expected mono; refusing to downmix"
        )
    data, _ = sf.read(str(file_path), dtype="int16", always_2d=False)
    return data.tobytes()


def validate_pcm_sha256(pcm: bytes, expect: str | None, *, path: Path) -> None:
    """Check decoded PCM bytes against a manifest `sha256`, when present."""
    if expect is None:
        return
    actual = hashlib.sha256(pcm).hexdigest()
    if actual != expect:
        raise AudioFormatError(
            f"audio {path} sha256 mismatch: manifest says {expect}, decoded {actual}"
        )


def manifest_corpus_id(path: Path) -> str:
    """`"sha256:" + sha256(manifest bytes).hexdigest()`."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
