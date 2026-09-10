# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests for manifest loading and PCM decoding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from verbatim_bench.corpus import (
    AudioFormatError,
    ManifestError,
    load_manifest,
    manifest_corpus_id,
    read_pcm16,
)

pytestmark = pytest.mark.cpu

RATE_HZ = 16000


def make_wav(path: Path, seconds: float, rate_hz: int = RATE_HZ, channels: int = 1) -> None:
    n = int(seconds * rate_hz)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(n) / rate_hz).astype(np.float32)
    if channels == 2:
        tone = np.stack([tone, tone], axis=1)
    sf.write(str(path), tone, rate_hz)


def write_manifest(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def entry(audio_path: str, duration: float, stream_id: str) -> dict:
    return {
        "audio_filepath": audio_path,
        "duration": duration,
        "text": f"reference for {stream_id}",
        "stream_id": stream_id,
    }


def test_load_manifest_preserves_order(tmp_path: Path) -> None:
    for name, seconds in (("a.wav", 0.5), ("b.wav", 0.3), ("c.wav", 0.1)):
        make_wav(tmp_path / name, seconds)
    manifest = tmp_path / "m.jsonl"
    write_manifest(
        manifest,
        [
            entry("a.wav", 0.5, "s-a"),
            entry("b.wav", 0.3, "s-b"),
            entry("c.wav", 0.1, "s-c"),
        ],
    )
    loaded = load_manifest(manifest)
    assert [u.stream_id for u in loaded] == ["s-a", "s-b", "s-c"]
    durations = [u.duration_s for u in loaded]
    assert durations == [0.5, 0.3, 0.1]
    assert durations != sorted(durations), "manifest was sorted by duration; order is the fixture"


def test_load_manifest_rejects_duplicate_stream_id(tmp_path: Path) -> None:
    make_wav(tmp_path / "a.wav", 0.1)
    manifest = tmp_path / "m.jsonl"
    write_manifest(manifest, [entry("a.wav", 0.1, "dup"), entry("a.wav", 0.1, "dup")])
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(manifest)


def test_load_manifest_rejects_missing_key(tmp_path: Path) -> None:
    make_wav(tmp_path / "a.wav", 0.1)
    manifest = tmp_path / "m.jsonl"
    bad = entry("a.wav", 0.1, "s-a")
    del bad["stream_id"]
    write_manifest(manifest, [bad])
    with pytest.raises(ManifestError, match="stream_id"):
        load_manifest(manifest)


def test_relative_audio_paths_resolve_against_the_manifest_directory(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    make_wav(audio_dir / "utt.wav", 0.2)
    manifest = tmp_path / "m.jsonl"
    write_manifest(manifest, [entry("audio/utt.wav", 0.2, "s-a")])
    (other := tmp_path / "elsewhere").mkdir()
    assert (other / "utt.wav").exists() is False
    loaded = load_manifest(manifest)
    assert loaded[0].audio_path == audio_dir / "utt.wav"


def test_read_pcm16_returns_expected_length(tmp_path: Path) -> None:
    wav = tmp_path / "one.wav"
    make_wav(wav, 1.0)
    assert len(read_pcm16(wav)) == 32000


def test_read_pcm16_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    wav = tmp_path / "eight.wav"
    make_wav(wav, 0.5, rate_hz=8000)
    with pytest.raises(AudioFormatError, match="8000"):
        read_pcm16(wav)


def test_read_pcm16_rejects_stereo(tmp_path: Path) -> None:
    wav = tmp_path / "stereo.wav"
    make_wav(wav, 0.2, channels=2)
    with pytest.raises(AudioFormatError, match=r"[Mm]ono"):
        read_pcm16(wav)


def test_manifest_corpus_id_is_stable_and_prefixed(tmp_path: Path) -> None:
    make_wav(tmp_path / "a.wav", 0.1)
    manifest = tmp_path / "m.jsonl"
    write_manifest(manifest, [entry("a.wav", 0.1, "s-a")])
    first = manifest_corpus_id(manifest)
    assert first.startswith("sha256:")
    assert manifest_corpus_id(manifest) == first
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert manifest_corpus_id(manifest) != first
