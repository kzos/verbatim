"""Build the corpus manifest the benchmark harness needs. Audio to disk, never to git.

`corpora/README.md` sets the contract: NeMo-compatible JSONL plus a YAML carrying the dataset id and
revision, the licence, the selection seed, and the manifest's own sha256, which becomes the corpus_id
stamped on every row. It also records one trap: the load generator must NOT sort by duration, because
arrival order is part of the fixture and duration-sorting is the batch-composition variable the whole
invariance argument is about. This writer preserves dataset order and says so in the YAML.
"""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
OUT = Path(os.environ.get("CORPUS_DIR", "/elm/optim/verbatim_orchestration/corpus/librispeech-test-other-256"))
DATASET, CONFIG, SPLIT = "openslr/librispeech_asr", "other", "test"
WAV = OUT / "audio"
WAV.mkdir(parents=True, exist_ok=True)

ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
revision = getattr(getattr(ds, "info", None), "version", None)
ds = ds.cast_column("audio", Audio(decode=False))

rows, total_s = [], 0.0
for rec in ds:
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="int16")
    if sr != 16000 or data.ndim != 1:
        raise SystemExit(f"{rec['id']}: expected 16 kHz mono, got {sr} Hz ndim={data.ndim}")
    path = WAV / f"{rec['id']}.wav"
    sf.write(path, data, sr, subtype="PCM_16")
    # The harness hashes the DECODED PCM it sends, not the WAV container.
    digest = hashlib.sha256(data.tobytes()).hexdigest()
    dur = len(data) / sr
    total_s += dur
    rows.append({
        "audio_filepath": str(path),
        "duration": round(dur, 4),
        "text": rec["text"].lower(),
        "stream_id": rec["id"],
        "sha256": digest,
    })
    if len(rows) >= N:
        break

jsonl = OUT / "librispeech-test-other-256.jsonl"
with jsonl.open("w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
corpus_id = "sha256:" + hashlib.sha256(jsonl.read_bytes()).hexdigest()

refs = hashlib.sha256("\n".join(r["text"] for r in rows).encode()).hexdigest()
(OUT / "librispeech-test-other-256.yaml").write_text(
    f"""# Corpus record. Audio is on disk and never in git; this file plus the JSONL reproduce it.
id: librispeech-test-other-256
corpus_id: {corpus_id}          # sha256 of the JSONL; stamped on every row
dataset: {DATASET}
config: {CONFIG}
split: {SPLIT}
revision: {revision!r}
licence: CC-BY-4.0
utterances: {len(rows)}
total_audio_s: {round(total_s, 2)}
sample_rate_hz: 16000
channels: 1
selection: "first {len(rows)} in dataset order"
selection_seed: null            # no sampling: order is the dataset's, unsorted
sorted_by_duration: false       # deliberately NOT sorted; arrival order is part of the fixture
reference_digest: {refs}
""", encoding="utf-8")

print(f"utterances   {len(rows)}")
print(f"total audio  {total_s/60:.1f} min")
print(f"corpus_id    {corpus_id[:16]}...")
print(f"manifest     {jsonl}")
