# `corpora/` — corpus manifests only, never audio

Each corpus is `<id>.jsonl` plus `<id>.yaml`:

- the **JSONL** is NeMo-manifest-compatible (`audio_filepath`, `duration`, `text`, a stable `stream_id`,
  a per-file `sha256`), so NeMo's own `asr_streaming_infer.py` — benchmark arm (b) — reads it with no
  conversion;
- the **YAML** carries the Hugging Face dataset id *and revision*, the licence, the selection seed, the
  reference transcripts' digest and the manifest's sha256.

The manifest's sha256 is the `corpus_id` stamped on every row.

Audio never enters git. LibriSpeech and FLEURS are CC-BY-4.0 and redistribution would be lawful; the
reason not to is that a repository carrying a gigabyte of WAV is a repository nobody clones, and a
pinned revision plus a checksum is a stronger reproducibility claim than a copy.

One trap, recorded here because it is easy to reintroduce: NeMo's `prepare_audio_data(...,
sort_by_duration=True)` reorders a corpus by duration. The load generator must **not** sort — arrival
order is part of the fixture, and duration-sorting is exactly the batch-composition variable the
invariance work is about.
