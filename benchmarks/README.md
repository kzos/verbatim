# `benchmarks/` — everything derived from `rows/`

- `schema/row.schema.json` — the contract CI validates every row against (schema id `vb-results/1`).
  
- `reference/<checkpoint>/<chunk_mode>.digest` — the NeMo batch-1 streaming reference, computed once on
  owned hardware and committed as a digest, so the invariance gate compares against a recorded number
  rather than spending GPU-hours recomputing a constant.
- `METHODOLOGY.md` — frozen and committed *before* the first head-to-head row exists.
- `gate.json` — the last invariance-gate result. Written by CI only.
- `ROWS.md` — the generated table: streams x p95 x WER x invariance. Never hand-edited.

Corpus attribution lives here too: LibriSpeech (`openslr/librispeech_asr`) and FLEURS
(`google/fleurs`) are CC-BY-4.0.
