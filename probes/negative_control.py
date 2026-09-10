# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""The control every previous run should have had, plus one arm nobody asked for.

Everything measured so far attributes divergence to BATCH COMPOSITION. That attribution is only valid if
an identical call, repeated, is bit-identical. Nobody checked. If the decode is nondeterministic run to
run, some fraction of every count published is GPU noise rather than batch composition, and the whole
argument is confounded.

Three arms, same utterances, same padding to a common length:

  A  REPEAT          transcribe the identical batch twice, compare.
                     Expected: zero. If not zero, every other number here is contaminated.
  B  COMPOSITION     alone versus inside a batch of 32. What every prior run measured.
  C  ROW ORDER       the SAME 32 utterances, permuted, comparing each utterance to itself.
                     Batch size, shapes and members are identical; only a row's POSITION changes.
                     Fixed-shape bucketing does not fix this if it bites, so it is worth knowing.

Arm C is the one an engineer asks about and nobody here had run: a session's row index in a batch is its
rank among that tick's live streams, which changes as sessions come and go.

Compares transcript text AND word start/end offsets, because the project promises both.
"""
import io
import os
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

os.makedirs("probe-output", exist_ok=True)
OUT = os.environ.get("PROBE_OUT", "probe-output/negative_control.json")
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
BATCH = 32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
SEED = 20260909
dev = torch.device("cuda:0")

from nemo.collections.asr.models import ASRModel

m = ASRModel.from_pretrained(MODEL, map_location="cpu").to(dev).eval()
print(f"[env] {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)


def hyps(rows, L):
    padded = []
    for r in rows:
        a = np.zeros(L, dtype=np.float32)
        a[: len(r)] = r
        padded.append(a)
    with torch.no_grad():
        return m.transcribe(padded, batch_size=len(padded), verbose=False, timestamps=True)


def key(h):
    """Text plus word timing, which is the pair the project actually promises."""
    ts = getattr(h, "timestamp", None)
    words = None
    if isinstance(ts, dict) and ts.get("word"):
        words = [(w.get("word"), w.get("start_offset"), w.get("end_offset")) for w in ts["word"]]
    return (getattr(h, "text", None), words)


def pcm_of(rec):
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return np.asarray(data, dtype=np.float32)


ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
pool, ids = [], []
for rec in ds:
    pool.append(pcm_of(rec))
    ids.append(rec["id"])
    if len(pool) >= N:
        break
print(f"[data] {len(pool)} utterances", flush=True)

rng = np.random.default_rng(SEED)
state = {
    "machine": torch.cuda.get_device_name(0),
    "n": len(pool),
    "batch": BATCH,
    "seed": SEED,
    "compared": "transcript text AND word start/end offsets",
    "arms": {
        "A_repeat": {"text": 0, "timing": 0, "checked": 0, "examples": []},
        "B_composition": {"text": 0, "timing": 0, "checked": 0, "examples": []},
        "C_row_order": {"text": 0, "timing": 0, "checked": 0, "examples": []},
    },
}
t0 = time.time()

for start in range(0, len(pool) - BATCH + 1, BATCH):
    group = pool[start : start + BATCH]
    gids = ids[start : start + BATCH]
    L = max(len(a) for a in group)

    b1 = [key(h) for h in hyps(group, L)]
    b2 = [key(h) for h in hyps(group, L)]  # arm A: identical call, again
    perm = rng.permutation(BATCH)
    bp = [key(h) for h in hyps([group[i] for i in perm], L)]  # arm C: same members, new positions
    unperm = [None] * BATCH
    for pos, orig in enumerate(perm):
        unperm[orig] = bp[pos]

    for j in range(BATCH):
        alone = key(hyps([group[j]], L)[0])  # arm B
        for arm, other in (("A_repeat", b2[j]), ("B_composition", alone), ("C_row_order", unperm[j])):
            a = state["arms"][arm]
            a["checked"] += 1
            dt = b1[j][0] != other[0]
            dtime = b1[j][1] != other[1]
            a["text"] += dt
            a["timing"] += dtime
            if (dt or dtime) and len(a["examples"]) < 25:
                a["examples"].append(
                    {
                        "id": gids[j],
                        "text_differs": bool(dt),
                        "timing_differs": bool(dtime),
                        "ref_text": b1[j][0],
                        "other_text": other[0],
                    }
                )
                print(f"  [{arm}] {gids[j]} text={dt} timing={dtime}", flush=True)
    if state["arms"]["A_repeat"]["checked"] % 320 == 0:
        s = {k: (v["text"], v["timing"]) for k, v in state["arms"].items()}
        print(
            f"  {state['arms']['A_repeat']['checked']}: (text,timing) {s}  {time.time() - t0:.0f}s",
            flush=True,
        )
        json.dump(state, open(OUT, "w"), indent=1)

state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)
print("\n" + "=" * 74)
for arm, label in (
    ("A_repeat", "A  identical call repeated  (MUST be 0/0)"),
    ("B_composition", "B  alone vs batch of 32    (the effect under study)"),
    ("C_row_order", "C  same batch, rows permuted (position only)"),
):
    a = state["arms"][arm]
    print(f"{label:48s} text {a['text']:4d}   word-timing {a['timing']:4d}   of {a['checked']}")
print("=" * 74)
if state["arms"]["A_repeat"]["text"] or state["arms"]["A_repeat"]["timing"]:
    print("ARM A IS NOT ZERO. The decode is not reproducible run to run, and every count this project")
    print("has published conflates batch composition with run-to-run nondeterminism.")
else:
    print("Arm A is clean: the decode is reproducible run to run, so arms B and C measure what they claim.")
