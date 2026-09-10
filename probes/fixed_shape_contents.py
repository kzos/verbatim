# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Does FIXED-SHAPE batching actually fix the transcript, or only the encoder's bits?

This is the mechanism the whole server is built on and nobody has tested it end to end.

Every previous probe compared batch 1 against batch 32, which changes the batch SHAPE. The project's
claim is different and stronger: that once the shape is fixed, the CONTENTS of the batch stop mattering.
Section 5 of the evidence file showed pad content is irrelevant at the level of encoder floats. It has
never been shown at the level of a transcript, and it has never been shown against real neighbours
rather than against padding.

So: batch size pinned at 32 in both arms, every row padded to one common length in both arms. The only
thing that differs is who the other 31 rows are.

    arm SILENCE    row 0 is the target, rows 1..31 are zeros
    arm CROWD      row 0 is the target, rows 1..31 are real utterances

Identical shape, identical dtype, identical row position, identical everything the scheduler controls.
If the target's transcript or word timings differ between these two arms, fixed-shape bucketing does not
deliver what this project claims for it, and the claim has to change rather than the wording.

Compares text AND word start/end offsets, because the project promises both.

Exploratory. NOT a harness row.
"""

import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

os.makedirs("probe-output", exist_ok=True)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PROBE_OUT", "probe-output/fixed_shape_contents.json")
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
BATCH = 32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
dev = torch.device("cuda:0")

from nemo.collections.asr.models import ASRModel

print(f"[env] {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
m = ASRModel.from_pretrained(MODEL, map_location="cpu").to(dev).eval()


def hyps(rows, L):
    padded = []
    for r in rows:
        a = np.zeros(L, dtype=np.float32)
        a[: len(r)] = r
        padded.append(a)
    with torch.no_grad():
        return m.transcribe(padded, batch_size=len(padded), verbose=False, timestamps=True)


def key(h):
    ts = getattr(h, "timestamp", None)
    words = None
    if isinstance(ts, dict) and ts.get("word"):
        words = [(w.get("word"), w.get("start_offset"), w.get("end_offset")) for w in ts["word"]]
    return (getattr(h, "text", None), words)


def pcm_of(rec):
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == 16000, sr
    return np.asarray(data, dtype=np.float32)


ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
pool, ids = [], []
for rec in ds:
    pool.append(pcm_of(rec))
    ids.append(rec["id"])
    if len(pool) >= N + BATCH:
        break
print(f"[data] {len(pool)} utterances held", flush=True)

state = {
    "machine": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nemo": __import__("nemo").__version__,
    "machine": torch.cuda.get_device_name(0),
    "batch": BATCH,
    "question": "with batch SHAPE held fixed, do the batch CONTENTS change the target's output?",
    "arms": "row 0 is the target in both; rows 1..31 are zeros in one arm and real utterances in the other",
    "compared": "transcript text AND word start/end offsets",
    "checked": 0,
    "text_differs": 0,
    "timing_differs": 0,
    "examples": [],
}
t0 = time.time()

for i in range(N):
    target = pool[i]
    neighbours = [pool[(i + 1 + k) % len(pool)] for k in range(BATCH - 1)]
    L = max(len(a) for a in [target] + neighbours)

    silence = [target] + [np.zeros(1, dtype=np.float32) for _ in range(BATCH - 1)]
    crowd = [target] + neighbours

    a = key(hyps(silence, L)[0])
    b = key(hyps(crowd, L)[0])
    state["checked"] += 1
    dt = a[0] != b[0]
    dtime = a[1] != b[1]
    state["text_differs"] += dt
    state["timing_differs"] += dtime
    if dt or dtime:
        state["examples"].append(
            {
                "id": ids[i],
                "text_differs": bool(dt),
                "timing_differs": bool(dtime),
                "alone_in_silence": a[0],
                "in_a_crowd": b[0],
            }
        )
        print(f"  *** {ids[i]} text={dt} timing={dtime}", flush=True)
        print(f"      silence: {a[0]}", flush=True)
        print(f"      crowd  : {b[0]}", flush=True)
    if state["checked"] % 128 == 0:
        print(
            f"  {state['checked']}: text {state['text_differs']}, "
            f"timing {state['timing_differs']}, {time.time() - t0:.0f}s",
            flush=True,
        )
        json.dump(state, open(OUT, "w"), indent=1)

state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)
print("\n" + "=" * 72)
print(f"batch shape held fixed at {BATCH}, only the other rows' contents changed")
print(f"  checked                    {state['checked']}")
print(f"  target's TEXT differed     {state['text_differs']}")
print(f"  target's WORD TIMING differed {state['timing_differs']}")
print("=" * 72)
if state["text_differs"] == 0 and state["timing_differs"] == 0:
    print("Fixed shape is sufficient: batch contents did not reach the target's output.")
    print("This is the mechanism the server is built on, tested past the encoder for the first time.")
else:
    print("FIXED SHAPE IS NOT SUFFICIENT. Contents reached the output with the shape held constant.")
    print("The project's central claim needs changing, not rewording.")
