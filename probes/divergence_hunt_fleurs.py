# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Same batch-composition divergence hunt, on a second corpus: FLEURS en_us.

LibriSpeech is read speech from audiobooks. FLEURS is read speech from a different source with different
recording conditions, and it is the second corpus the day-45 gate names. Running both answers the obvious
objection that the effect is an artefact of one dataset.

FLEURS ships `transcription` (lowercased, normalised) and `raw_transcription` (cased, punctuated). We compare
against `transcription`, which is the closer match to what this checkpoint emits. Numbers are spelled out in
FLEURS but the model may emit digits; such an error appears in BOTH conditions and so cannot manufacture a
divergence, which is what this experiment measures.

Exploratory evidence. NOT a harness row.
"""
import io
import os, json, time, warnings, difflib
warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
from datasets import load_dataset, Audio

os.makedirs("probe-output", exist_ok=True)
OUT = os.environ.get("PROBE_OUT", "probe-output/divergence_hunt_fleurs.json")
BATCH = 32
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
dev = torch.device("cuda:0")


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_errors(ref, hyp):
    rw, hw = ref.split(), hyp.split()
    sm = difflib.SequenceMatcher(None, rw, hw)
    err = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
    ops = [{"op": t, "ref": " ".join(rw[i1:i2]), "hyp": " ".join(hw[j1:j2])}
           for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal"]
    return err, len(rw), ops


def pcm_of(rec):
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == 16000, sr
    return np.asarray(data, dtype=np.float32)


from nemo.collections.asr.models import ASRModel
print(f"[model] loading {MODEL}", flush=True)
m = ASRModel.from_pretrained(MODEL, map_location="cpu").to(dev).eval()
print("[model] ready", flush=True)


def texts(rows, L):
    padded = []
    for r in rows:
        a = np.zeros(L, dtype=np.float32)
        a[:len(r)] = r
        padded.append(a)
    with torch.no_grad():
        hyps = m.transcribe(padded, batch_size=len(padded), verbose=False)
    return [(h.text if hasattr(h, "text") else str(h)) for h in hyps]


print("[data] streaming google/fleurs en_us test", flush=True)
ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

state = {"model": MODEL, "corpus": "google/fleurs en_us test", "batch_size": BATCH,
         "reference_field": "transcription", "control": "all rows padded to one common length",
         "checked": 0, "divergences": []}
buf, meta, t0 = [], [], time.time()


def flush(group):
    L = max(len(a) for a in group)
    batched = texts(group, L)
    for j, a in enumerate(group):
        alone = texts([a], L)[0]
        state["checked"] += 1
        if alone != batched[j]:
            ref = meta[j]["reference"]
            ea, na, oa = word_errors(ref, alone)
            eb, nb, ob = word_errors(ref, batched[j])
            state["divergences"].append({
                "n": state["checked"], "fleurs_id": meta[j]["id"], "reference": ref,
                "alone": alone, "in_batch": batched[j], "ref_words": na,
                "alone_word_errors": ea, "in_batch_word_errors": eb,
                "alone_ops": oa, "in_batch_ops": ob,
                "changed_tokens": [{"reference": x["ref"], "alone": x["hyp"], "in_batch": y["hyp"],
                                    "lev_alone": lev(x["ref"], x["hyp"]), "lev_in_batch": lev(y["ref"], y["hyp"])}
                                   for x, y in zip(oa, ob) if x["hyp"] != y["hyp"]]})
            print(f"  *** DIVERGENCE #{len(state['divergences'])} n={state['checked']} id={meta[j]['id']}", flush=True)
            print(f"      ref     : {ref}", flush=True)
            print(f"      alone   : {alone}   ({ea}/{na})", flush=True)
            print(f"      batch32 : {batched[j]}   ({eb}/{nb})", flush=True)


for rec in ds:
    buf.append(pcm_of(rec))
    meta.append({"id": str(rec.get("id")), "reference": rec["transcription"].lower()})
    if len(buf) < BATCH:
        continue
    flush(buf)
    buf, meta = [], []
    if state["checked"] % 320 == 0:
        print(f"  {state['checked']} checked, {len(state['divergences'])} divergent, {time.time()-t0:.0f}s", flush=True)
        json.dump(state, open(OUT, "w"), indent=1)
if buf:
    state["tail_group_size"] = len(buf)
    flush(buf)

state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)
print(f"\n=== {len(state['divergences'])} divergences in {state['checked']} FLEURS en_us utterances "
      f"(batch 1 vs batch {BATCH}), {state['seconds']}s ===", flush=True)
for d in state["divergences"]:
    print(f"  {d['fleurs_id']}: alone {d['alone_word_errors']}/{d['ref_words']}, "
          f"batch {d['in_batch_word_errors']}/{d['ref_words']}", flush=True)
