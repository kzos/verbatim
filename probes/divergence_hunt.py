# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Batch-composition divergence hunt over full LibriSpeech test splits, with references kept.

For each utterance: transcribe ALONE, then inside a batch of 32, and compare the strings exactly.
Every utterance in a group is zero-padded to one common length before either call, so BATCH SIZE is
the only variable; without that, a different neighbour changes the padded length and a divergence
could be blamed on either.

Unlike the first run (reviews/transcript_hunt.json, 1,024 utterances), this one persists the
LibriSpeech reference and utterance id alongside both hypotheses at collection time. Recovering them
afterwards cost a day.

Exploratory evidence. NOT a harness row: no frozen methodology, no environment record.
"""
import sys, os, io, json, time, warnings, difflib
warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
from datasets import load_dataset, Audio

os.makedirs("probe-output", exist_ok=True)
OUT = os.environ.get("PROBE_OUT", "probe-output/divergence_hunt_full.json")
BATCH = 32
SPLITS = [("other", "test"), ("clean", "test")]
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
dev = torch.device("cuda:0")   # physical GPU chosen by CUDA_VISIBLE_DEVICES


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


state = {"model": MODEL, "batch_size": BATCH, "control": "all rows padded to one common length",
         "machine": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "nemo": __import__("nemo").__version__,
         "splits": {}, "divergences": [], "total_checked": 0}
t0 = time.time()

for cfg, split in SPLITS:
    tag = f"{split}-{cfg}"
    print(f"\n[data] {tag}", flush=True)
    ds = load_dataset("openslr/librispeech_asr", cfg, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    buf, meta, checked = [], [], 0
    for rec in ds:
        buf.append(pcm_of(rec))
        meta.append({"id": rec["id"], "reference": rec["text"].lower()})
        if len(buf) < BATCH:
            continue
        L = max(len(a) for a in buf)
        batched = texts(buf, L)
        for j, a in enumerate(buf):
            alone = texts([a], L)[0]
            checked += 1
            state["total_checked"] += 1
            if alone != batched[j]:
                ref = meta[j]["reference"]
                ea, na, oa = word_errors(ref, alone)
                eb, nb, ob = word_errors(ref, batched[j])
                d = {"split": tag, "n_in_split": checked, "librispeech_id": meta[j]["id"],
                     "reference": ref, "alone": alone, "in_batch": batched[j],
                     "alone_word_errors": ea, "in_batch_word_errors": eb, "ref_words": na,
                     "alone_ops": oa, "in_batch_ops": ob,
                     "changed_tokens": [{"alone": x["hyp"], "in_batch": y["hyp"], "reference": x["ref"],
                                         "lev_alone": lev(x["ref"], x["hyp"]), "lev_in_batch": lev(y["ref"], y["hyp"])}
                                        for x, y in zip(oa, ob) if x["hyp"] != y["hyp"]]}
                state["divergences"].append(d)
                print(f"  *** DIVERGENCE #{len(state['divergences'])} {tag} n={checked} id={meta[j]['id']}", flush=True)
                print(f"      ref     : {ref}", flush=True)
                print(f"      alone   : {alone}   ({ea}/{na} word err)", flush=True)
                print(f"      batch32 : {batched[j]}   ({eb}/{nb} word err)", flush=True)
        buf, meta = [], []
        if checked % 320 == 0:
            print(f"  {tag}: {checked} checked, {len(state['divergences'])} divergent, {time.time()-t0:.0f}s", flush=True)
            state["splits"][tag] = checked
            json.dump(state, open(OUT, "w"), indent=1)
    # the ragged tail: fewer than BATCH left over, still worth checking as its own group
    if buf:
        L = max(len(a) for a in buf)
        batched = texts(buf, L)
        for j, a in enumerate(buf):
            alone = texts([a], L)[0]
            checked += 1
            state["total_checked"] += 1
            if alone != batched[j]:
                ref = meta[j]["reference"]
                ea, na, oa = word_errors(ref, alone)
                eb, nb, ob = word_errors(ref, batched[j])
                state["divergences"].append({"split": tag, "n_in_split": checked, "tail_group_size": len(buf),
                                             "librispeech_id": meta[j]["id"], "reference": ref,
                                             "alone": alone, "in_batch": batched[j],
                                             "alone_word_errors": ea, "in_batch_word_errors": eb, "ref_words": na,
                                             "alone_ops": oa, "in_batch_ops": ob})
                print(f"  *** DIVERGENCE (tail) {tag} n={checked} id={meta[j]['id']}", flush=True)
    state["splits"][tag] = checked
    json.dump(state, open(OUT, "w"), indent=1)
    print(f"[done] {tag}: {checked} utterances, {len(state['divergences'])} divergences so far", flush=True)

state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)
print(f"\n=== {len(state['divergences'])} divergences in {state['total_checked']} utterances "
      f"(batch 1 vs batch {BATCH}), {state['seconds']}s ===", flush=True)
for d in state["divergences"]:
    print(f"  {d['split']} {d['librispeech_id']}: alone {d['alone_word_errors']}/{d['ref_words']} err, "
          f"batch {d['in_batch_word_errors']}/{d['ref_words']} err", flush=True)
