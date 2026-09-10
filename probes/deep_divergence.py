# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Compare what the decoder ACTUALLY returns, not just the string.

Every divergence run so far compared `hypothesis.text`. That is the least sensitive instrument the
decoder offers, and it makes every count published so far a LOWER BOUND: two different token sequences
can normalise to the same string, and a session's timestamps can move without a word changing.

The Hypothesis object carries, and we were discarding:
    y_sequence        the token ids actually emitted
    timestamp         the frame index of each emitted token   <- the project claims these are invariant
                                                                 and has never once compared one
    score             the hypothesis score, the most sensitive detector of all
    token_confidence  per-token confidence  <- would MEASURE the "sits near a decision boundary" claim
                                               that the evidence file currently only asserts
    alignments        the per-frame token alignment

Same control as before: every row in a group zero-padded to one common length, so batch size is the only
variable. Alone versus a batch of 32.

Exploratory. NOT a harness row.
"""
import io
import os, json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
from datasets import load_dataset, Audio
from omegaconf import OmegaConf, open_dict

os.makedirs("probe-output", exist_ok=True)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PROBE_OUT", "probe-output/deep_divergence.json")
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
BATCH = 32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
dev = torch.device("cuda:0")

from nemo.collections.asr.models import ASRModel

print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}", flush=True)
m = ASRModel.from_pretrained(MODEL, map_location="cpu").to(dev).eval()

# ---- turn on everything the decoder is willing to hand back -------------------------------------
d = OmegaConf.create(OmegaConf.to_container(m.cfg.decoding, resolve=True))
enabled = {}
with open_dict(d):
    d.strategy = "greedy_batch"
    d.compute_timestamps = True
    d.preserve_alignments = True
    if "greedy" not in d:
        d.greedy = {}
    d.greedy.preserve_alignments = True
    d.greedy.preserve_frame_confidence = True
    d.confidence_cfg = {"preserve_frame_confidence": True,
                        "preserve_token_confidence": True,
                        "preserve_word_confidence": True,
                        "exclude_blank": True,
                        "aggregation": "min",
                        "method_cfg": {"name": "max_prob"}}
try:
    m.change_decoding_strategy(d, decoder_type="rnnt")
    enabled["requested"] = True
except Exception as e:  # a config shape we guessed wrong should not cost the whole run
    print(f"[warn] full config rejected ({e}); retrying with timestamps only", flush=True)
    d2 = OmegaConf.create(OmegaConf.to_container(m.cfg.decoding, resolve=True))
    with open_dict(d2):
        d2.strategy = "greedy_batch"
        d2.compute_timestamps = True
    m.change_decoding_strategy(d2, decoder_type="rnnt")
    enabled["requested"] = "timestamps-only"
print(f"[model] ready, decoding extras: {enabled}", flush=True)


def hyps_of(rows, L):
    padded = []
    for r in rows:
        a = np.zeros(L, dtype=np.float32)
        a[: len(r)] = r
        padded.append(a)
    with torch.no_grad():
        return m.transcribe(padded, batch_size=len(padded), verbose=False, timestamps=True)


def snap(h):
    """Everything the hypothesis will give us, normalised to comparable python."""
    def listify(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, (list, tuple)):
            return [listify(i) for i in x] if x and isinstance(x[0], torch.Tensor) else list(x)
        return x
    return {
        "text": getattr(h, "text", None),
        "tokens": listify(getattr(h, "y_sequence", None)),
        "timestamp": listify(getattr(h, "timestamp", None)),
        "score": float(getattr(h, "score", float("nan"))) if getattr(h, "score", None) is not None else None,
        "token_confidence": listify(getattr(h, "token_confidence", None)),
        "word_confidence": listify(getattr(h, "word_confidence", None)),
    }


def pcm_of(rec):
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == 16000, sr
    return np.asarray(data, dtype=np.float32)


ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
pool, meta = [], []
for rec in ds:
    pool.append(pcm_of(rec))
    meta.append({"id": rec["id"], "reference": rec["text"].lower()})
    if len(pool) >= N:
        break
print(f"[data] {len(pool)} utterances", flush=True)

state = {"machine": torch.cuda.get_device_name(0), "torch": torch.__version__,
         "model": MODEL, "batch": BATCH, "n": len(pool),
         "control": "every row padded to one common length; batch size is the only variable",
         "note": "text divergence is a LOWER BOUND; the finer channels are what this run adds",
         "fields_available": None,
         "counts": {"text": 0, "tokens": 0, "timestamp": 0, "score_exact": 0, "checked": 0},
         "score_deltas": [], "examples": []}

t0 = time.time()
for start in range(0, len(pool) - BATCH + 1, BATCH):
    group = pool[start : start + BATCH]
    L = max(len(a) for a in group)
    bh = hyps_of(group, L)
    for j, a in enumerate(group):
        ah = hyps_of([a], L)[0]
        A, Bs = snap(ah), snap(bh[j])
        if state["fields_available"] is None:
            state["fields_available"] = {k: (v is not None and v != []) for k, v in A.items()}
            print(f"[fields] actually populated: {state['fields_available']}", flush=True)
        state["counts"]["checked"] += 1
        diff_text = A["text"] != Bs["text"]
        diff_tok = A["tokens"] != Bs["tokens"]
        diff_ts = A["timestamp"] != Bs["timestamp"]
        same_score = (A["score"] is not None and Bs["score"] is not None and A["score"] == Bs["score"])
        state["counts"]["text"] += diff_text
        state["counts"]["tokens"] += diff_tok
        state["counts"]["timestamp"] += diff_ts
        state["counts"]["score_exact"] += (not same_score)
        if A["score"] is not None and Bs["score"] is not None:
            state["score_deltas"].append(abs(A["score"] - Bs["score"]))
        if diff_text or diff_tok or diff_ts:
            mj = meta[start + j]
            state["examples"].append({
                "librispeech_id": mj["id"], "reference": mj["reference"],
                "text_differs": diff_text, "tokens_differ": diff_tok, "timestamps_differ": diff_ts,
                "alone_text": A["text"], "batch_text": Bs["text"],
                "alone_tokens": A["tokens"], "batch_tokens": Bs["tokens"],
                "alone_timestamp": A["timestamp"], "batch_timestamp": Bs["timestamp"],
                "alone_score": A["score"], "batch_score": Bs["score"],
                "alone_token_confidence": A["token_confidence"], "batch_token_confidence": Bs["token_confidence"],
            })
            kinds = [k for k, v in (("text", diff_text), ("tokens", diff_tok), ("timestamps", diff_ts)) if v]
            print(f"  *** {mj['id']} differs in: {', '.join(kinds)}", flush=True)
    if state["counts"]["checked"] % 320 == 0:
        c = state["counts"]
        print(f"  {c['checked']}: text {c['text']}, tokens {c['tokens']}, timestamps {c['timestamp']}, "
              f"score-not-exact {c['score_exact']}, {time.time()-t0:.0f}s", flush=True)
        json.dump(state, open(OUT, "w"), indent=1)

sd = state["score_deltas"]
state["score_delta_summary"] = ({"n": len(sd), "max": max(sd), "mean": sum(sd) / len(sd),
                                "nonzero": sum(1 for x in sd if x != 0.0)} if sd else None)
state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)

c = state["counts"]
print("\n" + "=" * 66)
print(f"checked                          {c['checked']}")
print(f"differ in TEXT                   {c['text']}   <- all previous runs measured only this")
print(f"differ in TOKEN IDS              {c['tokens']}")
print(f"differ in TIMESTAMPS             {c['timestamp']}   <- never measured before today")
print(f"score not bit-identical          {c['score_exact']}")
if state["score_delta_summary"]:
    s = state["score_delta_summary"]
    print(f"score delta: max {s['max']:.6g}, mean {s['mean']:.6g}, nonzero in {s['nonzero']} of {s['n']}")
print("=" * 66)
