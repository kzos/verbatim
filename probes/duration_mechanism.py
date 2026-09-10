# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Does the length-coupling claim hold as a MEASURED relationship, or only as a reading of examples?

Section 13 of the evidence file says a short utterance streaming beside longer neighbours keeps
receiving trailing chunks and decodes further than it does alone. That was concluded by reading the
38 divergent cases and noticing every one differed at the tail. It was never tested as a relationship.

It is testable, and the data needed is free. Each utterance's DURATION is in the dataset, and the
streaming probe's batch grouping is deterministic: the target at index i is grouped with i+1..i+31,
wrapping. So for every utterance we can compute how much SHORTER it is than the longest member of its
own group, and ask whether the divergent ones sit in the tail of that distribution.

Prediction if the mechanism is right: divergent utterances have a large deficit against their group
maximum. If divergence is unrelated to duration, the two distributions overlap.

No GPU. No model. Pure arithmetic over the dataset and a run already on disk.
"""
import io
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
from datasets import load_dataset, Audio

RUN = os.environ.get("PROBE_OUT", "probe-output/streaming_divergence.json")
os.makedirs("probe-output", exist_ok=True)
OUT = os.environ.get("PROBE_OUT", "probe-output/duration_mechanism.json")
BATCH = 32

run = json.load(open(RUN))
div_ids = {d["librispeech_id"] for d in run["divergences"]}
print(f"[run] {run['checked']} checked, {len(div_ids)} divergent")

print("[data] measuring every utterance's duration", flush=True)
ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
dur, ids = [], []
for rec in ds:
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    info = sf.info(io.BytesIO(raw))
    dur.append(info.frames / info.samplerate)
    ids.append(rec["id"])
    if len(dur) >= run["checked"]:
        break
dur = np.asarray(dur)
n = len(dur)
print(f"[data] {n} durations, median {np.median(dur):.2f}s, max {dur.max():.2f}s", flush=True)

# the probe groups target i with i+1..i+31, wrapping
deficit = np.empty(n)
for i in range(n):
    idx = [(i + k) % n for k in range(BATCH)]
    deficit[i] = dur[idx].max() - dur[i]

is_div = np.array([u in div_ids for u in ids])
d_div, d_non = deficit[is_div], deficit[~is_div]

def pct(x, q):
    return float(np.percentile(x, q))

summary = {
    "machine": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nemo": __import__("nemo").__version__,
    "n": int(n), "divergent": int(is_div.sum()),
    "own_duration_s": {"divergent_median": float(np.median(dur[is_div])),
                       "other_median": float(np.median(dur[~is_div]))},
    "deficit_vs_group_max_s": {
        "divergent": {"median": float(np.median(d_div)), "p25": pct(d_div, 25), "p75": pct(d_div, 75),
                      "min": float(d_div.min()), "max": float(d_div.max())},
        "other": {"median": float(np.median(d_non)), "p25": pct(d_non, 25), "p75": pct(d_non, 75),
                  "min": float(d_non.min()), "max": float(d_non.max())},
    },
}
# how far up the deficit distribution does a typical divergent utterance sit?
ranks = [float((d_non < x).mean()) for x in d_div]
summary["divergent_percentile_within_others"] = {
    "median": float(np.median(ranks)), "min": float(min(ranks)), "max": float(max(ranks)),
    "above_90th": int(sum(1 for r in ranks if r >= 0.90)),
    "above_75th": int(sum(1 for r in ranks if r >= 0.75)),
}
json.dump(summary, open(OUT, "w"), indent=1)

print("\n" + "=" * 70)
print("HOW MUCH SHORTER IS AN UTTERANCE THAN THE LONGEST IN ITS OWN BATCH GROUP?")
print(f"  divergent  median {summary['deficit_vs_group_max_s']['divergent']['median']:.2f}s "
      f"(p25 {summary['deficit_vs_group_max_s']['divergent']['p25']:.2f}, "
      f"p75 {summary['deficit_vs_group_max_s']['divergent']['p75']:.2f})")
print(f"  all others median {summary['deficit_vs_group_max_s']['other']['median']:.2f}s "
      f"(p25 {summary['deficit_vs_group_max_s']['other']['p25']:.2f}, "
      f"p75 {summary['deficit_vs_group_max_s']['other']['p75']:.2f})")
r = summary["divergent_percentile_within_others"]
print(f"\n  a divergent utterance sits at the {r['median']*100:.0f}th percentile of that deficit, typically")
print(f"  {r['above_75th']} of {summary['divergent']} are above the 75th percentile")
print(f"  {r['above_90th']} of {summary['divergent']} are above the 90th percentile")
print(f"\n  own duration: divergent median {summary['own_duration_s']['divergent_median']:.2f}s "
      f"vs {summary['own_duration_s']['other_median']:.2f}s for the rest")
print("=" * 70)
