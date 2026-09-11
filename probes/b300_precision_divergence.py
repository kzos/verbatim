"""Does the server's default precision cost batch invariance on Blackwell too?

DR-0003 says `serve` defaults to bfloat16, and that on an A6000 bfloat16 diverges 287 in 2,939 on the
server's own code path where float32 diverges 6, with the length control removing nothing. It also says
the measurement nobody has taken is the one that decides the product.

This is that measurement, on a B300, through `verbatim.pipelines.nemo_runtime` -- the same code path
`serve` uses to build its pipeline -- so the thing measured is the server's configuration and not a
probe's imitation of it.

Two arms per precision, mirroring every earlier streaming probe so the numbers are comparable:
  ragged     rows keep their own lengths; a short stream's last frame lands early with real right padding
  equalised  every row zero-padded to one common length, so batch size is the only variable

Each arm compares a target ALONE against the same target in slot 0 of a batch of BATCH.

Exploratory. NOT a harness row.
"""

import io
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

sys.path.insert(0, os.environ.get("VERBATIM_SRC", "/opt/verbatim/repo/src"))

from verbatim.config import ChunkMode
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, att_context_size, build_pipeline

OUT = os.environ.get("OUT", "/opt/verbatim/b300_precision_divergence.json")
MODEL = os.environ.get("MODEL", "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
BATCH = int(os.environ.get("BATCH", "32"))
N_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
CHUNK_MS = int(os.environ.get("CHUNK_MS", "1120"))
DTYPES = os.environ.get("DTYPES", "bfloat16,float32").split(",")
GRAPHS = os.environ.get("GRAPHS", "0") == "1"

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

chunk = ChunkMode(CHUNK_MS)
att = att_context_size(MODEL, chunk, left=70)

print("[data] streaming LibriSpeech test-other", flush=True)
ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
pool, meta = [], []
for rec in ds:
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == 16000, sr
    pool.append(np.asarray(data, dtype=np.float32))
    meta.append({"id": rec["id"], "reference": rec["text"].lower()})
    if len(pool) >= N_TARGETS + BATCH:
        break
print(f"[data] {len(pool)} utterances held", flush=True)

state = {
    "model": MODEL,
    "machine": torch.cuda.get_device_name(0),
    "capability": "sm_" + "".join(map(str, torch.cuda.get_device_capability(0))),
    "torch": torch.__version__,
    "path": "verbatim.pipelines.nemo_runtime.build_pipeline -> CacheAwareRNNTPipeline.transcribe_step",
    "note": "built through the same code serve uses, so this measures the server's configuration",
    "att_context_size": list(att),
    "chunk_ms": CHUNK_MS,
    "batch": BATCH,
    "use_cuda_graphs": GRAPHS,
    "runs": {},
}

for dtype in DTYPES:
    spec = NeMoPipelineSpec(
        model=MODEL, chunk=chunk, att_context=tuple(att),
        num_slots=max(256, BATCH * 4), batch_size=BATCH,
        use_cuda_graphs=GRAPHS, compute_dtype=dtype, matmul_precision="highest",
    )
    print(f"\n[build] {dtype}, graphs={GRAPHS}", flush=True)
    pipeline = build_pipeline(spec)
    import nemo
    state["nemo"] = nemo.__version__
    N = int(round(float(pipeline.chunk_size_in_secs) * int(pipeline.sample_rate)))
    print(f"[build] chunk {pipeline.chunk_size_in_secs}s = {N} samples", flush=True)
    nid = [1]

    def stream(audios, pad_to=None):
        rows = []
        for a in audios:
            if pad_to is not None and len(a) < pad_to:
                p = np.zeros(pad_to, dtype=np.float32); p[: len(a)] = a; a = p
            rows.append(np.asarray(a, dtype=np.float32))
        ids = []
        for _ in rows:
            ids.append(nid[0]); nid[0] += 1
        steps = max(int(math.ceil(len(a) / N)) for a in rows)
        finals = {s: [] for s in ids}
        opts = ASRRequestOptions()
        for step in range(steps):
            frames = []
            for sid, a in zip(ids, rows):
                total = int(math.ceil(len(a) / N))
                if step >= total:
                    continue
                piece = a[step * N : step * N + N]
                valid = len(piece)
                if valid < N:
                    b = np.zeros(N, dtype=np.float32); b[:valid] = piece; piece = b
                frames.append(Frame(
                    samples=torch.from_numpy(np.ascontiguousarray(piece)), stream_id=sid,
                    is_first=(step == 0), is_last=(step == total - 1), length=valid,
                    options=opts if step == 0 else None))
            if not frames:
                break
            with torch.inference_mode():
                outs = pipeline.transcribe_step(frames)
            for o in outs:
                txt = str(o.final_transcript or "").strip()
                if txt:
                    finals[int(o.stream_id)].append(txt)
        return [" ".join(finals[s]).strip() for s in ids]

    probe = stream([pool[0], pool[1]])
    print(f"  [guard] ref {meta[0]['reference'][:70]}", flush=True)
    print(f"  [guard] hyp {probe[0][:70]}", flush=True)
    if not all(x.strip() for x in probe):
        raise SystemExit("[guard] FAILED: empty transcript; counts would be meaningless")
    ov = len(set(meta[0]["reference"].split()) & set(probe[0].lower().split())) / max(1, len(set(meta[0]["reference"].split())))
    print(f"  [guard] overlap {ov:.2f}", flush=True)
    if ov < 0.5:
        raise SystemExit(f"[guard] FAILED: overlap {ov:.2f}")

    state["runs"][dtype] = {}
    for arm in ("ragged", "equalised"):
        res = {"checked": 0, "divergences": []}
        t0 = time.time()
        for i in range(N_TARGETS):
            target = pool[i]
            nbrs = [pool[(i + 1 + k) % len(pool)] for k in range(BATCH - 1)]
            common = max(len(a) for a in [target] + nbrs) if arm == "equalised" else None
            alone = stream([target], pad_to=common)[0]
            batched = stream([target] + nbrs, pad_to=common)[0]
            res["checked"] += 1
            if alone != batched:
                res["divergences"].append({
                    "n": i, "librispeech_id": meta[i]["id"], "reference": meta[i]["reference"],
                    "alone": alone, "in_batch": batched})
            if (i + 1) % 16 == 0:
                print(f"  {dtype}/{arm} {i+1}/{N_TARGETS}: {len(res['divergences'])} divergent, {time.time()-t0:.0f}s", flush=True)
                state["runs"][dtype][arm] = res
                json.dump(state, open(OUT, "w"), indent=1)
        state["runs"][dtype][arm] = res
        json.dump(state, open(OUT, "w"), indent=1)
        print(f"=== {dtype}/{arm}: {len(res['divergences'])} in {res['checked']} ===", flush=True)
    del pipeline
    torch.cuda.empty_cache()

json.dump(state, open(OUT, "w"), indent=1)
for dt, arms in state["runs"].items():
    for arm, r in arms.items():
        print(f"FINAL {dt}/{arm}: {len(r['divergences'])} in {r['checked']}", flush=True)
sys.stdout.flush()
os._exit(0)
