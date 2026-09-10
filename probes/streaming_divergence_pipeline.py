"""Does batch composition reach the transcript on the path a SERVER runs, not the example script?

Every streaming number this project has published was measured through
`CacheAwareStreamingAudioBuffer` + `conformer_stream_step`, which is the shipped EXAMPLE path.
`docs/SCOPE.md` scopes the server to `nemo.collections.asr.inference`, a different implementation:
its `cache_feature_bufferer.update()` computes a per-stream `right_padding` for every frame, and
`transcribe_step_for_frames` passes those paddings ONLY for the final sub-batch, splitting finals
away from the steady rows with `keep_all_outputs=True`. Steady rows are therefore uniform in length
by construction and ragged lengths exist only at end of stream.

So the published "38 in 2,939, on the path a server actually runs" is not supported by what produced
it, and this script is what settles that. It drives `CacheAwareRNNTPipeline.transcribe_step` directly,
frame by frame, exactly as a server would.

Two arms, mirroring the two existing streaming probes so the numbers are comparable:

  UNCONTROLLED   rows keep their own lengths, so a short stream's `is_last` lands in an earlier tick
                 than its neighbours' and carries a real `right_padding`. This is the arm the
                 example-path probe scored 38 on.
  CONTROLLED     every row zero-padded to one common length BEFORE the pipeline sees it, so every
                 stream's `is_last` lands in the same tick with the same padding. Batch size is then
                 the only variable. The example-path probe scored 1 on this arm.

Each arm compares the target's transcript ALONE (batch of 1) against the same target in slot 0 of a
batch of BATCH. Same model, same weights, same att_context_size, same chunk schedule, same corpus and
the same batch size as the example-path runs.

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
from omegaconf import OmegaConf

OUT = os.environ.get(
    "OUT", "/elm/optim/verbatim_orchestration/reviews/streaming_divergence_pipeline.json"
)
CFG = "/tmp/claude-1001/Speech/examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml"
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
BATCH = int(os.environ.get("BATCH", "32"))
N_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
ARM = os.environ.get("ARM", "both")  # uncontrolled | controlled | both

from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

cfg = OmegaConf.load(CFG)
cfg.asr.model_name = MODEL
cfg.asr.device_id = 0  # CUDA_VISIBLE_DEVICES pins the physical card
cfg.asr.compute_dtype = os.environ.get("DTYPE", "bfloat16")
cfg.asr.use_cuda_graphs = False
cfg.asr.decoding.greedy.enable_per_stream_biasing = False
cfg.streaming.batch_size = BATCH
cfg.streaming.num_slots = max(256, BATCH * 4)
cfg.streaming.att_context_size = [70, 13]
cfg.enable_itn = False
cfg.enable_nmt = False
cfg.calculate_wer = False
cfg.calculate_bleu = False

print(f"[model] building CacheAwareRNNTPipeline over {MODEL}", flush=True)
MATMUL = os.environ.get("MATMUL", str(cfg.matmul_precision))
PipelineBuilder.set_matmul_precision(MATMUL)
pipeline = PipelineBuilder.build_pipeline(cfg)
CHUNK = int(round(float(pipeline.chunk_size_in_secs) * int(pipeline.sample_rate)))
print(
    f"[model] chunk {pipeline.chunk_size_in_secs}s = {CHUNK} samples, "
    f"sample_rate {pipeline.sample_rate}, num_slots {pipeline.num_slots}",
    flush=True,
)

_next_stream_id = [1]


def stream(audios, pad_to=None):
    """Run a list of float32 arrays as one batch through the pipeline; return final transcripts.

    Every stream gets a fresh id, so NeMo allocates and frees its slots exactly as it would for
    real sessions. A stream's last frame is zero-filled to a whole chunk and carries `length` =
    its real sample count, which is what makes `right_padding` non-zero for that row and only
    that row -- the mechanism the whole question is about.
    """
    rows = []
    for a in audios:
        if pad_to is not None and len(a) < pad_to:
            padded = np.zeros(pad_to, dtype=np.float32)
            padded[: len(a)] = a
            a = padded
        rows.append(np.asarray(a, dtype=np.float32))

    ids = []
    for _ in rows:
        ids.append(_next_stream_id[0])
        _next_stream_id[0] += 1

    steps = max(int(math.ceil(len(a) / CHUNK)) for a in rows)
    finals = {sid: [] for sid in ids}
    options = ASRRequestOptions()

    for step in range(steps):
        frames = []
        for sid, a in zip(ids, rows):
            total = int(math.ceil(len(a) / CHUNK))
            if step >= total:
                continue  # this stream already sent its last frame
            start = step * CHUNK
            piece = a[start : start + CHUNK]
            valid = len(piece)
            if valid < CHUNK:
                block = np.zeros(CHUNK, dtype=np.float32)
                block[:valid] = piece
                piece = block
            frames.append(
                Frame(
                    samples=torch.from_numpy(np.ascontiguousarray(piece)),
                    stream_id=sid,
                    is_first=(step == 0),
                    is_last=(step == total - 1),
                    length=valid,
                    options=options if step == 0 else None,
                )
            )
        if not frames:
            break
        with torch.inference_mode():
            outputs = pipeline.transcribe_step(frames)
        for out in outputs:
            text = str(out.final_transcript or "").strip()
            if text:
                finals[int(out.stream_id)].append(text)

    return [" ".join(finals[sid]).strip() for sid in ids]


def pcm_of(rec):
    a = rec["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == 16000, sr
    return np.asarray(data, dtype=np.float32)


print("[data] streaming LibriSpeech test-other", flush=True)
ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
pool, meta = [], []
for rec in ds:
    pool.append(pcm_of(rec))
    meta.append({"id": rec["id"], "reference": rec["text"].lower()})
    if len(pool) >= N_TARGETS + BATCH:
        break
print(f"[data] {len(pool)} utterances held", flush=True)

# --- degeneracy guard -------------------------------------------------------------------
# A driver that transcribes nothing reports zero divergences and looks like a clean result.
# This project has already shipped one degenerate experiment whose two arms could not differ,
# so the driver proves it is transcribing before any count is recorded.
print("\n[guard] checking the driver actually transcribes", flush=True)
_probe = stream([pool[0], pool[1]])
for _i, _text in enumerate(_probe):
    print(f"  ref  {_i}: {meta[_i]['reference'][:90]}", flush=True)
    print(f"  hyp  {_i}: {_text[:90]}", flush=True)
if not all(t.strip() for t in _probe):
    raise SystemExit(
        "[guard] FAILED: the pipeline returned an empty transcript; counts would be meaningless"
    )
_ref0 = set(meta[0]["reference"].split())
_hyp0 = set(_probe[0].lower().split())
_overlap = len(_ref0 & _hyp0) / max(1, len(_ref0))
print(f"[guard] word overlap with the reference on utterance 0: {_overlap:.2f}", flush=True)
if _overlap < 0.5:
    raise SystemExit(
        f"[guard] FAILED: overlap {_overlap:.2f} means the driver is not decoding correctly"
    )
print("[guard] passed\n", flush=True)

arms = ["uncontrolled", "controlled"] if ARM == "both" else [ARM]
state = {
    "model": MODEL,
    "machine": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "compute_dtype": str(cfg.asr.compute_dtype),
    "matmul_precision": MATMUL,
    "path": "nemo.collections.asr.inference CacheAwareRNNTPipeline.transcribe_step",
    "att_context_size": list(cfg.streaming.att_context_size),
    "batch": BATCH,
    "chunk_samples": CHUNK,
    "note": "the SERVER path, not the example path: per-stream right_paddings, finals in their own sub-batch",
    "arms": {},
}
t0 = time.time()

for arm in arms:
    controlled = arm == "controlled"
    res = {
        "control": (
            "every row zero-padded to one common length in BOTH arms, so every stream's is_last "
            "lands on the same step with the same right_padding; batch size is the only variable"
            if controlled
            else "rows keep their own lengths, so a short stream's is_last lands early and carries "
            "a real right_padding while its neighbours keep streaming"
        ),
        "checked": 0,
        "divergences": [],
    }
    print(f"\n=== arm {arm} ===", flush=True)
    for i in range(N_TARGETS):
        target = pool[i]
        neighbours = [pool[(i + 1 + k) % len(pool)] for k in range(BATCH - 1)]
        common = max(len(a) for a in [target] + neighbours) if controlled else None
        alone = stream([target], pad_to=common)[0]
        batched = stream([target] + neighbours, pad_to=common)[0]
        res["checked"] += 1
        if alone != batched:
            d = {
                "n": i,
                "librispeech_id": meta[i]["id"],
                "reference": meta[i]["reference"],
                "alone": alone,
                "in_batch": batched,
            }
            res["divergences"].append(d)
            print(f"  *** DIVERGENCE #{len(res['divergences'])} id={meta[i]['id']}", flush=True)
            print(f"      ref     : {d['reference']}", flush=True)
            print(f"      alone   : {alone}", flush=True)
            print(f"      batch{BATCH} : {batched}", flush=True)
        if (i + 1) % 8 == 0:
            print(
                f"  {i + 1}/{N_TARGETS} checked, {len(res['divergences'])} divergent, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
            state["arms"][arm] = res
            json.dump(state, open(OUT, "w"), indent=1)
    state["arms"][arm] = res
    json.dump(state, open(OUT, "w"), indent=1)
    print(f"=== arm {arm}: {len(res['divergences'])} in {res['checked']} ===", flush=True)

state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)
for arm, res in state["arms"].items():
    print(f"FINAL {arm}: {len(res['divergences'])} divergences in {res['checked']}", flush=True)

# NeMo's decoder leaves a non-Python thread that trips PyGILState_Release during interpreter
# finalization; the results are already on disk, so leave without running finalizers.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
