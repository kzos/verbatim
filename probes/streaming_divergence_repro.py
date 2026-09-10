# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Does batch composition change the transcript on the CACHE-AWARE STREAMING path?

Everything measured so far used NeMo's offline transcribe(). That path does not exercise the streaming
padding arithmetic that NVIDIA-NeMo/Speech issue #12840 is about, so the evidence to date is not a
reproducer for that issue. This script closes that gap: it drives conformer_stream_step chunk by chunk,
exactly as the shipped cache-aware streaming example does, and compares one utterance's final transcript
when it streams ALONE against the same utterance streaming inside a batch.

Arm A: the target utterance alone, batch of 1.
Arm B: the same utterance in slot 0 of a batch of BATCH, with unrelated neighbours.
Everything else is held constant: same model, same weights, same att_context_size, same chunk schedule.

If the transcripts differ, batch composition reached the transcript on the streaming path.
"""
import io
import os, json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import soundfile as sf
from datasets import load_dataset, Audio

os.makedirs("probe-output", exist_ok=True)
OUT = os.environ.get("PROBE_OUT", "probe-output/streaming_divergence.json")
MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
BATCH = 32
N_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
dev = torch.device("cuda:0")

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

print(f"[model] loading {MODEL}", flush=True)
m = ASRModel.from_pretrained(MODEL, map_location="cpu").to(dev).eval()
# The multi-lookahead checkpoint ships several context sizes; pin one so it is not a hidden variable.
att = m.encoder.att_context_size
print(f"[model] att_context_size {att}", flush=True)
m.encoder.setup_streaming_params()
print(f"[model] streaming_cfg {m.encoder.streaming_cfg}", flush=True)


def texts_of(hyps):
    return [h.text if isinstance(h, Hypothesis) else h for h in hyps]


def drop_extra(step_num):
    return 0 if step_num == 0 else m.encoder.streaming_cfg.drop_extra_pre_encoded


def stream(audios):
    """Stream a list of float32 arrays together as one batch; return their final transcripts."""
    buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
    for a in audios:
        buf.append_audio(a, stream_id=-1)
    bs = len(buf.streams_length)
    cache_ch, cache_t, cache_len = m.encoder.get_initial_cache_state(batch_size=bs)
    prev_hyp, pred_out = None, None
    transcribed = None
    for step, (chunk_audio, chunk_lengths) in enumerate(iter(buf)):
        with torch.inference_mode(), torch.no_grad():
            (pred_out, transcribed, cache_ch, cache_t, cache_len, prev_hyp) = m.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_ch,
                cache_last_time=cache_t,
                cache_last_channel_len=cache_len,
                keep_all_outputs=buf.is_buffer_empty(),
                previous_hypotheses=prev_hyp,
                previous_pred_out=pred_out,
                drop_extra_pre_encoded=drop_extra(step),
                return_transcription=True,
            )
    return texts_of(transcribed)


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

state = {"model": MODEL, "path": "cache-aware streaming, conformer_stream_step",
         "machine": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "nemo": __import__("nemo").__version__,
         "att_context_size": list(att) if not isinstance(att, int) else att,
         "batch": BATCH, "checked": 0, "divergences": []}
t0 = time.time()

for i in range(N_TARGETS):
    target = pool[i]
    alone = stream([target])[0]
    neighbours = [pool[(i + 1 + k) % len(pool)] for k in range(BATCH - 1)]
    batched = stream([target] + neighbours)[0]
    state["checked"] += 1
    if alone != batched:
        d = {"n": i, "librispeech_id": meta[i]["id"], "reference": meta[i]["reference"],
             "alone": alone, "in_batch": batched}
        state["divergences"].append(d)
        print(f"  *** STREAMING DIVERGENCE #{len(state['divergences'])} id={meta[i]['id']}", flush=True)
        print(f"      ref     : {d['reference']}", flush=True)
        print(f"      alone   : {alone}", flush=True)
        print(f"      batch{BATCH} : {batched}", flush=True)
    if (i + 1) % 16 == 0:
        print(f"  {i+1}/{N_TARGETS} checked, {len(state['divergences'])} divergent, {time.time()-t0:.0f}s", flush=True)
        json.dump(state, open(OUT, "w"), indent=1)

state["seconds"] = round(time.time() - t0, 1)
json.dump(state, open(OUT, "w"), indent=1)
print(f"\n=== {len(state['divergences'])} streaming divergences in {state['checked']} utterances "
      f"(batch 1 vs batch {BATCH}), {state['seconds']}s ===", flush=True)
