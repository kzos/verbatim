# SPDX-License-Identifier: Apache-2.0
"""Does the resampler's output recognise the same as native-rate audio?

The G.711 and resampling path decodes mu-law and resamples to 16 kHz in a worker
thread, and its unit tests compare it against itself: sample counts, chunked
output against one-shot output, and a level check.  None of them asks whether the
model recognises the result, which is the only question that matters to a caller
sending telephone audio.

Takes each utterance at 16 kHz PCM16 as the reference arm, encodes the same audio
to 8 kHz mu-law for the test arm, streams both, and compares the transcripts to
each other and to the corpus reference.  Run it at float32, where the baseline is
near-deterministic, or a precision difference will be read as a resampler one.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urlencode

import websockets

from verbatim_bench.corpus import load_manifest, read_pcm16

SAMPLE_RATE = 16000
_BIAS = 0x84
_CLIP = 32635


def _linear_to_mulaw(sample: int) -> int:
    """The standard G.711 mu-law encoder, so the test arm is real telephone audio."""
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    sample = min(sample, _CLIP) + _BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not sample & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _downsample_to_8k(pcm: bytes) -> bytes:
    """Decimate by two with a short symmetric filter, so the test arm is band-limited
    telephone audio rather than an aliased mess that would flatter nobody."""
    samples = array.array("h")
    samples.frombytes(pcm)
    taps = (1, 3, 6, 3, 1)
    total = sum(taps)
    out = array.array("h")
    for i in range(0, len(samples) - len(taps), 2):
        acc = sum(t * samples[i + j] for j, t in enumerate(taps)) // total
        out.append(max(-32768, min(32767, acc)))
    return out.tobytes()


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def _wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1])
            )
    return d[len(r)][len(h)] / max(1, len(r))


async def _stream(endpoint: str, payload: bytes, chunk_ms: int, encoding: str, rate: int) -> str:
    query = urlencode(
        {
            "chunk_ms": chunk_ms,
            "lang": "en-US",
            "words": "0",
            "encoding": encoding,
            "sample_rate_hz": rate,
        }
    )
    width = 1 if encoding == "MULAW" else 2
    frame_bytes = width * int(rate * chunk_ms / 1000)
    texts: list[str] = []

    async def read(ws) -> None:
        while True:
            try:
                raw = await ws.recv()
            except websockets.ConnectionClosed:
                return
            if isinstance(raw, bytes):
                continue
            frame = json.loads(raw)
            if frame.get("type") == "final":
                texts.append(frame.get("text", ""))
            elif frame.get("type") == "error":
                texts.append(f"<<ERROR {frame.get('code')}: {frame.get('message')}>>")
                return

    async with websockets.connect(f"{endpoint}?{query}", max_size=None) as ws:
        await ws.recv()
        reader = asyncio.ensure_future(read(ws))
        for i in range(0, len(payload), frame_bytes):
            await ws.send(payload[i : i + frame_bytes])
            await asyncio.sleep(chunk_ms / 1000.0)
        await ws.send(json.dumps({"type": "end"}))
        try:
            await asyncio.wait_for(reader, timeout=30.0)
        except TimeoutError:
            reader.cancel()
    return " ".join(t for t in texts if t)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--chunk-ms", type=int, default=160)
    ap.add_argument("--utterances", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pool = sorted(load_manifest(args.manifest), key=lambda u: -u.duration_s)[: args.utterances]
    rows = []
    for utterance in pool:
        pcm = read_pcm16(utterance.audio_path)
        native = await _stream(args.endpoint, pcm, args.chunk_ms, "LINEAR_PCM", SAMPLE_RATE)
        narrow = _downsample_to_8k(pcm)
        samples = array.array("h")
        samples.frombytes(narrow)
        mulaw = bytes(_linear_to_mulaw(s) for s in samples)
        phone = await _stream(args.endpoint, mulaw, args.chunk_ms, "MULAW", 8000)
        row = {
            "stream_id": utterance.stream_id,
            "duration_s": utterance.duration_s,
            "native_wer": _wer(utterance.text, native),
            "phone_wer": _wer(utterance.text, phone),
            "native_vs_phone_wer": _wer(native, phone),
            "same_text": _norm(native) == _norm(phone),
            "native_text": native,
            "phone_text": phone,
        }
        rows.append(row)
        print(
            f"{utterance.stream_id:18s} {utterance.duration_s:5.1f}s  "
            f"native WER {row['native_wer']:.4f}  8k mu-law WER {row['phone_wer']:.4f}  "
            f"delta {row['phone_wer'] - row['native_wer']:+.4f}  "
            f"{'identical' if row['same_text'] else 'differs'}"
        )

    ok = [r for r in rows if "<<ERROR" not in r["phone_text"]]
    doc = {
        "endpoint": args.endpoint,
        "chunk_ms": args.chunk_ms,
        "utterances": len(rows),
        "errored": len(rows) - len(ok),
        "mean_native_wer": (sum(r["native_wer"] for r in ok) / len(ok)) if ok else None,
        "mean_phone_wer": (sum(r["phone_wer"] for r in ok) / len(ok)) if ok else None,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    if ok:
        print(
            f"\n{len(ok)} utterances: mean WER {doc['mean_native_wer']:.4f} at 16 kHz PCM, "
            f"{doc['mean_phone_wer']:.4f} at 8 kHz mu-law, "
            f"delta {doc['mean_phone_wer'] - doc['mean_native_wer']:+.4f}"
        )
    if doc["errored"]:
        print(f"{doc['errored']} utterances errored on the mu-law arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
