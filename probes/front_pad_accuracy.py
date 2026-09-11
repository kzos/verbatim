# SPDX-License-Identifier: Apache-2.0
"""Does leading silence change what the cache-aware model recognises?

The tick-phase fix aligns a session's chunk grid by emitting a first chunk of
``[pad(n-r) | real(r)]``, so the model meets up to one chunk period of silence
before the first real sample, against a cold cache rather than a warmed one.
That is an acoustic question, not an accounting one, and it needs the GPU.

Streams the same utterance repeatedly with a leading pad stepped across one
period, asks for word timings, and compares each run's transcript and timings
against the unpadded run with the pad subtracted from every timestamp.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

import websockets

from verbatim_bench.corpus import load_manifest, read_pcm16

SAMPLE_RATE = 16000


async def one(endpoint: str, pcm: bytes, chunk_ms: int, pad_ms: int) -> dict[str, object]:
    pad = b"\x00\x00" * int(SAMPLE_RATE * pad_ms / 1000)
    stream = pad + pcm
    query = urlencode({"chunk_ms": chunk_ms, "lang": "en-US", "words": "1"})
    frame_bytes = 2 * int(SAMPLE_RATE * chunk_ms / 1000)
    texts: list[str] = []
    words: list[dict[str, object]] = []

    async def read(ws) -> None:
        """Collect every final, not the first.

        Endpointing segments a stream into several utterances, and a leading pad
        moves where the segmentation falls.  Reading one final measures where the
        server happened to split, which is why the first version of this probe
        reported transcripts changing when the recognition had not.
        """
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
                words.extend(frame.get("words", []) or [])

    async with websockets.connect(f"{endpoint}?{query}", max_size=None) as ws:
        await ws.recv()  # the session frame
        reader = asyncio.ensure_future(read(ws))
        for i in range(0, len(stream), frame_bytes):
            await ws.send(stream[i : i + frame_bytes])
            await asyncio.sleep(chunk_ms / 1000.0)
        await ws.send(json.dumps({"type": "end"}))
        try:
            await asyncio.wait_for(reader, timeout=30.0)
        except TimeoutError:
            reader.cancel()
    return {
        "pad_ms": pad_ms,
        "text": " ".join(t for t in texts if t),
        "words": words,
        "finals": len(texts),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--chunk-ms", type=int, default=160)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--utterances", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pool = sorted(load_manifest(args.manifest), key=lambda u: -u.duration_s)[: args.utterances]
    step = args.chunk_ms // args.steps
    report: list[dict[str, object]] = []
    text_mismatch = 0
    controls_differ = 0
    timing_rows: list[float] = []

    for utterance in pool:
        pcm = read_pcm16(utterance.audio_path)
        runs = []
        for i in range(args.steps):
            runs.append(await one(args.endpoint, pcm, args.chunk_ms, i * step))
        # The negative control, and the reason it is not optional: this server is
        # bfloat16, the precision measured to lose batch invariance, so the same audio
        # is not guaranteed to give the same transcript twice.  Without a pad-zero
        # repeat, "the padded run differs" says nothing about padding.
        control = await one(args.endpoint, pcm, args.chunk_ms, 0)
        base = runs[0]
        control_same = base["text"].split() == control["text"].split()
        print(
            f"{utterance.stream_id}  CONTROL pad 0 repeated: "
            f"text {'SAME' if control_same else 'DIFFERENT'}  "
            f"tokens {len(base['text'].split())} vs {len(control['text'].split())}  "
            f"finals {base['finals']} vs {control['finals']}"
        )
        report.append(
            {
                "stream_id": utterance.stream_id,
                "pad_ms": 0,
                "control": True,
                "same_text": control_same,
                "run_finals": control["finals"],
                "base_finals": base["finals"],
            }
        )
        if not control_same:
            controls_differ += 1
        for run in runs[1:]:
            # Compare the token sequence, not the joined string.  Endpointing splits
            # the stream into a different number of segments when the pad moves, so
            # the joined text differs in whitespace alone while the recognition is
            # identical.  Words are what the transcript is made of.
            base_tokens = base["text"].split()
            run_tokens = run["text"].split()
            same_text = base_tokens == run_tokens
            if not same_text:
                text_mismatch += 1
            drift = None
            bw, rw = base["words"], run["words"]
            if same_text and bw and rw and len(bw) == len(rw):
                # Every timestamp should move by exactly the pad and no more.
                deltas = [(w["s"] - b["s"]) - run["pad_ms"] for b, w in zip(bw, rw, strict=True)]
                drift = max(abs(d) for d in deltas)
                timing_rows.append(drift)
            report.append(
                {
                    "stream_id": utterance.stream_id,
                    "pad_ms": run["pad_ms"],
                    "same_text": same_text,
                    "base_words": len(bw),
                    "run_words": len(rw),
                    "base_finals": base["finals"],
                    "run_finals": run["finals"],
                    "max_timing_drift_ms": drift,
                    "text": run["text"] if not same_text else None,
                    "base_text": base["text"] if not same_text else None,
                    "first_difference": (
                        next(
                            (
                                f"#{i}: {b!r} vs {r!r}"
                                for i, (b, r) in enumerate(zip(base_tokens, run_tokens))
                                if b != r
                            ),
                            f"lengths {len(base_tokens)} vs {len(run_tokens)}",
                        )
                        if not same_text
                        else None
                    ),
                }
            )
            print(
                f"{utterance.stream_id}  pad {run['pad_ms']:3d} ms  "
                f"text {'SAME' if same_text else 'DIFFERENT'}  "
                f"tokens {len(base_tokens)} vs {len(run_tokens)}  "
                f"finals {base['finals']} vs {run['finals']}  "
                f"max timing drift {drift if drift is None else f'{drift:.0f} ms'}"
            )

    doc = {
        "endpoint": args.endpoint,
        "chunk_ms": args.chunk_ms,
        "pad_step_ms": step,
        "utterances": [u.stream_id for u in pool],
        "comparisons": len(report),
        "text_mismatches": text_mismatch,
        "controls_that_differed": controls_differ,
        "controls_run": len(pool),
        "max_timing_drift_ms": max(timing_rows) if timing_rows else None,
        "rows": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    worst = doc["max_timing_drift_ms"]
    worst_text = "no word timings compared" if worst is None else f"{worst:.0f} ms"
    print(
        f"\n{len(report)} comparisons, {text_mismatch} padded transcripts differed, "
        f"{controls_differ} of {len(pool)} unpadded repeats differed, "
        f"worst timing drift beyond the pad: {worst_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
