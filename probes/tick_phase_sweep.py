# SPDX-License-Identifier: Apache-2.0
"""Is a session's latency a constant fixed by when it connected, relative to the tick?

A windowed load run shows each session's watermark latency steady to a few
milliseconds for its whole life, while sessions differ from one another by up to
140 ms.  If that is a phase offset between the session's chunk cadence and the
server's tick, then stepping the start delay across one tick period should trace a
sawtooth, and the wrap should move when the same sweep is run again at a different
moment.

Both hold.  Measured against a live A6000 bfloat16 server on 2026-09-11, sixteen
sessions streaming one 33.7 s utterance with start delays stepping 10 ms:

    delay   0 ms -> 219.9 ms      delay  80 ms -> 301.0 ms
    delay  50 ms -> 168.8 ms      delay 150 ms -> 230.7 ms
    delay  60 ms -> 319.2 ms   (the wrap)

Slope exactly minus one, amplitude 160 ms, range 168.8 to 321.6 ms, and the wrap
moved from 55 ms to 95 ms when the sweep was repeated.  So

    latency = chunk_ms + ((tick phase - connect time) mod tick period)

drawn once at connect and held for the session's life, with no recovery: the
client's chunk cadence and the server's tick are the same period and stay in
lockstep.  Against a 310 ms budget and a 95th-percentile criterion, clients
arriving at arbitrary times give a p95 near 320 ms at any concurrency, including
one stream.

Time to the first word rides the same sawtooth: measured across every phase it is
the steady-state latency plus the 800 ms endpointing window, constant to within a
millisecond.  The first word carries the same phase penalty as every later chunk,
so there is no trade between the two latencies to make.

Run it against any live server; it takes about twenty seconds:

    PYTHONPATH=bench/src python probes/tick_phase_sweep.py \
        --endpoint ws://127.0.0.1:8081/v1/stream \
        --manifest <corpus>.jsonl --out sweep.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from verbatim_bench.client import ChunkMode, run_session
from verbatim_bench.corpus import load_manifest, read_pcm16


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--period-ms", type=int, default=160)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--utterance", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    utterances = load_manifest(args.manifest)
    # One utterance long enough to give many chunks, so a median means something.
    pool = sorted(utterances, key=lambda u: -u.duration_s)
    utterance = pool[args.utterance]
    pcm = read_pcm16(utterance.audio_path)
    chunk = ChunkMode.parse(args.period_ms)
    step_ms = args.period_ms / args.steps

    async def one(i: int) -> dict[str, object]:
        delay = i * step_ms / 1000.0
        result = await run_session(
            args.endpoint,
            session_id=f"phase{i:02d}",
            utterance=utterance,
            pcm=pcm,
            chunk=chunk,
            start_delay_s=delay,
            words=False,
            lang="en-US",
            frame_ms=20,
            frame_seed=20260915 + i,
        )
        samples = list(result.partial_ms)
        return {
            "step": i,
            "start_delay_ms": i * step_ms,
            "error": result.error,
            # Time from the first audio sent to the first partial carrying any text.
            # The phase fix is expected to improve the steady-state percentile and can
            # regress this, so both arms need it measured rather than assumed.
            "first_word_ms": result.first_partial_ms,
            "n": len(samples),
            "median_ms": statistics.median(samples) if samples else None,
            "min_ms": min(samples) if samples else None,
            "max_ms": max(samples) if samples else None,
            "spread_ms": (max(samples) - min(samples)) if samples else None,
        }

    rows = list(await asyncio.gather(*(one(i) for i in range(args.steps))))
    doc = {
        "endpoint": args.endpoint,
        "period_ms": args.period_ms,
        "steps": args.steps,
        "utterance": utterance.stream_id,
        "duration_s": utterance.duration_s,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(
        f"utterance {utterance.stream_id}, {utterance.duration_s:.1f} s, period {args.period_ms} ms"
    )
    print(" delay(ms)  n   median   min     max    spread   first word")
    for r in rows:
        if r["median_ms"] is None:
            print(f"  {r['start_delay_ms']:7.1f}  --  error: {r['error']}")
            continue
        fw = r["first_word_ms"]
        print(
            f"  {r['start_delay_ms']:7.1f} {r['n']:3d}  {r['median_ms']:7.1f} "
            f"{r['min_ms']:7.1f} {r['max_ms']:7.1f}   {r['spread_ms']:6.1f}"
            f"   {'--' if fw is None else f'{fw:8.1f}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
