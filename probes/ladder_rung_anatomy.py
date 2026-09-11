# SPDX-License-Identifier: Apache-2.0
"""Reproduce one ladder rung exactly and dump the anatomy of its p95.

The ladder's rung executor (`_make_rung` in the bench CLI) built its `LoadSpec`
with `ramp_s=0.0` and no `window_s`, so the frozen 180-second window and the
60-second warm-up never reached a measurement while every rung was still stamped
`canonical_window: true`.  This script reproduces that spec literally rather
than approximating it, and takes `--window-s` so the same rung can be run with
the window applied and the two compared.  It is kept as the record of the run
that found the defect; the executor itself was fixed on 2026-09-11 by DR-0005
and no longer behaves like this.

Run it against a live server with the bench package importable:

    PYTHONPATH=bench/src python probes/ladder_rung_anatomy.py \
        --endpoint ws://127.0.0.1:8081/v1/stream \
        --manifest <corpus>.jsonl --sessions 6 --seed 20260915 \
        --out rung.json

Measured on 2026-09-11 against an A6000 bfloat16 eager server: without the
window a six-stream rung is 15.1 s and 106 samples, and four repeats of one
identical rung give p95 values of 289.8, 320.3, 302.6 and 301.8 ms against the
310 ms threshold.  With the window it is 188.6 s and 4,798 samples.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from verbatim_bench.client import ChunkMode
from verbatim_bench.pace import LoadSpec, run_load
from verbatim_bench.results import percentile


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--window-s", type=float, default=None)
    ap.add_argument("--ramp-s", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = LoadSpec(
        endpoint=args.endpoint,
        manifest=args.manifest,
        sessions=args.sessions,
        chunk=ChunkMode.parse(160),
        profile="uniform",
        seed=args.seed,
        ramp_s=args.ramp_s,
        window_s=args.window_s,
    )
    result = await run_load(spec)

    per_session = [list(s.partial_ms) for s in result.sessions]
    flat = [v for row in per_session for v in row]
    if not flat:
        print("no samples")
        return 1
    p95 = percentile(flat, 95)
    # Where do the samples at or above the p95 sit inside their own session?
    above = [
        {"session": i, "index": j, "of": len(row), "ms": v}
        for i, row in enumerate(per_session)
        for j, v in enumerate(row)
        if v >= p95
    ]
    first_three = [v for row in per_session for v in row[:3]]
    rest = [v for row in per_session for v in row[3:]]
    doc = {
        "sessions": args.sessions,
        "seed": args.seed,
        "window_s": args.window_s,
        "ramp_s": args.ramp_s,
        "wall_clock_s": result.wall_clock_s,
        "session_count": len(per_session),
        "samples_total": len(flat),
        "samples_per_session": [len(r) for r in per_session],
        "p95_ms": p95,
        "p50_ms": percentile(flat, 50),
        "max_ms": max(flat),
        "at_or_above_p95": above,
        "at_or_above_p95_count": len(above),
        "first_three_chunks": {
            "n": len(first_three),
            "mean_ms": statistics.fmean(first_three),
            "max_ms": max(first_three),
        },
        "after_three_chunks": {
            "n": len(rest),
            "mean_ms": statistics.fmean(rest) if rest else None,
            "max_ms": max(rest) if rest else None,
            "p95_ms": percentile(rest, 95) if rest else None,
        },
        "per_session_partial_ms": per_session,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(
        f"n={args.sessions} seed={args.seed} window={args.window_s} "
        f"sessions_run={len(per_session)} samples={len(flat)} "
        f"p50={doc['p50_ms']:.1f} p95={p95:.1f} max={doc['max_ms']:.1f} "
        f"wall={result.wall_clock_s:.1f}s"
    )
    print(
        f"  first three chunks of each session: mean {doc['first_three_chunks']['mean_ms']:.1f} ms, "
        f"max {doc['first_three_chunks']['max_ms']:.1f} ms"
    )
    a = doc["after_three_chunks"]
    if a["mean_ms"] is not None:
        print(f"  everything after:                   mean {a['mean_ms']:.1f} ms, p95 {a['p95_ms']:.1f} ms")
    early = sum(1 for x in above if x["index"] < 3)
    print(f"  samples at or above p95: {len(above)}, of which {early} are in a session's first three chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
