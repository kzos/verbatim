"""Where a final's latency actually goes: grid wait, steady step, edge batches, or the rest.

DR-0017 closed the capacity question and left exactly one thing open, in its own words:

    At bucket 256 a step costs 56 ms, still well inside the budget, yet that arm failed on
    *latency* at 46 to 47 streams with p95 319 ms. The step alone does not account for that.
    That gap between step cost and observed latency is the next thing to profile, and
    nothing here should be read as explaining it.

`step_phase_split.py` measures the steady step and nothing else. The ladder measures the
end-to-end final latency and nothing else. Neither can see the space between them, and every
account of that space so far has been arithmetic -- which is how this project published a
per-row cost that was wrong by six times.

What this measures. A final's latency, as the gate defines it, is the time from the last
sample a client sent to the moment that session's terminal row is published. The tick loop
already knows every term:

    L = wait + overhead + step + edge + rest

    wait     the last sample landed mid-period, so it waits for the next tick boundary.
             This is DR-0012's phase offset plus the grid itself, and it is irreducible:
             you cannot step a chunk before the client has finished sending it.
    overhead the collect phase, the stamping phase, and any late start inherited from the
             previous tick -- `TickStats.lateness_ms` minus the two step costs.
    step     the steady batch, padded to the bucket. What step_phase_split measures.
    edge     every edge batch in this tick, run SERIALLY after the steady step. A final is
             produced by the edge peel, so it waits behind all of them. Edge batches are
             padded to `edge_batch` (8) and NOT to the bucket -- but the encoder is fixed
             cost (DR-0017: 14.5 ms at 32 rows, 14.6 at 256), so an 8-row edge batch is not
             eight rows' worth of cheap. That is the term arithmetic keeps missing.
    rest     publish: queueing the rows and waking the loop.

Nothing here is derived. `wait` comes from the tick loop's own recorded boundaries, `step`,
`edge` and `overhead` from the `TickStats` the server writes for itself, and `L` from the two
timestamps that bracket it. `rest` is the residual, which is the only honest place to put the
time nothing claimed.

Run it on the machine under test:

    PYTHONPATH=src:bench/src python probes/latency_decomposition.py \
        --bucket 256 --streams 46 --manifest <corpus.jsonl> --out <row.json>

It loads the checkpoint and serves real sessions in process, so it needs a GPU and the model
runtime. The transport is not in the path, and the feeders share a GIL with the tick thread:
the first omits time a real client pays and the second adds time it does not, so read the
SHARES rather than treating the total as the ladder's number. Nothing here is part of the
server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Streamed:
    """One session that ran to a terminal row, with the two timestamps that bracket it."""

    stream_id: int
    utterance: str
    t_last_sample: float
    chunks: int
    tick_id: int | None = None
    t_publish: float | None = None


@dataclass
class Arm:
    bucket: int
    streams: int
    sessions: list[Streamed] = field(default_factory=list)
    refused: int = 0


def _pct(values: list[float], q: float) -> float | None:
    """The q-th percentile by nearest rank, which is what the gate uses."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": _pct(values, 50),
        "p95": _pct(values, 95),
        "p99": _pct(values, 99),
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


async def run_arm(
    *,
    bucket: int,
    streams: int,
    window_s: float,
    manifest: Path,
    args: argparse.Namespace,
) -> tuple[Arm, list[dict[str, Any]], dict[int, float]]:
    from verbatim_bench.corpus import load_manifest, read_pcm16

    from verbatim.config import ChunkMode, EngineConfig
    from verbatim.engine import Engine
    from verbatim.pipelines import registry
    from verbatim.pipelines.cache_aware_rnnt import NeMoBoundary
    from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, build_pipeline
    from verbatim.protocols.base import SessionOptions
    from verbatim.core.errors import ResourceExhausted

    chunk = ChunkMode(args.chunk_ms)
    config = EngineConfig(
        chunk=chunk,
        buckets=(bucket,),
        pipeline=args.pipeline,
        padding="fixed",
        stop_history_eou_ms=args.stop_history_eou_ms,
        idle_timeout_s=None,
    )
    if args.pipeline == "fake":
        # For proving the instrument, not for measuring a server: the fake's step costs
        # nothing, so every term but `wait` collapses and a real reading is impossible.
        # What it does prove is that the join, the decomposition and the residual are
        # right, which is not something a GPU run would tell you.
        adapter = registry.build_for(config)
    else:
        spec = NeMoPipelineSpec(
            model=args.model,
            chunk=chunk,
            att_context=(args.att_context_left, chunk.ms // 80 - 1),
            num_slots=config.num_slots,
            batch_size=bucket,
            decoding="rnnt",
            stop_history_eou_ms=args.stop_history_eou_ms,
            use_cuda_graphs=False,
            compute_dtype=args.compute_dtype,
            device_id=args.device_id,
        )
        # The same construction the CLI uses, so this is the server's adapter and not a
        # reconstruction of it.
        boundary = NeMoBoundary.from_pipeline(build_pipeline(spec), graph_step_available=False)
        adapter = registry.build_for(config, boundary=boundary, use_cuda_graphs=False)

    engine = Engine(config, adapter, loop=asyncio.get_running_loop())

    arm = Arm(bucket=bucket, streams=streams)
    publish_of: dict[int, tuple[int, float]] = {}
    original_publish = engine._publish

    def stamping_publish(results: list[Any]) -> None:
        """Observe the terminal row at the seam the server publishes it, then hand the
        tick on untouched. The timestamp is taken BEFORE the real publish so it cannot
        be inflated by this probe's own bookkeeping."""
        now = time.monotonic()
        for result in results:
            if result.is_last and result.stream_id >= 0:
                publish_of.setdefault(result.stream_id, (result.tick_id, now))
        original_publish(results)

    engine._publish = stamping_publish  # type: ignore[method-assign]

    utterances = list(load_manifest(manifest))
    if not utterances:
        raise SystemExit(f"{manifest} holds no utterances")
    period = chunk.period_s
    frame_bytes = chunk.samples * 2
    rng = random.Random(args.seed)

    await engine.start()
    started = time.monotonic()
    deadline = started + window_s + args.warm_up_s

    # The tick loop retains only the most recent STATS_RETAINED ticks, which at a 160 ms
    # period is about 164 seconds -- shorter than any window worth measuring. Reading
    # `stats` once at the end would silently lose every tick before that, and the sessions
    # that ran in them would drop out of the join. So drain it as it goes, keyed by the
    # tick's own id and never by its position in the deque.
    collected: dict[int, Any] = {}

    async def collector() -> None:
        while time.monotonic() < deadline + 5.0:
            for stats in engine._tick.stats:
                collected.setdefault(stats.tick_id, stats)
            await asyncio.sleep(period * 100)

    collector_task = asyncio.create_task(collector())

    async def worker(index: int) -> None:
        cursor = index
        while time.monotonic() < deadline:
            # DR-0012: the phase is drawn once per connect and paid for the session's life.
            await asyncio.sleep(rng.random() * period)
            utterance = utterances[cursor % len(utterances)]
            cursor += streams
            pcm = read_pcm16(utterance.audio_path)
            try:
                handle = engine.open_session(SessionOptions(chunk_ms=chunk.ms))
            except ResourceExhausted:
                arm.refused += 1
                await asyncio.sleep(period)
                continue
            stream_id = handle._session.session_id
            t_start = time.monotonic()
            sent = 0
            t_last = t_start
            for offset in range(0, len(pcm), frame_bytes):
                target = t_start + (sent + 1) * period
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                handle.feed(pcm[offset : offset + frame_bytes])
                t_last = time.monotonic()
                sent += 1
            handle.end()
            record = Streamed(
                stream_id=stream_id,
                utterance=utterance.stream_id,
                t_last_sample=t_last,
                chunks=sent,
            )
            # Wait for this session's terminal row to be published, which is what the
            # gate's final latency ends on.
            waited = 0.0
            while stream_id not in publish_of and waited < args.final_timeout_s:
                await asyncio.sleep(period / 4)
                waited += period / 4
            if stream_id in publish_of:
                record.tick_id, record.t_publish = publish_of[stream_id]
            # Only sessions whose whole life was inside the measurement window count.
            if t_start >= started + args.warm_up_s:
                arm.sessions.append(record)

    workers = [asyncio.create_task(worker(i)) for i in range(streams)]
    await asyncio.gather(*workers)
    collector_task.cancel()
    for stats in engine._tick.stats:
        collected.setdefault(stats.tick_id, stats)

    tick_stats = [
        {
            "tick_id": s.tick_id,
            "steady_rows": s.steady_rows,
            "live_rows": s.live_rows,
            "pad_rows": s.pad_rows,
            "edge_batches": s.edge_batches,
            "step_ms": s.step_ms,
            "edge_ms": s.edge_ms,
            "lateness_ms": s.lateness_ms,
        }
        for s in sorted(collected.values(), key=lambda s: s.tick_id)
    ]
    # A tick's boundary is a fixed multiple of the period since the loop started -- the
    # tick loop's own definition, applied to the tick's own id. The `boundaries` deque
    # is NOT usable for this: it retains only the last STATS_RETAINED entries, so its
    # index stops being the tick id as soon as it wraps.
    tick_start = engine._tick._start
    boundaries = {s["tick_id"]: tick_start + s["tick_id"] * period for s in tick_stats}
    await engine.stop()
    return arm, tick_stats, boundaries


def decompose(
    arm: Arm,
    tick_stats: list[dict[str, Any]],
    boundaries: dict[int, float],
    *,
    period_ms: float,
) -> dict[str, Any]:
    by_tick = {t["tick_id"]: t for t in tick_stats}
    rows: list[dict[str, float]] = []
    unjoined = 0
    # A join that lines a session up with the wrong tick does not fail, it just produces
    # a number. This one can fail, on the one thing that is true by construction: a tick
    # collects the frame at some point between its boundary and its own end, so the end
    # of the collecting tick cannot precede the last sample that tick collected --
    #
    #     wait + lateness >= 0
    #
    # The first version of this guard asserted `wait >= 0` instead and flagged 31 of 1156
    # sessions on the first good run. Those were not a broken join: a tick running 400 ms
    # late collects audio that arrived well after its own boundary, so a negative wait is
    # ordinary and only the SUM is bounded. The guard is kept because the bug it was
    # written for -- reading a tick boundary out of a bounded deque by position, which put
    # the boundary 91 seconds from its tick -- violates this form just as loudly.
    impossible: list[float] = []
    for session in arm.sessions:
        if session.tick_id is None or session.t_publish is None:
            unjoined += 1
            continue
        stats = by_tick.get(session.tick_id)
        boundary = boundaries.get(session.tick_id)
        if stats is None or boundary is None:
            unjoined += 1
            continue
        total = (session.t_publish - session.t_last_sample) * 1000.0
        wait = (boundary - session.t_last_sample) * 1000.0
        lateness = stats["lateness_ms"]
        if wait + lateness < -1.0 or wait > 10.0 * period_ms:
            impossible.append(wait)
            continue
        step = stats["step_ms"]
        edge = stats["edge_ms"]
        rows.append(
            {
                "total_ms": total,
                "wait_ms": wait,
                "overhead_ms": lateness - step - edge,
                "step_ms": step,
                "edge_ms": edge,
                "rest_ms": total - wait - lateness,
                "edge_batches": float(stats["edge_batches"]),
                "live_rows": float(stats["live_rows"]),
            }
        )
    terms = ("total_ms", "wait_ms", "overhead_ms", "step_ms", "edge_ms", "rest_ms")
    out: dict[str, Any] = {
        "bucket": arm.bucket,
        "streams": arm.streams,
        "sessions_measured": len(rows),
        "sessions_unjoined": unjoined,
        "sessions_refused": arm.refused,
        "sessions_impossible_wait": len(impossible),
        "impossible_wait_examples_ms": sorted(impossible)[:5],
        "terms": {term: _summary([r[term] for r in rows]) for term in terms},
        "edge_batches_per_tick": _summary([t["edge_batches"] * 1.0 for t in tick_stats]),
        "ticks": len(tick_stats),
    }
    # The decisive reading: of the p95 latency, how much is the steady step?
    p95 = out["terms"]["total_ms"]["p95"]
    if p95:
        # Attribute at the p95 SESSION, not by comparing independent percentiles: the
        # p95 of a sum is not the sum of the p95s, and quoting it that way would be
        # exactly the kind of arithmetic this probe exists to replace.
        worst = sorted(rows, key=lambda r: r["total_ms"])
        rank = max(1, min(len(worst), int(round(0.95 * len(worst) + 0.5))))
        out["at_p95_session"] = worst[rank - 1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default="256:46,128:124",
        help="bucket:streams pairs. The default is the two the open question compares: the "
        "bucket-256 arm that failed on latency at 46, and the bucket-128 arm that passed at 124",
    )
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--att-context-left", type=int, default=70)
    parser.add_argument("--window-s", type=float, default=180.0)
    parser.add_argument("--warm-up-s", type=float, default=60.0)
    parser.add_argument("--final-timeout-s", type=float, default=30.0)
    parser.add_argument("--stop-history-eou-ms", type=int, default=800)
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--pipeline",
        default="cache_aware_rnnt",
        choices=("cache_aware_rnnt", "fake"),
        help="`fake` runs the whole path on CPU with a stub recognizer. It measures nothing "
        "about a server and exists to prove the probe's own join and residual before any "
        "GPU time is spent on it",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    arms_spec = []
    for pair in args.arms.split(","):
        bucket, _, streams = pair.strip().partition(":")
        arms_spec.append((int(bucket), int(streams)))

    results = []
    for bucket, streams in arms_spec:
        print(f"\n=== bucket {bucket}, {streams} streams ===", flush=True)
        arm, tick_stats, boundaries = asyncio.run(
            run_arm(
                bucket=bucket,
                streams=streams,
                window_s=args.window_s,
                manifest=args.manifest,
                args=args,
            )
        )
        summary = decompose(
            arm, tick_stats, boundaries, period_ms=float(args.chunk_ms)
        )
        results.append(summary)
        terms = summary["terms"]
        print(
            f"  measured {summary['sessions_measured']} sessions, "
            f"{summary['sessions_refused']} refused, "
            f"{summary['sessions_unjoined']} unjoined, {summary['ticks']} ticks"
        )
        if summary["sessions_impossible_wait"]:
            print(
                f"  *** {summary['sessions_impossible_wait']} sessions had an IMPOSSIBLE "
                f"wait (examples {summary['impossible_wait_examples_ms']}): the join is "
                "wrong and nothing below should be read"
            )
        print(f"  {'term':<10} {'p50':>9} {'p95':>9} {'p99':>9}")
        for term in ("total_ms", "wait_ms", "overhead_ms", "step_ms", "edge_ms", "rest_ms"):
            t = terms[term]
            if t["n"]:
                print(
                    f"  {term:<10} {t['p50']:>8.1f}ms {t['p95']:>8.1f}ms {t['p99']:>8.1f}ms"
                )
        at = summary.get("at_p95_session")
        if at:
            print(
                f"  at the p95 session: total {at['total_ms']:.1f} = wait {at['wait_ms']:.1f}"
                f" + overhead {at['overhead_ms']:.1f} + step {at['step_ms']:.1f}"
                f" + edge {at['edge_ms']:.1f} ({at['edge_batches']:.0f} batches)"
                f" + rest {at['rest_ms']:.1f}"
            )

    record = {
        "record": "vb-latency-decomposition/1",
        "model": args.model,
        "chunk_ms": args.chunk_ms,
        "compute_dtype": args.compute_dtype,
        "window_s": args.window_s,
        "warm_up_s": args.warm_up_s,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "question": (
            "DR-0017 left open why bucket 256 blew the latency budget at 46 streams when its "
            "steady step costs 56 ms. This splits a final's latency into the grid wait, the "
            "tick overhead, the steady step, the serialised edge batches, and the publish."
        ),
        "note": (
            "In-process. No socket is in the path, which omits time the ladder's client pays; "
            "but the feeders share a GIL with the tick thread, which adds time the ladder's "
            "client does not. The two push opposite ways and neither is measured here, so "
            "this is NOT a bound on the ladder's figure in either direction -- it is a "
            "decomposition of the engine's own latency, and only the SHARES should be read "
            "across to the ladder. wait/step/edge/overhead come from the tick loop's own "
            "recorded boundaries and TickStats; rest is the residual."
        ),
        "arms": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nrecord: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
