# SPDX-License-Identifier: Apache-2.0
"""Three questions about the graph path that only the box can answer.

1. **Does the crossed-wire guard have anything to hold on to?** The RNNT and CTC
   adapters refuse a boundary bound to the sibling pipeline by looking for
   ``greedy_rnnt_decoder`` / ``greedy_ctc_decoder``. Those names come from NeMo's
   source as read through ``docs/ISSUES.md``; if the built pipeline declares neither,
   the guard accepts both and is a guard that cannot fail.

2. **Does a graph captured on the constructing thread replay on the tick thread?**
   Warm-up runs where the ``TickLoop`` is built; live steps run on the tick thread, and
   torch's current stream is thread-local. If the graph does not carry across, live
   ticks would re-capture or run eager, and the retained count would not say so.

3. **Does a live steady batch present the key warm-up captured?** Upstream's key
   includes ``drop_extra_pre_encoded``, and a session's first frame may set a different
   one. If it does, the first frame costs a second key rather than sharing one, and the
   graph budget is understated.

Everything here is read off NeMo's own ``_graphs`` and ``_key_counts``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Any

import numpy as np

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.types import PcmFrame
from verbatim.pipelines import registry
from verbatim.pipelines.cache_aware import NeMoBoundary
from verbatim.pipelines.nemo_runtime import (
    NeMoPipelineSpec,
    att_context_size,
    build_pipeline,
)
from verbatim.scheduler.boundary import PadPool
from verbatim.scheduler.capture import CaptureController

GRAPH_STEP_PATHS = (
    ("asr_model", "asr_model", "encoder", "_stream_step_cuda_graphs"),
    ("asr_model", "encoder", "_stream_step_cuda_graphs"),
)


def graph_step(pipeline: Any) -> Any:
    for path in GRAPH_STEP_PATHS:
        obj: Any = pipeline
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def keys(step: Any) -> dict[str, int]:
    return {str([str(p) for p in k]): v for k, v in step._key_counts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
    ap.add_argument("--chunk", type=int, default=160)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--att-context-left", type=int, default=70)
    ap.add_argument("--compute-dtype", default="bfloat16")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    chunk = ChunkMode(args.chunk)
    config = EngineConfig(chunk=chunk, buckets=(args.bucket,))
    att = att_context_size(args.model, chunk, left=args.att_context_left)
    spec = NeMoPipelineSpec(
        model=args.model,
        chunk=chunk,
        att_context=(att[0], att[1]),
        num_slots=config.num_slots,
        batch_size=args.bucket,
        use_cuda_graphs=True,
        compute_dtype=args.compute_dtype,
    )
    built = build_pipeline(spec)
    report: dict[str, Any] = {"model": args.model, "compute_dtype": args.compute_dtype}

    # --- 1. the crossed-wire guard --------------------------------------------------
    inner = getattr(built, "asr_model", None)
    report["crossed_wire_guard"] = {
        "pipeline_type": type(built).__name__,
        "inner_wrapper_type": type(inner).__name__ if inner is not None else None,
        "declares_greedy_rnnt_decoder": hasattr(built, "greedy_rnnt_decoder"),
        "declares_greedy_ctc_decoder": hasattr(built, "greedy_ctc_decoder"),
        "inner_declares_greedy_rnnt_decoder": hasattr(inner, "greedy_rnnt_decoder"),
        "inner_declares_greedy_ctc_decoder": hasattr(inner, "greedy_ctc_decoder"),
    }

    adapter = registry.build(
        "cache_aware_rnnt", config, boundary=NeMoBoundary.from_pipeline(built), use_cuda_graphs=True
    )
    step = graph_step(built)
    controller = CaptureController(config, adapter)
    pads = PadPool(max(config.buckets) + config.edge_batch, chunk)
    controller.warmup(pads)
    report["after_warmup"] = {"graphs": len(step._graphs), "key_counts": keys(step)}

    # --- 2. the same shape, stepped from another thread ------------------------------
    outcome: dict[str, Any] = {}

    def on_another_thread() -> None:
        try:
            pads.reset()
            adapter.transcribe_step(pads.take(args.bucket), keep_all_outputs=False, graph=True)
            outcome["raised"] = None
        except BaseException as exc:  # noqa: BLE001 - the answer is what it raised
            outcome["raised"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=on_another_thread, name="tick")
    worker.start()
    worker.join()
    pads.reset()
    report["other_thread_step"] = {
        "raised": outcome.get("raised"),
        "graphs": len(step._graphs),
        "key_counts": keys(step),
    }

    # --- 3. a live steady batch: one real first frame, the rest pads ------------------
    rng = np.random.default_rng(20260913)
    adapter.open_stream(1, None)
    samples = rng.uniform(-0.5, 0.5, size=chunk.samples).astype(np.float32)
    first = PcmFrame(
        stream_id=1, samples=samples, is_first=True, is_last=False, valid_samples=chunk.samples
    )
    pads.reset()
    batch = [first, *pads.take(args.bucket - 1)]
    adapter.transcribe_step(batch, keep_all_outputs=False, graph=True)
    report["live_first_frame_step"] = {"graphs": len(step._graphs), "key_counts": keys(step)}

    later = PcmFrame(
        stream_id=1,
        samples=rng.uniform(-0.5, 0.5, size=chunk.samples).astype(np.float32),
        is_first=False,
        is_last=False,
        valid_samples=chunk.samples,
    )
    pads.reset()
    adapter.transcribe_step(
        [later, *pads.take(args.bucket - 1)], keep_all_outputs=False, graph=True
    )
    report["live_steady_step"] = {"graphs": len(step._graphs), "key_counts": keys(step)}
    adapter.close_stream(1)

    text = json.dumps(report, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
