# SPDX-License-Identifier: Apache-2.0
"""Does Verbatim's warm-up actually leave a CUDA graph behind?

``CaptureController.warmup`` steps each bucket shape ``GRAPH_WARMUP_STEPS`` times and
then records the key as captured. Nothing in that path asks NeMo whether a capture
happened -- ``captured_keys`` is a record of intent. NeMo's own
``CudaGraphsStreamingEncoderStep`` keeps the truth in ``_graphs`` (captured) and
``_key_counts`` (calls seen per key), and captures only when a key's count exceeds
``warmup_steps``. It also falls back to eager silently for any step it cannot capture,
including, by its own docstring, every step under CUDA autocast.

So this reads NeMo's two dictionaries: after the build, after Verbatim's warm-up, and
after further steps taken one at a time. It writes no conclusion it did not read off
the object.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from verbatim.config import ChunkMode, EngineConfig
from verbatim.pipelines import registry
from verbatim.pipelines.cache_aware import NeMoBoundary
from verbatim.pipelines.nemo_runtime import (
    NeMoPipelineSpec,
    att_context_size,
    build_pipeline,
)
from verbatim.scheduler.boundary import PadPool
from verbatim.scheduler.capture import CaptureController

#: Where ``StreamingEncoder.set_streaming_cuda_graphs`` puts it, read from NeMo's own
#: source on the box: ``streaming.py`` sets ``self._stream_step_cuda_graphs`` on the
#: encoder, and the inference wrapper reaches it as ``asr_model.encoder``.
KNOWN_PATHS = (
    "asr_model.asr_model.encoder._stream_step_cuda_graphs",
    "asr_model.encoder._stream_step_cuda_graphs",
    "encoder._stream_step_cuda_graphs",
)


def _by_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_graph_step(pipeline: Any) -> Any:
    """The ``CudaGraphsStreamingEncoderStep`` NeMo attached: the documented places
    first, then a bounded walk if it moved."""
    for path in KNOWN_PATHS:
        found = _by_path(pipeline, path)
        if found is not None:
            return path, found
    seen: set[int] = set()
    queue: list[tuple[str, Any]] = [("pipeline", pipeline)]
    while queue:
        path, obj = queue.pop(0)
        if id(obj) in seen or len(seen) > 4000:
            continue
        seen.add(id(obj))
        if type(obj).__name__ == "CudaGraphsStreamingEncoderStep":
            return path, obj
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if callable(child) or isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            if type(child).__module__.split(".")[0] in {"nemo", "torch", "omegaconf"}:
                queue.append((f"{path}.{name}", child))
    return None, None


def snapshot(step: Any) -> dict[str, Any]:
    if step is None:
        return {"graphs": None, "key_counts": None}
    return {
        "mode": str(step.cuda_graphs_mode),
        "warmup_steps": step.warmup_steps,
        "max_graphs": step.max_graphs,
        "allow_fallback": step.cuda_graphs_allow_fallback,
        "graphs": len(step._graphs),
        "graph_keys": [list(map(str, k)) for k in step._graphs],
        "key_counts": {str(list(map(str, k))): v for k, v in step._key_counts.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
    ap.add_argument("--chunk", type=int, default=160)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--att-context-left", type=int, default=70)
    ap.add_argument("--compute-dtype", default="bfloat16")
    ap.add_argument(
        "--pipeline",
        default="cache_aware_rnnt",
        choices=("cache_aware_rnnt", "cache_aware_ctc"),
        help="which cache-aware branch to build and probe; the CTC arm is the one "
        "whose graph switch cannot be read from an installed wheel",
    )
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--extra-steps", type=int, default=6)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    chunk = ChunkMode(args.chunk)
    config = EngineConfig(chunk=chunk, buckets=(args.bucket,))
    att_context = att_context_size(args.model, chunk, left=args.att_context_left)
    spec = NeMoPipelineSpec(
        model=args.model,
        chunk=chunk,
        att_context=(att_context[0], att_context[1]),
        num_slots=config.num_slots,
        batch_size=args.bucket,
        use_cuda_graphs=True,
        compute_dtype=args.compute_dtype,
        device_id=args.device_id,
        decoding="ctc" if args.pipeline == "cache_aware_ctc" else "rnnt",
    )
    nemo_pipeline = build_pipeline(spec)
    adapter = registry.build(
        args.pipeline,
        config,
        boundary=NeMoBoundary.from_pipeline(nemo_pipeline),
        use_cuda_graphs=True,
    )
    path, step = find_graph_step(nemo_pipeline)

    report: dict[str, Any] = {
        "model": args.model,
        "chunk_ms": args.chunk,
        "bucket": args.bucket,
        "compute_dtype": args.compute_dtype,
        "pipeline": args.pipeline,
        "graph_step_found_at": path,
        "verbatim_capability": str(adapter.graph_capability()),
        "after_build": snapshot(step),
    }

    controller = CaptureController(config, adapter)
    # The same pad pool the bucket table builds, so warm-up sees the shapes it would.
    pads = PadPool(max(config.buckets) + config.edge_batch, chunk)
    claimed = controller.warmup(pads)
    report["verbatim_warmup_steps"] = controller.warmup_steps
    report["verbatim_claims_captured"] = [str(k) for k in claimed]
    report["after_verbatim_warmup"] = snapshot(step)

    # One step at a time past warm-up: the count at which NeMo actually captures is the
    # number this whole question turns on.
    per_step = []
    for i in range(args.extra_steps):
        pads.reset()
        frames = pads.take(args.bucket)
        adapter.transcribe_step(frames, keep_all_outputs=False, graph=True)
        per_step.append({"extra_step": i + 1, "graphs": 0 if step is None else len(step._graphs)})
    pads.reset()
    report["one_step_at_a_time"] = per_step
    report["after_extra_steps"] = snapshot(step)

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
