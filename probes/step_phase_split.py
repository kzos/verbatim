"""Where the tick's time actually goes: encoder, decoder, or the Python around them.

`step_ms` in `verbatim.pipelines.cache_aware` times NeMo's whole `transcribe_step` as one
block, so every capacity conclusion this project has drawn about *why* the step costs what
it costs is arithmetic and code-reading rather than measurement. DR-0017 says so in those
words and asks for this.

The question it settles. From the file-driven ceiling row the step costs about
`28 ms + 0.85 ms per row`, and that slope is what puts the capacity fixed point at 124.
Two readings of the same number:

* the encoder dominates, and the answer is a faster or disaggregated encoder;
* the per-row *host* work around it dominates -- a `get_state` and a
  `cleanup_after_response` per row, a context dictionary walk per row, a greedy decode per
  row, a tokenizer `ids_to_text` per row per tick, and the label-looping decoder's
  `while active_mask.any()` synchronisations -- and an encoder change buys almost nothing.

NeMo splits cleanly at `CacheAwareRNNTInferenceWrapper.execute_step`, which calls
`encoder_step` and then `rnnt_decoder_predictions_tensor`. Both are wrapped here with
device synchronisation on either side, so the numbers are wall clock and not queue depth.
Everything in the step that is neither is reported as "the rest", which is the term the
two readings disagree about.

Run it on the machine under test:

    PYTHONPATH=src:bench/src python probes/step_phase_split.py --batches 32,128,256

It loads the checkpoint, so it needs a GPU and the model runtime. It writes a JSON record
and prints a table. Nothing here is part of the server.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Phase:
    """One wrapped callable's synchronised wall clock, in milliseconds."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[float] = []

    def wrap(self, torch: Any, fn: Any) -> Any:
        def timed(*args: Any, **kwargs: Any) -> Any:
            _sync(torch)
            started = time.perf_counter()
            out = fn(*args, **kwargs)
            _sync(torch)
            self.samples.append((time.perf_counter() - started) * 1000.0)
            return out

        return timed

    def summary(self, warmup: int) -> dict[str, float | int | None]:
        kept = self.samples[warmup:]
        if not kept:
            return {"calls": len(self.samples), "median_ms": None, "mean_ms": None}
        return {
            "calls": len(kept),
            "median_ms": statistics.median(kept),
            "mean_ms": statistics.fmean(kept),
            "min_ms": min(kept),
            "max_ms": max(kept),
        }


def _speech_rows(
    manifest: Path, batch: int, samples_per_chunk: int, steps: int
) -> list[list[np.ndarray]]:
    """`batch` streams of real audio, each a list of `steps` consecutive chunks.

    Every row gets a different utterance, cycling the corpus if it is smaller than the
    batch, so the decoder does the work real speech makes it do rather than the work
    silence makes it do.
    """
    from verbatim_bench.corpus import load_manifest, read_pcm16

    utterances = list(load_manifest(manifest))
    if not utterances:
        raise SystemExit(f"{manifest} holds no utterances")
    rows: list[list[np.ndarray]] = []
    for index in range(batch):
        pcm = read_pcm16(utterances[index % len(utterances)].audio_path)
        floats = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        chunks: list[np.ndarray] = []
        for start in range(0, len(floats) - samples_per_chunk, samples_per_chunk):
            chunks.append(np.ascontiguousarray(floats[start : start + samples_per_chunk]))
            if len(chunks) >= steps:
                break
        if not chunks:
            chunks = [np.zeros(samples_per_chunk, dtype=np.float32)]
        rows.append(chunks)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    )
    parser.add_argument("--batches", default="32,128,256")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--att-context-left", type=int, default=70)
    parser.add_argument("--steps", type=int, default=24, help="steps per batch size")
    parser.add_argument("--warmup", type=int, default=6, help="leading steps discarded")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "drive real corpus audio instead of uniform noise. It matters: noise decodes "
            "to almost nothing, so the greedy loop, the endpointer and the per-row "
            "ids_to_text all do their cheapest possible work and the per-row cost is "
            "understated. A number meant to describe a server has to be taken on speech"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    from verbatim.config import ChunkMode
    from verbatim.core.types import PcmFrame
    from verbatim.pipelines.cache_aware import NeMoBoundary
    from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter
    from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, build_pipeline

    chunk = ChunkMode(args.chunk_ms)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    arms: list[dict[str, Any]] = []

    for batch in batches:
        slots = batch * 2 + 16
        spec = NeMoPipelineSpec(
            model=args.model,
            chunk=chunk,
            att_context=(args.att_context_left, chunk.ms // 80 - 1),
            num_slots=slots,
            batch_size=batch,
            decoding="rnnt",
            use_cuda_graphs=False,
            compute_dtype=args.compute_dtype,
            device_id=args.device_id,
        )
        pipeline = build_pipeline(spec)

        # The seam: execute_step calls encoder_step and then the RNNT decoder. Both are
        # resolved rather than assumed, and the probe says what it found.
        wrapper = pipeline.asr_model
        decoding = wrapper.asr_model.decoding
        encoder = Phase("encoder_step")
        decoder = Phase("rnnt_decoder_predictions_tensor")
        wrapper.encoder_step = encoder.wrap(torch, wrapper.encoder_step)
        decoding.rnnt_decoder_predictions_tensor = decoder.wrap(
            torch, decoding.rnnt_decoder_predictions_tensor
        )

        adapter = CacheAwareRNNTAdapter(
            chunk,
            NeMoBoundary.from_pipeline(pipeline, graph_step_available=False),
            buckets=(batch,),
            required_slots=slots,
        )
        for stream_id in range(batch):
            adapter.open_stream(stream_id, None)

        if args.manifest is not None:
            audio = _speech_rows(args.manifest, batch, chunk.samples, args.steps)
            source = f"manifest {args.manifest.name}"
        else:
            rng = np.random.default_rng(20260916)
            audio = [
                [rng.uniform(-0.4, 0.4, size=chunk.samples).astype(np.float32)] * args.steps
                for _ in range(batch)
            ]
            source = "uniform noise"
        totals: list[float] = []
        for _step in range(args.steps):
            frames = [
                PcmFrame(
                    stream_id=i,
                    samples=audio[i][_step % len(audio[i])],
                    is_first=False,
                    is_last=False,
                    valid_samples=chunk.samples,
                )
                for i in range(batch)
            ]
            _sync(torch)
            started = time.perf_counter()
            adapter.transcribe_step(frames, keep_all_outputs=False, graph=False)
            _sync(torch)
            totals.append((time.perf_counter() - started) * 1000.0)

        kept = totals[args.warmup :]
        step_median = statistics.median(kept)
        enc = encoder.summary(args.warmup)
        dec = decoder.summary(args.warmup)
        rest = step_median - (enc["median_ms"] or 0.0) - (dec["median_ms"] or 0.0)
        arms.append(
            {
                "batch": batch,
                "steps": len(kept),
                "step_median_ms": step_median,
                "step_mean_ms": statistics.fmean(kept),
                "encoder": enc,
                "decoder": dec,
                "rest_ms": rest,
                "encoder_share": (enc["median_ms"] or 0.0) / step_median,
                "decoder_share": (dec["median_ms"] or 0.0) / step_median,
                "rest_share": rest / step_median,
                "audio": source,
            }
        )
        for stream_id in range(batch):
            adapter.close_stream(stream_id)
        del adapter, pipeline
        torch.cuda.empty_cache()

    record = {
        "record": "vb-step-phase/1",
        "model": args.model,
        "chunk_ms": args.chunk_ms,
        "compute_dtype": args.compute_dtype,
        "steps_per_batch": args.steps,
        "warmup_discarded": args.warmup,
        "note": (
            "encoder_step and rnnt_decoder_predictions_tensor are wrapped with device "
            "synchronisation on either side; 'rest' is the remainder of NeMo's "
            "transcribe_step, which is the per-row host work the two readings of "
            "DR-0017 disagree about"
        ),
        "arms": arms,
    }
    header = f"{'batch':>6} {'step':>9} {'encoder':>9} {'decoder':>9} {'rest':>9}"
    print(f"{header}   shares (enc/dec/rest)")
    for arm in arms:
        print(
            f"{arm['batch']:>6} {arm['step_median_ms']:>8.1f}ms "
            f"{(arm['encoder']['median_ms'] or 0):>8.1f}ms "
            f"{(arm['decoder']['median_ms'] or 0):>8.1f}ms "
            f"{arm['rest_ms']:>8.1f}ms   "
            f"{arm['encoder_share']:.0%} / {arm['decoder_share']:.0%} / {arm['rest_share']:.0%}"
        )
    if len(arms) >= 2:
        first, last = arms[0], arms[-1]
        span = last["batch"] - first["batch"]
        print()
        for label, key in (
            ("step", "step_median_ms"),
            ("encoder", None),
            ("decoder", None),
            ("rest", "rest_ms"),
        ):
            if key is None:
                lo = first["encoder" if label == "encoder" else "decoder"]["median_ms"] or 0.0
                hi = last["encoder" if label == "encoder" else "decoder"]["median_ms"] or 0.0
            else:
                lo, hi = first[key], last[key]
            print(f"  marginal {label:<8} {(hi - lo) / span:.3f} ms per row")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nrecord: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
