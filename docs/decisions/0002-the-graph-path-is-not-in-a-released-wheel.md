# DR-0002 — The CUDA-graph encoder step is not in a released NeMo wheel

**Date:** 2026-09-10
**Status:** accepted

## The finding

`verbatim doctor` refuses the graph path when the installed NeMo lacks the graphed streaming encoder
step, rather than running eager unasked. On first use it refused, and the refusal is correct.

Checked directly in both installed environments, not inferred from a version number:

| toolkit | `streaming_encoder_cuda_graphs` module | `CudaGraphsStreamingEncoderStep` | `set_streaming_cuda_graphs` |
|---|---|---|---|
| NeMo 2.7.3 | absent | absent | absent |
| NeMo 3.0.0 | absent | absent | absent |
| Speech source tree | present | present | present |

Both halves of [NeMo PR #15863](https://github.com/NVIDIA-NeMo/Speech/pull/15863) exist only in the
source tree. No released wheel we have carries them.

## Why this matters more than a packaging detail

The CUDA-graph encoder step is the premise of this project. The README's argument is that PR #15863
moved the bottleneck from the GPU to the host, which is what makes a server rather than a kernel the
interesting problem. The 3.08–5.14x figure cited there is upstream's, measured on that path.

Both kill gates read against it:

- **Day 21** compares the prototype against `asr_streaming_infer.py` with `use_cuda_graphs=true` as the
  file-driven graphed ceiling. From a released wheel, that reference arm cannot be graphed either.
- **Day 45** asks whether concurrency leaves transcripts bit-identical *on the graph path*, and says to
  de-scope if that is unattainable there.

## Decision

1. **`serve` refuses the graph path rather than degrading**, and `--eager` must be passed explicitly.
   A row can therefore never look graphed without being graphed. This is why the guard was written to
   refuse rather than warn.
2. **Every row records which it ran.** An eager row and a graphed row are not comparable and must not
   share a table column without saying so.
3. **The first load run may be eager, and its row will say so.** Installing NeMo from source is the
   alternative and is a change of dependency, not a flag, so it is a decision with its own consequences
   for anyone reproducing the row.
4. **No kill threshold moves because of this.** The thresholds were frozen before any number existed and
   they stay frozen. What changes is that a row must name its execution mode, which is a reporting rule,
   not a threshold.

## What would retire this record

A released NeMo wheel carrying both halves of PR #15863. At that point `doctor` stops refusing on a
default install, and the day-21 and day-45 runs can be repeated on the graph path and compared against
whatever eager rows exist by then.
