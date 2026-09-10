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

## Amended 2026-09-11: the path was reached, and it needs two things, not one

The graph path now runs. It took a source build **and** a driver, and the second was not in this record
when it was written.

| | A6000 box | B300 workspace |
|---|---|---|
| driver | 550.144.03, CUDA 12.4 | 580.126.09, CUDA 13.1 |
| NeMo | 3.0.0 wheel | **3.1.0+abb8254da, built from source** |
| graphed encoder step | absent | **present** |
| `doctor` verdict | eager only, exit 3 | **graph path available, exit 0** |
| decoder's own CUDA graphs | **disabled by the driver** | enabled |

**The driver is the harder half.** Beyond the missing encoder step, the A6000 box's driver also refuses
the decoder's graphs: *"Cuda graphs with while loops are disabled ... Driver supports cuda toolkit
version 12.4, but the driver needs to support at least 12.6."* That warning appears three times on the
A6000 and **zero times** on the B300. A source build cannot fix it and neither can any package; it needs
a driver upgrade, which needs root on the host. So **an A6000 box on driver 550 is eager-only, whatever
is installed.**

`doctor` printed `verdict: graph path available` and exited 0 for the first time on 2026-09-11. That
positive verdict had until then only ever been tested against the shape of the upstream change, never
against a machine that had it.

`serve` then ran the graph path end to end on the B300: it built the pipeline, reported
`graphs  graph path (CUDA graphs, NeMo PR #15863)`, served both wires, transcribed real audio and shut
down on SIGTERM with exit 0.

**One observation, and it is one utterance, not a claim.** The same LibriSpeech utterance transcribed
through both machines gave the identical string *and the identical word timings to the millisecond*,
across two architectures, two NeMo versions and eager against graphed:

```
A6000, eager,   NeMo 3.0.0 wheel   ->  "i'm from the cutter lying off the coast"  audio_s 2.095
B300,  graphed, NeMo 3.1.0 source  ->  "i'm from the cutter lying off the coast"  audio_s 2.095
words in both: [i'm 480-720] [from 720-800] [the 800-960] [cutter 960-1280]
```

That is one utterance out of one. It is recorded because it is the first cross-machine agreement this
project has ever observed through its own server rather than through a probe, and because if it had
*dis*agreed that would have been a finding. It is not evidence of invariance and must not be cited as
any.

**Consequence for the gates.** A graphed row is now producible, on the B300 and not on the A6000. The
day-21 condition names one RTX A6000 and requires a graphed reference arm. **That arm cannot be produced
on the die the condition names.** No threshold is being moved here; the conflict is recorded and belongs
to the author.

## What would retire this record

A released NeMo wheel carrying both halves of PR #15863. At that point `doctor` stops refusing on a
default install, and the day-21 and day-45 runs can be repeated on the graph path and compared against
whatever eager rows exist by then.
