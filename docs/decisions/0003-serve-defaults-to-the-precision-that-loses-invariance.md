# DR-0003 — `serve` defaults to bfloat16, which is the precision that loses batch invariance

**Date:** 2026-09-11
**Status:** accepted, and it blocks the day-45 gate as written

## What happened

`verbatim serve` ran against a real NeMo pipeline on an A6000 for the first time and **worked on the
first attempt**: it built the pipeline, reserved its slots, served both wires, transcribed real audio
with word timings, and shut down cleanly on SIGTERM. Its banner:

```
[verbatim] pipeline     NeMo cache-aware RNNT, bfloat16 on cuda:0 (NVIDIA RTX A6000)
[verbatim] chunk mode   160 ms (att_context_size [70, 1])
[verbatim] graphs       EAGER encoder step, by --eager
[verbatim] admission    bucket 32 streams, UNCALIBRATED (named for the ladder run)
[verbatim] slots        79 NeMo slots: bucket 32, 32 steady pads, 7 edge pads, 8 margin
```

## The problem the banner states in its own second line

**`bfloat16`.** That is the compute dtype `serve` chooses by default, inherited from NeMo's own example
configuration. It is also the precision measured on 2026-09-10 as diverging **287 times in 2,939
utterances** on this exact code path, where float32 diverges **6**, and where equalising row lengths
removes nothing (328 against 287). See evidence section 21.

So the server, as it ships today, defaults to the one setting that costs it the property it is named
for. Nothing was hidden: the banner says `bfloat16` and the evidence says what bfloat16 does. They had
simply never been read next to each other, because until tonight the server had never run.

## Confirmed on a second architecture, 2026-09-11

Repeated on a B300 with NeMo built from source, through `serve`'s own pipeline-building code:

| precision | A6000 ragged / equalised | B300 ragged / equalised |
|---|---|---|
| bfloat16 | 287 / 328 | **264 / 295** |
| float32 | 6 / 6 | **1 / 1** |

The effect follows the precision, not the die. Newer silicon does not fix it; the gap is **wider** there,
about 264x against the A6000's 48x, because float32 on Blackwell is so nearly clean. Equalising row
lengths makes it slightly worse on both machines.

## Decision

1. **This is recorded before any row is produced.** No benchmark row may be published from a bfloat16
   run without this record cited beside it, because a throughput number obtained at a precision that
   fails the invariance gate is not a number about this project's product.
2. **The day-45 invariance gate cannot pass at the default.** `KILL.md` condition 3 asks whether
   concurrency and batch composition leave transcripts and timestamps bit-identical. At bfloat16 on this
   path they do not, by a factor of about forty-eight against float32. **No threshold moves**; what
   changes is that the gate must name its precision, exactly as it must now name eager or graphed.
3. **The measurement nobody has taken is the one that decides the product.** float32 costs throughput
   and buys invariance; bfloat16 costs invariance and buys throughput. Which trade the server should
   default to is a measured question, and the ladder run should be taken at **both** precisions rather
   than at the default alone.
4. **`serve` should say this at startup**, not only in a document: when the chosen dtype is not float32,
   the banner should say that batch invariance has been measured to fail at that precision. Recorded
   here, not implemented; it is a source change.

## A second, independent reason the graph path is unavailable here

Beyond [DR-0002](0002-the-graph-path-is-not-in-a-released-wheel.md), which is about the wheel, this
machine's driver also refuses the decoder's own graphs:

```
Cuda graphs with while loops are disabled, decoding speed will be slower
Reason: Driver supports cuda toolkit version 12.4, but the driver needs to support at least 12.6.
```

So a graphed row on this box needs **both** a NeMo built from source **and** a driver upgrade. That is
worth knowing before the day-21 run is scheduled, and it was invisible until something actually loaded a
model.

## What went right, and is worth keeping

The first real build was expected to fail. It did not, and the reason it did not is that the round-2
adapter guards, the round-4 attention-context derivation and the slot arithmetic were all written from
NeMo's source rather than guessed, then tested against a fake standing at NeMo's own seam. The slot line
in the banner, `79 = 32 + 32 + 7 + 8`, is the arithmetic that was argued about in review, printed by the
running server and accepted by NeMo.
