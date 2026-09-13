# DR-0014 — The ragged control arm, so the invariance gate can fail

**Date:** 2026-09-14
**Status:** accepted

## What was wrong with a passing gate

On 2026-09-13 the batch-invariance gate ran against a live server for the first time, on a B300:
256 streams at concurrency 1 / 32a / 32b / 38, zero errors at every level, and one identical digest
`f68cbccfa089…` across all four, text and word timings both compared, at bfloat16. That result is
the centre of what this project claims.

It is also, on its own, close to a restatement of the design. `scheduler/buckets.py` pads the steady
batch to exactly `B` at every occupancy, so concurrency 1 and concurrency 38 present the encoder the
**identical shape**. The digests matching is what the padding is *for*. The question a reader is
entitled to ask — and which three independent reviews asked, in the same words — is:

> What outcome would have made this gate fail?

Without an answer, the headline is a tautology presented as a measurement. `benchmarks/METHODOLOGY.md`
already knew this: kill condition 3's 2026-09-10 amendment requires "the same corpus re-run with
fixed-shape batching disabled", and until now that arm was **unrunnable** — `cli.py` offered
`--ceiling` xor `--bucket`, one integer, and no way to turn the padding off.

## The decision

`EngineConfig.padding` is `"fixed"` (the shipped policy, the default, unchanged behaviour) or
`"ragged"`. Under ragged the steady batch is exactly the live rows and the edge batches are exactly
the finals: no pad row anywhere, so the shape the encoder sees tracks occupancy. `serve --padding
ragged` reaches it, and the banner says so in the same register as the precision warning, because an
operator who arrives there by accident is running a server that is **not** batch-invariant.

Three details that are decisions rather than mechanics:

1. **Ragged disables padding on the edge batches too.** An arm that unpadded the steady batch and
   kept padding the edge batch would not be the arm the amendment asks for. `_ragged_edge_batches`
   exists beside `partition_edge_batches` rather than growing a flag inside it, so the shipped path
   is byte-for-byte what it was.
2. **Ragged and the graph path are refused together, at construction.** A ragged batch changes shape
   with occupancy, so there is no stable key and nothing could be captured. These are not a
   trade-off to resolve silently in one direction: a run that quietly dropped one of them would
   publish a row naming both. The refusal names the eager arm as the way to run it.
3. **Ragged is not a serving mode and is not offered as one.** It is a control. If it turns out to
   be the better default, that is a separate decision taken on evidence, not a side effect of
   building the control.

## What each outcome of the arm would mean

This is written down **before** the arm is run, which is the point of writing it down at all.

- **Ragged diverges, fixed does not.** The expected result, and the one that turns the gate from a
  restatement into a measurement. Fixed-shape padding is then demonstrably what buys invariance, its
  cost in padded rows is the price of the property, and both numbers get published together.
- **Ragged is clean too.** Then the padding is **not** what buys invariance on this hardware and
  precision — something else is holding the property, or the corpus and concurrency span are not
  discriminating. In that case the shipped default should be reconsidered, because padding a 32-row
  step for 22 live sessions spends roughly a third of the encoder on rows that exist only to hold a
  shape. This outcome is not a failure of the arm; it is the arm working and telling us the design
  rationale is wrong. Saying so here, in advance, is what stops it being explained away afterwards.
- **Fixed diverges.** The gate's own fake-leak tests already show it can go red, so this would be a
  finding about the server rather than about the gate, and the more urgent of the two.

## Rejected alternatives

**Leave the caveat in the commit message and move on.** It was already written down in three places
by the time this was built, which is exactly the shape of a known problem nobody is fixing. A claim
this central cannot rest on a control that does not exist.

**Vary the bucket size between levels instead of disabling padding.** Cheaper, and it tests a
different thing: it would compare two fixed shapes rather than fixed against no shape at all, so a
leak that depends on occupancy *within* a shape would survive it. The amendment asks for padding
disabled, and that is what this does.

**Ship ragged as a performance option now.** It would very likely raise the sustained stream count,
since the encoder would stop paying for pad rows. It would also mean shipping a mode that breaks the
property the server is named for, on the strength of a capacity number and before the control has
been run once. Capacity measured on the ragged arm is a control's number, not a product's.
