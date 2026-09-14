# DR-0015: a phrase list is an input to the transcript, so biasing is its own arm

**Status:** accepted, 2026-09-14.

## Context

Verbatim sells "run it again and get the same transcript". The buyers for that —
medical records, legal depositions, financial compliance — are also the buyers whose
vocabulary the model gets wrong: drug names, ICD-10 codes, case citations, the parties
on this particular call. A server that is reproducible and cannot spell `hydrochloro-
thiazide` is reproducibly wrong.

NeMo carries a per-stream boosting tree, and the Riva wire Verbatim already speaks
carries `SpeechContext{phrases, boost}`. Until this change both ends existed and nothing
joined them: `options_from_config` accepted `speech_contexts` and noted them as ignored,
and the WebSocket query had no way to send any.

## Decision

**Per-session phrase lists, on NeMo's per-stream boosting tree, behind `--biasing`, and
a run with biasing on is a separate arm from a run without it.**

Four parts, and the fourth is the one that is easy to skip.

1. A session's phrases reach NeMo as one `BiasingRequestItemConfig` on its
   `ASRRequestOptions`, built in `open_stream` and released in `close_stream`.
2. `--biasing` is off by default and is read back off the built adapter, not off the
   flag, for the banner, `/readyz` and every metric label.
3. Every way the feature can fail quietly is a refusal: a CTC pipeline, a decoder built
   without the biasing arena, a boundary that cannot release an entry, and a session that
   sends phrases to a server without `--biasing`.
4. Turning biasing on changes the decoder's arithmetic for **every** row, biased or not:
   the fused label-looping path takes a second max over the vocabulary view and a
   `torch.where` where the bare path takes one max. So the transcript digests published
   for a server without biasing do not carry over to one with it, and the bench refuses a
   run whose phrase lists and whose server disagree.

## What was rejected, and why

**One boosting tree for the whole process** (`asr.decoding.greedy.boosting_tree`, which
Verbatim already writes as an empty block). It is nearly free — no per-session build, no
arena churn — and it benchmarks beautifully. It is also a cross-tenant channel: one
deployment-wide vocabulary is fused into every stream's decode, so one customer's patient
names and case parties are boosted inside another customer's transcript. For the buyer
this server is for, that is a disclosure question, not a word-error-rate trade. The block
stays empty and says why.

**An n-gram language model under shallow fusion (NGPU-LM).** Also per-row and therefore
just as invariant, so there is no invariance argument for it — and it buys the wrong
thing. It gives domain *priors*; what is wrong per customer is per-*call* vocabulary,
which the wire already carries. It also needs a domain text corpus that does not exist
here and hundreds of megabytes of resident arena beside the encoder cache. It remains the
right complement later, for phrasing and formats, once a rare-term harness shows the
failures are collocational rather than term-identity.

**Fine-tuning or a LoRA per domain.** It changes the checkpoint the invariance rows are
named against, needs domain audio the customer does not have, and does not scale: one
adapter per tenant is unmanageable and a merged one dilutes every tenant.

**LLM post-correction.** It destroys the property being sold. "Reproducible transcript"
and "a language model rewrote it afterwards" cannot both be true.

**Beam search (MALSD), for now.** Greedy boosting has a real ceiling: the blank-versus-
emit decision is taken from the *unbiased* argmax, so boosting can respell a token the
model already chose to emit and can never turn a blank into an emission. Rare terms fail
as deletions about as often as substitutions and no weight recovers those. Deferred with
a trigger rather than dismissed: if a rare-term harness shows greedy boosting closing
less than about half the gap *and* the residual errors are deletions, that is the
signature of the blank ceiling and the only thing beam fixes that tuning cannot. It would
mean re-earning the invariance property under a different decoder.

**NeMo's phrase-tree cache (`BiasingRequestItemConfig.cache_key`).** This one was
planned, and dropped on reading the code. The cache is a module-level dictionary with no
eviction and no tenant scope, so a shared key hands one caller's compiled vocabulary to
another. It is the sharpest benchmarks-well/deploys-badly item on the list: it makes a
demo fast and makes the product reportable. The tree is rebuilt per session, and what
that costs a tick is measured rather than assumed.

## What this costs, named rather than estimated

The tree is built on the **tick thread**, inside the step: NeMo registers a stream's tree
from `_prepare_per_stream_biasing`, which runs before the encoder call, so a session
arriving pays tokenisation, a graph build and a host-to-device copy inside one tick that
every other row is waiting on. `build_multi_biasing_ids_np` also calls `.item()` on a
device buffer per biased stream per chunk. Neither number is written here because neither
has been measured; the gate reads tick cost and the ladder reads capacity, and those are
where the numbers come from.

Three deployment traps are recorded now so they are not discovered later: over-boosting
inserts list words into audio that merely sounds like them, which is why the negative
control exists and why the server caps the weight it will apply; the formats that matter
most (`E11.9`, `410 U.S. 113`) are exactly the ones that tokenise worst, so they are in
the shipped phrase book rather than plain words; and session churn is arena churn, one
add and one remove per biased session.

## The defect this found

NeMo releases a stream's biasing arena entry only on an `is_last` frame, inside
`transcribe_step`. `delete_state` frees nothing. Verbatim's `close_stream` therefore had
to release it too, or every session that ends without a final — a dropped socket, an idle
deadline, a failed step — would cost one arena entry for the life of the process. A
benchmark would never have shown it: every benchmark stream sends a final. The CPU fake
holds the same rule so the leak is reproducible without a GPU, and the test that pins it
goes red when the release is removed.

## How this is checked

The invariance gate takes a phrase book. A stream keeps the same list at every
concurrency level — the gate's question is what changes when batch composition changes,
so the only other input is held fixed — and half the corpus carries a list, interleaved,
so a biased row and an unbiased row sit in the same batch.

Two controls run first, at concurrency 1:

- **The positive control can void the run.** A phrase list that reaches nothing produces
  exactly the transcripts an unbiased server produces, every level agrees, and the gate
  would report "invariant" for a feature that was never on. Each control clip is run bare,
  the reference words that pass did **not** produce are boosted, and it is run again; at
  least one pair must differ, and when none does the verdict is `uncontrolled` and there
  is no verdict at all.

  Boosting the *missed* words rather than the reference's rare words is the whole design,
  and it was learned from a control that reported failure on a working server. A list of
  words the model already emitted changes nothing — correctly — so the first version of
  this control measured its own choice of words. A clip the bare pass got entirely right
  is skipped rather than counted against the server, and the scan walks further down the
  corpus to find one that has something to prove.
- **The negative control is a reading, not a gate.** One clip is run against an unrelated
  list and any word it inserted is recorded. Over-boosting is an accuracy fact about a
  weight; the verdict is about batch composition, and conflating them would mean a green
  gate could be bought by tuning.

The ragged control arm (DR-0014) matters more here, not less: a padded server passing its
own padding test is close to a restatement of the design, and a boost large enough to
swamp decision margins could collapse the ragged arm's divergence and quietly remove the
gate's power to fail. A biasing run reports alongside a ragged run with the same book.

## What it measured, 2026-09-14, B300, bfloat16, eager

Both arms: 256 LibriSpeech test-other utterances (`sha256:6a142a96…`), phrase book
`domains-v1` (`cc726a6a…`) at weight 2.0, 128 of the 256 streams carrying a list,
interleaved; levels 1 / 32 / 32 / 42 with the max level churned on a 20 s triangle wave.
Both arms' positive control changed 4 of 4 clips and both negative controls inserted
nothing.

| arm | verdict | digests | streams differing from concurrency 1 (32a / 32b / max) |
|---|---|---|---|
| fixed + biasing | invariant | one, `bbf5e18dc3bf…` | 0 / 0 / 0 |
| ragged + biasing | divergent | four, all different | 117 / 117 / 89 |
| ragged, no biasing (same shape, DR-0014) | divergent | four | 118 / 118 / 88 |

Two readings, and the second is the one that makes the first mean anything.

**The property holds with per-session vocabularies.** One digest across four levels,
churned, with biased and unbiased rows sharing every batch.

**Boosting did not blunt the control.** The ragged arm diverged 117 / 117 / 89 against
118 / 118 / 88 for the same configuration without biasing: one stream of difference in
each comparison. A weight high enough to swamp decision margins would have pulled those
numbers toward zero and left a gate that could no longer fail. It did not.

The fixed arm's digest is `bbf5e18dc3bf…` where the same server without biasing produced
`f68cbccfa089…`. That is the predicted cost of the fused decode path, and it is the whole
reason this is a separate arm rather than a feature folded into the existing rows.

One difference between the two ragged runs is **not** claimed here: the biased one has 2
streams differing between 32a and 32b where the unbiased one had none. On a server that
is divergent by construction, two streams of 256 is not distinguishable from the
batch-dependence already being measured, and one observation is not an interval. It wants
a second ragged run before anyone reads anything into it.

The weight was calibrated at both ends on the same hardware. At 2.0 the positive control
recovered `risdongram` → `risdon graeme`, `archie` → `archy` and `i'm` → `i am`, and an
unrelated list inserted nothing. At the ceiling of 10 the domain list moved 12 transcripts
of 12 and inserted `certiorari`, `metformin` and `tortious interference` into audio that
merely sounds like them. That is what makes 2.0 a default rather than a guess, and it is
also the reason the ceiling exists.
