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

## The default weight, and the correction to it

**This section previously said the default of 2.0 was "a default rather than a guess". It
was a guess, and a better instrument has since falsified it.** The evidence was two
readings at opposite extremes on a handful of clips: at 2.0 the positive control recovered
`risdongram` → `risdon graeme`, `archie` → `archy` and `i'm` → `i am`, and a six-word
unrelated list inserted nothing into one clip; at the ceiling of 10 the domain list moved
12 transcripts of 12 and inserted `certiorari`, `metformin` and `tortious interference`
into audio that merely sounds like them. Two points and a gap between them is not a curve.

`verbatim-bench rare-terms` measured the curve: 256 utterances, 256 terms derived from the
corpus's own references by document frequency, the whole list sent to every session, one
pass per weight. B300, 2026-09-15.

| weight | recall | hits | missed | false accepts | precision | WER |
|---|---|---|---|---|---|---|
| bare | 0.911 | 266 | 26 | 3 | 0.989 | 0.0739 |
| **1.0** | **0.945** | **276** | **16** | **17** | **0.942** | **0.0737** |
| 2.0 | 0.942 | 275 | 17 | 105 | 0.724 | 0.0911 |
| 4.0 | 0.921 | 269 | 23 | 591 | 0.313 | 0.2543 |
| 10.0 | 0.438 | 128 | 164 | 1867 | 0.064 | 0.8175 |

**The knee is at 1.0 and the shipped default is now 1.0.** It recovers nine terms
(`baghdad`, `forelock`, `orficer`, `weevilly` among them), loses none, costs fourteen
false accepts and leaves word error rate where it was. **2.0 buys one occurrence less
than 1.0 and pays six times the false accepts and 23 percent relative word error rate.**
By 4.0 the transcript is coming apart, and at 10.0 recall falls *below bare*: the decoder
emits list words so freely that it loses the real occurrences too.

Two things this also says about the design:

**The ceiling of 10 is not a safety margin.** It is past the point where the feature
inverts. It stays as a refusal boundary because a client asking for 20 has misunderstood
the scale, but nothing between 4 and 10 is a usable setting and the documentation should
not imply otherwise.

**The gate's negative control is a smoke test, not the measurement.** It reported zero
insertions at 2.0 — from one clip and a six-word list, while the corpus-wide reading at
the same weight shows 105 false accepts. It was not wrong; it was underpowered by three
orders of magnitude of exposure, and read as reassurance it should not have given. It now
runs over every control clip and records its exposure so its power is visible rather than
assumed. The instrument for false accepts is `rare-terms`.

**What biasing actually buys, stated plainly:** the model already gets 91.1 percent of
rare terms right unaided. Boosting at the knee closes **ten of the twenty-six missing
occurrences — 38 percent of the gap — at roughly one false accept per word recovered.**
That is a real gain and a modest one, and it is under the half-the-gap line the beam-search
trigger names. Whether the residual sixteen are deletions rather than substitutions is the
other half of that trigger and has not been checked; it is the next measurement, not a
claim.

The invariance arms in the table above were run at 2.0. They are unaffected: the gate
compares digests across batch compositions and is not an accuracy measurement. A re-run at
1.0 would produce a different digest and the same verdict.
