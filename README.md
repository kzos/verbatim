# Verbatim

**An Apache-2.0, asynchronous, multi-client server for NVIDIA's cache-aware streaming speech pipelines. No API key.**

> ### There is no working server here yet
>
> Both wires now run every session through the tick-scheduled engine, with admission, live back-pressure
> and an idle deadline, over a CPU fake. The adapter onto NeMo exists and **has never run on a GPU**, so
> nothing here has yet transcribed real audio through the real pipeline, and the command-line tool still
> builds nothing and prints `not implemented yet`. Nothing in this repository has been benchmarked.
>
> What is finished and worth your time is the **evidence**: a measured account of how batch composition
> changes a transcript and a word's timing on stock NeMo, with the probe scripts under
> [`probes/`](probes/) so you can re-run every number rather than believe it. The benchmark harness and
> its frozen methodology are real too. The server is the argument those measurements make, and it is
> being built in the open.

**License:** Apache-2.0 (see `LICENSE`)

> Verbatim has not been benchmarked. Every number describing Verbatim's own behaviour in this README is an angle-bracketed placeholder until a row under `rows/` fills it. The batch-invariance section below reports exploratory probes on stock NeMo, not on Verbatim, which is not implemented. Two external figures are cited to their sources: the checkpoint's monthly download count and the CUDA-graph speedup range; neither is a measurement of this project.
>
> [KILL.md](KILL.md) — the dated conditions under which this stops, written before the runs.<br>
> [docs/SCOPE.md](docs/SCOPE.md) — fixed boundary and growth trigger.<br>
> [docs/not-here.md](docs/not-here.md) — common contribution refusals with redirects.<br>
> [CONTRIBUTING.md](CONTRIBUTING.md) — contribution units and measurement rules.

## The problem

NVIDIA's cache-aware streaming speech stack is open and fast, and it has no server. NeMo's
Apache-2.0 `nemo.collections.asr.inference` package already contains the multi-stream slot manager,
dynamic stream add/remove, endpointing and — since [NeMo PR #15863](https://github.com/NVIDIA-NeMo/Speech/pull/15863)
(merged 2026-08-12) — a CUDA-graph encoder step worth **3.08–5.14x** at the 80 ms chunk mode (upstream's figure, and **not yet in a released wheel**: checked 2026-09-11, neither NeMo 2.7.3 nor 3.0.0 carries it, only the source tree — [DR-0002](docs/decisions/0002-the-graph-path-is-not-in-a-released-wheel.md)), because
low-latency streaming "spends most of its time waiting on the host": each step launches "around 1.5k
small kernels, so the GPU is idle for most of the step while the host enqueues them." But that package
has zero occurrences of `asyncio`, `websocket` or `grpc`; its entry point takes a file, a directory or
a manifest and writes JSON. So the checkpoint everyone is downloading —
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
**942,647 downloads in the month to 2026-09-07, per the Hugging Face model card** — has no open multi-client server behind it: NVIDIA's own open
runtimes are batch-1 or a bounded four-thread pool, and NeMo-Speech.cpp's docs say plainly that
"NVIDIA NIM is the supported production deployment path" — today, production means the paid path.

## The design

Verbatim is a **tick scheduler**: sessions join and leave every chunk period, but the batch handed to
NeMo is always one of a few fixed bucket sizes whose CUDA graphs stay captured, and the ragged
final steps are peeled into a small eager side-batch — so a churning multi-tenant workload never
falls off the graph path that PR #15863 opened. The property it is built for is **batch invariance**:
one session's transcript and word timestamps identical whether that session is alone on the GPU or
sharing it. Whether that property holds on the graph path is an open question with a dated gate, and
the section below says exactly what has been measured and what has not.

## Batch invariance: what is measured, and what is not

Three things have been measured in exploratory probes on stock NeMo, not on Verbatim, which is not
implemented. The probes used `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` — a cache-aware
streaming FastConformer with a hybrid RNNT decoder, loaded through NeMo — on an RTX A6000. They are not
rows under `rows/`, and no number below is a benchmark result. None of it transfers to another checkpoint
or another card without being run again. The transcript probe used two concurrencies: one session
(concurrency 1) and batch 32. Both calls used the same audio, weights and decode path. Every utterance in
each group was zero-padded to one common length before either call, so batch size was the only variable.

- Running one session's audio inside a batch with other sessions changes the encoder's floating-point
  output, against running it alone. It was present at the smallest concurrency tested, 8, and does not
  require load to provoke it.
- Padding every step to a fixed batch shape restores bit-identical output at every occupancy tested.
  Turning TF32 off raises precision and does not fix the problem, so precision is not the lever. Shape is.
- **That numerical difference reaches the transcript.** The superseding run found seven transcript
  divergences across the three corpora below.

| corpus | utterances | divergences |
|---|---:|---:|
| LibriSpeech `test-other` | 2,939 | 4 |
| LibriSpeech `test-clean` | 2,620 | 2 |
| FLEURS `en_us` | 647 | 1 |
| **total** | **6,206** | **7** |

The two LibriSpeech divergences already printed in the earlier report are shown here with their
recovered references:

```text
LibriSpeech 8280-266249-0049
  reference   ... then mister LILBURN   waking from his first sleep in a stateroom near by ...
  alone       ... then mister LOBBOURNE waking from his first sleep in a stateroom near by ...
  in batch 32 ... then mister LILBURNE  waking from his first sleep in a stateroom near by ...

LibriSpeech 533-131562-0010
  reference   ... and as for the household matters missus GREAVES must be very particular ...
  alone       ... and as for the household matters missus CRIBBS  must be very particular ...
  in batch 32 ... and as for the household matters missus GRIEVES must be very particular ...
```

Neither hypothesis is the reference in either utterance, so neither is the true transcript. Both
conditions are wrong in both utterances, and word error rate is identical between them: one error in
twenty-one words for the first and three in twenty-four for the second.


On a synthetic FastConformer-shaped micro-benchmark, not on this checkpoint, turning TF32 off shrank the
difference and made it appear sooner. Precision is not the lever. Shape is.

The seven divergences are not all proper nouns: three are, and the other four are not. One two-word span
`an extra` was heard as `a natural` or `an answer`; the article `a` was heard as `it`; `ranged` was heard
as `arranged`; and the FLEURS deletion below is the fourth non-name case. Every divergence sat at a token
the model was already unsure of, and in most cases the other condition got that token wrong too. Proper
nouns are over-represented relative to their share of the text, but this seven-case sample does not
quantify that over-representation.

The direction is mixed. Three cases tie on word error rate. In one the batched hypothesis is better by
one word. In one the batched hypothesis is exactly the reference and running alone introduced the only
error. In one the reverse. Over the six LibriSpeech divergences, twelve word errors alone against eleven
batched; adding the FLEURS case makes it twenty-three against twenty-one over all seven. This is a comparison over seven hand-picked utterances, not a corpus word error rate; the
utterances that did not diverge are not in it.

The FLEURS divergence is a deletion, not merely a token substitution. In FLEURS `en_us`, utterance id
`1695`, a five-word span containing a proper noun is absent entirely when the utterance is decoded alone
and partly recovered when it is decoded in a batch of thirty-two:

```text
FLEURS en_us 1695
  reference   ... during the summer is THE PA AMB OLI BREAD WITH olive oil tomato and any available ...
  alone       ... during the summer is                          olive oil tomato on any available ...
  in batch 32 ... during the summer is                     BRED  olive oil tomato on any available ...
```

Both hypotheses are poor on this utterance, with eleven word errors alone and ten in batch 32 out of
twenty-seven, so this is not a case of one arm being right. Batch composition changed whether that span
was emitted at all, not merely which token was emitted. It matters beyond its count: a deletion is the one
failure mode that phrase boosting provably cannot repair on the greedy decoder, because the choice to emit
nothing is taken before any boost is applied and is then preserved. The path that can reach it is the
batched beam decoder, where the fusion score enters before pruning.

**Seven divergences in 6,206 utterances is a count with a denominator, not a rate, and this README will
not convert it into one.** Seven events do not support a frequency, a percentage or a probability, and any
such figure printed here would be a number this project did not measure. This run covered three corpora,
one card and the two concurrencies named above, under an exploratory script rather than the frozen
methodology the gate requires. The figure this project publishes comes from the day-45 gate, or it does
not get published.

Five things have **not** been measured, and the run above settles none of them.

- **How often it happens.** See above: a count, not a rate.
- ~~**Word timestamps.**~~ Measured now, and they are the most exposed channel of all. See below.
- **That the fix holds at the level of text.** Fixed-shape batching gives bit-identical encoder output at
  every occupancy tested, and identical output cannot decode to different text — but the corpus above has
  not been re-run with the bucket in place, so the fix is demonstrated on the numbers rather than on the
  transcripts.
- **The graph path, on this card.** The A6000's driver refuses NeMo's graphed decoding, so everything
  measured here ran eager. On a B300, where graphs do capture, the graphed and eager arms both returned
  zero transcript divergences over 2,912 utterances. That is one card and one corpus, and it does not
  settle the question that decides whether there is a server here at all.
- **Whether either effect compounds over a real session.** Both runs decide one utterance at a time. A
  live session re-rolls its batch composition every tick for the length of a call, and whether divergences
  are independent across ticks is not known. A per-utterance count does not convert into a per-session one.

### The streaming path shows a second, larger effect

The probe above used the offline transcription path. Re-running it through the cache-aware streaming
pipeline, chunk by chunk, turns up something the offline path cannot see. Over the same 2,939 `test-other`
utterances, at the same two concurrencies:

| run | control | divergences |
|---|---|---:|
| streaming, no length control | none | 38 |
| streaming, every row padded to one common length | lengths equal in both arms | 1 |

**All 38 differ only at or after the last word the two arms share. Not one differs mid-sentence.**

```text
alone       ... the remnant of the philistines shall per
in batch 32 ... the remnant of the philistines shall perish     (the reference says "perish")
```

The cause is not arithmetic. A short session streaming beside longer neighbours keeps receiving trailing
chunks after its own audio ends, and its decoder keeps emitting. Alone, the buffer empties at its own
length and the tail is dropped. **So the transcript depends on how long the other sessions in the batch
are** — and padding every row to a common length, which is what fixed-shape batching does, takes it from
38 to 1.

That leaves two distinct effects, and the common one is not the exotic one:

- **Length coupling**, 38 in 2,939, on the path a server actually runs.
- **Numerical perturbation**, 1 in 2,939, which survives the length control. The single
  divergence that survives is the same utterance, with the same two hypotheses, that the offline run
  found. Two decode paths reaching the same pair of wrong answers for the same token is better evidence
  for that effect than either run alone.

### Word timestamps are the channel this reaches most often

Every probe above compared transcript text. The decoder also returns the emitted token ids, per-word and
per-character timestamps, and a hypothesis score, and none of that had ever been looked at. Since this
server promises transcripts **and** word timestamps, half its claim had no evidence behind it.

Over 2,912 `test-other` utterances on the A6000, padded control, one session against a batch of 32:

| channel compared | utterances differing |
|---|---:|
| transcript text | 4 |
| token ids | 4 |
| **word timestamps** | **35**, of which **31** have identical text |
| segment timestamps | 8 |
| hypothesis score, bit-identical | 3 of 2,912 |

**A word's timing moves without the word changing 31 times, against 4 times that a word itself
changes.** The 35 above includes those 4, whose timings differ because the words do. In every case exactly one
word moves, by between one and five frames; at 80 ms per frame that is 80 ms in most cases and 400 ms in
the largest. A reader of words sees nothing. Anything that cuts subtitles, aligns to picture, or feeds a
diarisation stage gets a different answer.

The score is worth one sentence and no more. It is an accumulated log-probability over hundreds of steps,
so any perturbation anywhere changes it, and it differs in all but three utterances. That is arithmetic,
not a finding, and it is reported here only so nobody mistakes its absence for stability.

### Two controls, and both come back clean

Every count on this page attributes a difference to batch composition. That is only sound if an identical
call, repeated, gives an identical answer. Over 1,024 utterances, comparing text and word timing together:

| what changes between the two calls | text | word timing |
|---|---:|---:|
| nothing, the identical call twice | 0 | 0 |
| alone against a batch of 32 | 2 | 9 |
| same 32 members, same shapes, rows permuted | 0 | 0 |

The first row is the control that makes the rest mean anything: **the decode is reproducible run to run**,
so these are measurements of batch composition and not of GPU noise.

The third row settles a design question. A session's row index inside a batch is its rank among the streams
live in that tick, so it changes whenever a neighbour joins or leaves. Had permuting rows changed anything,
fixed-shape batching would not have been enough and the scheduler would have had to pin positions too. It
does not, so it does not.

### The same experiment on two architectures, with the software held fixed

| experiment | A6000, torch 2.6 / NeMo 2.7.3 | A6000, torch 2.11 / NeMo 3.0.0 | B300, torch 2.11 / NeMo 3.0.0 |
|---|---:|---:|---:|
| transcript, offline padded control | 4 in 2,912 | 5 in 2,912 | **0 in 2,912** |
| word timing, offline padded control | 35 | 27 | not measured |
| streaming, no length control | 38 in 2,939 | — | 38 in 2,939 |
| streaming, length-controlled | 1 in 2,939 | — | 0 in 2,939 |

**Transcript divergence tracks the silicon.** Holding the software fixed and changing only the card takes
it from 5 to 0. Holding the card fixed and changing the whole stack leaves it at 4 against 5. That is a
property of the hardware, and it is consistent with the mechanism: a linear-algebra library choosing
different kernels, and therefore different reduction orders, for different batch shapes on the
architecture it was tuned for.

**Word timing does not track the silicon, and this is the more honest half.** The same card with a
different stack moves it from 35 to 27. Software matters for the channel this page calls the most
exposed, and the B300's timing channel has never been measured at all. So the sentence that survives is
narrow: *transcript* divergence is architecture-dependent, and nothing here says anything about
timestamps on Blackwell.

**Length coupling reproduces at exactly 38 on both cards**, which is what a deterministic algorithmic
artefact looks like rather than an arithmetic one. Two caveats travel with that number. The probe drives
NeMo's example streaming path, not the inference pipeline this server is designed to wrap, and those are
different code with different boundary handling — see the honesty note below. And the two result files
carry no machine stamp, so they are indistinguishable except by elapsed time; the runs were separate but
the artefacts cannot prove it, and the next run fixes that.

### Fixed shape, different neighbours

Every table above compares batch 1 against batch 32, which changes the batch *shape*. The claim this
server rests on is stronger: that once the shape is pinned, the batch *contents* stop mattering. That had
never been tested past the encoder.

Batch size held at 32 in both arms, every row padded to one common length, the target in row 0. The only
difference is who the other 31 rows are.

| the other 31 rows are | target's text differs | target's word timing differs |
|---|---:|---:|
| silence | — | — |
| real utterances | **0 in 1,024** | **0 in 1,024** |

Zero on both channels. As a positive control, the composition arm of the same corpus — one session
against a batch of 32, which changes the shape — moved text twice and word timing nine times in the same
1,024 utterances. The instrument could detect a difference and did not find one here.

That is the first evidence that fixed-shape batching is sufficient at the level of output rather than of
encoder floats. It is one card, one checkpoint, the offline path, and 1,024 utterances.

### An honesty note about which code path this measures

The streaming probes drive NeMo's example streaming path, stepping the model chunk by chunk through the
streaming audio buffer. This server is designed to wrap a different implementation in the same
repository, `nemo.collections.asr.inference`, which computes per-stream right paddings and carries an
explicit flag for whether tokens past the clip boundary are returned. Those are different code with
different boundary handling.

So the 38 is a real measurement of the example path and **not** evidence about the pipeline this server
wraps. The re-run against the inference pipeline is the next experiment. Until it lands, read the length
coupling result as a property of NeMo's example driver, and read the transcript and timestamp results,
which use the offline path, as unaffected by this caveat.

One count against four is not a difference these numbers can resolve; treat both paths as showing the same
rare numerical effect. The honest summary is that **fixed-shape batching addresses both, and the effect it
addresses most often is the ordinary one.**

So the claim, at its full and honest width: **the claim is irreproducibility, not degradation.** Batch
composition changes what a user reads on real speech and a real cache-aware streaming checkpoint, but this
probe does not show that batching harms or improves accuracy. Neither arm should be called more accurate
when the hypotheses differ. Read this as observed irreproducibility, not as a known accuracy delta or rate.

A dated invariance gate is what settles it, over 1,000 LibriSpeech test-other
and 500 FLEURS utterances, comparing transcripts and timestamps exactly. That run carries both
arms, bucketed and ragged; the table's rows read the ragged arm, with bucketed as the control. Three
outcomes and what each one means, written down before the run rather than after:

| what the ragged arm finds | what this project then claims |
|---|---|
| transcripts differ at any rate | the claim above, as written |
| none across those 1,500 utterances | a bounded rate, not zero; the guarantee is worth having because it cannot be ruled out, not because it is common |
| none across a far larger corpus | the case for this server is weak, and that gets published as plainly as any other result |

The middle row is a harder outcome to reach than it was when it was written, and that is what writing it
down in advance is for: if the gate finds nothing where this probe found seven, the disagreement is itself
the finding and it gets chased before anything is published.

This section is provisional and will be rewritten from measured rows, not edited to fit them.

## Demo

> **None of this runs yet.** The commands below are the interface this server is being built toward, not
> something you can execute today: the console script exists and prints `not implemented yet`. They are
> here so the shape of the thing is arguable before it is built, and so the acceptance criteria are fixed
> in public rather than chosen afterwards. The distribution name is also unsettled — see
> [docs/ISSUES.md](docs/ISSUES.md) — **`pip install verbatim` today installs an unrelated and actively
> maintained speech-to-text project by a different author, not this one.** Everything measured on this
> page was measured against **stock NeMo**
> with `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`, not against the checkpoint the demo
> advertises and not against Verbatim, which does not exist yet.

```bash
pip install verbatim
verbatim serve nvidia/nemotron-3.5-asr-streaming-0.6b --chunk 160ms
```

```text
[verbatim] checkpoint   nvidia/nemotron-3.5-asr-streaming-0.6b  (NeMo cache-aware RNNT)
[verbatim] device       NVIDIA RTX A6000  ·  sm_86  ·  48 GB
[verbatim] chunk mode   160 ms   (att_context_size [56, 1], fp32)
[verbatim] graphs       steady bucket B=<N> captured at warm-up (fixed-shape default) + eager edge batch B_edge=8  (1 of max_graphs)
[verbatim] admission    ceiling <N> streams @ p95 <= 310 ms  (measured tick budget, calibrated on this GPU — never typed in)
[verbatim] riva  grpc   0.0.0.0:50051   StreamingRecognize (Riva-compatible subset)
[verbatim] websocket    0.0.0.0:8080    /v1/stream  ·  demo page at /  ·  /healthz /readyz /metrics /admission
```

Then, on the same GPU, put your own microphone in the same batch as `<N>` replayed LibriSpeech test-other
sessions — `<N>` being the ceiling the command has just calibrated on *your* card, not a number typed in:

```bash
verbatim demo nvidia/nemotron-3.5-asr-streaming-0.6b --chunk 160ms
```

```text
concurrency <N>/<N>   tick 160 ms   graphed steps <g> %   eager side-batch <e> %   (target: eager <= 2 %)

  streams (RTX A6000, 160 ms)  <N>        <F>x of NeMo graphed file-driven ceiling   (MVP target: >= 0.8x)
  p95 partial latency          <p95> ms   budget: chunk + 150 ms = 310 ms
  WER, LibriSpeech test-other  <W> %      NeMo batch-1 streaming reference: <W_ref> %  (target: within 0.1 abs)
  batch invariance 1/32/<N>    PASS/FAIL  transcript diffs: <d_t>, timestamp diffs: <d_ts>  (target: 0 and 0)

  your mic  ->  "the quick brown fox jumps over the lazy dog"
  same text, byte for byte, at 1 session and at <N>.
```

> **Placeholders, filled only from a measured row.** `<N>`, `<F>`, `<p95>`, `<W>`, `<g>` and `<e>` are angle-bracketed
> because nothing has been measured yet; the README fills them from a row directory under `rows/` and never by
> hand. The targets beside them are the MVP bar, not results.
> The ceiling they are a fraction of is NeMo's own `asr_streaming_infer.py` with
> `use_cuda_graphs=true`, measured on the same GPU, same checkpoint, same chunk mode; that ceiling
> and these five rows are published together, on an RTX A6000 and a B300, pass or fail.
>
> **The ratio is a property of the box, not of the architecture.** A slower die lowers the ceiling and
> so raises the ratio for anything that is host-bound rather than GPU-bound. Every row therefore names
> its die, and a result on one die is never reported as a result on another.

## Installation

```bash
pip install verbatim
```

- **Python** 3.11–3.13 (the range the wheel is tested on).
- **CUDA** 12.9 or 13.2 — the two tracks NeMo itself ships (`nemo-toolkit[cu12]` → torch `cu129`,
  `[cu13]` → torch `cu132`). These cover `sm_86` (RTX A6000); confirm the SM version of any Blackwell
  datacenter part on the box itself before relying on a prebuilt wheel. The `cu128` torch index
  stops at torch 2.11.0 and is not a target; a CUDA 12.8 driver runs the cu129 wheel under
  minor-version compatibility.
- **One NVIDIA GPU.** Verbatim is single-GPU by design; run one process per GPU.
- **Model.** Downloaded from Hugging Face on first start —
  `nvidia/nemotron-3.5-asr-streaming-0.6b` or `nvidia/nemotron-speech-streaming-en-0.6b`, or any
  cache-aware FastConformer checkpoint NeMo loads. Set `HF_HOME` to control the cache.
- **NeMo.** The model runtime is an extra, not a default dependency: `pip install 'verbatim[nemo]'`.
  The CUDA-graph encoder step this server is built around (PR #15863) merged on 2026-08-12, five days
  *after* the latest NeMo release (3.0.0, 2026-08-07), so as of 2026-09-07 **no released NeMo wheel
  carries it**. `verbatim doctor` reports which track you are on and refuses to start the graph path —
  rather than silently running eager — when the installed NeMo lacks it.
- **Ports.** `50051` — the Riva-compatible gRPC `StreamingRecognize` subset, on the Riva convention
  (`livekit-plugins-nvidia` and Pipecat's `nvidia` STT service take any `host:port`; NeMo-Speech.cpp's
  `riva_server` binds `0.0.0.0:50051` by default). `8080` — the plain WebSocket for the demo
  (`/v1/stream`), the demo page, and `/healthz`, `/readyz`, `/metrics`, `/admission`. No other listener.

```bash
verbatim serve nvidia/nemotron-3.5-asr-streaming-0.6b \
  --chunk 160ms --grpc-port 50051 --ws-port 8080
```

## What Verbatim is not

- Whisper, in any mode (SGLang / vLLM / faster-whisper own it)
- AED decoders and Canary (NeMo, vLLM)
- paged encoder caches (unnecessary: fixed-size state, NeMo's slot table)
- CuTe/CUTLASS encoder kernels (the streaming step is host-launch-bound rather than kernel-bound — NeMo
  #15863 reports roughly 1.5k small kernels per step and a 3.08-5.14x range from graph capture — so kernel
  work is not where the cost sits; the split is not measured here)
- FP8/NVFP4 encoders (fp32 models; graphs skip autocast; NeMo #16147 is doing mixed precision)
- speculative decoding
- OpenAI-Realtime and Deepgram protocol compatibility
- LiveKit/Pipecat plugin packages (the Riva path replaces them)
- diarization, PII, translation, TTS and speech-to-speech in year 1 (NeMo pipelines; the growth path)
- a control plane, dashboards or a hosted service
- CPU/edge/mobile runtimes (NeMo-Speech.cpp, parakeet.cpp, sherpa-onnx)
- ROCm
- any re-implementation of NeMo's pipeline

## Status

**Pre-alpha, and openly conditional.** Dated gates (clock anchored 2026-09-14) decide whether this
becomes a project: by **day 21 (2026-10-05)** a five-arm head-to-head on one RTX A6000 must put Verbatim
at **>= 0.7x** NeMo's graphed file-driven ceiling in streams at equal WER, and must not find an existing
open server already at **>= 0.5x** it; by **day 45 (2026-10-29)** the invariance gate must be
bit-identical at concurrency 1 / 32 / max over 1,000 LibriSpeech test-other and 500 FLEURS utterances
with <= 2% of steps eager, or the server is dropped and only the harness survives; by **day 60
(2026-11-13)** `livekit-plugins-nvidia` and Pipecat's `nvidia` STT service must complete live calls
against Verbatim with zero plugin code at <= chunk + 150 ms p95 at the published ceiling, or the fallback
is WebSocket-only and the case for the gRPC surface is materially weaker; by **day 90 (2026-12-13)** there must be a named installer
at >= 100 concurrent streams reporting numbers publicly, an NVIDIA mention, or three outside
contributors with merged measurement rows. If any gate fails — or if NVIDIA ships its own Apache-2.0
asynchronous multi-client batched server, or vLLM merges a Nemotron/Parakeet streaming model with
cross-session batching (the day-60 absorption watch) — the table is published anyway, the README records
the outcome, and work stops at a maintained utility rather than pretending otherwise. Each verdict is
recorded on its date, whatever it says.

## Licence

Apache-2.0. Model checkpoints carry their own licences from Hugging Face
(`nvidia/nemotron-3.5-asr-streaming-0.6b` and `nvidia/nemotron-speech-streaming-en-0.6b` are released
under **OpenMDW-1.1** — `license: other`, `license_name: openmdw-1.1` on the card — not Apache-2.0);
Verbatim downloads them, it does not redistribute them.

---

## Where this repository is today

Nothing above is implemented as a running server. What exists is the module tree as importable,
documented placeholders; the vendored Riva protos and their generated stubs; the benchmark harness with
its frozen methodology; and the probes under [`probes/`](probes/) that produced most of the measurements on
this page, with the exceptions named in that directory. Start with [`docs/SCOPE.md`](docs/SCOPE.md) for the boundary and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how a change lands.

Every numeral in the demo blocks above is an angle-bracketed placeholder for exactly the reason
`CONTRIBUTING.md` gives: **No number that the harness did not produce**, anywhere, ever.

- [`docs/third_party.md`](docs/third_party.md) — the vendored Riva protos: source, licence, pinned SHA.
- [`docs/protocols/riva.md`](docs/protocols/riva.md) — the supported-subset table (placeholder; it will
  be generated from `protocols/riva/conformance.py`).
