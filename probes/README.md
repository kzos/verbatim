# `probes/` — the scripts behind every number in the README's batch-invariance section

**These are not benchmark rows and never will be.** A row under `rows/` comes from the frozen methodology
in [`benchmarks/METHODOLOGY.md`](../benchmarks/METHODOLOGY.md), with an environment record collected by
the harness. These are exploratory scripts, run by hand, with no environment record and no frozen
definitions. They exist so that a sceptical reader can re-run the claims rather than take them, which is
the only reason those claims are in the README at all.

They also do not test Verbatim. **Every one of them drives stock NeMo.** That was originally because
Verbatim did not exist; it is now a deliberate property, because a measurement of the defect this server
is built to fix must not depend on the server. What they establish is that the defect is real.

## What each one answers

| script | question | evidence it produced |
|---|---|---|
| `divergence_hunt.py` | Does batch composition change the transcript, offline, on LibriSpeech? | 6 in 5,559 across both test splits |
| `divergence_hunt_fleurs.py` | Does it happen on a second corpus? | 1 in 647 on FLEURS `en_us` |
| `streaming_divergence_repro.py` | Does it happen on the cache-aware **streaming** path? | 38 in 2,939, every one at the tail — **but see the next row: this is NeMo's example path, not the server's** |
| `streaming_divergence_controlled.py` | Does padding every row to one length remove it? | 38 becomes 1, on the example path |
| `streaming_divergence_pipeline.py` | The same question on the path a **server** runs, `CacheAwareRNNTPipeline` | float32: **6 in 2,939, none at the tail**. bfloat16: **287**, and equalising lengths does not help (**328**) |
| `deep_divergence.py` | Which channel does it actually reach: text, tokens, timestamps or score? | word timing 35, text 4, of 2,912 |
| `negative_control.py` | Is any of this just run-to-run noise? Does row position matter? | both arms zero |
| `duration_mechanism.py` | Is the length-coupling explanation a measured relationship? | only weakly; see the caveat below |
| `nemo_biasing_repro.py` | The upstream registry defects, reproduced without a checkpoint | filed as NVIDIA-NeMo/Speech#16236 |
| `fixed_shape_contents.py` | With the batch shape pinned, do the batch contents reach the target's output? | 0 text and 0 timing differences in 1,024 |

## Running them

```bash
pip install torch nemo_toolkit[asr] datasets soundfile
python probes/negative_control.py 1024
```

Each writes JSON to `probe-output/` beside where you run it; set `PROBE_OUT` to send it elsewhere. They
stream their corpora from the Hugging Face hub, so the first run downloads. A full 2,939-utterance sweep
took about six minutes on an RTX A6000 and about three on a B300.

`negative_control.py` is the one to run first, and `fixed_shape_contents.py` is the one that tests the
mechanism this server is built on rather than the defect it exists to prevent.

Not every claim on the front page comes from a script here: the encoder-level results in the first
bullets, and the graphed-versus-eager comparison on the B300, were run from scratch files that were not
kept. Their raw outputs are cited where they are used, and re-running them is open work.

`negative_control.py` is the one to run first. If its repeat arm is not zero on your hardware, nothing
else here means what it says on yours.

## Read these caveats before quoting any of it

- **Counts, not rates.** Seven divergences in 6,206 utterances is a count with a denominator. It does not
  support a frequency, and converting it into one prints a number nobody measured.
- **`duration_mechanism.py` partly refutes the explanation it was written to confirm.** Divergent
  utterances sit at about the 61st percentile of the duration-deficit distribution, not the 90th. The
  direction survives, the strength of the story does not, and the README says so.
- **The architecture claim covers transcripts only.** Running the newer stack on the older card
  separated silicon from software: transcript divergence went 4, 5, 0 across old-stack Ampere,
  new-stack Ampere and new-stack Blackwell, so it tracks the hardware. Word timing went 35 to 27 on the
  same card with only the software changed, so it does not, and the Blackwell timing channel has never
  been measured.
- **The streaming probes carry no machine stamp.** Two result files from different cards are
  distinguishable only by elapsed time. Re-runs should stamp the device, torch and NeMo version the way
  `deep_divergence.py` does.
- **Everything ran eager on the A6000.** Its driver refuses NeMo's graphed decoding.
- **`deep_divergence.py` needs `timestamps=True` on the transcribe call**, not a decoding-config field.
  Two earlier attempts set the config and silently got no timestamps at all, which is why the run that
  finally produced them is the one that matters.
