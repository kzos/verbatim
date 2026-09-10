# `probes/` — the scripts behind every number in the README's batch-invariance section

**These are not benchmark rows and never will be.** A row under `rows/` comes from the frozen methodology
in [`benchmarks/METHODOLOGY.md`](../benchmarks/METHODOLOGY.md), with an environment record collected by
the harness. These are exploratory scripts, run by hand, with no environment record and no frozen
definitions. They exist so that a sceptical reader can re-run the claims rather than take them, which is
the only reason those claims are in the README at all.

They also do not test Verbatim. **Every one of them drives stock NeMo**, because Verbatim is not
implemented. What they establish is that the defect this server is being built to fix is real.

## What each one answers

| script | question | evidence it produced |
|---|---|---|
| `divergence_hunt.py` | Does batch composition change the transcript, offline, on LibriSpeech? | 6 in 5,559 across both test splits |
| `divergence_hunt_fleurs.py` | Does it happen on a second corpus? | 1 in 647 on FLEURS `en_us` |
| `streaming_divergence_repro.py` | Does it happen on the cache-aware **streaming** path? | 38 in 2,939, every one at the tail |
| `streaming_divergence_controlled.py` | Does padding every row to one length remove it? | 38 becomes 1 |
| `deep_divergence.py` | Which channel does it actually reach: text, tokens, timestamps or score? | word timing 35, text 4, of 2,912 |
| `negative_control.py` | Is any of this just run-to-run noise? Does row position matter? | both arms zero |
| `duration_mechanism.py` | Is the length-coupling explanation a measured relationship? | only weakly; see the caveat below |
| `nemo_biasing_repro.py` | The upstream registry defects, reproduced without a checkpoint | filed as NVIDIA-NeMo/Speech#16236 |

## Running them

```bash
pip install torch nemo_toolkit[asr] datasets soundfile
python probes/negative_control.py 1024
```

Each writes JSON to `probe-output/` beside where you run it; set `PROBE_OUT` to send it elsewhere. They
stream their corpora from the Hugging Face hub, so the first run downloads. A full 2,939-utterance sweep
took about six minutes on an RTX A6000 and about three on a B300.

`negative_control.py` is the one to run first. If its repeat arm is not zero on your hardware, nothing
else here means what it says on yours.

## Read these caveats before quoting any of it

- **Counts, not rates.** Seven divergences in 6,206 utterances is a count with a denominator. It does not
  support a frequency, and converting it into one prints a number nobody measured.
- **`duration_mechanism.py` partly refutes the explanation it was written to confirm.** Divergent
  utterances sit at about the 61st percentile of the duration-deficit distribution, not the 90th. The
  direction survives, the strength of the story does not, and the README says so.
- **Two machines, two software stacks.** The Ampere and Blackwell results differ in silicon *and* in
  torch and NeMo versions. Nothing here attributes a difference to an architecture.
- **Everything ran eager on the A6000.** Its driver refuses NeMo's graphed decoding.
- **`deep_divergence.py` needs `timestamps=True` on the transcribe call**, not a decoding-config field.
  Two earlier attempts set the config and silently got no timestamps at all, which is why the run that
  finally produced them is the one that matters.
