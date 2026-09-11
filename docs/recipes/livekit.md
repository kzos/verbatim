# Recipe: LiveKit agents against Verbatim

**There is no Verbatim plugin, and there will not be one.**
[`docs/not-here.md`](../not-here.md) rules plugin packages out: a plugin package belongs with the
framework that loads it, not with the server it points at. The artefact here is a recipe, and the
day-60 gate is a live call with **zero plugin code**.

That is possible because Verbatim serves the Riva-compatible gRPC subset those clients already speak,
so `livekit.plugins.nvidia.STT` connects to it unchanged. Three arguments do the work: the server
address, `use_ssl=False`, and a model name Verbatim actually serves.

## The example

[`examples/livekit/verbatim_livekit_stt.py`](../../examples/livekit/verbatim_livekit_stt.py) drives the
real plugin against a Verbatim server and prints the events it emits. Nothing in it subclasses,
patches or wraps the plugin.

Without a GPU, the fake pipeline serves scripted transcripts over the same wire, which is enough to
prove the protocol:

```bash
verbatim serve nvidia/stt_en_fastconformer_hybrid_large_streaming_multi \
    --pipeline fake --bucket 8 --grpc-port 50051
```

```bash
uv run --with livekit-plugins-nvidia --with livekit-agents --with soundfile \
    examples/livekit/verbatim_livekit_stt.py --server 127.0.0.1:50051 \
    --model nvidia/stt_en_fastconformer_hybrid_large_streaming_multi audio.wav
```

Against a real pipeline, drop `--pipeline fake` and give the server a GPU. The client command is
unchanged, which is the point.

## Two interpreters, and why

`nvidia-riva-client` pins protobuf 6.33.5 and Verbatim's vendored stubs need the 7.36 runtime. They
cannot share an interpreter, so the recipe runs the plugin in its own `uv` environment and the server
in the project's. This is a real constraint of the two packages rather than a workaround, and any
deployment putting both in one process will meet it.

## What the plugin sends, and what Verbatim does with it

- **`function-id` metadata, and `authorization` when an API key is set.** Accepted and ignored.
- **16 kHz mono PCM16**, interim results and word times requested. Answered with partials, then a
  final whose words carry millisecond offsets the plugin divides by a thousand.
- **A model name the server does not serve** is refused as `NOT_FOUND`. Nothing is substituted,
  because the plugin's default names NVIDIA's hosted model and a silent substitution would report a
  transcript from a model the caller did not ask for.

## The proof

[`tests/examples/test_livekit_live_call.py`](../../tests/examples/test_livekit_live_call.py) runs the
example against a subprocess `verbatim serve --pipeline fake`, with the plugin in an isolated `uv`
environment. It skips where the plugin is not installed, and where it runs it asserts the two things
that matter: nine audio chunks produce nine tokens and a final through the plugin's own event stream,
and a model name Verbatim does not serve is refused rather than answered.

That test is the gate's evidence. A recipe nobody runs is a claim; a recipe with a live-call test is a
measurement.
