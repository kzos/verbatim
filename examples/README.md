# `examples/` — one directory per claim, so the claims are falsifiable

Planned, none written yet:

- `livekit_agent/` — LiveKit Agents against a self-hosted Verbatim endpoint, **zero plugin code**
- `pipecat_bot/` — Pipecat's `nvidia` STT service pointed at Verbatim, **zero plugin code**
- `browser_demo/` — vanilla JS plus an `AudioWorklet`; no framework, no bundler, no build step
- `grpc_client.py`, `ws_client.py` — the two surfaces, minimally

The first two are the day-60 gate in example form: if they need plugin code, the drop-in claim is
false and the README has to say so.
