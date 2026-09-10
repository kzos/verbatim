# `tests/`

Everything here is CPU-only by default and must pass with **no GPU, no CUDA and no NeMo installed**.
Markers: `cpu` (default), `gpu` (self-hosted runners only), `slow` (the gates).

- `test_skeleton.py` — the module tree imports; the generated Riva stubs import and expose the right
  service surface.
- `protocol/` — Riva and WebSocket conformance against a stub recognizer.

- `scheduler/` — the tick loop, buckets, admission and boundary bookkeeping against a simulated clock
  and a fake CPU pipeline.
- `harness/` — the load generator and the results schema.

The GPU is needed to find out what the tick budget *is*. It is not needed to test what the scheduler
does with it, and a test that needs one is a test most maintainers cannot run on most days.
