# `docs/upstream/` — one file per NeMo shim

Upstream first, shim second, with an expiry. When NeMo lacks a hook Verbatim needs, the pull request
goes to NeMo *first*. Only then may `src/verbatim/pipelines/nemo_compat.py` carry a shim, behind a
flag, with an upstream PR number and a deletion date recorded here.

**A shim past its deletion date fails CI.** That is what turns "we'll upstream it eventually" into a
build error.

Each file: what the shim does, why it exists, the upstream PR number, the deletion date.
