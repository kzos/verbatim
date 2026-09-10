# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The server's only entry point: ``verbatim serve|demo|invariance|doctor``.

The harness has its own CLI (``verbatim-bench``, under ``bench/``); the two never
share a process and the harness never imports this package.

``doctor`` is load-bearing rather than decorative: it reports which NeMo track is
installed and refuses to start the CUDA-graph path -- rather than silently
degrading to eager -- when the installed NeMo lacks the graphed streaming encoder
step (NeMo PR #15863).
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Not implemented yet."""
    raise SystemExit(
        "verbatim: not implemented yet. The server is under construction; "
        "see docs/SCOPE.md for what is in scope and README.md for what has been measured."
    )
