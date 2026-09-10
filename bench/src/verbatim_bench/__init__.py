# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``verbatim-bench`` -- the Verbatim measurement harness.

A separate distribution with **no dependency on the server**, so the harness, the
rows, the corpora and the invariance gate survive the server being de-scoped.
``verify``, ``validate`` and ``replay`` are torch-free, so a laptop can recompute
a published artifact.

Honesty rule 1, which governs this package and the README alike: *no number that
the harness did not produce*. A number is a link to a row.
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

__all__ = ["__version__"]
