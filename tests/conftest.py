# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Pin test imports to this worktree.

The shared virtualenv may carry an editable install that points at a different
checkout, which would otherwise shadow this worktree's packages. Inserting this
worktree's ``src`` (and ``bench/src``) at the front of ``sys.path`` keeps the
suite hermetic without touching the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _dirname in ("bench/src", "src"):
    _path = str(ROOT / _dirname)
    if _path not in sys.path:
        sys.path.insert(0, _path)
