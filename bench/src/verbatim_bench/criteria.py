# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The rung criteria vocabulary, on its own so `verify` can read it without the load stack.

`ladder.py` is where a rung is measured, and measuring needs the generator, the corpus
decoder and numpy. `verify.py` recomputes a published row with nothing but the standard
library, and a reviewer's machine may have nothing else, so the names the two share live
here, importing nothing. `ladder` re-exports them; import from either.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = ["RUNG_PASS_CRITERIA", "Criterion"]


class Criterion(StrEnum):
    LATENCY = "latency"
    WER = "wer"
    INTEGRITY_REFUSED = "integrity:refused"
    INTEGRITY_DROPPED = "integrity:dropped"
    INTEGRITY_NO_FINAL = "integrity:no_final"
    THERMAL = "thermal"
    UNSTABLE = "unstable"
    INVALID_HOST = "invalid_host"


#: The four criteria the frozen methodology names for a rung to pass, written in the
#: vocabulary of `Criterion`. Its third criterion, integrity, is three values here,
#: because a stream can be refused, dropped, or end without a final transcript.
#: `UNSTABLE` and `INVALID_HOST` are deliberately absent: they name a rung that produced
#: no measurement window and a host that was unfit to measure on, neither of which is a
#: criterion the run is asked to establish.
RUNG_PASS_CRITERIA: Final = frozenset(
    {
        Criterion.LATENCY,
        Criterion.WER,
        Criterion.INTEGRITY_REFUSED,
        Criterion.INTEGRITY_DROPPED,
        Criterion.INTEGRITY_NO_FINAL,
        Criterion.THERMAL,
    }
)
