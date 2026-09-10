# SPDX-License-Identifier: Apache-2.0
"""The shared result cadence, kept separate from both protocol encoders.

The real adapter over NeMo's cache-aware pipeline is a LATER TASK, on a machine
with a GPU. This module only turns CPU-side ``StepResult`` records into the
hypotheses a future protocol adapter will encode.
"""

from __future__ import annotations

from dataclasses import dataclass

from verbatim.core.types import StepResult
from verbatim.protocols.base import Hypothesis, SessionOptions

__all__ = ["Emission", "emissions_for"]


@dataclass(frozen=True, slots=True)
class Emission:
    """What one ``StepResult`` turns into on the wire, in order."""

    hypotheses: tuple[Hypothesis, ...]
    end_of_session: bool


def emissions_for(result: StepResult, options: SessionOptions) -> Emission:
    """Apply the one cadence rule on the asyncio loop.

    A valid frame produces a partial when interim results are enabled, even when
    its text did not change. Finals carry words and confidence; a mid-stream
    final is followed by the empty partial used by the Riva surface.
    """
    hypotheses: list[Hypothesis] = []
    if result.valid_samples > 0 and options.interim_results:
        hypotheses.append(
            Hypothesis(
                text=result.partial_text,
                is_final=False,
                audio_processed_s=result.audio_processed_s,
            )
        )

    if result.final_text is not None:
        hypotheses.append(
            Hypothesis(
                text=result.final_text,
                is_final=True,
                audio_processed_s=result.audio_processed_s,
                words=result.words,
                confidence=result.confidence,
            )
        )
        if not result.is_last and result.valid_samples > 0 and options.interim_results:
            hypotheses.append(
                Hypothesis(
                    text="",
                    is_final=False,
                    audio_processed_s=result.audio_processed_s,
                )
            )

    return Emission(tuple(hypotheses), end_of_session=result.is_last)
