# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Adapter onto NeMo's ``CacheAwareRNNTPipeline``. The MVP path.

The lifecycle this adapter drives -- how a stream is opened, how pad rows keep the
batch shape, when the record of what NeMo holds is written, and how a step output
becomes a ``StepResult`` -- lives in ``verbatim.pipelines.cache_aware`` and is shared
with the CTC adapter, because NeMo shares it too: both cache-aware pipelines inherit
``BasePipeline.transcribe_step`` unchanged. Read that module first.

What is this pipeline's own, all of it behind that seam and none of it crossing it:

- A per-stream decoder state. ``CacheAwareRNNTPipeline.cache_aware_transcribe_step``
  collects ``previous_hypotheses`` from the states, hands them to ``stream_step`` and
  writes the returned hypothesis back, resetting it on ``is_last``. The adapter never
  sees a hypothesis; it only has to not strand the state that holds one, which is
  what the post-step bookkeeping in the shared body is for.
- Per-stream biasing, through ``Hypothesis.biasing_cfg`` and the decoding computer's
  biasing multi-model. This is the pipeline that carries a session's phrase list, and
  the only one: ``supports_biasing`` is true here and false on the CTC sibling. It is
  off unless an operator passes ``--biasing``, because turning it on changes the
  decoder's arithmetic for every row, biased or not, so a run with it on is a different
  configuration and does not inherit the transcript digests of a run with it off.
- Prompt conditioning. ``create_state`` fills a default ``language_code`` when the
  checkpoint is prompt-enabled and resolves it to a prompt index, raising
  ``ValueError`` for a code the checkpoint's prompt dictionary does not carry. That
  raise lands in ``open_stream``, on the tick thread, where it fails one session
  instead of a batch -- which is why ``open_stream`` calls ``init_state`` itself.

``NeMoBoundary`` and ``CacheAwarePipelineLike`` are re-exported here: this module
owned them before the CTC adapter existed, and ``NeMoBoundary`` is the object a
third-party adapter is handed through the entry-point group.
"""

from __future__ import annotations

from typing import ClassVar

from verbatim.pipelines.cache_aware import (
    CacheAwareAdapter,
    CacheAwarePipelineLike,
    NeMoBoundary,
)

__all__ = ["CacheAwarePipelineLike", "CacheAwareRNNTAdapter", "NeMoBoundary"]


class CacheAwareRNNTAdapter(CacheAwareAdapter):
    """Verbatim's ``PipelineAdapter`` over NeMo's cache-aware RNNT pipeline, one per
    chunk mode. See ``verbatim.pipelines.cache_aware`` for what it owns."""

    kind: ClassVar[str] = "cache-aware RNNT"
    registry_name: ClassVar[str] = "cache_aware_rnnt"
    decoder_attribute: ClassVar[str] = "greedy_rnnt_decoder"
    sibling_decoder_attribute: ClassVar[str] = "greedy_ctc_decoder"
    sibling_registry_name: ClassVar[str] = "cache_aware_ctc"
