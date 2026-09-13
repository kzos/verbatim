# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Adapter onto NeMo's ``CacheAwareCTCPipeline``.

The lifecycle this adapter drives is the one in ``verbatim.pipelines.cache_aware``,
shared with the RNNT adapter because NeMo shares it: ``CacheAwareCTCPipeline`` and
``CacheAwareRNNTPipeline`` both inherit ``BasePipeline.transcribe_step`` unchanged,
allocate the same bufferer and the same ``CacheAwareContextManager``, and split
``is_last`` rows into the same ``keep_all_outputs=True`` sub-batch. Read that module
first; everything about slots, pad rows, post-step bookkeeping and word timings is
there and is not restated here.

Where CTC genuinely differs, read from
``nemo/collections/asr/inference/pipelines/cache_aware_ctc_pipeline.py`` and
``factory/cache_aware_pipeline_builder.py``:

- **No per-stream decoder state.** ``cache_aware_transcribe_step`` passes features
  and context to ``stream_step`` and gets log probabilities back; there is no
  ``previous_hypotheses`` in, none out, and nothing to reset on ``is_last``. What
  the state carries across steps is the endpointer's label buffer and the tokens of
  the utterance so far, both of which NeMo owns. This adapter therefore models no
  decoder state at all, and there is no shared decoder-state abstraction pretending
  the two pipelines have one.
- **No prompt path.** ``CacheAwareCTCPipeline.__init__`` never calls
  ``init_prompt_support``, and its ``create_state`` calls ``options.fill_defaults``
  without ``default_language_code`` and never resolves a prompt index. A session's
  ``language_code`` is therefore recorded on NeMo's per-stream request options and
  never acted on: unlike the prompt-conditioned RNNT path, an unknown code cannot
  fail ``open_stream``, because nothing reads it. Verbatim keeps recording it rather
  than dropping it -- the options object is the record of what the client asked for
  -- and ``verbatim serve`` says so in its startup banner instead of leaving an
  operator to infer it from a transcript.
- **No per-stream biasing.** Biasing is reached through the RNNT decoding computer;
  there is no CTC equivalent in the inference package.
- **The decoding config is NeMo's, not ours.**
  ``CacheAwarePipelineBuilder.get_ctc_decoding_cfg`` takes no argument: it builds a
  ``CTCDecodingConfig`` with ``strategy="greedy"`` and ignores ``asr.decoding``
  entirely, where the RNNT branch merges ours. Scope item 2 is satisfied by not
  fighting that.
- **A tail-result path that RNNT does not have.** CTC's ``decode_log_probs`` can
  decode ``tail_log_probs`` into the partial transcript. The wrapper only produces
  them when ``valid_out_len`` is set and ``keep_all_outputs`` is false, and
  ``valid_out_len`` is ``None`` whenever ``use_cache`` is true, which is how Verbatim
  builds; ``return_tail_result`` stays false as well, so the path is inert here. It
  is named because it is a real difference in the shipped code, not because it runs.

The crossed-wire guard in the shared body matters most here: an RNNT-built pipeline
served under this adapter's name would transcribe perfectly well and put "CTC" on
the row. It is refused instead.
"""

from __future__ import annotations

from typing import ClassVar

from verbatim.pipelines.cache_aware import CacheAwareAdapter

__all__ = ["CacheAwareCTCAdapter"]


class CacheAwareCTCAdapter(CacheAwareAdapter):
    """Verbatim's ``PipelineAdapter`` over NeMo's cache-aware CTC pipeline, one per
    chunk mode. See the module docstring for where it differs from its RNNT sibling
    and ``verbatim.pipelines.cache_aware`` for everything it shares with it."""

    kind: ClassVar[str] = "cache-aware CTC"
    registry_name: ClassVar[str] = "cache_aware_ctc"
    decoder_attribute: ClassVar[str] = "greedy_ctc_decoder"
    sibling_decoder_attribute: ClassVar[str] = "greedy_rnnt_decoder"
    sibling_registry_name: ClassVar[str] = "cache_aware_rnnt"
