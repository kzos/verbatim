# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``PipelineAdapter`` -- Verbatim's stable surface over NeMo's unstable one.

An adapter owns: constructing the NeMo pipeline through its builder, translating
``core.types`` to NeMo ``Frame``/``FeatureBuffer``, exposing the graph key
parameters to ``scheduler/graph_budget.py``, and declaring which chunk modes it
supports.

The NeMo adapter is ``verbatim.pipelines.cache_aware_rnnt``. This ABC lets the
scheduler's whole logic be tested without a GPU, against the deterministic CPU
fake in ``verbatim.pipelines.fake``. This module must not depend on the NeMo
toolkit or on PyTorch.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

from verbatim.config import ChunkMode
from verbatim.core.types import PcmFrame, StepResult
from verbatim.protocols.base import SessionOptions

__all__ = ["PipelineAdapter"]


class PipelineAdapter(abc.ABC):
    """Verbatim's stable surface over the model runtime. One per chunk mode.

        The NeMo adapter is ``verbatim.pipelines.cache_aware_rnnt``; the CPU fake in
    ``verbatim.pipelines.fake`` exists so the scheduler's whole logic is testable
    without a GPU.
    """

    @property
    @abc.abstractmethod
    def chunk(self) -> ChunkMode:
        """The chunk mode this adapter steps."""
        ...

    @abc.abstractmethod
    def supported_buckets(self) -> tuple[int, ...]:
        """Bucket sizes this adapter was warmed up (graph-captured) for."""
        ...

    @abc.abstractmethod
    def open_stream(self, stream_id: int, options: SessionOptions | None) -> None:
        """Called once, on the tick thread, before this stream's first frame. The real
        adapter needs the request options for its per-stream state; the fake uses them to
        pick a script. `options` is None for a Session built outside the engine (every
        Session in tests/scheduler/), and an adapter that needs them must say so rather
        than assume them."""

    @abc.abstractmethod
    def close_stream(self, stream_id: int) -> None:
        """Called once, on the tick thread, after this stream's is_last frame, or when
        the tick loop fails the stream. It may be called for a stream whose
        `open_stream` raised, and must tolerate that."""

    @abc.abstractmethod
    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool
    ) -> list[StepResult]:
        """One step over a whole batch. `keep_all_outputs=True` marks the edge path.

        Implementations MUST NOT set `audio_processed_s`, `valid_samples` or `is_last`:
        the tick loop stamps all three from the Session and the frame. An adapter that
        stamped its own clock could not tell real bytes from the padding the server
        invented, which is the bug this rule exists to prevent.
        """
