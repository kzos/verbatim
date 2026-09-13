# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``PipelineAdapter`` -- Verbatim's stable surface over NeMo's unstable one.

An adapter owns: constructing the NeMo pipeline through its builder, translating
``core.types`` to NeMo ``Frame``/``FeatureBuffer``, exposing the graph key
parameters to ``scheduler/graph_budget.py``, reporting whether its CUDA-graph path
is available (``graph_capability``), and declaring which chunk modes it supports.

The NeMo adapters are ``verbatim.pipelines.cache_aware_rnnt`` and
``verbatim.pipelines.cache_aware_ctc``, over the lifecycle they share in
``verbatim.pipelines.cache_aware``. This ABC lets the scheduler's whole logic be
tested without a GPU, against the deterministic CPU fake in
``verbatim.pipelines.fake``. This module must not depend on the NeMo toolkit or on
PyTorch.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass

from verbatim.config import ChunkMode
from verbatim.core.types import PcmFrame, StepResult
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.graph_budget import ConfigError

__all__ = ["GraphCapability", "GraphKey", "GraphPathUnavailable", "PipelineAdapter"]


class GraphPathUnavailable(ConfigError):
    """The graph path was asked for and the runtime does not have it.

    A refusal, not a downgrade: it carries the reason the adapter gave, so an
    operator reads the same sentence here that ``verbatim doctor`` prints.
    """


@dataclass(frozen=True, slots=True)
class GraphKey:
    """The part of NeMo's capture key that a scheduler decision moves.

    ``dtype``, ``device``, ``att_context_size``, ``last_channel_cache_size`` and
    ``valid_out_len`` are fixed by the checkpoint and the chunk mode for the life of
    a process, and ``drop_extra_pre_encoded`` is a per-batch constant, so what is
    left for the scheduler to change is the chunk mode, the batch size and whether
    the batch is a final (``keep_all_outputs``) one.
    """

    chunk_ms: int
    batch_size: int
    keep_all_outputs: bool

    def __post_init__(self) -> None:
        if self.chunk_ms < 1:
            raise ConfigError(f"chunk_ms must be >= 1, got {self.chunk_ms!r}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size!r}")

    def __str__(self) -> str:
        kind = "edge" if self.keep_all_outputs else "steady"
        return f"{self.chunk_ms}ms x {self.batch_size} ({kind})"


@dataclass(frozen=True, slots=True)
class GraphCapability:
    """What an adapter says about its own CUDA-graph path. Three states, no fourth.

    - ``requested and available``: the pipeline was built for graphs and the runtime
      carries them. The scheduler captures the steady buckets at warm-up.
    - ``not requested``: eager because an operator asked for eager (``--eager``), or
      because this adapter has no graph path at all. ``reason`` says which.
    - ``requested and not available``: refuse. This is the state
      ``docs/decisions/0002-the-graph-path-is-not-in-a-released-wheel.md`` is about,
      and the one that must never be answered by quietly running eager instead.

    It lives beside the adapter contract rather than in the scheduler because it is
    something an adapter reports about itself; ``scheduler/capture.py`` consumes it.
    """

    requested: bool
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and not self.requested:
            raise ConfigError("a graph path that was never requested cannot be available")
        if self.available and self.reason is not None:
            raise ConfigError("an available graph path carries no reason; the reason is why not")
        if not self.available and not self.reason:
            raise ConfigError("an unavailable graph path must name the reason it is unavailable")

    @classmethod
    def graphed(cls) -> GraphCapability:
        """Requested and present: the steady buckets get captured."""
        return cls(requested=True, available=True, reason=None)

    @classmethod
    def eager(cls, reason: str) -> GraphCapability:
        """Eager by choice or by design, and the reason says which."""
        return cls(requested=False, available=False, reason=reason)

    @classmethod
    def missing(cls, reason: str) -> GraphCapability:
        """Asked for and absent. ``CaptureController`` turns this into a refusal."""
        return cls(requested=True, available=False, reason=reason)


class PipelineAdapter(abc.ABC):
    """Verbatim's stable surface over the model runtime. One per chunk mode.

        The NeMo adapters are ``verbatim.pipelines.cache_aware_rnnt`` and
    ``verbatim.pipelines.cache_aware_ctc``; the CPU fake in
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

    def graph_capability(self) -> GraphCapability:
        """Whether this adapter's CUDA-graph path was asked for and is there.

        The default is the honest answer for an adapter that has no graph path: eager,
        with the reason named. It is not abstract because most adapters have nothing
        to say here, and it is not silent because ``GraphCapability`` cannot be
        constructed without a reason for running eager.
        """
        return GraphCapability.eager(
            f"{type(self).__name__} implements no CUDA-graph path and runs the encoder step eager"
        )

    @abc.abstractmethod
    def transcribe_step(
        self, frames: Sequence[PcmFrame], *, keep_all_outputs: bool, graph: bool
    ) -> list[StepResult]:
        """One step over a whole batch. `keep_all_outputs=True` marks the edge path.

        `graph` is the scheduler's intent for this call: True only for a steady batch
        whose shape ``scheduler/capture.py`` captured at warm-up, and never for an
        edge batch. It is a required keyword and not a defaulted one on purpose --
        a default would let an adapter run the steady batch eager without anyone
        being told, which is the failure this whole path exists to prevent. An
        adapter asked for `graph=True` when its ``graph_capability()`` is not
        available MUST raise rather than run the step eager.

        Implementations MUST NOT set `audio_processed_s`, `valid_samples` or `is_last`:
        the tick loop stamps all three from the Session and the frame. An adapter that
        stamped its own clock could not tell real bytes from the padding the server
        invented, which is the bug this rule exists to prevent.
        """
