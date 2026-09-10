# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Typed configuration: chunk modes, bucket sets, the SLO and admission limits.

Chunk mode is the audio period of one tick. For ``nemotron-3.5-asr-streaming-0.6b``
an ``att_context_size`` of ``[56, k]`` gives a ``(k + 1) * 80 ms`` period, so
k in {0, 1, 3, 6, 13} yields 80/160/320/560/1120 ms. The English checkpoint uses
the ``[70, *]`` family and has no 320 ms mode.

The SLO is ``p95 partial latency <= chunk + 150 ms``. The concurrency ceiling is
never typed in: it is calibrated against a measured tick budget.

This module holds validated configuration only: no model, no timing
measurement, no number presented as observed. This module must not depend on
the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from verbatim.core.errors import InvalidArgument
from verbatim.scheduler.graph_budget import ConfigError, assert_graph_budget

__all__ = [
    "SAMPLE_RATE_HZ",
    "VALID_CHUNK_MS",
    "ChunkMode",
    "EngineConfig",
]

VALID_CHUNK_MS: Final = (80, 160, 560, 1120)
SAMPLE_RATE_HZ: Final = 16000

#: Placeholder bucket used when neither buckets nor a calibrated ceiling is given.
#: This is a CPU-test convenience default, not a measurement of any GPU.
_DEFAULT_BUCKET: Final = (8,)


@dataclass(frozen=True, slots=True)
class ChunkMode:
    """One tick's audio period. The tick budget *is* this period; the GPU is only
    needed to find out how much work fits inside it, not to test the bookkeeping."""

    ms: int

    def __post_init__(self) -> None:
        if self.ms not in VALID_CHUNK_MS:
            valid = ", ".join(str(v) for v in VALID_CHUNK_MS)
            raise InvalidArgument(f"invalid ms {self.ms!r}: must be one of {valid}")

    @property
    def samples(self) -> int:
        """Samples per chunk: ms * sample_rate_hz // 1000."""
        return self.ms * SAMPLE_RATE_HZ // 1000

    @property
    def period_s(self) -> float:
        """Tick period in seconds."""
        return self.ms / 1000


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """One chunk mode's scheduler configuration.

    The bucket list defaults to a single bucket at the calibrated ceiling -- the
    fixed-shape policy: at any occupancy every tick replays the same shape. Elastic
    buckets (``n > 1``) are opt-in via ``elastic_buckets=True``.

    ``calibrated_ceiling`` is supplied from a measurement row for this GPU and mode;
    there is no default and none may be invented, because a ceiling this project did
    not measure is a number this project does not have. When it is ``None`` the
    admission controller has no ceiling and rejects nothing on that basis (slot
    capacity still binds).

    ``pipeline`` names a factory in ``verbatim.pipelines.registry``; the default is
    the CPU fake, so no configuration silently loads a model. ``stop_history_eou_ms``
    is the end-of-utterance silence NeMo's endpointer waits for when a session does
    not carry its own; 0 disables endpointing and a final then comes only at
    half-close.

    ``idle_timeout_s`` is the longest a live session may go without a chunk of audio
    before the engine closes it with ``DEADLINE_EXCEEDED`` and frees its slot. It is
    counted in ticks on the engine's own clock, so it lives here rather than in
    either transport and both wires inherit it. ``None`` disables it, which is what
    the CPU test harness does; a server should not.
    """

    chunk: ChunkMode
    buckets: tuple[int, ...] | None = None
    edge_batch: int = 8
    pad_pool: int = 0
    drain_margin: int = 8
    max_graphs: int = 8
    utilisation_target: float = 0.70
    calibrated_ceiling: int | None = None
    ring_seconds: float = 3.0
    elastic_buckets: bool = False
    pipeline: str = "fake"
    stop_history_eou_ms: int = 800
    idle_timeout_s: float | None = 30.0

    def __post_init__(self) -> None:
        buckets = self.buckets
        if buckets is None:
            # The default is n = 1, B_1 = ceiling(mode). Without a calibration
            # there is no ceiling, so fall back to a placeholder bucket: a
            # CPU-test convenience, explicitly not a measurement.
            buckets = (
                (self.calibrated_ceiling,)
                if self.calibrated_ceiling is not None
                else _DEFAULT_BUCKET
            )
            object.__setattr__(self, "buckets", buckets)
        if len(buckets) == 0:
            raise ConfigError("buckets must be non-empty")
        if any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in buckets):
            raise ConfigError(f"buckets must be positive integers, got {buckets!r}")
        if tuple(buckets) != tuple(sorted(set(buckets))):
            raise ConfigError(f"buckets must be ascending and unique, got {buckets!r}")
        object.__setattr__(self, "buckets", tuple(buckets))
        if self.edge_batch < 1:
            raise ConfigError(f"edge_batch must be >= 1, got {self.edge_batch!r}")
        if self.pad_pool < 0:
            raise ConfigError(f"pad_pool must be >= 0, got {self.pad_pool!r}")
        if self.drain_margin < 0:
            raise ConfigError(f"drain_margin must be >= 0, got {self.drain_margin!r}")
        if self.max_graphs < 1:
            raise ConfigError(f"max_graphs must be >= 1, got {self.max_graphs!r}")
        if not 0 < self.utilisation_target <= 1:
            raise ConfigError(
                f"utilisation_target must satisfy 0 < t <= 1, got {self.utilisation_target!r}"
            )
        if self.ring_seconds <= 0:
            raise ConfigError(f"ring_seconds must be positive, got {self.ring_seconds!r}")
        if self.calibrated_ceiling is not None and self.calibrated_ceiling < 1:
            raise ConfigError(
                f"calibrated_ceiling must be a positive integer, got {self.calibrated_ceiling!r}"
            )
        if self.elastic_buckets and len(buckets) == 1:
            raise ConfigError("elastic_buckets requires more than one bucket")
        if not self.pipeline:
            raise ConfigError("pipeline must name a registered adapter")
        if isinstance(self.stop_history_eou_ms, bool) or self.stop_history_eou_ms < 0:
            raise ConfigError(f"stop_history_eou_ms must be >= 0, got {self.stop_history_eou_ms!r}")
        if self.idle_timeout_s is not None and (
            isinstance(self.idle_timeout_s, bool) or not self.idle_timeout_s > 0
        ):
            raise ConfigError(
                f"idle_timeout_s must be positive or None, got {self.idle_timeout_s!r}"
            )
        assert_graph_budget({self.chunk.ms: buckets}, self.max_graphs)

    @property
    def effective_pad(self) -> int:
        """Pad rows holding the steady shape: explicit ``pad_pool``, else ``max(buckets)``."""
        assert self.buckets is not None
        return self.pad_pool if self.pad_pool > 0 else max(self.buckets)

    @property
    def num_slots(self) -> int:
        """``max(buckets) + pad_pool + edge_batch + drain_margin``."""
        assert self.buckets is not None
        return max(self.buckets) + self.effective_pad + self.edge_batch + self.drain_margin

    @property
    def budget_ms(self) -> float:
        """``chunk.ms * utilisation_target``. The rest absorbs edge batches,
        end-of-utterance post-processing and interpreter jitter."""
        return self.chunk.ms * self.utilisation_target
