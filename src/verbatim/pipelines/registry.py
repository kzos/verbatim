# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Built-in adapters register by name here; ``EngineConfig.pipeline`` selects one.

Third-party adapters arrive through the ``verbatim.pipelines`` entry-point group,
so an operator can carry a private adapter without forking: an entry point whose
name is not registered here is loaded on first use.

A factory takes the ``EngineConfig`` and keyword arguments that only that adapter
understands. ``cache_aware_rnnt`` needs a built NeMo pipeline, passed as
``pipeline=``, or an already-bound ``boundary=``; it cannot build one itself,
because building one is a model download and a forty-field NeMo config that belong
to the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from verbatim.config import EngineConfig
from verbatim.core.errors import InvalidArgument
from verbatim.pipelines.base import PipelineAdapter
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter, NeMoBoundary
from verbatim.pipelines.fake import FakePipelineAdapter

__all__ = ["ENTRY_POINT_GROUP", "PipelineFactory", "build", "build_for", "names", "register"]

ENTRY_POINT_GROUP = "verbatim.pipelines"

PipelineFactory = Callable[..., PipelineAdapter]

_REGISTRY: dict[str, PipelineFactory] = {}


def register(name: str, factory: PipelineFactory, *, replace: bool = False) -> None:
    """Register a factory under ``name``. Re-registering a name is a caller bug unless
    ``replace`` says so."""
    if not name:
        raise InvalidArgument("a pipeline name must be non-empty")
    if name in _REGISTRY and not replace:
        raise InvalidArgument(f"pipeline {name!r} is already registered")
    _REGISTRY[name] = factory


def names() -> tuple[str, ...]:
    """Registered names, in registration order."""
    return tuple(_REGISTRY)


def _load_entry_point(name: str) -> PipelineFactory | None:
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name == name:
            factory = entry_point.load()
            register(name, factory)
            return factory
    return None


def build(name: str, config: EngineConfig, **kwargs: Any) -> PipelineAdapter:
    """Build the adapter registered as ``name`` for ``config``."""
    factory = _REGISTRY.get(name)
    if factory is None:
        factory = _load_entry_point(name)
    if factory is None:
        known = ", ".join(names()) or "<none>"
        raise InvalidArgument(f"unknown pipeline {name!r}: registered pipelines are {known}")
    return factory(config, **kwargs)


def build_for(config: EngineConfig, **kwargs: Any) -> PipelineAdapter:
    """Build the adapter ``config.pipeline`` names."""
    return build(config.pipeline, config, **kwargs)


def _fake_factory(config: EngineConfig, **kwargs: Any) -> PipelineAdapter:
    return FakePipelineAdapter(config.chunk, buckets=config.buckets or (), **kwargs)


def _cache_aware_rnnt_factory(
    config: EngineConfig,
    *,
    pipeline: Any = None,
    boundary: NeMoBoundary | None = None,
    language_code: str | None = None,
) -> PipelineAdapter:
    if boundary is None:
        if pipeline is None:
            raise InvalidArgument(
                "cache_aware_rnnt needs a built NeMo CacheAwareRNNTPipeline: "
                "pass pipeline=<pipeline> or boundary=<NeMoBoundary>"
            )
        boundary = NeMoBoundary.from_pipeline(pipeline)
    return CacheAwareRNNTAdapter(
        config.chunk,
        boundary,
        buckets=config.buckets or (),
        required_slots=config.num_slots,
        stop_history_eou_ms=config.stop_history_eou_ms,
        language_code=language_code,
    )


register("fake", _fake_factory)
register("cache_aware_rnnt", _cache_aware_rnnt_factory)
