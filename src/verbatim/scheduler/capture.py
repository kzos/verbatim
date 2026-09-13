# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Which shapes get a CUDA graph, who captures them, and what happens when there is none.

The scope's first item rests on this: every chunk period the scheduler hands NeMo a
batch whose shape it has already captured, and replays it. Three things have to be
true for that to be more than a hope, and this module owns all three.

**The keys are decided before the server runs, not discovered.** A capture key is
``(chunk mode, batch size, keep_all_outputs)`` -- the part of NeMo's key
(``shape, dtype, device, keep_all_outputs, drop_extra_pre_encoded, att_context_size,
last_channel_cache_size, valid_out_len``) that a scheduler decision moves. One key
per bucket, and the total must fit ``max_graphs`` (``graph_budget.py``, which also
reserves one for the edge shape) or the configuration is refused at construction.
First frames are deliberately not a key of their own: NeMo keys on the per-batch
pre-encode drop, which a first frame does not change, so a first frame is safe inside
the steady batch.

**Capture is warm-up, not a tick.** ``CaptureController.warmup`` steps one all-pad
batch at each bucket shape before the boundary grid starts. NeMo captures a shape the
first time it sees it, so warming the steady shapes first is what makes them the ones
that get the budget; a shape first seen on a live tick would be captured inside that
tick, against the tick's own deadline. Nothing here calls a NeMo capture API: there
isn't one to call, and inventing one that this machine cannot run would be a fiction.
Warm-up is a step at the shape we want retained, which is what NeMo's own key means.

**An absent graph path is refused, never downgraded.** The graphed streaming encoder
step (NeMo PR #15863) is in no released wheel; see
``docs/decisions/0002-the-graph-path-is-not-in-a-released-wheel.md``. So the mode is
selected from a capability the adapter reports, and a pipeline built for graphs on a
runtime that has none raises ``GraphPathUnavailable`` naming the reason, exactly as
``serve`` refuses today. An eager run is a thing an operator asks for by name; it is
never a thing that happens because a check failed quietly.

The edge batch is the one shape that is never captured, and upstream agrees: per the
reading in ``docs/ISSUES.md`` (issue 3), ``_can_use_graphs()`` never captures a
``keep_all_outputs=True`` call at all. So the peel is not Verbatim declining a graph
it could have had. What the peel buys is the *steady* key: NeMo's pipeline splits each
tick into a non-final and a final sub-batch, and a final left in the steady batch
shrinks the non-final sub-batch by one row, which is a different key every tick.
``edge_graph()`` returns ``False`` unconditionally, and the adapter refuses a
``keep_all_outputs=True`` batch that arrives claiming otherwise.

This module is arithmetic over configured shapes plus one warm-up call per bucket. It
must not depend on the NeMo toolkit or on PyTorch, and the whole of it is exercised on
the CPU fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from verbatim.config import EngineConfig
from verbatim.pipelines.base import (
    UPSTREAM_WARMUP_STEPS,
    GraphCapability,
    GraphKey,
    GraphPathUnavailable,
    PipelineAdapter,
)
from verbatim.scheduler.boundary import PadPool
from verbatim.scheduler.graph_budget import ConfigError

__all__ = [
    "GRAPH_WARMUP_STEPS",
    "UPSTREAM_WARMUP_STEPS",
    "CaptureController",
    "CaptureNotRetained",
    "CapturePlan",
    "UncapturedShape",
]

#: Steps taken at each bucket shape during warm-up: one more than upstream's count,
#: because upstream captures a key only when its call count *exceeds* ``warmup_steps``
#: (``if self._key_counts[key] > self.warmup_steps``). Three steps leave the count at
#: three, three is not greater than three, and nothing is captured.
#:
#: This was three until it was measured. On the B300 on 2026-09-13, after a warm-up of
#: three steps at the 160 ms bucket-32 shape, NeMo held **zero** graphs and its own
#: call count for the shape read three; the very next step took the count to four and
#: the graph count to one. The capture had been landing on the first live tick, against
#: that tick's deadline, which is the thing warm-up exists to avoid -- and warm-up had
#: been recording it as done. See ``docs/decisions/0011``.
GRAPH_WARMUP_STEPS: Final = UPSTREAM_WARMUP_STEPS + 1


class CaptureNotRetained(RuntimeError):
    """Warm-up stepped a shape and the runtime is holding no more graphs than before.

    Upstream captures on a best-effort basis and falls back to eager in silence -- on a
    capture failure, and for any step its own ``_can_use_graphs`` disqualifies. Neither
    is visible from the outside, so a server would come up announcing the graph path,
    run every tick eager, and publish rows saying graphed. Raised at construction, where
    it costs a start-up failure with a reason, rather than a whole measurement campaign.
    """


class UncapturedShape(RuntimeError):
    """A steady batch reached the graph path at a shape nobody captured.

    Raised before the step rather than after: NeMo would capture the new shape
    inside the tick (or, with the budget full, run it eager for the life of the
    process) and neither is visible from outside. Both are the silent loss of the
    graph path this design exists to prevent.
    """


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """One chunk mode's capture keys: one per bucket, plus the reserved edge key.

    ``keys`` is what the scheduler captures and retains. ``edge_key`` names the shape
    the peeled finals run at; it is reserved in ``graph_budget``'s arithmetic and never
    captured here, and upstream's ``_can_use_graphs()`` would refuse to capture it
    anyway (``docs/ISSUES.md``, issue 3). It is carried so that "the edge shape is not
    in the captured set" is a statement a test can make about a named key rather than
    about an absence.
    """

    chunk_ms: int
    keys: tuple[GraphKey, ...]
    edge_key: GraphKey

    @classmethod
    def for_config(cls, config: EngineConfig) -> CapturePlan:
        buckets = config.buckets
        if not buckets:
            raise ConfigError("a capture plan needs at least one bucket")
        chunk_ms = config.chunk.ms
        keys = tuple(
            GraphKey(chunk_ms=chunk_ms, batch_size=bucket, keep_all_outputs=False)
            for bucket in sorted(buckets)
        )
        edge_key = GraphKey(chunk_ms=chunk_ms, batch_size=config.edge_batch, keep_all_outputs=True)
        # The budget itself is asserted where the bucket list is set, in
        # ``EngineConfig.__post_init__``; re-asserting it here over the same numbers
        # would be a branch no input could reach. What this plan owes the budget is
        # that it produces exactly as many keys as ``graph_keys_required`` counts,
        # and that is checked by a test that can fail.
        return cls(chunk_ms=chunk_ms, keys=keys, edge_key=edge_key)

    @property
    def reserved_keys(self) -> tuple[GraphKey, ...]:
        """Every key this mode can consume: the captured buckets and the edge shape."""
        return (*self.keys, self.edge_key)

    def steady_key(self, batch_size: int) -> GraphKey:
        """The key a steady batch of ``batch_size`` rows replays."""
        return GraphKey(chunk_ms=self.chunk_ms, batch_size=batch_size, keep_all_outputs=False)


class CaptureController:
    """Selects the execution mode from the adapter's capability, captures the steady
    buckets at warm-up, and answers, per step, whether it takes the graph path."""

    def __init__(
        self,
        config: EngineConfig,
        pipeline: PipelineAdapter,
        *,
        warmup_steps: int = GRAPH_WARMUP_STEPS,
    ) -> None:
        if warmup_steps < 1:
            raise ConfigError(f"warmup_steps must be >= 1, got {warmup_steps!r}")
        self._warmup_steps = warmup_steps
        self._plan = CapturePlan.for_config(config)
        declared = tuple(pipeline.supported_buckets())
        missing = sorted(set(config.buckets or ()) - set(declared))
        if declared and missing:
            # A bucket the adapter was not built for is a shape NeMo's own batch
            # dimension cannot serve, so its graph could never be captured. Caught
            # here, where both numbers are in one place, rather than as
            # "No free slots available" on some later tick.
            raise ConfigError(
                f"the scheduler is configured for buckets {tuple(config.buckets or ())} "
                f"but {type(pipeline).__name__} was built for {declared}: "
                f"{missing} would be stepped at a shape it has no graph for"
            )
        if config.padding == "ragged" and _capability_of(pipeline).available:
            # A ragged batch changes shape with occupancy, so it has no stable graph key
            # and nothing could be captured for it. The two settings are not a trade-off
            # to be resolved silently in one direction; they are incoherent together, and
            # a run that quietly dropped one of them would publish a row naming both.
            raise ConfigError(
                "padding='ragged' and the graph path cannot both be asked for: a ragged "
                "steady batch changes shape with occupancy, so there is no shape to "
                "capture. The ragged arm is the eager control (docs/decisions/0014); "
                "pass --eager with it"
            )
        capability = _capability_of(pipeline)
        if capability.requested and not capability.available:
            raise GraphPathUnavailable(
                "the graph path was requested and the runtime does not have it: "
                f"{capability.reason}. Refusing rather than running the steady batch "
                "eager unasked, so a row can never look graphed without being graphed; "
                "pass --eager to run the encoder step eager and say so in the row"
            )
        self._pipeline = pipeline
        self._capability = capability
        self._captured: set[GraphKey] = set()
        self._warmed = False

    @property
    def capability(self) -> GraphCapability:
        return self._capability

    @property
    def plan(self) -> CapturePlan:
        return self._plan

    @property
    def graphed(self) -> bool:
        """True when the steady batch replays a captured graph."""
        return self._capability.available

    @property
    def mode(self) -> str:
        """``"graph path"`` or ``"eager"``, the words ``serve`` prints."""
        return "graph path" if self.graphed else "eager"

    @property
    def warmed(self) -> bool:
        return self._warmed

    @property
    def warmup_steps(self) -> int:
        """Steps taken at each bucket shape during warm-up."""
        return self._warmup_steps

    @property
    def captured_keys(self) -> frozenset[GraphKey]:
        """The keys captured and retained. Empty in eager mode, by construction."""
        return frozenset(self._captured)

    def warmup(self, pads: PadPool) -> tuple[GraphKey, ...]:
        """Capture one graph per bucket, before the boundary grid starts.

        Each shape is stepped ``warmup_steps`` times over pad rows only, so no session's
        audio is involved and no result of it is ever emitted. Returns the distinct keys
        captured, one per bucket, in ascending bucket order. In eager mode it captures
        nothing and says so with an empty tuple: there is no partial warm-up here.

        The runtime's own retained-graph count is read before and after each shape, and
        a count that did not rise is ``CaptureNotRetained`` rather than a key recorded
        as captured. So is an adapter that claims the graph path and cannot report a
        count at all: "cannot tell" is an honest answer from an eager adapter, which
        never reaches here, and an inadmissible one from a graphed adapter, because it
        would skip the only check in this loop.
        """
        if self._warmed:
            raise ConfigError("warmup runs once; the captured graphs are retained after it")
        self._warmed = True
        if not self.graphed:
            return ()
        captured: list[GraphKey] = []
        for key in self._plan.keys:
            before = self._pipeline.retained_graphs()
            if before is None:
                # An adapter that claims the graph path and cannot say how many graphs it
                # holds gives this loop nothing to check, and the branch below would be
                # skipped -- which is DR-0011's guard restored, in silence, in exactly the
                # case it exists for. "Cannot tell" is an honest answer from an eager
                # adapter and an inadmissible one from a graphed adapter.
                raise CaptureNotRetained(
                    f"{type(self._pipeline).__name__} reports the graph path available and "
                    "cannot report how many graphs the runtime holds, so a capture here "
                    "could never be observed. Refusing rather than recording a warm-up "
                    "nothing checked"
                )
            for _ in range(self._warmup_steps):
                pads.reset()
                frames = pads.take(key.batch_size)
                self._pipeline.transcribe_step(frames, keep_all_outputs=False, graph=True)
            after = self._pipeline.retained_graphs()
            if after is not None and after <= before:
                raise CaptureNotRetained(
                    f"warm-up stepped {key} {self._warmup_steps} times and the runtime is "
                    f"holding {after} graphs, the same as before ({before}). Upstream "
                    "captures a shape only once its call count exceeds its own "
                    f"warmup_steps (upstream default {UPSTREAM_WARMUP_STEPS}), and falls "
                    "back to eager in silence on any capture it cannot make. Refusing "
                    "rather than serving a graph path that is not there"
                )
            self._captured.add(key)
            captured.append(key)
        pads.reset()
        return tuple(captured)

    def steady_graph(self, batch_size: int) -> bool:
        """Whether this tick's steady batch takes the graph path, checked against the
        keys actually captured. A shape nobody captured is refused, not run."""
        if not self.graphed:
            return False
        key = self._plan.steady_key(batch_size)
        if key not in self._captured:
            captured = ", ".join(str(k) for k in sorted(self._captured, key=lambda k: k.batch_size))
            raise UncapturedShape(
                f"the steady batch is {key} and that graph was never captured "
                f"(captured: {captured or '<none>'}). NeMo would capture it inside this "
                "tick, or run it eager for the life of the process once the budget is "
                "full, and neither is visible from outside"
            )
        return True

    def edge_graph(self) -> bool:
        """Always False, in every mode. Final steps are peeled into an eager side-batch
        so that the steady sub-batch never changes size, and NeMo's own
        ``_can_use_graphs()`` refuses a ``keep_all_outputs=True`` capture regardless, so
        there is no graph here to ask for."""
        return False


def _capability_of(pipeline: PipelineAdapter) -> GraphCapability:
    """The adapter's reported capability. An adapter that does not implement the
    surface at all is eager and says so, which is a state, not a silence."""
    query = getattr(pipeline, "graph_capability", None)
    if query is None:
        return GraphCapability.eager(
            f"{type(pipeline).__name__} does not implement graph_capability(); "
            "an adapter with no graph path runs eager"
        )
    capability = query()
    if not isinstance(capability, GraphCapability):
        raise ConfigError(
            f"{type(pipeline).__name__}.graph_capability() returned "
            f"{type(capability).__name__}, not a GraphCapability"
        )
    return capability
