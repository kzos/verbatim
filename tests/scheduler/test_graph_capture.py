# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""CUDA-graph capture acceptance tests, run entirely on the CPU fake.

The graph path is in no released NeMo wheel and not on this machine
(``docs/decisions/0002``), so none of these tests can run a real captured graph.
What they can do, and what they do, is drive the capture *decision* and the capture
*requests* through the same seam the NeMo adapter uses, with the capability faked in
all three of its states. Every one of them asserts something that is false when the
seam is broken; none of them passes merely because this box has no graph path -- the
graphed arm here reports an available graph path and is asserted to have captured
and replayed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.pipelines.base import GraphCapability, GraphKey, GraphPathUnavailable
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.scheduler.boundary import PadPool
from verbatim.scheduler.capture import (
    GRAPH_WARMUP_STEPS,
    CaptureController,
    CapturePlan,
    UncapturedShape,
)
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.graph_budget import ConfigError, graph_keys_required
from verbatim.scheduler.tick import TickLoop

CHUNK = ChunkMode(160)
N = CHUNK.samples

#: The reason a test's faked runtime gives for having no graphed step. It stands in
#: for NeMo PR #15863 being absent; the words are this test's, not a measurement.
ABSENT = "the faked runtime carries no graphed streaming encoder step"


def _config(
    buckets: tuple[int, ...] = (8,), *, edge_batch: int = 8, elastic: bool = False
) -> EngineConfig:
    return EngineConfig(
        chunk=CHUNK,
        buckets=buckets,
        edge_batch=edge_batch,
        elastic_buckets=elastic,
        calibrated_ceiling=max(buckets),
    )


def _pipeline(
    buckets: tuple[int, ...] = (8,), *, graphs: GraphCapability | None = None
) -> FakePipelineAdapter:
    """A fake whose graph path is available unless a test says otherwise. The default
    is the graphed arm on purpose: a suite that only ever ran the eager arm would pass
    on this box for the wrong reason."""
    return FakePipelineAdapter(
        CHUNK, buckets=buckets, graphs=graphs if graphs is not None else GraphCapability.graphed()
    )


def _loop(
    buckets: tuple[int, ...] = (8,),
    *,
    graphs: GraphCapability | None = None,
    edge_batch: int = 8,
    elastic: bool = False,
) -> tuple[TickLoop, FakePipelineAdapter]:
    config = _config(buckets, edge_batch=edge_batch, elastic=elastic)
    pipeline = _pipeline(buckets, graphs=graphs)
    loop = TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock())
    return loop, pipeline


def _join(loop: TickLoop, session_id: int, chunks: int, *, drain: bool) -> Session:
    rng = np.random.default_rng(session_id)
    session = Session(session_id, CHUNK, ring_seconds=8.0)
    session.configure()
    decision = loop.admit_session(session)
    assert decision.admitted, decision.reason
    audio = rng.uniform(-0.5, 0.5, size=chunks * N).astype(np.float32)
    assert session.ring.write(audio) == len(audio)
    if drain:
        session.begin_draining()
    return session


# --- what warm-up captures ---------------------------------------------------------


def test_warmup_captures_one_graph_per_bucket_before_the_first_tick() -> None:
    """Every configured bucket is captured, at construction, and nothing else is."""
    loop, pipeline = _loop((4, 8), elastic=True)
    assert loop.capture.mode == "graph path"
    assert loop.tick_id == 0
    assert loop.boundaries == []
    assert loop.captured_graphs == (
        GraphKey(chunk_ms=160, batch_size=4, keep_all_outputs=False),
        GraphKey(chunk_ms=160, batch_size=8, keep_all_outputs=False),
    )
    assert pipeline.captures == list(loop.captured_graphs)
    # Upstream's own warmup_steps is 3, so each shape is stepped three times and the
    # two after the capture are replays of it. Nothing beyond the two buckets is
    # stepped: warm-up is capture, not traffic.
    assert pipeline.steps == 2 * GRAPH_WARMUP_STEPS
    assert sorted(key.batch_size for key in pipeline.replays) == [4, 4, 8, 8]


def test_warmup_captures_pad_rows_only_so_no_session_audio_is_involved() -> None:
    loop, pipeline = _loop((8,))
    del loop
    assert pipeline.calls_seen == [(8, False, True)] * GRAPH_WARMUP_STEPS
    # No session exists yet: pad rows are the whole of every warm-up batch.
    assert pipeline.steps == GRAPH_WARMUP_STEPS


def test_each_shape_is_warmed_often_enough_for_upstream_to_capture_it() -> None:
    """Upstream's ``CudaGraphsStreamingEncoderStep`` takes ``warmup_steps: int = 3``
    (docs/ISSUES.md, issue 3), so a single step at a shape is not on its own enough to
    be sure that shape ends up captured. Every bucket gets the same number of them."""
    assert GRAPH_WARMUP_STEPS >= 3
    loop, pipeline = _loop((4, 8), elastic=True)
    del loop
    per_shape = Counter(size for size, _keep, _graph in pipeline.calls_seen)
    assert per_shape == {4: GRAPH_WARMUP_STEPS, 8: GRAPH_WARMUP_STEPS}


def test_a_warmup_of_no_steps_is_refused() -> None:
    with pytest.raises(ConfigError, match="warmup_steps"):
        CaptureController(_config((8,)), _pipeline((8,)), warmup_steps=0)


def test_a_bucket_the_adapter_was_not_built_for_is_refused() -> None:
    """The scheduler's bucket and NeMo's batch dimension are two numbers that have to
    agree; disagreeing means capturing a shape the pipeline cannot step."""
    config = _config((16,))
    pipeline = _pipeline((8,), graphs=GraphCapability.graphed())
    with pytest.raises(ConfigError, match=r"no graph for"):
        CaptureController(config, pipeline)


# --- the steady batch replays, the edge batch does not ------------------------------


def test_every_steady_step_replays_the_captured_graph_for_its_shape() -> None:
    loop, pipeline = _loop((8,))
    _join(loop, 1, 3, drain=False)
    _join(loop, 2, 3, drain=False)
    loop.run_for(3)
    steady = [call for call in pipeline.calls_seen if not call[1]]
    assert len(steady) == GRAPH_WARMUP_STEPS + 3  # warm-up, then three ticks
    assert all(graph for _size, _keep, graph in steady)
    assert all(size == 8 for size, _keep, _graph in steady)
    key = GraphKey(chunk_ms=160, batch_size=8, keep_all_outputs=False)
    # Captured exactly once, then replayed by every warm-up step after the first and
    # by every tick: the shape never changes, so the key never does either.
    assert pipeline.captures == [key]
    assert pipeline.replays == [key] * (GRAPH_WARMUP_STEPS - 1 + 3)
    assert all(stat.steady_graphed for stat in loop.stats)


def test_the_edge_batch_never_takes_the_graph_path() -> None:
    """The peel. Finals go to a side-batch that is asked for eagerly, and no key the
    adapter ever saw on the graph path is a final shape."""
    loop, pipeline = _loop((8,), edge_batch=4)
    _join(loop, 1, 1, drain=True)
    loop.run_for(3)
    edge_calls = [call for call in pipeline.calls_seen if call[1]]
    assert edge_calls, "no edge batch ran: this test would assert nothing"
    assert all(not graph for _size, _keep, graph in edge_calls)
    assert loop.capture.edge_graph() is False
    seen_on_the_graph_path = pipeline.captures + pipeline.replays
    assert seen_on_the_graph_path, "nothing took the graph path: this test would prove nothing"
    assert all(not key.keep_all_outputs for key in seen_on_the_graph_path)


def test_the_edge_shape_is_named_reserved_and_never_captured() -> None:
    """The edge key exists so its absence from the captured set is an assertion about a
    named shape rather than about nothing."""
    loop, _pipeline = _loop((8,), edge_batch=4)
    plan = loop.capture.plan
    assert plan.edge_key == GraphKey(chunk_ms=160, batch_size=4, keep_all_outputs=True)
    assert plan.edge_key in plan.reserved_keys
    assert plan.edge_key not in plan.keys
    assert plan.edge_key not in loop.capture.captured_keys


def test_a_first_frame_stays_in_the_steady_batch_and_costs_no_extra_key() -> None:
    """First steps are deliberately not peeled: NeMo keys the graph on the per-batch
    pre-encode drop, which a first frame does not change. So a tick carrying a first
    frame replays the same key as one that does not, and captures nothing new."""
    loop, pipeline = _loop((8,))
    loop.run_for(1)  # a tick with no sessions at all
    before = len(pipeline.captures)
    _join(loop, 1, 2, drain=False)
    loop.run_for(1)  # the tick that carries session 1's first frame
    assert pipeline.captures[before:] == []
    assert {key.batch_size for key in pipeline.replays} == {8}
    assert all(stat.steady_graphed for stat in loop.stats)


# --- the refusal --------------------------------------------------------------------


def test_a_requested_graph_path_the_runtime_lacks_is_refused_not_downgraded() -> None:
    config = _config((8,))
    pipeline = _pipeline((8,), graphs=GraphCapability.missing(ABSENT))
    with pytest.raises(GraphPathUnavailable) as raised:
        TickLoop(config, pipeline, SessionRegistry(), clock=SimulatedClock())
    message = str(raised.value)
    assert ABSENT in message  # the reason is named, not swallowed
    assert "--eager" in message
    # And it refused before running anything, rather than stepping eagerly.
    assert pipeline.steps == 0


def test_eager_by_choice_captures_nothing_and_never_claims_a_graph() -> None:
    """The other half of the refusal: an eager server must not look graphed either."""
    loop, pipeline = _loop((8,), graphs=GraphCapability.eager("--eager was passed"))
    assert loop.capture.mode == "eager"
    assert loop.captured_graphs == ()
    assert loop.capture.captured_keys == frozenset()
    _join(loop, 1, 2, drain=True)
    loop.run_for(3)
    assert pipeline.captures == []
    assert pipeline.replays == []
    assert all(not graph for _size, _keep, graph in pipeline.calls_seen)
    assert not any(stat.steady_graphed for stat in loop.stats)


def test_an_adapter_that_implements_no_graph_path_is_eager_and_says_why() -> None:
    capability = FakePipelineAdapter(CHUNK, buckets=(8,)).graph_capability()
    assert capability.available is False
    assert capability.requested is False
    assert capability.reason


def test_a_capability_cannot_be_unavailable_without_naming_a_reason() -> None:
    with pytest.raises(ConfigError):
        GraphCapability(requested=False, available=False, reason=None)
    with pytest.raises(ConfigError):
        GraphCapability(requested=False, available=True, reason=None)


def test_the_fake_refuses_a_graph_step_it_has_no_graph_for() -> None:
    """The adapter half of "never downgrade": handed graph=True with no graph path, it
    raises rather than stepping. Called directly, because the controller's job is to
    make this unreachable through the loop."""
    pipeline = _pipeline((8,), graphs=GraphCapability.eager("no graphs here"))
    pads = PadPool(8, CHUNK)
    with pytest.raises(GraphPathUnavailable, match="no graphs here"):
        pipeline.transcribe_step(pads.take(8), keep_all_outputs=False, graph=True)


# --- the budget ---------------------------------------------------------------------


def test_the_plan_produces_exactly_the_keys_the_budget_counts() -> None:
    """The plan and ``graph_budget`` must agree, or the budget is guarding a different
    number from the one the scheduler will actually consume. One key per bucket plus
    the reserved edge shape, all distinct."""
    for buckets in ((8,), (2, 4, 8), (1, 2, 3, 4, 5, 6, 7)):
        config = _config(buckets, elastic=len(buckets) > 1)
        plan = CapturePlan.for_config(config)
        assert len(plan.reserved_keys) == graph_keys_required({CHUNK.ms: buckets})
        assert len(set(plan.reserved_keys)) == len(plan.reserved_keys)
        assert len(plan.keys) == len(buckets)


def test_a_bucket_list_that_does_not_fit_the_budget_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"9.*max_graphs=8"):
        _config(tuple(range(1, 9)), elastic=True)


def test_a_steady_shape_nobody_captured_is_refused_rather_than_stepped() -> None:
    """The keying guard. A batch size outside the captured set means NeMo would capture
    inside the tick, or run eager for the life of the process; both are silent."""
    loop, _pipeline = _loop((8,))
    assert loop.capture.steady_graph(8) is True
    with pytest.raises(UncapturedShape, match=r"160ms x 7"):
        loop.capture.steady_graph(7)


def test_warmup_runs_once_and_the_graphs_are_retained() -> None:
    loop, pipeline = _loop((8,))
    with pytest.raises(ConfigError, match="runs once"):
        loop.capture.warmup(loop.scheduler.pads)
    assert pipeline.captures == list(loop.captured_graphs)
