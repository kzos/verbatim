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
    UPSTREAM_WARMUP_STEPS,
    CaptureController,
    CaptureNotRetained,
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
    # Upstream captures on the step whose count exceeds its own warmup_steps, so the
    # capture is the LAST of the warm-up steps at a shape and none of them is a replay.
    # Nothing beyond the two buckets is stepped: warm-up is capture, not traffic.
    assert pipeline.steps == 2 * GRAPH_WARMUP_STEPS
    assert pipeline.replays == []


def test_warmup_captures_pad_rows_only_so_no_session_audio_is_involved() -> None:
    loop, pipeline = _loop((8,))
    del loop
    assert pipeline.calls_seen == [(8, False, True)] * GRAPH_WARMUP_STEPS
    # No session exists yet: pad rows are the whole of every warm-up batch.
    assert pipeline.steps == GRAPH_WARMUP_STEPS


def test_each_shape_is_warmed_often_enough_for_upstream_to_capture_it() -> None:
    """Upstream captures a shape on the call whose count at that shape *exceeds*
    ``warmup_steps``, so warming exactly ``warmup_steps`` times captures nothing.

    This is the test that was not here. It used to assert ``>= 3`` against upstream's
    default of 3, which is exactly the off-by-one that shipped: on the B300 NeMo held
    zero graphs after a three-step warm-up, and one after the next step. Every bucket
    gets the same number of steps."""
    assert GRAPH_WARMUP_STEPS > UPSTREAM_WARMUP_STEPS
    loop, pipeline = _loop((4, 8), elastic=True)
    del loop
    per_shape = Counter(size for size, _keep, _graph in pipeline.calls_seen)
    assert per_shape == {4: GRAPH_WARMUP_STEPS, 8: GRAPH_WARMUP_STEPS}


def test_warmup_that_leaves_the_runtime_holding_nothing_is_refused() -> None:
    """The guard that was missing. Upstream falls back to eager in silence -- on a
    capture failure and for any step its own rules disqualify -- so a warm-up that
    records a capture it did not make is how a server comes up announcing the graph
    path and runs every tick eager.

    The fake is given upstream's threshold one higher than the controller's warm-up,
    which is precisely the shape of the shipped defect, and the controller must refuse
    rather than record the key."""
    pipeline = FakePipelineAdapter(
        CHUNK,
        buckets=(8,),
        graphs=GraphCapability.graphed(),
        graph_warmup_steps=GRAPH_WARMUP_STEPS,
    )
    controller = CaptureController(_config((8,)), pipeline)
    with pytest.raises(CaptureNotRetained, match="holding 0 graphs"):
        controller.warmup(PadPool(16, CHUNK))
    assert controller.captured_keys == frozenset()


def test_a_warmup_that_did_capture_is_accepted() -> None:
    """The positive control: without it the refusal above would pass on a controller
    that refused every warm-up."""
    pipeline = _pipeline((8,))
    controller = CaptureController(_config((8,)), pipeline)
    captured = controller.warmup(PadPool(16, CHUNK))
    assert captured == (controller.plan.steady_key(8),)
    assert pipeline.retained_graphs() == 1


def test_an_adapter_that_claims_the_graph_path_and_cannot_count_is_refused() -> None:
    """The guard that disabled itself. ``retained_graphs()`` returning None means "no
    answer", and warm-up used to take that at its word and record the key -- which skips
    the only check in the loop, in exactly the case the check exists for.

    The reachable instance is a pipeline the installed package can graph and this object
    cannot: the package-level probe says available, nothing is attached to the object, so
    nothing could ever observe a capture. "Cannot tell" is honest from an eager adapter,
    which never reaches this loop, and inadmissible from a graphed one."""

    class _Speechless(FakePipelineAdapter):
        def retained_graphs(self) -> int | None:
            return None

    pipeline = _Speechless(CHUNK, buckets=(8,), graphs=GraphCapability.graphed())
    controller = CaptureController(_config((8,)), pipeline)
    with pytest.raises(CaptureNotRetained, match="cannot report how many graphs"):
        controller.warmup(PadPool(16, CHUNK))
    assert controller.captured_keys == frozenset()


def test_an_eager_adapter_that_cannot_count_graphs_warms_up_fine() -> None:
    """The other side of it: the refusal above must not reach an adapter that never
    claimed the graph path. Eager warm-up captures nothing and asks nothing."""

    class _Speechless(FakePipelineAdapter):
        def retained_graphs(self) -> int | None:
            return None

    pipeline = _Speechless(
        CHUNK, buckets=(8,), graphs=GraphCapability.eager("this fake runs no graphs")
    )
    controller = CaptureController(_config((8,)), pipeline)
    assert controller.warmup(PadPool(16, CHUNK)) == ()
    assert controller.mode == "eager"


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
    # Captured exactly once, on the last warm-up step, and replayed by every tick after
    # it: the shape never changes, so the key never does either. No live tick pays for
    # a capture, which is the whole point of warming up.
    assert pipeline.captures == [key]
    assert pipeline.replays == [key] * 3
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


def test_ragged_padding_and_the_graph_path_cannot_both_be_asked_for() -> None:
    """The control arm has no shape to capture.

    A ragged steady batch tracks live occupancy, so its shape changes every time a
    session joins or leaves and upstream's key changes with it. Asking for both is not a
    trade-off to resolve silently in one direction -- a run that dropped one of them
    would publish a row naming both -- so it is refused where both numbers are in one
    place.
    """
    config = EngineConfig(
        chunk=CHUNK, buckets=(8,), calibrated_ceiling=8, edge_batch=8, padding="ragged"
    )
    with pytest.raises(ConfigError, match="no shape to capture"):
        CaptureController(config, _pipeline((8,)))


def test_ragged_padding_is_fine_on_the_eager_arm() -> None:
    """The positive control: the refusal above must not reach the arm the ragged
    control is actually run on."""
    config = EngineConfig(
        chunk=CHUNK, buckets=(8,), calibrated_ceiling=8, edge_batch=8, padding="ragged"
    )
    pipeline = _pipeline((8,), graphs=GraphCapability.eager("the control arm runs eager"))
    controller = CaptureController(config, pipeline)
    assert controller.mode == "eager"
    assert controller.warmup(PadPool(16, CHUNK)) == ()
