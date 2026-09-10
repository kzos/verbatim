# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The NeMo adapter against a fake standing at NeMo's own seam.

Nothing here needs torch or NeMo. What is exercised is the adapter's translation:
our frames to NeMo frames, our options to NeMo request options, NeMo step outputs
to our step results, and the stream lifecycle NeMo drives from ``is_first`` and
``is_last``.
"""

from __future__ import annotations

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import InvalidArgument
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.core.types import PcmFrame
from verbatim.pipelines import registry
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter, NeMoBoundary
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.pipelines.nemo_fake import FakeCacheAwarePipeline, FakeFrame, boundary_for
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.graph_budget import ConfigError
from verbatim.scheduler.tick import TickLoop

CHUNK = ChunkMode(160)
N = CHUNK.samples
BUCKET = 4


def _config(**overrides) -> EngineConfig:
    args: dict = {
        "chunk": CHUNK,
        "buckets": (BUCKET,),
        "calibrated_ceiling": BUCKET,
        "edge_batch": 2,
        "pipeline": "cache_aware_rnnt",
    }
    args.update(overrides)
    return EngineConfig(**args)


def _adapter(
    config: EngineConfig | None = None, **kwargs
) -> tuple[CacheAwareRNNTAdapter, FakeCacheAwarePipeline]:
    config = config or _config()
    pipeline = FakeCacheAwarePipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareRNNTAdapter(
        config.chunk,
        boundary_for(pipeline),
        buckets=config.buckets or (),
        required_slots=config.num_slots,
        stop_history_eou_ms=config.stop_history_eou_ms,
        **kwargs,
    )
    return adapter, pipeline


def _speech(seed: int, chunks: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, size=chunks * N).astype(np.float32)


def _silence(chunks: int = 1) -> np.ndarray:
    return np.zeros(chunks * N, dtype=np.float32)


def _frame(
    stream_id: int, samples: np.ndarray, *, is_first: bool = False, is_last: bool = False
) -> PcmFrame:
    valid = int(np.count_nonzero(samples != 0)) if is_last else len(samples)
    return PcmFrame(
        stream_id=stream_id,
        samples=samples,
        is_first=is_first,
        is_last=is_last,
        valid_samples=valid if is_last else len(samples),
    )


def _pad(stream_id: int) -> PcmFrame:
    return PcmFrame(
        stream_id=stream_id, samples=_silence(), is_first=False, is_last=False, valid_samples=N
    )


# --- construction ---------------------------------------------------------------------


def test_chunk_mode_must_match_the_pipelines_chunk_size() -> None:
    pipeline = FakeCacheAwarePipeline(560, num_slots=64)
    with pytest.raises(ConfigError, match="does not match"):
        CacheAwareRNNTAdapter(CHUNK, boundary_for(pipeline), buckets=(4,), required_slots=8)


def test_sample_rate_must_be_16k() -> None:
    pipeline = FakeCacheAwarePipeline(CHUNK.ms, sample_rate=8000, num_slots=64)
    with pytest.raises(ConfigError, match="sample rate"):
        CacheAwareRNNTAdapter(CHUNK, boundary_for(pipeline), buckets=(4,), required_slots=8)


def test_pipeline_slots_must_cover_the_schedulers_need() -> None:
    config = _config()
    pipeline = FakeCacheAwarePipeline(CHUNK.ms, num_slots=config.num_slots - 1)
    with pytest.raises(ConfigError, match="No free slots"):
        CacheAwareRNNTAdapter(
            CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
        )


# --- open_stream and the request options ----------------------------------------------


def test_open_stream_creates_nemo_state_with_the_sessions_endpointing_silence() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, SessionOptions(stop_history_eou_ms=480, language_code="de-DE"))
    adapter.open_stream(2, SessionOptions())
    assert pipeline.get_state(1) is not None
    assert pipeline.get_state(1).options.stop_history_eou == 480
    assert pipeline.get_state(1).options.language_code == "de-DE"
    # No per-session value: the engine default rides instead.
    assert pipeline.get_state(2).options.stop_history_eou == 800
    assert adapter.open_streams == frozenset({1, 2})


def test_a_server_pinned_language_code_overrides_the_sessions() -> None:
    adapter, pipeline = _adapter(language_code="en-US")
    adapter.open_stream(1, SessionOptions(language_code="fr-FR"))
    assert pipeline.get_state(1).options.language_code == "en-US"


def test_open_stream_refuses_pad_ids_and_double_opens() -> None:
    adapter, _ = _adapter()
    with pytest.raises(InvalidArgument):
        adapter.open_stream(-1, None)
    adapter.open_stream(1, None)
    with pytest.raises(InvalidArgument, match="already open"):
        adapter.open_stream(1, None)


def test_a_failed_open_leaves_no_state_behind() -> None:
    class RefusingPipeline(FakeCacheAwarePipeline):
        def init_state(self, stream_id, options):
            if options is not None and options.language_code == "xx-XX":
                raise ValueError("Language code 'xx-XX' not found in prompt dictionary")
            return super().init_state(stream_id, options)

    config = _config()
    pipeline = RefusingPipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareRNNTAdapter(
        CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
    )
    with pytest.raises(ValueError, match="prompt dictionary"):
        adapter.open_stream(1, SessionOptions(language_code="xx-XX"))
    assert adapter.open_streams == frozenset()
    assert pipeline.get_state(1) is None


# --- the frames NeMo receives ---------------------------------------------------------


def test_first_frame_is_flagged_first_and_carries_the_options_once() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, SessionOptions(stop_history_eou_ms=320))
    adapter.transcribe_step([_frame(1, _speech(1), is_first=True)], keep_all_outputs=False)
    adapter.transcribe_step([_frame(1, _speech(2))], keep_all_outputs=False)
    first, second = pipeline.seen
    assert (first.is_first, first.is_last, first.length) == (True, False, N)
    assert first.options is not None and first.options.stop_history_eou == 320
    assert (second.is_first, second.is_last, second.options) == (False, False, None)


def test_a_drain_frame_reaches_nemo_with_its_valid_length_and_is_last() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    tail = np.zeros(N, dtype=np.float32)
    tail[:100] = 0.25
    frame = PcmFrame(stream_id=1, samples=tail, is_first=True, is_last=True, valid_samples=100)
    adapter.transcribe_step([frame], keep_all_outputs=True)
    (seen,) = pipeline.seen
    assert seen.is_last is True
    assert seen.length == 100
    assert seen.size - seen.valid_size == N - 100  # NeMo's right padding for this row
    assert pipeline.get_state(1) is None  # NeMo deleted the state on is_last


def test_steady_pad_rows_are_opened_once_and_never_end() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    batch = [_frame(1, _speech(3), is_first=True), _pad(-1), _pad(-2)]
    adapter.transcribe_step(batch, keep_all_outputs=False)
    adapter.transcribe_step([_frame(1, _speech(4)), _pad(-1), _pad(-2)], keep_all_outputs=False)
    pads = [f for f in pipeline.seen if f.stream_id < 0]
    assert [(f.stream_id, f.is_first, f.is_last) for f in pads] == [
        (-1, True, False),
        (-2, True, False),
        (-1, False, False),
        (-2, False, False),
    ]
    assert pipeline.get_state(-1) is not None
    assert pipeline.live_slots == 3


def test_edge_pad_rows_are_one_shot_so_the_final_sub_batch_keeps_its_shape() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    last = PcmFrame(stream_id=1, samples=_speech(5), is_first=True, is_last=True, valid_samples=N)
    adapter.transcribe_step([last, _pad(-3)], keep_all_outputs=True)
    pad = next(f for f in pipeline.seen if f.stream_id == -3)
    assert (pad.is_first, pad.is_last) == (True, True)
    assert pipeline.get_state(-3) is None  # created and deleted inside one step
    assert pipeline.live_slots == 0


def test_a_stream_that_was_never_opened_is_refused_before_nemo_sees_it() -> None:
    adapter, pipeline = _adapter()
    with pytest.raises(InvalidArgument, match="without open_stream"):
        adapter.transcribe_step([_frame(7, _speech(6), is_first=True)], keep_all_outputs=False)
    assert pipeline.seen == []
    assert pipeline.step_calls == 0


def test_outputs_must_come_back_in_row_order() -> None:
    class ShufflingPipeline(FakeCacheAwarePipeline):
        def transcribe_step(self, requests):
            return list(reversed(super().transcribe_step(requests)))

    config = _config()
    pipeline = ShufflingPipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareRNNTAdapter(
        CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
    )
    adapter.open_stream(1, None)
    adapter.open_stream(2, None)
    with pytest.raises(RuntimeError, match="arrived in the row"):
        adapter.transcribe_step(
            [_frame(1, _speech(8), is_first=True), _frame(2, _speech(9), is_first=True)],
            keep_all_outputs=False,
        )


# --- the results the tick loop gets back -----------------------------------------------


def test_partials_grow_per_speech_frame_and_the_clock_fields_are_left_to_the_loop() -> None:
    adapter, _ = _adapter()
    adapter.open_stream(1, None)
    first = adapter.transcribe_step([_frame(1, _speech(10), is_first=True)], keep_all_outputs=False)
    second = adapter.transcribe_step([_frame(1, _speech(11))], keep_all_outputs=False)
    assert len(first) == 1 and len(second) == 1
    assert first[0].partial_text.count(" ") == 0 and first[0].partial_text
    assert second[0].partial_text.startswith(first[0].partial_text + " ")
    assert first[0].final_text is None and second[0].final_text is None
    for result in (*first, *second):
        assert result.audio_processed_s == 0.0
        assert result.valid_samples == 0
        assert result.is_last is False
        assert result.eager is False


def test_silence_of_the_endpointing_history_produces_a_final_without_a_last_frame() -> None:
    adapter, _ = _adapter()
    adapter.open_stream(1, SessionOptions(stop_history_eou_ms=3 * CHUNK.ms))
    adapter.transcribe_step([_frame(1, _speech(12), is_first=True)], keep_all_outputs=False)
    adapter.transcribe_step([_frame(1, _speech(13))], keep_all_outputs=False)
    finals = []
    for _ in range(3):
        (result,) = adapter.transcribe_step([_frame(1, _silence())], keep_all_outputs=False)
        finals.append(result.final_text)
    assert finals[:2] == [None, None]
    assert finals[2] is not None and finals[2].count(" ") == 1  # two words, one final
    (after,) = adapter.transcribe_step([_frame(1, _silence())], keep_all_outputs=False)
    assert after.final_text is None and after.partial_text == ""


def test_a_final_carries_integer_millisecond_words_and_their_mean_confidence() -> None:
    adapter, _ = _adapter()
    adapter.open_stream(1, None)
    adapter.transcribe_step([_frame(1, _speech(14), is_first=True)], keep_all_outputs=False)
    last = PcmFrame(stream_id=1, samples=_speech(15), is_first=False, is_last=True, valid_samples=N)
    (result,) = adapter.transcribe_step([last], keep_all_outputs=True)
    assert result.final_text is not None and result.final_text.count(" ") == 1
    assert [(w.start_ms, w.end_ms) for w in result.words] == [(0, 160), (160, 320)]
    assert result.words[0].word == result.final_text.split()[0]
    assert result.confidence == pytest.approx(0.9)
    assert result.eager is True


def test_a_last_frame_with_nothing_said_still_ends_with_an_empty_final() -> None:
    adapter, _ = _adapter()
    adapter.open_stream(1, None)
    last = PcmFrame(stream_id=1, samples=_silence(), is_first=True, is_last=True, valid_samples=0)
    (result,) = adapter.transcribe_step([last], keep_all_outputs=True)
    assert result.final_text == ""
    assert result.words == ()


def test_step_costs_are_measured_wall_time() -> None:
    adapter, _ = _adapter()
    adapter.open_stream(1, None)
    assert adapter.step_ms == 0.0 and adapter.edge_step_ms == 0.0
    adapter.transcribe_step([_frame(1, _speech(16), is_first=True)], keep_all_outputs=False)
    assert adapter.step_ms > 0.0 and adapter.edge_step_ms == 0.0
    last = PcmFrame(stream_id=1, samples=_speech(17), is_first=False, is_last=True, valid_samples=N)
    adapter.transcribe_step([last], keep_all_outputs=True)
    assert adapter.edge_step_ms > 0.0


# --- close_stream and slot release -----------------------------------------------------


def test_close_stream_releases_nemo_slots_only_for_a_stream_that_never_ended() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    adapter.open_stream(2, None)
    adapter.transcribe_step(
        [_frame(1, _speech(18), is_first=True), _frame(2, _speech(19), is_first=True)],
        keep_all_outputs=False,
    )
    last = PcmFrame(stream_id=2, samples=_silence(), is_first=False, is_last=True, valid_samples=0)
    adapter.transcribe_step([last], keep_all_outputs=True)
    adapter.close_stream(2)  # ended through NeMo: nothing to release
    adapter.close_stream(1)  # stepped, never ended: the failed-session shape
    adapter.close_stream(3)  # never opened at all: a no-op
    assert pipeline.released == [1]
    assert pipeline.live_slots == 0
    assert adapter.open_streams == frozenset()


def test_the_tick_loop_never_steps_a_stream_whose_open_failed() -> None:
    class RefusingPipeline(FakeCacheAwarePipeline):
        def init_state(self, stream_id, options):
            if stream_id == 2:
                raise ValueError("refused")
            return super().init_state(stream_id, options)

    config = _config()
    pipeline = RefusingPipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareRNNTAdapter(
        CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
    )
    loop = TickLoop(config, adapter, SessionRegistry(), clock=SimulatedClock())
    sessions = []
    for sid in (1, 2):
        session = Session(sid, CHUNK, ring_seconds=8.0, options=SessionOptions())
        session.configure()
        assert loop.admit_session(session).admitted
        assert session.ring.write(_speech(20 + sid, 3)) == 3 * N
        sessions.append(session)
    results = loop.run_tick()
    assert [r.stream_id for r in results] == [1]
    assert 2 not in {f.stream_id for f in pipeline.seen}
    assert dict(loop.drain_errors()).keys() == {2}
    assert sessions[1].state.value == "CLOSED"
    assert loop.registry.live == 1
    assert pipeline.get_state(2) is None


# --- the registry --------------------------------------------------------------------


def test_registry_builds_the_fake_by_default_and_the_adapter_by_name() -> None:
    fake = registry.build_for(EngineConfig(chunk=CHUNK, buckets=(BUCKET,)))
    assert isinstance(fake, FakePipelineAdapter)
    config = _config()
    pipeline = FakeCacheAwarePipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = registry.build_for(config, boundary=boundary_for(pipeline))
    assert isinstance(adapter, CacheAwareRNNTAdapter)
    assert adapter.supported_buckets() == (BUCKET,)
    assert "fake" in registry.names() and "cache_aware_rnnt" in registry.names()


def test_registry_refuses_an_unknown_name_and_an_adapter_without_a_pipeline() -> None:
    with pytest.raises(InvalidArgument, match="unknown pipeline 'nope'"):
        registry.build("nope", _config())
    with pytest.raises(InvalidArgument, match="needs a built NeMo"):
        registry.build_for(_config())


def test_registry_loads_a_third_party_entry_point_on_first_use(monkeypatch) -> None:
    class _EntryPoint:
        name = "third-party"

        @staticmethod
        def load():
            return lambda config, **kwargs: FakePipelineAdapter(
                config.chunk, buckets=config.buckets or ()
            )

    monkeypatch.setattr(registry, "entry_points", lambda group: [_EntryPoint()])
    built = registry.build("third-party", EngineConfig(chunk=CHUNK, buckets=(BUCKET,)))
    assert isinstance(built, FakePipelineAdapter)
    assert "third-party" in registry.names()


def test_the_boundary_for_nemo_is_built_lazily() -> None:
    # No NeMo here: the constructor must not import it until asked to bind a pipeline.
    boundary = NeMoBoundary(
        pipeline=FakeCacheAwarePipeline(CHUNK.ms),
        make_frame=FakeFrame,
        make_options=lambda **kw: kw,
        to_samples=lambda s: s,
    )
    assert boundary.release_stream is None
    with pytest.raises((ImportError, ModuleNotFoundError)):
        NeMoBoundary.from_pipeline(object())
