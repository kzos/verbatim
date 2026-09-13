# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The CTC adapter against a fake standing at NeMo's own seam.

Nothing here needs torch or NeMo. Two things are under test. First, that the CTC
adapter satisfies the same ``PipelineAdapter`` contract the RNNT one does, so the
engine, the slot table, admission and both wires take it unchanged -- the lifecycle
tests below drive the same seam the RNNT suite drives, on this class. Second, the
places NeMo's two cache-aware pipelines genuinely differ: which greedy decoder the
bound pipeline declares, and that CTC has no prompt path, so a language code is
recorded and acted on by nothing.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import InvalidArgument
from verbatim.core.registry import SessionRegistry
from verbatim.core.session import Session
from verbatim.core.types import PcmFrame
from verbatim.pipelines import registry
from verbatim.pipelines.cache_aware_ctc import CacheAwareCTCAdapter
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter
from verbatim.pipelines.nemo_fake import (
    FakeCacheAwareCTCPipeline,
    FakeCacheAwarePipeline,
    FakeCacheAwareRNNTPipeline,
    boundary_for,
)
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, pipeline_config
from verbatim.protocols.base import SessionOptions
from verbatim.scheduler.clock import SimulatedClock
from verbatim.scheduler.graph_budget import ConfigError
from verbatim.scheduler.tick import TickLoop

pytestmark = pytest.mark.cpu

CHUNK = ChunkMode(160)
N = CHUNK.samples
BUCKET = 4


def _config(**overrides: Any) -> EngineConfig:
    args: dict[str, Any] = {
        "chunk": CHUNK,
        "buckets": (BUCKET,),
        "calibrated_ceiling": BUCKET,
        "edge_batch": 2,
        "pipeline": "cache_aware_ctc",
    }
    args.update(overrides)
    return EngineConfig(**args)


def _adapter(
    config: EngineConfig | None = None, **kwargs: Any
) -> tuple[CacheAwareCTCAdapter, FakeCacheAwareCTCPipeline]:
    config = config or _config()
    pipeline = FakeCacheAwareCTCPipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareCTCAdapter(
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
    return PcmFrame(
        stream_id=stream_id,
        samples=samples,
        is_first=is_first,
        is_last=is_last,
        valid_samples=len(samples),
    )


def _pad(stream_id: int) -> PcmFrame:
    return PcmFrame(
        stream_id=stream_id, samples=_silence(), is_first=False, is_last=False, valid_samples=N
    )


# --- which pipeline is on the other end of the boundary --------------------------------


def test_an_rnnt_pipeline_served_as_ctc_is_refused_rather_than_transcribed() -> None:
    """A crossed wire here does not fail: NeMo's RNNT pipeline drives the same
    ``BasePipeline.transcribe_step``, so it would run, transcribe well, and put CTC on
    the row. The only thing that distinguishes the two at this seam is which greedy
    decoder the pipeline built for itself."""
    config = _config()
    pipeline = FakeCacheAwareRNNTPipeline(CHUNK.ms, num_slots=config.num_slots)
    with pytest.raises(ConfigError, match="greedy_rnnt_decoder") as raised:
        CacheAwareCTCAdapter(
            CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
        )
    assert "--pipeline cache_aware_rnnt" in str(raised.value)


def test_a_ctc_pipeline_served_as_rnnt_is_refused_the_same_way() -> None:
    config = _config()
    pipeline = FakeCacheAwareCTCPipeline(CHUNK.ms, num_slots=config.num_slots)
    with pytest.raises(ConfigError, match="greedy_ctc_decoder") as raised:
        CacheAwareRNNTAdapter(
            CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
        )
    assert "--pipeline cache_aware_ctc" in str(raised.value)


def test_a_pipeline_that_declares_neither_decoder_is_taken_by_both() -> None:
    """The guard fires on a positive contradiction only. A third party's pipeline, or a
    double, that declares neither decoder says nothing to contradict."""
    config = _config()
    for adapter_class in (CacheAwareCTCAdapter, CacheAwareRNNTAdapter):
        pipeline = FakeCacheAwarePipeline(CHUNK.ms, num_slots=config.num_slots)
        built = adapter_class(
            CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
        )
        assert built.chunk == CHUNK


# --- construction ---------------------------------------------------------------------


def test_chunk_mode_must_match_the_pipelines_chunk_size() -> None:
    pipeline = FakeCacheAwareCTCPipeline(560, num_slots=64)
    with pytest.raises(ConfigError, match="does not match"):
        CacheAwareCTCAdapter(CHUNK, boundary_for(pipeline), buckets=(4,), required_slots=8)


def test_pipeline_slots_must_cover_the_schedulers_need() -> None:
    config = _config()
    pipeline = FakeCacheAwareCTCPipeline(CHUNK.ms, num_slots=config.num_slots - 1)
    with pytest.raises(ConfigError, match="No free slots"):
        CacheAwareCTCAdapter(
            CHUNK, boundary_for(pipeline), buckets=(BUCKET,), required_slots=config.num_slots
        )


# --- the language code CTC cannot act on ----------------------------------------------


def test_a_language_code_is_recorded_on_the_request_and_ctc_resolves_no_prompt() -> None:
    """``CacheAwareCTCPipeline.create_state`` calls ``fill_defaults`` without
    ``default_language_code`` and never resolves a prompt index, so a code the
    checkpoint never heard of cannot fail here the way it does on the prompt-enabled
    RNNT path. The adapter still records it: the request options are the record of
    what the client asked for."""

    class PromptResolvingRNNTPipeline(FakeCacheAwareRNNTPipeline):
        """The RNNT contrast: a prompt-enabled checkpoint raises inside ``create_state``."""

        def init_state(self, stream_id: int, options: Any) -> Any:
            if options is not None and options.language_code not in (None, "en-US"):
                raise ValueError(
                    f"Language code '{options.language_code}' not found in prompt dictionary"
                )
            return super().init_state(stream_id, options)

    config = _config()
    ctc_adapter, ctc_pipeline = _adapter(config)
    ctc_adapter.open_stream(1, SessionOptions(language_code="xx-XX"))
    assert ctc_pipeline.get_state(1).options.language_code == "xx-XX"
    assert ctc_adapter.open_streams == frozenset({1})

    rnnt_pipeline = PromptResolvingRNNTPipeline(CHUNK.ms, num_slots=config.num_slots)
    rnnt_adapter = CacheAwareRNNTAdapter(
        CHUNK, boundary_for(rnnt_pipeline), buckets=(BUCKET,), required_slots=config.num_slots
    )
    with pytest.raises(ValueError, match="prompt dictionary"):
        rnnt_adapter.open_stream(1, SessionOptions(language_code="xx-XX"))


# --- open_stream and the request options ----------------------------------------------


def test_open_stream_creates_nemo_state_with_the_sessions_endpointing_silence() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, SessionOptions(stop_history_eou_ms=480))
    adapter.open_stream(2, SessionOptions())
    assert pipeline.get_state(1).options.stop_history_eou == 480
    assert pipeline.get_state(2).options.stop_history_eou == 800  # the engine default
    assert adapter.open_streams == frozenset({1, 2})


def test_open_stream_refuses_pad_ids_and_double_opens() -> None:
    adapter, _ = _adapter()
    with pytest.raises(InvalidArgument):
        adapter.open_stream(-1, None)
    adapter.open_stream(1, None)
    with pytest.raises(InvalidArgument, match="already open"):
        adapter.open_stream(1, None)


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


def test_steady_pad_rows_are_opened_once_and_edge_pad_rows_are_one_shot() -> None:
    """The graph path's whole point: the steady batch keeps one shape because its pads
    never end, and the edge batch keeps its own because NeMo peels finals into their own
    sub-batch and a pad only stays in it by being a final too."""
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    adapter.transcribe_step(
        [_frame(1, _speech(3), is_first=True), _pad(-1), _pad(-2)], keep_all_outputs=False
    )
    adapter.transcribe_step([_frame(1, _speech(4)), _pad(-1), _pad(-2)], keep_all_outputs=False)
    pads = [f for f in pipeline.seen if f.stream_id < 0]
    assert [(f.stream_id, f.is_first, f.is_last) for f in pads] == [
        (-1, True, False),
        (-2, True, False),
        (-1, False, False),
        (-2, False, False),
    ]
    assert pipeline.live_slots == 3

    last = PcmFrame(stream_id=1, samples=_speech(5), is_first=False, is_last=True, valid_samples=N)
    adapter.transcribe_step([last, _pad(-3)], keep_all_outputs=True)
    edge_pad = next(f for f in pipeline.seen if f.stream_id == -3)
    assert (edge_pad.is_first, edge_pad.is_last) == (True, True)
    assert pipeline.get_state(-3) is None  # created and deleted inside one step
    assert pipeline.live_slots == 2  # the two steady pads, and nothing else


def test_a_stream_that_was_never_opened_is_refused_before_nemo_sees_it() -> None:
    """CTC dereferences the missing state later than RNNT does -- in
    ``decode_log_probs``, after the encoder step and after the context manager reset the
    finished rows' slots -- so the blast radius of letting one through is larger, not
    smaller. It never gets that far."""
    adapter, pipeline = _adapter()
    with pytest.raises(InvalidArgument, match="without open_stream"):
        adapter.transcribe_step([_frame(7, _speech(6), is_first=True)], keep_all_outputs=False)
    assert pipeline.seen == []
    assert pipeline.step_calls == 0
    assert pipeline.missing_state_attribute == "label_buffer"


def test_outputs_must_come_back_in_row_order() -> None:
    class ShufflingPipeline(FakeCacheAwareCTCPipeline):
        def transcribe_step(self, requests: Any) -> Any:
            return list(reversed(super().transcribe_step(requests)))

    config = _config()
    pipeline = ShufflingPipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = CacheAwareCTCAdapter(
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


def test_partials_grow_and_the_clock_fields_are_left_to_the_loop() -> None:
    adapter, _ = _adapter()
    adapter.open_stream(1, None)
    first = adapter.transcribe_step([_frame(1, _speech(10), is_first=True)], keep_all_outputs=False)
    second = adapter.transcribe_step([_frame(1, _speech(11))], keep_all_outputs=False)
    assert first[0].partial_text and second[0].partial_text.startswith(first[0].partial_text + " ")
    assert first[0].final_text is None and second[0].final_text is None
    for result in (*first, *second):
        assert result.audio_processed_s == 0.0
        assert result.valid_samples == 0
        assert result.is_last is False
        assert result.eager is False


def test_endpointing_silence_produces_a_final_without_a_last_frame() -> None:
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


# --- a step that raises ----------------------------------------------------------------


def test_a_ctc_step_that_raises_on_a_final_frame_does_not_strand_the_streams_slot() -> None:
    """The RNNT adapter's round-2 finding, on this class. NeMo raises from the encoder,
    after its bufferer and context manager allocated and after the bufferer freed a
    final frame's slot, but before the context manager freed its own or the state was
    deleted. The record of what NeMo holds is therefore written AFTER the step, from
    what NeMo did: bookkeeping written before it would call the final stream gone, and
    ``close_stream`` would skip the release and strand the context slot for good."""
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    adapter.transcribe_step([_frame(1, _speech(30), is_first=True)], keep_all_outputs=False)
    assert pipeline.live_slots == 1
    pipeline.raise_in_encoder = RuntimeError("CUDA error: device-side assert triggered")
    last = PcmFrame(stream_id=1, samples=_speech(31), is_first=False, is_last=True, valid_samples=N)
    with pytest.raises(RuntimeError, match="device-side assert"):
        adapter.transcribe_step([last], keep_all_outputs=True)
    assert pipeline.live_slots == 1  # NeMo never reached its cleanup
    adapter.close_stream(1)  # what fail_live does for every live session
    assert pipeline.released == [1]
    assert pipeline.live_slots == 0
    assert pipeline.get_state(1) is None


def test_a_ctc_step_that_raises_on_a_first_frame_releases_what_nemo_allocated() -> None:
    adapter, pipeline = _adapter()
    adapter.open_stream(1, None)
    pipeline.raise_in_encoder = RuntimeError("encoder failed")
    with pytest.raises(RuntimeError, match="encoder failed"):
        adapter.transcribe_step([_frame(1, _speech(32), is_first=True)], keep_all_outputs=False)
    assert pipeline.live_slots == 1
    adapter.close_stream(1)
    assert pipeline.released == [1]
    assert pipeline.live_slots == 0


# --- the tick loop drives it unchanged -------------------------------------------------


def test_the_tick_loop_runs_sessions_through_the_ctc_adapter_unchanged() -> None:
    """The contract claim, exercised rather than asserted: the engine's loop, registry
    and slot table take this adapter with no change of their own."""
    config = _config()
    adapter, pipeline = _adapter(config)
    loop = TickLoop(config, adapter, SessionRegistry(), clock=SimulatedClock())
    sessions = []
    for sid in (1, 2):
        session = Session(sid, CHUNK, ring_seconds=8.0, options=SessionOptions())
        session.configure()
        assert loop.admit_session(session).admitted
        assert session.ring.write(_speech(40 + sid, 2)) == 2 * N
        sessions.append(session)

    results = loop.run_tick()
    assert sorted(r.stream_id for r in results) == [1, 2]
    assert all(r.partial_text for r in results)
    assert dict(loop.drain_errors()) == {}
    # A transcript is the stream's own audio and never its neighbour's.
    assert results[0].partial_text != results[1].partial_text
    assert pipeline.step_calls == 1
    assert adapter.open_streams == frozenset({1, 2})


# --- selection: the registry, and the config that builds the NeMo pipeline --------------


def test_the_registry_builds_the_ctc_adapter_by_name() -> None:
    config = _config()
    pipeline = FakeCacheAwareCTCPipeline(CHUNK.ms, num_slots=config.num_slots)
    adapter = registry.build_for(config, boundary=boundary_for(pipeline))
    assert isinstance(adapter, CacheAwareCTCAdapter)
    assert adapter.supported_buckets() == (BUCKET,)
    assert "cache_aware_ctc" in registry.names()


def test_the_registry_refuses_the_ctc_adapter_without_a_pipeline_and_names_it() -> None:
    with pytest.raises(InvalidArgument, match="CacheAwareCTCPipeline"):
        registry.build_for(_config())


def test_the_registry_refuses_a_crossed_wire_by_name() -> None:
    config = _config()
    pipeline = FakeCacheAwareRNNTPipeline(CHUNK.ms, num_slots=config.num_slots)
    with pytest.raises(ConfigError, match="cache_aware_rnnt"):
        registry.build_for(config, boundary=boundary_for(pipeline))


def test_the_nemo_config_takes_the_builders_ctc_branch_and_claims_no_rnnt_knobs() -> None:
    """``CacheAwarePipelineBuilder.get_ctc_decoding_cfg`` takes no argument and reads
    nothing from ``asr.decoding``, so writing the RNNT greedy block there would record a
    configuration nothing applies."""
    spec = NeMoPipelineSpec(
        model="nvidia/nemotron-speech-streaming-en-0.6b",
        chunk=CHUNK,
        att_context=(70, 1),
        num_slots=79,
        batch_size=32,
        decoding="ctc",
    )
    cfg = pipeline_config(spec)
    assert cfg["asr_decoding_type"] == "ctc"
    assert cfg["pipeline_type"] == "cache_aware"
    assert cfg["asr"]["decoding"] == {"strategy": "greedy", "preserve_alignments": False}
    assert cfg["return_tail_result"] is False
    assert cfg["streaming"]["use_cache"] is True
    # And the RNNT branch is untouched by the CTC one.
    rnnt = pipeline_config(dataclasses.replace(spec, decoding="rnnt"))
    assert rnnt["asr_decoding_type"] == "rnnt"
    assert rnnt["asr"]["decoding"]["strategy"] == "greedy_batch"


def test_the_spec_refuses_a_decoding_type_nemos_builder_has_no_branch_for() -> None:
    with pytest.raises(ConfigError, match="decoding must be rnnt or ctc"):
        NeMoPipelineSpec(
            model="nvidia/nemotron-speech-streaming-en-0.6b",
            chunk=CHUNK,
            att_context=(70, 1),
            num_slots=79,
            batch_size=32,
            decoding="malsd",
        )
