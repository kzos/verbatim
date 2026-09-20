# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Per-session phrase lists, from the session options down to NeMo's seam.

No torch and no NeMo: the fake in ``verbatim.pipelines.nemo_fake`` stands where the
cache-aware RNNT pipeline stands and holds its biasing rules -- an arena entry is taken
inside the step and given back only on an ``is_last`` frame, and ``delete_state`` frees
nothing. That is what makes the leak on an aborted session reproducible here.

Every way this feature can fail quietly has a test, because every one of them returns a
transcript the client cannot tell from a biased one: a pipeline that cannot bias, a
decoder built without the arena, a boundary that cannot release, a session that sends
phrases to a server without ``--biasing``, and a stream that dies mid-utterance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import InvalidArgument
from verbatim.core.types import PcmFrame
from verbatim.pipelines import registry
from verbatim.pipelines.cache_aware import _biasing_seam
from verbatim.pipelines.cache_aware_ctc import CacheAwareCTCAdapter
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter
from verbatim.pipelines.nemo_fake import (
    FakeCacheAwareCTCPipeline,
    FakeCacheAwareRNNTPipeline,
    boundary_for,
)
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, pipeline_config
from verbatim.protocols.base import Phrase, SessionOptions
from verbatim.scheduler.graph_budget import ConfigError

pytestmark = pytest.mark.cpu

CHUNK = ChunkMode(160)
N = CHUNK.samples
BUCKET = 4

PHRASES = (Phrase("metformin"), Phrase("E11.9", boost=3.0))


def _pipeline(**kwargs: Any) -> FakeCacheAwareRNNTPipeline:
    return FakeCacheAwareRNNTPipeline(CHUNK.ms, num_slots=32, **kwargs)


def _adapter(pipeline: Any, *, biasing: bool = True) -> CacheAwareRNNTAdapter:
    return CacheAwareRNNTAdapter(
        CHUNK,
        boundary_for(pipeline),
        buckets=(BUCKET,),
        required_slots=1,
        biasing=biasing,
    )


def _speech(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=N).astype(np.float32)


def _frame(stream_id: int, samples: np.ndarray, *, is_last: bool = False) -> PcmFrame:
    return PcmFrame(
        stream_id=stream_id,
        samples=samples,
        is_first=False,
        is_last=is_last,
        valid_samples=N,
    )


def _options(**kwargs: Any) -> SessionOptions:
    return SessionOptions(chunk_ms=CHUNK.ms, **kwargs)


# --- what reaches NeMo -------------------------------------------------------------


def test_a_sessions_phrases_reach_nemos_request_options() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(7, _options(phrases=PHRASES, boost=2.0))

    state = pipeline.get_state(7)
    assert state is not None
    cfg = state.options.biasing_cfg
    assert cfg is not None
    assert cfg.phrases == (("metformin", None), ("E11.9", 3.0))
    # The session's own weight is the multi-model alpha; a phrase's own weight rides
    # on the phrase, because Riva carries it per context and NeMo takes it per phrase.
    assert cfg.alpha == 2.0


def test_a_session_without_phrases_carries_no_biasing_request() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(7, _options())
    state = pipeline.get_state(7)
    assert state is not None
    assert state.options.biasing_cfg is None
    assert not state.has_biasing_request()


def test_pad_rows_never_carry_a_phrase_list() -> None:
    """A pad row exists to hold the batch's shape. If it carried a vocabulary it would
    take an arena entry for the life of the process and put it in every steady batch."""
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(1, _options(phrases=PHRASES))
    adapter.transcribe_step(
        [_frame(1, _speech()), _frame(-1, np.zeros(N, dtype=np.float32))],
        keep_all_outputs=False,
        graph=False,
    )
    pad_state = pipeline.get_state(-1)
    assert pad_state is not None
    assert not pad_state.has_biasing_request()
    assert pipeline.biasing_arena is not None
    assert len(pipeline.biasing_arena.active) == 1


def test_the_phrase_list_changes_the_transcript_and_only_its_own_streams() -> None:
    """The positive control, at the seam: same audio, different vocabulary, different
    text -- and a neighbour's list never moves a stream that sent none."""
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(1, _options(phrases=PHRASES))
    adapter.open_stream(2, _options())
    audio = _speech()
    results = adapter.transcribe_step(
        [_frame(1, audio, is_last=True), _frame(2, audio, is_last=True)],
        keep_all_outputs=True,
        graph=False,
    )
    biased, plain = results[0].final_text, results[1].final_text
    assert biased and plain
    assert biased != plain

    # And the unbiased stream alone reads exactly as it did beside a biased one.
    alone = _pipeline(per_stream_biasing=True)
    alone_adapter = _adapter(alone)
    alone_adapter.open_stream(2, _options())
    solo = alone_adapter.transcribe_step(
        [_frame(2, audio, is_last=True)], keep_all_outputs=True, graph=False
    )
    assert solo[0].final_text == plain


# --- the arena lifecycle -----------------------------------------------------------


def test_the_arena_entry_is_taken_on_the_step_not_at_open() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(1, _options(phrases=PHRASES))
    assert pipeline.biasing_arena is not None
    assert pipeline.biasing_arena.active == set()

    adapter.transcribe_step([_frame(1, _speech())], keep_all_outputs=False, graph=False)
    assert len(pipeline.biasing_arena.active) == 1


def test_a_final_frame_gives_the_arena_entry_back() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(1, _options(phrases=PHRASES))
    adapter.transcribe_step([_frame(1, _speech())], keep_all_outputs=False, graph=False)
    adapter.transcribe_step(
        [_frame(1, _speech(2), is_last=True)], keep_all_outputs=True, graph=False
    )
    adapter.close_stream(1)
    assert pipeline.biasing_arena is not None
    assert pipeline.biasing_arena.active == set()


def test_a_stream_that_aborts_mid_utterance_does_not_leak_its_arena_entry() -> None:
    """NeMo frees the entry only on an ``is_last`` frame and ``delete_state`` frees
    nothing, so without ``close_stream`` releasing it, every dropped socket, idle
    deadline and failed step would cost one entry for the life of the process."""
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    for stream_id in range(1, 6):
        adapter.open_stream(stream_id, _options(phrases=PHRASES))
        adapter.transcribe_step(
            [_frame(stream_id, _speech(stream_id))], keep_all_outputs=False, graph=False
        )
        adapter.close_stream(stream_id)  # no is_last: the client vanished
    assert pipeline.biasing_arena is not None
    assert pipeline.biasing_arena.registrations == 5
    assert pipeline.biasing_arena.active == set()


def test_closing_a_stream_twice_is_safe() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline)
    adapter.open_stream(1, _options(phrases=PHRASES))
    adapter.transcribe_step([_frame(1, _speech())], keep_all_outputs=False, graph=False)
    adapter.close_stream(1)
    adapter.close_stream(1)
    assert pipeline.biasing_arena is not None
    assert pipeline.biasing_arena.active == set()


def test_a_server_without_biasing_never_touches_the_arena() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline, biasing=False)
    adapter.open_stream(1, _options())
    adapter.transcribe_step([_frame(1, _speech())], keep_all_outputs=False, graph=False)
    adapter.close_stream(1)
    assert pipeline.biasing_arena is not None
    assert pipeline.biasing_arena.registrations == 0


# --- the refusals ------------------------------------------------------------------


def test_phrases_to_a_server_without_biasing_are_refused() -> None:
    pipeline = _pipeline(per_stream_biasing=True)
    adapter = _adapter(pipeline, biasing=False)
    with pytest.raises(InvalidArgument, match="without --biasing"):
        adapter.open_stream(1, _options(phrases=PHRASES))


def test_a_decoder_built_without_the_arena_refuses_to_serve_biasing() -> None:
    """The case that would otherwise log a warning per step and decode unbiased."""
    pipeline = _pipeline()  # built without per_stream_biasing
    with pytest.raises(ConfigError, match="no biasing arena"):
        _adapter(pipeline)


def test_a_ctc_server_refuses_biasing_outright() -> None:
    pipeline = FakeCacheAwareCTCPipeline(CHUNK.ms, num_slots=32, per_stream_biasing=True)
    with pytest.raises(ConfigError, match="cannot serve it"):
        CacheAwareCTCAdapter(
            CHUNK,
            boundary_for(pipeline),
            buckets=(BUCKET,),
            required_slots=1,
            biasing=True,
        )


def test_a_boundary_that_cannot_release_refuses_to_serve_biasing() -> None:
    import dataclasses

    pipeline = _pipeline(per_stream_biasing=True)
    boundary = dataclasses.replace(boundary_for(pipeline), release_biasing=None)
    with pytest.raises(ConfigError, match="cannot release"):
        CacheAwareRNNTAdapter(CHUNK, boundary, buckets=(BUCKET,), required_slots=1, biasing=True)


def test_the_adapter_reports_biasing_from_what_it_was_built_with() -> None:
    assert _adapter(_pipeline(per_stream_biasing=True)).biasing is True
    assert _adapter(_pipeline(per_stream_biasing=True), biasing=False).biasing is False


def test_the_registry_carries_the_engine_configs_biasing_flag() -> None:
    config = EngineConfig(
        chunk=CHUNK,
        buckets=(BUCKET,),
        calibrated_ceiling=BUCKET,
        edge_batch=2,
        pipeline="cache_aware_rnnt",
        biasing=True,
    )
    pipeline = FakeCacheAwareRNNTPipeline(
        CHUNK.ms, num_slots=config.num_slots, per_stream_biasing=True
    )
    adapter = registry.build_for(config, boundary=boundary_for(pipeline))
    assert adapter.biasing is True


def test_the_seam_is_absent_without_nemo_installed() -> None:
    """On a machine with no NeMo the seam reports nothing rather than a stub, which is
    what makes the adapter's refusal fire instead of a silent unbiased decode."""
    assert _biasing_seam(object()) == (None, None)


# --- what the built config asks NeMo for -------------------------------------------


def _spec(**overrides: Any) -> NeMoPipelineSpec:
    args: dict[str, Any] = {
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "chunk": CHUNK,
        "att_context": (70, 1),
        "num_slots": 32,
        "batch_size": BUCKET,
    }
    args.update(overrides)
    return NeMoPipelineSpec(**args)


def test_biasing_is_off_in_the_built_config_unless_asked_for() -> None:
    greedy = pipeline_config(_spec())["asr"]["decoding"]["greedy"]
    assert greedy["enable_per_stream_biasing"] is False


def test_biasing_on_reaches_nemos_greedy_block_on_the_batched_strategy() -> None:
    decoding = pipeline_config(_spec(enable_per_stream_biasing=True))["asr"]["decoding"]
    assert decoding["greedy"]["enable_per_stream_biasing"] is True
    # Plain `greedy` never receives the flag upstream and never complains about it.
    assert decoding["strategy"] == "greedy_batch"
    assert decoding["greedy"]["loop_labels"] is True


def test_the_deployment_wide_boosting_tree_stays_empty() -> None:
    """One tree for the whole process would fuse every tenant's vocabulary into every
    other tenant's decode. A session's list travels on its own request instead."""
    greedy = pipeline_config(_spec(enable_per_stream_biasing=True))["asr"]["decoding"]["greedy"]
    assert greedy["boosting_tree"]["key_phrases_list"] is None
    assert greedy["boosting_tree"]["key_phrase_items_list"] is None
    assert greedy["boosting_tree_alpha"] == 0.0


def test_a_ctc_spec_refuses_biasing() -> None:
    with pytest.raises(ConfigError, match="decoding='rnnt'"):
        _spec(decoding="ctc", enable_per_stream_biasing=True)


def test_the_inert_per_stream_biasing_defaults_block_is_not_written() -> None:
    """NeMo reads it only on the offline manifest path; the pipeline builder never
    does. A block that looks like it sets the tokenisation mode and does not is the
    defect this project keeps finding."""
    assert "per_stream_biasing_defaults" not in pipeline_config(_spec())["asr"]
