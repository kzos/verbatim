# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Word confidence on the WebSocket wire, and what ``/readyz`` observes off the built pipeline.

The "off" side is held to bytes, not to a description of them.
``fixtures/ws_confidence_off_golden.json`` was captured from a ``git archive`` export of
the commit it names, before any of this existed: finals and partials from a real
``run_server`` over the fake pipeline, finals over the scripted stub engine, frames built
directly (including words whose confidences are not 1.0, which the old code dropped), and
``/readyz`` bodies. With word confidence off every one of them must come back byte for
byte, and ``/readyz`` may only have grown at the end.

The "on" side follows the value from where the server computes it: a NeMo-shaped fake
pipeline puts known confidences on its final segments, ``cache_aware._words_of`` copies
``float(segment.conf)`` onto each ``Word``, the engine delivers the hypothesis, and the
frame must carry exactly those values as ``"c"``.

The observed facts are read off attributes at the paths NeMo's built objects keep them
(``verbatim.pipelines.observed`` cites each). The fake's encoder deliberately holds an
attention context the settings would not give, so a reading taken from the spec cannot
pass; and the NeMo-backed test at the end runs the same readers over NeMo's own config
and decoder classes, skipped where NeMo is not installed.

``/readyz``'s ``code`` names the package directories the serving process imported. The
answer expected here is computed from this test file's own location, the checkout the
suite belongs to, and never from ``verbatim.__file__``, which is the value under test.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import verbatim_bench
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

import verbatim
from verbatim.config import ChunkMode
from verbatim.core.errors import ErrorCode
from verbatim.core.types import Word
from verbatim.engine import Engine, stub_engine
from verbatim.obs.counters import Counters
from verbatim.obs.metrics import MetricsSnapshot
from verbatim.pipelines import observed as observed_module
from verbatim.pipelines import registry
from verbatim.pipelines.base import PipelineAdapter
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.pipelines.nemo_fake import (
    FakeCacheAwareCTCPipeline,
    FakeCacheAwareRNNTPipeline,
    boundary_for,
)
from verbatim.pipelines.nemo_runtime import (
    WORD_CONFIDENCE_MODES,
    NeMoPipelineSpec,
    pipeline_config,
)
from verbatim.pipelines.observed import (
    ObservedFacts,
    configured_word_confidence,
    disagreement,
    observe,
)
from verbatim.protocols.base import Hypothesis, SessionOptions
from verbatim.protocols.health import (
    CODE_KEYS,
    OBSERVED_KEYS,
    HealthReporter,
    ServiceFacts,
    code_paths,
)
from verbatim.protocols.ws.frames import (
    ErrorFrame,
    FinalFrame,
    PartialFrame,
    SessionFrame,
    confidence_on_wire,
)
from verbatim.protocols.ws.server import WsServer, WsServerConfig
from verbatim.scheduler.graph_budget import ConfigError
from verbatim.serve import Endpoints, ServeSettings, engine_config, run_server

pytestmark = pytest.mark.cpu

GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "ws_confidence_off_golden.json").read_text(
        encoding="utf-8"
    )
)
NEW_READYZ_KEYS = ["word_confidence", "observed", "code"]
NULL_OBSERVED = dict.fromkeys(OBSERVED_KEYS)
#: The checkout this file belongs to, and the two package directories in it that a
#: server started from it must report it imported: ``/readyz``'s ``code`` object.
CHECKOUT = Path(__file__).resolve().parents[1]
OWN_CODE = {
    "verbatim_path": os.path.realpath(CHECKOUT / "src" / "verbatim"),
    "bench_path": os.path.realpath(CHECKOUT / "bench" / "src" / "verbatim_bench"),
}

#: A checkpoint whose family fixes the left context at 70, so the settings of a 160 ms
#: server imply [70, 1]. The NeMo-shaped fake's encoder holds [56, 1] instead: a
#: reading derived from the settings or the spec cannot agree with the built encoder.
NEMO_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
SPEC_CONTEXT = [70, 1]
BUILT_CONTEXT = (56, 1)
#: Per-word confidences the fake puts on its final segments, in word order. 0.3, 0.999,
#: 0.05 and 0.6 are not exact in binary, so a float32 hop on the way changes them. None
#: needs more than 7 decimal places, so a rounding to 7 or more survives them all:
#: ``test_c_is_the_confidence_unrounded_and_unclamped`` holds the values that do not.
CONFIDENCES = (0.125, 0.8671875, 0.3, 0.999, 0.05, 0.6)


# --- helpers ---------------------------------------------------------------------------


def _words(name: str) -> tuple[Word, ...] | None:
    spec = GOLDEN["word_cases"][name]
    if spec is None:
        return None
    return tuple(Word(word=w, start_ms=s, end_ms=e, confidence=c) for w, s, e, c in spec)


def _pcm(chunks: int, extra_bytes: int, seed: int) -> list[bytes]:
    """The capture's PCM: a 31-bit linear congruential generator, no library RNG."""
    state = seed

    def samples(n: int) -> bytes:
        nonlocal state
        values = []
        for _ in range(n):
            state = (1103515245 * state + 12345) % (1 << 31)
            values.append((state >> 16) % 6001 - 3000)
        return np.asarray(values, dtype="<i2").tobytes()

    out = [samples(160 * 16) for _ in range(chunks)]
    if extra_bytes:
        out.append(samples(extra_bytes // 2))
    return out


class _FixedSnapshot:
    """The snapshot the golden ``/readyz`` bodies were rendered from."""

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            chunk_ms=160,
            period_ms=160.0,
            budget_ms=112.0,
            bucket=4,
            started=True,
            running=True,
            dead=False,
            tick_id=42,
            last_tick_age_s=0.01,
            live_sessions=0,
            slots_capacity=8,
            slots_reserved=0,
            slots_free=8,
            ceiling=None,
            calibrated_ceiling=None,
            degradation_level=0,
            consecutive_overruns=0,
            p95_tick_ms=1.0,
            eager_step_fraction=0.0,
            counters=Counters(),
            latency_count=0,
            partial_latency_ms=(0.0, 0.0, 0.0),
            tick_cost_ms=(0.0, 0.0, 0.0),
            last_refusal_reason=None,
        )


def _without_new_keys(body: str, *, tick_id: int | None = None) -> str:
    """``body`` with the added keys removed, re-serialised the way ``health._json`` does.

    Key order survives ``json.loads``, so this equals the golden bytes only if every
    earlier key kept its place and its value. ``tick_id`` is overwritten for a live
    server, whose tick count is the one field that moves between runs.
    """
    doc = json.loads(body)
    kept = {key: value for key, value in doc.items() if key not in NEW_READYZ_KEYS}
    if tick_id is not None:
        kept["tick_id"] = tick_id
    return json.dumps(kept, separators=(",", ":")) + "\n"


async def _get(url: str) -> tuple[int, str]:
    def fetch() -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    return await asyncio.to_thread(fetch)


async def _ready_body(endpoints: Endpoints) -> str:
    """The first 200 ``/readyz`` body: engine started and one tick done."""
    url = f"http://127.0.0.1:{endpoints.ws_port}/readyz"
    for _ in range(400):
        status, body = await _get(url)
        if status == 200:
            return body
        await asyncio.sleep(0.025)
    raise AssertionError(f"/readyz never answered 200: {status} {body}")


@asynccontextmanager
async def _serving(
    settings: ServeSettings, adapter: PipelineAdapter, **kwargs: Any
) -> AsyncIterator[Endpoints]:
    """``run_server`` in a task, yielding its endpoints once it says it is listening."""
    shutdown = asyncio.Event()
    ready: asyncio.Future[Endpoints] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        run_server(settings, adapter, shutdown=shutdown, on_ready=ready.set_result, **kwargs)
    )
    done, _ = await asyncio.wait({task, ready}, timeout=15, return_when=asyncio.FIRST_COMPLETED)
    assert ready in done, f"run_server did not come up: {task}"
    try:
        yield ready.result()
    finally:
        shutdown.set()
        await asyncio.wait_for(task, timeout=15)


async def _session_frames(endpoint: str, query: str, pieces: Sequence[bytes]) -> list[str]:
    """Every text frame after the session frame, raw, until the clean close."""
    raws: list[str] = []
    async with connect(f"{endpoint}?{query}") as ws:
        first = await asyncio.wait_for(ws.recv(), timeout=10)
        assert isinstance(first, str) and json.loads(first)["type"] == "session"
        for piece in pieces:
            await ws.send(piece)
        await ws.send('{"type": "end"}')
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                assert isinstance(raw, str)
                raws.append(raw)
        except ConnectionClosedOK:
            pass
    return raws


def _fake_settings() -> ServeSettings:
    golden = GOLDEN["serve_fake"]["settings"]
    return ServeSettings(
        model=golden["model"],
        chunk=ChunkMode(golden["chunk_ms"]),
        bucket=golden["bucket"],
        pipeline=golden["pipeline"],
        host="127.0.0.1",
        ws_port=0,
        grpc_port=0,
    )


def _nemo_settings(pipeline: str = "cache_aware_rnnt") -> ServeSettings:
    return ServeSettings(
        model=NEMO_MODEL,
        chunk=ChunkMode(160),
        bucket=4,
        pipeline=pipeline,
        host="127.0.0.1",
        ws_port=0,
        grpc_port=0,
    )


def _spec(mode: str) -> NeMoPipelineSpec:
    return NeMoPipelineSpec(
        model=NEMO_MODEL,
        chunk=ChunkMode(160),
        att_context=(SPEC_CONTEXT[0], SPEC_CONTEXT[1]),
        num_slots=64,
        batch_size=4,
        word_confidence=mode,
    )


#: NeMo's library defaults for ``ConfidenceConfig``, which stay in ``confidence_cfg`` when
#: ``preserve_frame_confidence`` is false and the builder copies nothing.
_LIBRARY_DEFAULT = (
    "min",
    {"name": "entropy", "entropy_type": "tsallis", "alpha": 0.33, "entropy_norm": "exp"},
)


def _decoding_cfg(mode: str) -> SimpleNamespace:
    """The decoding configuration NeMo's builder derives for ``mode``, on plain objects.

    What ``pipeline_config`` writes goes through the rule of ``base_builder.py:86-105``:
    the confidence block is copied into ``confidence_cfg`` only when the greedy (or beam)
    ``preserve_frame_confidence`` is true.
    """
    config = pipeline_config(_spec(mode))
    preserve = config["asr"]["decoding"]["greedy"]["preserve_frame_confidence"]
    aggregation, method = _LIBRARY_DEFAULT
    if preserve:
        aggregation = config["confidence"]["aggregation"]
        method = dict(config["confidence"]["method_cfg"])
    return SimpleNamespace(
        greedy=SimpleNamespace(preserve_frame_confidence=preserve),
        beam=SimpleNamespace(preserve_frame_confidence=False),
        confidence_cfg=SimpleNamespace(
            aggregation=aggregation, method_cfg=SimpleNamespace(**method)
        ),
    )


class _NeMoShaped(FakeCacheAwareRNNTPipeline):
    """``nemo_fake``'s RNNT pipeline with NeMo's attributes at the paths ``observed`` reads,
    and scripted confidences on its final segments.

    ``pipeline.asr_model.asr_model.encoder.att_context_size``, ``pipeline.asr_model
    .decoding_cfg`` and ``pipeline.decoding_computer.{preserve_step_confidence,
    cuda_graphs_mode}`` are where NeMo's built pipeline keeps them; the NeMo-backed test
    below reads the same attributes off NeMo's own classes.
    """

    def __init__(
        self,
        *,
        mode: str = "off",
        att_context: tuple[int, int] = BUILT_CONTEXT,
        step_confidence: bool | None = None,
        graphs_mode: str | None = "no_while_loops",
        confidences: Sequence[float] = CONFIDENCES,
    ) -> None:
        super().__init__(160, num_slots=64)
        self.asr_model = SimpleNamespace(
            asr_model=SimpleNamespace(encoder=SimpleNamespace(att_context_size=list(att_context))),
            decoding_cfg=_decoding_cfg(mode),
        )
        self.decoding_computer = SimpleNamespace(
            preserve_step_confidence=(mode != "off")
            if step_confidence is None
            else step_confidence,
            cuda_graphs_mode=graphs_mode,
        )
        self.confidences = tuple(confidences)

    def transcribe_step(self, requests: list[Any]) -> list[Any]:
        outputs = super().transcribe_step(requests)
        for output in outputs:
            for index, segment in enumerate(output.final_segments):
                segment.conf = self.confidences[index]
        return outputs


class _CTCShaped(FakeCacheAwareCTCPipeline):
    """``nemo_fake``'s CTC pipeline shaped like NeMo's ``CacheAwareCTCPipeline``: the
    wrapper, its encoder and its decoding configuration, and no ``decoding_computer``,
    which NeMo's CTC pipeline never names (checked against NeMo's source below)."""

    def __init__(self) -> None:
        super().__init__(160, num_slots=64)
        self.asr_model = SimpleNamespace(
            asr_model=SimpleNamespace(
                encoder=SimpleNamespace(att_context_size=list(BUILT_CONTEXT))
            ),
            decoding_cfg=_decoding_cfg("off"),
        )


def _adapter(settings: ServeSettings, pipeline: Any) -> PipelineAdapter:
    return registry.build_for(
        engine_config(settings),
        boundary=boundary_for(pipeline),
        language_code="en-US",
        use_cuda_graphs=False,
    )


RUNTIME = {"nemo": "nemo-under-test", "torch": "torch-under-test", "device": "cpu-fake"}


# --- word confidence off: the golden bytes ---------------------------------------------


@pytest.mark.parametrize("case", GOLDEN["direct_finals"], ids=lambda c: c["case"])
def test_a_final_without_confidence_is_the_golden_bytes(case: dict[str, Any]) -> None:
    """Default, explicit False, and the rule's answer for "off" and for an unread
    configuration: all four are the bytes the code wrote before "c" existed, including
    for words whose confidences are not 1.0. A final with no word to carry a "c" (words
    not asked for, or none recognised) is those bytes with confidence on as well: "on"
    must not invent a "words" key the client did not ask for."""
    words = _words(case["words"])
    confidences = [False, confidence_on_wire("off"), confidence_on_wire(None)]
    if not words:
        confidences.append(True)
    for confidence in confidences:
        frame = FinalFrame(case["text"], case["audio_s"], words, confidence=confidence)
        assert frame.to_json() == case["json"]
    assert FinalFrame(case["text"], case["audio_s"], words).to_json() == case["json"]


def test_the_golden_finals_include_one_without_words_and_one_with_none_recognised() -> None:
    """The ``if not words`` above is only a check if the fixture holds both cases."""
    kinds = {c["case"]: _words(c["words"]) for c in GOLDEN["direct_finals"]}
    assert kinds["words_none"] is None
    assert kinds["words_empty"] == ()


def test_the_other_frames_are_the_golden_bytes() -> None:
    by_case = {c["case"]: c for c in GOLDEN["other_frames"]}
    session = by_case["session"]
    assert (
        SessionFrame(session["id"], session["chunk_ms"], session["invariance_class"]).to_json()
        == session["json"]
    )
    partial = by_case["partial"]
    assert PartialFrame(partial["text"], partial["audio_s"]).to_json() == partial["json"]
    error = by_case["error"]
    assert ErrorFrame(ErrorCode(error["code"]), error["message"]).to_json() == error["json"]


async def test_serving_the_fake_sends_the_golden_bytes_and_readyz_only_grew() -> None:
    """The real ``run_server`` path over the fake pipeline: every frame after the session
    frame is the captured bytes, and ``/readyz`` is the captured body plus three keys at
    the end. The fake runs no decoder, so it is configured "off" and observes nothing; it
    still names the code it imported, which is this checkout's."""
    settings = _fake_settings()
    adapter = registry.build_for(engine_config(settings))
    async with _serving(settings, adapter, execution="fake", runtime={}) as endpoints:
        body = await _ready_body(endpoints)
        for case in GOLDEN["serve_fake"]["sessions"]:
            pieces = _pcm(case["chunks"], case["extra_bytes"], case["seed"])
            frames = await _session_frames(endpoints.ws_endpoint, case["query"], pieces)
            assert frames == case["frames"], case["case"]
    golden = GOLDEN["serve_fake"]["readyz_body"]
    golden_tick = json.loads(golden)["tick_id"]
    assert _without_new_keys(body, tick_id=golden_tick) == golden
    doc = json.loads(body)
    assert list(doc)[-len(NEW_READYZ_KEYS) :] == NEW_READYZ_KEYS
    assert doc["word_confidence"] == "off"
    assert doc["observed"] == NULL_OBSERVED
    assert doc["code"] == OWN_CODE


@pytest.mark.parametrize("case", GOLDEN["stub_scripted"], ids=lambda c: c["case"])
async def test_the_scripted_stub_over_the_transport_sends_the_golden_bytes(
    case: dict[str, Any],
) -> None:
    engine = stub_engine(script=case["script"])
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        frames = await _session_frames(
            server.endpoint, case["query"], [b"\x00" * (160 * 32)] * case["chunks"]
        )
    assert frames == case["frames"]


@pytest.mark.parametrize("case", GOLDEN["readyz_reporter"], ids=lambda c: c["case"])
def test_readyz_keeps_every_key_and_value_and_adds_three_at_the_end(
    case: dict[str, Any],
) -> None:
    answer = HealthReporter(_FixedSnapshot(), ServiceFacts(**case["facts"])).route("/readyz")
    assert answer is not None
    assert (answer.status, answer.content_type) == (case["status"], case["content_type"])
    assert _without_new_keys(answer.body) == case["body"]
    doc = json.loads(answer.body)
    assert list(doc) == [*json.loads(case["body"]), *NEW_READYZ_KEYS]
    # Nothing told this reporter a configuration or gave it anything to observe. What
    # code answered is not something to be told: it is read, so it is there regardless.
    assert doc["word_confidence"] is None
    assert doc["observed"] == NULL_OBSERVED
    assert doc["code"] == OWN_CODE


# --- word confidence on: "c" carries the value the server computed ---------------------


def test_the_rule_puts_c_on_the_wire_only_for_a_configuration_other_than_off() -> None:
    assert confidence_on_wire("off") is False
    assert confidence_on_wire(None) is False
    for mode in WORD_CONFIDENCE_MODES:
        assert confidence_on_wire(mode) is (mode != "off")


def test_on_each_word_carries_its_confidence_as_c_and_nothing_else_changes() -> None:
    case = next(c for c in GOLDEN["direct_finals"] if c["case"] == "words_mixed_confidence")
    words = _words(case["words"])
    assert words is not None
    raw = FinalFrame(case["text"], case["audio_s"], words, confidence=True).to_json()
    doc = json.loads(raw)
    assert [w["c"] for w in doc["words"]] == [w.confidence for w in words]
    # C3 says "c" is a float. json.loads reads "0" as the int 0, which == 0.0, so the
    # type is checked on its own: this case holds 0.0 and 1.0.
    assert {0.0, 1.0} <= {w.confidence for w in words}
    assert all(type(w["c"]) is float for w in doc["words"])
    assert all(list(w) == ["w", "s", "e", "c"] for w in doc["words"])
    for word in doc["words"]:
        del word["c"]
    assert json.dumps(doc, separators=(",", ":")) == case["json"]


def test_c_is_written_as_a_json_float_even_when_the_confidence_is_integral() -> None:
    """The exact bytes, for 0.0, 1.0, an int 1 and 0.5: "c" is always a JSON number with
    a fraction, never "0" or "1"."""
    words = (
        Word("a", 0, 80, 0.0),
        Word("b", 80, 160, 1.0),
        Word("c", 160, 240, 1),
        Word("d", 240, 320, 0.5),
    )
    assert FinalFrame("a b c d", 0.32, words, confidence=True).to_json() == (
        '{"type":"final","text":"a b c d","words":['
        '{"w":"a","s":0,"e":80,"c":0.0},'
        '{"w":"b","s":80,"e":160,"c":1.0},'
        '{"w":"c","s":160,"e":240,"c":1.0},'
        '{"w":"d","s":240,"e":320,"c":0.5}'
        '],"audio_s":0.32}'
    )


def test_c_is_the_confidence_unrounded_and_unclamped() -> None:
    """The "c" on the wire is ``Word.confidence`` itself. A rounding to 16 or fewer
    decimal places changes 0.1 + 0.2, and one to 15 or fewer changes 1/3; -0.25, 1.5 and
    the float just above 1.0 lie outside [0, 1], and a clamp would hide that the server
    computed them. Each must come back from the JSON exactly, and the bytes are pinned."""
    values = (0.1 + 0.2, 1 / 3, -0.25, 1.5, 1.0000000000000002)
    words = tuple(Word(f"w{i}", 80 * i, 80 * (i + 1), v) for i, v in enumerate(values))
    raw = FinalFrame("w0 w1 w2 w3 w4", 0.4, words, confidence=True).to_json()
    doc = json.loads(raw)
    assert [w["c"] for w in doc["words"]] == list(values)
    assert all(type(w["c"]) is float for w in doc["words"])
    assert raw == (
        '{"type":"final","text":"w0 w1 w2 w3 w4","words":['
        '{"w":"w0","s":0,"e":80,"c":0.30000000000000004},'
        '{"w":"w1","s":80,"e":160,"c":0.3333333333333333},'
        '{"w":"w2","s":160,"e":240,"c":-0.25},'
        '{"w":"w3","s":240,"e":320,"c":1.5},'
        '{"w":"w4","s":320,"e":400,"c":1.0000000000000002}'
        '],"audio_s":0.4}'
    )


def test_a_non_finite_confidence_is_sent_as_null_and_the_frame_stays_json() -> None:
    words = (
        Word("a", 0, 80, float("nan")),
        Word("b", 80, 160, float("inf")),
        Word("c", 160, 240, 0.5),
    )
    raw = FinalFrame("a b c", 0.24, words, confidence=True).to_json()
    assert "NaN" not in raw and "Infinity" not in raw
    doc = json.loads(raw, parse_constant=lambda name: pytest.fail(f"{name} on the wire"))
    assert [w["c"] for w in doc["words"]] == [None, None, 0.5]


#: Audio positions a final can carry that a rounding other than the 6 places "off" writes
#: would change: 0.83125 is the golden fake session's final, and the others need 4, 5 or
#: 6 places, or more than 6, where the on side must round exactly as the off side does.
#: An ``audio_s`` that rounds to the same value at 3 places as at 6, as every final pinned
#: to bytes above does, cannot show a rounding to 3 on the on side.
_AUDIO_S_NEEDING_PLACES = (0.83125, 0.0005, 1.23456, 1.234567, 0.1234567, 1 / 3, 2.5e-06)


@pytest.mark.parametrize("audio_s", _AUDIO_S_NEEDING_PLACES, ids=repr)
def test_confidence_on_changes_nothing_but_c(audio_s: float) -> None:
    """C3: with confidence on a final only gains each word's "c". With every "c" removed
    it is, byte for byte, the final the same words and ``audio_s`` write with it off,
    whatever places ``audio_s`` needs; and a final with no words is unchanged by it."""
    case = next(c for c in GOLDEN["direct_finals"] if c["case"] == "words_mixed_confidence")
    words = _words(case["words"])
    assert words is not None
    off = FinalFrame(case["text"], audio_s, words).to_json()
    on = json.loads(FinalFrame(case["text"], audio_s, words, confidence=True).to_json())
    assert on["audio_s"] == json.loads(off)["audio_s"] == round(audio_s, 6)
    assert [w.pop("c") for w in on["words"]] == [w.confidence for w in words]
    assert json.dumps(on, separators=(",", ":")) == off
    for none in (None, ()):
        assert (
            FinalFrame(case["text"], audio_s, none, confidence=True).to_json()
            == FinalFrame(case["text"], audio_s, none).to_json()
        )


class _Written:
    """The two calls ``WsServer._write_results`` makes on a connection, recorded."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: int | None = None

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000) -> None:
        self.closed = code


class _Delivered:
    """A session whose engine delivers a fixed list of hypotheses, then ends."""

    def __init__(self, hypotheses: Sequence[Hypothesis]) -> None:
        self._hypotheses = tuple(hypotheses)

    async def results(self) -> AsyncIterator[Hypothesis]:
        for hypothesis in self._hypotheses:
            yield hypothesis


async def test_every_final_of_a_session_carries_c_and_nothing_else_moves() -> None:
    """A session with an endpoint mid-stream is delivered several finals, with partials
    between them. Each final carries its own words' "c", not only the first final; with
    them removed every frame, partials included, is the one confidence off writes."""
    hypotheses = [
        Hypothesis("a", True, 0.83125, (Word("a", 0, 160, 0.125),)),
        Hypothesis("b", False, 1.1234567),
        Hypothesis("b c", True, 1.234567, (Word("b", 160, 320, 0.5), Word("c", 320, 480, 0.3))),
        Hypothesis("d", False, 1.5),
        Hypothesis("d", True, 1.6, (Word("d", 480, 640, 0.999),)),
    ]
    options = SessionOptions(chunk_ms=160, word_timestamps=True)

    async def written(confidence: bool) -> list[str]:
        ws = _Written()
        await WsServer._write_results(
            ws,  # type: ignore[arg-type]
            _Delivered(hypotheses),  # type: ignore[arg-type]
            options,
            confidence=confidence,
        )
        assert ws.closed == 1000
        return ws.sent

    off, on = await written(False), await written(True)
    assert [json.loads(raw)["type"] for raw in on] == [
        "final",
        "partial",
        "final",
        "partial",
        "final",
    ]
    finals = [json.loads(raw) for raw in on if json.loads(raw)["type"] == "final"]
    assert [[w["c"] for w in final["words"]] for final in finals] == [[0.125], [0.5, 0.3], [0.999]]
    stripped = []
    for raw in on:
        doc = json.loads(raw)
        for word in doc.get("words", ()):
            del word["c"]
        stripped.append(json.dumps(doc, separators=(",", ":")))
    assert stripped == off
    assert all("c" not in w for raw in off for w in json.loads(raw).get("words", ()))


@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
async def test_the_c_a_final_carries_is_the_segment_confidence_the_pipeline_computed(
    mode: str,
) -> None:
    """From NeMo's seam to the frame: the fake puts known confidences on its final
    segments, the adapter copies them onto each word, the engine delivers the final, and
    a frame built the way the server builds one carries them exactly, or not at all
    when the configuration read off the pipeline is "off"."""
    settings = _nemo_settings()
    adapter = _adapter(settings, _NeMoShaped(mode=mode))
    configured = configured_word_confidence(adapter)
    assert configured == mode
    engine = Engine(engine_config(settings), adapter)
    async with engine:
        session = engine.open_session(SessionOptions(chunk_ms=160, word_timestamps=True))
        for piece in _pcm(4, 0, seed=3):
            assert session.feed(piece) == len(piece)
        session.end()
        finals = [h async for h in session.results() if h.is_final]
    assert len(finals) == 1 and len(finals[0].words) == 4
    words = tuple(finals[0].words)
    assert [w.confidence for w in words] == list(CONFIDENCES[:4])
    frame = FinalFrame(
        finals[0].text,
        finals[0].audio_processed_s,
        words,
        confidence=confidence_on_wire(configured),
    )
    doc = json.loads(frame.to_json())
    if mode == "off":
        assert all("c" not in w for w in doc["words"])
        assert (
            frame.to_json()
            == FinalFrame(finals[0].text, finals[0].audio_processed_s, words).to_json()
        )
    else:
        assert [w["c"] for w in doc["words"]] == list(CONFIDENCES[:4])


@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
async def test_a_served_final_carries_c_exactly_when_the_configuration_is_not_off(
    mode: str,
) -> None:
    """Through ``run_server`` and the socket: the configuration read off the pipeline is
    what ``/readyz`` says, and the final carries each word's computed confidence as a
    float "c" under it, or no "c" at all under "off". A words=0 session on the same
    server gets its final without a "words" key under every mode."""
    settings = _nemo_settings()
    adapter = _adapter(settings, _NeMoShaped(mode=mode))
    async with _serving(settings, adapter, execution="eager", runtime=RUNTIME) as endpoints:
        ready = json.loads(await _ready_body(endpoints))
        frames = await _session_frames(endpoints.ws_endpoint, "words=1", _pcm(4, 0, seed=3))
        unasked = await _session_frames(endpoints.ws_endpoint, "words=0", _pcm(4, 0, seed=3))
    assert ready["word_confidence"] == mode
    final = json.loads(frames[-1])
    assert final["type"] == "final" and len(final["words"]) == 4
    if mode == "off":
        assert all(list(w) == ["w", "s", "e"] for w in final["words"])
    else:
        assert [w.get("c") for w in final["words"]] == list(CONFIDENCES[:4])
        assert all(type(w["c"]) is float for w in final["words"])
    del final["words"]
    assert unasked[-1] == json.dumps(final, separators=(",", ":"))


@pytest.mark.parametrize("word_confidence", [*WORD_CONFIDENCE_MODES, None])
async def test_the_listener_config_alone_decides_c_on_the_scripted_stub(
    word_confidence: str | None,
) -> None:
    """The transport without ``run_server``: ``WsServerConfig.word_confidence`` is what
    puts "c" on a final. The stub's words carry ``Word``'s own 1.0; with a mode other than
    "off" every word gains exactly ``"c":1.0`` and nothing else moves, and with "off" or
    null the final is the golden bytes. A default config is "off"."""
    case = next(c for c in GOLDEN["stub_scripted"] if c["case"] == "stub_scripted_words_1")
    assert WsServerConfig().word_confidence == "off"
    engine = stub_engine(script=case["script"])
    config = WsServerConfig(port=0, word_confidence=word_confidence)
    async with engine, WsServer(engine, config) as server:
        frames = await _session_frames(
            server.endpoint, case["query"], [b"\x00" * (160 * 32)] * case["chunks"]
        )
    assert frames[:-1] == case["frames"][:-1]
    if not confidence_on_wire(word_confidence):
        assert frames[-1] == case["frames"][-1]
        return
    final = json.loads(frames[-1])
    assert all(list(w) == ["w", "s", "e", "c"] for w in final["words"])
    assert '"c":1.0}' in frames[-1]
    assert [w.pop("c") for w in final["words"]] == [1.0] * len(final["words"])
    assert json.dumps(final, separators=(",", ":")) == case["frames"][-1]


@pytest.mark.parametrize("word_confidence", [*WORD_CONFIDENCE_MODES, None])
@pytest.mark.parametrize("case", GOLDEN["stub_scripted"], ids=lambda c: c["case"])
async def test_a_words_0_client_gets_the_off_servers_bytes_under_every_setting(
    case: dict[str, Any], word_confidence: str | None
) -> None:
    """Word confidence is the server's, but whether words are sent is the client's. A
    client that asks ``words=0`` gets, under every setting, the bytes an "off" server
    sends it: the golden words=1 frames with the final's "words" key dropped, and no
    "words" list, empty or carrying "c", that it did not ask for."""
    query = case["query"].replace("words=1", "words=0")
    assert query != case["query"]
    pieces = [b"\x00" * (160 * 32)] * case["chunks"]

    async def frames(config: WsServerConfig) -> list[str]:
        engine = stub_engine(script=case["script"])
        async with engine, WsServer(engine, config) as server:
            return await _session_frames(server.endpoint, query, pieces)

    golden_final = json.loads(case["frames"][-1])
    del golden_final["words"]
    off = await frames(WsServerConfig(port=0, word_confidence="off"))
    assert off == [*case["frames"][:-1], json.dumps(golden_final, separators=(",", ":"))]
    assert await frames(WsServerConfig(port=0, word_confidence=word_confidence)) == off


#: Values a client might send to try to set word confidence itself: each mode, four of
#: the bool spellings the query parser accepts for other switches, and an empty value.
_CLIENT_CONFIDENCE_VALUES = ("off", "nemo-shipped", "paper-best", "0", "1", "true", "false", "")


@pytest.mark.parametrize("word_confidence", [*WORD_CONFIDENCE_MODES, None])
async def test_the_client_cannot_switch_c_on_or_off(word_confidence: str | None) -> None:
    """Whether a final carries "c" is the server's configuration and never the client's:
    a words=1 client that also asks for word confidence on or off, under four names a
    client might try, gets exactly the frames the same server sends a client that asked
    for nothing, which carry "c" exactly when the server's own setting is not off.
    """
    case = next(c for c in GOLDEN["stub_scripted"] if c["case"] == "stub_scripted_words_1")
    pieces = [b"\x00" * (160 * 32)] * case["chunks"]
    engine = stub_engine(script=case["script"])
    config = WsServerConfig(port=0, word_confidence=word_confidence)
    async with engine, WsServer(engine, config) as server:
        plain = await _session_frames(server.endpoint, case["query"], pieces)
        asked: dict[str, list[str]] = {}
        for value in _CLIENT_CONFIDENCE_VALUES:
            query = "&".join(
                [case["query"]]
                + [f"{name}={value}" for name in ("word_confidence", "word-confidence")]
                + [f"{name}={value}" for name in ("confidence", "c")]
            )
            asked[value] = await _session_frames(server.endpoint, query, pieces)
    final = json.loads(plain[-1])
    assert final["words"]
    assert all(("c" in w) is confidence_on_wire(word_confidence) for w in final["words"])
    if not confidence_on_wire(word_confidence):
        assert plain == case["frames"]
    assert {value: frames == plain for value, frames in asked.items()} == dict.fromkeys(
        _CLIENT_CONFIDENCE_VALUES, True
    )


async def test_a_configuration_that_cannot_be_named_is_served_as_null_not_as_off() -> None:
    """A decoder that computes step confidence under a decoding configuration none of
    the three modes wrote (alpha 0.7) is not "off" and not any mode: ``/readyz`` says
    null, the server starts (there is nothing to contradict), and no "c" is sent, since
    no mode names what the value would mean. Reading the null as "off" would refuse to
    start, because "off" over a decoder that computes confidence is a contradiction."""
    settings = _nemo_settings()
    pipeline = _NeMoShaped(mode="nemo-shipped")
    pipeline.asr_model.decoding_cfg.confidence_cfg.method_cfg.alpha = 0.7
    adapter = _adapter(settings, pipeline)
    assert configured_word_confidence(adapter) is None
    async with _serving(settings, adapter, execution="eager", runtime=RUNTIME) as endpoints:
        doc = json.loads(await _ready_body(endpoints))
        frames = await _session_frames(endpoints.ws_endpoint, "words=1", _pcm(4, 0, seed=3))
    assert doc["word_confidence"] is None
    assert doc["observed"]["decoder_step_confidence"] is True
    final = json.loads(frames[-1])
    assert final["type"] == "final" and len(final["words"]) == 4
    assert all("c" not in w for w in final["words"])


# --- /readyz: the configuration and what the built pipeline is -------------------------


async def test_readyz_reports_what_the_built_pipeline_is_not_what_the_settings_imply() -> None:
    """The encoder holds [56, 1]; the settings' checkpoint family and chunk give [70, 1].
    ``/readyz`` must say [56, 1]. Every earlier key keeps its meaning."""
    settings = _nemo_settings()
    adapter = _adapter(settings, _NeMoShaped(mode="nemo-shipped"))
    async with _serving(settings, adapter, execution="eager", runtime=RUNTIME) as endpoints:
        doc = json.loads(await _ready_body(endpoints))
    assert doc["observed"] == {
        "att_context_size": list(BUILT_CONTEXT),
        "decoder_step_confidence": True,
        "decoder_graphs": True,
        # The fake keeps no model decoding object; tests/test_confidence_word_aggregation_crash.py
        # reads one.
        "decoder_word_confidence": None,
    }
    assert doc["observed"]["att_context_size"] != SPEC_CONTEXT
    assert doc["word_confidence"] == "nemo-shipped"
    assert list(doc)[-len(NEW_READYZ_KEYS) :] == NEW_READYZ_KEYS
    assert doc["code"] == OWN_CODE
    assert {k: doc[k] for k in ("model", "pipeline", "chunk_ms", "precision", "execution")} == {
        "model": NEMO_MODEL,
        "pipeline": "cache_aware_rnnt",
        "chunk_ms": 160,
        "precision": "bfloat16",
        "execution": "eager",
    }
    assert doc["biasing"] is False and doc["ready"] is True
    assert (doc["nemo_version"], doc["torch_version"], doc["device_name"]) == (
        RUNTIME["nemo"],
        RUNTIME["torch"],
        RUNTIME["device"],
    )


async def test_readyz_reads_the_built_objects_on_every_request() -> None:
    """A decoder whose graphs are switched off, or an encoder whose context is changed,
    after the server started is reported as it is now, not as it was at startup."""
    settings = _nemo_settings()
    pipeline = _NeMoShaped(mode="off")
    adapter = _adapter(settings, pipeline)
    async with _serving(settings, adapter, execution="eager", runtime=RUNTIME) as endpoints:
        before = json.loads(await _ready_body(endpoints))["observed"]
        pipeline.decoding_computer.cuda_graphs_mode = None
        pipeline.asr_model.asr_model.encoder.att_context_size = [70, 13]
        after = json.loads(await _ready_body(endpoints))["observed"]
    assert before == {
        "att_context_size": list(BUILT_CONTEXT),
        "decoder_step_confidence": False,
        "decoder_graphs": True,
        "decoder_word_confidence": None,
    }
    assert after == {
        "att_context_size": [70, 13],
        "decoder_step_confidence": False,
        "decoder_graphs": False,
        "decoder_word_confidence": None,
    }


async def test_a_ctc_server_observes_its_encoder_and_no_decoder() -> None:
    """NeMo's CTC pipeline has an encoder and no decoding computer. ``/readyz`` reports
    the encoder's context and null for both decoder facts. False would say "read, and it
    computes no step confidence and runs no graphs", which a runbook that wants False
    would then accept, although a CTC decoder computes confidence on every step."""
    settings = _nemo_settings("cache_aware_ctc")
    adapter = _adapter(settings, _CTCShaped())
    async with _serving(settings, adapter, execution="eager", runtime=RUNTIME) as endpoints:
        doc = json.loads(await _ready_body(endpoints))
    assert doc["pipeline"] == "cache_aware_ctc"
    assert doc["observed"] == {
        "att_context_size": list(BUILT_CONTEXT),
        "decoder_step_confidence": None,
        "decoder_graphs": None,
        "decoder_word_confidence": None,
    }


@pytest.mark.parametrize(
    ("mode", "step_confidence"), [("nemo-shipped", False), ("paper-best", False), ("off", True)]
)
async def test_a_configuration_the_built_decoder_contradicts_is_refused_before_binding(
    mode: str, step_confidence: bool
) -> None:
    settings = _nemo_settings()
    adapter = _adapter(settings, _NeMoShaped(mode=mode, step_confidence=step_confidence))
    bound: list[Endpoints] = []
    with pytest.raises(ConfigError, match="the label and the decoder disagree"):
        await asyncio.wait_for(
            run_server(
                settings,
                adapter,
                shutdown=asyncio.Event(),
                on_ready=bound.append,
                execution="eager",
                runtime=RUNTIME,
            ),
            timeout=15,
        )
    assert bound == []


def test_disagreement_needs_both_sides_read() -> None:
    assert disagreement("off", ObservedFacts(decoder_step_confidence=False)) is None
    assert disagreement("paper-best", ObservedFacts(decoder_step_confidence=True)) is None
    assert disagreement(None, ObservedFacts(decoder_step_confidence=True)) is None
    assert disagreement("nemo-shipped", ObservedFacts()) is None
    assert disagreement("nemo-shipped", ObservedFacts(decoder_step_confidence=False))
    assert disagreement("off", ObservedFacts(decoder_step_confidence=True))


@pytest.mark.parametrize("mode", WORD_CONFIDENCE_MODES)
def test_the_configured_mode_is_read_back_from_the_decoding_configuration(mode: str) -> None:
    pipeline = SimpleNamespace(asr_model=SimpleNamespace(decoding_cfg=_decoding_cfg(mode)))
    assert (
        configured_word_confidence(SimpleNamespace(_boundary=SimpleNamespace(pipeline=pipeline)))
        == mode
    )


def _reading_adapter(decoding: Any) -> Any:
    """An adapter whose boundary's pipeline carries ``decoding`` where NeMo's wrapper
    keeps the decoding configuration it applied."""
    pipeline = SimpleNamespace(asr_model=SimpleNamespace(decoding_cfg=decoding))
    return SimpleNamespace(_boundary=SimpleNamespace(pipeline=pipeline))


#: One value per field ``configured_word_confidence`` compares, none of which either on
#: mode writes (nemo-shipped: mean, entropy, tsallis, 0.5, exp; paper-best: min, entropy,
#: tsallis, 0.33, exp). The two on modes differ only in aggregation and alpha, so a
#: comparison that dropped name, entropy_type or entropy_norm would still tell them apart
#: and only a change to that one field shows it was dropped.
_UNWRITTEN_FIELD_VALUES = (
    ("aggregation", "max"),
    ("name", "max_prob"),
    ("entropy_type", "gibbs"),
    ("alpha", 0.7),
    ("entropy_norm", "lin"),
)


@pytest.mark.parametrize("mode", [m for m in WORD_CONFIDENCE_MODES if m != "off"])
@pytest.mark.parametrize(("field", "value"), _UNWRITTEN_FIELD_VALUES, ids=lambda v: str(v))
def test_each_compared_field_alone_makes_a_configuration_unnamed(
    mode: str, field: str, value: Any
) -> None:
    """A decoding configuration that matches ``mode`` in every compared field but one is
    not ``mode`` and not any other mode: it is null. Named instead, ``/readyz`` would
    report a mode, and the wire would carry "c", for a method no mode wrote."""
    decoding = _decoding_cfg(mode)
    assert configured_word_confidence(_reading_adapter(decoding)) == mode
    confidence = decoding.confidence_cfg
    target = confidence if field == "aggregation" else confidence.method_cfg
    assert getattr(target, field) != value
    setattr(target, field, value)
    assert configured_word_confidence(_reading_adapter(decoding)) is None


def test_a_configuration_that_cannot_be_named_is_null_not_a_guess() -> None:
    unread = _decoding_cfg("off")
    unread.greedy.preserve_frame_confidence = "false"
    assert configured_word_confidence(_reading_adapter(unread)) is None
    # The beam block's flag counts as NeMo's builder counts it (base_builder.py:86-88).
    beam = _decoding_cfg("paper-best")
    beam.greedy.preserve_frame_confidence, beam.beam.preserve_frame_confidence = False, True
    assert configured_word_confidence(_reading_adapter(beam)) == "paper-best"
    # A NeMo pipeline without a decoding configuration, and an adapter with no pipeline
    # that is not the scripted fake: nothing was read, so neither is "off".
    no_config = SimpleNamespace(_boundary=SimpleNamespace(pipeline=SimpleNamespace()))
    assert configured_word_confidence(no_config) is None
    assert configured_word_confidence(SimpleNamespace()) is None
    assert configured_word_confidence(SimpleNamespace(_boundary=None)) is None


def test_only_the_scripted_fake_is_off_without_a_pipeline_to_read() -> None:
    """The fake runs no decoder, which is a fact about its class, so it is "off". The same
    answer for any adapter without a NeMo pipeline would be a default, not a reading."""
    adapter = registry.build_for(engine_config(_fake_settings()))
    assert isinstance(adapter, FakePipelineAdapter)
    assert configured_word_confidence(adapter) == "off"
    assert configured_word_confidence(object()) is None


def test_a_configuration_that_preserves_no_confidence_is_named_only_when_one_mode_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the flag false NeMo copies no confidence block, so the flag is all there is
    to match. Today exactly one mode writes it false; the name comes from that row, and
    if two did, the reading could not tell them apart and must be null."""
    assert [row[0] for row in observed_module._written() if not row[1]] == ["off"]
    pipeline = SimpleNamespace(asr_model=SimpleNamespace(decoding_cfg=_decoding_cfg("off")))
    adapter = SimpleNamespace(_boundary=SimpleNamespace(pipeline=pipeline))
    quiet = ("quiet", False, ("min", "entropy", "tsallis", 0.33, "exp"))
    loud = ("loud", True, ("mean", "max_prob", None, 1.0, None))
    monkeypatch.setattr(observed_module, "_written", lambda: (quiet, loud))
    assert configured_word_confidence(adapter) == "quiet"
    monkeypatch.setattr(observed_module, "_written", lambda: (("off", *quiet[1:]), quiet, loud))
    assert configured_word_confidence(adapter) is None


def test_observed_facts_carry_exactly_the_readyz_keys() -> None:
    assert tuple(ObservedFacts().to_json_dict()) == OBSERVED_KEYS
    reporter = HealthReporter(
        _FixedSnapshot(),
        ServiceFacts(model="m", chunk_ms=160, precision="none", execution="fake", pipeline="fake"),
        observe=lambda: {"att_context_size": [1, 2], "extra": "dropped"},
    )
    assert reporter.observed() == {**NULL_OBSERVED, "att_context_size": [1, 2]}


def test_readyz_writes_the_observed_keys_in_wire_order_whatever_the_reading_gives() -> None:
    """``OBSERVED_KEYS`` says wire order; a reading that lists them backwards must still be
    written in that order, so a consumer comparing bodies byte for byte sees one body."""
    reporter = HealthReporter(
        _FixedSnapshot(),
        ServiceFacts(model="m", chunk_ms=160, precision="none", execution="fake", pipeline="fake"),
        observe=lambda: {
            "decoder_graphs": False,
            "decoder_step_confidence": True,
            "att_context_size": [1, 2],
        },
    )
    answer = reporter.route("/readyz")
    assert answer is not None
    observed = json.loads(answer.body)["observed"]
    assert list(observed) == [
        "att_context_size",
        "decoder_step_confidence",
        "decoder_graphs",
        "decoder_word_confidence",
    ]
    assert '"observed":{"att_context_size":[1,2],' in answer.body


# --- /readyz: the code that answered ---------------------------------------------------


def _code_reporter() -> HealthReporter:
    return HealthReporter(
        _FixedSnapshot(),
        ServiceFacts(model="m", chunk_ms=160, precision="none", execution="fake", pipeline="fake"),
    )


def _readyz_code(reporter: HealthReporter) -> dict[str, Any]:
    """``/readyz``'s ``code`` object off ``reporter``'s wire body, asked now."""
    answer = reporter.route("/readyz")
    assert answer is not None
    doc = json.loads(answer.body)
    assert list(doc)[-1] == "code"
    code: dict[str, Any] = doc["code"]
    return code


async def test_readyz_code_names_this_checkout_s_own_packages() -> None:
    """C7: a server started from this checkout reports the ``src/verbatim`` and the
    ``bench/src/verbatim_bench`` of this checkout, resolved, under those two key names in
    that order. The expected paths come from this file's location, not from the modules:
    a suite run against another checkout's installed packages fails here."""
    assert list(CODE_KEYS) == ["verbatim_path", "bench_path"]
    settings = _fake_settings()
    adapter = registry.build_for(engine_config(settings))
    async with _serving(settings, adapter, execution="fake", runtime={}) as endpoints:
        doc = json.loads(await _ready_body(endpoints))
    assert list(doc["code"]) == ["verbatim_path", "bench_path"]
    assert doc["code"] == OWN_CODE
    assert doc["code"]["verbatim_path"] == os.path.realpath(CHECKOUT / "src" / "verbatim")
    assert os.path.isabs(doc["code"]["verbatim_path"])
    assert code_paths() == OWN_CODE


def test_readyz_code_is_read_off_the_imported_modules_at_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The paths are ``os.path.dirname(module.__file__)`` of the modules the import system
    holds when ``/readyz`` is asked, with symlinks resolved: a package reached through a
    linked directory is reported at its real directory, and one whose ``__init__.py`` is
    a link is reported at the directory it was imported from, not at the link's target.
    One reporter answers throughout, so a reading taken once when it was built, or from
    this checkout's layout, cannot pass."""
    reporter = _code_reporter()
    assert _readyz_code(reporter) == OWN_CODE
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        (root / "real_pkg").mkdir()
        (root / "real_pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "linked_pkg").symlink_to(root / "real_pkg", target_is_directory=True)
        (root / "bench_pkg").mkdir()
        (root / "elsewhere").mkdir()
        (root / "elsewhere" / "__init__.py").write_text("", encoding="utf-8")
        (root / "bench_pkg" / "__init__.py").symlink_to(root / "elsewhere" / "__init__.py")
        monkeypatch.setattr(verbatim, "__file__", str(root / "linked_pkg" / "__init__.py"))
        monkeypatch.setattr(verbatim_bench, "__file__", str(root / "bench_pkg" / "__init__.py"))
        assert _readyz_code(reporter) == {
            "verbatim_path": str(root / "real_pkg"),
            "bench_path": str(root / "bench_pkg"),
        }
        monkeypatch.undo()
    assert _readyz_code(reporter) == OWN_CODE


def test_readyz_code_is_null_for_a_package_that_cannot_be_imported_or_names_no_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server does not need ``verbatim_bench``; where it cannot be imported its path
    is null, not a guess from where it would be in a checkout. A module with no file names
    no directory, so it is null too."""
    reporter = _code_reporter()
    no_bench = {"verbatim_path": OWN_CODE["verbatim_path"], "bench_path": None}
    monkeypatch.setitem(sys.modules, "verbatim_bench", None)
    assert _readyz_code(reporter) == no_bench
    monkeypatch.undo()
    assert _readyz_code(reporter) == OWN_CODE
    monkeypatch.setattr(verbatim_bench, "__file__", None)
    assert _readyz_code(reporter) == no_bench


@pytest.mark.parametrize(
    "value", [None, [70], [70, 1, 2], [70.0, 1], [True, 1], "70,1", 70], ids=repr
)
def test_an_attention_context_that_is_not_two_ints_is_not_observed(value: Any) -> None:
    pipeline = SimpleNamespace(
        asr_model=SimpleNamespace(
            asr_model=SimpleNamespace(encoder=SimpleNamespace(att_context_size=value))
        )
    )
    assert observe(pipeline).att_context_size is None


def test_an_encoder_without_a_decoding_computer_observes_no_decoder() -> None:
    """NeMo's CTC pipeline shape, read directly: the context, and None for both decoder
    facts, not False."""
    pipeline = SimpleNamespace(
        asr_model=SimpleNamespace(
            asr_model=SimpleNamespace(encoder=SimpleNamespace(att_context_size=[70, 1]))
        )
    )
    facts = observe(pipeline)
    assert facts.att_context_size == (70, 1)
    assert facts.decoder_step_confidence is None
    assert facts.decoder_graphs is None


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None], ids=repr)
def test_a_step_confidence_flag_that_is_not_a_bool_is_not_observed(flag: Any) -> None:
    """Only a bool is an observation; ``bool("false")`` would report True."""
    computer = SimpleNamespace(preserve_step_confidence=flag, cuda_graphs_mode=None)
    assert observe(SimpleNamespace(decoding_computer=computer)).decoder_step_confidence is None
    assert (
        observe(SimpleNamespace(decoding_computer=SimpleNamespace())).decoder_step_confidence
        is None
    )


_NO_ATTRIBUTE = object()


@pytest.mark.parametrize(
    ("mode", "graphs"),
    [
        (_NO_ATTRIBUTE, None),
        (None, False),
        ("full_graph", True),
        ("no_while_loops", True),
        ("no_graphs", False),
        (SimpleNamespace(value="no_while_loops"), True),
        (SimpleNamespace(value="no_graphs"), False),
        ("graphs_somehow", None),
        (1, None),
    ],
    ids=lambda value: "no-attribute" if value is _NO_ATTRIBUTE else repr(value),
)
def test_decoder_graphs_is_read_from_the_mode_the_computer_runs(mode: Any, graphs: Any) -> None:
    """A computer with no ``cuda_graphs_mode`` is not observed (None); NeMo's
    ``no_graphs`` mode runs no graph (False), like no mode; a mode NeMo does not define
    is None rather than a guess. ``SimpleNamespace(value=...)`` stands for NeMo's enum."""
    computer = SimpleNamespace(preserve_step_confidence=False)
    if mode is not _NO_ATTRIBUTE:
        computer.cuda_graphs_mode = mode
    assert observe(SimpleNamespace(decoding_computer=computer)).decoder_graphs is graphs


def test_nemo_own_objects_are_read_at_the_same_paths() -> None:
    """The readers run over NeMo's own classes, not only over this file's fakes: the RNNT
    decoding configuration its builder derives from what ``pipeline_config`` writes, the
    CTC one it builds itself, its label-looping computer, and a small ConformerEncoder.
    No model is loaded and no GPU is touched."""
    omegaconf = pytest.importorskip("omegaconf")
    builder = pytest.importorskip(
        "nemo.collections.asr.inference.factory.cache_aware_pipeline_builder"
    ).CacheAwarePipelineBuilder
    looping = pytest.importorskip(
        "nemo.collections.asr.parts.submodules.transducer_decoding.rnnt_label_looping"
    )
    conformer = pytest.importorskip("nemo.collections.asr.modules.conformer_encoder")

    def adapter(decoding: Any) -> Any:
        pipeline = SimpleNamespace(asr_model=SimpleNamespace(decoding_cfg=decoding))
        return SimpleNamespace(_boundary=SimpleNamespace(pipeline=pipeline))

    for mode in WORD_CONFIDENCE_MODES:
        decoding = builder.get_rnnt_decoding_cfg(
            omegaconf.OmegaConf.create(pipeline_config(_spec(mode)))
        )
        assert configured_word_confidence(adapter(decoding)) == mode
    assert configured_word_confidence(adapter(builder.get_ctc_decoding_cfg())) == "off"

    # Each of NeMo's graph modes is set with force_cuda_graphs_mode, which only assigns
    # the attribute: allow_cuda_graphs=True would query the CUDA driver for conditional
    # nodes, and this test touches no device. None is what allow_cuda_graphs=False gives.
    graph_modes = {None: False, "full_graph": True, "no_while_loops": True, "no_graphs": False}
    assert {m.value for m in looping.GreedyBatchedRNNTLabelLoopingComputer.CudaGraphsMode} == {
        mode for mode in graph_modes if mode is not None
    }
    for preserve in (False, True):
        computer = looping.GreedyBatchedRNNTLabelLoopingComputer(
            decoder=None,
            joint=None,
            blank_index=10,
            max_symbols_per_step=10,
            preserve_step_confidence=preserve,
            exclude_blank_from_confidence=True,
            allow_cuda_graphs=False,
        )
        assert computer.cuda_graphs_mode is None
        for mode, graphs in graph_modes.items():
            computer.force_cuda_graphs_mode(mode)
            facts = observe(SimpleNamespace(decoding_computer=computer))
            assert (facts.decoder_step_confidence, facts.decoder_graphs) == (preserve, graphs)

    # NeMo's CTC pipeline keeps no decoding computer, which is the shape _CTCShaped has.
    ctc = pytest.importorskip("nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline")
    rnnt = pytest.importorskip("nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline")
    assert "decoding_computer" not in inspect.getsource(ctc)
    assert "self.decoding_computer" in inspect.getsource(rnnt)

    encoder = conformer.ConformerEncoder(
        feat_in=16,
        n_layers=1,
        d_model=16,
        n_heads=2,
        att_context_size=[[70, 13], [70, 1]],
        att_context_style="chunked_limited",
        subsampling_factor=8,
        subsampling_conv_channels=8,
        ff_expansion_factor=2,
    )
    wrapper = SimpleNamespace(asr_model=SimpleNamespace(encoder=encoder))
    assert observe(SimpleNamespace(asr_model=wrapper)).att_context_size == (70, 13)
    encoder.set_default_att_context_size([70, 1])
    assert observe(SimpleNamespace(asr_model=wrapper)).att_context_size == (70, 1)
