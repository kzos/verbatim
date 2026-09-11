# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The health endpoints on the WebSocket listener, over plain HTTP."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import pytest
from websockets.asyncio.client import connect

from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.errors import ResourceExhausted
from verbatim.core.types import StepResult
from verbatim.engine import Engine, stub_engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.base import SessionOptions
from verbatim.protocols.health import STALE_TICKS, HealthReporter, ServiceFacts
from verbatim.protocols.ws.server import WsServer, WsServerConfig
from verbatim.scheduler.clock import ScaledMonotonicClock

pytestmark = pytest.mark.cpu

FACTS = ServiceFacts(
    model="fake-model", chunk_ms=160, precision="none", execution="fake", pipeline="fake"
)
OPTIONS = SessionOptions(chunk_ms=160)


async def _get(url: str) -> tuple[int, dict[str, str], str]:
    def fetch() -> tuple[int, dict[str, str], str]:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, dict(resp.headers), resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), err.read().decode()

    return await asyncio.to_thread(fetch)


def _base(server: WsServer) -> str:
    return f"http://127.0.0.1:{server.port}"


def _reporting(engine: Engine) -> WsServer:
    return WsServer(engine, WsServerConfig(port=0), health=HealthReporter(engine, FACTS))


async def test_readyz_follows_the_engine_from_unstarted_to_ticking_to_stopped() -> None:
    engine = stub_engine()
    async with _reporting(engine) as server:
        status, headers, body = await _get(_base(server) + "/readyz")
        assert status == 503
        assert headers["Content-Type"].startswith("application/json")
        assert json.loads(body)["ready"] is False
        assert json.loads(body)["reason"] == "engine not started"

        await engine.start()
        await engine.wait_for_ticks(1)
        status, _, body = await _get(_base(server) + "/readyz")
        ready = json.loads(body)
        assert status == 200 and ready["ready"] is True and ready["reason"] is None
        assert ready["model"] == "fake-model" and ready["pipeline"] == "fake"
        assert ready["chunk_ms"] == 160 and ready["precision"] == "none"
        assert ready["execution"] == "fake" and ready["tick_id"] >= 1

        await engine.stop()
        status, _, body = await _get(_base(server) + "/readyz")
        assert status == 503 and json.loads(body)["reason"] == "tick loop not running"


async def test_healthz_is_the_tick_loop_liveness(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = stub_engine()
    async with _reporting(engine) as server:
        await engine.start()
        await engine.wait_for_ticks(1)
        status, _, body = await _get(_base(server) + "/healthz")
        alive = json.loads(body)
        assert status == 200 and alive["alive"] is True and alive["running"] is True
        assert alive["tick_id"] >= 1 and alive["last_tick_age_s"] >= 0.0

        real_run_tick = engine._tick.run_tick

        def run_tick(*, lock: object = None) -> list[StepResult]:
            raise MemoryError("tick thread died")

        monkeypatch.setattr(engine._tick, "run_tick", run_tick)
        for _ in range(500):
            if engine.snapshot().dead:
                break
            await asyncio.sleep(0.01)
        status, _, body = await _get(_base(server) + "/healthz")
        assert status == 503 and json.loads(body)["reason"] == "tick loop dead"
        monkeypatch.setattr(engine._tick, "run_tick", real_run_tick)
        await engine.stop()

    engine2 = stub_engine()
    async with _reporting(engine2) as server:
        async with engine2:
            await engine2.wait_for_ticks(1)
        status, _, body = await _get(_base(server) + "/healthz")
        assert status == 503 and json.loads(body)["reason"] == "tick loop not running"


class _HeldClock:
    """Real time for `now`; the tick thread's sleeps block until released."""

    def __init__(self) -> None:
        self._releases = threading.Semaphore(0)
        self._open = False
        self.sleeps = 0

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        self.sleeps += 1
        if not self._open:
            self._releases.acquire()

    def release(self) -> None:
        self._releases.release()

    def open(self) -> None:
        self._open = True
        self._releases.release()


async def _until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.005)


async def test_a_stalled_tick_loop_is_not_alive() -> None:
    """A running thread whose last tick is older than STALE_TICKS periods is a stall,
    and /healthz says so: the thread is held in its sleep on a real clock."""
    clock = _HeldClock()
    config = EngineConfig(chunk=ChunkMode(160), buckets=(1,), edge_batch=1)
    engine = Engine(config, FakePipelineAdapter(ChunkMode(160), buckets=(1,)), clock=clock)
    async with _reporting(engine) as server, engine:
        try:
            await _until(lambda: clock.sleeps >= 1)  # parked before tick 1
            wakes = engine._processed_wakes
            clock.release()
            await _until(lambda: engine._processed_wakes > wakes)  # tick 1 done, parked
            status, _, body = await _get(_base(server) + "/healthz")
            assert status == 200 and json.loads(body)["alive"] is True
            await asyncio.sleep(STALE_TICKS * 0.16 + 0.2)  # the thread stays parked
            status, _, body = await _get(_base(server) + "/healthz")
            stalled = json.loads(body)
            assert status == 503 and stalled["alive"] is False
            assert stalled["reason"].startswith("last tick ") and "periods" in stalled["reason"]
            assert stalled["running"] is True
        finally:
            clock.open()


class _HeldFirstStep(FakePipelineAdapter):
    """The first step blocks until released: on a real GPU that is the warm-up, and it
    is the window in which the engine is started and running but has never stepped."""

    def __init__(self) -> None:
        super().__init__(ChunkMode(160), buckets=(1,))
        self.release = threading.Event()
        self.calls = 0

    def transcribe_step(self, frames, *, keep_all_outputs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            self.release.wait(timeout=10.0)
        return super().transcribe_step(frames, keep_all_outputs=keep_all_outputs)


async def test_readyz_waits_for_the_first_tick() -> None:
    """Started and running is not ready: until the first tick completes the server has
    never stepped, and a probe that said ready would route traffic at its warm-up."""
    pipeline = _HeldFirstStep()
    config = EngineConfig(chunk=ChunkMode(160), buckets=(1,), edge_batch=1)
    engine = Engine(config, pipeline, clock=ScaledMonotonicClock(100.0))
    session = engine.open_session(OPTIONS)
    assert session.feed(b"\x00" * OPTIONS.chunk_bytes) == OPTIONS.chunk_bytes
    async with _reporting(engine) as server, engine:
        try:
            await _until(lambda: pipeline.calls >= 1)  # inside the first step, tick 0
            status, _, body = await _get(_base(server) + "/readyz")
            not_yet = json.loads(body)
            assert status == 503 and not_yet["ready"] is False
            assert not_yet["reason"] == "no tick completed yet" and not_yet["tick_id"] == 0
            pipeline.release.set()
            await engine.wait_for_ticks(1)
            status, _, body = await _get(_base(server) + "/readyz")
            assert status == 200 and json.loads(body)["tick_id"] >= 1
        finally:
            pipeline.release.set()


async def test_admission_reports_the_controller_and_the_last_refusal() -> None:
    engine = stub_engine(bucket=2)
    async with _reporting(engine) as server, engine:
        engine.open_session(OPTIONS)
        engine.open_session(OPTIONS)
        with pytest.raises(ResourceExhausted):
            engine.open_session(OPTIONS)
        status, headers, body = await _get(_base(server) + "/admission")
        report = json.loads(body)
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert report["bucket"] == 2 and report["live"] == 2
    assert report["ceiling"] is None and report["calibrated_ceiling"] is None
    assert report["admitted_total"] == 2 and report["refused_total"] == 1
    assert "fill the largest bucket" in report["last_refusal_reason"]
    assert report["degradation_level"] == 0 and report["consecutive_overruns"] == 0
    slots = report["slots"]
    assert slots["free"] == slots["capacity"] - slots["reserved"]


async def test_metrics_is_the_exposition_with_the_facts_as_labels() -> None:
    engine = stub_engine()
    async with _reporting(engine) as server, engine:
        engine.open_session(OPTIONS)
        await engine.wait_for_ticks(2)
        status, headers, body = await _get(_base(server) + "/metrics")
    assert status == 200
    assert headers["Content-Type"] == "text/plain; version=0.0.4; charset=utf-8"
    labels = 'chunk_ms="160",precision="none",execution="fake",model="fake-model"'
    assert f"verbatim_live_sessions{{{labels}}} 1" in body
    assert f"verbatim_up{{{labels}}} 1" in body
    assert "# TYPE verbatim_ticks_total counter" in body


async def test_the_health_paths_are_404_without_a_reporter() -> None:
    engine = stub_engine()
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        for path in ("/healthz", "/readyz", "/admission", "/metrics"):
            status, _, _ = await _get(_base(server) + path)
            assert status == 404, path


async def test_the_stream_still_upgrades_and_other_paths_are_404_beside_the_reporter() -> None:
    engine = stub_engine()
    async with engine, _reporting(engine) as server:
        async with connect(server.endpoint) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assert isinstance(raw, str) and json.loads(raw)["type"] == "session"
        status, _, _ = await _get(_base(server) + "/nothing")
        assert status == 404
