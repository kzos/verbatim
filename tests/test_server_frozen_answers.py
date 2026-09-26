# SPDX-License-Identifier: Apache-2.0
"""``probes/server_frozen_answers.py`` end to end, against ``verbatim serve`` over the NeMo fake,
and every guard it carries as a pure function.

The servers here are the real ``verbatim serve`` path -- ``main`` with ``--bucket``,
``--padding fixed`` or ``ragged``, ``--compute-dtype bfloat16``, ``--word-confidence``, the real
``CacheAwareRNNTAdapter``, engine, WebSocket listener and health endpoints -- with the one thing a
CPU cannot do, the model, replaced at NeMo's own seam by ``_NeMoShaped``: the CPU fake
``FakeCacheAwareRNNTPipeline``, carrying a built NeMo pipeline's attributes where
``verbatim.pipelines.observed`` reads them. Its recognition is a pure function of the audio, and
each final word's confidence a pure function of the word, so the tests can compute what the
server must answer for each synthetic recording and check the capture against that, per
recording id.

``/readyz``'s contract-C3 fields come from the real server code reading those attributes, so the
probe reads what a NeMo server at this setting reports. ``readyz_as_a_nemo_server_reports_it``
lets a test replace them (``READYZ_C3`` changed) to make a server that says something else, or
remove some or all of them to make a server that does not say. Likewise its contract-C7 ``code``,
which the real server reports from its own imports: the servers here run in this process, so
they report the ``verbatim`` these tests imported, which is this checkout's when the tests run
with its ``src`` first on the path (a test checks it); a test replaces it to make a server that
runs other code, or says nothing.

Two things a CPU run cannot have are supplied, each as a NeMo server on card 3 would show it:

* the host commands (``nvidia-smi``, ``ss``): ``FakeHostCommands``, which records every call.
* ``/proc/<pid>/cmdline`` and ``environ`` of the server: ``FakeProc``, the argv the fixture
  really served with and an environment that places it on card 3.

Because they are test doubles, every capture here is stamped ``"fake_pipeline": true`` and the
comparator refuses it; where a test goes on to compare, it does so on a copy without the stamp
(``_unstamped``), which is the shape a capture from a real server has.

What git says of the probe's checkout is fixed for these captures (``git_says_a_clean_checkout``):
the tests may run from a copy that is no git checkout, where every capture's commit would be null
and the comparator would refuse it. The real ``_git`` is tested on a repository of its own.

The guards that matter most are the lost finals, the server's identity across the three
readings, and the card. Each has an end-to-end test that makes a real run fail or be refused,
with the reason named; every other guard has a unit test that flips one field of a green input.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import websockets
from verbatim_bench import client as bench_client
from verbatim_bench import constants as bench_constants
from verbatim_bench.canonical import FinalRecord, finals_digest
from verbatim_bench.client import SessionResult
from verbatim_bench.corpus import Utterance
from verbatim_bench.serverfacts import ServerFacts

from verbatim.cli import EXIT_OK as SERVE_OK
from verbatim.cli import Hooks
from verbatim.cli import main as serve_main
from verbatim.pipelines.nemo_fake import FakeCacheAwareRNNTPipeline, boundary_for
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, RuntimeReport, pipeline_config
from verbatim.protocols.base import Hypothesis
from verbatim.protocols.health import HealthReporter, HealthResponse
from verbatim.protocols.ws.server import WsServer
from verbatim.serve import Endpoints

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("server_frozen_answers", "probes/server_frozen_answers.py")
compare_captures = _load("compare_captures", "scripts/compare_captures.py")
#: The probe's own git reader, kept before ``git_says_a_clean_checkout`` replaces it.
REAL_GIT = probe._git
#: The commit every capture here names.
TEST_COMMIT = "5e5e" * 10

BUCKET = 8
CHUNK_MS = 160
CHUNK = CHUNK_MS * 16  # samples per chunk at 16 kHz
PIN = probe.DEFAULT_REVISION
CARD_UUID = "GPU-00000000-test-double-card"
SERVER_PID = 4242
RAGGED_PID = 4343
BIASED_PID = 4444
CONFIDENT_PID = 4545
HELD_PID = 4646
#: A credential in the server's environment: it must never reach the record.
TOKEN = "hf_test_double_token_never_written"
#: The server's PYTHONPATH as /proc shows it: a placeholder, kept in the record as evidence.
SERVER_PYTHONPATH = "/checkout/src:/checkout/bench/src"

#: The CLI inspects the runtime before building; this report lets it take the eager NeMo path.
#: It is a label: the fake pipeline below touches no device.
TEST_DOUBLE_RUNTIME = RuntimeReport(
    nemo_version="test-double",
    torch_version="test-double",
    cuda_available=True,
    device_name="test double (no device is touched)",
    inference_package=True,
    graph_step=False,
)

#: What /readyz of a NeMo server at this setting reports under contract C3, word confidence off:
#: what the real server reports over ``_NeMoShaped`` (a test checks it), and the base a test
#: changes to make a server that says something else.
READYZ_C3: dict[str, Any] = {
    "word_confidence": "off",
    "observed": {
        "att_context_size": [70, 1],
        "decoder_step_confidence": False,
        "decoder_graphs": False,
    },
}
#: The C3 fields as the server reports them, untouched.
AS_SERVED = "as served"
#: What every test server's /readyz carries for C3 and for C7's ``code``: ``AS_SERVED``, or
#: what replaces the server's own (for C3, None or a partial mapping: a server that does not
#: report some or all; for ``code``, the object, or None for a server that reports none).
_READYZ: dict[str, Any] = {"c3": AS_SERVED, "code": AS_SERVED}
#: The directory the probe expects the server to have imported ``verbatim`` from by default:
#: its own checkout's, resolved.
THIS_VERBATIM = str((probe.REPO / "src" / "verbatim").resolve())


@pytest.fixture(scope="module", autouse=True)
def readyz_as_a_nemo_server_reports_it() -> Iterator[None]:
    original = HealthReporter.route

    def route(self: HealthReporter, path: str) -> HealthResponse | None:
        response = original(self, path)
        served = _READYZ["c3"] == AS_SERVED and _READYZ["code"] == AS_SERVED
        if path != "/readyz" or response is None or served:
            return response
        body = json.loads(response.body)
        if _READYZ["c3"] != AS_SERVED:
            # The server's C3 fields are replaced whole: what the test puts in, and nothing else.
            body.pop("word_confidence", None)
            body.pop("observed", None)
            body.update(copy.deepcopy(_READYZ["c3"] or {}))
        if _READYZ["code"] != AS_SERVED:
            body.pop("code", None)
            if _READYZ["code"] is not None:
                body["code"] = copy.deepcopy(_READYZ["code"])
        return HealthResponse(response.status, response.content_type, json.dumps(body) + "\n")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(HealthReporter, "route", route)
        yield


@pytest.fixture(scope="module", autouse=True)
def git_says_a_clean_checkout() -> Iterator[None]:
    """git's word on the probe's checkout, for every capture here: a clean tree at
    ``TEST_COMMIT``."""
    answers = {"rev-parse": TEST_COMMIT, "status": ""}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(probe, "_git", lambda *args: answers[args[0]])
        yield


# --- the model at NeMo's seam ---


def _confidence_of(word: str) -> float:
    """The confidence ``_NeMoShaped`` puts on a final word: a function of the word alone, and
    not exact in binary, so a rounded or float32 value on the way would not be this one."""
    return int(word[1:], 16) % 9973 / 9973


class _NeMoShaped(FakeCacheAwareRNNTPipeline):
    """The CPU fake with a built NeMo pipeline's attributes at the paths
    ``verbatim.pipelines.observed`` reads (``tests/test_ws_confidence_and_readyz.py`` checks
    those paths against NeMo's own classes): the encoder's attention context, as NeMo sets it
    from the spec; the decoding configuration NeMo's builder derives for the spec's word
    confidence (``preserve_frame_confidence``, and the confidence block it copies when that is
    on); and the decoding computer's step-confidence flag and graph mode (none: eager). Each
    final segment carries ``_confidence_of`` its word, where NeMo's decoder would put its
    aggregated confidence."""

    def __init__(self, spec: NeMoPipelineSpec, **kwargs: Any) -> None:
        super().__init__(spec.chunk.ms, **kwargs)
        config = pipeline_config(spec)
        preserve = bool(config["asr"]["decoding"]["greedy"]["preserve_frame_confidence"])
        block = config["confidence"]
        self.asr_model = SimpleNamespace(
            asr_model=SimpleNamespace(
                encoder=SimpleNamespace(att_context_size=list(spec.att_context))
            ),
            decoding_cfg=SimpleNamespace(
                greedy=SimpleNamespace(preserve_frame_confidence=preserve),
                beam=SimpleNamespace(preserve_frame_confidence=False),
                confidence_cfg=SimpleNamespace(
                    aggregation=block["aggregation"],
                    method_cfg=SimpleNamespace(**block["method_cfg"]),
                ),
            ),
        )
        self.decoding_computer = SimpleNamespace(
            preserve_step_confidence=preserve, cuda_graphs_mode=None
        )

    def transcribe_step(self, requests: list[Any]) -> list[Any]:
        outputs = super().transcribe_step(requests)
        for output in outputs:
            for segment in output.final_segments:
                segment.conf = _confidence_of(segment.text)
        return outputs


# --- synthetic recordings and what the fake must answer for them ---


def _noise(chunks: int, seed: int, extra: int = 0) -> np.ndarray:
    rng = np.random.default_rng([20260924, seed])
    return (rng.uniform(-0.5, 0.5, size=chunks * CHUNK + extra) * 32767.0).astype("<i2")


def _silence(chunks: int) -> np.ndarray:
    return np.zeros(chunks * CHUNK, dtype="<i2")


def _recording(rid: str, *parts: np.ndarray) -> Any:
    pcm = np.concatenate(parts).astype("<i2").tobytes()
    return probe.Recording(rid, pcm, f"reference of {rid}")


RECORDINGS = [
    _recording("syn-a", _noise(4, 1)),  # one segment, a whole number of chunks
    _recording("syn-b", _noise(3, 2, extra=1000)),  # a short tail the server pads
    # speech, 960 ms of silence (the endpointer fires at 800), speech: two finals
    _recording("syn-c", _noise(3, 3), _silence(6), _noise(2, 4)),
    _recording("syn-d", _silence(5)),  # nothing said: one final, empty
    _recording("syn-e", _noise(6, 5)),
    _recording("syn-f", _noise(2, 6)),
]
BY_ID = {r.rid: r for r in RECORDINGS}
#: Finals the server must send per recording: one per endpoint plus the terminal one.
EXPECTED_FINALS = {"syn-a": 1, "syn-b": 1, "syn-c": 2, "syn-d": 1, "syn-e": 1, "syn-f": 1}
THREE = [BY_ID["syn-a"], BY_ID["syn-c"], BY_ID["syn-f"]]
ONE = [BY_ID["syn-f"]]


def _expected_answer(pcm: bytes) -> tuple[str, list[list[Any]]]:
    """The fake's recognition, computed here from the audio alone: one word per non-silent
    chunk, hashed from its valid samples as the server's decoder hands them over (int16 / 32768
    in float32), timed to the chunk. Independent of neighbours, so it is the right answer at
    any concurrency and either padding."""
    samples = np.frombuffer(pcm, dtype="<i2")
    words: list[list[Any]] = []
    for k in range(math.ceil(len(samples) / CHUNK)):
        valid = samples[k * CHUNK : (k + 1) * CHUNK].astype(np.float32) * (1.0 / 32768.0)
        if np.any(valid != 0):
            digest = hashlib.sha256(np.ascontiguousarray(valid, dtype=np.float32).tobytes())
            words.append(["w" + digest.hexdigest()[:6], k * CHUNK_MS, (k + 1) * CHUNK_MS])
    return " ".join(word for word, _, _ in words), words


# --- the host: a corpus, a references record, a Hugging Face cache, a card, /proc ---


@dataclass
class Host:
    loader: Callable[[int | None], tuple[list[Any], dict[str, Any]]]
    references: Path
    hub: Path
    split: int
    during_load: list[Callable[[], None]] = field(default_factory=list)


def _hub(root: Path, *, refs: str = PIN, snapshots: Sequence[str] = (PIN,)) -> Path:
    repo = root / "hub" / ("models--" + probe.DEFAULT_MODEL.replace("/", "--"))
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "snapshots").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(refs + "\n", encoding="utf-8")
    for name in snapshots:
        (repo / "snapshots" / name).mkdir(parents=True, exist_ok=True)
    return root / "hub"


def _host(
    root: Path,
    chosen: Sequence[Any],
    *,
    references: Sequence[tuple[str, str]] | None = None,
    split: int | None = None,
    loaded: Sequence[Any] | None = None,
) -> Host:
    """A corpus of ``chosen`` (the whole split), the stock record's references for it, and a
    cache holding only the pin. ``references``/``split``/``loaded`` break one of them. Anything
    in ``during_load`` happens while the corpus loads, between the first reading and the one
    the run is stamped with."""
    whole = list(chosen)
    size = len(whole) if split is None else split
    host: Host

    def load(limit: int | None) -> tuple[list[Any], dict[str, Any]]:
        for action in host.during_load:
            action()
        got = list(loaded) if loaded is not None else (whole if limit is None else whole[:limit])
        return got, {"dataset": "synthetic test recordings", "split_size_observed": size}

    pairs = references if references is not None else [(r.rid, r.reference) for r in whole]
    record = root / "stock-record.json"
    record.write_text(json.dumps({"references": dict(pairs), "runs": {}}), encoding="utf-8")
    host = Host(load, record, _hub(root), len(whole))
    return host


class FakeHostCommands:
    """nvidia-smi and ss as a server alone on the named card would answer. Records every call,
    so a test can show that only queries were asked; a test changes a field to change the card."""

    def __init__(
        self,
        port: int,
        *,
        uuid: str = CARD_UUID,
        name: str | None = None,
        on_card: Sequence[int] = (SERVER_PID,),
        listeners: Sequence[int] = (SERVER_PID,),
    ) -> None:
        self.port, self.uuid, self.on_card, self.listeners = port, uuid, on_card, listeners
        self.name = TEST_DOUBLE_RUNTIME.device_name if name is None else name
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> str:
        self.calls.append(list(argv))
        if argv[:1] == ["nvidia-smi"] and "--query-gpu=uuid,name" in argv:
            return f"{self.uuid}, {self.name}\n"
        if argv[:1] == ["nvidia-smi"] and any(a.startswith("--query-compute-apps") for a in argv):
            return "".join(f"{pid}, python, 1024\n" for pid in self.on_card)
        if argv[:1] == ["ss"]:
            listen = f"LISTEN 0 100 127.0.0.1:{self.port} 0.0.0.0:*"
            return "".join(
                f'{listen} users:(("verbatim",pid={pid},fd=7))\n' for pid in self.listeners
            )
        raise AssertionError(f"unexpected host command {argv}")


class FakeProc:
    """``/proc/<pid>/cmdline`` and ``environ`` of the server process."""

    def __init__(
        self,
        argv: Sequence[str],
        env: Mapping[str, str],
        *,
        pid: int = SERVER_PID,
        unreadable: Sequence[str] = (),
    ) -> None:
        self.argv, self.env, self.pid = list(argv), dict(env), pid
        self.unreadable = set(unreadable)
        self.calls: list[tuple[int, str]] = []

    def __call__(self, pid: int, name: str) -> bytes:
        self.calls.append((pid, name))
        if pid != self.pid:
            raise FileNotFoundError(f"no process {pid}")
        if name in self.unreadable:
            raise PermissionError(13, "Permission denied", f"{name} of {pid}")
        if name == "cmdline":
            return b"".join(arg.encode() + b"\0" for arg in self.argv)
        if name == "environ":
            return b"".join(f"{k}={v}".encode() + b"\0" for k, v in self.env.items())
        raise AssertionError(f"unexpected /proc read {name}")


def _environ(hub: Path, **change: str | None) -> dict[str, str]:
    """A server environment that places cuda:0 on card 3 and reads the cache at ``hub``."""
    env: dict[str, str | None] = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "3",
        "HF_HUB_CACHE": str(hub),
        "HF_HUB_OFFLINE": "1",
        "HOME": str(hub.parent),
        "HF_TOKEN": TOKEN,
        "PYTHONPATH": SERVER_PYTHONPATH,
        **change,
    }
    return {k: v for k, v in env.items() if v is not None}


# --- the servers: `verbatim serve` in a worker thread, fixed, ragged and biased ---


@dataclass
class Served:
    endpoint: str
    stdout: io.StringIO
    argv: list[str]
    pid: int

    @property
    def port(self) -> int:
        return int(self.endpoint.rsplit(":", 1)[1].split("/", 1)[0])

    def cmdline(self, **flags: str) -> list[str]:
        """The server's argv as /proc shows it; ``flags`` replaces a flag's value."""
        argv = list(self.argv)
        for flag, value in flags.items():
            name = "--" + flag.replace("_", "-")
            argv[argv.index(name) + 1] = value
        return ["python3", "verbatim", *argv]

    def log(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.stdout.getvalue(), encoding="utf-8")
        return path


def _one_session_at_a_time(handle: Callable[..., Any]) -> Callable[..., Any]:
    """The listener's connection handler, made to hold every new connection -- after the
    WebSocket handshake, before its session frame -- until the session before it has ended."""
    lock: list[asyncio.Lock] = []

    async def held(self: WsServer, ws: Any) -> None:
        if not lock:
            lock.append(asyncio.Lock())  # made on the server's own loop
        async with lock[0]:
            await handle(self, ws)

    return held


@contextmanager
def _serving(
    padding: str,
    *,
    pid: int,
    biasing: bool = False,
    word_confidence: str | None = None,
    hold: bool = False,
) -> Iterator[Served]:
    """``verbatim serve`` in a worker thread. ``hold``: its listener takes one session at a
    time (``_one_session_at_a_time``), bound when it starts, so no other server sees it."""
    ready = threading.Event()
    found: list[Endpoints] = []
    stoppers: list[Callable[[], None]] = []

    def on_ready(where: Endpoints, request_shutdown: Callable[[], None]) -> None:
        found.append(where)
        stoppers.append(request_shutdown)
        ready.set()

    def build(spec: NeMoPipelineSpec) -> Any:
        return boundary_for(
            _NeMoShaped(
                spec,
                num_slots=max(64, spec.num_slots),
                stop_history_eou=spec.stop_history_eou_ms,
                per_stream_biasing=spec.enable_per_stream_biasing,
            )
        )

    stdout = io.StringIO()
    hooks = Hooks(
        inspect_runtime=lambda: TEST_DOUBLE_RUNTIME,
        build_boundary=build,
        on_ready=on_ready,
        stdout=stdout,
        stderr=io.StringIO(),
    )
    argv = [
        "serve", probe.DEFAULT_MODEL,
        "--chunk", f"{CHUNK_MS}ms",
        "--bucket", str(BUCKET),
        "--padding", padding,
        "--compute-dtype", "bfloat16",
        "--eager",
        "--host", "127.0.0.1", "--ws-port", "0", "--grpc-port", "0",
        *(["--biasing"] if biasing else []),
        *(["--word-confidence", word_confidence] if word_confidence else []),
    ]  # fmt: skip
    outcome: dict[str, int] = {}
    worker = threading.Thread(
        target=lambda: outcome.update(rc=serve_main(argv, hooks=hooks)), daemon=True
    )
    patch = pytest.MonkeyPatch()
    if hold:
        patch.setattr(WsServer, "_handle", _one_session_at_a_time(WsServer._handle))
    worker.start()
    try:
        if not ready.wait(20.0):
            raise RuntimeError(f"the server never reported ready: {hooks.stderr.getvalue()}")
        patch.undo()  # the listener bound its handler when it started
        yield Served(found[0].ws_endpoint, stdout, argv, pid)
    finally:
        patch.undo()
        for stop in stoppers:
            stop()
        worker.join(20.0)
    assert not worker.is_alive(), "serve did not stop when asked"
    assert outcome.get("rc") == SERVE_OK


@pytest.fixture(scope="module")
def fixed() -> Iterator[Served]:
    with _serving("fixed", pid=SERVER_PID) as served:
        yield served


@pytest.fixture(scope="module")
def ragged() -> Iterator[Served]:
    with _serving("ragged", pid=RAGGED_PID) as served:
        yield served


@pytest.fixture(scope="module")
def biased() -> Iterator[Served]:
    with _serving("fixed", pid=BIASED_PID, biasing=True) as served:
        yield served


@pytest.fixture(scope="module")
def confident() -> Iterator[Served]:
    """A server with ``--word-confidence nemo-shipped``: its finals carry each word's "c"."""
    with _serving("fixed", pid=CONFIDENT_PID, word_confidence="nemo-shipped") as served:
        yield served


def _admission(ws_endpoint: str) -> dict[str, Any]:
    url = ws_endpoint.replace("ws://", "http://").replace("/v1/stream", "/admission")
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


@dataclass
class Rig:
    """What one run of the probe is given: the host commands, /proc and the server log."""

    commands: FakeHostCommands
    proc: FakeProc
    log: Path


def _rig(served: Served, host: Host, out: Path) -> Rig:
    return Rig(
        FakeHostCommands(served.port, on_card=(served.pid,), listeners=(served.pid,)),
        FakeProc(served.cmdline(), _environ(host.hub), pid=served.pid),
        served.log(out.with_name(out.stem + ".server.log")),
    )


def _argv(
    served: Served,
    host: Host,
    out: Path,
    rig: Rig,
    *extra: str,
    padding: str = "fixed",
    concurrency: int = 1,
) -> list[str]:
    return [
        "--endpoint", served.endpoint,
        "--out", str(out),
        "--bucket", str(BUCKET),
        "--padding", padding,
        "--concurrency", str(concurrency),
        "--server-log", str(rig.log),
        "--gpu-index", "3",
        "--gpu-uuid", CARD_UUID,
        "--references-record", str(host.references),
        *extra,
    ]  # fmt: skip


def _run(
    served: Served,
    host: Host,
    out: Path,
    *extra: str,
    padding: str = "fixed",
    concurrency: int = 1,
    rig: Rig | None = None,
) -> int:
    rig = rig or _rig(served, host, out)
    argv = _argv(served, host, out, rig, *extra, padding=padding, concurrency=concurrency)
    return probe.main(
        argv,
        loader=host.loader,
        runner=rig.commands,
        proc=rig.proc,
        expected_split=host.split,
    )


# --- server mutations: withhold a final ---


class _WithholdTerminalFinal:
    """The victim session's result stream, less its terminal final. Each final is held until
    something follows it; the one nothing follows is the terminal one, and it is never sent."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def results(self) -> AsyncIterator[Hypothesis]:
        held: Hypothesis | None = None
        async for hypothesis in self._session.results():
            if held is not None:
                yield held
                held = None
            if hypothesis.is_final:
                held = hypothesis
                continue
            yield hypothesis


class _WithholdFirstFinal:
    """The victim session's result stream, less its FIRST final. Everything after it,
    including the empty partial that follows it and the terminal final, is sent."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def results(self) -> AsyncIterator[Hypothesis]:
        dropped = False
        async for hypothesis in self._session.results():
            if hypothesis.is_final and not dropped:
                dropped = True
                continue
            yield hypothesis


@contextmanager
def _withheld(monkeypatch: pytest.MonkeyPatch, victim: int, mutation: type) -> Iterator[None]:
    """Patch the listener so the ``victim``-th session it opens loses a final. The server still
    ends the results and closes the socket with 1000, as for a finished utterance."""
    original = WsServer._write_results
    opened = itertools.count()

    async def writer(ws: Any, session: Any, options: Any, **kwargs: Any) -> None:
        if next(opened) == victim:
            session = mutation(session)
        await original(ws, session, options, **kwargs)

    monkeypatch.setattr(WsServer, "_write_results", staticmethod(writer))
    yield


# --- the capture against the unmutated server ---


def test_every_recording_is_captured_with_the_servers_own_answer(
    fixed: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _host(tmp_path, RECORDINGS)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    code = _run(fixed, host, out, concurrency=3, rig=rig)
    assert code == probe.EXIT_OK, capsys.readouterr().err
    assert out.exists() and not probe.failed_path(out).exists()
    raw = out.read_text()
    record = json.loads(raw)
    assert record["record"] == probe.RECORD
    assert record["success"] is True and record["status"] == "complete"
    assert record["failures"] == [] and record["missing"] == []
    # Test doubles took it (the host, /proc and the corpus): stamped first, as C1's fake is.
    assert next(iter(record)) == "fake_pipeline" and record["fake_pipeline"] is True
    assert record["not_a_row"].startswith("FAKE: test doubles stood in")
    # Everything was observed; nothing stood in for anything. (Ticks over budget may warn on a
    # loaded host; that is about the card keeping time, not about what was observed.)
    assert not [w for w in record["warnings"] if "could not be read" in w or "C3" in w]

    # Sent against received, counted from what was on the wire.
    counts = record["counts"]
    assert counts["recordings"] == counts["recordings_sent"] == len(RECORDINGS)
    assert counts["sessions_acknowledged"] == len(RECORDINGS)
    assert counts["recordings_with_terminal_final"] == len(RECORDINGS)
    assert counts["recordings_missing_final"] == 0
    assert counts["recordings_with_unexplained_partial_resets"] == 0
    assert counts["recordings_without_partials"] == 0
    assert counts["final_frames_received"] == sum(EXPECTED_FINALS.values())
    assert counts["text_without_words"] == 0

    # Each recording id carries ITS answer, not a neighbour's: the text and the word timings are
    # what the fake computes from that recording's audio, exactly as the wire carried them.
    for rid, entry in record["recordings"].items():
        text, words = _expected_answer(BY_ID[rid].pcm)
        assert entry["text"] == text, rid
        assert entry["words"] == words, rid
        assert entry["finals"] == EXPECTED_FINALS[rid], rid
        assert entry["terminal_final"] is True and entry["sent"] is True
        assert entry["unexplained_partial_resets"] == []
        assert entry["partials_tapped"] > 0
        assert entry["problems"] == [] and entry["client_error"] is None
        assert entry["close_code"] == 1000
        assert entry["final_frames"][-1]["audio_s"] == pytest.approx(BY_ID[rid].duration_s)
        assert entry["reference"] == f"reference of {rid}"
    assert record["recordings"]["syn-d"]["text"] == ""  # an empty final is still a final
    assert [f["audio_s"] for f in record["recordings"]["syn-c"]["final_frames"]] == [1.28, 1.76]
    assert [e["n"] for e in record["recordings"].values()] == list(range(len(RECORDINGS)))

    # This server sends no per-word confidence ("c" only with --word-confidence on), and the
    # record says so from the wire.
    assert all(e["word_confidence_present"] == 0 for e in record["recordings"].values())
    assert all(len(w) == 3 for e in record["recordings"].values() for w in e["words"])
    confidence = record["word_confidence"]
    assert confidence["with_confidence"] == 0 and confidence["confidence_key"] == "c"
    assert confidence["word_entries"] == sum(len(_expected_answer(r.pcm)[1]) for r in RECORDINGS)
    assert confidence["unparsed_word_keys_seen"] == []
    assert confidence["final_frames_with_confidence"] == 0

    # The gate's digest, over the answers the fake must give.
    expected_finals = []
    for r in RECORDINGS:
        text, words = _expected_answer(r.pcm)
        expected_finals.append(FinalRecord(r.rid, text, tuple(tuple(w) for w in words)))
    assert record["finals_digest"] == finals_digest(expected_finals)

    # Stamps: observed where the server or the host can be asked.
    server = record["server"]
    assert server["url"] == fixed.endpoint
    assert server["model"]["reported_by_server"] == probe.DEFAULT_MODEL
    revision = server["model_revision"]
    assert revision["observed_in_hf_cache"] == PIN
    assert revision["hub_dir"] == str(host.hub)  # the SERVER's cache, from its environment
    assert revision["hub_dir_source"] == probe.HUB_FROM_SERVER_ENVIRON
    assert revision["after"]["snapshots"] == [PIN]
    assert server["bucket"] == {
        "reported_by_server": BUCKET,
        "observed_in_cmdline": BUCKET,
        "declared": BUCKET,
    }
    assert server["padding"]["reported_by_server"] is None
    assert server["padding"]["observed_in_cmdline"] == "fixed"
    assert server["padding"]["observed_in_server_log"] == "fixed"
    assert server["padding"]["declared"] == "fixed"
    assert server["execution"]["reported_by_server"] == "eager"
    assert server["execution"]["observed_in_cmdline"] == "eager"
    assert server["chunk_ms"]["reported_by_server"] == CHUNK_MS
    assert server["chunk_ms"]["session_frames"] == [str(CHUNK_MS)]
    assert server["dtype"] == {"reported_by_server": "bfloat16", "declared": "bfloat16"}
    assert server["matmul_precision"]["declared"] == "high"
    assert server["matmul_precision"]["installed_default"] == "high"
    assert server["att_context"]["observed_in_readyz"] == [70, 1]
    assert server["att_context"]["derived_from_cmdline"] == [70, 1]
    assert server["att_context"]["observed_in_server_log"] == [70, 1]
    assert server["word_confidence"]["reported_by_server"] == "off"
    assert server["word_confidence"]["decoder_step_confidence"] is False
    assert server["decoder_graphs"]["reported_by_server"] is False
    assert server["biasing"] == {
        "reported_by_server": False,
        "observed_in_cmdline": False,
        "observed_in_server_log": False,
        "declared": False,
    }
    for moment in ("first", "before", "after"):
        assert server["gpu"][moment]["uuid"] == CARD_UUID
        assert server["gpu"][moment]["compute_pids"] == [SERVER_PID]
        assert server["gpu"][moment]["listener_pids"] == [SERVER_PID]
        assert server["process"][moment]["pid"] == SERVER_PID
        assert server["process"][moment]["environ"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert server["process"][moment]["environ"]["PYTHONPATH"] == SERVER_PYTHONPATH
        assert server["server_log"][moment]["banner"]["banners"] == 1
    assert TOKEN not in raw  # the environment is digested, never written
    ticks = server["tick_ids"]
    assert ticks["first"] <= ticks["before"] < ticks["after"]
    load = server["load"]
    assert load["admitted_during_run"] == len(RECORDINGS)
    assert load["refused_during_run"] == 0
    assert load["ticks_during_run"] > 0
    for side in ("before", "after"):
        for key in (
            "p95_tick_ms",
            "degradation_level",
            "consecutive_overruns",
            "ticks_over_budget_total",
        ):
            assert load[side][key] is not None, (side, key)
    concurrency = record["client"]["concurrency"]
    assert concurrency["configured"] == 3 and concurrency["observed_peak_in_flight"] == 3
    assert concurrency["sessions_timed"] == len(RECORDINGS)
    spans = [tuple(e["in_flight_s"]) for e in record["recordings"].values()]
    # Seconds from the start of the run: inside it (the wall clock is rounded to milliseconds).
    assert all(0 < start < end <= record["wall_clock_s"] + 0.0005 for start, end in spans)
    # The stamped peak is the one the record's own stamps give, by the probe's rule and by the
    # comparator's recount.
    assert probe.peak_in_flight(spans) == 3
    assert compare_captures.recounted_peak(spans) == 3
    assert sum(concurrency["occupancy_at_first_audio"].values()) == len(RECORDINGS)
    assert max(map(int, concurrency["occupancy_at_first_audio"])) == 3
    assert record["started"] and record["finished"]
    assert record["corpus"]["loaded"] == record["corpus"]["expected"] == len(RECORDINGS)
    assert record["corpus"]["checked_against"]["references"] == len(RECORDINGS)
    # The comparator's whole rule, word for word, and nothing after it.
    assert record["frozen_claim"] == (
        "not claimed by one capture: one digest cannot tell invariant answers from "
        "batch-dependent ones. scripts/compare_captures.py claims it only for "
        f"{compare_captures.FROZEN_RULE}"
    )
    assert "tracked_files_modified" in record["client"]  # the comparator reads it
    # The code that ran, file by file, as this process imported it.
    assert record["client"]["imported_from"] == {
        "verbatim_bench.client": str(Path(bench_client.__file__).resolve()),
        "verbatim": str(Path(sys.modules["verbatim"].__file__).resolve()),
    }
    assert record["client"]["checkout"] == str(probe.REPO)
    assert record["client"]["verbatim_commit"] == TEST_COMMIT
    assert record["client"]["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES")
    # The hashes the comparator's client-code check compares are of the files that ran: this
    # probe and the session client, each by its own bytes.
    client = record["client"]
    assert client["probe_sha256"] == hashlib.sha256(probe.HERE.read_bytes()).hexdigest()
    bench_bytes = Path(bench_client.__file__).read_bytes()
    assert client["bench_client_sha256"] == hashlib.sha256(bench_bytes).hexdigest()
    assert client["probe_sha256"] != client["bench_client_sha256"]
    # Read at the start and again at the end, and the same.
    assert client["provenance_at_end"] == {key: client[key] for key in probe.PROVENANCE}
    assert set(probe.PROVENANCE) == {
        "verbatim_commit", "tracked_files_modified", "probe_sha256", "bench_client_sha256",
    }  # fmt: skip
    # Contract C7: which verbatim the server imported, as its /readyz reports it, and what the
    # run expected.
    import verbatim_bench

    assert server["code"]["reported_by_server"] == {
        "verbatim_path": THIS_VERBATIM,
        "bench_path": os.path.realpath(os.path.dirname(verbatim_bench.__file__)),
    }
    assert server["code"]["expected_verbatim_path"] == THIS_VERBATIM
    assert server["code"]["expected_from"] == "this client's own checkout"

    # The host was only asked questions, three times over, about the one server process.
    assert {c[0] for c in rig.commands.calls} == {"nvidia-smi", "ss"}
    assert all(
        any(a.startswith("--query") for a in c) for c in rig.commands.calls if c[0] == "nvidia-smi"
    )
    assert sorted(set(rig.proc.calls)) == [(SERVER_PID, "cmdline"), (SERVER_PID, "environ")]
    assert len(rig.proc.calls) == 6

    # The tap is gone once the run is over.
    assert bench_client.websockets is websockets


# --- the lost-final guards go red ---


def test_a_withheld_final_turns_the_run_red_though_the_client_reports_no_error(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """syn-a has one final. With it withheld, the server still closes the socket normally and
    ``run_session`` returns ``error`` None and an empty text: a probe that trusted ``error``
    would file an empty transcript as the server's answer."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    with _withheld(monkeypatch, 0, _WithholdTerminalFinal):
        code = _run(fixed, host, out)
    assert code == probe.EXIT_FAILED
    assert not out.exists(), "a failed capture must never be written where the answers go"
    record = json.loads(probe.failed_path(out).read_text())
    assert record["success"] is False and record["status"] == "FAILED"
    assert [m["id"] for m in record["missing"]] == ["syn-a"]
    victim = record["recordings"]["syn-a"]
    assert victim["client_error"] is None  # the harness client saw nothing wrong
    assert victim["finals"] == 0 and victim["text"] == ""
    # With no final it has no time in flight, and the record counts only the sessions timed.
    assert victim["in_flight_s"] is None
    assert [record["recordings"][r]["in_flight_s"] is None for r in ("syn-c", "syn-f")] == [
        False,
        False,
    ]
    assert record["client"]["concurrency"]["sessions_timed"] == 2
    assert sum(record["client"]["concurrency"]["occupancy_at_first_audio"].values()) == 2
    assert victim["sent"] is True and victim["close_code"] == 1000
    assert victim["terminal_final"] is False
    assert any(p.startswith("no final") for p in victim["problems"])
    counts = record["counts"]
    assert counts["recordings_sent"] == 3
    assert counts["recordings_with_terminal_final"] == 2
    assert counts["recordings_missing_final"] == 1
    assert record["failures"][0].startswith("1 of 3 recording(s) failed; first syn-a")
    for rid in ("syn-c", "syn-f"):
        assert record["recordings"][rid]["problems"] == []


def test_losing_only_the_terminal_final_is_caught_where_at_least_one_final_would_pass(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """syn-c endpoints mid-stream, so its first final arrives and only the terminal one is
    withheld. The client counts one final, no error, and a transcript missing its second half."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    with _withheld(monkeypatch, 1, _WithholdTerminalFinal):
        code = _run(fixed, host, out)
    assert code == probe.EXIT_FAILED
    assert not out.exists()
    record = json.loads(probe.failed_path(out).read_text())
    assert [m["id"] for m in record["missing"]] == ["syn-c"]
    victim = record["recordings"]["syn-c"]
    assert victim["client_error"] is None
    assert victim["finals"] == 1  # "at least one final" holds
    full_text, _ = _expected_answer(BY_ID["syn-c"].pcm)
    assert victim["text"] == " ".join(full_text.split()[:3])  # the first segment only
    assert victim["terminal_final"] is False
    assert any(p.startswith("no terminal final") for p in victim["problems"])


def test_losing_the_first_final_is_caught_though_the_terminal_final_arrives(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """syn-c's FIRST final is withheld and its terminal final arrives, covering all the audio:
    the terminal rule passes and the transcript is missing its first half. Only the partial
    reset at 1.28 s, with no final beside it, shows the loss."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    with _withheld(monkeypatch, 1, _WithholdFirstFinal):
        code = _run(fixed, host, out)
    assert code == probe.EXIT_FAILED
    assert not out.exists()
    record = json.loads(probe.failed_path(out).read_text())
    assert [m["id"] for m in record["missing"]] == ["syn-c"]
    victim = record["recordings"]["syn-c"]
    assert victim["client_error"] is None
    assert victim["terminal_final"] is True  # the terminal rule alone would pass
    assert victim["finals"] == 1
    full_text, _ = _expected_answer(BY_ID["syn-c"].pcm)
    assert victim["text"] == " ".join(full_text.split()[3:])  # the second segment only
    assert victim["unexplained_partial_resets"] == [1.28]
    assert [p for p in victim["problems"] if p.startswith("lost mid-recording final")]
    assert record["counts"]["recordings_with_unexplained_partial_resets"] == 1


# --- refusals before a byte is sent, against the running server ---


def _refused_without_a_session(
    served: Served,
    out: Path,
    run: Callable[[], int],
    capsys: pytest.CaptureFixture[str],
    *reasons: str,
) -> list[str]:
    """Refused, for EVERY named reason, with nothing written and no session admitted."""
    before = _admission(served.endpoint)["admitted_total"]
    capsys.readouterr()
    assert run() == probe.EXIT_REFUSED
    said = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[refused]")]
    for reason in reasons:
        assert [line for line in said if reason in line], (reason, said)
    assert not out.exists() and not probe.failed_path(out).exists()
    assert _admission(served.endpoint)["admitted_total"] == before
    return said


@pytest.mark.parametrize(
    ("extra", "reasons"),
    [
        (("--bucket", "16"), ["the server reports bucket 8, the run declares 16"]),
        (
            ("--dtype", "float32"),
            ["--dtype 'float32' and the server reports precision 'bfloat16'"],
        ),
        (("--model", "some/other-checkpoint"), ["the run declares 'some/other-checkpoint'"]),
        (
            ("--execution", "graph path"),
            ["reports execution 'eager', the run declares 'graph path'"],
        ),
        (("--pipeline", "cache_aware_ctc"), ["the run declares 'cache_aware_ctc'"]),
        (("--chunk-ms", "560"), ["configured for 560 ms chunks and the server reports 160 ms"]),
        (("--matmul", "highest"), ["--matmul 'highest' and the installed NeMoPipelineSpec"]),
        (("--concurrency", str(BUCKET + 1)), [f"concurrency {BUCKET + 1} exceeds the bucket"]),
        (("--model-revision", PIN[:8]), [f"not the pin '{PIN[:8]}'"]),
        (
            ("--word-confidence", "paper-best"),
            [
                "/readyz reports word confidence 'off', the run declares 'paper-best'",
                "/readyz observes decoder_step_confidence False on the built decoder",
                "the server was started with word confidence 'off' (/proc/4242/cmdline)",
                "the server's banner says word confidence off, the run declares paper-best",
            ],
        ),
        (
            ("--decoder-graphs", "on"),
            [
                "/readyz observes decoder_graphs False in the built pipeline, the run declares "
                "True",
                "the server was started with decoder graphs False (/proc/4242/cmdline)",
                "the server's banner says decoder graphs False, the run declares True",
            ],
        ),
        (
            ("--att-context", "70,13"),
            [
                "/readyz observes att_context_size [70, 1] on the built encoder, the run declares "
                "[70, 13]",
                "the server was started with attention context [70, 1] (/proc/4242/cmdline)",
                "the banner names att_context [70, 1], the run declares [70, 13]",
            ],
        ),
    ],
    ids=lambda value: value[0].lstrip("-") if isinstance(value, tuple) else "",
)
def test_a_declaration_the_server_or_host_contradicts_is_refused_before_any_session(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, str],
    reasons: list[str],
) -> None:
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out, *extra), capsys, *reasons)


@pytest.mark.parametrize(
    ("c3", "reason"),
    [
        ({"word_confidence": "nemo-shipped"}, "/readyz reports word confidence 'nemo-shipped'"),
        (
            {"observed": {**READYZ_C3["observed"], "att_context_size": [70, 13]}},
            "/readyz observes att_context_size [70, 13] on the built encoder, the run declares "
            "[70, 1]",
        ),
        (
            {"observed": {**READYZ_C3["observed"], "decoder_step_confidence": True}},
            "/readyz observes decoder_step_confidence True on the built decoder",
        ),
        (
            {"observed": {**READYZ_C3["observed"], "decoder_graphs": True}},
            "/readyz observes decoder_graphs True in the built pipeline, the run declares False",
        ),
    ],
    ids=["word-confidence", "att-context", "step-confidence", "decoder-graphs"],
)
def test_a_readyz_that_observes_another_model_than_declared_is_refused(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    c3: dict[str, Any],
    reason: str,
) -> None:
    """Contract C3: /readyz reports what the BUILT model is. A server whose flags and banner are
    right but whose built encoder or decoder is not is refused on the observation alone."""
    monkeypatch.setitem(_READYZ, "c3", {**READYZ_C3, **c3})
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    said = _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out), capsys, reason)
    assert all("/readyz" in line for line in said), said  # nothing else disagreed


NO_ATT_CONTEXT = "/readyz does not report observed.att_context_size (contract C3)"


@pytest.mark.parametrize(
    "c3",
    [
        None,
        {**READYZ_C3, "observed": {k: v for k, v in READYZ_C3["observed"].items()
                                   if k != "att_context_size"}},
        {**READYZ_C3, "observed": {**READYZ_C3["observed"], "att_context_size": None}},
    ],
    ids=["no-c3-fields", "att-context-absent", "att-context-null"],
)  # fmt: skip
def test_a_server_that_does_not_observe_its_attention_context_is_refused_before_any_session(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    c3: Any,
) -> None:
    """``compare_captures.py`` refuses every capture without ``/readyz``'s observed attention
    context, so a capture from such a server would be a real-time pass over the whole corpus
    that nothing can compare: it is refused before a byte is sent, not warned about after."""
    monkeypatch.setitem(_READYZ, "c3", c3)
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    said = _refused_without_a_session(
        fixed, out, lambda: _run(fixed, host, out), capsys, NO_ATT_CONTEXT
    )
    assert all(NO_ATT_CONTEXT in line for line in said), said  # nothing else was wrong


def test_a_server_without_the_other_c3_fields_is_captured_with_them_stamped_not_observed(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Word confidence, step confidence and decoder graphs have other sources the comparator
    reads (the command line, the banner); a server that does not report them is captured, with
    each named in a loud warning and stamped null."""
    monkeypatch.setitem(_READYZ, "c3", {"observed": {"att_context_size": [70, 1]}})
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    assert _run(fixed, host, out) == probe.EXIT_OK
    err = capsys.readouterr().err
    for name in ("word_confidence", "decoder_step_confidence", "decoder_graphs"):
        assert f"!!! [WARNING] /readyz does not report {name} (contract C3)" in err, name
    assert "att_context_size (contract C3)" not in err
    record = json.loads(out.read_text())
    assert record["server"]["att_context"]["observed_in_readyz"] == [70, 1]
    assert record["server"]["word_confidence"]["reported_by_server"] is None
    assert [w for w in record["warnings"] if "decoder_graphs (contract C3)" in w]


NO_CODE = "/readyz does not report code.verbatim_path (contract C7)"


def test_the_served_readyz_reports_the_code_this_process_imported(
    fixed: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract C7 as the real server code reports it. The servers here run in this process, so
    the ``verbatim`` they imported is the one these tests imported: this checkout's, which is
    what every capture here expects by default."""
    import verbatim_bench

    import verbatim

    code = probe.reported_code(probe.read_server(fixed.endpoint))
    assert code == {
        "verbatim_path": os.path.realpath(os.path.dirname(verbatim.__file__)),
        "bench_path": os.path.realpath(os.path.dirname(verbatim_bench.__file__)),
    }
    assert code["verbatim_path"] == THIS_VERBATIM, "run the tests with this checkout's src first"
    monkeypatch.setitem(_READYZ, "code", None)
    assert probe.reported_code(probe.read_server(fixed.endpoint)) is None


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (None, NO_CODE),
        ({"bench_path": None}, NO_CODE),
        ({"verbatim_path": None, "bench_path": None}, NO_CODE),
        (
            {"verbatim_path": "/elsewhere/src/verbatim", "bench_path": None},
            "/readyz says the server imported verbatim from /elsewhere/src/verbatim, and the run "
            f"expects {THIS_VERBATIM}: the server does not run the checkout's code",
        ),
    ],
    ids=["no-code", "no-verbatim-path", "verbatim-path-null", "another-checkout"],
)
def test_a_server_that_does_not_run_this_checkouts_code_is_refused_before_any_session(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    code: Any,
    reason: str,
) -> None:
    """Contract C7: a server started without the checkout's src first on the path answers from
    whatever ``verbatim`` its environment imports (an editable install of another checkout),
    with every flag, banner and C3 field right. Refused on ``/readyz`` ``code`` alone."""
    monkeypatch.setitem(_READYZ, "code", code)
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    said = _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out), capsys, reason)
    assert all(reason in line for line in said), said  # nothing else was wrong


def test_the_server_checkout_names_the_code_the_server_must_run(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--server-checkout DIR``: the server must have imported ``DIR/src/verbatim``, resolved,
    and nothing else; without it, this client's own checkout."""
    host = _host(tmp_path, ONE)
    other = tmp_path / "other-checkout"
    (other / "src" / "verbatim").mkdir(parents=True)
    other_verbatim = str((other / "src" / "verbatim").resolve())
    # This checkout's server, and --server-checkout naming the other: refused.
    out = tmp_path / "this.json"
    _refused_without_a_session(
        fixed,
        out,
        lambda: _run(fixed, host, out, "--server-checkout", str(other)),
        capsys,
        f"/readyz says the server imported verbatim from {THIS_VERBATIM}, and the run expects "
        f"{other_verbatim}",
    )
    # A server that runs the other checkout's code, named through a symbolic link: accepted,
    # and the record says what was expected and why.
    served_from = {"verbatim_path": other_verbatim, "bench_path": None}
    monkeypatch.setitem(_READYZ, "code", served_from)
    link = tmp_path / "link-to-other"
    link.symlink_to(other)
    out = tmp_path / "other.json"
    assert _run(fixed, host, out, "--server-checkout", str(link)) == probe.EXIT_OK
    code = json.loads(out.read_text())["server"]["code"]
    assert code["reported_by_server"] == served_from
    assert code["expected_verbatim_path"] == other_verbatim
    assert code["expected_from"] == "--server-checkout"
    # The same server without --server-checkout: this client's own checkout is expected.
    out = tmp_path / "default.json"
    _refused_without_a_session(
        fixed,
        out,
        lambda: _run(fixed, host, out),
        capsys,
        f"/readyz says the server imported verbatim from {other_verbatim}, and the run expects "
        f"{THIS_VERBATIM}",
    )


@pytest.mark.parametrize(
    ("hub", "reason"),
    [
        ({"snapshots": (PIN, "f" * 40)}, "2 snapshots in"),
        ({"refs": "f" * 40, "snapshots": ("f" * 40,)}, f"is '{'f' * 40}', not the pin"),
        ({"snapshots": ()}, "0 snapshots in"),
    ],
    ids=["two-snapshots", "refs-not-the-pin", "no-snapshot"],
)
def test_a_server_cache_that_does_not_hold_only_the_pin_is_refused(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    hub: dict[str, Any],
    reason: str,
) -> None:
    """The cache read is the one the SERVER's environment names, not this client's."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    rig.proc.env["HF_HUB_CACHE"] = str(_hub(tmp_path / "server-side", **hub))
    _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out, rig=rig), capsys, reason)


@pytest.mark.parametrize(
    ("references", "reason"),
    [
        (
            [
                ("syn-a", "reference of syn-a"),
                ("syn-c", "a different reference"),
                ("syn-f", "reference of syn-f"),
            ],
            "1 reference(s) differ from the stock record's",
        ),
        (
            [
                ("syn-c", "reference of syn-c"),
                ("syn-a", "reference of syn-a"),
                ("syn-f", "reference of syn-f"),
            ],
            "2 recording(s) out of the stock record's order",
        ),
        (
            [("syn-a", "reference of syn-a"), ("syn-c", "reference of syn-c")],
            "the references record holds 2 references, expected 3",
        ),
    ],
    ids=["reference-text", "order", "record-short"],
)
def test_a_corpus_the_stock_record_disagrees_with_is_refused(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    references: list[tuple[str, str]],
    reason: str,
) -> None:
    host = _host(tmp_path, THREE, references=references)
    out = tmp_path / "answers.json"
    _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out), capsys, reason)


def test_the_full_corpus_run_refuses_a_corpus_short_of_every_recording(
    fixed: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _host(tmp_path, THREE, loaded=THREE[:2])
    out = tmp_path / "answers.json"
    reason = "the corpus yielded 2 recordings, expected 3"
    _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out), capsys, reason)


def test_a_split_of_the_wrong_size_is_refused(
    fixed: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _host(tmp_path, THREE, split=4)
    out = tmp_path / "answers.json"
    reason = "the split holds 4 recordings, expected 3"
    _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out), capsys, reason)


@pytest.mark.parametrize(
    ("commands", "reason"),
    [
        ({"uuid": "GPU-some-other-card"}, "nvidia-smi -i 3 is 'GPU-some-other-card'"),
        ({"name": "some other device"}, "nvidia-smi -i 3 names 'some other device'"),
        ({"on_card": (777,)}, f"(pid {SERVER_PID}) holds no context on card 3"),
        (
            {"on_card": (SERVER_PID, 777)},
            f"card 3 is shared: pid(s) [777] hold a context on it beside the server (pid "
            f"{SERVER_PID})",
        ),
        ({"listeners": ()}, "0 process(es) listen on port"),
        ({"listeners": (SERVER_PID, 5151)}, "2 process(es) listen on port"),
    ],
    ids=[
        "uuid",
        "device-name",
        "server-not-on-card",
        "card-shared",
        "no-listener",
        "two-listeners",
    ],
)
def test_the_card_guard_refuses_a_server_it_cannot_place_alone_on_the_named_card(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    commands: dict[str, Any],
    reason: str,
) -> None:
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    for key, value in commands.items():
        setattr(rig.commands, key, value)
    _refused_without_a_session(fixed, out, lambda: _run(fixed, host, out, rig=rig), capsys, reason)


def _cmdline(**flags: str) -> Callable[[Served], list[str]]:
    return lambda served: served.cmdline(**flags)


@pytest.mark.parametrize(
    ("argv", "env", "extra", "reason"),
    [
        (
            _cmdline(padding="ragged"),
            {},
            (),
            "the server was started with padding 'ragged' (/proc/4242/cmdline), the run declares "
            "'fixed'",
        ),
        (_cmdline(bucket="16"), {}, (), "the server was started with bucket 16"),
        (_cmdline(chunk="560ms"), {}, (), "the server was started with chunk ms 560"),
        (_cmdline(compute_dtype="float32"), {}, (), "the server was started with compute dtype"),
        (
            lambda s: [*s.cmdline(), "--biasing"],
            {},
            (),
            "the server was started with biasing True",
        ),
        (
            lambda s: [*s.cmdline(), "--att-context-left", "56"],
            {},
            (),
            "the server was started with attention context [56, 1]",
        ),
        (
            lambda s: [x for x in s.cmdline() if x != "--eager"],
            {},
            (),
            "the server was started with execution 'graph path'",
        ),
        (
            lambda s: [*s.cmdline()[:3], "--no-such-flag"],
            {},
            (),
            "is not a server this client can read: verbatim's serve parser rejects",
        ),
        (
            lambda s: ["python3", "-m", "http.server", str(s.port)],
            {},
            (),
            "is not a server this client can read: not a `verbatim serve` command line",
        ),
        (
            None,
            {"CUDA_VISIBLE_DEVICES": "0"},
            (),
            "the server's cuda:0 is card '0' by its CUDA_VISIBLE_DEVICES='0', not card 3",
        ),
        (None, {"CUDA_VISIBLE_DEVICES": None}, (), "the server's cuda:0 is card '0'"),
        (
            None,
            {"CUDA_DEVICE_ORDER": None},
            (),
            "the server's CUDA_DEVICE_ORDER is None, not 'PCI_BUS_ID'",
        ),
        (
            None,
            {},
            ("--hf-hub-cache", "/nonexistent/hub"),
            "the server resolves its Hugging Face hub cache to",
        ),
    ],
    ids=[
        "padding",
        "bucket",
        "chunk",
        "dtype",
        "biasing",
        "att-context",
        "graph-path",
        "bad-flag",
        "not-verbatim",
        "visible-devices",
        "all-devices-visible",
        "device-order",
        "hub-cache",
    ],
)
def test_the_server_process_contradicting_the_run_is_refused(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: Callable[[Served], list[str]] | None,
    env: dict[str, str | None],
    extra: tuple[str, ...],
    reason: str,
) -> None:
    """The flags and environment are the server PROCESS's, read from /proc: a server started
    with other flags than its label, or on another card, is refused though /readyz agrees."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    if argv is not None:
        rig.proc.argv = argv(fixed)
    rig.proc.env = _environ(host.hub, **env)
    _refused_without_a_session(
        fixed, out, lambda: _run(fixed, host, out, *extra, rig=rig), capsys, reason
    )


@pytest.mark.parametrize(
    ("unreadable", "warning", "extra", "source"),
    [
        (("cmdline",), "the server's command line could not be read (PermissionError", (), None),
        (
            ("environ",),
            "the server's environment could not be read (PermissionError",
            ("--hf-hub-cache", "<hub>"),
            "--hf-hub-cache",
        ),
        (
            ("environ",),
            "the server's environment could not be read (PermissionError",
            (),
            "this client's own",
        ),
    ],
    ids=["cmdline", "environ-flag", "environ-client-cache"],
)
def test_an_unreadable_server_process_is_said_loudly_and_the_log_or_flag_stands_in(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    unreadable: tuple[str, ...],
    warning: str,
    extra: tuple[str, ...],
    source: str | None,
) -> None:
    """Without the server's environment, the revision is read from --hf-hub-cache or, without
    that, from this client's own cache (here the test's hub, as this process resolves it). The
    record names which, never the server's environment, and the comparator refuses it."""
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    rig.proc.unreadable = set(unreadable)
    monkeypatch.setattr(probe, "default_hub_dir", lambda: host.hub)
    extra = tuple(str(host.hub) if x == "<hub>" else x for x in extra)
    assert _run(fixed, host, out, *extra, rig=rig) == probe.EXIT_OK
    err = capsys.readouterr().err
    assert f"!!! [WARNING] {warning}" in err
    record = json.loads(out.read_text())
    assert [w for w in record["warnings"] if w.startswith(warning)]
    server = record["server"]
    if source is None:
        assert server["padding"]["observed_in_cmdline"] is None
        assert server["padding"]["observed_in_server_log"] == "fixed"
        assert server["model_revision"]["hub_dir_source"] == probe.HUB_FROM_SERVER_ENVIRON
    else:
        said = server["model_revision"]["hub_dir_source"]
        assert said.startswith(source), said
        assert said.endswith("(the server's environment was not read)"), said
        assert server["model_revision"]["hub_dir"] == str(host.hub)
        assert server["model_revision"]["observed_in_hf_cache"] == PIN
        # The revision came from a cache the server may not read: the comparator does not take
        # it as observed of the server, and refuses the capture.
        capture = compare_captures.load_capture(_unstamped(out))
        assert (
            f"{capture.path}: model_revision was not observed of the server: "
            f"server.model_revision.hub_dir_source is {said!r}, not "
            f"{compare_captures.SERVER_HUB_SOURCE!r}"
        ) in compare_captures.capture_problems(capture)


def test_a_server_started_with_biasing_is_refused_by_every_source_that_sees_it(
    biased: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    _refused_without_a_session(
        biased,
        out,
        lambda: _run(biased, host, out),
        capsys,
        "the server reports biasing on",
        f"the server was started with biasing True (/proc/{BIASED_PID}/cmdline)",
        "the server's banner says biasing ON",
    )


@pytest.mark.parametrize(
    ("server", "log_of", "padding", "reasons"),
    [
        (
            "fixed",
            "fixed",
            "ragged",
            [
                "the server's banner says padding fixed, the run declares ragged",
                "the server was started with padding 'fixed' (/proc/4242/cmdline), the run "
                "declares 'ragged'",
            ],
        ),
        (
            "ragged",
            "ragged",
            "fixed",
            [
                "the server's banner says padding ragged, the run declares fixed",
                "the server was started with padding 'ragged' (/proc/4343/cmdline), the run "
                "declares 'fixed'",
            ],
        ),
        ("fixed", "ragged", "fixed", ["the banner's server listens on port"]),
    ],
    ids=["ragged-declared-on-fixed", "fixed-declared-on-ragged", "another-servers-log"],
)
def test_a_padding_or_log_the_server_contradicts_is_refused(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    server: str,
    log_of: str,
    padding: str,
    reasons: list[str],
) -> None:
    served: Served = request.getfixturevalue(server)
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    rig = _rig(served, host, out)
    request.getfixturevalue(log_of).log(rig.log)
    _refused_without_a_session(
        served,
        out,
        lambda: _run(served, host, out, padding=padding, rig=rig),
        capsys,
        *reasons,
    )


@pytest.mark.parametrize(
    "drop",
    [("--server-log",), ("--gpu-uuid",), ("--gpu-index",)],
    ids=["server-log", "gpu-uuid", "gpu-index"],
)
def test_the_server_log_and_the_card_are_mandatory(
    fixed: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str], drop: tuple[str]
) -> None:
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    argv = _argv(fixed, host, out, _rig(fixed, host, out))
    at = argv.index(drop[0])
    del argv[at : at + 2]
    with pytest.raises(SystemExit) as stopped:
        probe.main(argv, loader=host.loader, expected_split=host.split)
    assert stopped.value.code == 2
    assert f"the following arguments are required: {drop[0]}" in capsys.readouterr().err


def test_without_a_references_record_the_run_is_refused(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(probe.REFERENCES_ENV, raising=False)
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    argv = _argv(fixed, host, out, rig)
    at = argv.index("--references-record")
    del argv[at : at + 2]

    def run() -> int:
        return probe.main(argv, loader=host.loader, runner=rig.commands, proc=rig.proc)

    _refused_without_a_session(fixed, out, run, capsys, "no references record")
    monkeypatch.setenv(probe.REFERENCES_ENV, str(tmp_path / "missing.json"))
    _refused_without_a_session(fixed, out, run, capsys, "the references record could not be read")


# --- a record is written once ---


EARLIER = "an earlier multi-hour record\n"


@pytest.mark.parametrize("existing", ["out", "failed"])
def test_a_path_a_record_already_holds_is_refused_before_anything_is_asked(
    fixed: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str], existing: str
) -> None:
    """Without this refusal a re-run with the same --out spends a real-time pass and then
    writes over the earlier record, or files its failure beside an earlier failure's."""
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    there, other = (out, probe.failed_path(out))[:: 1 if existing == "out" else -1]
    there.write_text(EARLIER, encoding="utf-8")
    rig = _rig(fixed, host, out)
    before = _admission(fixed.endpoint)["admitted_total"]
    capsys.readouterr()
    assert _run(fixed, host, out, rig=rig) == probe.EXIT_REFUSED
    err = capsys.readouterr().err
    assert f"[refused] {there} exists; a record is written once, pick a new --out" in err
    assert there.read_text(encoding="utf-8") == EARLIER
    assert not other.exists() and not list(tmp_path.glob("*.tmp"))
    assert rig.commands.calls == [] and rig.proc.calls == []  # refused before asking anything
    assert _admission(fixed.endpoint)["admitted_total"] == before


def test_a_record_that_appears_at_the_path_during_the_run_is_not_written_over(
    fixed: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Another capture given the same --out, finishing first: its record stays as it is, and
    this run's complete record is kept beside it under a name of its own."""
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    host.during_load.append(lambda: out.write_text(EARLIER, encoding="utf-8"))
    capsys.readouterr()
    assert _run(fixed, host, out) == probe.EXIT_FAILED
    assert out.read_text(encoding="utf-8") == EARLIER
    kept = list(tmp_path.glob("answers.json.*.tmp"))
    assert len(kept) == 1, kept
    record = json.loads(kept[0].read_text(encoding="utf-8"))
    assert record["success"] is True and record["counts"]["recordings"] == len(ONE)
    err = capsys.readouterr().err
    assert f"FINAL: NOT WRITTEN to {out}; the record is at {kept[0]}" in err
    assert not probe.failed_path(out).exists()


@pytest.mark.parametrize("links", [True, False], ids=["hard-link", "no-hard-links"])
@pytest.mark.parametrize("success", [True, False], ids=["complete", "failed"])
def test_write_record_never_replaces_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, links: bool, success: bool
) -> None:
    if not links:

        def no_links(*_: Any) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(probe.os, "link", no_links)
    out = tmp_path / "answers.json"
    target = out if success else probe.failed_path(out)
    assert probe.write_record({"success": success, "n": 1}, out) == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"success": success, "n": 1}
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(probe.RecordExists) as exists:
        probe.write_record({"success": success, "n": 2}, out)
    assert json.loads(target.read_text(encoding="utf-8")) == {"success": success, "n": 1}
    assert exists.value.target == target
    assert json.loads(exists.value.kept.read_text(encoding="utf-8"))["n"] == 2


# --- the same server after the corpus load: refused before a byte is sent ---


def _readings(monkeypatch: pytest.MonkeyPatch, changes: Mapping[int, Callable[[Any], Any]]) -> None:
    """Change the n-th reading of the server (1: first, 2: after the load, 3: after the run)."""
    real = probe.read_server
    calls = itertools.count(1)

    def read_server(endpoint: str) -> Any:
        reading = real(endpoint)
        change = changes.get(next(calls))
        return reading if change is None else change(reading)

    monkeypatch.setattr(probe, "read_server", read_server)


def _facts(**change: Any) -> Callable[[Any], Any]:
    return lambda r: dataclasses.replace(r, facts=dataclasses.replace(r.facts, **change))


#: A /readyz ``code`` of a server that imported another checkout's ``verbatim``.
ELSEWHERE_CODE = {"verbatim_path": "/elsewhere/src/verbatim", "bench_path": None}


def _code(code: Any) -> Callable[[Any], Any]:
    return lambda r: dataclasses.replace(r, readyz={**r.readyz, "code": code})


def _new_banner(served: Served, log: Path) -> Callable[[], None]:
    def append() -> None:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(served.stdout.getvalue())

    return append


@pytest.mark.parametrize(
    "case",
    [
        "identity",
        "code",
        "readyz-refused",
        "tick-back",
        "restarted-same-flags",
        "restarted-ragged",
        "environment",
        "new-banner",
        "stopped",
        "card-shared",
    ],
)
def test_a_server_that_is_not_the_same_after_the_corpus_load_is_refused(
    fixed: Served,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """The load takes minutes on the real corpus. Every check runs again after it, and the
    server must be the one first read: here it changes while the corpus loads."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    reasons: list[str]
    if case == "identity":
        _readings(monkeypatch, {2: _facts(device_name="another device")})
        reasons = ["after the corpus load: the server's /readyz identity changed while the corpus "
                   "loaded: device_name"]  # fmt: skip
    elif case == "code":
        # A server restarted from another checkout's code: C7 on the stamped reading, and the
        # identity across the two readings.
        _readings(monkeypatch, {2: _code(ELSEWHERE_CODE)})
        reasons = [
            "after the corpus load: /readyz says the server imported verbatim from "
            "/elsewhere/src/verbatim",
            "the server's /readyz identity changed while the corpus loaded: code",
        ]
    elif case == "readyz-refused":
        # Identity unchanged: only the refusals of the stamped reading can see this.
        _readings(monkeypatch, {2: lambda r: dataclasses.replace(r, admission=None)})
        reasons = ["after the corpus load: /admission did not answer"]
    elif case == "tick-back":
        _readings(monkeypatch, {2: _facts(tick_id=0)})
        reasons = ["while the corpus loaded: it restarted"]
    elif case == "restarted-same-flags":

        def restart() -> None:
            rig.commands.listeners = rig.commands.on_card = (5151,)
            rig.proc.pid = 5151

        host.during_load.append(restart)
        reasons = ["the process listening on the port went from [4242] to [5151] while the corpus"]
    elif case == "restarted-ragged":
        host.during_load.append(lambda: setattr(rig.proc, "argv", fixed.cmdline(padding="ragged")))
        reasons = [
            "after the corpus load: the server was started with padding 'ragged'",
            "the server's command line changed while the corpus loaded",
        ]
    elif case == "environment":
        host.during_load.append(lambda: rig.proc.env.update(PYTHONUNBUFFERED="1"))
        reasons = ["the server's environment changed while the corpus loaded"]
    elif case == "new-banner":
        host.during_load.append(_new_banner(fixed, rig.log))
        reasons = ["holds 2 while the corpus loaded: a server started again"]
    elif case == "stopped":

        def stop() -> None:
            with rig.log.open("a", encoding="utf-8") as handle:
                handle.write("[verbatim] stopped\n")

        host.during_load.append(stop)
        reasons = [
            "after the corpus load: the server log's last server has stopped",
            "the server log's last banner changed while the corpus loaded",
        ]
    else:
        host.during_load.append(lambda: setattr(rig.commands, "on_card", (SERVER_PID, 777)))
        reasons = ["after the corpus load: card 3 is shared: pid(s) [777]"]
    _refused_without_a_session(
        fixed, out, lambda: _run(fixed, host, out, rig=rig), capsys, *reasons
    )


# --- the same server after the run: a failed run, written only as FAILED ---


def _after_the_run(monkeypatch: pytest.MonkeyPatch, action: Callable[[str], Any]) -> None:
    """Do ``action(endpoint)`` once every session has ended, before the after-run readings."""
    real = probe.capture

    async def capture(endpoint: str, *args: Any, **kwargs: Any) -> Any:
        run = await real(endpoint, *args, **kwargs)
        done = action(endpoint)
        if hasattr(done, "__await__"):
            await done
        return run

    monkeypatch.setattr(probe, "capture", capture)


async def _another_client(endpoint: str) -> None:
    """One more session on the same server, from someone else."""
    other = BY_ID["syn-a"]
    await bench_client.run_session(
        endpoint,
        session_id="another-client",
        utterance=Utterance("another", Path("another"), other.duration_s, ""),
        pcm=other.pcm,
        chunk=bench_client.ChunkMode.parse(CHUNK_MS),
        start_delay_s=0.0,
        words=True,
        lang="en-US",
        frame_ms=bench_constants.FRAME_MS,
        frame_seed=1,
    )


@pytest.mark.parametrize(
    "case",
    [
        "another-client",
        "revision",
        "card-uuid",
        "card-shared",
        "listener",
        "cmdline",
        "environment",
        "new-banner",
        "tick-back",
        "identity",
        "code",
        "not-ready",
    ],
)
def test_a_server_that_is_not_the_same_after_the_run_fails_the_run(
    fixed: Served,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    refs = next(host.hub.glob("models--*")) / "refs" / "main"
    actions: dict[str, tuple[Callable[[str], Any], str]] = {
        "another-client": (
            _another_client,
            "the server admitted 2 session(s) during the run, and 1 were sent",
        ),
        "revision": (
            lambda _: refs.write_text("f" * 40 + "\n", encoding="utf-8"),
            f"after the run: refs/main in {refs.parent.parent} is '{'f' * 40}', not the pin",
        ),
        "card-uuid": (
            lambda _: setattr(rig.commands, "uuid", "GPU-another"),
            f"after the run: nvidia-smi -i 3 is 'GPU-another', not '{CARD_UUID}'",
        ),
        "card-shared": (
            lambda _: setattr(rig.commands, "on_card", (SERVER_PID, 777)),
            "after the run: card 3 is shared: pid(s) [777]",
        ),
        "listener": (
            lambda _: setattr(rig.commands, "listeners", (5151,)),
            "the process listening on the port went from [4242] to [5151] during the run",
        ),
        "cmdline": (
            lambda _: setattr(rig.proc, "argv", fixed.cmdline(padding="ragged")),
            "the server's command line changed during the run",
        ),
        "environment": (
            lambda _: rig.proc.env.update(PYTHONUNBUFFERED="1"),
            "the server's environment changed during the run",
        ),
        "new-banner": (
            lambda _: _new_banner(fixed, rig.log)(),
            "holds 2 during the run: a server started again",
        ),
    }
    if case in actions:
        action, failure = actions[case]
        _after_the_run(monkeypatch, action)
    elif case == "tick-back":
        _readings(monkeypatch, {3: _facts(tick_id=1)})
        failure = "from the first reading to the end of the run: it restarted"
    elif case == "identity":
        _readings(monkeypatch, {3: _facts(nemo_version="another")})
        failure = "the server's /readyz identity changed during the run: nemo_version"
    elif case == "code":
        _readings(monkeypatch, {3: _code(ELSEWHERE_CODE)})
        failure = "the server's /readyz identity changed during the run: code"
    else:
        _readings(monkeypatch, {3: _facts(ready=False)})
        failure = "/readyz says the server is not ready at the end of the run"
    assert _run(fixed, host, out, rig=rig) == probe.EXIT_FAILED
    assert not out.exists()
    record = json.loads(probe.failed_path(out).read_text())
    assert record["success"] is False and record["missing"] == []  # every answer arrived
    assert [f for f in record["failures"] if failure in f], record["failures"]


# --- the client's own code changes during the run: a failed run, naming the start ---


@pytest.mark.parametrize(
    "change",
    ["commit", "tracked-files", "probe-file", "bench-client-file", "probe-file-before-the-run"],
)
def test_a_client_whose_code_changes_during_the_run_fails_the_run(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Other sessions edit this checkout and move its HEAD while a capture runs for hours. The
    provenance is read at the start (git before the first reading, the files as imported) and
    again at the end; here one of the four changes while the corpus loads, after the start, or
    (the last case) the probe file was edited after it was imported and before the run began,
    so what ran is the imported file, not the one on disk at the start. The run fails, the
    record names the start, and says what the end found."""
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    git = {"rev-parse": TEST_COMMIT, "status": ""}
    monkeypatch.setattr(probe, "_git", lambda *args: git[args[0]])
    # The files the end reading hashes: copies of the two that were imported, so one can be
    # edited here without touching the real ones.
    copies = {}
    for key, path in probe.code_files().items():
        copies[key] = tmp_path / "code" / key / path.name
        copies[key].parent.mkdir(parents=True)
        copies[key].write_bytes(path.read_bytes())
    monkeypatch.setattr(probe, "code_files", lambda: dict(copies))

    def edit(key: str) -> Callable[[], None]:
        return lambda: copies[key].write_bytes(copies[key].read_bytes() + b"# edited\n")

    key, action = {
        "commit": ("verbatim_commit", lambda: git.update({"rev-parse": "7" * 40})),
        "tracked-files": (
            "tracked_files_modified",
            lambda: git.update({"status": " M probes/server_frozen_answers.py"}),
        ),
        "probe-file": ("probe_sha256", edit("probe_sha256")),
        "bench-client-file": ("bench_client_sha256", edit("bench_client_sha256")),
        "probe-file-before-the-run": ("probe_sha256", lambda: None),
    }[change]
    host.during_load.append(action)
    start = {
        "verbatim_commit": TEST_COMMIT,
        "tracked_files_modified": False,
        "probe_sha256": hashlib.sha256(probe.HERE.read_bytes()).hexdigest(),
        "bench_client_sha256": hashlib.sha256(Path(bench_client.__file__).read_bytes()).hexdigest(),
    }
    assert {k: start[k] for k in probe.LOADED_SHA256} == probe.LOADED_SHA256
    if change == "probe-file-before-the-run":
        # The file that was imported, and so ran, hashed otherwise than the one now on disk.
        start["probe_sha256"] = "0" * 64
        monkeypatch.setitem(probe.LOADED_SHA256, "probe_sha256", start["probe_sha256"])
    assert _run(fixed, host, out) == probe.EXIT_FAILED
    assert not out.exists()
    record = json.loads(probe.failed_path(out).read_text())
    assert record["success"] is False and record["missing"] == []  # every answer arrived
    end = {
        **start,
        "verbatim_commit": git["rev-parse"],
        "tracked_files_modified": bool(git["status"]),
        **{k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in copies.items()},
    }
    assert [k for k in start if start[k] != end[k]] == [key]
    assert record["failures"] == [
        f"the client's code changed during the run: {key} {start[key]!r} at the start and "
        f"{end[key]!r} at the end; the record names the start, which may not be what ran"
    ]
    assert {k: record["client"][k] for k in probe.PROVENANCE} == start
    assert record["client"]["provenance_at_end"] == end


@pytest.mark.parametrize("key", ["verbatim_commit", "tracked_files_modified", "probe_sha256",
                                 "bench_client_sha256"])  # fmt: skip
def test_provenance_changes_names_each_value_that_moved(key: str) -> None:
    start = {"verbatim_commit": "a", "tracked_files_modified": False, "probe_sha256": "p",
             "bench_client_sha256": "b", "checkout": "/c"}  # fmt: skip
    assert probe.provenance_changes(start, dict(start)) == []
    end = {**start, key: "moved"}
    assert probe.provenance_changes(start, end) == [
        f"the client's code changed during the run: {key} {start[key]!r} at the start and "
        "'moved' at the end; the record names the start, which may not be what ran"
    ]


# --- three captures and the comparison: the frozen claim needs a control that differs ---


def _unstamped(path: Path) -> Path:
    """A copy of a capture without its ``fake_pipeline`` stamp: the shape a capture from a real
    server has, for the comparison to run on the probe's real output."""
    record = json.loads(path.read_text())
    assert record.pop("fake_pipeline") is True
    copy_path = path.with_name(f"{path.stem}.unstamped.json")
    copy_path.write_text(json.dumps(record), encoding="utf-8")
    return copy_path


def test_three_captures_do_not_claim_frozen_when_the_ragged_control_does_not_differ(
    fixed: Served, ragged: Served, tmp_path: Path
) -> None:
    """The fake's answers are a pure function of the audio, so fixed and ragged captures are
    identical here. The fixed pair matches at two concurrencies, and the comparison still must
    not say "frozen": the ragged control showed no difference, so the instrument's ability to
    see one is unshown."""
    compare = compare_captures
    host = _host(tmp_path, THREE)
    stamped = {}
    for name, served, padding, concurrency in (
        ("fixed-c3", fixed, "fixed", 3),
        ("fixed-c1", fixed, "fixed", 1),
        ("ragged-c3", ragged, "ragged", 3),
    ):
        stamped[name] = tmp_path / name / "capture.json"
        stamped[name].parent.mkdir()
        code = _run(served, host, stamped[name], padding=padding, concurrency=concurrency)
        assert code == probe.EXIT_OK, name
    ragged_record = json.loads(stamped["ragged-c3"].read_text())
    assert ragged_record["server"]["padding"]["observed_in_server_log"] == "ragged"
    assert ragged_record["server"]["padding"]["observed_in_cmdline"] == "ragged"

    # As taken, test doubles took them, and the comparator refuses every one.
    with pytest.raises(compare.Refused) as refused:
        compare.compare(
            compare.load_capture(stamped["fixed-c3"]),
            compare.load_capture(stamped["fixed-c1"]),
            control=compare.load_capture(stamped["ragged-c3"]),
        )
    assert len(refused.value.reasons) == 3
    assert all("fake_pipeline is True" in r for r in refused.value.reasons)
    paths = {name: _unstamped(path) for name, path in stamped.items()}

    report = compare.compare(
        compare.load_capture(paths["fixed-c3"]),
        compare.load_capture(paths["fixed-c1"]),
        control=compare.load_capture(paths["ragged-c3"]),
    )
    assert report["identity"]["identical"] == len(THREE)
    assert report["frozen"] is False
    assert report["verdict"].startswith("NOT SHOWN FROZEN")

    places_out = tmp_path / "places.json"
    report = compare.compare(
        compare.load_capture(paths["fixed-c3"]),
        compare.load_capture(paths["ragged-c3"]),
        repeat=compare.load_capture(paths["fixed-c1"]),
        places_out=places_out,
    )
    assert report["verdict"].startswith("POSITIVE CONTROL ABSENT")
    assert "NOT SHOWN FROZEN" in report["verdict"] and report["frozen"] is False
    places = json.loads(places_out.read_text())
    run = places["runs"]["bfloat16"]
    assert run["positive_control"].startswith("ABSENT")
    assert run["repeat_verdict"] == compare.SAME_SHAPE_VERDICT  # fixed at 3 and 1 matched
    arm = run["arms"][compare.ARM]
    assert arm["checked"] == len(THREE)
    assert arm["text_divergent"] == arm["timing_only_divergent"] == 0
    assert places["references"] == {r.rid: r.reference for r in THREE}
    assert places["att_context_size"] == [70, 1] and places["gpu_uuid"] == CARD_UUID
    assert "fake_pipeline" not in places
    trees = [
        json.loads(paths[name].read_text())["client"]["tracked_files_modified"]
        for name in ("fixed-c3", "ragged-c3", "fixed-c1")
    ]
    assert [
        places["captures"][k]["tracked_files_modified"]
        for k in ("a_fixed", "b_ragged", "repeat_fixed")
    ] == trees
    assert places["tracked_files_modified"] is (
        True if True in trees else None if None in trees else False
    )

    # The same fixed capture re-serialised at another stated concurrency, given as the repeat
    # after the ragged one: every pair is checked, so it is refused in this order too.
    copy_of_fixed = json.loads(paths["fixed-c3"].read_text())
    copy_of_fixed["client"]["concurrency"]["configured"] = 8
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(copy_of_fixed), encoding="utf-8")
    with pytest.raises(compare.Refused, match=r"b .* and repeat .* are the same capture"):
        compare.compare(
            compare.load_capture(paths["ragged-c3"]),
            compare.load_capture(paths["fixed-c3"]),
            repeat=compare.load_capture(edited),
        )


def test_a_capture_below_its_concurrency_records_the_peak_it_reached(
    fixed: Served, tmp_path: Path
) -> None:
    """Three recordings at --concurrency 8 never have more than three sessions in flight. The
    record says 3, not 8, and the comparator, which reads the observed peaks, does not take it
    and a capture at --concurrency 3 for two occupancies."""
    compare = compare_captures
    host = _host(tmp_path, THREE)
    paths = {}
    for concurrency in (8, 3):
        paths[concurrency] = tmp_path / f"c{concurrency}" / "capture.json"
        paths[concurrency].parent.mkdir()
        code = _run(fixed, host, paths[concurrency], concurrency=concurrency)
        assert code == probe.EXIT_OK, concurrency
        paths[concurrency] = _unstamped(paths[concurrency])
    below = json.loads(paths[8].read_text())["client"]["concurrency"]
    assert below["configured"] == 8 and below["observed_peak_in_flight"] == 3
    assert below["occupancy_at_first_audio"] == {"1": 1, "2": 1, "3": 1}
    assert below["sessions_timed"] == 3
    assert json.loads(paths[3].read_text())["client"]["concurrency"]["observed_peak_in_flight"] == 3

    report = compare.compare(compare.load_capture(paths[8]), compare.load_capture(paths[3]))
    assert report["identity"]["identical"] == len(THREE) and report["frozen"] is False
    assert report["verdict"].startswith(
        "IDENTICAL AT ONE CONCURRENCY (3): the fixed captures at concurrency 8 and 3 (observed "
        "peak in flight 3 and 3)"
    ), report["verdict"]


def test_the_peak_is_what_the_wire_carried_not_what_the_client_admitted(
    fixed: Served, tmp_path: Path
) -> None:
    """A server that holds each new session, after the handshake and before its session frame,
    until the session before it has ended. The client admits all three recordings at once
    (``--concurrency 3``, three recordings), and the wire never carries more than one: the
    record says 1, where a count of admissions, or min(concurrency, recordings), says 3. The
    comparator then takes it and a capture at --concurrency 1 for one occupancy, and it and a
    capture at --concurrency 3 on a server that does not hold sessions for two."""
    compare = compare_captures
    host = _host(tmp_path, THREE)
    paths = {}
    with _serving("fixed", pid=HELD_PID, hold=True) as held:
        paths["held-c3"] = tmp_path / "held-c3" / "capture.json"
        paths["held-c3"].parent.mkdir()
        assert _run(held, host, paths["held-c3"], concurrency=3) == probe.EXIT_OK
    for name, concurrency in (("fixed-c3", 3), ("fixed-c1", 1)):
        paths[name] = tmp_path / name / "capture.json"
        paths[name].parent.mkdir()
        assert _run(fixed, host, paths[name], concurrency=concurrency) == probe.EXIT_OK
    paths = {name: _unstamped(path) for name, path in paths.items()}

    record = json.loads(paths["held-c3"].read_text())
    concurrency = record["client"]["concurrency"]
    assert concurrency["configured"] == 3 and len(record["recordings"]) == 3
    assert concurrency["observed_peak_in_flight"] == 1
    assert concurrency["occupancy_at_first_audio"] == {"1": 3}
    spans = sorted(tuple(e["in_flight_s"]) for e in record["recordings"].values())
    assert all(end <= after for (_, end), (after, _) in itertools.pairwise(spans)), spans
    assert probe.peak_in_flight(spans) == 1
    unheld = json.loads(paths["fixed-c3"].read_text())["client"]["concurrency"]
    assert unheld["configured"] == 3 and unheld["observed_peak_in_flight"] == 3
    assert (
        json.loads(paths["held-c3"].read_text())["finals_digest"]
        == json.loads(paths["fixed-c3"].read_text())["finals_digest"]
    )  # the same answers: only the occupancy differs

    one = compare.compare(
        compare.load_capture(paths["held-c3"]), compare.load_capture(paths["fixed-c1"])
    )
    assert one["verdict"].startswith(
        "IDENTICAL AT ONE CONCURRENCY (1): the fixed captures at concurrency 3 and 1 (observed "
        "peak in flight 1 and 1)"
    ), one["verdict"]
    two = compare.compare(
        compare.load_capture(paths["fixed-c3"]), compare.load_capture(paths["held-c3"])
    )
    assert two["verdict"] == (
        "IDENTICAL, FROZEN NOT CLAIMED: the fixed captures at concurrency 3 and 3 (observed peak "
        "in flight 3 and 1) match on all 3 recordings, and no ragged control was given"
    )


def test_sessions_whose_first_audio_went_out_at_one_instant_are_all_counted_in_flight(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record's concurrency block, written from the sessions' intervals: here every session
    is given the same one (first audio frames at one instant, as a clock that ticks coarsely can
    stamp them). All three were in flight at once, so the peak is 3 and each was counted at 3;
    a peak read off the number of distinct occupancies would say 1. The comparator's recount of
    the written in_flight_s agrees with the stamp."""
    monkeypatch.setattr(probe, "in_flight", lambda tap, start: (0.25, 1.5))
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    assert _run(fixed, host, out, concurrency=3) == probe.EXIT_OK
    record = json.loads(out.read_text())
    concurrency = record["client"]["concurrency"]
    assert concurrency["observed_peak_in_flight"] == 3
    assert concurrency["occupancy_at_first_audio"] == {"3": 3}
    assert concurrency["sessions_timed"] == 3
    assert [e["in_flight_s"] for e in record["recordings"].values()] == [[0.25, 1.5]] * 3
    assert compare_captures.peak_problems(compare_captures.load_capture(_unstamped(out))) == []


def test_a_capture_with_word_confidence_on_carries_each_words_c(
    confident: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract C3 end to end: ``verbatim serve --word-confidence nemo-shipped``, its /readyz
    as the real server code reads it off the built pipeline, and every final word on the wire
    carrying its "c". The capture keeps each value, exactly, as the fourth element of its word,
    and the digest and the timing channel stay (word, start, end)."""
    host = _host(tmp_path, THREE)
    out = tmp_path / "answers.json"
    code = _run(confident, host, out, "--word-confidence", "nemo-shipped")
    assert code == probe.EXIT_OK
    record = json.loads(out.read_text())
    assert record["success"] is True and record["failures"] == []
    server = record["server"]
    assert server["word_confidence"] == {
        **server["word_confidence"],
        "reported_by_server": "nemo-shipped",
        "decoder_step_confidence": True,
        "observed_in_cmdline": "nemo-shipped",
        "observed_in_server_log": "nemo-shipped",
        "declared": "nemo-shipped",
    }
    assert server["att_context"]["observed_in_readyz"] == [70, 1]
    assert server["decoder_graphs"]["reported_by_server"] is False
    total = 0
    for rid, entry in record["recordings"].items():
        _, words = _expected_answer(BY_ID[rid].pcm)
        assert entry["words"] == [[*w, _confidence_of(w[0])] for w in words], rid
        assert entry["word_confidence_present"] == len(words)
        assert entry["unparsed_word_keys"] == []
        total += len(words)
    assert total > 0
    assert record["word_confidence"]["with_confidence"] == total
    assert record["word_confidence"]["word_entries"] == total
    expected = [FinalRecord(r.rid, *_answer_digest_parts(r)) for r in THREE]
    assert record["finals_digest"] == finals_digest(expected)

    # The same server with "c" taken off the wire: every recording fails, by name.
    monkeypatch.setattr(probe, "CONFIDENCE_KEY", "not-c")
    failed = tmp_path / "without-c.json"
    assert _run(confident, host, failed, "--word-confidence", "nemo-shipped") == probe.EXIT_FAILED
    missing = json.loads(probe.failed_path(failed).read_text())["missing"]
    assert [m["id"] for m in missing] == [r.rid for r in THREE]
    assert all(any("carry no confidence 'not-c'" in p for p in m["problems"]) for m in missing)


def _answer_digest_parts(recording: Any) -> tuple[str, tuple[tuple[Any, ...], ...]]:
    text, words = _expected_answer(recording.pcm)
    return text, tuple(tuple(w) for w in words)


def test_a_capture_taken_without_the_test_seams_is_not_stamped_fake(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp follows ``main``'s seams, as C1's follows the stock probe's backend: with none
    given, ``main`` reads the dataset, runs the host commands and reads /proc through its own
    functions (replaced here, beneath it), and the record carries no stamp."""
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    monkeypatch.setattr(probe, "load_librispeech_test_other", host.loader)
    monkeypatch.setattr(probe, "run_command", rig.commands)
    monkeypatch.setattr(probe, "read_proc", rig.proc)
    argv = _argv(fixed, host, out, rig)
    assert probe.main(argv, expected_split=host.split) == probe.EXIT_OK
    record = json.loads(out.read_text())
    assert "fake_pipeline" not in record
    assert record["not_a_row"] == "Exploratory capture through a running server; not a harness row."
    assert rig.commands.calls and rig.proc.calls  # the replaced functions were the ones used


#: ``main``'s test seams, and the module function each stands in for.
SEAMS = {
    "loader": "load_librispeech_test_other",
    "runner": "run_command",
    "proc": "read_proc",
}


@pytest.mark.parametrize("seam", sorted(SEAMS))
def test_any_one_test_seam_alone_stamps_the_capture_fake(
    fixed: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    """One seam given to ``main``, the other two replaced beneath it as in the test above: the
    stamp follows each seam on its own, not one of them in particular."""
    host = _host(tmp_path, ONE)
    out = tmp_path / "answers.json"
    rig = _rig(fixed, host, out)
    doubles = {"loader": host.loader, "runner": rig.commands, "proc": rig.proc}
    for name, function in SEAMS.items():
        if name != seam:
            monkeypatch.setattr(probe, function, doubles[name])
    argv = _argv(fixed, host, out, rig)
    assert probe.main(argv, expected_split=host.split, **{seam: doubles[seam]}) == probe.EXIT_OK
    record = json.loads(out.read_text())
    assert next(iter(record)) == "fake_pipeline" and record["fake_pipeline"] is True
    assert record["not_a_row"].startswith("FAKE: test doubles stood in")
    assert rig.commands.calls and rig.proc.calls


# --- unit tests: the client's own code ---


def test_the_hub_source_the_comparator_takes_is_the_one_the_probe_writes() -> None:
    assert probe.HUB_FROM_SERVER_ENVIRON == compare_captures.SERVER_HUB_SOURCE


def test_the_imported_files_are_the_ones_this_process_loaded() -> None:
    assert probe.IMPORTED == ("verbatim_bench.client", "verbatim")
    assert probe.imported_files() == {
        "verbatim_bench.client": str(Path(bench_client.__file__).resolve()),
        "verbatim": str(Path(sys.modules["verbatim"].__file__).resolve()),
    }


def test_a_module_not_imported_yet_is_found_where_an_import_would_load_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verbatim

    monkeypatch.delitem(sys.modules, "verbatim")
    assert probe.imported_files()["verbatim"] == str(Path(verbatim.__file__).resolve())
    monkeypatch.setattr(probe, "IMPORTED", ("verbatim_no_such_module_here",))
    assert probe.imported_files() == {"verbatim_no_such_module_here": None}


#: Where the two modules came from, per case: inside the probe's checkout or elsewhere.
_INSIDE = {"verbatim_bench.client": "<repo>/bench/c.py", "verbatim": "<repo>/src/v.py"}


@pytest.mark.parametrize(
    ("files", "status", "modified", "outside"),
    [
        (_INSIDE, "", False, []),
        (_INSIDE, " M probes/x.py", True, []),
        (
            {**_INSIDE, "verbatim_bench.client": "<elsewhere>/bench/c.py"},
            "",
            None,
            ["verbatim_bench.client"],
        ),
        ({**_INSIDE, "verbatim": "<elsewhere>/src/v.py"}, " M probes/x.py", None, ["verbatim"]),
        ({**_INSIDE, "verbatim": None}, "", None, ["verbatim"]),
        (_INSIDE, None, None, []),
    ],
    ids=["clean", "modified", "client-elsewhere", "verbatim-elsewhere", "not-found", "no-git"],
)
def test_the_tree_state_describes_the_code_that_ran_or_is_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str | None],
    status: str | None,
    modified: bool | None,
    outside: list[str],
) -> None:
    """``tracked_files_modified`` is git's word on the probe's own checkout. It says what ran
    only when the session client and ``verbatim`` were imported from inside that checkout;
    otherwise it is null, and the record names what came from elsewhere."""
    repo, elsewhere = tmp_path / "checkout", tmp_path / "elsewhere"
    placed = {
        name: None
        if path is None
        else path.replace("<repo>", str(repo)).replace("<elsewhere>", str(elsewhere))
        for name, path in files.items()
    }
    monkeypatch.setattr(probe, "REPO", repo)
    monkeypatch.setattr(probe, "imported_files", lambda: placed)
    answers = {"status": status, "rev-parse": "f" * 40}
    monkeypatch.setattr(probe, "_git", lambda *args: answers[args[0]])
    client = probe.client_provenance()
    assert client["tracked_files_modified"] is modified
    assert client["code_outside_checkout"] == outside
    assert client["imported_from"] == placed
    assert client["verbatim_commit"] == "f" * 40
    assert client["checkout"] == str(repo)


def test_the_client_provenance_reads_git_of_the_probes_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real git reader, on a repository made here: the commit it is at, a clean tree, a
    tracked file changed (an untracked one is not a change), and a directory that is no
    checkout, where both are null."""
    repo = tmp_path / "checkout"
    (repo / "src").mkdir(parents=True)
    tracked = repo / "src" / "v.py"
    tracked.write_text("a = 1\n", encoding="utf-8")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t.invalid"]
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "commit.gpgsign=false", "commit", "-qm", "x"],
    ):
        subprocess.run([*git, *args], check=True, capture_output=True, timeout=30)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    monkeypatch.setattr(probe, "_git", REAL_GIT)
    monkeypatch.setattr(probe, "REPO", repo)
    monkeypatch.setattr(
        probe,
        "imported_files",
        lambda: {"verbatim_bench.client": str(repo / "b.py"), "verbatim": str(tracked)},
    )
    client = probe.client_provenance()
    assert (client["verbatim_commit"], client["tracked_files_modified"]) == (head, False)
    (repo / "untracked.txt").write_text("new", encoding="utf-8")
    assert probe.client_provenance()["tracked_files_modified"] is False
    tracked.write_text("a = 2\n", encoding="utf-8")
    assert probe.client_provenance()["tracked_files_modified"] is True
    elsewhere = tmp_path / "not-a-checkout"
    elsewhere.mkdir()
    monkeypatch.setattr(probe, "REPO", elsewhere)
    monkeypatch.setattr(probe, "imported_files", lambda: {"verbatim": str(elsewhere / "v.py")})
    client = probe.client_provenance()
    assert (client["verbatim_commit"], client["tracked_files_modified"]) == (None, None)


# --- unit tests: occupancy on the wire ---


@pytest.mark.parametrize(
    ("intervals", "occupancy", "peak"),
    [
        ([(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)], [1, 2, 2], 2),
        ([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [1, 1, 1], 1),
        ([(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)], [3, 3, 3], 3),
        ([(2.0, 3.0), (0.0, 9.0), (1.0, 2.5)], [3, 1, 2], 3),
        ([(0.0, 1.0), None, (0.5, 0.75)], [1, 2], 2),
        ([], [], 0),
    ],
    ids=["overlap", "one-after-another", "all-at-once", "out-of-order", "untimed", "none"],
)
def test_the_peak_is_the_most_sessions_in_flight_at_one_instant(
    intervals: list[Any], occupancy: list[int], peak: int
) -> None:
    """A session is in flight from its first audio frame to its last final; one whose final
    arrived at the instant another's first audio frame went out is not counted with it. The
    comparator recounts a capture's peak by the same rule."""
    assert probe.occupancy_at_first_audio(intervals) == occupancy
    assert probe.peak_in_flight(intervals) == peak
    assert compare_captures.recounted_peak(intervals) == peak


def test_in_flight_runs_from_the_first_audio_frame_to_the_last_final() -> None:
    """To the last FINAL: not the first final, and not a frame of another kind after it."""
    frames = [
        _session(),
        _final("w1", [{"w": "w1", "s": 0, "e": 160}], 0.32),
        *GREEN[1:],
        _partial("", RECORDING.duration_s),
    ]
    tap = probe.Tap(
        received=[json.dumps(f) for f in frames],
        received_at=[8.125, 8.75, 8.875, 9.0, 9.5, 9.875],
        first_audio_at=8.25,
    )
    assert probe.in_flight(tap, 8.0) == (0.25, 1.5)
    assert probe.in_flight(dataclasses.replace(tap, first_audio_at=None), 8.0) is None
    no_final = [f for f in frames if f["type"] != "final"]
    tap = probe.Tap(
        received=[json.dumps(f) for f in no_final],
        received_at=[8.125, 8.875, 9.0, 9.875],
        first_audio_at=8.25,
    )
    assert probe.in_flight(tap, 8.0) is None


class _Socket:
    """A connection that hands out ``messages`` and keeps what is sent."""

    def __init__(self, messages: Sequence[str]) -> None:
        self.messages = list(messages)
        self.sent: list[Any] = []

    async def recv(self) -> str:
        return self.messages.pop(0)

    async def send(self, message: Any) -> None:
        self.sent.append(message)


def test_the_tap_stamps_the_first_audio_frame_and_each_text_frame() -> None:
    """On the monotonic clock, when the send or the receive returned."""
    tap = probe.Tap()
    socket = _Socket(['{"type": "session"}', '{"type": "final"}'])
    conn = probe._TappedConnection(socket, tap)

    async def run() -> None:
        await conn.send('{"type": "end"}')  # text: not audio
        assert tap.first_audio_at is None
        before = time.monotonic()
        await conn.send(b"\x00\x01")
        after = time.monotonic()
        assert tap.first_audio_at is not None and before <= tap.first_audio_at <= after
        first = tap.first_audio_at
        await asyncio.sleep(0.01)
        await conn.send(b"\x00\x01")
        assert tap.first_audio_at == first  # the first frame's moment, not the latest
        for _ in range(2):
            before = time.monotonic()
            await conn.recv()
            after = time.monotonic()
            assert before <= tap.received_at[-1] <= after
        assert len(tap.received_at) == len(tap.received) == 2

    asyncio.run(run())
    assert socket.sent == ['{"type": "end"}', b"\x00\x01", b"\x00\x01"]


# --- unit tests: the tolerance ---


TAIL = probe.Recording("tail", b"\x01\x00" * 8681, "")  # 0.5425625 s: seven decimals


def test_a_terminal_final_one_sample_short_is_not_terminal() -> None:
    short = TAIL.duration_s - 1 / probe.SAMPLE_RATE_HZ
    frames = [{"type": "final", "text": "", "audio_s": short}]
    assert probe.terminal_final_present(frames, TAIL.duration_s) is False


def test_a_terminal_final_rounded_to_six_decimals_on_the_wire_is_terminal() -> None:
    assert round(TAIL.duration_s, 6) != TAIL.duration_s  # the rounding is really exercised
    frames = [{"type": "final", "text": "", "audio_s": round(TAIL.duration_s, 6)}]
    assert probe.terminal_final_present(frames, TAIL.duration_s) is True


# --- unit tests: judge, flipping one thing per case ---


RECORDING = BY_ID["syn-f"]
TEXT, WORDS = _expected_answer(RECORDING.pcm)


def _session(chunk_ms: int = CHUNK_MS) -> dict[str, Any]:
    return {"type": "session", "id": "x", "chunk_ms": chunk_ms, "invariance_class": "u"}


def _final(text: str = TEXT, words: Any = None, audio_s: float | None = None, **extra: Any) -> dict:
    wire = [{"w": w, "s": s, "e": e} for w, s, e in WORDS] if words is None else words
    audio = RECORDING.duration_s if audio_s is None else audio_s
    return {"type": "final", "text": text, "words": wire, "audio_s": audio, **extra}


def _partial(text: str, audio_s: float) -> dict[str, Any]:
    return {"type": "partial", "text": text, "audio_s": audio_s}


def _outcome(
    frames: list[dict[str, Any]],
    *,
    finals: int | None = None,
    text: str | None = None,
    words: list[tuple[str, int, int]] | None = None,
    error: str | None = None,
    connected: bool = True,
    sent: int | None = None,
    end_sent: bool = True,
    close_code: int = 1000,
) -> Any:
    wire_finals = [f for f in frames if f.get("type") == "final"]
    tap = probe.Tap(
        connected=connected,
        received=[json.dumps(f) for f in frames],
        binary_bytes_sent=len(RECORDING.pcm) if sent is None else sent,
        end_sent=end_sent,
        close_code=close_code,
    )
    result = SessionResult(session_id="s", stream_id=RECORDING.rid, server_session_id="x")
    result.finals_received = len(wire_finals) if finals is None else finals
    result.final_text = bench_client.join_final_texts(wire_finals) if text is None else text
    result.words = bench_client.join_final_words(wire_finals) if words is None else words
    result.error = error
    return probe.Outcome(0, RECORDING, result, tap)


GREEN = [_session(), _partial(TEXT.split()[0], 0.16), _partial(TEXT, 0.32), _final()]


def test_the_green_outcome_is_green() -> None:
    entry = probe.judge(_outcome(GREEN), chunk_ms=CHUNK_MS)
    assert entry["problems"] == [] and entry["terminal_final"] is True
    assert entry["words"] == WORDS
    assert probe.run_failures([entry], expected=1) == []


def test_the_in_flight_stamps_are_written_unrounded() -> None:
    """The comparator recounts the peak from these stamps, and its recount agrees with the
    probe's only on the same numbers: a rounded stamp can turn two sessions that did not
    overlap into two that did."""
    outcome = _outcome(GREEN)
    outcome.in_flight_s = (0.1234567890123, 2.9876543210987)
    assert probe.judge(outcome, chunk_ms=CHUNK_MS)["in_flight_s"] == [
        0.1234567890123,
        2.9876543210987,
    ]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"frames": [_session(560), *GREEN[1:]]}, "the session frame says chunk_ms 560"),
        ({"frames": GREEN[1:]}, "no session frame"),
        (
            {"frames": [*GREEN, {"type": "error", "code": "X", "message": "m"}]},
            "server error frame X",
        ),
        ({"close_code": 1011}, "the socket closed with code 1011"),
        (
            {
                "frames": [*GREEN[:3], {k: v for k, v in _final().items() if k != "words"}],
                "words": [],
            },
            "final 0 carried no words list",
        ),
        (
            {"frames": [*GREEN[:3], _final(words=[])], "words": []},
            f"final 0 carried text {TEXT!r} but no word timings",
        ),
        (
            {"frames": [*GREEN[:3], _final(audio_s=RECORDING.duration_s - 0.16)]},
            "no terminal final",
        ),
        ({"frames": [*GREEN[:3]], "text": "", "words": []}, "no final"),
        ({"error": "TimeoutError: boom"}, "client error: TimeoutError: boom"),
        ({"connected": False}, "the tap never saw"),
        ({"sent": len(RECORDING.pcm) - 2}, f"{len(RECORDING.pcm) - 2} of"),
        ({"end_sent": False}, "the end message never went on the wire"),
        ({"finals": 2}, "wiring: the tap saw 1 final(s), run_session counted 2"),
        ({"text": "not what the wire said"}, "wiring: the tap's joined final text"),
        ({"words": [("x", 0, 1)]}, "wiring: the tap's word timings"),
        (
            {"frames": [*GREEN[:3], _final(words=[{"w": "x", "s": 0.0, "e": 1}])], "words": []},
            "1 word entr(ies) on the wire were not integer-millisecond",
        ),
        (
            {"frames": [GREEN[0], _partial("w1", 0.16), _partial("", 0.16), *GREEN[2:]]},
            "lost mid-recording final: the partial text dropped to empty at audio_s [0.16]",
        ),
        ({"frames": [GREEN[0], GREEN[3]]}, "no partial arrived"),
        (
            {
                "frames": [
                    *GREEN[:3],
                    _final(words=[{"w": w, "s": s, "e": e, "c": 0.9} for w, s, e in WORDS]),
                ]
            },
            f"{len(WORDS)} word(s) carry a confidence 'c' and the run declares word confidence off",
        ),
    ],
    ids=[
        "session-chunk",
        "no-session",
        "error-frame",
        "close-code",
        "no-words-list",
        "text-without-timings",
        "terminal-short",
        "no-final",
        "client-error",
        "not-tapped",
        "bytes-short",
        "no-end",
        "wiring-count",
        "wiring-text",
        "wiring-words",
        "float-word",
        "partial-reset",
        "no-partials",
        "confidence-when-off",
    ],
)
def test_judge_names_each_problem(change: dict[str, Any], problem: str) -> None:
    change = dict(change)
    frames = change.pop("frames", GREEN)
    entry = probe.judge(_outcome(frames, **change), chunk_ms=CHUNK_MS)
    assert [p for p in entry["problems"] if p.startswith(problem)], entry["problems"]


def test_a_partial_reset_beside_a_final_at_the_same_audio_is_a_mid_recording_final() -> None:
    frames = [
        _session(),
        _partial("w1", 0.16),
        _final("w1", [{"w": "w1", "s": 0, "e": 160}], 0.32),
        _partial("", 0.32),
        _partial("w2", 0.48),
        _final("w2", [{"w": "w2", "s": 320, "e": 480}]),
    ]
    assert probe.unexplained_resets(frames) == []
    without = [f for f in frames if not (f["type"] == "final" and f["audio_s"] == 0.32)]
    assert probe.unexplained_resets(without) == [0.32]


def _with_c(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"w": w, "s": s, "e": e, "c": c} for (w, s, e), c in zip(WORDS, values, strict=True)]


def test_per_word_confidence_c_is_the_fourth_element_when_the_run_declares_it_on() -> None:
    """Contract C3: a word object gains "c" only on a server with word confidence on. The frames
    are constructed here, to reach cases a server does not send (a final-level confidence, a
    field under another name); ``test_a_capture_with_word_confidence_on_carries_each_words_c``
    takes them from a served --word-confidence nemo-shipped. The values need every digit a
    double carries, so a rounded or truncated fourth element is not the value on the wire."""
    values = [0.123456789012345, 1 / 3]  # syn-f has two words
    frames = [*GREEN[:3], _final(words=_with_c(values), confidence=0.7)]
    entry = probe.judge(_outcome(frames), chunk_ms=CHUNK_MS, word_confidence_mode="paper-best")
    assert entry["problems"] == []
    assert entry["words"] == [[*w, c] for w, c in zip(WORDS, values, strict=True)]
    assert [w[3] for w in entry["words"]] == values  # exactly, not approximately
    # ... and exactly again after the record's own JSON round trip.
    assert json.loads(json.dumps(entry["words"])) == entry["words"]
    assert entry["word_confidence_present"] == len(WORDS)
    assert entry["unparsed_word_keys"] == []
    assert entry["final_frames"][0]["confidence_present"] is True
    assert entry["final_frames"][0]["confidence"] == 0.7
    assert entry["final_frames"][0]["unparsed_keys"] == ["confidence"]

    plain = probe.judge(_outcome(GREEN), chunk_ms=CHUNK_MS)
    assert plain["words"] == WORDS and plain["word_confidence_present"] == 0
    assert plain["final_frames"][0]["confidence_present"] is False

    # A field under another name is not a confidence, and is still seen.
    odd = [{"w": w, "s": s, "e": e, "p": 0.9} for w, s, e in WORDS]
    seen = probe.judge(_outcome([*GREEN[:3], _final(words=odd)]), chunk_ms=CHUNK_MS)
    assert seen["words"] == WORDS and seen["unparsed_word_keys"] == ["p"]
    assert seen["problems"] == []


_W0, _W1 = ({"w": w, "s": s, "e": e} for w, s, e in WORDS)


@pytest.mark.parametrize(
    ("wire", "problem"),
    [
        (
            [{**_W0, "c": 0.5}, _W1],
            f"1 of {len(WORDS)} word(s) carry no confidence 'c' and the run declares word "
            "confidence paper-best",
        ),
        ([{**_W0, "c": "high"}, {**_W1, "c": 0.5}], "1 word confidence(s) on the wire are not"),
        # The server sends null for a non-finite confidence: present, and not a number.
        ([{**_W0, "c": None}, {**_W1, "c": None}], "2 word confidence(s) on the wire are not"),
    ],
    ids=["one-missing", "not-a-number", "null"],
)
def test_a_confidence_the_run_declares_on_must_be_a_number_on_every_word(
    wire: list[dict[str, Any]], problem: str
) -> None:
    frames = [*GREEN[:3], _final(words=wire)]
    entry = probe.judge(_outcome(frames), chunk_ms=CHUNK_MS, word_confidence_mode="paper-best")
    assert [p for p in entry["problems"] if p.startswith(problem)], entry["problems"]
    assert all(len(w) == 3 or isinstance(w[3], float) for w in entry["words"])


# --- unit tests: the run ---


def _entry(rid: str, text: str = "w", problems: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "id": rid,
        "n": 0,
        "text": text,
        "words": [["w", 0, 1]] if text else [],
        "problems": list(problems),
    }


@pytest.mark.parametrize(
    ("entries", "expected", "failure"),
    [
        ([_entry("a", ""), _entry("b", "")], 2, "vacuous"),
        ([_entry("a"), _entry("a")], 2, "recording ids are not unique"),
        ([_entry("a")], 2, "1 recordings were captured, 2 were expected"),
        ([], None, "no recording was captured"),
        ([_entry("a"), _entry("b", problems=["x"])], 2, "1 of 2 recording(s) failed; first b"),
    ],
    ids=["vacuous", "duplicate-ids", "count", "none", "a-bad-recording"],
)
def test_run_failures_names_each_failure(
    entries: list[dict[str, Any]], expected: int | None, failure: str
) -> None:
    assert [f for f in probe.run_failures(entries, expected=expected) if f.startswith(failure)]
    assert probe.run_failures([_entry("a"), _entry("b", "")], expected=2) == []


# --- unit tests: refusals from the server's own report ---


FACTS = ServerFacts(
    ready=True,
    model=probe.DEFAULT_MODEL,
    pipeline=probe.DEFAULT_PIPELINE,
    chunk_ms=CHUNK_MS,
    precision="bfloat16",
    execution="eager",
    tick_id=100,
    biasing=False,
    nemo_version="n",
    torch_version="t",
    device_name="d",
)
ADMISSION = {
    "bucket": BUCKET, "admitted_total": 10, "refused_total": 0, "p95_tick_ms": 1.0,
    "degradation_level": 0, "consecutive_overruns": 0, "live": 0,
}  # fmt: skip
METRICS = {"verbatim_ticks_over_budget_total": 0.0, "verbatim_ticks_total": 100.0}
#: What the unit-test readings say the server imported (contract C7), and what DECLARED expects:
#: placeholders, not directories on any machine.
SERVED_FROM = "/checkout/src/verbatim"
READYZ_CODE = {"verbatim_path": SERVED_FROM, "bench_path": "/checkout/bench/src/verbatim_bench"}
DECLARED = probe.Declared(
    model=probe.DEFAULT_MODEL,
    pipeline=probe.DEFAULT_PIPELINE,
    chunk_ms=CHUNK_MS,
    dtype="bfloat16",
    execution="eager",
    bucket=BUCKET,
    concurrency=BUCKET,
    padding="fixed",
    word_confidence="off",
    decoder_graphs=False,
    att_context=[70, 1],
    server_verbatim_path=SERVED_FROM,
)


def _reading(
    facts: Any = FACTS,
    admission: Any = ADMISSION,
    metrics: Any = METRICS,
    c3: Any = READYZ_C3,
    code: Any = READYZ_CODE,
) -> Any:
    """One reading; ``code`` None is a /readyz without a ``code`` object."""
    readyz = (
        None
        if facts is None
        else {
            **facts.to_json_dict(),
            **copy.deepcopy(c3 or {}),
            **({} if code is None else {"code": copy.deepcopy(code)}),
        }
    )
    return probe.ServerReading(
        facts, None if admission is None else dict(admission), metrics, readyz
    )


def _replace(facts: ServerFacts, **change: Any) -> ServerFacts:
    return ServerFacts(**{**facts.to_json_dict(), **change})


def _observed(**change: Any) -> dict[str, Any]:
    return {**READYZ_C3, "observed": {**READYZ_C3["observed"], **change}}


def test_the_green_reading_is_not_refused() -> None:
    assert probe.refusals(_reading(), DECLARED) == []
    assert probe.readyz_warnings(_reading()) == []


@pytest.mark.parametrize(
    ("reading", "declared", "refusal"),
    [
        (_reading(facts=None), {}, "/readyz did not answer"),
        (_reading(_replace(FACTS, ready=False)), {}, "/readyz says the server is not ready"),
        (
            _reading(_replace(FACTS, pipeline="cache_aware_ctc")),
            {},
            "the server reports pipeline 'cache_aware_ctc'",
        ),
        (
            _reading(_replace(FACTS, execution="graph path")),
            {},
            "the server reports execution 'graph path'",
        ),
        (_reading(_replace(FACTS, execution=None)), {}, "/readyz does not report execution"),
        (_reading(_replace(FACTS, model=None)), {}, "/readyz does not report model"),
        (_reading(_replace(FACTS, biasing=None)), {}, "/readyz does not report biasing"),
        (_reading(_replace(FACTS, biasing=True)), {}, "the server reports biasing on"),
        (_reading(_replace(FACTS, model="x/y")), {}, "the server reports model 'x/y'"),
        (_reading(_replace(FACTS, precision="float32")), {}, "--dtype 'bfloat16'"),
        (_reading(_replace(FACTS, chunk_ms=560)), {}, "the run is configured for 160 ms"),
        (_reading(), {"concurrency": BUCKET + 1}, f"concurrency {BUCKET + 1} exceeds the bucket"),
        (_reading(admission={**ADMISSION, "bucket": 16}), {}, "the server reports bucket 16"),
        (_reading(admission=None), {}, "/admission did not answer"),
        (
            _reading(admission={k: v for k, v in ADMISSION.items() if k != "p95_tick_ms"}),
            {},
            "/admission does not report p95_tick_ms",
        ),
        (_reading(metrics=None), {}, "/metrics did not answer"),
        (
            _reading(metrics={"verbatim_ticks_total": 1.0}),
            {},
            "/metrics has no verbatim_ticks_over_budget_total",
        ),
        (
            _reading(c3={**READYZ_C3, "word_confidence": "paper-best"}),
            {},
            "/readyz reports word confidence 'paper-best', the run declares 'off'",
        ),
        (
            _reading(c3=_observed(decoder_step_confidence=True)),
            {},
            "/readyz observes decoder_step_confidence True",
        ),
        (
            _reading(
                c3={**_observed(decoder_step_confidence=True), "word_confidence": "paper-best"}
            ),
            {"word_confidence": "paper-best"},
            None,
        ),
        (
            _reading(),
            {"word_confidence": "nemo-shipped"},
            "/readyz observes decoder_step_confidence False on the built decoder, and the run "
            "declares word confidence 'nemo-shipped'",
        ),
        (_reading(c3=_observed(decoder_graphs=True)), {}, "/readyz observes decoder_graphs True"),
        (
            _reading(c3=_observed(att_context_size=[70, 13])),
            {},
            "/readyz observes att_context_size [70, 13]",
        ),
        (_reading(), {"att_context": None}, "no attention context is declared"),
        (_reading(code=None), {}, "/readyz does not report code.verbatim_path (contract C7)"),
        (
            _reading(code={"bench_path": None}),
            {},
            "/readyz does not report code.verbatim_path (contract C7)",
        ),
        (
            _reading(code={**READYZ_CODE, "verbatim_path": 7}),
            {},
            "/readyz does not report code.verbatim_path (contract C7)",
        ),
        (
            _reading(code={**READYZ_CODE, "verbatim_path": "/elsewhere/src/verbatim"}),
            {},
            "/readyz says the server imported verbatim from /elsewhere/src/verbatim, and the run "
            f"expects {SERVED_FROM}",
        ),
        (
            _reading(),
            {"server_verbatim_path": "/elsewhere/src/verbatim"},
            f"/readyz says the server imported verbatim from {SERVED_FROM}, and the run expects "
            "/elsewhere/src/verbatim",
        ),
        (_reading(code={**READYZ_CODE, "bench_path": None}), {}, None),
    ],
    ids=[
        "no-readyz",
        "not-ready",
        "pipeline",
        "execution",
        "execution-unreported",
        "model-unreported",
        "biasing-unreported",
        "biasing-on",
        "model",
        "dtype",
        "chunk",
        "concurrency-over-bucket",
        "bucket",
        "no-admission",
        "admission-key",
        "no-metrics",
        "metrics-series",
        "c3-word-confidence",
        "c3-step-confidence-when-off",
        "c3-step-confidence-when-on-is-green",
        "c3-no-step-confidence-when-on",
        "c3-decoder-graphs",
        "c3-att-context",
        "no-declared-att-context",
        "c7-no-code",
        "c7-no-verbatim-path",
        "c7-verbatim-path-not-a-path",
        "c7-another-checkout",
        "c7-another-expected",
        "c7-no-bench-is-green",
    ],
)
def test_refusals_names_each_reason(
    reading: Any, declared: dict[str, Any], refusal: str | None
) -> None:
    found = probe.refusals(reading, dataclasses.replace(DECLARED, **declared))
    if refusal is None:
        assert found == [], found
    else:
        assert [f for f in found if f.startswith(refusal)], found


def test_a_reading_without_the_c3_fields_is_refused_for_the_attention_context_alone() -> None:
    """The observed attention context is required (the comparator refuses a capture without
    it); every other C3 field the server does not report is a warning."""
    bare = _reading(c3=None)
    found = probe.refusals(bare, DECLARED)
    assert len(found) == 1 and found[0].startswith(NO_ATT_CONTEXT), found
    assert probe.readyz_warnings(bare) == [
        f"/readyz does not report {name} (contract C3): it is stamped as NOT observed"
        for name in ("word_confidence", "decoder_step_confidence", "decoder_graphs")
    ]
    only_att = _reading(c3={"observed": {"att_context_size": [70, 1]}})
    assert probe.refusals(only_att, DECLARED) == []
    assert len(probe.readyz_warnings(only_att)) == 3


def test_read_server_reads_the_c3_fields_off_the_readyz_body(
    fixed: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    reading = probe.read_server(fixed.endpoint)
    assert probe.reported_c3(reading) == {
        "word_confidence": "off",
        "att_context_size": [70, 1],
        "decoder_step_confidence": False,
        "decoder_graphs": False,
    }
    assert reading.facts is not None and reading.facts.biasing is False
    monkeypatch.setitem(_READYZ, "c3", None)
    assert set(probe.reported_c3(probe.read_server(fixed.endpoint)).values()) == {None}


def test_a_server_that_is_not_ready_is_read_as_not_ready_rather_than_as_silent() -> None:
    """``/readyz`` answers 503, with the reason, while the server is not ready: the body is read
    and the refusal names the reason, instead of "did not answer"."""
    import http.server

    body = {**FACTS.to_json_dict(), **READYZ_C3, "ready": False, "reason": "no tick completed yet"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status = 503 if self.path == "/readyz" else 404
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode() if status == 503 else b"{}")

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        reading = probe.read_server(f"ws://127.0.0.1:{server.server_address[1]}/v1/stream")
    finally:
        server.shutdown()
        server.server_close()
    assert reading.facts is not None and reading.facts.ready is False
    assert reading.admission is None  # a 404 body is not read as an answer
    assert probe.refusals(reading, DECLARED)[0] == (
        "/readyz says the server is not ready (no tick completed yet)"
    )


# --- unit tests: the server either side of the run ---


def _after(admitted: int = 3, refused: int = 0, over: float = 0.0, **facts: Any) -> Any:
    return _reading(
        _replace(FACTS, **{"tick_id": 200, **facts}),
        {**ADMISSION, "admitted_total": 10 + admitted, "refused_total": refused},
        {**METRICS, "verbatim_ticks_over_budget_total": over, "verbatim_ticks_total": 300.0},
    )


def test_a_clean_run_has_no_server_failures() -> None:
    first = _reading(_replace(FACTS, tick_id=50))
    assert probe.server_failures(_reading(), _after(), recordings=3, first=first) == ([], [])


@pytest.mark.parametrize(
    ("after", "failure"),
    [
        (_after(admitted=4), "the server admitted 4 session(s) during the run, and 3"),
        (_after(admitted=2), "the server admitted 2 session(s) during the run, and 3"),
        (_after(refused=1), "the server refused 1 session(s) during the run"),
        (_after(tick_id=5), "the server's tick count went 100 -> 100 -> 5"),
        (_after(tick_id=100), "the server's tick count went 100 -> 100 -> 100"),
        (_after(ready=False), "/readyz says the server is not ready at the end of the run"),
        (
            _reading(facts=None, admission={**ADMISSION, "admitted_total": 13}),
            "/readyz stopped answering",
        ),
        (_reading(admission=None), "the admitted count could not be read"),
        (
            _reading(metrics=None, admission={**ADMISSION, "admitted_total": 13}),
            "ticks over budget could not be read",
        ),
    ],
    ids=[
        "admitted-one-over",
        "admitted-one-under",
        "refused",
        "restart-ticks",
        "no-tick",
        "not-ready",
        "readyz-gone",
        "admission-gone",
        "metrics-gone",
    ],
)
def test_server_failures_names_each_failure(after: Any, failure: str) -> None:
    failures, _ = probe.server_failures(_reading(), after, recordings=3)
    assert [f for f in failures if f.startswith(failure)], failures


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model", "x/y"),
        ("pipeline", "cache_aware_ctc"),
        ("chunk_ms", 560),
        ("precision", "float32"),
        ("execution", "graph path"),
        ("biasing", True),
        ("nemo_version", "other"),
        ("torch_version", "other"),
        ("device_name", "other"),
        ("word_confidence", "paper-best"),
        ("att_context_size", [70, 13]),
        ("decoder_step_confidence", True),
        ("decoder_graphs", True),
        ("code", {**READYZ_CODE, "verbatim_path": "/elsewhere/src/verbatim"}),
        ("code", {**READYZ_CODE, "bench_path": None}),
    ],
)
def test_every_identity_field_is_compared_across_the_run(key: str, value: Any) -> None:
    if key in {f.name for f in dataclasses.fields(ServerFacts)}:
        after = _after(**{key: value})
    elif key in ("word_confidence", "code"):
        after = dataclasses.replace(_after(), readyz={**_after().readyz, key: value})
    else:
        readyz = _after().readyz
        after = dataclasses.replace(
            _after(), readyz={**readyz, "observed": {**readyz["observed"], key: value}}
        )
    failures, _ = probe.server_failures(_reading(), after, recordings=3)
    assert failures == [f"the server's /readyz identity changed during the run: {key}"]
    stamped = probe.continuity_refusals(_reading(), after, "while the corpus loaded")
    assert stamped == [f"the server's /readyz identity changed while the corpus loaded: {key}"]


def test_the_tick_count_must_not_go_back_from_the_first_reading() -> None:
    first = _reading(_replace(FACTS, tick_id=150))
    failures, _ = probe.server_failures(_reading(), _after(), recordings=3, first=first)
    assert failures == [
        "the server's tick count went 150 -> 100 -> 200 from the first reading to the end of the "
        "run: it restarted"
    ]
    assert probe.continuity_refusals(first, _reading(), "while the corpus loaded") == [
        "the server's tick count went from 150 to 100 while the corpus loaded: it restarted"
    ]


def test_ticks_over_budget_are_a_warning_not_a_failure() -> None:
    failures, warnings = probe.server_failures(_reading(), _after(over=4.0), recordings=3)
    assert failures == []
    assert warnings and warnings[0].startswith("4 tick(s) went over budget")


def test_parse_metrics_reads_counters_and_skips_quantiles() -> None:
    text = (
        "# HELP verbatim_ticks_total Ticks completed.\n"
        '# TYPE verbatim_ticks_total counter\nverbatim_ticks_total{execution="graph path"} 12\n'
        'verbatim_ticks_over_budget_total{model="m"} 3\n'
        'verbatim_tick_cost_ms{quantile="0.5"} 9.5\n'
    )
    assert probe.parse_metrics(text) == {
        "verbatim_ticks_total": 12.0,
        "verbatim_ticks_over_budget_total": 3.0,
    }


# --- unit tests: the corpus ---


def _meta(split: int | None = 3) -> dict[str, Any]:
    return {} if split is None else {"split_size_observed": split}


REFS = [(r.rid, r.reference) for r in THREE]


def test_the_matching_corpus_is_not_refused() -> None:
    found = probe.corpus_refusals(THREE, _meta(), expected=3, expected_split=3, references=REFS)
    assert found == []
    limited = probe.corpus_refusals(
        THREE[:2], _meta(), expected=2, expected_split=3, references=REFS
    )
    assert limited == []


@pytest.mark.parametrize(
    ("recordings", "meta", "references", "refusal"),
    [
        (THREE, _meta(None), REFS, "the loader did not report the size"),
        (THREE, _meta(4), REFS, "the split holds 4 recordings, expected 3"),
        (THREE[:2], _meta(), REFS, "the corpus yielded 2 recordings, expected 3"),
        ([THREE[0], THREE[0], THREE[2]], _meta(), REFS, "the corpus has duplicate recording ids"),
        (
            THREE,
            _meta(),
            [REFS[0], (REFS[1][0], "other"), REFS[2]],
            "1 reference(s) differ from the stock record's; first syn-c",
        ),
        (THREE, _meta(), [REFS[1], REFS[0], REFS[2]], "2 recording(s) out of the stock record's"),
        (THREE, _meta(), REFS[:2], "the references record holds 2 references, expected 3"),
    ],
    ids=[
        "no-split-size",
        "split-size",
        "count",
        "duplicate-ids",
        "reference-text",
        "order",
        "record-short",
    ],
)
def test_corpus_refusals_names_each_reason(
    recordings: list[Any], meta: dict[str, Any], references: list[Any], refusal: str
) -> None:
    found = probe.corpus_refusals(
        recordings, meta, expected=3, expected_split=3, references=references
    )
    assert [f for f in found if f.startswith(refusal)], found


def test_the_references_record_is_read_again_while_it_is_being_replaced(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text('{"references": {"a": "x"', encoding="utf-8")  # caught mid-write
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        path.write_text(json.dumps({"references": {"a": "x", "b": "y"}}), encoding="utf-8")

    pairs, digest = probe.load_references(path, attempts=3, wait_s=0.25, sleep=sleep)
    assert pairs == [("a", "x"), ("b", "y")] and slept == [0.25]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON after 3 attempts"):
        probe.load_references(path, attempts=3, wait_s=0.0, sleep=lambda _: None)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'references'"):
        probe.load_references(path, attempts=3, wait_s=0.0, sleep=lambda _: None)


# --- unit tests: the host ---


def test_the_revision_is_read_from_the_cache_and_refused_unless_it_is_only_the_pin(
    tmp_path: Path,
) -> None:
    good = probe.observe_revision(_hub(tmp_path / "good"), probe.DEFAULT_MODEL)
    assert good["refs_main"] == PIN and good["snapshots"] == [PIN]
    assert probe.revision_refusals(good, PIN) == []
    assert probe.revision_refusals(good, PIN[:8])  # a prefix is not the pin

    two = probe.observe_revision(
        _hub(tmp_path / "two", snapshots=(PIN, "a" * 40)), probe.DEFAULT_MODEL
    )
    assert [r for r in probe.revision_refusals(two, PIN) if r.startswith("2 snapshots")]
    moved = probe.observe_revision(
        _hub(tmp_path / "moved", refs="a" * 40, snapshots=("a" * 40,)), probe.DEFAULT_MODEL
    )
    found = probe.revision_refusals(moved, PIN)
    assert [r for r in found if r.startswith("refs/main")]
    assert [r for r in found if r.startswith("the only snapshot is")]
    missing = probe.observe_revision(tmp_path / "nothing", probe.DEFAULT_MODEL)
    assert probe.revision_refusals(missing, PIN)[0].startswith("the model revision could not")


def test_the_card_guard_on_its_own() -> None:
    def observe(**change: Any) -> dict[str, Any]:
        return probe.observe_gpu(FakeHostCommands(8765, **change), index=3, port=8765)

    name = TEST_DOUBLE_RUNTIME.device_name
    green = observe()
    assert green["uuid"] == CARD_UUID and green["listener_pids"] == [SERVER_PID]
    assert green["compute_pids"] == [SERVER_PID]
    assert probe.gpu_refusals(green, expected_uuid=CARD_UUID, device_name=name) == []
    for change, refusal in (
        ({"uuid": "GPU-x"}, "nvidia-smi -i 3 is 'GPU-x'"),
        ({"name": "other"}, "nvidia-smi -i 3 names 'other'"),
        ({"on_card": (777,)}, "the process listening on port 8765 (pid 4242) holds no context"),
        ({"on_card": (777, SERVER_PID)}, "card 3 is shared: pid(s) [777]"),
        ({"on_card": ()}, "the process listening on port 8765 (pid 4242) holds no context"),
        ({"listeners": ()}, "0 process(es) listen on port 8765"),
        ({"listeners": (1, 2)}, "2 process(es) listen on port 8765"),
    ):
        found = probe.gpu_refusals(observe(**change), expected_uuid=CARD_UUID, device_name=name)
        assert [f for f in found if f.startswith(refusal)], (change, found)
    # The device /readyz names is compared even when it names none.
    assert probe.gpu_refusals(green, expected_uuid=CARD_UUID, device_name=None)

    def broken(argv: Sequence[str]) -> str:
        raise FileNotFoundError("nvidia-smi")

    failed = probe.observe_gpu(broken, index=3, port=8765)
    assert probe.gpu_refusals(failed, expected_uuid=CARD_UUID, device_name=name)[0].startswith(
        "the card could not be observed"
    )


#: A command line as the runbook starts the server (step1_gate_runbook.sh build_serve).
RUNBOOK_ARGV = [
    "python3", "verbatim", "serve", probe.DEFAULT_MODEL,
    "--chunk", "160ms", "--bucket", "32", "--padding", "fixed", "--eager",
    "--compute-dtype", "bfloat16", "--att-context-left", "70", "--pipeline", "cache_aware_rnnt",
    "--host", "127.0.0.1", "--ws-port", "8080", "--grpc-port", "0", "--device-id", "0",
    "--idle-timeout", "30.0", "--ring-seconds", "3.0", "--stop-history-eou-ms", "800",
    "--language-code", "en-US", "--traceback",
]  # fmt: skip


def test_the_server_command_line_is_read_by_the_servers_own_parser() -> None:
    flags = probe.parse_serve_argv(RUNBOOK_ARGV)
    assert flags == {
        "model": probe.DEFAULT_MODEL,
        "chunk_ms": 160,
        "bucket": 32,
        "ceiling": None,
        "padding": "fixed",
        "eager": True,
        "biasing": False,
        "decoder_graphs": False,
        "word_confidence": "off",
        "att_context_left": 70,
        "att_context": [70, 1],
        "compute_dtype": "bfloat16",
        "pipeline": "cache_aware_rnnt",
        "device_id": 0,
        "host": "127.0.0.1",
        "ws_port": 8080,
    }
    as_module = ["python3", "-m", "verbatim.cli", *RUNBOOK_ARGV[2:]]
    assert probe.parse_serve_argv(as_module) == flags
    with pytest.raises(ValueError, match="not a `verbatim serve` command line"):
        probe.parse_serve_argv(["python3", "-m", "http.server", "8080"])
    with pytest.raises(ValueError, match="serve parser rejects"):
        probe.parse_serve_argv([*RUNBOOK_ARGV, "--padding", "sideways"])


def _obs(argv: Sequence[str] = RUNBOOK_ARGV, **env: str | None) -> dict[str, Any]:
    base = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "3", "HOME": "/h", **env}
    proc = FakeProc(argv, {k: v for k, v in base.items() if v is not None})
    return probe.observe_process(proc, SERVER_PID)


RUNBOOK_DECLARED = dataclasses.replace(DECLARED, bucket=32, concurrency=32)


def _process_refusals(obs: Mapping[str, Any], **kw: Any) -> list[str]:
    return probe.process_refusals(
        obs,
        kw.pop("declared", RUNBOOK_DECLARED),
        gpu_index=3,
        gpu_uuid=CARD_UUID,
        port=8080,
        hub_cache=kw.pop("hub_cache", None),
    )


def test_the_process_guard_on_its_own() -> None:
    green = _obs()
    assert (
        green["environ"]["CUDA_VISIBLE_DEVICES"] == "3"
        and green["hub_dir"] == "/h/.cache/huggingface/hub"
    )
    assert _process_refusals(green) == []
    assert _process_refusals(_obs(CUDA_VISIBLE_DEVICES=CARD_UUID)) == []  # a UUID names it too
    assert _process_refusals(_obs(CUDA_VISIBLE_DEVICES="1,3"), declared=RUNBOOK_DECLARED) != []
    second = [*RUNBOOK_ARGV, "--device-id", "1"]  # the last --device-id wins, as in argparse
    assert _process_refusals(_obs(second, CUDA_VISIBLE_DEVICES="1,3")) == []
    cases: list[tuple[dict[str, Any], str]] = [
        ({"declared": dataclasses.replace(RUNBOOK_DECLARED, model="x/y")}, "with model"),
        ({"declared": dataclasses.replace(RUNBOOK_DECLARED, chunk_ms=560)}, "with chunk ms 160"),
        ({"declared": dataclasses.replace(RUNBOOK_DECLARED, bucket=8)}, "with bucket 32"),
        ({"declared": dataclasses.replace(RUNBOOK_DECLARED, padding="ragged")}, "with padding"),
        ({"declared": dataclasses.replace(RUNBOOK_DECLARED, dtype="float32")}, "compute dtype"),
        (
            {"declared": dataclasses.replace(RUNBOOK_DECLARED, pipeline="cache_aware_ctc")},
            "with pipeline",
        ),
        (
            {"declared": dataclasses.replace(RUNBOOK_DECLARED, execution="graph path")},
            "with execution 'eager'",
        ),
        (
            {"declared": dataclasses.replace(RUNBOOK_DECLARED, word_confidence="paper-best")},
            "with word confidence 'off'",
        ),
        (
            {"declared": dataclasses.replace(RUNBOOK_DECLARED, decoder_graphs=True)},
            "with decoder graphs False",
        ),
        (
            {"declared": dataclasses.replace(RUNBOOK_DECLARED, att_context=[70, 13])},
            "with attention context [70, 1]",
        ),
        ({"hub_cache": Path("/elsewhere/hub")}, "the server resolves its Hugging Face hub cache"),
    ]
    for kw, refusal in cases:
        found = _process_refusals(green, **kw)
        assert [f for f in found if refusal in f], (kw, found)
    biased = _obs([*RUNBOOK_ARGV, "--biasing"])
    assert [f for f in _process_refusals(biased) if "with biasing True" in f]
    port = _obs([*RUNBOOK_ARGV, "--ws-port", "9090"])
    assert [f for f in _process_refusals(port) if "listens on port 9090" in f]
    for env, refusal in (
        ({"CUDA_DEVICE_ORDER": None}, "CUDA_DEVICE_ORDER is None"),
        ({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, "CUDA_DEVICE_ORDER is 'FASTEST_FIRST'"),
        ({"CUDA_VISIBLE_DEVICES": "0"}, "is card '0' by its CUDA_VISIBLE_DEVICES='0'"),
        ({"CUDA_VISIBLE_DEVICES": ""}, "is card '' by its CUDA_VISIBLE_DEVICES=''"),
        ({"CUDA_VISIBLE_DEVICES": None}, "is card '0' by its CUDA_VISIBLE_DEVICES=None"),
        ({"HOME": None}, "the server's hub cache cannot be resolved"),
    ):
        found = _process_refusals(_obs(**env))
        assert [f for f in found if refusal in f], (env, found)
    # Unreadable: nothing is refused on what was not seen; process_warnings says it loudly.
    blind = probe.observe_process(
        FakeProc(RUNBOOK_ARGV, {}, unreadable=("cmdline", "environ")), 4242
    )
    assert _process_refusals(blind) == []
    said = probe.process_warnings(blind, hub_source="somewhere")
    assert [w for w in said if w.startswith("the server's command line could not be read")]
    assert [w for w in said if w.startswith("the server's environment could not be read")]
    assert probe.process_warnings(green, hub_source="x") == []
    nobody = probe.observe_process(FakeProc(RUNBOOK_ARGV, {}), None)
    assert nobody["cmdline_error"] == "no single process listens on the endpoint's port"


@pytest.mark.parametrize(
    "env",
    [
        {"HF_HUB_CACHE": "$ROOT/explicit"},
        {"HUGGINGFACE_HUB_CACHE": "~/legacy"},
        {"HF_HOME": "${ROOT}/home"},
        {"XDG_CACHE_HOME": "/xdg"},
        {},
    ],
    ids=["hf-hub-cache", "legacy", "hf-home", "xdg", "default"],
)
def test_the_servers_hub_cache_is_resolved_as_huggingface_hub_resolves_it(
    tmp_path: Path, env: dict[str, str]
) -> None:
    """Checked against huggingface_hub itself, in a process given exactly that environment."""
    full = {"HOME": str(tmp_path / "home-dir"), "ROOT": str(tmp_path), **env}
    ours = probe.hub_dir_from_environ(full)
    theirs = subprocess.run(
        [
            sys.executable,
            "-c",
            "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)",
        ],
        env={**full, "PATH": os.environ.get("PATH", ""), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    assert str(ours) == theirs


# --- unit tests: the server's banner ---


def _banner(served: Served, old: str | None = None, new: str = "") -> str:
    text = served.stdout.getvalue()
    if old is None:
        return text
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _banner_refusals(text: str, served: Served, **declared: Any) -> list[str]:
    return probe.banner_refusals(
        probe.banner_facts(text),
        endpoint=served.endpoint,
        declared=dataclasses.replace(DECLARED, **declared),
    )


def test_the_banner_green(fixed: Served) -> None:
    assert _banner_refusals(_banner(fixed), fixed) == []


@pytest.mark.parametrize(
    ("old", "new", "refusal"),
    [
        (
            "(att_context_size [70, 1])",
            "(att_context_size [70, 13])",
            "the banner names att_context [70, 13], the run declares [70, 1]",
        ),
        (
            f"checkpoint   {probe.DEFAULT_MODEL}",
            "checkpoint   some/other-model",
            "the banner names checkpoint 'some/other-model'",
        ),
        ("chunk mode   160 ms", "chunk mode   560 ms", "the banner names 560 ms chunks, not 160"),
        (
            f"admission    bucket {BUCKET} streams",
            "admission    bucket 16 streams",
            "the banner names bucket 16, not 8",
        ),
        ("[verbatim] ready\n", "[verbatim] ready\n[verbatim] stopped\n", "the server log's last"),
        (
            "[verbatim] ready\n",
            "[verbatim] ready\n[verbatim] biasing      ON: sessions may carry phrase lists\n",
            "the server's banner says biasing ON",
        ),
        (
            "[verbatim] ready\n",
            "[verbatim] ready\n[verbatim] decoder      CUDA graphs ON for the RNNT decoder.\n",
            "the server's banner says decoder graphs True, the run declares False",
        ),
    ],
    ids=["att-context", "checkpoint", "chunk", "bucket", "stopped", "biasing", "decoder-graphs"],
)
def test_the_banner_names_each_disagreement(
    fixed: Served, old: str, new: str, refusal: str
) -> None:
    found = _banner_refusals(_banner(fixed, old, new), fixed)
    assert [f for f in found if f.startswith(refusal)], found


@pytest.mark.parametrize(
    "line",
    [
        "[verbatim] checkpoint",
        "[verbatim] chunk mode",
        "[verbatim] admission",
        "[verbatim] websocket",
    ],
)
def test_an_incomplete_banner_says_nothing_about_padding(fixed: Served, line: str) -> None:
    """A banner missing any line it must have is not complete, and a missing RAGGED line in it
    is not evidence of fixed padding."""
    kept = [x for x in _banner(fixed).splitlines() if not x.startswith(line)]
    if line == "[verbatim] checkpoint":
        # Without its first line there is no banner at all; keep a stub start so the rest is read.
        kept.insert(0, "[verbatim] checkpoint ")
    facts = probe.banner_facts("\n".join(kept))
    assert facts["padding"] is None and facts["word_confidence"] is None
    assert _banner_refusals("\n".join(kept), fixed)[0].startswith(
        "the server log holds no complete startup banner"
    )


def test_the_banner_parser_reads_word_confidence_and_decoder_graphs(fixed: Served) -> None:
    text = fixed.stdout.getvalue()
    plain = probe.banner_facts(text)
    assert plain["word_confidence"] == "off" and plain["decoder_graphs"] is False
    assert plain["biasing"] is False
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("[verbatim] chunk mode"))
    lines[at + 1 : at + 1] = [
        "[verbatim] confidence   word confidence ON, paper-best: tsallis exp, min over a word.",
        "[verbatim] decoder      CUDA graphs ON for the RNNT decoder. A SEPARATE ARM: the path",
    ]
    on = probe.banner_facts("\n".join(lines))
    assert on["word_confidence"] == "paper-best" and on["decoder_graphs"] is True
    found = probe.banner_refusals(on, endpoint=fixed.endpoint, declared=DECLARED)
    assert found == [
        "the server's banner says word confidence paper-best, the run declares off",
        "the server's banner says decoder graphs True, the run declares False",
    ]


@pytest.mark.parametrize("order", ["fixed-then-ragged", "ragged-then-fixed"])
def test_a_log_appended_across_a_restart_is_read_from_its_last_banner(
    fixed: Served, ragged: Served, order: str
) -> None:
    """One log appended across a restart on the same port: a server killed after 'ready' with
    no 'stopped' line, then another started. Only the last banner is the server now running,
    and when ``/proc/<pid>/cmdline`` cannot be read it is the only source of the padding."""
    assert "[verbatim] stopped" not in fixed.stdout.getvalue()  # killed, not stopped
    assert ragged.endpoint in ragged.stdout.getvalue()
    # The restarted server listens on the port the endpoint names.
    texts = {
        "fixed": fixed.stdout.getvalue(),
        "ragged": ragged.stdout.getvalue().replace(ragged.endpoint, fixed.endpoint),
    }
    first, last = order.split("-then-")
    facts = probe.banner_facts(texts[first] + texts[last])
    assert facts["banners"] == 2
    assert facts["padding"] == last
    found = probe.banner_refusals(facts, endpoint=fixed.endpoint, declared=DECLARED)
    if last == "fixed":
        assert found == []
    else:
        assert found == ["the server's banner says padding ragged, the run declares fixed"]


def test_the_banner_parser_reads_padding_and_att_context(
    fixed: Served, ragged: Served, biased: Served
) -> None:
    for served, padding in ((fixed, "fixed"), (ragged, "ragged")):
        facts = probe.banner_facts(served.stdout.getvalue())
        assert facts["padding"] == padding
        assert facts["att_context"] == [70, 1]
        assert facts["bucket"] == BUCKET and facts["chunk_ms"] == CHUNK_MS
        assert facts["checkpoint"] == probe.DEFAULT_MODEL
        assert facts["banners"] == 1 and facts["biasing"] is False
        assert urllib.parse.urlparse(facts["websocket"]).port == served.port
    assert probe.banner_facts(biased.stdout.getvalue())["biasing"] is True
    # A banner cut off before 'ready' says nothing about padding.
    cut = fixed.stdout.getvalue().split("[verbatim] ready")[0]
    assert probe.banner_facts(cut)["padding"] is None
    assert probe.banner_refusals(
        probe.banner_facts(cut), endpoint=fixed.endpoint, declared=DECLARED
    )[0].startswith("the server log holds no complete startup banner")
