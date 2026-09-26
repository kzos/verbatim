"""Capture the SERVER's own answers for every LibriSpeech test-other recording.

The stock probes (``stock_divergence.py`` and the ones before it) measured where stock NeMo is
unstable. Nothing scored what a running Verbatim server itself answers. This client asks it:
every recording of ``openslr/librispeech_asr``, config ``other``, split ``test``, in the order
``stock_divergence.py`` reads them, streamed through a RUNNING server, and for each recording id
it keeps the final transcript, the word timings, and any per-word confidence exactly as the wire
carried them.

ONE CAPTURE IS NOT EVIDENCE THAT THE ANSWERS ARE FROZEN. One digest cannot tell invariant answers
from batch-dependent ones. The plan this client serves takes three captures of one server
setting (160 ms chunks, att_context [70, 1], bfloat16, the bucket chosen for the card):

    --padding fixed  --concurrency 32     the served answers
    --padding ragged --concurrency 32     the DR-0014 control arm: the places, and the proof
                                          that the instrument can see a batch effect
    --padding fixed  --concurrency 8      the served answers again at another occupancy

``scripts/compare_captures.py`` compares them. "Frozen" is claimed there, and only when the two
fixed captures match on both channels AND their clients observed a different peak number of
sessions in flight on the wire (the record's ``client.concurrency.observed_peak_in_flight``,
below) AND the ragged capture differs AND every capture's padding was observed AND none of them
comes from a test double or, unless ``--allow-modified-tree`` is given there, from a modified
tree. The places are where the fixed and ragged captures differ, both taken at the serving
setting, so a place and the answer scored at it come from one setting.

OCCUPANCY IS OBSERVED ON THE WIRE, not read off the configuration. ``--concurrency`` is how many
sessions the client ADMITS at once (``asyncio.Semaphore``), which says nothing about how many the
server had: a server that holds a session before acknowledging it, or recordings too short to
overlap, leave fewer in flight. The tap stamps, on the client's monotonic clock, the moment each
session's first audio frame went on the wire and the moment its last final arrived; a session is
in flight between the two, and ``observed_peak_in_flight`` is the most sessions in flight at one
instant (``peak_in_flight``). Every recording keeps its own two stamps (``in_flight_s``, seconds
from the start of the run, unrounded), and ``compare_captures.py`` recounts the peak from them and
refuses a capture whose stamp is not that count.

It does not start the server and it touches no GPU. Start the server separately, on the named
card and on no other, from the checkout whose code is to be captured. The reference for the
command line and the environment is ``scripts/step1_gate_runbook.sh`` (``build_serve``); by hand,
from a checkout ``<checkout>``, it is

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \\
    PYTHONPATH=<checkout>/src:<checkout>/bench/src \\
    verbatim serve nvidia/nemotron-speech-streaming-en-0.6b --chunk 160ms --bucket <bucket> \\
        --padding fixed --compute-dtype bfloat16 --eager --att-context-left 70 --device-id 0 \\
        --host 127.0.0.1 --ws-port 8080 > server.log 2>&1

``CUDA_DEVICE_ORDER=PCI_BUS_ID`` makes ``3`` the card ``nvidia-smi`` calls 3, and
``CUDA_VISIBLE_DEVICES=3`` keeps the server off every other card: without them it starts on
cuda:0, which is not this run's card. ``PYTHONPATH`` puts the checkout's own ``verbatim`` first:
without it an editable install of another checkout in the same environment answers, and this
client refuses a server whose ``/readyz`` names another ``verbatim`` (THE SERVER'S CODE, below).
Then, on the same host and as the same user (so ``ss`` names the server's PID and
``/proc/<pid>/environ`` is readable),

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<checkout>/src:<checkout>/bench/src \\
    python probes/server_frozen_answers.py \\
        --endpoint ws://127.0.0.1:8080/v1/stream --bucket <bucket> --padding fixed \\
        --concurrency 32 --server-log server.log \\
        --gpu-index 3 --gpu-uuid <uuid: nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader> \\
        --references-record <probe-output>/step1/nemotron-bf16-1120.json \\
        --out <probe-output>/step1/server-fixed-c32.json

``--bucket``, ``--padding``, ``--concurrency``, ``--server-log``, ``--gpu-index`` and
``--gpu-uuid`` have no defaults: the bucket is chosen per card from a step profile, the
concurrency (the max level) has to fit under it, and a capture that cannot see the server's
banner and card is not taken. ``--references-record`` may instead come from the environment
variable ``VERBATIM_STEP1_STOCK_RECORD``.

THE SERVER PATH. Every recording is one session through ``verbatim_bench.client.run_session``,
the session client the invariance gate drives (bench/src/verbatim_bench/invariance.py:783-796),
called with the gate's arguments: words on, canonical 20 ms framing with seeded jitter
(``constants.FRAME_MS``), no phrase list, and admission by ``asyncio.Semaphore(concurrency)`` as
in invariance.py:766-770. ``run_level`` itself is not called because it keeps only text and words
and files a session with no error and no final as an empty transcript (invariance.py:804-811).

PACING IS REAL TIME, because the harness has no faster mode (client.py:252, 367, 463-466;
pace.py:3). The batch-invariance evidence (docs/decisions/0014) was taken on paced arrivals.

A LOST FINAL IS A FAILURE, and it is not read off ``SessionResult.error``, which is set only for a
stream that produced no final at all (client.py:436, 482, 505). The connection is tapped: every
text frame the client received, every byte it sent and the close code are recorded beside it.
Three rules, each a pure function of the tapped frames:

* the LAST final carries an ``audio_s`` equal to the audio sent (the terminal final arrived);
* every drop of the partial text from non-empty to empty coincides with a final at the same
  ``audio_s``. A mid-recording final is always followed by an empty partial at its own
  ``audio_s`` (src/verbatim/protocols/emit.py:52-58), and a step result that carries a final is
  never dropped from the backlog (src/verbatim/engine.py:70-99), so a reset with no final beside
  it is a final that was lost on the way;
* a recording with audio received at least one partial. The rule above reads the partials, and
  they arrive only while interim results are on (the server's default, frames.py:207); without
  any, a lost mid-recording final could not be seen, so their absence is a failure too.

The tap forwards every call unchanged. It is compared with what ``run_session`` returned, but
that is a check of the WIRING, not a second reading: both see the same ``recv`` calls and go
through the same join functions, so they disagree only when the tap is bypassed or mis-wired.
The tap's value is that it keeps what ``run_session`` discards: ``audio_s``, the partials, and
any field of a final or a word that the client does not parse.

OBSERVED, NOT DECLARED, wherever the server or the host can be asked:

* ``/readyz``: model, pipeline, chunk mode, precision, execution and biasing (which must be
  reported, and off), and, where the server reports them, the configured ``word_confidence`` and
  ``observed``: ``att_context_size`` read from the built encoder, ``decoder_step_confidence`` and
  ``decoder_graphs``. A declaration any of them contradicts is refused. A server that does not
  report ``observed.att_context_size`` is refused before a byte is sent, since
  ``compare_captures.py`` refuses every capture without it; any other C3 field it does not
  report is stamped null with a warning. ``code`` (contract C7): which ``verbatim`` the server
  imported, refused unless it is the checkout's (THE SERVER'S CODE, below). ``/admission``: the
  bucket.
* the server process: ``ss`` names the one PID listening on the endpoint's port. Its command
  line, ``/proc/<pid>/cmdline`` parsed by verbatim's own ``serve`` parser, must name the declared
  model, chunk, bucket, padding, execution, dtype, pipeline, word confidence, decoder graphs and
  attention context, and no biasing. Its environment, ``/proc/<pid>/environ``, must put its
  ``--device-id`` on ``--gpu-index`` (``CUDA_DEVICE_ORDER=PCI_BUS_ID`` and
  ``CUDA_VISIBLE_DEVICES``), and gives the Hugging Face hub cache the server resolves, in
  huggingface_hub's order of variables (``hub_dir_from_environ``; unlike huggingface_hub, it
  takes a variable set empty as unset). When either file cannot be read, a loud warning says so
  and something else stands in: the server log's banner for the flags, and for the cache
  ``--hf-hub-cache`` or, without it, this client's own cache; the record names which.
* the card: ``nvidia-smi -i <index>`` must name ``--gpu-uuid`` and the device ``/readyz`` names,
  and the server's PID must be the ONLY process holding a context on that card.
* the server's startup banner (``--server-log``): checkpoint, chunk, attention context, bucket,
  port, padding, word confidence, decoder graphs and biasing; its last server must not have
  stopped.
* the model revision: ``refs/main`` and ``snapshots/`` of the server's hub cache, refused unless
  ``refs/main`` is the pin and it is the only snapshot. ``hub_dir_source`` says where the cache
  came from; ``compare_captures.py`` takes the revision as observed only when it is
  ``HUB_FROM_SERVER_ENVIRON``, the server's own environment.
* the corpus: every id and reference is checked, in order, against the step-1 stock record's
  ``references`` (read with retry, since that file is replaced while its run is going).
* the load: ``p95_tick_ms``, ``degradation_level`` and ``consecutive_overruns`` from
  ``/admission`` and ``verbatim_ticks_over_budget_total`` from ``/metrics``, before and after.
  The admitted count must rise by exactly the recordings sent and the refused count by none,
  which also catches another client on the same server.

THREE READINGS of all of it: before the corpus load, after it (the reading the run is stamped
with) and after the run. Each must pass the same checks, and each later one must show the SAME
server: the same ``/readyz`` identity, a ``tick_id`` never below the first reading's, the same
listening PID, the same command line and environment, the same last banner and no new one. A
server restarted on the same port while the corpus loads, even with every flag the same, is
refused before a byte is sent.

Matmul precision is not reported by the server. It is stamped as declared beside the default in
the installed ``NeMoPipelineSpec`` (the CLI passes none), and a declaration that contradicts the
installed default is refused.

Output: one JSON file. On success it is written to ``--out`` and the exit code is 0. If any
check fails after sending, NOTHING is written to ``--out``: the record goes to
``<out stem>.FAILED<suffix>`` with ``"success": false`` and the reasons, and the exit code is 1.
A refusal before sending (exit 2) writes nothing. A record is written once: a run whose
``--out`` or FAILED path already exists is refused before sending, and a file that appears at
the path during the run is not written over (the record stays beside it, exit 1).

A TEST DOUBLE IS STAMPED, as ``probes/stock_divergence.py`` stamps its CPU fake: when ``main`` is
given any of its test seams (``loader``, ``runner``, ``proc``), the record's first key is
``"fake_pipeline": true`` and its ``not_a_row`` says so, whatever the doubles are. The only
unstamped path is the one that reads the dataset, runs ``nvidia-smi`` and ``ss`` and reads
``/proc`` itself. ``scripts/compare_captures.py`` refuses a stamped capture, and
``scripts/step1_places.py`` refuses a stamped record.

THE SERVER'S CODE (contract C7). ``/readyz`` reports ``code``: the directory the server process
imported ``verbatim`` from (``verbatim_path``) and ``verbatim_bench`` (``bench_path``, null where
it cannot). A server that does not report ``code.verbatim_path``, or reports another than
``<this file's checkout>/src/verbatim`` (resolved), or, with ``--server-checkout DIR``,
``DIR/src/verbatim`` (resolved), is refused before a byte is sent. ``code`` is part of the
server's identity, so a server that answers from other code after the load or after the run is
refused or fails the run. The record keeps it (``server.code``), and
``scripts/compare_captures.py`` refuses two captures whose servers ran different code.

THE CLIENT'S CODE. ``client.verbatim_commit`` and ``tracked_files_modified`` are the git state of
the checkout this file lives in. The code that ran is also ``verbatim_bench`` (the session
client) and ``verbatim`` (the serve parser, the attention context), found on ``sys.path``; the
record names the file each came from (``client.imported_from``). When either lies outside this
file's checkout (``client.code_outside_checkout``), that checkout's state does not describe what
ran, and ``tracked_files_modified`` is null. ``probe_sha256`` and ``bench_client_sha256`` hash
this file and the session client's file as they were when this module was imported, the code
that runs. All four (``PROVENANCE``) are the START of the run: git is read before the first
reading of the server, and the files were hashed at import. They are read again after the run
(``client.provenance_at_end``), and a run in which any of them changed FAILS: other sessions
edit and switch checkouts, and a record must not name code that did not run.

Exploratory. NOT a harness row.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import websockets
from verbatim_bench import client as bench_client
from verbatim_bench import constants
from verbatim_bench.canonical import FinalRecord, finals_digest
from verbatim_bench.client import (
    ChunkMode,
    SessionResult,
    join_final_texts,
    join_final_words,
    run_session,
)
from verbatim_bench.corpus import Utterance
from verbatim_bench.invariance import DEFAULT_SEED
from verbatim_bench.results import percentile
from verbatim_bench.serverfacts import (
    ArmContradiction,
    ServerFacts,
    check_arm,
    health_url,
    read_server_facts,
)
from verbatim_bench.wer import corpus_wer

SAMPLE_RATE_HZ = 16000
DATASET = "openslr/librispeech_asr"
DATASET_CONFIG = "other"
DATASET_SPLIT = "test"
#: The test-other count every stock record was taken over (stock_divergence.py N_TARGETS default).
EXPECTED_RECORDINGS = 2939
DEFAULT_ENDPOINT = "ws://127.0.0.1:8080/v1/stream"
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
#: The full Hugging Face commit step 1 is pinned to; compared exactly, never as a prefix.
DEFAULT_REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
DEFAULT_CHUNK_MS = 160
DEFAULT_DTYPE = "bfloat16"
DEFAULT_PIPELINE = "cache_aware_rnnt"
#: The execution mode of the stock and published records the captures are read beside.
DEFAULT_EXECUTION = "eager"
EXECUTIONS = ("eager", "graph path")
DEFAULT_MATMUL = "high"
#: ``verbatim serve --word-confidence``; any value but off is a different decode path.
DEFAULT_WORD_CONFIDENCE = "off"
WORD_CONFIDENCE_MODES = ("off", "nemo-shipped", "paper-best")
#: The environment variable that names the step-1 stock record when --references-record does not.
REFERENCES_ENV = "VERBATIM_STEP1_STOCK_RECORD"

#: A final's ``audio_s`` is the valid samples consumed, rounded to 6 decimals on the wire
#: (src/verbatim/protocols/ws/frames.py FinalFrame). Half a sample plus that rounding: one missing
#: sample is caught, and a missing 160 ms chunk is a thousand times past it.
AUDIO_TOLERANCE_S = 0.5 / SAMPLE_RATE_HZ + 0.5e-6

#: The keys of a word entry the client parses (client.py join_final_words).
WORD_KEYS = ("w", "s", "e")
#: A word's confidence on the WebSocket wire: present only when the server runs with
#: --word-confidence other than off. Every other unparsed key is listed, so a differently named
#: field is still seen.
CONFIDENCE_KEY = "c"
#: The keys of a final frame the client parses.
FINAL_KEYS = ("type", "text", "words", "audio_s")

#: The /metrics series stamped before and after a run.
METRIC_SERIES = (
    "verbatim_ticks_over_budget_total",
    "verbatim_ticks_total",
    "verbatim_ticks_late_total",
    "verbatim_sessions_admitted_total",
    "verbatim_sessions_refused_total",
    "verbatim_degradation_level",
    "verbatim_tick_p95_ms",
)

#: The keys of the server's environment the record keeps: what places it on a card, where it
#: reads the model from, and the path it imports code from (what ``/readyz`` ``code`` resolves;
#: kept as evidence beside it). The rest is digested, never written: it may hold credentials.
ENV_KEYS = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "HOME",
    "HF_HUB_OFFLINE",
    "PYTHONPATH",
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

#: /5: the server's code is observed (``server.code``, contract C7) and the client's provenance
#: is the start of the run, checked again at its end. /4: ``observed_peak_in_flight`` is counted
#: from the wire (in /3 it was the client's own admission count), each recording keeps its
#: ``in_flight_s``, and a test double is stamped.
RECORD = "vb-server-frozen-answers/5"
#: What every record says of the frozen claim: scripts/compare_captures.py's FROZEN_RULE,
#: word for word (a test holds the two together).
FROZEN_CLAIM = (
    "one digest cannot tell invariant answers from batch-dependent ones. "
    "scripts/compare_captures.py claims it only for two fixed captures whose clients observed a "
    "different peak number of sessions in flight on the wire, with identical answers on both "
    "channels, AND a ragged capture of the same setting that differs, AND the padding of all "
    "three observed, AND none of them from a test double, AND none taken from a modified tree "
    "unless --allow-modified-tree is given"
)
HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
#: How ``observe_host`` names a hub cache read from the server process's own environment:
#: scripts/compare_captures.py's SERVER_HUB_SOURCE, word for word (a test holds the two together);
#: the one source it takes a revision from as an observation of the server.
HUB_FROM_SERVER_ENVIRON = (
    "the server's environment (/proc/<pid>/environ), by huggingface_hub's rule"
)
#: The modules besides this file whose code ran: the session client (its package is the one
#: every other ``verbatim_bench`` import here comes from) and ``verbatim``.
IMPORTED = ("verbatim_bench.client", "verbatim")
#: How ``observe_host`` names the hub cache it reads when the server's environment could not be
#: read and ``--hf-hub-cache`` was not given: this client's own, which the server may not read.
HUB_FROM_THIS_CLIENT = (
    "this client's own huggingface_hub cache (the server's environment was not read)"
)
#: The keys of ``/readyz``'s ``code`` object (contract C7).
CODE_KEYS = ("verbatim_path", "bench_path")
#: The client provenance read at the start of the run and again at its end; a run in which any
#: of them changed fails.
PROVENANCE = ("verbatim_commit", "tracked_files_modified", "probe_sha256", "bench_client_sha256")


# --- the corpus ---


@dataclass(frozen=True, slots=True)
class Recording:
    """One recording: 16 kHz mono PCM16 little-endian bytes and its reference, if any."""

    rid: str
    pcm: bytes
    reference: str = ""

    @property
    def samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_s(self) -> float:
        return self.samples / SAMPLE_RATE_HZ


Loader = Callable[[int | None], tuple[list[Recording], dict[str, Any]]]


def load_librispeech_test_other(limit: int | None) -> tuple[list[Recording], dict[str, Any]]:
    """Every test-other recording in ``stock_divergence.py``'s order: the same streaming
    ``load_dataset`` call, the same ``Audio(decode=False)`` cast, iterated in the order it yields.

    Decoded to int16 rather than the probe's float32: a 16-bit FLAC read as int16 is the exact
    stored samples, and the server's own decoder turns them into ``int16 / 32768`` float32
    (src/verbatim/audio/pcm.py), which is what ``sf.read(dtype="float32")`` gave the stock probe.
    References are lowercased as ``stock_divergence.py`` stores them.
    """
    import datasets
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    recordings: list[Recording] = []
    seen = 0
    for rec in ds:
        seen += 1
        if limit is not None and len(recordings) >= limit:
            # Read to the end rather than break: an abandoned streaming iterator made this
            # interpreter hang, or abort in PyGILState_Release, at exit (datasets 5.0.1). It
            # also lets a --limit run confirm the whole split is there.
            continue
        audio = rec["audio"]
        if audio.get("bytes"):
            raw = audio["bytes"]
        else:
            with open(audio["path"], "rb") as handle:
                raw = handle.read()
        data, rate = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
        if rate != SAMPLE_RATE_HZ:
            raise ValueError(
                f"{rec['id']}: {rate} Hz, expected {SAMPLE_RATE_HZ}; refusing to resample"
            )
        if data.ndim != 1:
            raise ValueError(f"{rec['id']}: {data.shape} is not mono; refusing to downmix")
        pcm = np.ascontiguousarray(data, dtype="<i2").tobytes()
        recordings.append(Recording(str(rec["id"]), pcm, str(rec["text"]).lower()))
    meta = {
        "split_size_observed": seen,
        "dataset": DATASET,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "order": (
            "load_dataset(streaming=True) iteration order after cast_column('audio', "
            "Audio(decode=False)), exactly as probes/stock_divergence.py"
        ),
        "references": "the dataset's text, lowercased, as probes/stock_divergence.py stores them",
        "datasets_version": datasets.__version__,
    }
    return recordings, meta


def load_references(
    path: Path,
    *,
    attempts: int = 10,
    wait_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[tuple[str, str]], str]:
    """The ``references`` of a stock record as ordered (id, reference) pairs, and the SHA-256 of
    the exact bytes parsed. The file is replaced atomically while its run goes on; a read that
    does not parse is retried, and ``attempts`` failures raise ValueError."""
    last: Exception | None = None
    for attempt in range(attempts):
        raw = path.read_bytes()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            last = exc
            if attempt + 1 < attempts:
                sleep(wait_s)
            continue
        references = body.get("references") if isinstance(body, dict) else None
        if not isinstance(references, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in references.items()
        ):
            raise ValueError(f"{path}: no 'references' mapping of id to text")
        return list(references.items()), hashlib.sha256(raw).hexdigest()
    raise ValueError(f"{path}: not valid JSON after {attempts} attempts: {last}")


def corpus_refusals(
    recordings: Sequence[Recording],
    meta: Mapping[str, Any],
    *,
    expected: int,
    expected_split: int,
    references: Sequence[tuple[str, str]],
) -> list[str]:
    """Every reason the loaded corpus is not the one the stock record was taken over."""
    found: list[str] = []
    split = meta.get("split_size_observed")
    if split is None:
        found.append("the loader did not report the size of the whole split")
    elif split != expected_split:
        found.append(f"the split holds {split} recordings, expected {expected_split}")
    if len(recordings) != expected:
        found.append(f"the corpus yielded {len(recordings)} recordings, expected {expected}")
    ids = [r.rid for r in recordings]
    if len(set(ids)) != len(ids):
        found.append("the corpus has duplicate recording ids")
    if len(references) != expected_split:
        found.append(
            f"the references record holds {len(references)} references, expected {expected_split}"
        )
    head = list(references[: len(recordings)])
    if len(head) < len(recordings):
        found.append(f"the references record has only {len(head)} of {len(recordings)} ids")
    pairs = list(enumerate(zip(recordings, head, strict=False)))
    order = [n for n, (r, (rid, _)) in pairs if r.rid != rid]
    if order:
        n = order[0]
        found.append(
            f"{len(order)} recording(s) out of the stock record's order; first n={n}: "
            f"loaded {recordings[n].rid!r}, the record has {head[n][0]!r}"
        )
    text = [n for n, (r, (rid, ref)) in pairs if r.rid == rid and r.reference != ref]
    if text:
        n = text[0]
        found.append(
            f"{len(text)} reference(s) differ from the stock record's; first {recordings[n].rid}: "
            f"loaded {recordings[n].reference!r}, the record has {head[n][1]!r}"
        )
    return found


# --- the tap: observes the harness client's connection, forwards everything unchanged ---


@dataclass(slots=True)
class Tap:
    """What went over one session's socket, observed beside ``run_session`` rather than
    reported by it. ``first_audio_at`` and ``received_at`` are ``time.monotonic()`` readings:
    when the first binary frame's send returned, and when each text frame in ``received`` was
    handed to the client."""

    connected: bool = False
    received: list[str] = field(default_factory=list)
    received_at: list[float] = field(default_factory=list)
    binary_bytes_sent: int = 0
    first_audio_at: float | None = None
    end_sent: bool = False
    close_code: int | None = None
    close_reason: str | None = None

    def frames(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in self.received:
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                event = None
            out.append(event if isinstance(event, dict) else {"type": "<unparseable>"})
        return out


_TAP: contextvars.ContextVar[Tap | None] = contextvars.ContextVar(
    "frozen_answers_tap", default=None
)


class _TappedConnection:
    """A client connection that records what it carries and changes nothing."""

    def __init__(self, ws: Any, tap: Tap) -> None:
        self._ws = ws
        self._tap = tap
        self._iterator: Any = None

    async def recv(self, *args: Any, **kwargs: Any) -> Any:
        message = await self._ws.recv(*args, **kwargs)
        self._note(message)
        return message

    def __aiter__(self) -> _TappedConnection:
        self._iterator = self._ws.__aiter__()
        return self

    async def __anext__(self) -> Any:
        message = await self._iterator.__anext__()
        self._note(message)
        return message

    async def send(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        answer = await self._ws.send(message, *args, **kwargs)
        if isinstance(message, bytes | bytearray | memoryview):
            if self._tap.first_audio_at is None:
                self._tap.first_audio_at = time.monotonic()
            self._tap.binary_bytes_sent += len(message)
        elif isinstance(message, str):
            with contextlib.suppress(json.JSONDecodeError):
                if json.loads(message) == {"type": "end"}:
                    self._tap.end_sent = True
        return answer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)

    def _note(self, message: Any) -> None:
        if isinstance(message, str):
            self._tap.received_at.append(time.monotonic())
            self._tap.received.append(message)


class _TappedConnect:
    """``websockets.connect(...)`` for ``async with``, handing out a tapped connection."""

    def __init__(self, inner: Any, tap: Tap | None) -> None:
        self._inner = inner
        self._tap = tap
        self._ws: Any = None

    async def __aenter__(self) -> Any:
        self._ws = await self._inner.__aenter__()
        if self._tap is None:
            return self._ws
        self._tap.connected = True
        return _TappedConnection(self._ws, self._tap)

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._inner.__aexit__(*exc)
        finally:
            if self._tap is not None and self._ws is not None:
                self._tap.close_code = getattr(self._ws, "close_code", None)
                self._tap.close_reason = getattr(self._ws, "close_reason", None)


class _TappingWebsockets:
    """Stands in for the ``websockets`` module inside ``verbatim_bench.client`` only."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def connect(self, uri: str, **kwargs: Any) -> _TappedConnect:
        return _TappedConnect(self._real.connect(uri, **kwargs), _TAP.get())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@contextlib.contextmanager
def tapped_client() -> Iterator[None]:
    """Route ``verbatim_bench.client``'s one ``websockets.connect`` call (client.py:384) through
    the tap for the duration, and put the module back afterwards."""
    original = bench_client.websockets
    bench_client.websockets = _TappingWebsockets(original)  # type: ignore[assignment]
    try:
        yield
    finally:
        bench_client.websockets = original  # type: ignore[assignment]


# --- one session, and the run ---


@dataclass(slots=True)
class Outcome:
    index: int
    recording: Recording
    result: SessionResult
    tap: Tap
    #: (first audio frame sent, last final received), seconds from the start of the run on
    #: the client's monotonic clock; None when the session sent no audio or got no final.
    in_flight_s: tuple[float, float] | None = None


@dataclass(slots=True)
class Capture:
    outcomes: list[Outcome]
    wall_clock_s: float


def frame_seed(seed: int, index: int) -> int:
    return seed * 100_000 + index


def in_flight(tap: Tap, start: float) -> tuple[float, float] | None:
    """When the session was in flight on the wire, from the tap's own stamps: from its first
    audio frame to the last final it received, in seconds from ``start``."""
    finals = [
        at
        for at, frame in zip(tap.received_at, tap.frames(), strict=True)
        if frame.get("type") == "final"
    ]
    if tap.first_audio_at is None or not finals:
        return None
    return (tap.first_audio_at - start, finals[-1] - start)


def occupancy_at_first_audio(intervals: Sequence[tuple[float, float] | None]) -> list[int]:
    """For each session with an interval, how many sessions were in flight on the wire at the
    instant its first audio frame went out: itself, and every other session whose first audio
    frame went out at or before that instant and whose last final arrived after it. A session
    whose last final arrived at that same instant is no longer in flight."""
    timed = [i for i in intervals if i is not None]
    return [
        1 + sum(1 for k, (s, e) in enumerate(timed) if k != n and s <= start < e)
        for n, (start, _) in enumerate(timed)
    ]


def peak_in_flight(intervals: Sequence[tuple[float, float] | None]) -> int:
    """The most sessions in flight on the wire at one instant. The count only rises when a
    session starts, so the peak is the largest occupancy at a first audio frame; 0 when no
    session has an interval."""
    return max(occupancy_at_first_audio(intervals), default=0)


async def capture(
    endpoint: str,
    recordings: Sequence[Recording],
    *,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    concurrency: int,
    seed: int = DEFAULT_SEED,
    lang: str = "en-US",
    progress: Callable[[int, int, int], None] | None = None,
) -> Capture:
    """Every recording once, in order, at most ``concurrency`` admitted at once, each on its own
    connection through ``run_session`` at real-time pace. Never raises for a session: a failed
    one is an outcome, judged afterwards. How many were in flight on the wire is each outcome's
    ``in_flight_s``, stamped by its tap; the admission bound is not a measurement of it."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency!r}")
    chunk = ChunkMode.parse(chunk_ms)
    gate = asyncio.Semaphore(concurrency)
    state = {"done": 0, "unanswered": 0}
    started = time.monotonic()

    async def one(index: int, recording: Recording) -> Outcome:
        async with gate:
            tap = Tap()
            token = _TAP.set(tap)
            try:
                result = await run_session(
                    endpoint,
                    session_id=f"frozen-{index:04d}",
                    utterance=Utterance(
                        stream_id=recording.rid,
                        audio_path=Path(recording.rid),
                        duration_s=recording.duration_s,
                        text=recording.reference,
                    ),
                    pcm=recording.pcm,
                    chunk=chunk,
                    start_delay_s=0.0,
                    words=True,
                    lang=lang,
                    frame_ms=constants.FRAME_MS,
                    frame_seed=frame_seed(seed, index),
                )
            finally:
                _TAP.reset(token)
        outcome = Outcome(index, recording, result, tap, in_flight(tap, started))
        state["done"] += 1
        frames = tap.frames()
        if not terminal_final_present(frames, recording.duration_s) or unexplained_resets(frames):
            state["unanswered"] += 1
        if progress is not None:
            progress(state["done"], len(recordings), state["unanswered"])
        return outcome

    with tapped_client():
        outcomes = await asyncio.gather(*(one(i, r) for i, r in enumerate(recordings)))
    return Capture(list(outcomes), time.monotonic() - started)


# --- judging: pure functions of what was observed ---


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finals(frames: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [frame for frame in frames if frame.get("type") == "final"]


def audio_matches(audio_s: object, duration_s: float) -> bool:
    """A wire ``audio_s`` equals ``duration_s`` within ``AUDIO_TOLERANCE_S``."""
    return _is_number(audio_s) and abs(float(audio_s) - duration_s) <= AUDIO_TOLERANCE_S  # type: ignore[arg-type]


def terminal_final_present(frames: Sequence[Mapping[str, Any]], duration_s: float) -> bool:
    """The last final on the wire covers every sample that was sent."""
    finals = _finals(frames)
    return bool(finals) and audio_matches(finals[-1].get("audio_s"), duration_s)


def _text(frame: Mapping[str, Any]) -> str:
    text = frame.get("text")
    return text.strip() if isinstance(text, str) else ""


def unexplained_resets(frames: Sequence[Mapping[str, Any]]) -> list[object]:
    """The ``audio_s`` of every partial whose text dropped from non-empty to empty with no
    final at the same ``audio_s``: a mid-recording final that never arrived."""
    at_finals = [float(f["audio_s"]) for f in _finals(frames) if _is_number(f.get("audio_s"))]
    previous = ""
    found: list[object] = []
    for frame in frames:
        if frame.get("type") != "partial":
            continue
        text = _text(frame)
        if previous and not text:
            at = frame.get("audio_s")
            if not any(audio_matches(at, final) for final in at_finals):
                found.append(at)
        previous = text
    return found


def _valid_word(entry: object) -> bool:
    if not isinstance(entry, Mapping):
        return False
    word, start, end = entry.get("w"), entry.get("s"), entry.get("e")
    return isinstance(word, str) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in (start, end)
    )


def word_confidence(finals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-word confidence as the wire carried it (``"c"``, contract C3), aligned with the
    client's word list (the well-formed entries, in order). Counted, never assumed."""
    values: list[Any] = []
    present = not_a_number = 0
    keys: set[str] = set()
    for frame in finals:
        raw = frame.get("words")
        if not isinstance(raw, list):
            continue
        for entry in raw:
            if not _valid_word(entry):
                continue
            keys.update(str(k) for k in entry if k not in (*WORD_KEYS, CONFIDENCE_KEY))
            if CONFIDENCE_KEY not in entry:
                values.append(None)
                continue
            present += 1
            value = entry[CONFIDENCE_KEY]
            if not _is_number(value):
                not_a_number += 1
            values.append(value)
    return {
        "values": values if present else None,
        "word_entries": len(values),
        "with_confidence": present,
        "not_a_number": not_a_number,
        "unparsed_word_keys": sorted(keys),
    }


def judge(
    outcome: Outcome, *, chunk_ms: int, word_confidence_mode: str = DEFAULT_WORD_CONFIDENCE
) -> dict[str, Any]:
    """One recording's entry: what the server answered, and every reason it does not count."""
    recording, result, tap = outcome.recording, outcome.result, outcome.tap
    frames = tap.frames()
    session = next((frame for frame in frames if frame.get("type") == "session"), None)
    finals = _finals(frames)
    partials = [frame for frame in frames if frame.get("type") == "partial"]
    errors = [frame for frame in frames if frame.get("type") == "error"]
    sent = tap.end_sent and tap.binary_bytes_sent == len(recording.pcm)
    terminal = terminal_final_present(frames, recording.duration_s)
    resets = unexplained_resets(frames)
    problems: list[str] = []
    if result.error is not None:
        problems.append(f"client error: {result.error}")
    if not tap.connected:
        problems.append(
            "the tap never saw this session's connection, so nothing below was observed"
        )
    if session is None:
        problems.append("no session frame: the server never acknowledged the session")
    elif session.get("chunk_ms") != chunk_ms:
        problems.append(
            f"the session frame says chunk_ms {session.get('chunk_ms')!r}, not {chunk_ms}"
        )
    if tap.binary_bytes_sent != len(recording.pcm):
        problems.append(
            f"{tap.binary_bytes_sent} of {len(recording.pcm)} audio bytes went on the wire"
        )
    if not tap.end_sent:
        problems.append("the end message never went on the wire")
    for frame in errors:
        problems.append(f"server error frame {frame.get('code')}: {frame.get('message')}")
    if not finals:
        problems.append("no final: the server sent no final frame for this recording")
    elif not terminal:
        problems.append(
            f"no terminal final: the last of {len(finals)} final(s) covers "
            f"{finals[-1].get('audio_s')!r} s of the {recording.duration_s:.6f} s sent"
        )
    if resets:
        problems.append(
            f"lost mid-recording final: the partial text dropped to empty at audio_s {resets} "
            "with no final at the same audio_s"
        )
    if recording.samples and not partials:
        problems.append(
            "no partial arrived: the lost-mid-recording-final rule reads the partials, so a "
            "final lost mid-recording could not have been seen"
        )
    for k, frame in enumerate(finals):
        raw = frame.get("words")
        if not isinstance(raw, list):
            problems.append(f"final {k} carried no words list although words=1 was asked")
        elif _text(frame) and not any(_valid_word(entry) for entry in raw):
            problems.append(f"final {k} carried text {_text(frame)!r} but no word timings")
    # A check of the wiring, not a second reading: the tap and run_session see the same recv
    # calls and join them with the same functions, so they disagree only when the tap is
    # bypassed or mis-wired.
    tapped_words = join_final_words(finals)
    if len(finals) != result.finals_received:
        problems.append(
            f"wiring: the tap saw {len(finals)} final(s), run_session counted "
            f"{result.finals_received}"
        )
    if join_final_texts(finals) != result.final_text:
        problems.append("wiring: the tap's joined final text is not run_session's")
    if tapped_words != list(result.words):
        problems.append("wiring: the tap's word timings are not run_session's")
    raw_words = sum(len(frame["words"]) for frame in finals if isinstance(frame.get("words"), list))
    if raw_words != len(result.words):
        problems.append(
            f"{raw_words - len(result.words)} word entr(ies) on the wire were not "
            "integer-millisecond (w, s, e) and were left out by the client"
        )
    if tap.close_code != 1000:
        problems.append(f"the socket closed with code {tap.close_code!r}, not 1000 (a normal end)")
    confidence = word_confidence(finals)
    with_c, entries = confidence["with_confidence"], confidence["word_entries"]
    if word_confidence_mode == "off" and with_c:
        problems.append(
            f"{with_c} word(s) carry a confidence {CONFIDENCE_KEY!r} and the run declares word "
            "confidence off: the server is not running what was declared"
        )
    elif word_confidence_mode != "off" and with_c != entries:
        problems.append(
            f"{entries - with_c} of {entries} word(s) carry no confidence {CONFIDENCE_KEY!r} and "
            f"the run declares word confidence {word_confidence_mode}"
        )
    if confidence["not_a_number"]:
        problems.append(
            f"{confidence['not_a_number']} word confidence(s) on the wire are not numbers"
        )
    words: list[list[Any]] = [list(word) for word in result.words]
    values = confidence["values"]
    if values is not None and len(values) == len(words):
        # The fourth element, only where the wire carried a number: what
        # scripts/step1_places.py --confidence reads. Digests and timing identity use
        # (word, start, end) only.
        words = [
            w + ([float(c)] if _is_number(c) else []) for w, c in zip(words, values, strict=True)
        ]
    return {
        "id": recording.rid,
        "n": outcome.index,
        "text": result.final_text,
        "words": words,
        "word_confidence_present": with_c,
        "unparsed_word_keys": confidence["unparsed_word_keys"],
        "finals": len(finals),
        "final_frames": [
            {
                "text": frame.get("text"),
                "audio_s": frame.get("audio_s"),
                "words": len(frame["words"]) if isinstance(frame.get("words"), list) else None,
                "confidence_present": "confidence" in frame,
                "confidence": frame.get("confidence"),
                "unparsed_keys": sorted(str(k) for k in frame if k not in FINAL_KEYS),
            }
            for frame in finals
        ],
        "terminal_final": terminal,
        "unexplained_partial_resets": resets,
        "sent": sent,
        "in_flight_s": None if outcome.in_flight_s is None else list(outcome.in_flight_s),
        "audio_samples": recording.samples,
        "audio_s": round(recording.duration_s, 6),
        "pcm_sha256": hashlib.sha256(recording.pcm).hexdigest(),
        "reference": recording.reference,
        "server_session_id": result.server_session_id,
        "session_chunk_ms": None if session is None else session.get("chunk_ms"),
        "invariance_class": None if session is None else session.get("invariance_class"),
        "partials": result.partials_received,
        "partials_tapped": len(partials),
        "close_code": tap.close_code,
        "client_error": result.error,
        "problems": problems,
    }


def run_failures(entries: Sequence[Mapping[str, Any]], *, expected: int | None) -> list[str]:
    """Why this run's answers may not be reported as a success. Empty means success."""
    failures: list[str] = []
    bad = [entry for entry in entries if entry["problems"]]
    if bad:
        first = bad[0]
        failures.append(
            f"{len(bad)} of {len(entries)} recording(s) failed; first {first['id']} "
            f"(n={first['n']}): {first['problems'][0]}"
        )
    if expected is not None and len(entries) != expected:
        failures.append(f"{len(entries)} recordings were captured, {expected} were expected")
    ids = [entry["id"] for entry in entries]
    if len(set(ids)) != len(ids):
        failures.append("recording ids are not unique, so answers cannot be told apart")
    if not entries:
        failures.append("no recording was captured")
    elif all(entry["text"] == "" for entry in entries):
        failures.append(
            "vacuous: every final is empty; a server that transcribes nothing produces exactly this"
        )
    return failures


def counts_of(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    with_terminal = sum(1 for entry in entries if entry["terminal_final"])
    return {
        "recordings": len(entries),
        "sessions_acknowledged": sum(1 for e in entries if e["server_session_id"] is not None),
        "recordings_sent": sum(1 for entry in entries if entry["sent"]),
        "recordings_with_terminal_final": with_terminal,
        "recordings_missing_final": len(entries) - with_terminal,
        "recordings_with_unexplained_partial_resets": sum(
            1 for entry in entries if entry["unexplained_partial_resets"]
        ),
        "recordings_without_partials": sum(1 for e in entries if not e["partials_tapped"]),
        "final_frames_received": sum(entry["finals"] for entry in entries),
        "recordings_with_problems": sum(1 for entry in entries if entry["problems"]),
        "empty_text": sum(1 for entry in entries if entry["text"] == ""),
        "text_without_words": sum(1 for e in entries if e["text"] and not e["words"]),
        "word_entries_with_confidence": sum(e["word_confidence_present"] for e in entries),
    }


# --- what the server says it is, and how it is coping ---


def _fetch_text(url: str, *, timeout_s: float = 5.0, error_body: bool = False) -> str | None:
    """The body at ``url``; for an HTTP error status, its body when ``error_body`` (``/readyz``
    answers 503 with the reason while the server is not ready), else None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if not error_body:
            return None
        try:
            return exc.read().decode("utf-8")
        except (OSError, ValueError):
            return None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def read_admission(endpoint: str) -> dict[str, Any] | None:
    text = _fetch_text(health_url(endpoint, "/admission"))
    try:
        body = json.loads(text) if text is not None else None
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def parse_metrics(text: str) -> dict[str, float]:
    """The unlabelled-by-quantile series of a Prometheus text exposition, by name."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if "quantile=" in head:
            continue
        try:
            out[head.split("{", 1)[0]] = float(value)
        except ValueError:
            continue
    return out


def read_metrics(endpoint: str) -> dict[str, float] | None:
    text = _fetch_text(health_url(endpoint, "/metrics"))
    return None if text is None else parse_metrics(text)


@dataclass(frozen=True, slots=True)
class ServerReading:
    """One reading of ``/readyz``, ``/admission`` and ``/metrics``; ``readyz`` is the raw body,
    which carries the fields ``ServerFacts`` does not (contract C3)."""

    facts: ServerFacts | None
    admission: dict[str, Any] | None
    metrics: dict[str, float] | None
    readyz: dict[str, Any] | None = None


def read_server(endpoint: str) -> ServerReading:
    text = _fetch_text(health_url(endpoint), error_body=True)
    body: dict[str, Any] | None = None
    if text is not None:
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        body = parsed if isinstance(parsed, dict) else None
    facts = None
    if body is not None:
        raw = text or ""
        facts = read_server_facts(endpoint, fetch=lambda _url: raw)
    return ServerReading(facts, read_admission(endpoint), read_metrics(endpoint), body)


def reported_c3(reading: ServerReading) -> dict[str, Any]:
    """The ``/readyz`` fields of contract C3, None where the server does not report them:
    ``word_confidence`` as configured, and ``observed`` read from the built model."""
    body = reading.readyz or {}
    observed = body.get("observed")
    observed = observed if isinstance(observed, dict) else {}
    att = observed.get("att_context_size")
    return {
        "word_confidence": body.get("word_confidence"),
        "att_context_size": list(att) if isinstance(att, list | tuple) else att,
        "decoder_step_confidence": observed.get("decoder_step_confidence"),
        "decoder_graphs": observed.get("decoder_graphs"),
    }


def reported_code(reading: ServerReading) -> dict[str, Any] | None:
    """``/readyz``'s ``code`` object (contract C7), each of ``CODE_KEYS`` as reported; None when
    the server reports no ``code`` object at all."""
    code = (reading.readyz or {}).get("code")
    if not isinstance(code, dict):
        return None
    return {key: code.get(key) for key in CODE_KEYS}


def _count(value: object) -> int | float | None:
    if not _is_number(value):
        return None
    number = float(value)  # type: ignore[arg-type]
    return int(number) if number.is_integer() else number


def load_stamp(reading: ServerReading) -> dict[str, Any]:
    """How the server was coping at one moment, from /admission and /metrics."""
    admission = reading.admission or {}
    metrics = reading.metrics or {}
    return {
        "p95_tick_ms": admission.get("p95_tick_ms"),
        "degradation_level": admission.get("degradation_level"),
        "consecutive_overruns": admission.get("consecutive_overruns"),
        "live": admission.get("live"),
        "admitted_total": admission.get("admitted_total"),
        "refused_total": admission.get("refused_total"),
        "result_partials_dropped": admission.get("result_partials_dropped"),
        "ticks_over_budget_total": _count(metrics.get("verbatim_ticks_over_budget_total")),
        "ticks_total": _count(metrics.get("verbatim_ticks_total")),
        "ticks_late_total": _count(metrics.get("verbatim_ticks_late_total")),
    }


def _delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int | None:
    a, b = before.get(key), after.get(key)
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, int)
        or not isinstance(b, int)
    ):
        return None
    return b - a


_IDENTITY = (
    "model",
    "pipeline",
    "chunk_ms",
    "precision",
    "execution",
    "biasing",
    "nemo_version",
    "torch_version",
    "device_name",
)


def identity(reading: ServerReading) -> dict[str, Any] | None:
    """What the server says it is, from one reading: every ``/readyz`` field but the readiness
    and the tick count, the C3 fields and the C7 ``code`` included."""
    if reading.facts is None:
        return None
    body = reading.facts.to_json_dict()
    return {
        **{key: body.get(key) for key in _IDENTITY},
        **reported_c3(reading),
        "code": reported_code(reading),
    }


def _changed(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> list[str]:
    if a is None or b is None:
        return ["the whole reading"]
    return sorted(k for k in {*a, *b} if a.get(k) != b.get(k))


def _tick(reading: ServerReading) -> int | None:
    tick = None if reading.facts is None else reading.facts.tick_id
    return tick if isinstance(tick, int) and not isinstance(tick, bool) else None


@dataclass(frozen=True, slots=True)
class Declared:
    """What the run declares the server is. Each field is compared with what is observed."""

    model: str
    pipeline: str
    chunk_ms: int
    dtype: str
    execution: str
    bucket: int
    concurrency: int
    padding: str
    word_confidence: str
    decoder_graphs: bool
    att_context: list[int] | None
    #: The directory the server must have imported ``verbatim`` from (contract C7), resolved:
    #: ``<checkout>/src/verbatim`` of this file's checkout, or of ``--server-checkout``.
    server_verbatim_path: str


def expected_verbatim_path(server_checkout: Path | None) -> tuple[str, str]:
    """(the directory the server must import ``verbatim`` from, resolved; where that comes
    from): ``--server-checkout DIR`` when given, else this file's own checkout."""
    if server_checkout is None:
        return str((REPO / "src" / "verbatim").resolve()), "this client's own checkout"
    return str((server_checkout / "src" / "verbatim").resolve()), "--server-checkout"


def refusals(reading: ServerReading, declared: Declared) -> list[str]:
    """Every reason not to send a byte: the server did not say what it is, or said something
    the declaration contradicts."""
    facts, admission, d = reading.facts, reading.admission, declared
    if facts is None:
        return ["/readyz did not answer beside the endpoint, so the run could not be stamped"]
    found: list[str] = []
    if not facts.ready:
        found.append(
            f"/readyz says the server is not ready ({(reading.readyz or {}).get('reason')})"
        )
    for name, value in (
        ("model", facts.model),
        ("pipeline", facts.pipeline),
        ("chunk_ms", facts.chunk_ms),
        ("precision", facts.precision),
        ("execution", facts.execution),
        ("biasing", facts.biasing),
    ):
        if value is None:
            found.append(f"/readyz does not report {name}, so the run cannot stamp it as observed")
    try:
        check_arm(facts, arm="", declared_dtype=d.dtype, declared_chunk_ms=d.chunk_ms)
    except ArmContradiction as exc:
        found.append(str(exc))
    if facts.biasing:
        found.append(
            "the server reports biasing on: it serves phrase lists, a separate arm whose decode "
            "path differs for every session, biased or not; start it without --biasing"
        )
    if facts.model is not None and facts.model != d.model:
        found.append(f"the server reports model {facts.model!r}, the run declares {d.model!r}")
    if facts.pipeline is not None and facts.pipeline != d.pipeline:
        found.append(
            f"the server reports pipeline {facts.pipeline!r}, the run declares {d.pipeline!r}"
        )
    if facts.execution is not None and facts.execution != d.execution:
        found.append(
            f"the server reports execution {facts.execution!r}, the run declares {d.execution!r}"
        )
    c3 = reported_c3(reading)
    if c3["word_confidence"] is not None and c3["word_confidence"] != d.word_confidence:
        found.append(
            f"/readyz reports word confidence {c3['word_confidence']!r}, the run declares "
            f"{d.word_confidence!r}"
        )
    step = c3["decoder_step_confidence"]
    if step is not None and step != (d.word_confidence != "off"):
        found.append(
            f"/readyz observes decoder_step_confidence {step!r} on the built decoder, and the run "
            f"declares word confidence {d.word_confidence!r}"
        )
    if c3["decoder_graphs"] is not None and c3["decoder_graphs"] != d.decoder_graphs:
        found.append(
            f"/readyz observes decoder_graphs {c3['decoder_graphs']!r} in the built pipeline, the "
            f"run declares {d.decoder_graphs!r}"
        )
    if c3["att_context_size"] is None:
        found.append(
            "/readyz does not report observed.att_context_size (contract C3), the attention "
            "context of the BUILT encoder: scripts/compare_captures.py refuses every capture "
            "without it, so this run would be a real-time pass over the corpus that nothing can "
            "compare; serve from a verbatim whose /readyz reports it"
        )
    if d.att_context is None:
        found.append(
            "no attention context is declared and none can be derived for this model and chunk"
        )
    elif c3["att_context_size"] is not None and c3["att_context_size"] != d.att_context:
        found.append(
            f"/readyz observes att_context_size {c3['att_context_size']} on the built encoder, "
            f"the run declares {d.att_context}"
        )
    served_from = (reported_code(reading) or {}).get("verbatim_path")
    if not isinstance(served_from, str):
        found.append(
            "/readyz does not report code.verbatim_path (contract C7), the directory the server "
            "imported verbatim from: which code answers is not observed, and "
            "scripts/compare_captures.py refuses a capture without it; serve from a verbatim "
            "whose /readyz reports it"
        )
    elif served_from != d.server_verbatim_path:
        found.append(
            f"/readyz says the server imported verbatim from {served_from}, and the run expects "
            f"{d.server_verbatim_path}: the server does not run the checkout's code; start it "
            "with PYTHONPATH=<checkout>/src:<checkout>/bench/src, or name its checkout with "
            "--server-checkout"
        )
    if admission is None:
        found.append("/admission did not answer, so the bucket cannot be observed")
    else:
        served = admission.get("bucket")
        if served != d.bucket:
            found.append(f"the server reports bucket {served!r}, the run declares {d.bucket}")
        elif d.concurrency > d.bucket:
            found.append(
                f"concurrency {d.concurrency} exceeds the bucket {d.bucket}: sessions past it are "
                "refused"
            )
        for key in (
            "admitted_total",
            "refused_total",
            "p95_tick_ms",
            "degradation_level",
            "consecutive_overruns",
        ):
            if key not in admission:
                found.append(f"/admission does not report {key}, so the run cannot stamp it")
    if reading.metrics is None:
        found.append("/metrics did not answer, so ticks over budget cannot be stamped")
    elif "verbatim_ticks_over_budget_total" not in reading.metrics:
        found.append("/metrics has no verbatim_ticks_over_budget_total series")
    return found


def readyz_warnings(reading: ServerReading) -> list[str]:
    """The C3 fields this server does not report, but for ``att_context_size``, which
    ``refusals`` requires: stamped as not observed, never assumed."""
    if reading.facts is None:
        return []
    return [
        f"/readyz does not report {name} (contract C3): it is stamped as NOT observed"
        for name, value in reported_c3(reading).items()
        if value is None and name != "att_context_size"
    ]


def continuity_refusals(first: ServerReading, later: ServerReading, stage: str) -> list[str]:
    """``later`` is the same server process as ``first``, saying the same thing."""
    found: list[str] = []
    if identity(later) != identity(first):
        found.append(
            f"the server's /readyz identity changed {stage}: "
            f"{', '.join(_changed(identity(first), identity(later)))}"
        )
    t0, t1 = _tick(first), _tick(later)
    if t0 is None or t1 is None or t1 < t0:
        found.append(f"the server's tick count went from {t0} to {t1} {stage}: it restarted")
    return found


def server_failures(
    before: ServerReading,
    after: ServerReading,
    *,
    recordings: int,
    first: ServerReading | None = None,
) -> tuple[list[str], list[str]]:
    """(failures, warnings) from the server's readings either side of the run. The server must
    be the same process saying the same thing, its tick count must never have gone back from
    the first reading on, and it must have admitted exactly this run's sessions and refused
    none."""
    failures: list[str] = []
    warnings: list[str] = []
    if after.facts is None:
        failures.append("/readyz stopped answering by the end of the run")
    else:
        if not after.facts.ready:
            failures.append("/readyz says the server is not ready at the end of the run")
        if identity(after) != identity(before):
            failures.append(
                "the server's /readyz identity changed during the run: "
                f"{', '.join(_changed(identity(before), identity(after)))}"
            )
        ticks = [_tick(r) for r in (first or before, before, after)]
        if None in ticks or not ticks[0] <= ticks[1] < ticks[2]:  # type: ignore[operator]
            failures.append(
                f"the server's tick count went {' -> '.join(str(t) for t in ticks)} from the "
                "first reading to the end of the run: it restarted"
            )
    stamp_before, stamp_after = load_stamp(before), load_stamp(after)
    admitted = _delta(stamp_before, stamp_after, "admitted_total")
    refused = _delta(stamp_before, stamp_after, "refused_total")
    if admitted is None:
        failures.append("the admitted count could not be read before and after the run")
    elif admitted != recordings:
        failures.append(
            f"the server admitted {admitted} session(s) during the run, and {recordings} were "
            "sent: another client used the server, or it restarted"
        )
    if refused is None:
        failures.append("the refused count could not be read before and after the run")
    elif refused != 0:
        failures.append(f"the server refused {refused} session(s) during the run")
    over = _delta(stamp_before, stamp_after, "ticks_over_budget_total")
    if over is None:
        failures.append("ticks over budget could not be read from /metrics before and after")
    elif over > 0:
        warnings.append(
            f"{over} tick(s) went over budget during the run: the card did not keep real time "
            "on every tick"
        )
    for key in ("degradation_level", "consecutive_overruns"):
        value = stamp_after.get(key)
        if isinstance(value, int) and value > 0:
            warnings.append(f"{key} was {value} at the end of the run")
    return failures, warnings


# --- what the host says: the card, the server process, the model revision, the banner ---


def default_hub_dir() -> Path:
    """Where huggingface_hub keeps its cache for THIS process, by its own resolution."""
    try:
        from huggingface_hub import constants as hf_constants

        return Path(hf_constants.HF_HUB_CACHE)
    except ImportError:
        return hub_dir_from_environ(os.environ)


def _expand(value: str, env: Mapping[str, str]) -> str:
    """``os.path.expandvars`` then ``expanduser``, against ``env`` rather than this process."""

    def var(match: re.Match[str]) -> str:
        return env.get(match.group(1) or match.group(2), match.group(0))

    value = re.sub(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))", var, value)
    if value == "~" or value.startswith("~/"):
        home = env.get("HOME")
        if not home:
            raise ValueError("the environment has no HOME to expand '~' with")
        value = home + value[1:]
    return value


def hub_dir_from_environ(env: Mapping[str, str]) -> Path:
    """The hub cache huggingface_hub resolves in a process with environment ``env`` (its
    constants.py: HF_HUB_CACHE, else HUGGINGFACE_HUB_CACHE, else HF_HOME/hub, HF_HOME defaulting
    to $XDG_CACHE_HOME/huggingface or ~/.cache/huggingface). A variable set empty is unset."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key):
            return Path(_expand(env[key], env))
    home = env.get("HF_HOME") or os.path.join(
        env.get("XDG_CACHE_HOME") or "~/.cache", "huggingface"
    )
    return Path(_expand(home, env)) / "hub"


def observe_revision(hub_dir: Path, model: str) -> dict[str, Any]:
    """``refs/main`` and the snapshot directories of ``model`` in the cache at ``hub_dir``."""
    repo = hub_dir / ("models--" + model.replace("/", "--"))
    obs: dict[str, Any] = {
        "hub_dir": str(hub_dir),
        "repo_dir": str(repo),
        "refs_main": None,
        "snapshots": None,
        "error": None,
    }
    try:
        obs["refs_main"] = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
        obs["snapshots"] = sorted(p.name for p in (repo / "snapshots").iterdir() if p.is_dir())
    except OSError as exc:
        obs["error"] = f"{type(exc).__name__}: {exc}"
    return obs


def revision_refusals(obs: Mapping[str, Any], pin: str) -> list[str]:
    if obs.get("error"):
        return [f"the model revision could not be read from the cache: {obs['error']}"]
    found: list[str] = []
    if obs.get("refs_main") != pin:
        found.append(
            f"refs/main in {obs['repo_dir']} is {obs.get('refs_main')!r}, not the pin {pin!r}"
        )
    snapshots = obs.get("snapshots") or []
    if len(snapshots) != 1:
        found.append(
            f"{len(snapshots)} snapshots in {obs['repo_dir']}, expected exactly one: which one the "
            "server loaded cannot be told"
        )
    elif snapshots[0] != pin:
        found.append(f"the only snapshot is {snapshots[0]!r}, not the pin {pin!r}")
    return found


Runner = Callable[[Sequence[str]], str]


def run_command(argv: Sequence[str]) -> str:
    return subprocess.run(list(argv), capture_output=True, text=True, check=True, timeout=30).stdout


def observe_gpu(runner: Runner, *, index: int, port: int | None) -> dict[str, Any]:
    """The card nvidia-smi calls ``index``, the processes holding a context on it, and the
    processes listening on the endpoint's port. Queries only: nothing is allocated on a card."""
    obs: dict[str, Any] = {
        "index": index,
        "uuid": None,
        "name": None,
        "compute_apps": None,
        "compute_pids": None,
        "port": port,
        "listener_pids": None,
        "error": None,
    }
    try:
        query = ["nvidia-smi", "-i", str(index), "--query-gpu=uuid,name", "--format=csv,noheader"]
        lines = [x.strip() for x in runner(query).splitlines() if x.strip()]
        if len(lines) != 1:
            raise ValueError(f"nvidia-smi -i {index} named {len(lines)} cards")
        uuid, _, name = lines[0].partition(",")
        obs["uuid"], obs["name"] = uuid.strip(), name.strip()
        query = [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        apps = [x.strip() for x in runner(query).splitlines() if x.strip()]
        obs["compute_apps"] = apps
        obs["compute_pids"] = sorted(
            {int(a.split(",")[0]) for a in apps if a.split(",")[0].strip().isdigit()}
        )
        if port is None:
            raise ValueError("the endpoint names no port, so its listener cannot be found")
        listeners = runner(["ss", "-H", "-ltnp", f"sport = :{port}"])
        obs["listener_pids"] = sorted({int(p) for p in re.findall(r"pid=(\d+)", listeners)})
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        obs["error"] = f"{type(exc).__name__}: {exc}"
    return obs


def gpu_refusals(
    obs: Mapping[str, Any], *, expected_uuid: str, device_name: str | None
) -> list[str]:
    """The card is the named one, the device ``/readyz`` names, and the server listening on the
    endpoint's port is the ONLY process holding a context on it."""
    if obs.get("error"):
        return [f"the card could not be observed: {obs['error']}"]
    found: list[str] = []
    index = obs.get("index")
    if obs.get("uuid") != expected_uuid:
        found.append(f"nvidia-smi -i {index} is {obs.get('uuid')!r}, not {expected_uuid!r}")
    if obs.get("name") != device_name:
        found.append(
            f"nvidia-smi -i {index} names {obs.get('name')!r} and /readyz names "
            f"{device_name!r}: the server is not on this card"
        )
    listeners = obs.get("listener_pids") or []
    on_card = obs.get("compute_pids") or []
    if len(listeners) != 1:
        found.append(
            f"{len(listeners)} process(es) listen on port {obs.get('port')}, expected exactly one "
            "server"
        )
        return found
    pid = listeners[0]
    if pid not in on_card:
        found.append(
            f"the process listening on port {obs.get('port')} (pid {pid}) holds no context on "
            f"card {index}: the server runs somewhere else"
        )
    others = [p for p in on_card if p != pid]
    if others:
        found.append(
            f"card {index} is shared: pid(s) {others} hold a context on it beside the server "
            f"(pid {pid}); the server must be the only process on the card"
        )
    return found


ProcReader = Callable[[int, str], bytes]


def read_proc(pid: int, name: str) -> bytes:
    """``/proc/<pid>/<name>``, raising OSError when it cannot be read."""
    return Path("/proc", str(pid), name).read_bytes()


def parse_serve_argv(argv: Sequence[str]) -> dict[str, Any]:
    """The flags a ``verbatim serve`` command line gives, by verbatim's own parser, and the
    attention context the server's own ``att_context_size`` derives from them. ValueError when
    the command line is not one ``verbatim serve`` accepts."""
    from verbatim.cli import _parser as serve_parser
    from verbatim.pipelines.nemo_runtime import att_context_size
    from verbatim.scheduler.graph_budget import ConfigError

    try:
        at = list(argv).index("serve", 1)
    except ValueError as exc:
        raise ValueError(f"not a `verbatim serve` command line: {' '.join(argv)!r}") from exc
    sink = io.StringIO()
    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            ns = serve_parser().parse_args(list(argv[at:]))
    except SystemExit as exc:
        raise ValueError(
            f"verbatim's serve parser rejects this command line: {sink.getvalue().strip()}"
        ) from exc
    try:
        att: list[int] | None = list(att_context_size(ns.model, ns.chunk, left=ns.att_context_left))
    except (ConfigError, ValueError):
        att = None
    return {
        "model": ns.model,
        "chunk_ms": ns.chunk.ms,
        "bucket": ns.bucket,
        "ceiling": ns.ceiling,
        "padding": ns.padding,
        "eager": ns.eager,
        "biasing": ns.biasing,
        "decoder_graphs": ns.decoder_graphs,
        "word_confidence": ns.word_confidence,
        "att_context_left": ns.att_context_left,
        "att_context": att,
        "compute_dtype": ns.compute_dtype,
        "pipeline": ns.pipeline,
        "device_id": ns.device_id,
        "host": ns.host,
        "ws_port": ns.ws_port,
    }


def observe_process(proc: ProcReader, pid: int | None) -> dict[str, Any]:
    """The server process's command line (parsed) and environment, read from ``/proc``."""
    obs: dict[str, Any] = {
        "pid": pid,
        "cmdline": None,
        "cmdline_error": None,
        "flags": None,
        "flags_error": None,
        "environ": None,
        "environ_sha256": None,
        "environ_error": None,
        "hub_dir": None,
        "hub_dir_error": None,
    }
    if pid is None:
        obs["cmdline_error"] = obs["environ_error"] = (
            "no single process listens on the endpoint's port"
        )
        return obs
    try:
        raw = proc(pid, "cmdline")
    except OSError as exc:
        obs["cmdline_error"] = f"{type(exc).__name__}: {exc}"
    else:
        argv = [p.decode("utf-8", "replace") for p in raw.rstrip(b"\0").split(b"\0")] if raw else []
        obs["cmdline"] = argv
        try:
            obs["flags"] = parse_serve_argv(argv)
        except ValueError as exc:
            obs["flags_error"] = str(exc)
    try:
        raw = proc(pid, "environ")
    except OSError as exc:
        obs["environ_error"] = f"{type(exc).__name__}: {exc}"
    else:
        items = raw.decode("utf-8", "replace").split("\0")
        env = dict(item.split("=", 1) for item in items if "=" in item)
        obs["environ"] = {key: env.get(key) for key in ENV_KEYS}
        obs["environ_sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            obs["hub_dir"] = str(hub_dir_from_environ(env))
        except ValueError as exc:
            obs["hub_dir_error"] = str(exc)
    return obs


def process_refusals(
    obs: Mapping[str, Any],
    declared: Declared,
    *,
    gpu_index: int,
    gpu_uuid: str,
    port: int,
    hub_cache: Path | None,
) -> list[str]:
    """The server's own command line and environment against the declaration and the card."""
    found: list[str] = []
    pid, flags, d = obs.get("pid"), obs.get("flags"), declared
    if obs.get("cmdline") is not None and flags is None:
        found.append(
            f"the process on the port (pid {pid}) is not a server this client can read: "
            f"{obs.get('flags_error')}"
        )
    if flags is not None:
        for name, have, want in (
            ("model", flags["model"], d.model),
            ("chunk ms", flags["chunk_ms"], d.chunk_ms),
            ("bucket", flags["bucket"], d.bucket),
            ("padding", flags["padding"], d.padding),
            ("compute dtype", flags["compute_dtype"], d.dtype),
            ("pipeline", flags["pipeline"], d.pipeline),
            ("execution", "eager" if flags["eager"] else "graph path", d.execution),
            ("word confidence", flags["word_confidence"], d.word_confidence),
            ("decoder graphs", flags["decoder_graphs"], d.decoder_graphs),
            ("biasing", flags["biasing"], False),
            ("attention context", flags["att_context"], d.att_context),
        ):
            if have != want:
                found.append(
                    f"the server was started with {name} {have!r} (/proc/{pid}/cmdline), the run "
                    f"declares {want!r}"
                )
        if flags["ws_port"] not in (0, port):
            found.append(
                f"the server's command line listens on port {flags['ws_port']}, the endpoint is "
                f"port {port}"
            )
    env = obs.get("environ")
    if env is not None and flags is not None:
        order = env.get("CUDA_DEVICE_ORDER")
        if order != "PCI_BUS_ID":
            found.append(
                f"the server's CUDA_DEVICE_ORDER is {order!r}, not 'PCI_BUS_ID': its "
                f"cuda:{flags['device_id']} cannot be tied to nvidia-smi index {gpu_index}"
            )
        visible = env.get("CUDA_VISIBLE_DEVICES")
        device = flags["device_id"]
        if visible is None:
            card: str | None = str(device)
        else:
            names = [x.strip() for x in visible.split(",")]
            card = names[device] if 0 <= device < len(names) else None
        if card not in (str(gpu_index), gpu_uuid):
            found.append(
                f"the server's cuda:{device} is card {card!r} by its CUDA_VISIBLE_DEVICES="
                f"{visible!r}, not card {gpu_index} ({gpu_uuid})"
            )
    hub = obs.get("hub_dir")
    if hub is not None and hub_cache is not None and Path(hub).resolve() != hub_cache.resolve():
        found.append(
            f"the server resolves its Hugging Face hub cache to {hub}, and --hf-hub-cache names "
            f"{hub_cache}"
        )
    if obs.get("hub_dir_error"):
        found.append(f"the server's hub cache cannot be resolved: {obs['hub_dir_error']}")
    return found


def process_warnings(obs: Mapping[str, Any], *, hub_source: str) -> list[str]:
    """What could not be read from the server process, said loudly."""
    found: list[str] = []
    if obs.get("cmdline_error"):
        found.append(
            f"the server's command line could not be read ({obs['cmdline_error']}): its flags are "
            "NOT observed from the process; padding, word confidence and decoder graphs rest on "
            "the server log's banner alone"
        )
    if obs.get("environ_error"):
        found.append(
            f"the server's environment could not be read ({obs['environ_error']}): which card its "
            f"cuda device is and which Hugging Face cache it reads are NOT observed; the revision "
            f"is read from {hub_source}, which is only assumed to be the server's"
        )
    return found


_BANNER = "[verbatim] "


def banner_facts(text: str) -> dict[str, Any]:
    """What the LAST startup banner in a ``verbatim serve`` log says (cli.py _banner and the
    ready lines). Padding is ragged when the RAGGED line is present, fixed when a complete
    banner lacks it: the server prints nothing for fixed; likewise word confidence, decoder
    graphs and biasing, which print a line only when on."""
    lines = [line.rstrip("\n") for line in text.splitlines()]
    starts = [i for i, line in enumerate(lines) if line.startswith(_BANNER + "checkpoint ")]
    facts: dict[str, Any] = {
        "banners": len(starts),
        "checkpoint": None,
        "chunk_ms": None,
        "att_context": None,
        "bucket": None,
        "websocket": None,
        "ready": False,
        "stopped_after_ready": False,
        "padding": None,
        "word_confidence": None,
        "decoder_graphs": None,
        "biasing": None,
        "lines": [],
    }
    if not starts:
        return facts
    block = lines[starts[-1] :]
    facts["lines"] = [line for line in block if line.startswith(_BANNER)]
    ragged = decoder_graphs = biasing = False
    confidence = "off"
    for line in facts["lines"]:
        body = line[len(_BANNER) :]
        if m := re.match(r"checkpoint\s+(\S+)", body):
            facts["checkpoint"] = m.group(1)
        elif m := re.match(r"chunk mode\s+(\d+) ms \(att_context_size \[(\d+), (\d+)\]\)", body):
            facts["chunk_ms"] = int(m.group(1))
            facts["att_context"] = [int(m.group(2)), int(m.group(3))]
        elif m := re.match(r"admission\s+bucket (\d+) streams", body):
            facts["bucket"] = int(m.group(1))
        elif m := re.match(r"websocket\s+(\S+)", body):
            facts["websocket"] = m.group(1)
        elif re.match(r"padding\s+RAGGED", body):
            ragged = True
        elif m := re.match(r"confidence\s+word confidence ON, ([^:\s]+):", body):
            confidence = m.group(1)
        elif re.match(r"decoder\s+CUDA graphs ON", body):
            decoder_graphs = True
        elif re.match(r"biasing\s+ON", body):
            biasing = True
        elif body == "ready":
            facts["ready"] = True
        elif body == "stopped" and facts["ready"]:
            facts["stopped_after_ready"] = True
    complete = all(facts[k] is not None for k in ("checkpoint", "chunk_ms", "bucket", "websocket"))
    if complete and facts["ready"]:
        facts["padding"] = "ragged" if ragged else "fixed"
        facts["word_confidence"] = confidence
        facts["decoder_graphs"] = decoder_graphs
        facts["biasing"] = biasing
    return facts


def banner_refusals(facts: Mapping[str, Any], *, endpoint: str, declared: Declared) -> list[str]:
    """The last banner in the server log against the declaration."""
    d = declared
    if facts.get("padding") is None:
        return ["the server log holds no complete startup banner ending in 'ready'"]
    found: list[str] = []
    if facts.get("stopped_after_ready"):
        found.append("the server log's last server has stopped")
    if facts["padding"] != d.padding:
        found.append(
            f"the server's banner says padding {facts['padding']}, the run declares {d.padding}"
        )
    if facts.get("word_confidence") != d.word_confidence:
        found.append(
            f"the server's banner says word confidence {facts.get('word_confidence')}, the run "
            f"declares {d.word_confidence}"
        )
    if facts.get("decoder_graphs") != d.decoder_graphs:
        found.append(
            f"the server's banner says decoder graphs {facts.get('decoder_graphs')}, the run "
            f"declares {d.decoder_graphs}"
        )
    if facts.get("biasing"):
        found.append("the server's banner says biasing ON: a separate arm")
    if facts.get("checkpoint") != d.model:
        found.append(f"the banner names checkpoint {facts.get('checkpoint')!r}, not {d.model!r}")
    if facts.get("chunk_ms") != d.chunk_ms:
        found.append(f"the banner names {facts.get('chunk_ms')} ms chunks, not {d.chunk_ms}")
    if facts.get("bucket") != d.bucket:
        found.append(f"the banner names bucket {facts.get('bucket')}, not {d.bucket}")
    if facts.get("att_context") != d.att_context:
        found.append(
            f"the banner names att_context {facts.get('att_context')}, the run declares "
            f"{d.att_context}"
        )
    listening = urlparse(str(facts.get("websocket") or "")).port
    if listening != urlparse(endpoint).port:
        found.append(
            f"the banner's server listens on port {listening}, the endpoint is {endpoint}: the "
            "log is from another server"
        )
    return found


def read_server_log(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}", "banner": None}
    return {
        "path": str(path),
        "error": None,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "banner": banner_facts(raw.decode("utf-8", errors="replace")),
    }


def installed_matmul_default() -> str | None:
    """The matmul precision ``NeMoPipelineSpec`` defaults to in the installed source; the CLI
    passes none (cli.py:337-350), so this is what a server built from this source runs."""
    from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec

    for spec_field in dataclasses.fields(NeMoPipelineSpec):
        if spec_field.name == "matmul_precision":
            return spec_field.default  # type: ignore[return-value]
    return None


def derived_att_context(model: str, chunk_ms: int) -> list[int] | None:
    """The attention context the server's own ``att_context_size`` gives this model and chunk:
    the declaration when ``--att-context`` is not given, never an observation."""
    from verbatim.config import ChunkMode as ServerChunk
    from verbatim.pipelines.nemo_runtime import att_context_size
    from verbatim.scheduler.graph_budget import ConfigError

    try:
        return list(att_context_size(model, ServerChunk(chunk_ms)))
    except (ConfigError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class HostReading:
    """What the host said at one moment."""

    gpu: dict[str, Any]
    process: dict[str, Any]
    server_log: dict[str, Any]
    revision: dict[str, Any]
    hub_dir: str
    hub_dir_source: str

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def observe_host(
    args: argparse.Namespace,
    *,
    runner: Runner,
    proc: ProcReader,
    port: int,
    hub: tuple[Path, str] | None = None,
) -> HostReading:
    """The card, the process on the endpoint's port, the server log and the model revision in
    the server's cache. ``hub`` fixes the cache and its source (the first reading's), so every
    later reading looks at the same one."""
    gpu = observe_gpu(runner, index=args.gpu_index, port=port)
    listeners = gpu.get("listener_pids") or []
    process = observe_process(proc, listeners[0] if len(listeners) == 1 else None)
    if hub is None:
        if process["hub_dir"] is not None:
            hub = (Path(process["hub_dir"]), HUB_FROM_SERVER_ENVIRON)
        elif args.hf_hub_cache is not None:
            hub = (args.hf_hub_cache, "--hf-hub-cache (the server's environment was not read)")
        else:
            hub = (default_hub_dir(), HUB_FROM_THIS_CLIENT)
    return HostReading(
        gpu=gpu,
        process=process,
        server_log=read_server_log(args.server_log),
        revision=observe_revision(hub[0], args.model),
        hub_dir=str(hub[0]),
        hub_dir_source=hub[1],
    )


def host_refusals(
    host: HostReading,
    args: argparse.Namespace,
    declared: Declared,
    *,
    port: int,
    device_name: str | None,
) -> list[str]:
    """Every check on the host at one moment."""
    found = gpu_refusals(host.gpu, expected_uuid=args.gpu_uuid, device_name=device_name)
    found += process_refusals(
        host.process,
        declared,
        gpu_index=args.gpu_index,
        gpu_uuid=args.gpu_uuid,
        port=port,
        hub_cache=args.hf_hub_cache,
    )
    log = host.server_log
    if log.get("error"):
        found.append(f"the server log could not be read: {log['error']}")
    else:
        found += banner_refusals(log["banner"], endpoint=args.endpoint, declared=declared)
    found += revision_refusals(host.revision, args.model_revision)
    return found


def host_continuity(earlier: HostReading, later: HostReading, stage: str) -> list[str]:
    """``later`` shows the same server process, started the same way, as ``earlier``."""
    found: list[str] = []
    a, b = earlier.gpu.get("listener_pids"), later.gpu.get("listener_pids")
    if a != b:
        found.append(f"the process listening on the port went from {a} to {b} {stage}: restarted")
    pa, pb = earlier.process, later.process
    if pa.get("cmdline") != pb.get("cmdline"):
        found.append(f"the server's command line changed {stage}: it restarted")
    if pa.get("environ_sha256") != pb.get("environ_sha256"):
        found.append(f"the server's environment changed {stage}: it restarted")
    la, lb = earlier.server_log.get("banner") or {}, later.server_log.get("banner") or {}
    if la.get("banners") != lb.get("banners"):
        found.append(
            f"the server log held {la.get('banners')} startup banner(s) and holds "
            f"{lb.get('banners')} {stage}: a server started again"
        )
    elif la.get("lines") != lb.get("lines"):
        found.append(f"the server log's last banner changed {stage}")
    return found


# --- provenance of this client ---


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def imported_files() -> dict[str, str | None]:
    """The file each of ``IMPORTED`` was loaded from in this process (resolved), or, for one
    not imported yet, the file an import would load; None when it cannot be found.
    ``importlib.util.find_spec`` gives both: an imported module's own spec, else a search."""
    found: dict[str, str | None] = {}
    for name in IMPORTED:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origin = None if spec is None else spec.origin
        found[name] = None if origin is None else str(Path(origin).resolve())
    return found


def outside_checkout(files: Mapping[str, str | None]) -> list[str]:
    """The names in ``files`` whose file is not inside ``REPO`` (or was not found)."""
    root = REPO.resolve()
    return sorted(
        name for name, path in files.items() if path is None or not Path(path).is_relative_to(root)
    )


def code_files() -> dict[str, Path]:
    """The files whose hashes name this client's code: this file and the session client's."""
    return {"probe_sha256": HERE, "bench_client_sha256": Path(bench_client.__file__)}


#: The SHA-256 of each of ``code_files`` when this module was imported: the code that runs.
LOADED_SHA256: dict[str, str | None] = {key: _sha256_file(p) for key, p in code_files().items()}


def client_provenance(file_sha256: Mapping[str, str | None] | None = None) -> dict[str, Any]:
    """This client's code: the git state of the checkout this file lives in, the files the
    rest of its code came from, and the hash of each of ``code_files``: as given
    (``LOADED_SHA256``, at the start of a run), else as the files are now. That state describes
    what ran only when every imported file lies inside the checkout; otherwise
    ``tracked_files_modified`` is null."""
    files = imported_files()
    outside = outside_checkout(files)
    tracked_changes = _git("status", "--porcelain", "--untracked-files=no")
    hashes = (
        {key: _sha256_file(path) for key, path in code_files().items()}
        if file_sha256 is None
        else dict(file_sha256)
    )
    return {
        "verbatim_commit": _git("rev-parse", "HEAD"),
        "tracked_files_modified": (
            None if outside or tracked_changes is None else bool(tracked_changes)
        ),
        "checkout": str(REPO),
        "imported_from": files,
        "code_outside_checkout": outside,
        **hashes,
    }


def provenance_changes(start: Mapping[str, Any], end: Mapping[str, Any]) -> list[str]:
    """Every ``PROVENANCE`` value that is not the same at the end of the run as at its start."""
    return [
        f"the client's code changed during the run: {key} {start.get(key)!r} at the start and "
        f"{end.get(key)!r} at the end; the record names the start, which may not be what ran"
        for key in PROVENANCE
        if start.get(key) != end.get(key)
    ]


# --- the record ---


def failed_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.FAILED{out.suffix or '.json'}")


@dataclass(frozen=True, slots=True)
class HostObservations:
    """What the host said before the corpus load, after it, and after the run."""

    first: HostReading
    before: HostReading
    after: HostReading
    matmul_default: str | None
    references: dict[str, Any]


def host_failures(
    host: HostObservations,
    args: argparse.Namespace,
    declared: Declared,
    *,
    port: int,
    device_name: str | None,
) -> list[str]:
    """Checks on the host after the run: every check the run was admitted on still holds, and
    the server is the same process, started the same way, as the one the run was stamped with."""
    failures = [
        f"after the run: {r}"
        for r in host_refusals(host.after, args, declared, port=port, device_name=device_name)
    ]
    failures += host_continuity(host.before, host.after, "during the run")
    return failures


def _observed_flag(host: HostReading, key: str) -> Any:
    flags = host.process.get("flags")
    return None if flags is None else flags.get(key)


def build_record(
    *,
    endpoint: str,
    args: argparse.Namespace,
    declared: Declared,
    corpus: Mapping[str, Any],
    recordings: Sequence[Recording],
    run: Capture,
    first: ServerReading,
    before: ServerReading,
    after: ServerReading,
    host: HostObservations,
    warnings_before: Sequence[str],
    started: str,
    finished: str,
    expected: int | None,
    fake: bool,
    client_start: Mapping[str, Any],
    client_end: Mapping[str, Any],
    server_code_from: str,
) -> dict[str, Any]:
    """The record. ``fake``: a test double stood in for the host, ``/proc`` or the corpus
    (``main``'s test seams); the record then says so first, and nothing downstream uses it.
    ``client_start`` and ``client_end``: ``client_provenance`` at the start of the run and at
    its end; the record carries the start, and any ``PROVENANCE`` change fails the run."""
    if before.facts is None or before.admission is None:
        raise ValueError("the run was stamped with a reading that has no /readyz or /admission")
    facts_before = before.facts
    port = urlparse(endpoint).port or 0
    entries = [
        judge(outcome, chunk_ms=args.chunk_ms, word_confidence_mode=declared.word_confidence)
        for outcome in run.outcomes
    ]
    failures = run_failures(entries, expected=expected)
    server_fail, warnings = server_failures(before, after, recordings=len(recordings), first=first)
    failures += server_fail
    device_after = None if after.facts is None else after.facts.device_name
    failures += host_failures(host, args, declared, port=port, device_name=device_after)
    failures += provenance_changes(client_start, client_end)
    warnings = [*warnings_before, *warnings]
    slips = [slip for outcome in run.outcomes for slip in outcome.result.pacing_slip_ms]
    stamp_before, stamp_after = load_stamp(before), load_stamp(after)
    wer = corpus_wer((e["reference"], e["text"]) for e in entries if e["reference"])
    counts = counts_of(entries)
    success = not failures
    order = "\n".join(recording.rid for recording in recordings)
    audio = "\n".join(entry["pcm_sha256"] for entry in entries)
    intervals = [outcome.in_flight_s for outcome in run.outcomes]
    histogram: dict[str, int] = {}
    for value in sorted(occupancy_at_first_audio(intervals)):
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    extra_word_keys = sorted({k for e in entries for k in e["unparsed_word_keys"]})
    extra_final_keys = sorted(
        {k for e in entries for f in e["final_frames"] for k in f["unparsed_keys"]}
    )
    banner = host.before.server_log.get("banner") or {}
    c3 = reported_c3(before)
    readings = {"first": host.first, "before": host.before, "after": host.after}
    record: dict[str, Any] = {"fake_pipeline": True} if fake else {}
    return record | {
        "record": RECORD,
        "success": success,
        "status": "complete" if success else "FAILED",
        "failures": failures,
        "warnings": warnings,
        "question": (
            "What does the running server itself answer, final text and word timings, for every "
            "LibriSpeech test-other recording, at one declared padding and concurrency?"
        ),
        "frozen_claim": f"not claimed by one capture: {FROZEN_CLAIM}",
        "not_a_row": (
            "FAKE: test doubles stood in for the host, /proc or the corpus (main's test seams); "
            "this capture measures nothing."
            if fake
            else "Exploratory capture through a running server; not a harness row."
        ),
        "server": {
            "url": endpoint,
            "model": {"reported_by_server": facts_before.model, "declared": args.model},
            "model_revision": {
                "observed_in_hf_cache": host.before.revision.get("refs_main"),
                "declared": args.model_revision,
                "hub_dir": host.before.hub_dir,
                "hub_dir_source": host.before.hub_dir_source,
                "first": host.first.revision,
                "before": host.before.revision,
                "after": host.after.revision,
                "note": "refs/main and snapshots/ of the hub cache the server's own environment "
                "resolves (hub_dir_source says when it could not be read); the server reports no "
                "revision on any endpoint",
            },
            "bucket": {
                "reported_by_server": before.admission.get("bucket"),
                "observed_in_cmdline": _observed_flag(host.before, "bucket"),
                "declared": args.bucket,
            },
            "padding": {
                "reported_by_server": None,
                "observed_in_cmdline": _observed_flag(host.before, "padding"),
                "observed_in_server_log": banner.get("padding"),
                "declared": args.padding,
                "note": "no endpoint reports the padding; it is observed in the server process's "
                "command line and in its startup banner, where fixed is the absence of the "
                "RAGGED line",
            },
            "execution": {
                "reported_by_server": facts_before.execution,
                "observed_in_cmdline": (
                    None
                    if _observed_flag(host.before, "eager") is None
                    else ("eager" if _observed_flag(host.before, "eager") else "graph path")
                ),
                "declared": args.execution,
            },
            "word_confidence": {
                "reported_by_server": c3["word_confidence"],
                "decoder_step_confidence": c3["decoder_step_confidence"],
                "observed_in_cmdline": _observed_flag(host.before, "word_confidence"),
                "observed_in_server_log": banner.get("word_confidence"),
                "declared": args.word_confidence,
                "note": "reported_by_server is /readyz word_confidence (configured); "
                "decoder_step_confidence is /readyz observed, read from the built decoding "
                "computer; null where the server does not report it",
            },
            "decoder_graphs": {
                "reported_by_server": c3["decoder_graphs"],
                "observed_in_cmdline": _observed_flag(host.before, "decoder_graphs"),
                "observed_in_server_log": banner.get("decoder_graphs"),
                "declared": declared.decoder_graphs,
            },
            "biasing": {
                "reported_by_server": facts_before.biasing,
                "observed_in_cmdline": _observed_flag(host.before, "biasing"),
                "observed_in_server_log": banner.get("biasing"),
                "declared": False,
            },
            "pipeline": {"reported_by_server": facts_before.pipeline, "declared": args.pipeline},
            "chunk_ms": {
                "reported_by_server": facts_before.chunk_ms,
                "declared": args.chunk_ms,
                "session_frames": sorted({str(e["session_chunk_ms"]) for e in entries}),
            },
            "att_context": {
                "observed_in_readyz": c3["att_context_size"],
                "derived_from_cmdline": _observed_flag(host.before, "att_context"),
                "observed_in_server_log": banner.get("att_context"),
                "declared": declared.att_context,
                "note": "observed_in_readyz is /readyz observed.att_context_size, read from the "
                "built encoder; the others are what the flags and the banner say. "
                "compare_captures.py compares observed_in_readyz only",
            },
            "code": {
                "reported_by_server": reported_code(before),
                "expected_verbatim_path": declared.server_verbatim_path,
                "expected_from": server_code_from,
                "note": "/readyz code (contract C7): the directories the server process "
                "imported verbatim and verbatim_bench from; the run was refused unless "
                "verbatim_path is the expected one, and it is part of the identity checked at "
                "every reading",
            },
            "dtype": {"reported_by_server": facts_before.precision, "declared": args.dtype},
            "matmul_precision": {
                "reported_by_server": None,
                "declared": args.matmul,
                "installed_default": host.matmul_default,
                "note": "the server does not report it; NeMoPipelineSpec.matmul_precision in the "
                "installed source, which the CLI does not override",
            },
            "gpu": {
                "expected_uuid": args.gpu_uuid,
                "index": args.gpu_index,
                **{name: reading.gpu for name, reading in readings.items()},
                "rule": "nvidia-smi names the expected UUID and the /readyz device, and the one "
                "process listening on the endpoint's port is the only process on the card",
            },
            "process": {name: reading.process for name, reading in readings.items()},
            "server_log": {name: reading.server_log for name, reading in readings.items()},
            "readyz_first": first.readyz,
            "readyz_before": facts_before.to_json_dict(),
            "readyz_before_raw": before.readyz,
            "readyz_after": None if after.facts is None else after.facts.to_json_dict(),
            "readyz_after_raw": after.readyz,
            "admission_before": dict(before.admission),
            "admission_after": None if after.admission is None else dict(after.admission),
            "metrics_before": {k: (before.metrics or {}).get(k) for k in METRIC_SERIES},
            "metrics_after": {k: (after.metrics or {}).get(k) for k in METRIC_SERIES},
            "tick_ids": {"first": _tick(first), "before": _tick(before), "after": _tick(after)},
            "load": {
                "before": stamp_before,
                "after": stamp_after,
                "admitted_during_run": _delta(stamp_before, stamp_after, "admitted_total"),
                "refused_during_run": _delta(stamp_before, stamp_after, "refused_total"),
                "ticks_during_run": _delta(stamp_before, stamp_after, "ticks_total"),
                "ticks_over_budget_during_run": _delta(
                    stamp_before, stamp_after, "ticks_over_budget_total"
                ),
                "ticks_late_during_run": _delta(stamp_before, stamp_after, "ticks_late_total"),
            },
            "sessions_this_run_opened": counts["sessions_acknowledged"],
            "invariance_class_on_session_frames": sorted(
                {str(e["invariance_class"]) for e in entries}
            ),
        },
        "client": {
            "path": "verbatim_bench.client.run_session, the session client "
            "verbatim_bench.invariance.run_level drives, with its arguments; admission by "
            "asyncio.Semaphore(concurrency) as run_level",
            "pacing": "real time: run_session sleeps to a wall-clock deadline per frame; the "
            "harness has no faster mode",
            "frame_ms": constants.FRAME_MS,
            "frame_jitter_ms": constants.FRAME_JITTER_MS,
            "seed": args.seed,
            "frame_seed": "seed * 100000 + n",
            "words": True,
            "interim_results": "the server's default; the client does not send the parameter",
            "lang": args.lang,
            "concurrency": {
                "configured": args.concurrency,
                "observed_peak_in_flight": peak_in_flight(intervals),
                "occupancy_at_first_audio": histogram,
                "sessions_timed": sum(1 for i in intervals if i is not None),
                "observed_how": "on the wire, on this client's monotonic clock: a session is in "
                "flight from its first audio frame sent to its last final received "
                "(recordings[id].in_flight_s); the peak is the most in flight at one instant",
                "configured_is": "how many sessions the client admits at once "
                "(asyncio.Semaphore); an upper bound it set, not an observation",
            },
            "pacing_slip_p99_ms": percentile(slips, 99) if slips else None,
            "missing_final_rules": [
                f"the last final's audio_s equals the audio sent, within {AUDIO_TOLERANCE_S:.7f} s",
                "every drop of the partial text from non-empty to empty has a final at the same "
                "audio_s",
                "a recording with audio received at least one partial",
            ],
            "python": sys.version.split()[0],
            "websockets": getattr(websockets, "__version__", None),
            **client_start,
            "provenance_at_end": {key: client_end.get(key) for key in PROVENANCE},
            "provenance_rule": "verbatim_commit and tracked_files_modified are git's word before "
            "the first reading of the server, probe_sha256 and bench_client_sha256 the files "
            "as imported; all four are read again after the run (provenance_at_end), and a "
            "change fails the run",
            # This client needs no GPU; what it was given, and whether anything imported torch.
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_imported": "torch" in sys.modules,
        },
        "corpus": {
            **dict(corpus),
            "expected": expected,
            "limit": args.limit,
            "loaded": len(recordings),
            "total_audio_s": round(sum(r.samples for r in recordings) / SAMPLE_RATE_HZ, 3),
            "order_sha256": hashlib.sha256(order.encode("utf-8")).hexdigest(),
            "audio_sha256": hashlib.sha256(audio.encode("utf-8")).hexdigest(),
            "checked_against": host.references,
        },
        "started": started,
        "finished": finished,
        "wall_clock_s": round(run.wall_clock_s, 3),
        "counts": counts,
        "missing": [
            {"id": e["id"], "n": e["n"], "problems": e["problems"]}
            for e in entries
            if e["problems"]
        ],
        "word_confidence": {
            "word_entries": sum(len(e["words"]) for e in entries),
            "with_confidence": counts["word_entries_with_confidence"],
            "confidence_key": CONFIDENCE_KEY,
            "unparsed_word_keys_seen": extra_word_keys,
            "final_frames_with_confidence": sum(
                1 for e in entries for f in e["final_frames"] if f["confidence_present"]
            ),
            "unparsed_final_keys_seen": extra_final_keys,
            "where": "a word's confidence, when the wire carried one, is the fourth element of "
            "its entry in recordings[id].words; three elements means none was sent",
            "note": "counted per word entry and per final frame as the wire carried them",
        },
        "finals_digest": finals_digest(
            FinalRecord(e["id"], e["text"], tuple(tuple(w[:3]) for w in e["words"]))
            for e in entries
        ),
        "finals_digest_note": "sha256 of verbatim_bench.canonical.canonicalise_finals over every "
        "recording, the invariance gate's digest, so a capture at another concurrency compares "
        "digest to digest",
        "reading": {
            "corpus_wer": wer.wer,
            "errors": wer.errors,
            "reference_words": wer.reference_words,
            "note": "pooled WER under verbatim_bench.wer.normalise against the lowercased "
            "references: a reading that the server recognises this audio at all, not a result "
            "and not a gate",
        },
        "recordings": {e["id"]: {k: v for k, v in e.items() if k != "id"} for e in entries},
    }


class RecordExists(FileExistsError):
    """The record's path came into being while the run went on; ``kept`` holds this run's."""

    def __init__(self, target: Path, kept: Path) -> None:
        super().__init__(f"{target} appeared during the run; this run's record is at {kept}")
        self.target, self.kept = target, kept


def write_record(record: Mapping[str, Any], out: Path) -> Path:
    """Write once: to ``out`` on success, else to its FAILED path, never over a file already
    there. ``main`` refuses an existing path before sending; a file that appears there during
    the run (another capture given the same ``--out``) is left as it is, and this run's record
    stays at a name of its own (``RecordExists``)."""
    target = out if record["success"] else failed_path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.link(tmp, target)  # atomic, and unlike os.replace it never replaces
    except FileExistsError:
        raise RecordExists(target, tmp) from None
    except OSError:
        # A filesystem without hard links: an exclusive create is not atomic, but it is still
        # never a replacement.
        try:
            with target.open("xb") as fh:
                fh.write(tmp.read_bytes())
        except FileExistsError:
            raise RecordExists(target, tmp) from None
    tmp.unlink()
    return target


# --- the command ---


def _att_context_arg(value: str) -> list[int]:
    parts = value.split(",")
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected LEFT,RIGHT, got {value!r}") from exc
    if len(numbers) != 2:
        raise argparse.ArgumentTypeError(f"expected LEFT,RIGHT, got {value!r}")
    return numbers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="server_frozen_answers",
        description="Stream every LibriSpeech test-other recording through a running Verbatim "
        "server and record its final text and word timings per recording id.",
    )
    add = parser.add_argument
    add("--endpoint", default=DEFAULT_ENDPOINT, help="the server's WebSocket URL")
    add("--out", required=True, type=Path, help="where the record goes on success")
    add(
        "--concurrency",
        type=int,
        required=True,
        help="streams in flight at once (the max level); must fit the bucket",
    )
    add(
        "--bucket",
        type=int,
        required=True,
        help="declared; must match /admission, the command line and the banner",
    )
    add(
        "--padding",
        choices=("fixed", "ragged"),
        required=True,
        help="declared; must match the server's command line and banner",
    )
    add(
        "--server-log",
        type=Path,
        required=True,
        help="the server's stdout: its startup banner is read before the load, after it and "
        "after the run",
    )
    add("--gpu-index", type=int, required=True, help="nvidia-smi index of the server's card")
    add(
        "--gpu-uuid",
        required=True,
        help="refused unless nvidia-smi -i --gpu-index is this card and the server on the port "
        "is the only process on it",
    )
    add(
        "--limit",
        type=int,
        default=None,
        help="only the first N recordings, for a smoke run; the record says so",
    )
    add("--model", default=DEFAULT_MODEL, help="declared; must match /readyz")
    add(
        "--model-revision",
        default=DEFAULT_REVISION,
        help="the pin; must be refs/main and the only snapshot in the server's HF cache",
    )
    add(
        "--hf-hub-cache",
        type=Path,
        default=None,
        help="the hub cache the server is expected to read: refused when the server's own "
        "environment resolves another; used, with a warning, only when that environment "
        "cannot be read",
    )
    add(
        "--server-checkout",
        type=Path,
        default=None,
        metavar="DIR",
        help="the checkout the server runs from: /readyz code.verbatim_path must be "
        "DIR/src/verbatim, resolved (default: this client's own checkout)",
    )
    add(
        "--execution",
        choices=EXECUTIONS,
        default=DEFAULT_EXECUTION,
        help="declared; must match /readyz (default %(default)s)",
    )
    add(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help="declared; must match /readyz and every session frame",
    )
    add("--dtype", default=DEFAULT_DTYPE, help="declared; must match /readyz")
    add("--pipeline", default=DEFAULT_PIPELINE, help="declared; must match /readyz")
    add(
        "--word-confidence",
        choices=WORD_CONFIDENCE_MODES,
        default=DEFAULT_WORD_CONFIDENCE,
        help="declared (verbatim serve --word-confidence); checked against /readyz, the "
        "command line, the banner and every word on the wire",
    )
    add(
        "--decoder-graphs",
        choices=("off", "on"),
        default="off",
        help="declared (verbatim serve --decoder-graphs); checked like --word-confidence",
    )
    add(
        "--att-context",
        type=_att_context_arg,
        default=None,
        metavar="LEFT,RIGHT",
        help="declared attention context (default: what the server's att_context_size derives "
        "for --model and --chunk-ms); must match /readyz observed, the flags and the banner",
    )
    add(
        "--matmul",
        default=DEFAULT_MATMUL,
        help="declared; must match the installed NeMoPipelineSpec default",
    )
    add(
        "--references-record",
        type=Path,
        default=None,
        help=f"the stock record whose 'references' fix corpus order and text (default: "
        f"${REFERENCES_ENV})",
    )
    add("--seed", type=int, default=DEFAULT_SEED, help="frame-jitter seed")
    add("--lang", default="en-US")
    return parser


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _refuse(reasons: Sequence[str]) -> int:
    for reason in reasons:
        _say(f"[refused] {reason}")
    return EXIT_REFUSED


def _warn(warnings: Sequence[str]) -> None:
    for warning in warnings:
        _say(f"!!! [WARNING] {warning}")


def main(
    argv: Sequence[str] | None = None,
    *,
    loader: Loader | None = None,
    runner: Runner | None = None,
    proc: ProcReader | None = None,
    expected_split: int = EXPECTED_RECORDINGS,
) -> int:
    """``loader``, ``runner``, ``proc`` and ``expected_split`` are test seams: the dataset, the
    host commands (nvidia-smi, ss), ``/proc`` and the size of the whole split. None is on the
    command line. A record taken with any of the first three given is stamped
    ``"fake_pipeline": true``, whatever they are."""
    args = _parser().parse_args(list(argv) if argv is not None else None)
    out: Path = args.out
    fake = any(seam is not None for seam in (loader, runner, proc))
    runner = runner or run_command
    proc = proc or read_proc
    if args.concurrency < 1:
        return _refuse([f"--concurrency must be >= 1, got {args.concurrency}"])
    if args.limit is not None and args.limit < 1:
        return _refuse([f"--limit must be >= 1, got {args.limit}"])
    references_record = args.references_record
    if references_record is None and os.environ.get(REFERENCES_ENV):
        references_record = Path(os.environ[REFERENCES_ENV])
    if references_record is None:
        return _refuse([f"no references record: pass --references-record or set {REFERENCES_ENV}"])
    port = urlparse(args.endpoint).port
    if port is None:
        return _refuse([f"the endpoint {args.endpoint} names no port"])
    for existing in (out, failed_path(out)):
        if existing.exists():
            return _refuse([f"{existing} exists; a record is written once, pick a new --out"])
    # The start of the run: git's word now, and the files as they were imported.
    client_start = client_provenance(LOADED_SHA256)
    server_verbatim_path, server_code_from = expected_verbatim_path(args.server_checkout)

    att = (
        args.att_context
        if args.att_context is not None
        else derived_att_context(args.model, args.chunk_ms)
    )
    declared = Declared(
        model=args.model,
        pipeline=args.pipeline,
        chunk_ms=args.chunk_ms,
        dtype=args.dtype,
        execution=args.execution,
        bucket=args.bucket,
        concurrency=args.concurrency,
        padding=args.padding,
        word_confidence=args.word_confidence,
        decoder_graphs=args.decoder_graphs == "on",
        att_context=att,
        server_verbatim_path=server_verbatim_path,
    )

    def device(reading: ServerReading) -> str | None:
        return None if reading.facts is None else reading.facts.device_name

    first = read_server(args.endpoint)
    found = refusals(first, declared)
    host_first = observe_host(args, runner=runner, proc=proc, port=port)
    found += host_refusals(host_first, args, declared, port=port, device_name=device(first))
    matmul_default = installed_matmul_default()
    if args.matmul != matmul_default:
        found.append(
            f"--matmul {args.matmul!r} and the installed NeMoPipelineSpec default is "
            f"{matmul_default!r}, which is what the server runs"
        )
    warnings = readyz_warnings(first) + process_warnings(
        host_first.process, hub_source=host_first.hub_dir_source
    )
    _warn(warnings)
    if found:
        return _refuse(found)

    try:
        references, references_sha = load_references(references_record)
    except (OSError, ValueError) as exc:
        return _refuse([f"the references record could not be read: {exc}"])

    _say(f"[data] loading {DATASET} {DATASET_CONFIG}/{DATASET_SPLIT}")
    recordings, corpus = (loader or load_librispeech_test_other)(args.limit)
    expected = expected_split if args.limit is None else args.limit
    found = corpus_refusals(
        recordings, corpus, expected=expected, expected_split=expected_split, references=references
    )
    if found:
        return _refuse(found)
    total_s = sum(r.samples for r in recordings) / SAMPLE_RATE_HZ
    _say(f"[data] {len(recordings)} recordings, {total_s:.1f} s of audio")

    # Everything again after the load, which can take minutes: this is the reading the run is
    # stamped with, and it must be the server first read.
    before = read_server(args.endpoint)
    found = refusals(before, declared)
    found += continuity_refusals(first, before, "while the corpus loaded")
    hub = (Path(host_first.hub_dir), host_first.hub_dir_source)
    host_before = observe_host(args, runner=runner, proc=proc, port=port, hub=hub)
    found += host_refusals(host_before, args, declared, port=port, device_name=device(before))
    found += host_continuity(host_first, host_before, "while the corpus loaded")
    if found:
        return _refuse([f"after the corpus load: {reason}" for reason in found])

    step = max(1, len(recordings) // 20)

    def progress(done: int, total: int, unanswered: int) -> None:
        if done % step == 0 or done == total:
            _say(f"[capture] {done}/{total} sessions ended, {unanswered} with a final missing")

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _say(f"[capture] {args.endpoint} {args.padding} at concurrency {args.concurrency}, real time")
    run = asyncio.run(
        capture(
            args.endpoint,
            recordings,
            chunk_ms=args.chunk_ms,
            concurrency=args.concurrency,
            seed=args.seed,
            lang=args.lang,
            progress=progress,
        )
    )
    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = read_server(args.endpoint)
    client_end = client_provenance()
    host = HostObservations(
        first=host_first,
        before=host_before,
        after=observe_host(args, runner=runner, proc=proc, port=port, hub=hub),
        matmul_default=matmul_default,
        references={
            "path": str(references_record),
            "sha256": references_sha,
            "references": len(references),
            "compared": len(recordings),
            "rule": "ids in order and reference text equal, for every loaded recording",
        },
    )
    record = build_record(
        endpoint=args.endpoint,
        args=args,
        declared=declared,
        corpus=corpus,
        recordings=recordings,
        run=run,
        first=first,
        before=before,
        after=after,
        host=host,
        warnings_before=warnings,
        started=started,
        finished=finished,
        expected=expected,
        fake=fake,
        client_start=client_start,
        client_end=client_end,
        server_code_from=server_code_from,
    )
    try:
        target = write_record(record, out)
    except RecordExists as exc:
        _warn(record["warnings"])
        for reason in record["failures"]:
            _say(f"*** {reason}")
        _say(f"*** {exc}: nothing was written over it")
        _say(f"FINAL: NOT WRITTEN to {exc.target}; the record is at {exc.kept}")
        return EXIT_FAILED
    counts = record["counts"]
    summary = (
        f"{counts['recordings']} recordings, {counts['recordings_sent']} sent, "
        f"{counts['recordings_with_terminal_final']} with a terminal final, "
        f"{counts['final_frames_received']} final frames"
    )
    _warn(record["warnings"])
    if record["success"]:
        _say(f"FINAL: complete: {summary}; digest {record['finals_digest'][:16]}; wrote {target}")
        return EXIT_OK
    for reason in record["failures"]:
        _say(f"*** {reason}")
    _say(f"FINAL: FAILED: {summary}; NOT written to {out}; the evidence is in {target}")
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
