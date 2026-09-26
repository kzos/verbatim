#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare server captures from ``probes/server_frozen_answers.py``, recording by recording.

    python scripts/compare_captures.py A.json B.json [--control RAGGED.json]
    python scripts/compare_captures.py FIXED.json RAGGED.json [--repeat FIXED2.json]
        [--places-out PLACES.json] [--allow-declared-padding]
    ... [--allow-modified-tree] [--out REPORT.json]
    python scripts/compare_captures.py --confidence-equivalence OFF.json ON.json [--out REPORT.json]

For every recording both captures hold it reports identity on the TEXT channel (the final text,
byte for byte) and on the TIMING channel (every word's ``(word, start_ms, end_ms)`` exactly as the
wire carried it). A recording whose text is equal and whose timings are not is a timing-only
difference, never "identical". A per-word confidence, where a capture carries one as a word's
fourth element, is not part of either channel; differences in it are counted separately. Then it
prints one verdict line.

FROZEN is claimed only for ``FROZEN_RULE``: two fixed-padding captures whose clients OBSERVED a
different peak number of sessions in flight on the wire (``client.concurrency
.observed_peak_in_flight``: the probe counts it from each session's first audio frame sent to its
last final received) have identical answers on both channels AND a ragged-padding capture of the
same setting differs from them AND the padding of all three was OBSERVED (in the server
process's command line or its startup banner), not only declared, AND none of them comes from a
test double AND none was taken from a modified tree (``client.tracked_files_modified`` false in
each, and ``client.code_outside_checkout`` recorded as an empty list: no client code imported
from outside the checkout it names; a capture that does not record it, or records anything but
a list, is "not recorded", never clean) unless ``--allow-modified-tree`` is given, and then the
verdict says so. The configured concurrency is
what the client admitted at once, not what the wire carried: a capture of 8 recordings at
``--concurrency 32`` peaks at 8 at most, and a server that holds new sessions back peaks lower
still. The stamped peak is not taken on trust: it is recounted from the recordings' own
``in_flight_s`` by the probe's rule (``recounted_peak``), and a capture whose stamp is not that
count, or is above the recordings it holds or the concurrency it was configured with, is refused.
A match with no ragged capture, or with a ragged capture that also matches, says the instrument
never showed it can see a batch effect; a match at one observed peak, or where a peak is not
recorded, is repeatability, not batch independence; a padding nobody observed may not be what
the label says; a tree that was modified, whose state was not recorded, or whose client imported
code from elsewhere, may not be the commit the capture names. None of them is called frozen.
Every verdict line ``compare`` prints on two fixed captures, and every places ``repeat_verdict``
but the same-shape one, names both their configured concurrency and their observed peak (or
says the peak is not recorded). The same-shape string (``SAME_SHAPE_VERDICT``) is step1_places'
contract and names neither; the places' ``captures`` block carries both for each capture.

PLACES. Given one fixed and one ragged capture, the recordings where they differ are the places
at the serving setting. ``--places-out`` writes them in the server-captures record shape
``scripts/step1_places.py`` reads: ``source`` "scripts/compare_captures.py", ``timing_unit``
"ms", ``runs[dtype].arms["server_fixed_vs_ragged"]``, top-level ``references``, ``bucket`` (and
``batch``, the same value) and ``execution``. Side ``a`` is the fixed capture, the answer the
server serves; side ``b`` is the ragged capture, the DR-0014 control arm. ``repeat_verdict`` is
step1_places' same-shape string only when a second fixed capture (``--repeat``) at a DIFFERENT
observed peak in flight has identical answers and the padding of both was observed;
``positive_control`` starts "present" only when the ragged capture differs. No
``every_recording`` is written: the scorer takes a server capture's per-word confidence from
``step1_places.py --confidence``. Each capture's ``tracked_files_modified`` is carried (under
``captures``, with its ``code_outside_checkout``), and the top-level ``tracked_files_modified``
is true when any input's tree was modified, null when none was but one did not record it or
imported code from outside its checkout, and false only when every input recorded a clean tree
and its own code. The model, revision, chunk, dtype, bucket, execution and attention context
are the OBSERVED values (``OBSERVED``), and ``build_places`` refuses a capture that lacks one,
whose other stamps (declared, command line, banner) contradict it, or whose revision was read
from a cache other than the one the server's own environment names (``OBSERVED_WHERE``); it
refuses a capture whose stamped peak its recordings do not give (``peak_problems``), and any two
captures not taken by the same client code (``CLIENT_CODE``) or not served by the same server
code (``SERVER_CODE``), as ``compare`` does.

THE SERVER'S CODE (contract C7). Each capture records the ``/readyz`` ``code`` object the server
reported (``server.code.reported_by_server``): the directory the server process imported
``verbatim`` from (``verbatim_path``) and ``verbatim_bench`` (``bench_path``, null where it could
not). The probe refuses, before sending, a server whose ``verbatim_path`` is not the checkout it
expects. Two captures whose servers imported different code are refused here, as two taken by
different client code are: a ragged control served by other code could differ because the code
changed, not the padding.

CONFIDENCE EQUIVALENCE (contract C8). ``--confidence-equivalence OFF.json ON.json`` asks whether a
capture taken with word confidence on is the same answers as one taken with it off. It holds when
(i) each capture's digest, recomputed over its own recordings (the text and every word's
``(word, start_ms, end_ms)``; the confidence is not in it), is its stored ``finals_digest``;
(ii) the two stored digests are equal; and (iii) both hold the same recording ids in the same
order with the same audio (``pcm_sha256``) and references, and every setting compared below
matches but word confidence (and decoder step confidence, which is word confidence observed on
the built decoder, and differs with it by construction). Both captures must also be usable
(``capture_problems``), OFF's word confidence must be off and ON's must not. It prints one
verdict line, ``VERDICT: IDENTICAL IN TEXT AND TIMINGS`` (exit 0) or ``VERDICT: NOT SHOWN
IDENTICAL:`` and every difference (exit 1). ``confidence_equivalence`` is that rule as a
function; ``scripts/step1_places.py --confidence`` calls it rather than repeat the digest.

OUTPUTS are written once. ``--out`` and ``--places-out`` are refused (exit 2) before any input is
read when the path exists, resolves to one of the inputs, or both name one file; and a file that
appears at the path before the write is not written over (the write is a hard link from a file
of this process's own, which never replaces).

REFUSED (exit 2), before anything is compared, when any two of the inputs (every pair of the
two or three) differ in model, model revision, chunk, dtype, execution, matmul precision, bucket,
the attention context the server OBSERVED on its built encoder (``/readyz`` ``observed``; the
value a client derives from model and chunk is never compared, since it cannot differ between
two captures at one model and chunk), or the card; or in pipeline, biasing, word confidence,
decoder step confidence, decoder graphs, NeMo or torch version or device; or in the client code
that took them (``CLIENT_CODE``: the commit, this probe's file and the session client's file,
each recorded in both: a ragged control from other code could differ because the code changed,
not the padding); or in the server code that served them (``SERVER_CODE``, contract C7); or do
not hold the same recordings in the same order with the same audio and
references; or when a capture is incomplete, is stamped ``fake_pipeline`` (a test double served
or observed it; the probe stamps every capture taken through its test seams), its stored digest
is not the digest of its own recordings, its stamped peak in flight is not what its own
recordings give, its padding contradicts what the server showed, one of the ``OBSERVED``
settings was not observed (or, for the revision, not read from the server's own cache) or
another of its stamps contradicts the observation, or any two inputs are the same capture (the
same bytes, the same start and finish, or the same server process at the same starting tick), in
whatever order they are given. Places are refused when a capture's padding is only declared
(``--allow-declared-padding`` to accept): which side is served must not rest on a label nobody
checked.

Nothing here touches a GPU or a server.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verbatim_bench.canonical import FinalRecord, finals_digest

#: /5: the server's code is observed (``server.code``, contract C7) and the client's provenance is
#: read at the start and the end of the run; /4 counted the peak in flight on the wire; /3
#: stamped the client's admission count.
RECORD_KIND = "vb-server-frozen-answers/5"
#: The whole rule; the probe's ``frozen_claim`` repeats it word for word.
FROZEN_RULE = (
    "two fixed captures whose clients observed a different peak number of sessions in flight on "
    "the wire, with identical answers on both channels, AND a ragged capture of the same setting "
    "that differs, AND the padding of all three observed, AND none of them from a test double, "
    "AND none taken from a modified tree unless --allow-modified-tree is given"
)
#: The places record's contract, as scripts/step1_places.py states it.
SOURCE = "scripts/compare_captures.py"
ARM = "server_fixed_vs_ragged"
TIMING_UNIT = "ms"
SAME_SHAPE_VERDICT = "run-to-run identical in the same shape"
SIDES = {
    "a": "the served answer: the server capture with --padding fixed",
    "b": "a server capture of the same recordings with --padding ragged",
}
#: The settings two captures must share, each recorded in both, for their answers to be compared.
#: ``att_context`` is the one the server observed on its built encoder (``/readyz``).
REQUIRED = (
    "model",
    "revision",
    "chunk_ms",
    "dtype",
    "execution",
    "matmul",
    "bucket",
    "att_context",
)
#: Also compared; a pair that differs here changed more than one variable.
ALSO_COMPARED = (
    "pipeline",
    "biasing",
    "word_confidence",
    "decoder_step_confidence",
    "decoder_graphs",
    "nemo_version",
    "torch_version",
    "device_name",
)
#: How a padding is known when neither the command line nor the banner showed it.
DECLARED_ONLY = "declared only"
#: The settings read as OBSERVED: ``server.<key>.<observed>`` is the value compared and written
#: to the places, and each of the other stamps, where a capture records one, must equal it.
#: The probe refuses before sending when they disagree; this holds a capture to that here, so
#: no declared value is ever read in place of an observed one.
OBSERVED: dict[str, tuple[str, tuple[str, ...]]] = {
    "model": ("reported_by_server", ("declared",)),
    "model_revision": ("observed_in_hf_cache", ("declared",)),
    "chunk_ms": ("reported_by_server", ("declared",)),
    "dtype": ("reported_by_server", ("declared",)),
    "bucket": ("reported_by_server", ("observed_in_cmdline", "declared")),
    "execution": ("reported_by_server", ("observed_in_cmdline", "declared")),
    "att_context": (
        "observed_in_readyz",
        ("derived_from_cmdline", "observed_in_server_log", "declared"),
    ),
}
#: How the probe names a hub cache read from the server process's own environment. The probe's
#: ``HUB_FROM_SERVER_ENVIRON`` is this, word for word (a test holds the two together); any other
#: source (``--hf-hub-cache``, or the client's own cache) is a cache the server may not read.
SERVER_HUB_SOURCE = "the server's environment (/proc/<pid>/environ), by huggingface_hub's rule"
#: An ``OBSERVED`` setting that is an observation of the server only when read from the right
#: place: ``server.<key>.<field>`` must be the value given, or the setting was not observed.
OBSERVED_WHERE: dict[str, tuple[str, str]] = {
    "model_revision": ("hub_dir_source", SERVER_HUB_SOURCE),
}
#: The client code that took a capture (``client.<key>``): two captures compared, or read
#: together into places, must each record all three, and the same.
CLIENT_CODE = ("verbatim_commit", "probe_sha256", "bench_client_sha256")
#: The server code that served a capture (``server.code.reported_by_server.<key>``, contract C7):
#: the directory the server process imported each package from. Two captures compared, or read
#: together into places, must record the same; ``verbatim_path`` must be recorded, ``bench_path``
#: is null where the server could not import verbatim_bench.
SERVER_CODE = ("verbatim_path", "bench_path")
#: What ``--confidence-equivalence`` prints when contract C8 holds.
CONFIDENCE_EQUIVALENT = "IDENTICAL IN TEXT AND TIMINGS"
#: The settings a capture with word confidence on differs in from one with it off by
#: construction: the word confidence, and the decoder step confidence, which is the word
#: confidence observed on the built decoder. ``confidence_equivalence`` compares every other.
CONFIDENCE_SETTINGS = ("word_confidence", "decoder_step_confidence")

EXIT_OK = 0
#: ``--confidence-equivalence``: the two captures are not shown to be the same answers.
EXIT_DIFFERENT = 1
EXIT_REFUSED = 2


class Refused(ValueError):
    """The captures cannot be compared; ``reasons`` says why."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


@dataclass(frozen=True, slots=True)
class Loaded:
    path: str
    sha256: str
    record: dict[str, Any]


def load_capture(path: Path) -> Loaded:
    raw = Path(path).read_bytes()
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise Refused([f"{path}: not a JSON object"])
    return Loaded(str(path), hashlib.sha256(raw).hexdigest(), body)


def _get(record: Mapping[str, Any], *keys: str) -> Any:
    node: Any = record
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


#: Where a capture stamps a setting, most direct observation first.
_SOURCES = ("reported_by_server", "observed_in_cmdline", "observed_in_server_log", "declared")


def _observed_or_declared(record: Mapping[str, Any], name: str) -> Any:
    for source in _SOURCES:
        value = _get(record, "server", name, source)
        if value is not None:
            return value
    return None


def observed_value(record: Mapping[str, Any], key: str) -> Any:
    """The OBSERVED value of one of the ``OBSERVED`` settings, never another stamp."""
    return _get(record, "server", key, OBSERVED[key][0])


def observed_problems(loaded: Loaded) -> list[str]:
    """Why a capture's ``OBSERVED`` settings cannot be read as observed: one was not observed,
    was read somewhere other than ``OBSERVED_WHERE`` requires, or another of its stamps (the
    declaration, the command line, the banner) contradicts it."""
    found: list[str] = []
    for key, (source, others) in OBSERVED.items():
        seen = _get(loaded.record, "server", key, source)
        if seen is None:
            found.append(f"{loaded.path}: {key} was not observed (server.{key}.{source} is null)")
            continue
        if key in OBSERVED_WHERE:
            field, required = OBSERVED_WHERE[key]
            said = _get(loaded.record, "server", key, field)
            if said != required:
                found.append(
                    f"{loaded.path}: {key} was not observed of the server: server.{key}.{field} "
                    f"is {said!r}, not {required!r}"
                )
        for other in others:
            value = _get(loaded.record, "server", key, other)
            if value is not None and value != seen:
                found.append(
                    f"{loaded.path}: {key} {other} {value!r} contradicts the observed {source} "
                    f"{seen!r}"
                )
    return found


def settings_of(record: Mapping[str, Any]) -> dict[str, Any]:
    """The settings a capture ran at, as its stamps give them (observed where it observed)."""
    readyz = _get(record, "server", "readyz_before") or {}
    return {
        "model": observed_value(record, "model"),
        "revision": observed_value(record, "model_revision"),
        "chunk_ms": observed_value(record, "chunk_ms"),
        "dtype": observed_value(record, "dtype"),
        "execution": observed_value(record, "execution"),
        "matmul": _get(record, "server", "matmul_precision", "declared"),
        "bucket": observed_value(record, "bucket"),
        "pipeline": readyz.get("pipeline"),
        "biasing": readyz.get("biasing"),
        "word_confidence": _observed_or_declared(record, "word_confidence"),
        "decoder_step_confidence": _get(
            record, "server", "word_confidence", "decoder_step_confidence"
        ),
        "decoder_graphs": _observed_or_declared(record, "decoder_graphs"),
        "nemo_version": readyz.get("nemo_version"),
        "torch_version": readyz.get("torch_version"),
        "device_name": readyz.get("device_name"),
        "att_context": observed_value(record, "att_context"),
        "gpu_uuid": _get(record, "server", "gpu", "before", "uuid"),
    }


def padding_of(record: Mapping[str, Any]) -> tuple[str | None, str]:
    """(padding, how it is known): observed in the server process's command line or in its
    startup banner, or declared only. Raises when an observation contradicts the declaration,
    which ``capture_problems`` refuses before any verdict is read."""
    declared = _get(record, "server", "padding", "declared")
    cmdline = _get(record, "server", "padding", "observed_in_cmdline")
    log = _get(record, "server", "padding", "observed_in_server_log")
    seen = [value for value in (cmdline, log) if value is not None]
    if not seen:
        return declared, DECLARED_ONLY
    if any(value != declared for value in seen):
        # capture_problems refuses such a capture before anything asks for its padding.
        raise RuntimeError(
            f"padding declared {declared!r} and observed {seen}: a contradicted capture "
            "reached a verdict; capture_problems must refuse it first"
        )
    return declared, (
        "observed in the server's command line" if cmdline is not None else "observed in the log"
    )


def concurrency_of(record: Mapping[str, Any]) -> Any:
    """The concurrency the capture was CONFIGURED with: what was asked for, never evidence of
    the occupancy it reached."""
    return _get(record, "client", "concurrency", "configured")


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def peak_of(record: Mapping[str, Any]) -> int | None:
    """The most sessions the capture's client OBSERVED in flight on the wire at once (from a
    session's first audio frame to its last final), or None when the capture does not record
    a positive whole number for it. ``capture_problems`` refuses a recorded peak that is not
    what the capture's own recordings give (``peak_problems``) before any verdict reads it."""
    value = _get(record, "client", "concurrency", "observed_peak_in_flight")
    if not _whole(value) or value < 1:
        return None
    return value


def intervals_of(record: Mapping[str, Any]) -> tuple[list[tuple[float, float] | None], list[str]]:
    """Each recording's ``in_flight_s`` in order (None where it has none: the session sent no
    audio or got no final), and the ids of those whose stamp is neither null nor two numbers."""
    intervals: list[tuple[float, float] | None] = []
    malformed: list[str] = []
    for rid, entry in ordered(record):
        span = entry.get("in_flight_s")
        if span is None:
            intervals.append(None)
        elif isinstance(span, list) and len(span) == 2 and all(_number(v) for v in span):
            intervals.append((float(span[0]), float(span[1])))
        else:
            malformed.append(rid)
    return intervals, malformed


def recounted_peak(intervals: Sequence[tuple[float, float] | None]) -> int:
    """The probe's rule (``occupancy_at_first_audio`` and ``peak_in_flight`` in
    probes/server_frozen_answers.py), counted again here: at each session's first audio frame,
    itself and every other session whose first audio frame went out at or before that instant
    and whose last final arrived after it; the peak is the largest such count, 0 for none."""
    timed = [span for span in intervals if span is not None]
    return max(
        (
            1 + sum(1 for k, (s, e) in enumerate(timed) if k != n and s <= start < e)
            for n, (start, _) in enumerate(timed)
        ),
        default=0,
    )


def peak_problems(loaded: Loaded) -> list[str]:
    """Why a capture's recorded peak in flight cannot be read as observed: it is not the peak its
    own recordings' ``in_flight_s`` give (``recounted_peak``), it is above the recordings the
    capture holds or the concurrency it was configured with, or that concurrency is not a whole
    number. A peak that is not recorded is never read as an occupancy (``peak_of``), so it is
    not refused; it withholds FROZEN and the same-shape verdict."""
    record, name = loaded.record, loaded.path
    peak = peak_of(record)
    if peak is None:
        return []
    found: list[str] = []
    intervals, malformed = intervals_of(record)
    if malformed:
        found.append(
            f"{name}: in_flight_s of {len(malformed)} recording(s) is neither null nor two "
            f"numbers; first {malformed[0]}"
        )
    elif (recount := recounted_peak(intervals)) != peak:
        found.append(
            f"{name}: observed_peak_in_flight {peak} is not the {recount} its recordings' "
            "in_flight_s give"
        )
    held = len(record.get("recordings") or {})
    if peak > held:
        found.append(
            f"{name}: observed_peak_in_flight {peak} is above the {held} recording(s) it holds"
        )
    configured = concurrency_of(record)
    if not _whole(configured):
        found.append(f"{name}: the configured concurrency {configured!r} is not a whole number")
    elif peak > configured:
        found.append(
            f"{name}: observed_peak_in_flight {peak} is above the configured concurrency "
            f"{configured}"
        )
    return found


#: What ``tree_state`` says of a capture whose ``client.tracked_files_modified`` is not a bool.
TREE_NOT_RECORDED = "not recorded"
TREE_MODIFIED = "modified"
#: ... and of one whose client imported code from outside the checkout its tree state describes.
TREE_OUTSIDE = "code imported from outside its checkout"


def tree_state(record: Mapping[str, Any]) -> str | None:
    """None when the capture's client recorded a clean tree (``tracked_files_modified``
    false) and recorded that it imported no code from elsewhere (``code_outside_checkout`` an
    empty list), else why it cannot be taken for the commit it names: modified, code from
    outside, or not recorded. A ``code_outside_checkout`` that is absent, null or not a list
    says nothing about where the code came from, so it is "not recorded", never clean."""
    value = _get(record, "client", "tracked_files_modified")
    outside = _get(record, "client", "code_outside_checkout")
    if value is True:
        return TREE_MODIFIED
    if not isinstance(outside, list):
        return TREE_NOT_RECORDED
    if outside:
        return TREE_OUTSIDE
    return None if value is False else TREE_NOT_RECORDED


def unclean_trees(*captures: Loaded | None) -> list[str]:
    """``path (modified)`` or ``path (not recorded)`` for every capture not from a clean tree."""
    return [
        f"{x.path} ({state})"
        for x in captures
        if x is not None and (state := tree_state(x.record)) is not None
    ]


def is_fake(record: Mapping[str, Any]) -> bool:
    """Stamped by a test double: ``fake_pipeline`` present and anything but false, the rule
    ``scripts/step1_places.py`` applies to a record."""
    return record.get("fake_pipeline", False) is not False


@dataclass(frozen=True, slots=True)
class Occupancy:
    """Two captures' configured concurrency and OBSERVED peak in flight. Only the peaks decide
    whether the captures ran at two occupancies: a capture of fewer recordings than its
    concurrency never reaches it, and a server that holds sessions back never lets it."""

    configured: tuple[Any, Any]
    peaks: tuple[int | None, int | None]
    unrecorded: tuple[str, ...]

    @property
    def one(self) -> bool:
        """Both peaks recorded, and equal: one occupancy."""
        return not self.unrecorded and self.peaks[0] == self.peaks[1]

    def unrecorded_in(self) -> str:
        return f"the observed peak in flight is not recorded in {', '.join(self.unrecorded)}"

    def __str__(self) -> str:
        (cx, cy), (px, py) = self.configured, self.peaks
        return f"concurrency {cx} and {cy} (observed peak in flight {px} and {py})"


def occupancy(x: Loaded, y: Loaded) -> Occupancy:
    peaks = (peak_of(x.record), peak_of(y.record))
    return Occupancy(
        (concurrency_of(x.record), concurrency_of(y.record)),
        peaks,
        tuple(c.path for c, peak in zip((x, y), peaks, strict=True) if peak is None),
    )


def ordered(record: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    recordings = record.get("recordings") or {}
    return sorted(recordings.items(), key=lambda item: item[1].get("n", 0))


def triples(words: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """The timing channel: ``(word, start_ms, end_ms)``, without any confidence element."""
    return [list(w[:3]) for w in words]


def digest_of(record: Mapping[str, Any]) -> str:
    """The invariance gate's digest over a capture's own recordings."""
    return finals_digest(
        FinalRecord(rid, e["text"], tuple(tuple(w) for w in triples(e["words"])))
        for rid, e in ordered(record)
    )


def capture_problems(loaded: Loaded) -> list[str]:
    """Why one capture cannot be used at all."""
    record, name = loaded.record, loaded.path
    found: list[str] = []
    if record.get("record") != RECORD_KIND:
        found.append(f"{name}: record kind {record.get('record')!r}, expected {RECORD_KIND!r}")
        return found
    if is_fake(record):
        found.append(
            f"{name}: fake_pipeline is {record['fake_pipeline']!r}: test doubles took this "
            "capture, so it measures nothing"
        )
    if record.get("success") is not True or record.get("failures"):
        found.append(f"{name}: not a complete capture (success {record.get('success')!r})")
    if not record.get("recordings"):
        found.append(f"{name}: holds no recordings")
        return found
    if digest_of(record) != record.get("finals_digest"):
        found.append(f"{name}: the stored finals_digest is not the digest of its own recordings")
    mp = _get(record, "server", "matmul_precision") or {}
    if mp.get("declared") != mp.get("installed_default"):
        found.append(
            f"{name}: matmul declared {mp.get('declared')!r} against the installed default "
            f"{mp.get('installed_default')!r}"
        )
    declared = _get(record, "server", "padding", "declared")
    if declared not in ("fixed", "ragged"):
        found.append(f"{name}: padding {declared!r} is neither fixed nor ragged")
    for source, label in (
        ("observed_in_cmdline", "the server's command line"),
        ("observed_in_server_log", "the server log"),
    ):
        seen = _get(record, "server", "padding", source)
        if seen is not None and seen != declared:
            found.append(f"{name}: padding declared {declared!r} but {label} says {seen!r}")
    return found + observed_problems(loaded) + peak_problems(loaded)


def same_capture(x: Loaded, y: Loaded) -> str | None:
    """Why ``x`` and ``y`` are one capture, or None. Re-serialising a capture changes its bytes
    and can change any field; the start and finish, or the server process and the tick it
    started at, are one run's."""
    if x.sha256 == y.sha256:
        return "the same bytes"
    rx, ry = x.record, y.record
    if rx.get("started") is not None and (rx.get("started"), rx.get("finished")) == (
        ry.get("started"),
        ry.get("finished"),
    ):
        return "the same start and finish"
    kx, ky = (
        (
            _get(r, "server", "process", "before", "pid"),
            _get(r, "server", "readyz_before", "tick_id"),
        )
        for r in (rx, ry)
    )
    if None not in kx and kx == ky:
        return "the same server process at the same starting tick"
    return None


def code_problems(a: Loaded, b: Loaded) -> list[str]:
    """Why two captures cannot be taken as one client code's: a ``CLIENT_CODE`` stamp one of
    them does not record, or that differs between them."""
    found: list[str] = []
    for key in CLIENT_CODE:
        x, y = _get(a.record, "client", key), _get(b.record, "client", key)
        if x is None or y is None:
            found.append(f"client.{key} is not recorded in both captures ({x!r}, {y!r})")
        elif x != y:
            found.append(
                f"the captures were taken by different client code: client.{key} {x!r} and {y!r}"
            )
    return found


def server_code_problems(a: Loaded, b: Loaded) -> list[str]:
    """Why two captures cannot be taken as served by one server code (contract C7): the
    ``verbatim_path`` one of them does not record (then nothing else is compared: the code was
    not observed at all), or a ``SERVER_CODE`` path that differs."""
    paths = [
        {key: _get(c.record, "server", "code", "reported_by_server", key) for key in SERVER_CODE}
        for c in (a, b)
    ]
    x, y = (p["verbatim_path"] for p in paths)
    if x is None or y is None:
        return [
            f"server.code.verbatim_path is not recorded in both captures ({x!r}, {y!r}): which "
            "code the server ran was not observed"
        ]
    return [
        f"the servers ran different code: server.code.{key} {paths[0][key]!r} and {paths[1][key]!r}"
        for key in SERVER_CODE
        if paths[0][key] != paths[1][key]
    ]


def pair_problems(a: Loaded, b: Loaded, *, ignoring: Sequence[str] = ()) -> list[str]:
    """Why two usable captures' answers cannot be compared recording by recording.
    ``ignoring`` names settings of ``ALSO_COMPARED`` left out (``confidence_equivalence``'s)."""
    found = code_problems(a, b) + server_code_problems(a, b)
    sa, sb = settings_of(a.record), settings_of(b.record)
    for key in REQUIRED:
        if sa[key] is None or sb[key] is None:
            found.append(f"{key} is not recorded in both captures ({sa[key]!r}, {sb[key]!r})")
        elif sa[key] != sb[key]:
            found.append(f"the captures differ in {key}: {sa[key]!r} and {sb[key]!r}")
    for key in ALSO_COMPARED:
        if key not in ignoring and sa[key] != sb[key]:
            found.append(f"the captures differ in {key}: {sa[key]!r} and {sb[key]!r}")
    if sa["gpu_uuid"] is None or sb["gpu_uuid"] is None:
        found.append(
            f"a card is not recorded in both captures ({sa['gpu_uuid']!r}, {sb['gpu_uuid']!r})"
        )
    elif sa["gpu_uuid"] != sb["gpu_uuid"]:
        found.append(f"the captures ran on different cards: {sa['gpu_uuid']} and {sb['gpu_uuid']}")
    ra, rb = ordered(a.record), ordered(b.record)
    if [rid for rid, _ in ra] != [rid for rid, _ in rb]:
        found.append("the captures do not hold the same recordings in the same order")
        return found
    pairs = list(zip(ra, rb, strict=True))
    audio = [rid for (rid, x), (_, y) in pairs if x["pcm_sha256"] != y["pcm_sha256"]]
    if audio:
        found.append(f"{len(audio)} recording(s) were sent different audio; first {audio[0]}")
    refs = [rid for (rid, x), (_, y) in pairs if x["reference"] != y["reference"]]
    if refs:
        found.append(f"{len(refs)} recording(s) carry different references; first {refs[0]}")
    return found


def identity(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Per recording: is the text the same, are the word timings the same. Both channels."""
    per: dict[str, dict[str, bool]] = {}
    text_differs: list[str] = []
    timing_only: list[str] = []
    confidence_differs = 0
    for rid, x in ordered(a):
        y = b["recordings"][rid]
        same_text = x["text"] == y["text"]
        same_timings = triples(x["words"]) == triples(y["words"])
        per[rid] = {"text": same_text, "timings": same_timings}
        if not same_text:
            text_differs.append(rid)
        elif not same_timings:
            timing_only.append(rid)
        confidence_differs += x["words"] != y["words"] and same_timings
    identical = len(per) - len(text_differs) - len(timing_only)
    digests_match = a["finals_digest"] == b["finals_digest"]
    # The digest covers text and every word timing; the per-recording comparison must agree
    # with it, or one of the two is not comparing what it says.
    if digests_match != (identical == len(per)):
        raise RuntimeError(
            f"the digests {'match' if digests_match else 'differ'} but {identical} of {len(per)} "
            "recordings compare identical: the comparison and the digest disagree"
        )
    return {
        "recordings": len(per),
        "identical": identical,
        "text_differs": len(text_differs),
        "timing_only": len(timing_only),
        "text_differs_ids": text_differs,
        "timing_only_ids": timing_only,
        "confidence_only_differs": confidence_differs,
        "digests_match": digests_match,
        "per_recording": per,
    }


def _differ(same: Mapping[str, Any]) -> str:
    return f"{same['text_differs']} in text and {same['timing_only']} in word timings only"


def _answers_apart(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """Where two captures' answers differ, recording by recording, for a message: descriptive
    only, and tolerant of recordings one of them does not hold."""
    text: list[str] = []
    timing: list[str] = []
    others = b.get("recordings") or {}
    for rid, x in ordered(a):
        y = others.get(rid)
        if y is None:
            continue
        if x.get("text") != y.get("text"):
            text.append(rid)
        elif triples(x.get("words") or []) != triples(y.get("words") or []):
            timing.append(rid)
    first = (text + timing)[:1]
    return f"{len(text)} recording(s) differ in text and {len(timing)} in word timings only" + (
        f"; first {first[0]}" if first else ""
    )


def confidence_equivalence(off: Loaded, on: Loaded) -> list[str]:
    """Why ``on``, a capture taken with word confidence on, is not shown to be the same answers
    as ``off``, one taken with it off (contract C8); empty when it is. The digest compared is
    the invariance gate's, over the text and every word's ``(word, start_ms, end_ms)``; a word's
    confidence is not in it, so equal digests are the same answers in text and timings.

    (i) each capture's stored ``finals_digest`` is the digest of its own recordings, which
    ``capture_problems`` checks with every other reason a capture cannot be used; (ii) the two
    stored digests are equal; (iii) ``pair_problems`` finds nothing but word confidence: the
    same recording ids in the same order, the same audio and references, the same client and
    server code, and every other setting. ``off``'s word confidence must be off and ``on``'s
    must not be."""
    found = capture_problems(off) + capture_problems(on)
    if found:
        return found
    off_mode = settings_of(off.record)["word_confidence"]
    on_mode = settings_of(on.record)["word_confidence"]
    if off_mode != "off":
        found.append(
            f"{off.path}: word confidence {off_mode!r}, not 'off': the first capture is the one "
            "taken with word confidence off"
        )
    if on_mode is None or on_mode == "off":
        found.append(
            f"{on.path}: word confidence {on_mode!r}: the second capture is the one taken with "
            "word confidence on"
        )
    found += pair_problems(off, on, ignoring=CONFIDENCE_SETTINGS)
    digest_off, digest_on = off.record.get("finals_digest"), on.record.get("finals_digest")
    if digest_off != digest_on:
        found.append(
            f"the finals digests differ ({str(digest_off)[:16]} and {str(digest_on)[:16]}): "
            f"{_answers_apart(off.record, on.record)}"
        )
    return found


def declared_only(*captures: Loaded | None) -> list[str]:
    """The paths of the captures whose padding nobody observed."""
    return [x.path for x in captures if x is not None and padding_of(x.record)[1] == DECLARED_ONLY]


def frozen_reading(
    fixed: Loaded,
    second: Loaded | None,
    ragged: Loaded | None,
    fixed_vs_second: Mapping[str, Any] | None,
    fixed_vs_ragged: Mapping[str, Any] | None,
    *,
    allow_modified_tree: bool = False,
) -> tuple[bool, str]:
    """(frozen, the line that says why), by ``FROZEN_RULE``. ``allow_modified_tree`` accepts
    captures not taken from a clean tree, and the FROZEN line then names them."""
    if second is None or fixed_vs_second is None:
        return False, "FROZEN NOT ASSESSED: no second fixed capture was given"
    n = fixed_vs_second["recordings"]
    at = occupancy(fixed, second)
    if fixed_vs_second["identical"] != n:
        return False, (
            f"NOT FROZEN: of {n} recordings, {_differ(fixed_vs_second)} between fixed captures "
            f"at {at}"
        )
    if at.unrecorded:
        return False, (
            f"IDENTICAL, OCCUPANCY NOT OBSERVED: the fixed captures at {at} match on all {n} "
            f"recordings, but {at.unrecorded_in()}, so two occupancies are not shown; frozen not "
            "claimed"
        )
    if at.one:
        return False, (
            f"IDENTICAL AT ONE CONCURRENCY ({at.peaks[0]}): the fixed captures at {at} match on "
            f"all {n} recordings, but both peaked at {at.peaks[0]} in flight: run-to-run "
            "repeatability, not batch independence; frozen not claimed"
        )
    if ragged is None or fixed_vs_ragged is None:
        return False, (
            f"IDENTICAL, FROZEN NOT CLAIMED: the fixed captures at {at} match on all {n} "
            "recordings, and no ragged control was given"
        )
    if fixed_vs_ragged["identical"] == n:
        return False, (
            f"NOT SHOWN FROZEN: the fixed captures at {at} match on all {n} recordings, but the "
            "ragged capture matches them too, so this instrument has not shown it can see a "
            "batch effect"
        )
    unobserved = declared_only(fixed, second, ragged)
    if unobserved:
        return False, (
            f"FROZEN NOT CLAIMED: the fixed captures at {at} match on all {n} recordings and the "
            f"ragged capture differs, but the padding is only declared in {', '.join(unobserved)}"
        )
    unclean = unclean_trees(fixed, second, ragged)
    if unclean and not allow_modified_tree:
        return False, (
            f"FROZEN NOT CLAIMED: the fixed captures at {at} match on all {n} recordings and the "
            f"ragged capture differs, but not every capture was taken from a clean tree: "
            f"{', '.join(unclean)}; a capture from such a tree may not be the commit it names "
            "(--allow-modified-tree to accept)"
        )
    accepted = (
        f" (--allow-modified-tree: accepted although not taken from a clean tree: "
        f"{', '.join(unclean)})"
        if unclean
        else ""
    )
    return True, (
        f"FROZEN{accepted}: the fixed captures at {at} match on all {n} recordings (digest "
        f"{fixed.record['finals_digest'][:16]}), and the ragged capture differs on "
        f"{_differ(fixed_vs_ragged)}"
    )


def _describe(loaded: Loaded) -> dict[str, Any]:
    record = loaded.record
    padding, source = padding_of(record)
    return {
        "path": loaded.path,
        "sha256": loaded.sha256,
        "padding": padding,
        "padding_source": source,
        "concurrency": concurrency_of(record),
        "observed_peak_in_flight": _get(record, "client", "concurrency", "observed_peak_in_flight"),
        "tracked_files_modified": _get(record, "client", "tracked_files_modified"),
        "code_outside_checkout": _get(record, "client", "code_outside_checkout"),
        **{key: _get(record, "client", key) for key in CLIENT_CODE},
        "server_code": _get(record, "server", "code", "reported_by_server"),
        "seed": _get(record, "client", "seed"),
        "finals_digest": record.get("finals_digest"),
        "started": record.get("started"),
        "finished": record.get("finished"),
        "warnings": record.get("warnings"),
    }


def build_places(
    fixed: Loaded,
    ragged: Loaded,
    *,
    repeat: Loaded | None,
    fixed_vs_repeat: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The places between a fixed and a ragged capture, in step1_places' server-captures shape:
    side a = fixed (served), side b = ragged (control); timings in milliseconds.

    Raises ``Refused`` when an ``OBSERVED`` setting of any input was not observed (or not where
    ``OBSERVED_WHERE`` requires) or another of its stamps contradicts the observation, when a
    recorded peak in flight is not what the capture's recordings give, or when two inputs were
    not taken by the same client code or served by the same server code: the places carry
    observed values only, and the one commit they name is every input's. A capture a test
    double took makes the record ``"fake_pipeline": true``, which step1_places refuses
    (``compare`` refuses such a capture before it gets here)."""
    given = [fixed, ragged] + ([] if repeat is None else [repeat])
    problems = [p for x in given for p in observed_problems(x) + peak_problems(x)]
    problems += [
        f"{x.path} and {y.path}: {p}"
        for x, y in itertools.combinations(given, 2)
        for p in code_problems(x, y) + server_code_problems(x, y)
    ]
    if problems:
        raise Refused(problems)
    fs, rs = fixed.record, ragged.record
    settings = settings_of(fs)
    arm: dict[str, Any] = {
        "checked": 0,
        "text_divergent": 0,
        "timing_only_divergent": 0,
        "divergences": [],
        "transcripts": {},
    }
    for rid, x in ordered(fs):
        y = rs["recordings"][rid]
        a_text, b_text = x["text"], y["text"]
        a_time, b_time = triples(x["words"]), triples(y["words"])
        arm["checked"] += 1
        arm["transcripts"][rid] = {"a": a_text, "b": b_text} if a_text != b_text else {"a": a_text}
        if a_text != b_text:
            arm["text_divergent"] += 1
        elif a_time != b_time:
            arm["timing_only_divergent"] += 1
        if a_text != b_text or a_time != b_time:
            arm["divergences"].append(
                {
                    "n": x["n"],
                    "librispeech_id": rid,
                    "text_differs": a_text != b_text,
                    "a": a_text,
                    "b": b_text,
                    "a_timings": a_time,
                    "b_timings": b_time,
                }
            )
    diverged = arm["text_divergent"] + arm["timing_only_divergent"]
    if repeat is None or fixed_vs_repeat is None:
        repeat_verdict = "not measured: no second fixed capture was given (--repeat)"
    elif fixed_vs_repeat["identical"] != fixed_vs_repeat["recordings"]:
        repeat_verdict = (
            f"NOT run-to-run identical: two fixed captures differ on {_differ(fixed_vs_repeat)}, "
            f"at {occupancy(fixed, repeat)}"
        )
    elif (at := occupancy(fixed, repeat)).unrecorded:
        repeat_verdict = (
            f"identical, but {at.unrecorded_in()} (configured {at.configured[0]} and "
            f"{at.configured[1]}): not shown across occupancy"
        )
    elif at.one:
        repeat_verdict = (
            f"identical at one concurrency ({at.peaks[0]}) only: both captures peaked at "
            f"{at.peaks[0]} in flight (configured {at.configured[0]} and {at.configured[1]}); "
            "not shown across occupancy"
        )
    elif declared_only(fixed, repeat):
        repeat_verdict = (
            "identical at two concurrencies, but the padding is only declared in "
            f"{', '.join(declared_only(fixed, repeat))}, at {at}: not the same-shape verdict"
        )
    else:
        repeat_verdict = SAME_SHAPE_VERDICT
    started = sorted(filter(None, (x.record.get("started") for x in given)))
    finished = sorted(filter(None, (x.record.get("finished") for x in given)))
    states = [tree_state(x.record) for x in given]
    fakes = [x.path for x in given if is_fake(x.record)]
    record: dict[str, Any] = {"fake_pipeline": True} if fakes else {}
    return record | {
        "source": SOURCE,
        "timing_unit": TIMING_UNIT,
        "question": "Where do the server's fixed-padding answers (served) and its ragged-padding "
        "answers (the control arm) differ, at one serving setting?",
        "not_a_row": (
            f"FAKE: derived from captures test doubles took ({', '.join(fakes)}); it measures "
            "nothing."
            if fakes
            else "Derived from server captures by scripts/compare_captures.py; its settings are "
            "the captures' own observed stamps. Not a harness row."
        ),
        "model": settings["model"],
        "model_revision": settings["revision"],
        "machine": settings["device_name"],
        "capability": None,
        "torch": settings["torch_version"],
        "nemo": settings["nemo_version"],
        "nemo_commit": None,
        # Every input's commit: build_places refuses inputs that name different ones.
        "verbatim_commit": _get(fs, "client", "verbatim_commit"),
        # True when any input capture's tree was modified, None when none was but one did not
        # record it or imported code from outside its checkout, False only when every input
        # recorded a clean tree and its own code; each is under captures.
        "tracked_files_modified": (
            True
            if TREE_MODIFIED in states
            else None
            if any(state is not None for state in states)
            else False
        ),
        "att_context_size": settings["att_context"],
        "chunk_ms": settings["chunk_ms"],
        "bucket": settings["bucket"],
        "batch": settings["bucket"],
        "execution": settings["execution"],
        "matmul_precision": settings["matmul"],
        "word_confidence": settings["word_confidence"],
        "decoder_graphs": settings["decoder_graphs"],
        "gpu_uuid": settings["gpu_uuid"],
        "arms": [ARM],
        "sides": SIDES,
        "captures": {
            "a_fixed": _describe(fixed),
            "b_ragged": _describe(ragged),
            "repeat_fixed": None if repeat is None else _describe(repeat),
        },
        "compared_channels": ["final text", "word timings (word, start_ms, end_ms)"],
        "targets": arm["checked"],
        "started": started[0] if started else None,
        "finished": finished[-1] if finished else None,
        "references": {rid: x["reference"] for rid, x in ordered(fs)},
        "runs": {
            settings["dtype"]: {
                "finished": finished[-1] if finished else None,
                "repeat": None
                if fixed_vs_repeat is None
                else {k: fixed_vs_repeat[k] for k in ("recordings", "identical", "digests_match")},
                "repeat_verdict": repeat_verdict,
                "positive_control": (
                    f"present: ragged padding diverged from fixed on {diverged} recording(s)"
                    if diverged
                    else "ABSENT: the ragged capture is identical to the fixed one"
                ),
                "arms": {ARM: arm},
            }
        },
    }


def _write_json(body: Mapping[str, Any], path: Path) -> None:
    """Write once: never over a file already at ``path``, whatever put it there. ``main``
    refuses an existing path before reading any input; one that appears there before this
    write (another run given the same path) is left as it is, and ``Refused`` says so."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
    appeared = [f"{path} exists: nothing was written over it"]
    try:
        os.link(tmp, path)  # atomic, and unlike os.replace it never replaces
    except FileExistsError:
        raise Refused(appeared) from None
    except OSError:
        # A filesystem without hard links: an exclusive create is not atomic, and still never
        # a replacement.
        try:
            with path.open("xb") as handle:
                handle.write(tmp.read_bytes())
        except FileExistsError:
            raise Refused(appeared) from None
    finally:
        tmp.unlink()


def output_problems(outputs: Mapping[str, Path | None], inputs: Sequence[Path]) -> list[str]:
    """Why an output path may not be written: it is one of the inputs, it exists (an output is
    written once), or two outputs name one file. Paths are compared resolved, so a relative
    path, a ``..`` or a symbolic link that names an input is caught."""
    found: list[str] = []
    named = [(flag, path) for flag, path in outputs.items() if path is not None]
    for flag, path in named:
        target = path.resolve()
        same = [given for given in inputs if given.resolve() == target]
        if same:
            found.append(f"{flag} {path} is the input {same[0]}: an input is never written over")
        if os.path.lexists(path):
            found.append(f"{flag} {path} exists: an output is written once, pick a new path")
    for (fx, px), (fy, py) in itertools.combinations(named, 2):
        if px.resolve() == py.resolve():
            found.append(f"{fx} {px} and {fy} {py} are one file: each output needs its own path")
    return found


def compare(
    a: Loaded,
    b: Loaded,
    *,
    control: Loaded | None = None,
    repeat: Loaded | None = None,
    places_out: Path | None = None,
    allow_declared_padding: bool = False,
    allow_modified_tree: bool = False,
) -> dict[str, Any]:
    """The report for ``a`` against ``b``; raises ``Refused`` when they cannot be compared.

    ``control`` is a ragged capture for two fixed ones; ``repeat`` a second fixed capture for a
    fixed and a ragged one. Either way the three captures are two fixed and one ragged.
    ``allow_modified_tree`` lets FROZEN be claimed on captures not taken from a clean tree, and
    the verdict then names them."""
    if control is not None and repeat is not None:
        raise Refused(["--control and --repeat do not go together"])
    inputs = [("a", a), ("b", b)]
    inputs += [(name, x) for name, x in (("control", control), ("repeat", repeat)) if x is not None]
    reasons = [r for _, x in inputs for r in capture_problems(x)]
    if reasons:
        raise Refused(reasons)
    for (nx, x), (ny, y) in itertools.combinations(inputs, 2):
        why = same_capture(x, y)
        if why is not None:
            reasons.append(f"{nx} ({x.path}) and {ny} ({y.path}) are the same capture: {why}")
    for (nx, x), (ny, y) in itertools.combinations(inputs, 2):
        label = "" if (nx, ny) == ("a", "b") else f"{nx}/{ny}: "
        reasons += [f"{label}{r}" for r in pair_problems(x, y)]
    if reasons:
        raise Refused(reasons)
    (pa, source_a), (pb, source_b) = padding_of(a.record), padding_of(b.record)
    kinds = {pa, pb}
    if control is not None:
        if kinds != {"fixed"}:
            raise Refused(["--control goes with two fixed captures"])
        if padding_of(control.record)[0] != "ragged":
            raise Refused([f"the control is {padding_of(control.record)[0]}, not ragged"])
    if repeat is not None:
        if kinds != {"fixed", "ragged"}:
            raise Refused(["--repeat goes with one fixed and one ragged capture"])
        if padding_of(repeat.record)[0] != "fixed":
            raise Refused([f"the repeat is {padding_of(repeat.record)[0]}, not fixed"])
    if places_out is not None:
        if kinds != {"fixed", "ragged"}:
            raise Refused(["--places-out needs one fixed and one ragged capture"])
        unobserved = declared_only(a, b, repeat)
        if unobserved and not allow_declared_padding:
            raise Refused(
                [
                    f"padding is only declared in {', '.join(unobserved)}: which side is served "
                    "would rest on a label nobody checked (--allow-declared-padding to accept)"
                ]
            )

    same = identity(a.record, b.record)
    n = same["recordings"]
    second_identity = None
    places_written = None
    if kinds == {"fixed"}:
        second_identity = None if control is None else identity(a.record, control.record)
        frozen, verdict = frozen_reading(
            a, b, control, same, second_identity, allow_modified_tree=allow_modified_tree
        )
    elif kinds == {"fixed", "ragged"}:
        fixed, ragged = (a, b) if pa == "fixed" else (b, a)
        second_identity = None if repeat is None else identity(fixed.record, repeat.record)
        frozen, reading = frozen_reading(
            fixed, repeat, ragged, second_identity, same, allow_modified_tree=allow_modified_tree
        )
        if same["identical"] != n:
            verdict = (
                f"POSITIVE CONTROL PRESENT: ragged padding differs from fixed on {_differ(same)}, "
                f"of {n} recordings; those are the places. {reading}"
            )
        else:
            verdict = (
                f"POSITIVE CONTROL ABSENT: the ragged capture matches the fixed one on all {n} "
                f"recordings; no places. {reading}"
            )
        if places_out is not None:
            _write_json(
                build_places(fixed, ragged, repeat=repeat, fixed_vs_repeat=second_identity),
                places_out,
            )
            places_written = str(places_out)
    else:
        frozen = False
        verdict = f"RAGGED VS RAGGED: of {n} recordings, {_differ(same)}; frozen is not assessable"

    return {
        "question": "Are two server captures the same answers, recording by recording, on the "
        "text and on the word timings?",
        "captures": {"a": _describe(a), "b": _describe(b)},
        "control": None if control is None else _describe(control),
        "repeat": None if repeat is None else _describe(repeat),
        "settings": settings_of(a.record),
        "padding": {"a": [pa, source_a], "b": [pb, source_b]},
        "identity": same,
        "third_capture_identity": second_identity,
        "frozen": frozen,
        "frozen_rule": FROZEN_RULE,
        "trees_not_clean": unclean_trees(a, b, control, repeat),
        "allow_modified_tree": allow_modified_tree,
        "places_written": places_written,
        "verdict": verdict,
    }


def _refused(reasons: Sequence[str]) -> int:
    for reason in reasons:
        print(f"[refused] {reason}", file=sys.stderr)
    return EXIT_REFUSED


def _confidence_main(args: argparse.Namespace) -> int:
    """``--confidence-equivalence OFF ON``: one VERDICT line, exit 0 when contract C8 holds and
    1 when it does not; 2 when the captures cannot be read or the arguments do not go
    together."""
    off_path, on_path = args.confidence_equivalence
    others = [
        name
        for name, given in (
            ("A", args.a),
            ("B", args.b),
            ("--control", args.control),
            ("--repeat", args.repeat),
            ("--places-out", args.places_out),
            ("--allow-declared-padding", args.allow_declared_padding),
            ("--allow-modified-tree", args.allow_modified_tree),
        )
        if given
    ]
    if others:
        return _refused([f"--confidence-equivalence takes no {', '.join(others)}"])
    problems = output_problems({"--out": args.out}, [off_path, on_path])
    if problems:
        return _refused(problems)
    try:
        off, on = load_capture(off_path), load_capture(on_path)
    except (Refused, OSError, json.JSONDecodeError) as exc:
        return _refused(getattr(exc, "reasons", [str(exc)]))
    differences = confidence_equivalence(off, on)
    verdict = (
        f"NOT SHOWN IDENTICAL: {'; '.join(differences)}" if differences else CONFIDENCE_EQUIVALENT
    )
    if args.out is not None:
        report = {
            "question": "Is a capture taken with word confidence on the same answers as one "
            "taken with it off, in the final text and every word's (word, start_ms, end_ms)?",
            "rule": "contract C8: each capture's recomputed finals digest is its stored one, the "
            "two stored digests are equal, and the recordings, their order and audio, and every "
            "compared setting but word confidence match",
            "off": _describe(off),
            "on": _describe(on),
            "identical": not differences,
            "differences": differences,
            "verdict": verdict,
        }
        try:
            _write_json(report, args.out)
        except (Refused, OSError) as exc:
            return _refused(getattr(exc, "reasons", [str(exc)]))
    print(f"VERDICT: {verdict}")
    return EXIT_DIFFERENT if differences else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path, nargs="?", default=None)
    parser.add_argument("b", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--confidence-equivalence",
        type=Path,
        nargs=2,
        default=None,
        metavar=("OFF", "ON"),
        help="contract C8: is the capture with word confidence ON the same answers as OFF? One "
        "VERDICT line; exit 0 when it is, 1 when it is not",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=None,
        help="with two fixed captures: a ragged capture of the same setting",
    )
    parser.add_argument(
        "--repeat",
        type=Path,
        default=None,
        help="with a fixed and a ragged capture: a second fixed capture",
    )
    parser.add_argument(
        "--places-out",
        type=Path,
        default=None,
        help="with one fixed and one ragged capture: write the places here",
    )
    parser.add_argument("--allow-declared-padding", action="store_true")
    parser.add_argument(
        "--allow-modified-tree",
        action="store_true",
        help="claim FROZEN on captures not taken from a clean tree; the verdict names them",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write the full report here (a new path)"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.confidence_equivalence is not None:
        return _confidence_main(args)
    if args.a is None or args.b is None:
        return _refused(["two captures are needed: A and B (or --confidence-equivalence OFF ON)"])
    inputs = [path for path in (args.a, args.b, args.control, args.repeat) if path is not None]
    problems = output_problems({"--out": args.out, "--places-out": args.places_out}, inputs)
    if problems:
        return _refused(problems)
    try:
        report = compare(
            load_capture(args.a),
            load_capture(args.b),
            control=None if args.control is None else load_capture(args.control),
            repeat=None if args.repeat is None else load_capture(args.repeat),
            places_out=args.places_out,
            allow_declared_padding=args.allow_declared_padding,
            allow_modified_tree=args.allow_modified_tree,
        )
        if args.out is not None:
            _write_json(report, args.out)
    except (Refused, OSError, json.JSONDecodeError) as exc:
        return _refused(getattr(exc, "reasons", [str(exc)]))
    same = report["identity"]
    print(
        f"recordings {same['recordings']}: identical {same['identical']}, text differs "
        f"{same['text_differs']}, timing only {same['timing_only']}; digests "
        f"{'match' if same['digests_match'] else 'differ'}"
    )
    if report["places_written"]:
        print(f"places written to {report['places_written']}")
    print(f"VERDICT: {report['verdict']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
