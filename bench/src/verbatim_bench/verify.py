# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Torch-free recomputation of a published run artifact.

The summary in a results file is never trusted, it is **recomputed** from the raw
per-sample records in the same file, and a mismatch is a rejection.

Verification is not a security boundary: the keyed checksum deters casual editing
of a JSON file. It does not stop a determined forger, and nothing in a JSON file
could.

"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from verbatim_bench import constants
from verbatim_bench import schema as _schema
from verbatim_bench.canonical import verify_checksum
from verbatim_bench.ladder import RUNG_PASS_CRITERIA
from verbatim_bench.results import percentile

PERCENTILE_TOLERANCE_MS: Final = 0.1

_CORPUS_ID_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_INVARIANCE_HASHES: Final = ("hash_1", "hash_32a", "hash_32b", "hash_max")
_LATENCY_BLOCKS: Final = ("first_partial", "partial", "final")


class Level(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    level: Level
    code: str  # stable, greppable: "SCHEMA", "PERCENTILE_MISMATCH", "COUNTER_MISMATCH",
    # "ORDER", "INVARIANCE_INCONSISTENT", "CHECKSUM", "NOT_MIN_OF_THREE", ...
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class VerifyReport:
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        """True when no ERROR findings are present."""
        return all(finding.level is not Level.ERROR for finding in self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.level is Level.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.level is Level.WARNING)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _sessions_of(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sessions = doc.get("sessions", [])
    if not isinstance(sessions, list):
        return []
    return [entry for entry in sessions if isinstance(entry, Mapping)]


def _result_of(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    result = doc.get("result", {})
    return result if isinstance(result, Mapping) else {}


def _check_latency_block(
    name: str, block: Any, samples: list[float], findings: list[Finding]
) -> None:
    base = f"result.latency_ms.{name}"
    if not isinstance(block, Mapping):
        return
    expected_n = len(samples)
    actual_n = block.get("n")
    if actual_n != expected_n:
        findings.append(
            Finding(
                Level.ERROR,
                "N_MISMATCH",
                f"n is {actual_n!r} but sessions imply {expected_n} samples",
                f"{base}.n",
            )
        )
    for label, rank in (("p50", 50), ("p95", 95), ("p99", 99)):
        actual = block.get(label)
        if not _is_number(actual):
            continue
        expected = percentile(samples, rank)
        if abs(float(actual) - expected) > PERCENTILE_TOLERANCE_MS + 1e-9:
            findings.append(
                Finding(
                    Level.ERROR,
                    "PERCENTILE_MISMATCH",
                    f"{label} is {float(actual):.4f}ms but recomputed {expected:.4f}ms "
                    f"from {expected_n} samples (tolerance {PERCENTILE_TOLERANCE_MS}ms)",
                    f"{base}.{label}",
                )
            )
    numbers = [block.get(label) for label in ("p50", "p95", "p99")]
    if all(_is_number(v) for v in numbers):
        p50, p95, p99 = (float(v) for v in numbers)
        if not (p50 <= p95 <= p99):
            findings.append(
                Finding(
                    Level.ERROR,
                    "ORDER",
                    f"p50 <= p95 <= p99 violated: {p50!r} {p95!r} {p99!r}",
                    base,
                )
            )


def _check_pacing_slip(result: Mapping[str, Any], findings: list[Finding]) -> None:
    """Check pacing_slip_ms ordering only (p50 <= p95 <= max).

    The summary cannot be recomputed from per-session slip samples because
    sessions[] carries no slip samples and the writer emits none, so ordering
    is all this check can claim.
    """
    slip = result.get("pacing_slip_ms")
    if not isinstance(slip, Mapping):
        return
    values = [slip.get(label) for label in ("p50", "p95", "max")]
    if all(_is_number(v) for v in values):
        p50, p95, maximum = (float(v) for v in values)
        if not (p50 <= p95 <= maximum):
            findings.append(
                Finding(
                    Level.ERROR,
                    "ORDER",
                    f"p50 <= p95 <= max violated: {p50!r} {p95!r} {maximum!r}",
                    "result.pacing_slip_ms",
                )
            )


def _check_counters(
    doc: Mapping[str, Any], sessions: list[Mapping[str, Any]], findings: list[Finding]
) -> None:
    result = _result_of(doc)
    completed = sum(1 for s in sessions if s.get("error") is None)
    failed = sum(1 for s in sessions if s.get("error") is not None)
    final_count = sum(1 for s in sessions if s.get("final_ms") is not None)
    expected = {
        "sessions_started": len(sessions),
        "sessions_completed": completed,
        "sessions_failed": failed,
        "finals_received": final_count,
    }
    if all(
        isinstance(s.get("partials_received"), int)
        and not isinstance(s.get("partials_received"), bool)
        for s in sessions
    ):
        # Reconcile against the per-session message counts the client records,
        # not against len(partial_ms): the two coincide only for servers that
        # emit exactly one partial per chunk. Older documents without the field
        # cannot support the claim, so the check is skipped for them.
        expected["partials_received"] = sum(
            s["partials_received"] for s in sessions if isinstance(s, Mapping)
        )
    for key, want in expected.items():
        actual = result.get(key)
        if actual is None:
            continue
        if actual != want:
            findings.append(
                Finding(
                    Level.ERROR,
                    "COUNTER_MISMATCH",
                    f"{key} is {actual!r} but sessions imply {want}",
                    f"result.{key}",
                )
            )
    started = result.get("sessions_started")
    done = result.get("sessions_completed")
    bad = result.get("sessions_failed")
    if (
        _is_number(started)
        and _is_number(done)
        and _is_number(bad)
        and float(done) + float(bad) > float(started)
    ):
        findings.append(
            Finding(
                Level.ERROR,
                "COUNTER_MISMATCH",
                f"sessions_completed + sessions_failed ({done!r} + {bad!r}) "
                f"exceeds sessions_started ({started!r})",
                "result.sessions_completed",
            )
        )
    for key in (
        "sessions_started",
        "sessions_completed",
        "sessions_failed",
        "chunks_sent",
        "partials_received",
        "finals_received",
    ):
        value = result.get(key)
        if value is not None and _is_number(value) and float(value) < 0:
            findings.append(
                Finding(
                    Level.ERROR,
                    "COUNTER_MISMATCH",
                    f"{key} is negative: {value!r}",
                    f"result.{key}",
                )
            )


def _check_corpus_and_date(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    config = doc.get("config", {})
    corpus = config.get("corpus_id") if isinstance(config, Mapping) else None
    if isinstance(corpus, str) and _CORPUS_ID_RE.match(corpus) is None:
        findings.append(
            Finding(
                Level.ERROR,
                "CORPUS_ID",
                f"corpus_id {corpus!r} does not match ^sha256:[0-9a-f]{{64}}$",
                "config.corpus_id",
            )
        )
    date = doc.get("date")
    if isinstance(date, str):
        text = date.strip()
        candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            datetime.fromisoformat(candidate)
        except ValueError:
            findings.append(Finding(Level.ERROR, "DATE", f"date {date!r} is not ISO-8601", "date"))


def _check_invariance(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    result = _result_of(doc)
    if "invariance" not in result:
        return
    block = result.get("invariance")
    if block is None:
        return
    if not isinstance(block, Mapping):
        return
    present = {name: block.get(name) for name in _INVARIANCE_HASHES if block.get(name) is not None}
    malformed = False
    for name, value in present.items():
        if not isinstance(value, str) or _HEX64_RE.match(value) is None:
            findings.append(
                Finding(
                    Level.ERROR,
                    "INVARIANCE_INCONSISTENT",
                    f"{name} is not 64 lowercase hex characters: {value!r}",
                    f"result.invariance.{name}",
                )
            )
            malformed = True
            continue
    if not present or malformed:
        return
    equal = block.get("equal")
    distinct = sorted(set(present.values()))
    if equal is True and len(distinct) > 1:
        first = next(n for n in _INVARIANCE_HASHES if present.get(n) == distinct[0])
        second = next(n for n in _INVARIANCE_HASHES if present.get(n) == distinct[1])
        findings.append(
            Finding(
                Level.ERROR,
                "INVARIANCE_INCONSISTENT",
                f"equal is true but {first} != {second}",
                "result.invariance.equal",
            )
        )
    elif equal is False and len(distinct) == 1:
        findings.append(
            Finding(
                Level.ERROR,
                "INVARIANCE_INCONSISTENT",
                "equal is false but all hashes are identical",
                "result.invariance.equal",
            )
        )
    elif equal is None and len(present) >= 2:
        findings.append(
            Finding(
                Level.ERROR,
                "INVARIANCE_INCONSISTENT",
                "hashes are present but equal states no verdict",
                "result.invariance.equal",
            )
        )


def _check_honesty(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    config = doc.get("config", {})
    paced = config.get("paced_real_time") if isinstance(config, Mapping) else None
    if not isinstance(paced, bool):
        findings.append(
            Finding(
                Level.ERROR,
                "PACED_REAL_TIME",
                "config.paced_real_time must be present and boolean",
                "config.paced_real_time",
            )
        )
    result = _result_of(doc)
    runs = result.get("streams_runs")
    if runs is None:
        return
    if not isinstance(runs, list) or len(runs) != 3:
        findings.append(
            Finding(
                Level.ERROR,
                "NOT_MIN_OF_THREE",
                f"streams_runs must hold exactly three entries, got {runs!r}",
                "result.streams_runs",
            )
        )
        return
    streams = result.get("streams")
    # Schema errors were only recorded, so the list can hold anything and
    # min() must never see a non-integer. A bare None default cannot mark
    # "not found" because None itself may be the offender.
    _clean = object()
    offender = next(
        (entry for entry in runs if not isinstance(entry, int) or isinstance(entry, bool)),
        _clean,
    )
    if offender is not _clean:
        findings.append(
            Finding(
                Level.ERROR,
                "NOT_MIN_OF_THREE",
                f"streams_runs holds a non-integer entry: {offender!r}",
                "result.streams_runs",
            )
        )
        return
    if streams != min(runs):
        findings.append(
            Finding(
                Level.ERROR,
                "NOT_MIN_OF_THREE",
                f"streams is {streams!r} but min(streams_runs) is {min(runs)!r}: "
                "min-of-three, never best-of-three",
                "result.streams",
            )
        )


def _check_checksum(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    if "checksum" not in doc:
        findings.append(
            Finding(
                Level.WARNING,
                "CHECKSUM",
                "no checksum block: a hand-built fixture is legitimate, "
                "a hand-edited row is what this catches",
                "checksum",
            )
        )
        return
    if not verify_checksum(doc):
        findings.append(
            Finding(
                Level.ERROR,
                "CHECKSUM",
                "checksum does not recompute: the file was edited after stamping",
                "checksum",
            )
        )


def verify_document(doc: Mapping[str, Any]) -> VerifyReport:
    """Recompute every checkable claim in `doc` and return all findings at once.

    Dispatches on the document's `schema` field: `vb-results/1` keeps every
    check it ever had, unchanged; `vb-results/2` runs those plus the
    environment, ladder and comparability checks below; `vb-results/3` runs the
    same set against its own schema, with the ladder checks reading the rung's
    `criteria_evaluated`.
    """
    schema_id = doc.get("schema", _schema.SCHEMA_ID)
    if schema_id in (_schema.SCHEMA_ID_V2, _schema.SCHEMA_ID_V3):
        return _verify_row_document(doc, str(schema_id))
    findings: list[Finding] = []
    for error in _schema.validate(doc):
        findings.append(Finding(Level.ERROR, "SCHEMA", error.message, error.path))
    sessions = _sessions_of(doc)
    result = _result_of(doc)
    latency = result.get("latency_ms", {})
    if isinstance(latency, Mapping):
        firsts = [s["first_partial_ms"] for s in sessions if _is_number(s.get("first_partial_ms"))]
        partials = [
            v
            for s in sessions
            if isinstance(s.get("partial_ms"), list)
            for v in s["partial_ms"]
            if _is_number(v)
        ]
        finals = [s["final_ms"] for s in sessions if _is_number(s.get("final_ms"))]
        for name, samples in (
            ("first_partial", [float(v) for v in firsts]),
            ("partial", [float(v) for v in partials]),
            ("final", [float(v) for v in finals]),
        ):
            _check_latency_block(name, latency.get(name), samples, findings)
    _check_pacing_slip(result, findings)
    _check_counters(doc, sessions, findings)
    _check_corpus_and_date(doc, findings)
    _check_invariance(doc, findings)
    _check_honesty(doc, findings)
    _check_checksum(doc, findings)
    return VerifyReport(findings=tuple(findings))


def _ladder_of(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ladder = doc.get("ladder", [])
    if not isinstance(ladder, list):
        return []
    return [entry for entry in ladder if isinstance(entry, Mapping)]


def _contains_notes_key(value: Any) -> list[str]:
    hits: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == "notes":
                    hits.append(child_path)
                _walk(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                _walk(child, f"{path}[{index}]")

    _walk(value, "")
    return hits


def _check_row_schema(doc: Mapping[str, Any], findings: list[Finding], schema_id: str) -> None:
    try:
        schema = _schema.load_schema(schema_id)
    except FileNotFoundError:
        return
    for error in _schema.validate(doc, schema):
        findings.append(Finding(Level.ERROR, "SCHEMA", error.message, error.path))
    for path in _contains_notes_key(doc):
        findings.append(Finding(Level.ERROR, "SCHEMA", "notes key is never a schema field", path))


def _check_v2_env(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    gpu = doc.get("gpu")
    if gpu is None:
        findings.append(
            Finding(Level.ERROR, "ENV_INCOMPLETE", "gpu block is null: NVML only", "gpu")
        )
        return
    if not isinstance(gpu, Mapping):
        findings.append(Finding(Level.ERROR, "ENV_INCOMPLETE", "gpu block is not an object", "gpu"))
        return
    name = gpu.get("name")
    sm = gpu.get("sm")
    if (
        isinstance(name, str)
        and isinstance(sm, str)
        and name == constants.VERDICT_DIE
        and sm != constants.VERDICT_SM
    ):
        findings.append(
            Finding(
                Level.ERROR,
                "SM_DIE_MISMATCH",
                f"die {name!r} with SM {sm!r} disagrees with {constants.VERDICT_SM!r}",
                "gpu.sm",
            )
        )
    reasons = gpu.get("throttle_reasons")
    if isinstance(reasons, Mapping):
        total = 0
        for value in reasons.values():
            if isinstance(value, bool | int | float):
                total += int(value)
        if total > constants.NVML_THROTTLE_EVENTS_MAX:
            findings.append(
                Finding(
                    Level.ERROR,
                    "THROTTLE_NONZERO",
                    f"throttle reason counters sum to {total}",
                    "gpu.throttle_reasons",
                )
            )
    foreign = gpu.get("foreign_processes_count")
    pids = gpu.get("compute_process_pids")
    foreign_count = 0
    if isinstance(foreign, int) and not isinstance(foreign, bool):
        foreign_count = foreign
    elif isinstance(pids, list):
        foreign_count = len(pids)
    if foreign_count > constants.FOREIGN_GPU_PROCESSES_MAX:
        findings.append(
            Finding(
                Level.ERROR,
                "FOREIGN_GPU_PROCESS",
                f"{foreign_count} foreign compute processes on the target GPU",
                "gpu.foreign_processes_count",
            )
        )
    if gpu.get("persistence_mode") is False and constants.GPU_PERSISTENCE_MODE_REQUIRED:
        findings.append(Finding(Level.ERROR, "PERSISTENCE_OFF", "persistence mode is off", "gpu"))
    host = doc.get("host")
    if isinstance(host, Mapping):
        client_cpu = host.get("client_cpu_pct_of_cpuset")
        if _is_number(client_cpu):
            fraction = float(client_cpu) / 100.0 if float(client_cpu) > 1.0 else float(client_cpu)
            if fraction > constants.CLIENT_CPU_MAX_FRACTION_OF_CPUSET:
                findings.append(
                    Finding(
                        Level.ERROR,
                        "CLIENT_CPU_OVER_BUDGET",
                        f"client CPU {client_cpu!r} exceeds the frozen fraction",
                        "host.client_cpu_pct_of_cpuset",
                    )
                )


def _check_v2_floor_and_box(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    cal = doc.get("cal")
    floor = cal.get("null_floor_p95_ms") if isinstance(cal, Mapping) else None
    if floor is None:
        findings.append(Finding(Level.ERROR, "FLOOR_MISSING", "null floor was not taken", "cal"))
    box = doc.get("box")
    box_id_value = box.get("id") if isinstance(box, Mapping) else None
    if not isinstance(box_id_value, str) or not box_id_value:
        findings.append(Finding(Level.ERROR, "BOX_ID_MISSING", "box id is missing", "box.id"))


def _rung_criteria_evaluated(rung: Mapping[str, Any]) -> set[str]:
    listed = rung.get("criteria_evaluated")
    if not isinstance(listed, list):
        return set()
    return {value for value in listed if isinstance(value, str)}


def _check_rung_criteria(ladder: list[Mapping[str, Any]], findings: list[Finding]) -> None:
    """Check each rung against what it says it evaluated.

    A rung may report a pass only for criteria it established, may not label a failure
    with a criterion it never evaluated, and is not required to name a failing criterion
    when the reason it did not pass is that the evaluation was incomplete.
    """
    required = {criterion.value for criterion in RUNG_PASS_CRITERIA}
    for rung in ladder:
        evaluated = _rung_criteria_evaluated(rung)
        unevaluated = sorted(required - evaluated)
        if rung.get("passed") is True and unevaluated:
            findings.append(
                Finding(
                    Level.ERROR,
                    "PASSED_WITHOUT_EVALUATING_EVERY_CRITERION",
                    f"rung n={rung.get('n')} reports a pass but never evaluated "
                    f"{', '.join(unevaluated)}",
                    "ladder",
                )
            )
        criterion = rung.get("first_failing_criterion")
        if isinstance(criterion, str) and criterion in required and criterion not in evaluated:
            findings.append(
                Finding(
                    Level.ERROR,
                    "CRITERION_NOT_EVALUATED",
                    f"rung n={rung.get('n')} fails on {criterion}, which it never evaluated",
                    "ladder",
                )
            )


def _check_ladder(doc: Mapping[str, Any], findings: list[Finding], *, criteria_aware: bool) -> None:
    ladder = _ladder_of(doc)
    if not ladder:
        return
    messages: list[str] = []
    by_seed: dict[Any, list[Mapping[str, Any]]] = {}
    for rung in ladder:
        if rung.get("valid") is True:
            by_seed.setdefault(rung.get("seed"), []).append(rung)
    for seed, entries in by_seed.items():
        ordered = sorted(entries, key=lambda rung: rung.get("n", 0))
        for lower in ordered:
            if lower.get("passed") is True:
                continue
            for higher in ordered:
                if higher.get("n", 0) > lower.get("n", 0) and higher.get("passed") is True:
                    messages.append(
                        f"seed {seed}: rung n={higher.get('n')} passed above "
                        f"failed n={lower.get('n')}"
                    )
    for message in messages:
        findings.append(Finding(Level.ERROR, "LADDER_NOT_MONOTONE", message, "ladder"))
    passing = [rung for rung in ladder if rung.get("passed") is True and rung.get("valid") is True]
    if passing:
        highest = max(rung.get("n", 0) for rung in passing)
        streams = (_result_of(doc)).get("streams")
        if isinstance(streams, int) and not isinstance(streams, bool) and streams != highest:
            findings.append(
                Finding(
                    Level.ERROR,
                    "S_NOT_HIGHEST_PASSING",
                    f"streams is {streams!r} but highest passing rung is {highest!r}",
                    "result.streams",
                )
            )
    if criteria_aware:
        _check_rung_criteria(ladder, findings)
    required = {criterion.value for criterion in RUNG_PASS_CRITERIA}
    for rung in ladder:
        if rung.get("passed") is not False or rung.get("first_failing_criterion") is not None:
            continue
        # A rung that did not pass because something it evaluated failed owes a criterion.
        # One that did not pass because the evaluation was incomplete does not, and saying
        # otherwise would name a criterion nobody measured.
        if criteria_aware and not required <= _rung_criteria_evaluated(rung):
            continue
        findings.append(
            Finding(
                Level.ERROR,
                "CRITERION_MISSING",
                f"rung n={rung.get('n')} failed without a labelled criterion",
                "ladder",
            )
        )
        break
    criteria = {rung.get("first_failing_criterion") for rung in ladder}
    if "latency" not in criteria:
        # A table with no latency failure anywhere is the signature of a
        # latency metric that cannot grow, which is the defect this harness
        # exists to remove.
        findings.append(
            Finding(
                Level.WARNING,
                "NO_LATENCY_CRITERION_ANYWHERE",
                "no ladder rung in this row ever failed on latency",
                "ladder",
            )
        )
    pairs = [(rung.get("n"), rung.get("seed")) for rung in ladder]
    if len(pairs) != len(set(pairs)):
        findings.append(
            Finding(
                Level.ERROR,
                "SEEDS_NOT_DISTINCT",
                "ladder repeats a rung with the same seed: repeats need distinct seeds",
                "ladder",
            )
        )
    config = doc.get("config")
    if isinstance(config, Mapping) and config.get("canonical_window") is False:
        findings.append(
            Finding(
                Level.ERROR,
                "NON_CANONICAL_WINDOW",
                "rung durations were overridden: row may exist but never publishes",
                "config.canonical_window",
            )
        )


def _check_v2_ratio(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    result = _result_of(doc)
    ceiling = doc.get("ceiling")
    if not isinstance(ceiling, Mapping):
        return
    c_runs = ceiling.get("c_runs")
    s_runs = result.get("streams_runs")
    stored_f = result.get("fraction_of_ceiling")
    if (
        s_runs is not None
        and c_runs is not None
        and stored_f is not None
        and isinstance(s_runs, list)
        and isinstance(c_runs, list)
        and _is_number(stored_f)
        and len(s_runs) == constants.S_REPEATS
        and len(c_runs) == constants.CEILING_MEDIAN_OF
    ):
        try:
            from verbatim_bench.killrule import fraction_of_ceiling

            recomputed = fraction_of_ceiling(s_runs, c_runs)
        except ValueError:
            recomputed = None
        if recomputed is not None and abs(recomputed - float(stored_f)) > 1e-12:
            findings.append(
                Finding(
                    Level.ERROR,
                    "F_MISMATCH",
                    f"fraction_of_ceiling is {stored_f!r} but recomputes to {recomputed!r}",
                    "result.fraction_of_ceiling",
                )
            )
    box = doc.get("box")
    box_id_value = box.get("id") if isinstance(box, Mapping) else None
    ceiling_box = ceiling.get("box_id")
    if (
        isinstance(box_id_value, str)
        and isinstance(ceiling_box, str)
        and box_id_value != ceiling_box
        and s_runs is not None
        and c_runs is not None
    ):
        findings.append(
            Finding(
                Level.ERROR,
                "CROSS_BOX_RATIO",
                "S and C are from different box sessions",
                "result.fraction_of_ceiling",
            )
        )
    arm = doc.get("arm")
    arm_dtype = arm.get("dtype_class") if isinstance(arm, Mapping) else None
    ceiling_dtype = ceiling.get("dtype_class")
    if (
        isinstance(arm_dtype, str)
        and isinstance(ceiling_dtype, str)
        and arm_dtype != ceiling_dtype
        and stored_f is not None
    ):
        findings.append(
            Finding(
                Level.ERROR,
                "DTYPE_CLASS_MISMATCH",
                f"arm dtype {arm_dtype!r} != ceiling dtype {ceiling_dtype!r}",
                "arm.dtype_class",
            )
        )
    if isinstance(ceiling.get("input_streams_per_slot"), int | float):
        per_slot = float(ceiling["input_streams_per_slot"])
        if per_slot < constants.CEILING_MIN_INPUT_STREAMS_PER_BATCH_SLOT:
            findings.append(
                Finding(
                    Level.ERROR,
                    "CEILING_UNDERFILLED",
                    f"ceiling has {per_slot!r} input streams per slot, "
                    "fewer than four per batch slot",
                    "ceiling",
                )
            )
    elif isinstance(ceiling.get("input_streams"), int | float) and isinstance(
        ceiling.get("batch_size"), int | float
    ):
        per_slot = float(ceiling["input_streams"]) / float(ceiling["batch_size"])
        if per_slot < constants.CEILING_MIN_INPUT_STREAMS_PER_BATCH_SLOT:
            findings.append(
                Finding(
                    Level.ERROR,
                    "CEILING_UNDERFILLED",
                    "ceiling input streams under-fill the batch",
                    "ceiling",
                )
            )


def _check_v2_arm(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    arm = doc.get("arm")
    if not isinstance(arm, Mapping):
        return
    status = arm.get("status")
    reason = arm.get("reason")
    if status in ("not_run", "invalid") and reason is None:
        findings.append(
            Finding(
                Level.ERROR,
                "ARM_STATUS_MISSING_REASON",
                f"arm status {status!r} requires a reason from the closed enum",
                "arm.reason",
            )
        )


def _check_v2_invariance(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    result = _result_of(doc)
    block = result.get("invariance")
    if not isinstance(block, Mapping):
        return
    present = {name: block.get(name) for name in _INVARIANCE_HASHES if block.get(name) is not None}
    if block.get("equal") is True and len(present) < 4:
        # `equal` may never be true with fewer than four hashes present. A row
        # that ran one concurrency would otherwise render as invariant.
        findings.append(
            Finding(
                Level.ERROR,
                "INVARIANCE_UNDER_FOUR_HASHES",
                f"equal is true with only {len(present)} hashes present: invariance needs all four",
                "result.invariance.equal",
            )
        )


def _check_v2_typed_fields(doc: Mapping[str, Any], findings: list[Finding]) -> None:
    declared = doc.get("declared")
    if isinstance(declared, Mapping):
        for key in declared:
            if not str(key).startswith("declared_"):
                findings.append(
                    Finding(
                        Level.ERROR,
                        "TYPED_FIELD_USED_FOR_COMPARABILITY",
                        f"declared key {key!r} must start with declared_",
                        "declared",
                    )
                )

    def _walk(node: Any, path: str, inside_declared: bool) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_inside = inside_declared or path == "declared" or key == "declared"
                if str(key).startswith("declared_") and not (
                    inside_declared or key == "declared" or path.startswith("declared")
                ):
                    findings.append(
                        Finding(
                            Level.ERROR,
                            "TYPED_FIELD_USED_FOR_COMPARABILITY",
                            f"declared key {key!r} outside the declared block",
                            f"{path}.{key}" if path else str(key),
                        )
                    )
                _walk(child, f"{path}.{key}" if path else str(key), child_inside)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                _walk(child, f"{path}[{index}]", inside_declared)

    _walk(doc, "", False)


def _verify_row_document(doc: Mapping[str, Any], schema_id: str) -> VerifyReport:
    """Run the row checks for `vb-results/2` and every version after it.

    The only difference between the versions is the schema the document is validated
    against and, from `vb-results/3` on, that the ladder checks read what each rung says
    it evaluated.
    """
    findings: list[Finding] = []
    _check_row_schema(doc, findings, schema_id)
    sessions = _sessions_of(doc)
    result = _result_of(doc)
    latency = result.get("latency_ms", {})
    if isinstance(latency, Mapping):
        firsts = [s["first_partial_ms"] for s in sessions if _is_number(s.get("first_partial_ms"))]
        partials = [
            v
            for s in sessions
            if isinstance(s.get("partial_ms"), list)
            for v in s["partial_ms"]
            if _is_number(v)
        ]
        finals = [s["final_ms"] for s in sessions if _is_number(s.get("final_ms"))]
        for name, samples in (
            ("first_partial", [float(v) for v in firsts]),
            ("partial", [float(v) for v in partials]),
            ("final", [float(v) for v in finals]),
        ):
            _check_latency_block(name, latency.get(name), samples, findings)
    _check_pacing_slip(result, findings)
    _check_counters(doc, sessions, findings)
    _check_corpus_and_date(doc, findings)
    _check_invariance(doc, findings)
    _check_v2_invariance(doc, findings)
    _check_honesty(doc, findings)
    _check_checksum(doc, findings)
    _check_v2_env(doc, findings)
    _check_v2_floor_and_box(doc, findings)
    _check_ladder(doc, findings, criteria_aware=schema_id != _schema.SCHEMA_ID_V2)
    _check_v2_ratio(doc, findings)
    _check_v2_arm(doc, findings)
    _check_v2_typed_fields(doc, findings)
    return VerifyReport(findings=tuple(findings))


def verify_table(docs: list[Mapping[str, Any]]) -> VerifyReport:
    """Verify several rows as one table: one harness tag, and latency somewhere."""
    findings: list[Finding] = []
    tags = {
        doc.get("harness", {}).get("tag") for doc in docs if isinstance(doc.get("harness"), Mapping)
    }
    tags.discard(None)
    if len(tags) > 1:
        findings.append(
            Finding(
                Level.ERROR,
                "HARNESS_TAG_MIXED",
                f"table mixes harness tags {sorted(tags)!r}: one tag per table",
                "harness.tag",
            )
        )
    latencies = 0
    for doc in docs:
        for rung in _ladder_of(doc):
            if rung.get("first_failing_criterion") == "latency":
                latencies += 1
    if docs and latencies == 0:
        findings.append(
            Finding(
                Level.WARNING,
                "NO_LATENCY_CRITERION_ANYWHERE",
                "no arm in this table ever failed on latency",
                "ladder",
            )
        )
    for doc in docs:
        findings.extend(verify_document(doc).findings)
    return VerifyReport(findings=tuple(findings))


def verify_file(path: Path) -> VerifyReport:
    """Verify a results file, or `<dir>/results.json` when given a directory."""
    target = Path(path)
    if target.is_dir():
        target = target / "results.json"
    doc = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return VerifyReport(
            findings=(Finding(Level.ERROR, "SCHEMA", "top-level JSON value must be an object", ""),)
        )
    return verify_document(doc)
