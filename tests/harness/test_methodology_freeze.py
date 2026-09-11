# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Acceptance tests that keep the frozen method equal to its instrument."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from verbatim_bench import constants

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
METHODOLOGY = ROOT / constants.METHODOLOGY_PATH
REQUIRED_HEADINGS = [
    "## 1. The freeze and its proof",
    "## 2. Latency",
    "## 3. Constants",
    "## 4. The arm registry",
    "## 5. The kill arithmetic",
    "## 6. The ceiling",
    "## 7. Validity",
    "## 8. The host and the box",
    "## 9. Schema version and harness tag",
    "## 10. Comparability",
    "## 11. The constants test",
    "## 12. Refresh cadence and staleness",
    "## 13. The die the verdict is read on",
]


def _read_methodology() -> str:
    return METHODOLOGY.read_text(encoding="utf-8")


def _section(document: str, heading: str) -> str:
    start = document.index(heading) + len(heading)
    ends = [
        index
        for index in (
            document.find("\n## ", start),
            document.find(constants.FROZEN_BLOCK_MARKER, start),
        )
        if index != -1
    ]
    return document[start : min(ends)] if ends else document[start:]


def _frozen_block(document: str) -> dict[str, Any]:
    match = re.search(
        rf"{re.escape(constants.FROZEN_BLOCK_MARKER)}\n```json\n(.*?)\n```",
        document,
        flags=re.DOTALL,
    )
    assert match is not None
    value = json.loads(match.group(1))
    assert isinstance(value, dict)
    return value


def test_methodology_document_exists() -> None:
    assert METHODOLOGY.is_file()
    assert METHODOLOGY.read_text(encoding="utf-8").strip()


def test_document_has_every_required_section() -> None:
    headings = [line for line in _read_methodology().splitlines() if line.startswith("## ")]
    assert headings == REQUIRED_HEADINGS


def test_document_has_exactly_one_frozen_constants_block() -> None:
    document = _read_methodology()
    assert document.count(constants.FROZEN_BLOCK_MARKER) == 1
    assert len(re.findall(r"^```json$", document, flags=re.MULTILINE)) == 1
    marker_line = document.splitlines().index(constants.FROZEN_BLOCK_MARKER)
    assert document.splitlines()[marker_line + 1] == "```json"


def test_frozen_constants_block_parses_as_json() -> None:
    assert isinstance(_frozen_block(_read_methodology()), dict)


def test_document_constants_equal_the_code_constants() -> None:
    document_values = _frozen_block(_read_methodology())
    code_values = constants.frozen_constants()
    sentinel = object()
    differing = sorted(
        name
        for name in document_values.keys() | code_values.keys()
        if document_values.get(name, sentinel) != code_values.get(name, sentinel)
    )
    assert not differing, (
        f"frozen constants differ for keys {differing}; change the document block or constants.py"
    )


def test_no_code_constant_is_missing_from_the_document() -> None:
    missing = set(constants.frozen_constants()) - set(_frozen_block(_read_methodology()))
    assert not missing, (
        f"code constants missing from document: {sorted(missing)}; change document block"
    )


def test_no_document_constant_is_unknown_to_the_code() -> None:
    unknown = set(_frozen_block(_read_methodology())) - set(constants.frozen_constants())
    assert not unknown, (
        f"document constants unknown to code: {sorted(unknown)}; change document block"
    )


def test_unfrozen_thresholds_are_null_in_the_document_and_listed_by_name() -> None:
    document = _read_methodology()
    section = _section(document, "## 7. Validity")
    values = _frozen_block(document)
    for name in constants.UNFROZEN_THRESHOLDS:
        assert values[name] is None
        assert name in section


def test_one_primary_latency_definition() -> None:
    section = _section(_read_methodology(), "## 2. Latency")
    assert section.count("word_emission") == 1
    assert section.lower().count("primary") == 1


def test_deleted_latency_definition_is_named_as_deleted() -> None:
    section = _section(_read_methodology(), "## 2. Latency")
    assert constants.LATENCY_DELETED in section
    assert "deleted" in section.lower()


def test_no_competing_partial_latency_definition_survives() -> None:
    phrase = "first partial received after"
    source_root = ROOT / "bench" / "src" / "verbatim_bench"
    assert not any(
        phrase in path.read_text(encoding="utf-8")
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix == ".py"
    )

    # The superseded definition must not survive anywhere a reader could mistake it for current.
    # Documentation may name it only to say it was replaced, so a mention has to sit on a line that
    # also says so.
    for doc in (ROOT / "docs", ROOT / "benchmarks"):
        for path in doc.rglob("*.md"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if phrase in line:
                    assert "superseded" in line.lower() or "deleted" in line.lower(), (
                        f"{path}: the replaced definition appears without saying it was replaced"
                    )


def test_kill_thresholds_appear_in_the_same_sentence_as_the_die() -> None:
    section = _section(_read_methodology(), "## 13. The die the verdict is read on")
    sentences = re.split(r"(?<=[.!?])\s+", section)
    required_terms = (
        str(constants.KILL_RULE_1_THRESHOLD),
        str(constants.KILL_RULE_2_THRESHOLD),
        constants.VERDICT_DIE,
        f"sm_{constants.VERDICT_SM.replace('.', '')}",
    )
    assert any(all(term in sentence for term in required_terms) for sentence in sentences)


def test_bias_direction_is_stated() -> None:
    section = _section(_read_methodology(), "## 13. The die the verdict is read on").lower()
    assert "slower die lowers `c`, so `f` rises" in section


def test_unowned_dies_are_declared_not_measured() -> None:
    section = _section(_read_methodology(), "## 13. The die the verdict is read on")
    prose = section.split(constants.FROZEN_BLOCK_MARKER, 1)[0]
    for die in constants.NOT_MEASURED_DIES:
        line = next(line for line in prose.splitlines() if die in line)
        assert "not measured" in line.lower()


def test_arm_registry_lists_every_pre_declared_arm() -> None:
    section = _section(_read_methodology(), "## 4. The arm registry")
    for arm in ("(a)", "(b)", "(c)", "(d1)", "(d2)", "(d3)", "(e)", "(f)", "(g)", "(h)"):
        assert f"| {arm} |" in section


def test_every_arm_has_a_pre_declared_not_run_reason() -> None:
    section = _section(_read_methodology(), "## 4. The arm registry")
    lines = section.splitlines()
    header = next(line for line in lines if line.startswith("| Arm |"))
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    not_run_index = columns.index("not_run")
    rows = [line for line in lines if line.startswith("| (")]
    assert len(rows) == 10
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[not_run_index]


def test_rule_two_asymmetry_is_stated() -> None:
    section = _section(_read_methodology(), "## 5. The kill arithmetic").lower()
    assert "not evaluable" in section
    assert re.search(r"measured `s_d >= 0\.5 c` kills", section)


def test_no_rounding_is_stated_with_a_worked_boundary() -> None:
    section = _section(_read_methodology(), "## 5. The kill arithmetic").lower()
    assert "no rounding" in section
    assert "0.699" in section
    assert "exactly `f = 0.7`" in section
    assert "exactly `s_d = 0.5 c`" in section


def test_refresh_cadence_has_exactly_one_alternative() -> None:
    section = _section(_read_methodology(), "## 12. Refresh cadence and staleness")
    lower = section.lower()
    assert "alternative a" not in lower
    assert "alternative b" not in lower
    assert "withdrawn" not in lower
    assert len([paragraph for paragraph in section.split("\n\n") if paragraph.strip()]) == 1


def test_ceiling_section_requires_four_times_batch_input_streams() -> None:
    section = _section(_read_methodology(), "## 6. The ceiling").lower()
    multiplication_sign = "\N{MULTIPLICATION SIGN}"
    assert f"4 {multiplication_sign} b" in section
    assert "under-filled" in section
    assert "lowers `c`" in section
    assert "raises `f`" in section


def test_invalid_is_rerun_never_footnoted() -> None:
    section = _section(_read_methodology(), "## 7. Validity").lower()
    assert "an invalid run is re-run, never footnoted." in section


def test_harness_tag_rule_is_one_tag_per_table() -> None:
    section = _section(_read_methodology(), "## 9. Schema version and harness tag").lower()
    assert "one committed harness tag runs every arm" in section
    assert "mid-run" in section
    assert "re-run" in section and "every arm" in section


def test_methodology_is_referenced_by_the_benchmarks_readme() -> None:
    readme = ROOT / "benchmarks" / "README.md"
    assert "METHODOLOGY.md" in readme.read_text(encoding="utf-8")
    assert (readme.parent / "METHODOLOGY.md").is_file()


def test_document_contains_no_measured_number() -> None:
    prose = _read_methodology().split(constants.FROZEN_BLOCK_MARKER, 1)[0]
    lines = prose.splitlines()

    null_server_lines = [line for line in lines if "166" in line]
    assert len(null_server_lines) == 1
    assert "measured against a null server" in null_server_lines[0].lower()

    allowed_literals = {
        str(value)
        for value in constants.frozen_constants().values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    } | {"0.699"}
    for line in lines:
        for match in re.finditer(r"\b(?:F|S|S_d|C|p95)\s*=\s*`?(\d+(?:\.\d+)?)", line):
            literal = match.group(1)
            assert literal in allowed_literals, (
                f"unpermitted result literal {literal!r} in line: {line}"
            )

    vocabulary_matches = [
        line
        for line in lines
        if re.search(r"\b(?:sustained|achieved|observed|came in at)\b.*\d", line)
    ]
    assert not vocabulary_matches, f"unpermitted result vocabulary: {vocabulary_matches}"

    allowed = "measured against a null server"
    suspicious: list[str] = []
    for line in lines:
        lower = line.lower()
        if allowed in lower:
            continue
        if re.search(
            r"\b(?:measured|measurement)\b.*\b(?:was|were|recorded|reported)\b.*\d", lower
        ):
            suspicious.append(line)
    assert not suspicious, f"unpermitted measured number: {suspicious}"


def test_a_produced_run_stamps_the_frozen_schema_version() -> None:
    """The freeze binds the code, not only the document.

    Before this, `SCHEMA_VERSION_FOR_RUN` was referenced nowhere outside `constants.py`
    and the frozen block: the document and the constant agreed, this file confirmed they
    agreed, and `results.py` stamped a literal. Changing the frozen constant changed
    nothing a row would carry, which is a freeze that cannot bind.
    """
    from verbatim_bench.results import RunResult

    doc = RunResult(spec_dict={"arm": "t", "chunk_ms": 160}, sessions=[]).to_json_dict_v2()
    assert doc["schema"] == constants.SCHEMA_VERSION_FOR_RUN
