# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``scripts/audit_published_numbers.py``: each guard it adds is shown able to fail.

* The committed records pass.
* A copy of the summary with one number changed fails, naming every sentence that prints it.
* A rounded claim's tolerance is the rounding and no more: a record half a unit past what the
  paper prints fails, and one just inside it passes.
* The white paper's numbers mapping is compared both ways: a mapping built from the audit's own
  keys passes; with one printed number changed, a row dropped, or a row added, it fails.
* A tolerance without the rounding it stands for is refused, and so is a stated rounding with
  no tolerance.
* A claim whose string does not print the value it checks fails, whatever the record says.
* The Draft 2 abstract's "66 to 88", checked against Draft 2's own records, fails: Table 1's
  weight-2.0 control row is 89. The audit, edited back to 88, says so.
* The Draft 2 numbers this audit did not check before (§5.4's 79% and 88%, Table 1's "about
  1.2%", §5.2's 720-800 ms example, §4.2's 20-second churn) each fail when the audit is edited
  to a value the records do not give.
* The paper-best tie range (Table 4's caption and §7) and the served spans right by the
  flag's rule each fail when the summary gives another value.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_published_numbers.py"
SUMMARY = ROOT / "rows" / "exploratory" / "step1-summary-2026-09-26.json"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
audit_module = importlib.import_module("audit_published_numbers")

#: The unpadded server against itself, words changed: printed in five places.
WORD_LEVEL = ("server", "ragged_vs_ragged", "a207de6", "word_level")
FIVE_PLACES = ("abstract p1", "§1 contributions", "Figure 4", "§6.5 p2", "§10")


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = audit_module.main(list(argv))
    return code, capsys.readouterr().out


def summary_with(tmp_path: Path, keys: tuple[str, ...], value: object) -> Path:
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    node = document
    for key in keys[:-1]:
        node = node[key]
    assert keys[-1] in node
    node[keys[-1]] = value
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def mapping_from_the_audit() -> list[str]:
    """A numbers mapping in the paper's format, holding exactly the audit's own keys."""
    a = audit_module.audit()
    lines = ["# built by the test", "section\tprinted\tsource\tvalue"]
    for (section, printed, pointer), n in sorted(a.printed_keys.items()):
        source = f"summary {pointer} (a note)" if pointer else "Draft 2 record"
        lines += [f"{section}\t{printed}\t{source}\tv"] * n
    for section, printed in audit_module.NOT_FROM_A_RECORD:
        lines.append(f"{section}\t{printed}\tno record\tv")
    return lines


def write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "numbers.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def mutant(old: str, new: str) -> types.ModuleType:
    """The audit with one source edit, run from its own place so it reads the same records."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    module = types.ModuleType("audit_published_numbers_mutant")
    module.__file__ = str(SCRIPT)
    exec(compile(source.replace(old, new), str(SCRIPT), "exec"), module.__dict__)
    return module


def test_the_committed_records_pass(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys)
    assert code == 0, out
    assert out.rstrip().endswith("no mismatch")
    a = audit_module.audit()
    # Every Draft 3 claim is counted, and none failed silently.
    assert sum(a.printed_keys.values()) > 300
    assert a.failures == []


def test_a_changed_summary_number_fails_in_every_sentence_that_prints_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = summary_with(tmp_path, WORD_LEVEL, 148)
    code, out = run(capsys, "--summary", str(path))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "record=148 published=147" in line]
    assert failing == [
        f"paper {section}: '147': record=148 published=147" for section in FIVE_PLACES
    ]


@pytest.mark.parametrize(("precision", "code"), [(0.84499, 0), (0.84501, 1)])
def test_a_rounded_claim_allows_the_rounding_and_no_more(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, precision: float, code: int
) -> None:
    keys = ("flips_vs_confidence", "modes", "paper-best", "confidence", "precision")
    got, out = run(capsys, "--summary", str(summary_with(tmp_path, keys, precision)))
    assert got == code, out
    if code:
        assert out.count("'84%' (rounded to a whole percent): record=84.501 published=84") == 3


def test_the_numbers_mapping_is_compared_both_ways(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    lines = mapping_from_the_audit()
    code, out = run(capsys, "--numbers", str(write(tmp_path, lines)))
    assert code == 0, out
    assert f"numbers.tsv: all {len(lines) - 2} rows are claims here or listed" in out

    # One printed number changed: the row is unchecked and the claim is unlisted.
    target = next(i for i, line in enumerate(lines) if line.startswith("abstract p1\t147\t"))
    changed = [*lines]
    changed[target] = changed[target].replace("\t147\t", "\t148\t", 1)
    code, out = run(capsys, "--numbers", str(write(tmp_path, changed)))
    assert code == 1
    assert "numbers.tsv abstract p1: '148'" in out
    assert "claim abstract p1: '147'" in out

    # A row dropped: the claim is one the paper no longer prints.
    dropped = [*lines[:target], *lines[target + 1 :]]
    code, out = run(capsys, "--numbers", str(write(tmp_path, dropped)))
    assert code == 1
    assert "claim abstract p1: '147' (/server/ragged_vs_ragged/a207de6/word_level) x1" in out

    # A row added: a number the paper prints that nothing checks.
    added = [*lines, "§6.6 p1\t112 ms\tthe tick budget\tv"]
    code, out = run(capsys, "--numbers", str(write(tmp_path, added)))
    assert code == 1
    assert "numbers.tsv §6.6 p1: '112 ms' (no pointer) x1: no claim here checks it" in out


def test_a_tolerance_is_the_rounding_it_states() -> None:
    a = audit_module.Audit()
    with pytest.raises(ValueError, match="rounding"):
        a.printed("s", "84%", "", 84, record=84.07, tol=0.5)
    with pytest.raises(ValueError, match="rounding"):
        a.printed("s", "84%", "", 84, record=84.07, rounds="to a whole percent")
    a.printed("s", "84%", "", 84, record=84.07, tol=0.5, rounds="to a whole percent")
    assert (a.matched, a.failures) == (1, [])


def test_a_claim_must_print_the_value_it_checks() -> None:
    a = audit_module.Audit()
    # The record agrees with the value checked, but the string prints something else.
    a.printed("s", "264 to 328", "", (264, 329), record=(264, 329))
    assert a.matched == 0
    assert a.failures == [
        "paper s: '264 to 328': checks (264, 329), which the string does not print"
    ]
    # A string in words carries the reading it applies instead.
    a.printed("s", "two commits", "", 2, record=2, rule="'two': the commits")
    assert a.matched == 1


def test_draft_2s_range_that_left_out_89_fails_against_draft_2s_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = mutant(
        '        "66 to 89 of 256",\n        "",\n        (66, 89, [256]),',
        '        "66 to 88 of 256",\n        "",\n        (66, 88, [256]),',
    )
    code = module.main([])
    out = capsys.readouterr().out
    assert code == 1
    assert (
        "paper abstract p2: '66 to 88 of 256': record=(66, 89, [256]) published=(66, 88, [256])"
        in out
    )


@pytest.mark.parametrize(
    ("old", "new", "failing"),
    [
        (
            '(("equalised", 295, 79), ("ragged", 264, 88))',
            '(("equalised", 295, 80), ("ragged", 264, 88))',
            "paper §5.4 unstable in both B300 arms, 80% of the 295 (to a whole percent): record=",
        ),
        (
            "        (1 - 0.05 ** (1 / n_streams)) * 100,\n        1.2,",
            "        (1 - 0.05 ** (1 / n_streams)) * 100,\n        1.3,",
            "'about 1.2%' (to one decimal place): record=1.16",
        ),
        (
            r"""re.fullmatch(r"\('[\w']+', 800, 880\)", d["right"])""",
            r"""re.fullmatch(r"\('[\w']+', 800, 960\)", d["right"])""",
            "paper §5.2 a word moved from 720-800 ms to 800-880 ms: record=False published=True",
        ),
        (
            "        [20.0],\n    )",
            "        [15.0],\n    )",
            "paper §4.2 the churn period, seconds: record=[20.0] published=[15.0]",
        ),
    ],
)
def test_draft_2_numbers_checked_now_fail_when_the_audit_says_otherwise(
    capsys: pytest.CaptureFixture[str], old: str, new: str, failing: str
) -> None:
    code = mutant(old, new).main([])
    out = capsys.readouterr().out
    assert code == 1
    assert failing in out


def test_the_tie_range_and_the_served_spans_right_fail_when_the_summary_moves(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    keys = (
        "flips_vs_confidence",
        "modes",
        "paper-best",
        "tie_range",
        "range",
        "word_errors_removable",
    )
    code, out = run(capsys, "--summary", str(summary_with(tmp_path, keys, [219, 222])))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "'219 to 221'" in line]
    assert failing == [
        f"paper {section}: '219 to 221': record=[219, 222] published=[219, 221]"
        for section in ("Table 4 caption", "§7 p3")
    ]
    keys = ("flips_vs_confidence", "served_right", "span_rule")
    code, out = run(capsys, "--summary", str(summary_with(tmp_path, keys, 63)))
    assert code == 1
    assert "paper Table 4 caption: '64': record=63 published=64" in out
