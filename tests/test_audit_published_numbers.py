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
* The B300 summary is read as the A6000 one is: a copy with one number changed fails,
  naming every sentence that prints it; the draft's date fails when the newest record moves
  to another day; "no capacity search" fails when a capacity ladder is among the records.
* A mapping row that checks several values is keyed by all of its pointers, in order: a row
  that drops or reorders one fails, and so does a B300 row with one printed number changed.
* The audit edited to print another number fails, for a claim read as the string's figures
  and for one read by a stated rule: a claim whose string prints a figure it does not check
  fails unless the figure is set aside with the reason it is not a measurement.
* The mapping's rows must stand in the order the claims are made, the paper's: two rows of one
  section swapped, or a row moved to another section's place, fails, so two numbers swapped in
  the paper cannot be covered by reordering the mapping to match.
* "One card of each" counts GPU UUIDs, not device names: a second A6000 card in the B300
  summary's a6000 section fails, and so does a card count no record backs.
* The A6000's smoke-stopped runbook, the two server processes of its unpadded pair at 32
  streams, and "more of the B300's ticks ended after the next was due" each fail when the
  summary says otherwise.
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
B300_SUMMARY = ROOT / "rows" / "exploratory" / "step1-b300-summary-2026-09-26.json"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
audit_module = importlib.import_module("audit_published_numbers")

#: The unpadded server against itself on the A6000, words changed: printed in seven places.
WORD_LEVEL = ("server", "ragged_vs_ragged", "a207de6", "word_level")
PLACES_147 = (
    ("abstract p2", "'on 147 recordings at 32 streams': record=[148, 32] published=[147, 32]"),
    ("§1 contributions", "'147': record=148 published=147"),
    ("Figure 4", "'147': record=148 published=147"),
    ("§6.5 p2", "'147': record=148 published=147"),
    ("Table 5 A6000 32", "'147': record=148 published=147"),
    ("§6.8 p2", "'on 147 recordings at 32 streams': record=[148, 32] published=[147, 32]"),
    ("§10", "'147': record=148 published=147"),
)
#: The B300's unpadded answer against its padded one, words changed, and the five sentences
#: that print it (each printing is its own string).
B300_WORD_LEVEL = ("b300", "ragged_vs_fixed", "word_level")
PLACES_128 = (
    ("abstract p2", "on 128 on the B300"),
    ("§6.7 p2", "128"),
    ("§6.8 p2", "at word level on 128 recordings"),
    ("§8 why (changed)", "128 recordings' words differed"),
    ("§10", "of 128 on the B300"),
)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = audit_module.main(list(argv))
    return code, capsys.readouterr().out


def summary_with(
    tmp_path: Path, keys: tuple[str | int, ...], value: object, source: Path = SUMMARY
) -> Path:
    """A copy of a summary with the value at `keys` replaced (the key must already be there)."""
    document = json.loads(source.read_text(encoding="utf-8"))
    node = document
    for key in keys[:-1]:
        node = node[key]
    if isinstance(node, dict):
        assert keys[-1] in node
    node[keys[-1]] = value
    path = tmp_path / source.name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def mapping_from_the_audit() -> list[str]:
    """A numbers mapping in the paper's format, holding exactly the audit's own keys."""
    a = audit_module.audit()
    lines = ["# built by the test", "section\tprinted\tsource\tvalue"]
    for section, printed, pointer in a.printed_order:
        if pointer.startswith(audit_module.B300):
            source = f"{pointer} (a note)"
        else:
            source = f"summary {pointer} (a note)" if pointer else "Draft 2 record"
        lines.append(f"{section}\t{printed}\t{source}\tv")
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
    assert sum(a.printed_keys.values()) > 600
    assert a.failures == []


def test_a_changed_summary_number_fails_in_every_sentence_that_prints_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = summary_with(tmp_path, WORD_LEVEL, 148)
    code, out = run(capsys, "--summary", str(path))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "record=148" in line or "[148" in line]
    assert failing == [f"paper {section}: {reading}" for section, reading in PLACES_147]


def test_a_changed_b300_summary_number_fails_in_every_sentence_that_prints_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = summary_with(tmp_path, B300_WORD_LEVEL, 129, B300_SUMMARY)
    code, out = run(capsys, "--b300-summary", str(path))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "record=129 published=128" in line]
    assert failing == [
        f"paper {section}: {printed!r}: record=129 published=128" for section, printed in PLACES_128
    ]


def test_the_drafts_date_is_the_day_of_its_newest_record(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    keys = ("b300", "retest_c64", "timeline", 3, "finished")
    path = summary_with(tmp_path, keys, "2026-09-28T00:00:04+0000", B300_SUMMARY)
    code, out = run(capsys, "--b300-summary", str(path))
    assert code == 1
    assert (
        "paper header: '27 September 2026': record='28 September 2026' "
        "published='27 September 2026'"
    ) in out


def test_no_capacity_search_fails_when_a_ladder_is_among_the_records(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    b300 = json.loads(B300_SUMMARY.read_text(encoding="utf-8"))
    raw = {**b300["inputs"]["b300"]["sha256_raw"], "ladder-ba20c49-fixed.json.gz": "0" * 64}
    path = summary_with(tmp_path, ("inputs", "b300", "sha256_raw"), raw, B300_SUMMARY)
    code, out = run(capsys, "--b300-summary", str(path))
    assert code == 1
    assert (
        "paper §6.6 p2: 'No capacity search was run for Nemotron on either card' "
        "['no capacity search': none of the Step 1 records, on either card, is a capacity "
        "ladder]: record=(True, False) published=(True, True)"
    ) in out


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
    target = next(i for i, line in enumerate(lines) if line.startswith("§1 contributions\t147\t"))
    changed = [*lines]
    changed[target] = changed[target].replace("\t147\t", "\t148\t", 1)
    code, out = run(capsys, "--numbers", str(write(tmp_path, changed)))
    assert code == 1
    assert "numbers.tsv §1 contributions: '148'" in out
    assert "claim §1 contributions: '147'" in out

    # A row dropped: the claim is one the paper no longer prints.
    dropped = [*lines[:target], *lines[target + 1 :]]
    code, out = run(capsys, "--numbers", str(write(tmp_path, dropped)))
    assert code == 1
    assert "claim §1 contributions: '147' (/server/ragged_vs_ragged/a207de6/word_level) x1" in out

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
        "paper abstract p3: '66 to 88 of 256': record=(66, 89, [256]) published=(66, 88, [256])"
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


def test_mapping_keys_read_both_summaries_and_every_pointer() -> None:
    key = audit_module.mapping_key
    assert key("summary /server/setting/bucket (the A6000's)") == "/server/setting/bucket"
    assert key("summary /a/b + /c/d") == "/a/b + /c/d"
    assert key("b300 summary /b300/x + /b300/y (both true)") == "b300 summary /b300/x + /b300/y"
    assert key("derived: summary /a / /b") == ""
    assert key("Draft 2 abstract (audit: ...)") == ""


RATE_64 = (
    "b300 summary "
    "/b300/captures/server-ba20c49-ragged-c64-run2.json.gz/ticks_over_budget_rate + "
    "/b300/captures/server-ba20c49-ragged-c64-run1.json.gz/ticks_over_budget_rate"
)


def test_a_row_of_several_pointers_must_name_every_one_in_order(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    lines = mapping_from_the_audit()
    target = next(i for i, line in enumerate(lines) if line.startswith("§6.9 p2\t12% to 14%\t"))
    assert RATE_64 in lines[target]
    first, second = RATE_64.removeprefix("b300 summary ").split(" + ")
    for source in (f"b300 summary {first}", f"b300 summary {second} + {first}"):
        edited = [*lines]
        edited[target] = f"§6.9 p2\t12% to 14%\t{source} (a note)\tv"
        code, out = run(capsys, "--numbers", str(write(tmp_path, edited)))
        assert code == 1
        assert f"numbers.tsv §6.9 p2: '12% to 14%' ({source}) x1: no claim here checks it" in out
        assert f"claim §6.9 p2: '12% to 14%' ({RATE_64}) x1: the paper's numbers mapping" in out


def test_a_b300_row_with_a_changed_printed_number_fails(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    lines = mapping_from_the_audit()
    target = next(
        i for i, line in enumerate(lines) if line.startswith("Table 5 B300 64\t285 of 2,016\t")
    )
    lines[target] = lines[target].replace("\t285 of 2,016\t", "\t286 of 2,016\t", 1)
    code, out = run(capsys, "--numbers", str(write(tmp_path, lines)))
    assert code == 1
    assert "numbers.tsv Table 5 B300 64: '286 of 2,016'" in out
    assert "claim Table 5 B300 64: '285 of 2,016'" in out


@pytest.mark.parametrize(
    ("old", "new", "failing"),
    [
        (
            '("285 of 2,016", "248 of 2,001")',
            '("286 of 2,016", "248 of 2,001")',
            "paper Table 5 B300 64: '286 of 2,016': record=[285, 2016] published=[286.0, 2016.0]",
        ),
        (
            '"run once at 64",',
            '"run once at 65",',
            "paper §6.9 p1: 'run once at 65' ['once': the B300's padded records at 64 streams, "
            "captures and refused runs]: prints [65.0], which the claim does not check",
        ),
    ],
)
def test_the_audit_edited_to_print_another_number_fails(
    capsys: pytest.CaptureFixture[str], old: str, new: str, failing: str
) -> None:
    code = mutant(old, new).main([])
    out = capsys.readouterr().out
    assert code == 1
    assert failing in out


def test_a_claim_must_check_every_figure_it_prints() -> None:
    a = audit_module.Audit()
    # A rule reads the words, but the string's figures must still be among the values checked.
    a.printed("s", "held 32 but not 64", "", (32,), record=(32,), rule="'held': 32")
    assert a.matched == 0
    assert a.failures == [
        "paper s: 'held 32 but not 64' ['held': 32]: prints [64.0], which the claim does not check"
    ]
    # A figure that is not a measurement is set aside with its reason, and then passes.
    a.printed(
        "s", "held 32 but not 64", "", (32,), record=(32,), rule="'held': 32", aside={64: "x"}
    )
    assert a.matched == 1
    # A section, a commit or "95th" is not a figure, and a claim on text compares the text.
    a.printed("s", "the same answer (§6.7)", "", True, record=True, rule="'the same'")
    a.printed("s", "26eaebb", "", "26eaebb", record="26eaebb")
    assert (a.matched, len(a.failures)) == (3, 1)


def test_the_rows_must_stand_in_the_claims_order(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    lines = mapping_from_the_audit()
    code, out = run(capsys, "--numbers", str(write(tmp_path, lines)))
    assert code == 0, out
    assert "the claims in the paper's order" in out

    # Two rows of one section swapped, as a swap of 280 and 147 in §6.5 would need.
    first = next(i for i, line in enumerate(lines) if line.startswith("§6.5 p2\t280\t"))
    assert lines[first + 1].startswith("§6.5 p2\t147\t")
    swapped = [*lines]
    swapped[first], swapped[first + 1] = swapped[first + 1], swapped[first]
    code, out = run(capsys, "--numbers", str(write(tmp_path, swapped)))
    assert code == 1
    assert (
        "numbers.tsv claim row" in out
        and "is §6.5 p2: '147', where the claim here is §6.5 p2: '280'" in out
    )

    # A row moved to another section's place in the same part of the paper.
    caption = next(i for i, line in enumerate(lines) if line.startswith("Table 5 caption\t2,939\t"))
    header = next(i for i, line in enumerate(lines) if line.startswith("Table 5 header\t"))
    moved = [*lines]
    row = moved.pop(caption)
    moved.insert(header, row)
    code, out = run(capsys, "--numbers", str(write(tmp_path, moved)))
    assert code == 1
    assert "is Table 5 caption: '2,939', where the claim here is Table 5 header" in out


@pytest.mark.parametrize(
    ("keys", "value", "record"),
    [
        (("a6000", "card", "cards"), 2, "record=([1, 2], True) published=([1, 1], True)"),
        (
            ("a6000", "card", "records_naming_a_card"),
            0,
            "record=([1, 1], False) published=([1, 1], True)",
        ),
    ],
)
def test_one_card_of_each_counts_card_identities(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    keys: tuple[str, ...],
    value: int,
    record: str,
) -> None:
    path = summary_with(tmp_path, keys, value, B300_SUMMARY)
    code, out = run(capsys, "--b300-summary", str(path))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if line.startswith("    paper")]
    assert len(failing) == 1
    assert failing[0].startswith("paper abstract p1: 'one card of each measured'")
    assert failing[0].endswith(record)


def test_the_smoke_stops_fail_when_a_smoke_check_passed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    keys = ("a6000_gate_smoke", "b32", "stopped_before_the_gate")
    code, out = run(
        capsys, "--b300-summary", str(summary_with(tmp_path, keys, False, B300_SUMMARY))
    )
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "record=([64, 32], False)" in line]
    assert [line.split(" [")[0] for line in failing] == [
        "paper abstract p3: '64 or 32 rows'",
        "paper §6.6 p2: 'buckets of 64 and of 32 rows'",
        "paper §6.9 p1: 'buckets of 64 and of 32 rows'",
    ]


def test_two_server_processes_fail_when_the_pair_ran_on_one(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    keys = ("a6000", "ragged_c32_pair", "server_processes")
    code, out = run(capsys, "--b300-summary", str(summary_with(tmp_path, keys, 1, B300_SUMMARY)))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "record=(32, 1, True)" in line]
    assert [line.split(": ")[0] for line in failing] == [
        "paper Table 5 caption",
        "paper §6.8 p3",
    ]


def test_more_late_ticks_fail_when_a_b300_run_was_late_less_often(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    keys = ("b300", "captures", "server-ba20c49-ragged-c64-run2.json.gz", "ticks_late_during_run")
    code, out = run(capsys, "--b300-summary", str(summary_with(tmp_path, keys, 20, B300_SUMMARY)))
    assert code == 1
    failing = [line.strip() for line in out.splitlines() if "'more ... late'" in line]
    assert [line.split(": ")[0] for line in failing] == [
        "paper abstract p2",
        "paper §1 contributions",
        "paper §10",
    ]
    assert all("False" in line.split("record=")[1].split(" published=")[0] for line in failing)
