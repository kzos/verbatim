# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``scripts/compare_stock_replay.py``: a replay of the first N targets against the stock record.

The records here are synthetic, built from one table of which targets diverge in which arm: a
stock record in the shape the probe wrote at 6583a83 (three-element timings, no
``every_recording``, six targets) and a replay in the shape the probe writes now (four-element
timings, ``every_recording``, the observed stamps, the first four targets). The stock lists
divergences past the replay's last target, which must be left out of every comparison.

Each check has its own case below: one edit to one record that the check, and only the check,
has to report. The expected line is asserted, not only the exit code, so a check removed from
the comparator turns its case red even where another check still fires.
``tests/test_stock_divergence.py`` runs the comparator on records the probe itself writes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]


def _load() -> Any:
    path = REPO / "scripts" / "compare_stock_replay.py"
    spec = importlib.util.spec_from_file_location("compare_stock_replay_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


csr = _load()

STOCK_TARGETS, N = 6, 4
IDS = [f"s-{n}" for n in range(STOCK_TARGETS)]
ARMS = ["ragged", "equalised", "fixed"]
#: Which targets diverge, and how, in each arm. n = 5 and n = 4 are past the replay's targets.
TRUTH: dict[str, dict[int, str]] = {
    "ragged": {1: "text", 2: "timing", 5: "text"},
    "equalised": {2: "timing", 4: "timing"},
    "fixed": {},
}
SAME_SHAPE = "run-to-run identical in the same shape"


def _words(n: int, shift: float = 0.0) -> list[list[Any]]:
    return [[f"w{n}", 0.0, 0.08 + shift], [f"x{n}.", 0.08 + shift, 0.16]]


def _sides(n: int, kind: str | None) -> tuple[str, str, list[list[Any]], list[list[Any]]]:
    a_text = f"w{n} x{n}."
    if kind == "text":
        return a_text, f"w{n} x{n}", _words(n), [[f"w{n}", 0.0, 0.08], [f"x{n}", 0.08, 0.16]]
    if kind == "timing":
        return a_text, a_text, _words(n), _words(n, 0.04)
    return a_text, a_text, _words(n), _words(n)


def _arm(kinds: dict[int, str], targets: int, *, replay: bool) -> dict[str, Any]:
    def stored(timings: list[list[Any]]) -> list[list[Any]]:
        return [[*t, 0.0] for t in timings] if replay else [list(t) for t in timings]

    arm: dict[str, Any] = {
        "checked": targets,
        "text_divergent": 0,
        "timing_only_divergent": 0,
        "divergences": [],
        "transcripts": {},
    }
    if replay:
        arm["confidence_only_divergent"] = 0
        arm["every_recording"] = {}
    for n in range(targets):
        kind = kinds.get(n)
        a_text, b_text, a_t, b_t = _sides(n, kind)
        rid = IDS[n]
        arm["transcripts"][rid] = {"a": a_text, "b": b_text} if kind == "text" else {"a": a_text}
        if kind is not None:
            arm["text_divergent" if kind == "text" else "timing_only_divergent"] += 1
            arm["divergences"].append(
                {
                    "n": n,
                    "librispeech_id": rid,
                    "text_differs": kind == "text",
                    "a": a_text,
                    "b": b_text,
                    "a_timings": stored(a_t),
                    "b_timings": stored(b_t),
                }
            )
        if replay:
            arm["every_recording"][rid] = {
                "a_text": a_text,
                "b_text": b_text,
                "a_words": stored(a_t),
                "b_words": stored(b_t),
            }
    return arm


def _record(targets: int, *, replay: bool) -> dict[str, Any]:
    run: dict[str, Any] = {
        "arms": {arm: _arm(TRUTH[arm], targets, replay=replay) for arm in ARMS},
        "repeat": {"alone_identical": 3, "batch_identical": 3, "checked": 3},
        "repeat_verdict": SAME_SHAPE,
    }
    record: dict[str, Any] = {
        "question": "Does a recording's transcript depend on its batch neighbours?",
        "model": "example/streaming-model",
        "model_revision": "rev0",
        "chunk_ms": 1120,
        "batch": 32,
        "matmul_precision": "highest",
        "att_context_size": [70, 13],
        "arms": list(ARMS),
        "targets": targets,
        "references": {IDS[n]: f"reference {n}" for n in range(targets)},
        "runs": {"bfloat16": run},
    }
    if replay:
        record["word_confidence"] = "off"
        record["att_context_size_observed"] = [70, 13]
        run["att_context_size_observed"] = [70, 13]
        run["word_confidence_observed"] = {
            "mode_requested": "off",
            "decoder_step_confidence": False,
            "nonzero_conf_words_on_guard_recording": 0,
        }
    return record


def _pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return _record(STOCK_TARGETS, replay=False), _record(N, replay=True)


def test_the_synthetic_records_are_what_they_claim() -> None:
    stock, replay = _pair()
    ragged = stock["runs"]["bfloat16"]["arms"]["ragged"]
    assert [d["n"] for d in ragged["divergences"]] == [1, 2, 5]
    assert all(len(t) == 3 for d in ragged["divergences"] for t in d["a_timings"])
    assert "every_recording" not in ragged
    mine = replay["runs"]["bfloat16"]["arms"]["ragged"]
    assert [d["n"] for d in mine["divergences"]] == [1, 2]
    assert all(len(t) == 4 for d in mine["divergences"] for t in d["a_timings"])
    assert list(mine["every_recording"]) == IDS[:N]


def test_a_replay_of_the_stock_answers_matches() -> None:
    stock, replay = _pair()
    assert csr.differences(stock, replay) == []


def test_the_stock_divergences_past_the_replays_targets_are_left_out() -> None:
    """The stock's n = 5 (ragged) and n = 4 (equalised) are not the replay's to reproduce."""
    stock, replay = _pair()
    assert csr.differences(stock, replay) == []
    # The same records at N = 6 would have to hold them.
    full = _record(STOCK_TARGETS, replay=True)
    assert csr.differences(stock, full) == []


def test_the_confidence_a_replay_stores_is_not_compared() -> None:
    stock, replay = _pair()
    arm = replay["runs"]["bfloat16"]["arms"]["ragged"]
    (d,) = [d for d in arm["divergences"] if d["n"] == 2]
    for t in d["a_timings"]:
        t[3] = 0.5
    for t in arm["every_recording"]["s-2"]["a_words"]:
        t[3] = 0.5
    assert csr.differences(stock, replay) == []


def test_a_fake_pipeline_false_is_not_a_fake() -> None:
    stock, replay = _pair()
    replay["fake_pipeline"] = False
    assert csr.differences(stock, replay) == []


# --- one edit, one reported difference ---------------------------------------------------------


def _arm_of(record: dict[str, Any], arm: str = "ragged") -> dict[str, Any]:
    return record["runs"]["bfloat16"]["arms"][arm]


def _div(arm: dict[str, Any], rid: str) -> dict[str, Any]:
    (d,) = [d for d in arm["divergences"] if d["librispeech_id"] == rid]
    return d


def _drop_divergence(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """The replay reads s-2 (a timing divergence in the stock) as identical, consistently."""
    arm = _arm_of(replay)
    arm["divergences"] = [d for d in arm["divergences"] if d["librispeech_id"] != "s-2"]
    arm["timing_only_divergent"] -= 1
    entry = arm["every_recording"]["s-2"]
    entry["b_words"] = copy.deepcopy(entry["a_words"])


def _add_divergence(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """The replay reads s-0 (identical in the stock) as a timing divergence, consistently."""
    arm = _arm_of(replay)
    entry = arm["every_recording"]["s-0"]
    entry["b_words"] = [[*t[:2], t[2] + 0.04, t[3]] for t in entry["a_words"]]
    arm["divergences"].insert(
        0,
        {
            "n": 0,
            "librispeech_id": "s-0",
            "text_differs": False,
            "a": entry["a_text"],
            "b": entry["b_text"],
            "a_timings": copy.deepcopy(entry["a_words"]),
            "b_timings": copy.deepcopy(entry["b_words"]),
        },
    )
    arm["timing_only_divergent"] += 1


def _set_b_text(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """s-1's batched text reads differently in the replay, consistently in all three places."""
    arm = _arm_of(replay)
    _div(arm, "s-1")["b"] = "w1 x1!"
    arm["every_recording"]["s-1"]["b_text"] = "w1 x1!"
    arm["transcripts"]["s-1"]["b"] = "w1 x1!"


def _set_a_text(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    arm = _arm_of(replay)
    _div(arm, "s-1")["a"] = "w1 x1?"
    arm["every_recording"]["s-1"]["a_text"] = "w1 x1?"
    arm["transcripts"]["s-1"]["a"] = "w1 x1?"


def _shift(side: str, index: int, value: Any) -> Callable[[dict[str, Any], dict[str, Any]], None]:
    """One element of one of s-2's timings, changed in the divergence and in every_recording."""

    def edit(stock: dict[str, Any], replay: dict[str, Any]) -> None:
        arm = _arm_of(replay)
        _div(arm, "s-2")[f"{side}_timings"][0][index] = value
        arm["every_recording"]["s-2"][f"{side}_words"][0][index] = value

    return edit


def _rename_texts_of_s0(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """s-0 is identical on both sides in both runs, with other words in the replay."""
    arm = _arm_of(replay)
    arm["every_recording"]["s-0"]["a_text"] = "other words"
    arm["every_recording"]["s-0"]["b_text"] = "other words"
    arm["transcripts"]["s-0"] = {"a": "other words"}


def _set(path: tuple[Any, ...], value: Any, *, stock: bool = False) -> Any:
    def edit(stock_record: dict[str, Any], replay_record: dict[str, Any]) -> None:
        target = stock_record if stock else replay_record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _delete(path: tuple[Any, ...], *, stock: bool = False) -> Any:
    def edit(stock_record: dict[str, Any], replay_record: dict[str, Any]) -> None:
        target = stock_record if stock else replay_record
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return edit


RUN = ("runs", "bfloat16")
RAGGED = (*RUN, "arms", "ragged")


def _reorder_references(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    items = list(replay["references"].items())
    replay["references"] = dict([items[1], items[0], *items[2:]])


def _rename_dtype(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    replay["runs"] = {"float32": replay["runs"]["bfloat16"]}


def _reorder_every_recording(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    arm = _arm_of(replay)
    items = list(arm["every_recording"].items())
    arm["every_recording"] = dict([items[1], items[0], *items[2:]])


def _entry_off_its_divergence(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    _arm_of(replay)["every_recording"]["s-2"]["b_words"][0][2] = 0.3


def _entry_text_off_its_divergence(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """s-2's every_recording texts are not its divergence's; its timings still are."""
    entry = _arm_of(replay)["every_recording"]["s-2"]
    entry["a_text"] = entry["b_text"] = "other words"


def _timing_divergence_with_equal_sides(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    arm = _arm_of(replay)
    d = _div(arm, "s-2")
    d["b_timings"] = copy.deepcopy(d["a_timings"])
    arm["every_recording"]["s-2"]["b_words"] = copy.deepcopy(d["a_timings"])


def _text_divergence_with_equal_texts(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    arm = _arm_of(replay)
    d = _div(arm, "s-1")
    d["b"] = d["a"]
    arm["every_recording"]["s-1"]["b_text"] = d["a"]
    arm["transcripts"]["s-1"]["b"] = d["a"]


def _stock_n_off(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    _div(_arm_of(stock), "s-1")["n"] = 3


def _replay_n_off(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    _div(_arm_of(replay), "s-1")["n"] = 0


def _replay_n_not_an_index(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    _div(_arm_of(replay), "s-1")["n"] = "1"


def _replay_n_a_bool(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    _div(_arm_of(replay), "s-1")["n"] = True


def _replay_n_negative(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    _div(_arm_of(replay), "s-1")["n"] = -3


def _listed_twice(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    arm = _arm_of(replay)
    arm["divergences"].append(copy.deepcopy(_div(arm, "s-1")))


def _extra_arm(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    replay["runs"]["bfloat16"]["arms"]["other"] = copy.deepcopy(_arm_of(replay))


def _stray_divergence_past_the_targets(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """The replay's fixed arm lists a divergence at n = 5, past its four targets, that its
    counts and its every_recording leave out: nothing else in the replay shows it."""
    a_text, b_text, a_t, b_t = _sides(5, "text")
    _arm_of(replay, "fixed")["divergences"].append(
        {
            "n": 5,
            "librispeech_id": "s-5",
            "text_differs": True,
            "a": a_text,
            "b": b_text,
            "a_timings": [[*t, 0.0] for t in a_t],
            "b_timings": [[*t, 0.0] for t in b_t],
        }
    )


def _two_dtypes_in_the_other_order(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """Both records ran both dtypes, each the same in both; only the order of the runs differs."""
    stock["runs"]["float32"] = copy.deepcopy(stock["runs"]["bfloat16"])
    replay["runs"] = {
        "float32": copy.deepcopy(replay["runs"]["bfloat16"]),
        "bfloat16": replay["runs"]["bfloat16"],
    }


def _second_dtype_checked_short(stock: dict[str, Any], replay: dict[str, Any]) -> None:
    """Both records ran two dtypes in the same order; only the second one's ragged arm of the
    replay is short. A comparator that looks at the first dtype alone passes it."""
    stock["runs"]["float32"] = copy.deepcopy(stock["runs"]["bfloat16"])
    replay["runs"]["float32"] = copy.deepcopy(replay["runs"]["bfloat16"])
    replay["runs"]["float32"]["arms"]["ragged"]["checked"] = 3


CASES: list[tuple[str, Any, str]] = [
    ("replay-fake", _set(("fake_pipeline",), True), "refused: the replay record has fake_pipeline"),
    # Present and null is not false: only an absent or false stamp is a real run.
    (
        "replay-fake-null",
        _set(("fake_pipeline",), None),
        "refused: the replay record has fake_pipeline None",
    ),
    (
        "stock-fake",
        _set(("fake_pipeline",), True, stock=True),
        "refused: the stock record has fake_pipeline True",
    ),
    ("replay-fake-a-string", _set(("fake_pipeline",), "yes"), "fake_pipeline 'yes'"),
    ("model", _set(("model",), "example/other"), "model is 'example/streaming-model' in the stock"),
    ("revision", _set(("model_revision",), "rev1"), "model_revision is 'rev0' in the stock"),
    # A setting the replay does not stamp at all is a difference, not a pass.
    (
        "revision-missing",
        _delete(("model_revision",)),
        "model_revision is 'rev0' in the stock, None",
    ),
    ("chunk", _set(("chunk_ms",), 560), "chunk_ms is 1120 in the stock"),
    ("batch", _set(("batch",), 16), "batch is 32 in the stock"),
    ("matmul", _set(("matmul_precision",), "high"), "matmul_precision is 'highest' in the stock"),
    ("att-requested", _set(("att_context_size",), [70, 6]), "att_context_size is [70, 13] in"),
    ("arms-setting", _set(("arms",), ["ragged", "fixed"]), "arms is ['ragged', 'equalised'"),
    (
        "att-observed",
        _set(("att_context_size_observed",), [70, 1]),
        "the replay's att_context_size_observed is [70, 1], not the stock's",
    ),
    (
        "att-observed-missing",
        _delete(("att_context_size_observed",)),
        "the replay's att_context_size_observed is None",
    ),
    (
        "att-observed-in-the-run",
        _set((*RUN, "att_context_size_observed"), [70, 1]),
        "bfloat16: the replay's att_context_size_observed is [70, 1]",
    ),
    (
        "att-observed-missing-in-the-run",
        _delete((*RUN, "att_context_size_observed")),
        "bfloat16: the replay's att_context_size_observed is None",
    ),
    (
        "word-confidence-asked",
        _set(("word_confidence",), "nemo-shipped"),
        "asked for word confidence 'nemo-shipped', not off",
    ),
    (
        "word-confidence-kept",
        _set((*RUN, "word_confidence_observed", "decoder_step_confidence"), True),
        "bfloat16: the replay's built decoder kept step confidence True",
    ),
    (
        "word-confidence-unobserved",
        _delete((*RUN, "word_confidence_observed")),
        "kept step confidence None",
    ),
    ("dtype", _rename_dtype, "the runs are ['bfloat16'] in the stock, ['float32']"),
    (
        "dtype-order",
        _two_dtypes_in_the_other_order,
        "the runs are ['bfloat16', 'float32'] in the stock, ['float32', 'bfloat16']",
    ),
    (
        "second-dtype-compared",
        _second_dtype_checked_short,
        "float32/ragged: the replay checked 3, not its 4",
    ),
    ("targets-past-the-stock", _set(("targets",), 7), "the replay's targets is 7"),
    ("targets-zero", _set(("targets",), 0), "the replay's targets is 0"),
    (
        "reference-text",
        _set(("references", "s-2"), "another reference"),
        "they part at target 2",
    ),
    ("reference-order", _reorder_references, "they part at target 0"),
    ("divergent-set-stock-only", _drop_divergence, "divergent in the stock only ['s-2']"),
    ("divergent-set-replay-only", _add_divergence, "in the replay only ['s-0']"),
    (
        "text-differs",
        _set((*RAGGED, "divergences", 1, "text_differs"), True),
        "s-2: text_differs is False in the stock, True",
    ),
    ("b-text", _set_b_text, "s-1: b is 'w1 x1' in the stock, 'w1 x1!'"),
    ("a-text", _set_a_text, "s-1: a is 'w1 x1.' in the stock, 'w1 x1?'"),
    ("b-end", _shift("b", 2, 0.2), "s-2: b_timings (word, start, end) differ"),
    ("a-start", _shift("a", 1, 0.01), "s-2: a_timings (word, start, end) differ"),
    ("timing-word", _shift("b", 0, "other"), "s-2: b_timings (word, start, end) differ"),
    ("stock-n", _stock_n_off, "bfloat16/ragged stock: divergence n=3 names 's-1', but target 3"),
    ("replay-n", _replay_n_off, "bfloat16/ragged replay: divergence n=0 names 's-1'"),
    ("replay-n-compared", _replay_n_off, "s-1: n is 1 in the stock, 0"),
    ("replay-n-not-an-index", _replay_n_not_an_index, "has index n '1', not a target index"),
    ("replay-n-a-bool", _replay_n_a_bool, "has index n True, not a target index"),
    ("replay-n-negative", _replay_n_negative, "has index n -3, not a target index"),
    ("listed-twice", _listed_twice, "'s-1' is listed as a divergence twice"),
    (
        "replay-divergence-past-its-targets",
        _stray_divergence_past_the_targets,
        "bfloat16/fixed replay: divergence n=5 names 's-5', but target 5 is None",
    ),
    (
        "replay-divergence-past-its-targets-in-the-set",
        _stray_divergence_past_the_targets,
        "bfloat16/fixed: divergent in the stock only [], in the replay only ['s-5']",
    ),
    (
        "count-against-the-stock",
        _set((*RAGGED, "text_divergent"), 2),
        "text_divergent is 2 in the replay, 1 in the stock over n < 4",
    ),
    (
        "count-against-its-own-divergences",
        _set((*RAGGED, "timing_only_divergent"), 2),
        "the replay's timing_only_divergent 2 disagrees with its own divergences (1)",
    ),
    ("count-set-stock-only", _drop_divergence, "timing_only_divergent is 0 in the replay, 1"),
    ("checked", _set((*RAGGED, "checked"), 3), "the replay checked 3, not its 4"),
    ("texts-of-every-target", _rename_texts_of_s0, "s-0: texts (a, b) are ('w0 x0.', 'w0 x0.')"),
    (
        "transcript",
        _set((*RAGGED, "transcripts", "s-0"), {"a": "w0 x0"}),
        "s-0: the transcript is {'a': 'w0 x0.'} in the stock, {'a': 'w0 x0'}",
    ),
    (
        "stock-transcript-missing",
        _delete((*RAGGED, "transcripts", "s-3"), stock=True),
        "s-3: the stock record has no transcript",
    ),
    (
        "every-recording-missing",
        _delete((*RAGGED, "every_recording", "s-3")),
        "every_recording holds 3 recordings, not its 4 targets in order",
    ),
    (
        "every-recording-order",
        _reorder_every_recording,
        "every_recording holds 4 recordings, not its 4 targets in order",
    ),
    (
        "undivergent-timing",
        _set((*RAGGED, "every_recording", "s-0", "b_words", 1, 1), 0.1),
        "s-0: the replay lists no divergence, but its every_recording sides differ",
    ),
    (
        "undivergent-text",
        _set((*RAGGED, "every_recording", "s-3", "b_text"), "w3"),
        "s-3: the replay lists no divergence, but its every_recording sides differ",
    ),
    (
        "entry-against-its-divergence",
        _entry_off_its_divergence,
        "s-2: the replay's every_recording disagrees with its divergence",
    ),
    (
        "entry-text-against-its-divergence",
        _entry_text_off_its_divergence,
        "s-2: the replay's every_recording disagrees with its divergence",
    ),
    (
        "timing-divergence-with-equal-sides",
        _timing_divergence_with_equal_sides,
        "s-2: the replay's divergence says text_differs False, and its sides do not differ",
    ),
    (
        "text-divergence-with-equal-texts",
        _text_divergence_with_equal_texts,
        "s-1: the replay's divergence says text_differs True, and its sides do not differ",
    ),
    (
        "repeat-verdict",
        _set((*RUN, "repeat_verdict"), "NOT run-to-run deterministic"),
        "bfloat16: repeat_verdict is 'run-to-run identical in the same shape' in the stock",
    ),
    (
        "repeat-counts",
        _set((*RUN, "repeat", "batch_identical"), 2),
        "bfloat16: repeat is {'alone_identical': 3, 'batch_identical': 3",
    ),
    (
        "arm-missing",
        _delete((*RUN, "arms", "fixed")),
        "bfloat16/fixed: the replay has no such arm",
    ),
    ("arm-extra", _extra_arm, "bfloat16/other: the stock has no such arm"),
]


@pytest.mark.parametrize(("edit", "expected"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_each_check_reports_its_own_difference(edit: Any, expected: str) -> None:
    stock, replay = _pair()
    edit(stock, replay)
    found = csr.differences(stock, replay)
    assert any(expected in line for line in found), found


@pytest.mark.parametrize("who", ["stock", "replay"])
def test_a_fake_record_is_refused_before_anything_else_is_compared(who: str) -> None:
    stock, replay = _pair()
    {"stock": stock, "replay": replay}[who]["fake_pipeline"] = True
    replay["model"] = "example/other"  # a difference that would be reported, were it compared
    found = csr.differences(stock, replay)
    assert found == [
        f"refused: the {who} record has fake_pipeline True; the CPU test fake wrote it, so it"
        " measures nothing"
    ]


# --- the repeat control over fewer targets than the stock checked ------------------------------

NOT_DETERMINISTIC = "NOT run-to-run deterministic"


def _repeat(
    stock_counts: tuple[int, int, int],
    replay_counts: tuple[int, int, int],
    verdict: str = SAME_SHAPE,
) -> list[str]:
    """The default pair with these repeat counts, (alone, batch, checked), and one verdict."""
    stock, replay = _pair()
    for record, counts in ((stock, stock_counts), (replay, replay_counts)):
        run = record["runs"]["bfloat16"]
        run["repeat"] = dict(
            zip(("alone_identical", "batch_identical", "checked"), counts, strict=True)
        )
        run["repeat_verdict"] = verdict
    return csr.differences(stock, replay)


def test_the_repeat_counts_are_the_stocks_over_the_targets_the_replay_checked() -> None:
    """REPEAT = 6 with six stock targets, all identical: a replay of four checks four, all
    identical, and that is the stock's answer over those four."""
    assert _repeat((6, 6, 6), (4, 4, 4)) == []
    assert _repeat((6, 6, 6), (6, 6, 6)) == [
        "bfloat16: repeat is {'alone_identical': 4, 'batch_identical': 4, 'checked': 4} in the"
        " stock over its first 4 targets, {'alone_identical': 6, 'batch_identical': 6,"
        " 'checked': 6} in the replay"
    ]
    assert _repeat((6, 6, 6), (4, 3, 4)) != []
    assert _repeat((6, 6, 6), (4, 4, 3)) != []


def test_a_total_of_zero_restricts_to_zero() -> None:
    assert _repeat((0, 6, 6), (0, 4, 4), NOT_DETERMINISTIC) == []
    assert _repeat((6, 0, 6), (4, 0, 4), NOT_DETERMINISTIC) == []
    assert _repeat((0, 6, 6), (1, 4, 4), NOT_DETERMINISTIC) != []


def test_a_partial_total_is_compared_whole_when_the_replay_checked_every_stock_target() -> None:
    """The stock checked 3 and the replay's 4 targets cover all 3: 2 of 3 is 2 of 3."""
    assert _repeat((2, 3, 3), (2, 3, 3), NOT_DETERMINISTIC) == []
    assert _repeat((3, 2, 3), (3, 2, 3), NOT_DETERMINISTIC) == []
    assert _repeat((2, 3, 3), (3, 3, 3), NOT_DETERMINISTIC) != []


@pytest.mark.parametrize(
    ("stock_counts", "key", "total"),
    [((5, 6, 6), "alone_identical", 5), ((6, 1, 6), "batch_identical", 1)],
    ids=["alone", "batch"],
)
def test_a_partial_total_over_more_targets_than_the_replay_cannot_be_restricted(
    stock_counts: tuple[int, int, int], key: str, total: int
) -> None:
    """5 of 6 says one of the six differed, not which: over the first four it is 3 or 4. The
    comparator says so instead of passing either."""
    for replay_counts in ((4, 4, 4), (3, 4, 4), (4, 1, 4), (1, 1, 4)):
        found = _repeat(stock_counts, replay_counts, NOT_DETERMINISTIC)
        assert (
            f"bfloat16: the stock's repeat {key} is {total} of 6, which cannot be restricted to"
            " the replay's first 4; replay at least 6 targets"
        ) in found


# --- the command line --------------------------------------------------------------------------


def _files(tmp_path: Path, stock: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    paths = [tmp_path / "stock.json", tmp_path / "replay.json"]
    for path, record in zip(paths, (stock, replay), strict=True):
        path.write_text(json.dumps(record))
    return [str(p) for p in paths]


def test_main_exits_zero_on_a_match_and_says_what_matched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert csr.main(_files(tmp_path, *_pair())) == 0
    out = capsys.readouterr().out
    assert "DIFFERENCE" not in out
    assert "bfloat16/ragged: 4 targets, text 1, timing-only 1:" in out
    assert "bfloat16/equalised: 4 targets, text 0, timing-only 1:" in out
    assert out.rstrip().endswith(
        "VERDICT: reproduced: over its first 4 targets the replay gives exactly the stock"
        " record's answers"
    )


def test_main_prints_every_difference_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stock, replay = _pair()
    replay["batch"] = 16
    _set_b_text(stock, replay)
    assert csr.main(_files(tmp_path, stock, replay)) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "DIFFERENCE: batch is 32 in the stock, 16",
        "DIFFERENCE: bfloat16/ragged s-1: b is 'w1 x1' in the stock, 'w1 x1!'",
    ]
    assert "DIFFERENCE: bfloat16/ragged s-1: texts (a, b) are ('w1 x1.', 'w1 x1')" in lines[2]
    assert lines[-1].startswith("VERDICT: NOT reproduced: 4 difference(s)")
    assert len(lines) == 5


def test_main_exits_one_when_a_record_cannot_be_compared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stock, replay = _pair()
    del replay["runs"]["bfloat16"]["arms"]["ragged"]["every_recording"]
    assert csr.main(_files(tmp_path, stock, replay)) == 1
    assert "DIFFERENCE: cannot compare: KeyError: 'every_recording'" in capsys.readouterr().out
    assert csr.main([str(tmp_path / "stock.json"), str(tmp_path / "absent.json")]) == 1
    assert "cannot compare: FileNotFoundError" in capsys.readouterr().out
