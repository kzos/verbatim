#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a replay of the stock probe's first N targets with the finished stock record.

    python scripts/compare_stock_replay.py STOCK.json REPLAY.json

STOCK is the record the stock probe wrote at 6583a83, when it was one top-level script: word
timings as (word, start_s, end_s), ``divergences`` listed only for the recordings that diverged,
each with its target index ``n``, and ``transcripts`` for every target. REPLAY is a record that
``probes/stock_divergence.py`` as it is now wrote over the first N targets of the same stream at
the same settings (N is the replay's ``targets``). The question is whether the restructured probe
gives the stock run's answers, exactly, where the two can be compared.

Why the first N targets can be compared recording by recording: the probe's pool is the first
``targets + BATCH`` recordings of the stream, or all D of them when the stream holds fewer, and
target n's neighbour k is ``pool[(n + 1 + k) % len(pool)]`` for k < BATCH - 1. The modulo does
wrap: test-other holds D = 2,939 recordings, the stock run asked for 2,939 + 32, so its pool is
the whole set and its last 31 targets (n >= 2,908) have neighbours from the start of the set.
Still, each of the first N targets has the same neighbours in both runs:

* N + BATCH <= D: the replay's pool is the first N + BATCH recordings and the stock's is at
  least that long. For n < N, n + 1 + k <= N + BATCH - 1, so neither modulo wraps and both
  pick the same recordings.
* N + BATCH > D: both pools are the whole set, the same length, so both wrap the same way.

The replay stamps its pool's length (``pool``); the stock record does not, so D = 2,939 is
inferred from the dataset's size.

Every check below is a difference when it fails:

* Neither record carries ``fake_pipeline`` other than false. A fake record is refused before
  anything else is compared: the CPU test fake wrote it, so it measures nothing.
* Settings. ``model``, ``model_revision``, ``chunk_ms``, ``batch``, ``matmul_precision``,
  ``att_context_size`` and ``arms`` are equal. The runs are the same dtypes, in the same order.
  The replay's ``att_context_size_observed``, read off its built encoder, is the stock's
  ``att_context_size`` at the top level and in every run. The replay asked for word confidence
  off, and its built decoder kept no step confidence (``word_confidence_observed`` in every
  run): the stock run predates the switch, so off is the configuration it ran.
* Targets. The replay's ``references`` are the stock's first N: the same ids, in the same order,
  with the same text.
* Per dtype and arm, over the targets n < N:
  - the divergent recordings are the same, and each divergence's ``n`` names the recording at
    that index in its own record's ``references``;
  - for each, ``text_differs``, ``a``, ``b`` and both sides' (word, start, end) timings are
    equal (the replay's fourth element, the confidence, is not compared);
  - ``text_divergent`` and ``timing_only_divergent`` are the stock's counts restricted to n < N,
    and agree with the replay's own divergences; ``checked`` is N;
  - every target's a and b texts, which the stock keeps in ``transcripts``, are the replay's, in
    its ``every_recording`` and in its ``transcripts``;
  - the replay agrees with itself: its ``every_recording`` holds exactly its N targets in order;
    a divergent recording's entry has the divergence's texts and timings, and its sides differ
    as ``text_differs`` says; any other recording has equal a and b texts and (word, start, end)
    timings.
* Per dtype, the repeat control's verdict is equal, and so are its counts over the targets both
  runs checked. The control checks the first ``min(REPEAT, targets)`` targets, so a replay of N
  at the stock's REPEAT checks the stock's first ``min(N, checked)``. The stock keeps only
  totals. A total restricts to that prefix exactly when it is 0 or equal to ``checked`` (the
  stock run's 8 of 8), or when the prefix is every checked target. Any other total cannot be
  restricted, and the comparator says so instead of guessing: replay at least the stock's
  ``checked`` targets.

Exit 0 only when there is no difference. Otherwise every difference is printed and the exit is
1, including when a record cannot be read or lacks what is compared. Nothing here touches a GPU,
the data or the model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_MATCH, EXIT_DIFFER = 0, 1
#: Settings both records stamp, compared as they are.
SETTINGS = (
    "model",
    "model_revision",
    "chunk_ms",
    "batch",
    "matmul_precision",
    "att_context_size",
    "arms",
)


def wse(timings: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """(word, start, end) of each timing; the stock stores three elements, the replay four."""
    return [list(t[:3]) for t in timings]


def refusals(stock: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    """A record stamped by the CPU fake, with ``fake_pipeline`` present and not false."""
    out = []
    for name, record in (("stock", stock), ("replay", replay)):
        fake = record.get("fake_pipeline", False)
        if fake is not False:
            out.append(
                f"refused: the {name} record has fake_pipeline {fake!r}; the CPU test fake wrote"
                " it, so it measures nothing"
            )
    return out


def _is_index(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def _divergences(
    arm: dict[str, Any], ids: list[str], below: float, where: str, diffs: list[str]
) -> dict[str, dict[str, Any]]:
    """The arm's divergences with n < ``below``, by recording id, each ``n`` checked against
    the record's own order of targets."""
    out: dict[str, dict[str, Any]] = {}
    for d in arm["divergences"]:
        n, rid = d.get("n"), d.get("librispeech_id")
        if not _is_index(n):
            diffs.append(f"{where}: divergence {rid!r} has index n {n!r}, not a target index")
            continue
        if n >= below:
            continue
        at = ids[n] if n < len(ids) else None
        if at != rid:
            diffs.append(f"{where}: divergence n={n} names {rid!r}, but target {n} is {at!r}")
        if rid in out:
            diffs.append(f"{where}: {rid!r} is listed as a divergence twice")
        out[rid] = d
    return out


def compare_arm(
    stock_arm: dict[str, Any],
    replay_arm: dict[str, Any],
    stock_ids: list[str],
    replay_ids: list[str],
    where: str,
) -> list[str]:
    """Every difference between one arm of the stock record, over its first N targets, and the
    same arm of the replay; and every way the replay's arm disagrees with itself."""
    diffs: list[str] = []
    count = len(replay_ids)
    stock_div = _divergences(stock_arm, stock_ids, count, f"{where} stock", diffs)
    replay_div = _divergences(replay_arm, replay_ids, math.inf, f"{where} replay", diffs)
    order = {rid: n for n, rid in enumerate(replay_ids)}

    def by_n(rids: set[str]) -> list[str]:
        return sorted(rids, key=lambda rid: (order.get(rid, count), rid))

    only_stock = by_n(set(stock_div) - set(replay_div))
    only_replay = by_n(set(replay_div) - set(stock_div))
    if only_stock or only_replay:
        diffs.append(
            f"{where}: divergent in the stock only {only_stock}, in the replay only {only_replay}"
        )
    for rid in by_n(set(stock_div) & set(replay_div)):
        s, r = stock_div[rid], replay_div[rid]
        for key in ("n", "text_differs", "a", "b"):
            if s.get(key) != r.get(key):
                diffs.append(f"{where} {rid}: {key} is {s.get(key)!r} in the stock, {r.get(key)!r}")
        for key in ("a_timings", "b_timings"):
            if wse(s[key]) != wse(r[key]):
                diffs.append(f"{where} {rid}: {key} (word, start, end) differ")

    stock_text = sum(1 for d in stock_div.values() if d.get("text_differs") is True)
    stock_timing = len(stock_div) - stock_text
    own_text = sum(1 for d in replay_div.values() if d.get("text_differs") is True)
    own_timing = len(replay_div) - own_text
    for key, stock_count, own in (
        ("text_divergent", stock_text, own_text),
        ("timing_only_divergent", stock_timing, own_timing),
    ):
        if replay_arm[key] != stock_count:
            diffs.append(
                f"{where}: {key} is {replay_arm[key]} in the replay, {stock_count} in the stock"
                f" over n < {count}"
            )
        if replay_arm[key] != own:
            diffs.append(
                f"{where}: the replay's {key} {replay_arm[key]} disagrees with its own"
                f" divergences ({own})"
            )
    if replay_arm["checked"] != count:
        diffs.append(f"{where}: the replay checked {replay_arm['checked']}, not its {count}")

    every = replay_arm["every_recording"]
    stock_transcripts, replay_transcripts = stock_arm["transcripts"], replay_arm["transcripts"]
    for rid in replay_ids:
        s = stock_transcripts.get(rid)
        if s is None:
            diffs.append(f"{where} {rid}: the stock record has no transcript")
            continue
        entry = every.get(rid)
        wanted = (s["a"], s.get("b", s["a"]))
        if entry is not None and (entry["a_text"], entry["b_text"]) != wanted:
            diffs.append(
                f"{where} {rid}: texts (a, b) are {wanted!r} in the stock,"
                f" {(entry['a_text'], entry['b_text'])!r} in the replay's every_recording"
            )
        if replay_transcripts.get(rid) != s:
            diffs.append(
                f"{where} {rid}: the transcript is {s!r} in the stock,"
                f" {replay_transcripts.get(rid)!r} in the replay"
            )

    if list(every) != replay_ids:
        diffs.append(
            f"{where}: the replay's every_recording holds {len(every)} recordings, not its"
            f" {count} targets in order"
        )
    for rid, entry in every.items():
        a_words, b_words = wse(entry["a_words"]), wse(entry["b_words"])
        d = replay_div.get(rid)
        if d is None:
            if entry["a_text"] != entry["b_text"] or a_words != b_words:
                diffs.append(
                    f"{where} {rid}: the replay lists no divergence, but its every_recording"
                    " sides differ"
                )
            continue
        agrees = (entry["a_text"], entry["b_text"]) == (d["a"], d["b"])
        if not agrees or (a_words, b_words) != (wse(d["a_timings"]), wse(d["b_timings"])):
            diffs.append(
                f"{where} {rid}: the replay's every_recording disagrees with its divergence"
            )
        text_differs = d["a"] != d["b"]
        if d.get("text_differs") is not text_differs or (
            not text_differs and wse(d["a_timings"]) == wse(d["b_timings"])
        ):
            diffs.append(
                f"{where} {rid}: the replay's divergence says text_differs"
                f" {d.get('text_differs')!r}, and its sides do not differ that way"
            )
    return diffs


REPEAT_COUNTS = ("alone_identical", "batch_identical")


def restricted_repeat(stock_rep: dict[str, Any], count: int) -> dict[str, int] | str:
    """The stock's repeat counts over its first ``min(count, checked)`` targets, which is what
    a replay of ``count`` targets at the same REPEAT checks; or why they cannot be known.

    The stock keeps only totals. Over a prefix of its checked targets a total is known exactly
    when it is 0 or all of them, or when the prefix is every checked target. Otherwise which of
    the targets the replay checks were the identical ones is not in the record."""
    checked = stock_rep["checked"]
    first = min(count, checked)
    out: dict[str, int] = {}
    for key in REPEAT_COUNTS:
        total = stock_rep[key]
        if first == checked or total == 0:
            out[key] = total
        elif total == checked:
            out[key] = first
        else:
            return (
                f"the stock's repeat {key} is {total} of {checked}, which cannot be restricted"
                f" to the replay's first {first}; replay at least {checked} targets"
            )
    out["checked"] = first
    return out


def differences(stock: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    """Every difference between the stock record and the replay over the replay's targets."""
    refused = refusals(stock, replay)
    if refused:
        return refused
    diffs: list[str] = []
    for key in SETTINGS:
        if stock.get(key) != replay.get(key):
            diffs.append(f"{key} is {stock.get(key)!r} in the stock, {replay.get(key)!r}")
    att = stock.get("att_context_size")
    if replay.get("att_context_size_observed") != att:
        diffs.append(
            f"the replay's att_context_size_observed is {replay.get('att_context_size_observed')!r}"
            f", not the stock's att_context_size {att!r}"
        )
    if replay.get("word_confidence") != "off":
        diffs.append(
            f"the replay asked for word confidence {replay.get('word_confidence')!r}, not off"
        )
    if list(stock["runs"]) != list(replay["runs"]):
        diffs.append(f"the runs are {list(stock['runs'])} in the stock, {list(replay['runs'])}")

    count = replay.get("targets")
    if not _is_index(count) or count == 0 or count > stock["targets"]:
        diffs.append(
            f"the replay's targets is {count!r}; it must be 1 to the stock's {stock['targets']}"
        )
        return diffs
    stock_refs = list(stock["references"].items())
    replay_refs = list(replay["references"].items())
    if replay_refs != stock_refs[:count]:
        first = next(
            (n for n, pair in enumerate(replay_refs) if n >= count or pair != stock_refs[n]),
            min(len(replay_refs), count),
        )
        diffs.append(
            f"the replay's references are not the stock's first {count}: they part at target"
            f" {first}"
        )
    stock_ids = [rid for rid, _ in stock_refs]
    replay_ids = [rid for rid, _ in replay_refs]

    for dtype in (d for d in stock["runs"] if d in replay["runs"]):
        stock_run, replay_run = stock["runs"][dtype], replay["runs"][dtype]
        observed_att = replay_run.get("att_context_size_observed")
        if observed_att != att:
            diffs.append(
                f"{dtype}: the replay's att_context_size_observed is {observed_att!r}, not the"
                f" stock's att_context_size {att!r}"
            )
        kept = (replay_run.get("word_confidence_observed") or {}).get("decoder_step_confidence")
        if kept is not False:
            diffs.append(
                f"{dtype}: the replay's built decoder kept step confidence {kept!r}, not False"
            )
        wanted = restricted_repeat(stock_run["repeat"], count)
        if isinstance(wanted, str):
            diffs.append(f"{dtype}: {wanted}")
        elif replay_run.get("repeat") != wanted:
            diffs.append(
                f"{dtype}: repeat is {wanted!r} in the stock over its first"
                f" {wanted['checked']} targets, {replay_run.get('repeat')!r} in the replay"
            )
        if stock_run.get("repeat_verdict") != replay_run.get("repeat_verdict"):
            diffs.append(
                f"{dtype}: repeat_verdict is {stock_run.get('repeat_verdict')!r} in the stock,"
                f" {replay_run.get('repeat_verdict')!r} in the replay"
            )
        for arm, stock_arm in stock_run["arms"].items():
            if arm not in replay_run["arms"]:
                diffs.append(f"{dtype}/{arm}: the replay has no such arm")
                continue
            diffs += compare_arm(
                stock_arm, replay_run["arms"][arm], stock_ids, replay_ids, f"{dtype}/{arm}"
            )
        for arm in replay_run["arms"]:
            if arm not in stock_run["arms"]:
                diffs.append(f"{dtype}/{arm}: the stock has no such arm")
    return diffs


def summary(replay: dict[str, Any]) -> list[str]:
    """One line per dtype and arm of what matched."""
    lines = []
    for dtype, run in replay["runs"].items():
        for arm, a in run["arms"].items():
            lines.append(
                f"{dtype}/{arm}: {a['checked']} targets, text {a['text_divergent']},"
                f" timing-only {a['timing_only_divergent']}: the same recordings, texts and"
                " timings as the stock record"
            )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stock", type=Path, help="the stock record, written at 6583a83")
    parser.add_argument("replay", type=Path, help="the replay record, written by the probe now")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        stock = json.loads(args.stock.read_text())
        replay = json.loads(args.replay.read_text())
        diffs = differences(stock, replay)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        diffs = [f"cannot compare: {type(exc).__name__}: {exc}"]
    if diffs:
        for line in diffs:
            print(f"DIFFERENCE: {line}")
        print(
            f"VERDICT: NOT reproduced: {len(diffs)} difference(s) between the replay and the"
            " stock record"
        )
        return EXIT_DIFFER
    for line in summary(replay):
        print(line)
    print(
        f"VERDICT: reproduced: over its first {replay['targets']} targets the replay gives"
        " exactly the stock record's answers"
    )
    return EXIT_MATCH


if __name__ == "__main__":
    raise SystemExit(main())
