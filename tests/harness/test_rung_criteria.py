# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""What a rung establishes from one executed load, and what it refuses to claim.

`rung_from_run` is a pure reduction from a load result to a rung, so everything here
except the last test builds the run by hand instead of spending four minutes of wall
clock on one. Each criterion gets a test that reddens if the rung stops evaluating it,
and thermal gets one that reddens if the rung ever starts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from verbatim_bench import constants
from verbatim_bench.client import SessionResult
from verbatim_bench.env import fake_gpu_facts
from verbatim_bench.ladder import (
    Criterion,
    InvalidReason,
    RungPlan,
    integrity_counts,
    pacing_slip_validity,
    rung_from_run,
    rung_validity,
)
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.results import RunResult
from verbatim_bench.wer import Batch1Reference

pytestmark = pytest.mark.cpu

SEED = 20260914
THRESHOLD_MS = 160 + constants.X_MS

#: The phases of a hand-built run. The window is the frozen length and the warm-up is the
#: frozen floor, so a run assembled from these is canonical and the criteria tests are
#: not silently reading a rung that ran nothing.
ALL_LIVE_S = 10.0
WINDOW_OPEN_S = ALL_LIVE_S + constants.WARM_UP_S
WINDOW_CLOSE_S = WINDOW_OPEN_S + constants.WINDOW_S

#: A reference and a transcript that agree word for word, so a clean run's corpus WER is
#: zero and the WER criterion is decided by the reference it is compared against.
SPOKEN = "the quick brown fox jumps over the lazy dog"


def _session(
    index: int = 0,
    *,
    final_text: str | None = None,
    finals_received: int = 1,
    error: str | None = None,
    acknowledged: bool = True,
    latency_ms: float = 20.0,
    slip_ms: float = 1.0,
) -> SessionResult:
    """One session result, inside the window, with four samples and ten slip readings."""
    session = SessionResult(session_id=f"s{index:04d}-0000", stream_id=f"utt-{index}")
    session.server_session_id = f"null-{index}" if acknowledged else None
    session.started_at_s = WINDOW_OPEN_S
    session.chunks = 4
    session.partial_ms = [latency_ms] * 4
    session.partial_recv_s = [WINDOW_OPEN_S + 1.0 + step for step in range(4)]
    session.pacing_slip_ms = [slip_ms] * 10
    session.partials_received = 4
    session.finals_received = finals_received
    session.final_text = SPOKEN if final_text is None else final_text
    session.reference_text = SPOKEN
    session.error = error
    return session


def _run(sessions: list[SessionResult], **overrides: object) -> RunResult:
    """A converged, canonical run holding `sessions`, with its phases filled in."""
    fields: dict = {
        "spec_dict": {
            "window_s": float(constants.WINDOW_S),
            "warm_up_s": float(constants.WARM_UP_S),
            "warm_up_reading_s": float(constants.WARM_UP_READING_S),
            "warm_up_convergence": float(constants.WARM_UP_CONVERGENCE),
            "warm_up_cap_s": float(constants.WARM_UP_CAP_S),
        },
        "sessions": sessions,
        "wall_clock_s": WINDOW_CLOSE_S,
        "all_live_at_s": ALL_LIVE_S,
        "warm_up_end_s": WINDOW_OPEN_S,
        "window_open_s": WINDOW_OPEN_S,
        "window_close_s": WINDOW_CLOSE_S,
        "warm_up_converged": True,
    }
    fields.update(overrides)
    return RunResult(**fields)


def _rung(sessions: list[SessionResult], *, batch1_wer: float | None = None, **overrides: object):
    return rung_from_run(
        _run(sessions, **overrides),
        plan=RungPlan(n=len(sessions), seed=SEED),
        threshold_ms=THRESHOLD_MS,
        batch1_wer=batch1_wer,
    )


def test_a_clean_run_evaluates_every_criterion_the_data_supports() -> None:
    """Latency, word error rate and all three integrity counts, in the document's order."""
    rung = _rung([_session(index) for index in range(4)], batch1_wer=0.0)
    assert rung.criteria_evaluated == (
        Criterion.LATENCY,
        Criterion.WER,
        Criterion.INTEGRITY_REFUSED,
        Criterion.INTEGRITY_DROPPED,
        Criterion.INTEGRITY_NO_FINAL,
    )
    assert rung.first_failing_criterion is None
    assert rung.valid is True
    assert rung.canonical_window is True
    assert (rung.sessions_refused, rung.sessions_dropped, rung.sessions_without_final) == (0, 0, 0)


def test_thermal_is_never_evaluated_so_no_rung_can_pass() -> None:
    """The criterion nobody collects, and the guard against pretending otherwise.

    Zero GPU throttle events over the window needs a collection this harness does not
    have. A rung that is otherwise perfect still does not pass, and thermal is the one
    thing missing. If a later change fakes it, this reddens.
    """
    rung = _rung([_session(index) for index in range(4)], batch1_wer=0.0)
    assert Criterion.THERMAL not in rung.criteria_evaluated
    assert rung.criteria_unevaluated == {Criterion.THERMAL}
    assert rung.passed is False
    # And it is not smuggled in as a failure either: nothing evaluated failed.
    assert rung.first_failing_criterion is None


def test_word_error_rate_is_evaluated_only_when_a_batch1_reference_is_supplied() -> None:
    """The same load, read twice. Without a reference there is nothing to compare to.

    This reddens two ways: if the rung stops evaluating WER when a reference is present,
    and if it starts claiming WER when none is.
    """
    sessions = [_session(index) for index in range(3)]
    without = _rung(sessions)
    assert Criterion.WER not in without.criteria_evaluated
    assert without.wer_vs_batch1 is None
    with_reference = _rung(sessions, batch1_wer=0.05)
    assert Criterion.WER in with_reference.criteria_evaluated
    # Every transcript matched its reference, so the corpus WER is zero and the rung sits
    # a flat 0.05 below the reference it was compared against.
    assert with_reference.wer_vs_batch1 == pytest.approx(-0.05)
    assert with_reference.first_failing_criterion is None


def test_a_corpus_wer_outside_the_frozen_window_fails_the_wer_criterion() -> None:
    """Nine reference words, nine wrong words back: corpus WER 1.0 against a 0.05 batch-1."""
    sessions = [_session(index, final_text="a b c d e f g h i") for index in range(3)]
    rung = _rung(sessions, batch1_wer=0.05)
    assert rung.wer_vs_batch1 == pytest.approx(0.95)
    assert rung.first_failing_criterion is Criterion.WER
    assert Criterion.WER in rung.criteria_evaluated
    assert rung.passed is False
    # Inside the window the same load passes the criterion, so the threshold is what is
    # being tested and not merely the arithmetic.
    one_word_wrong = "the quick brown fox jumps over the lazy cat"
    inside = _rung(
        [_session(index, final_text=one_word_wrong) for index in range(3)],
        batch1_wer=0.05,
    )
    assert inside.wer_vs_batch1 == pytest.approx(1.0 / 9.0 - 0.05)
    assert inside.first_failing_criterion is None


def test_a_corpus_with_no_reference_words_leaves_word_error_rate_unevaluated() -> None:
    """Undefined is not zero: nothing to divide by is nothing to compare."""
    sessions = [_session(index) for index in range(2)]
    for session in sessions:
        session.reference_text = ""
    rung = _rung(sessions, batch1_wer=0.05)
    assert Criterion.WER not in rung.criteria_evaluated
    assert rung.wer_vs_batch1 is None


def test_streams_that_ended_without_a_final_are_counted_and_fail_integrity() -> None:
    """`sessions_without_final` was the literal 0. It is now a count of `finals_received`.

    The count is only meaningful because the load client reads every final a stream
    carries: while it returned on the first, `finals_received` could not exceed one.
    """
    sessions = [_session(0), _session(1, finals_received=0), _session(2, finals_received=0)]
    rung = _rung(sessions, batch1_wer=0.0)
    assert rung.sessions_without_final == 2
    assert Criterion.INTEGRITY_NO_FINAL in rung.criteria_evaluated
    assert rung.first_failing_criterion is Criterion.INTEGRITY_NO_FINAL
    # A stream endpointed several times carries several finals and is not a failure.
    many = _rung([_session(0, finals_received=4), _session(1, finals_received=2)], batch1_wer=0.0)
    assert many.sessions_without_final == 0
    assert many.first_failing_criterion is None


def test_a_stream_the_server_accepted_and_lost_is_dropped_rather_than_refused() -> None:
    """`sessions_dropped` was the literal 0, and every failure was filed as a refusal.

    The server's own session acknowledgement is the line: a stream that failed without
    one never got in, and one that failed after it was admitted and then lost.
    """
    dropped = _rung(
        [_session(0), _session(1, error="ServerError overload: gone", acknowledged=True)],
        batch1_wer=0.0,
    )
    assert (dropped.sessions_refused, dropped.sessions_dropped) == (0, 1)
    assert Criterion.INTEGRITY_DROPPED in dropped.criteria_evaluated
    assert dropped.first_failing_criterion is Criterion.INTEGRITY_DROPPED

    refused = _rung(
        [
            _session(0),
            _session(
                1,
                error="ConnectionRefusedError: [Errno 111]",
                acknowledged=False,
                finals_received=0,
            ),
        ],
        batch1_wer=0.0,
    )
    assert (refused.sessions_refused, refused.sessions_dropped) == (1, 0)
    # A stream that never got in also never produced a final, and both are true of it. The
    # counts overlap and the diagnosis order picks the refusal, which is the cause.
    assert refused.sessions_without_final == 1
    assert refused.first_failing_criterion is Criterion.INTEGRITY_REFUSED


def test_integrity_counts_read_one_session_three_ways() -> None:
    """The counts are not disjoint, on purpose, and each answers its own question."""
    never_got_in = _session(
        0, error="TimeoutError: no session message", acknowledged=False, finals_received=0
    )
    counts = integrity_counts([never_got_in])
    assert counts.refused == 1
    assert counts.dropped == 0
    assert counts.without_final == 1


def test_a_window_p95_over_the_threshold_still_fails_on_latency() -> None:
    rung = _rung([_session(index, latency_ms=THRESHOLD_MS + 1.0) for index in range(3)])
    assert rung.p95_ms == pytest.approx(THRESHOLD_MS + 1.0)
    assert rung.first_failing_criterion is Criterion.LATENCY
    assert Criterion.LATENCY in rung.criteria_evaluated


def test_a_window_that_produced_no_samples_does_not_evaluate_latency() -> None:
    sessions = [_session(index) for index in range(2)]
    for session in sessions:
        session.partial_ms = []
        session.partial_recv_s = []
    rung = _rung(sessions)
    assert Criterion.LATENCY not in rung.criteria_evaluated
    assert rung.p95_ms == float("inf")


def test_a_pooled_pacing_slip_p99_over_the_frozen_tolerance_makes_the_rung_invalid() -> None:
    """The validity check, on the data every session already carries.

    A generator that missed its own send schedule did not deliver the paced real-time
    load the rung names, so the latency that came back is not an answer to it. That is
    host fitness, not a server failure: the rung is invalid and the ladder re-runs it.
    """
    over = constants.PACING_SLIP_P99_MAX_MS + 0.1
    rung = _rung([_session(index, slip_ms=over) for index in range(3)], batch1_wer=0.0)
    assert rung.valid is False
    assert rung.invalid_reason is InvalidReason.PACING_SLIP
    # An invalid rung is discarded whole, so it claims no criterion and reports no
    # percentile: the window it would have come from is not a measurement.
    assert rung.criteria_evaluated == ()
    assert rung.first_failing_criterion is None
    assert rung.p95_ms == float("inf")
    assert rung.passed is False

    under = _rung(
        [_session(index, slip_ms=constants.PACING_SLIP_P99_MAX_MS) for index in range(3)],
        batch1_wer=0.0,
    )
    assert under.valid is True
    assert under.invalid_reason is None
    assert Criterion.LATENCY in under.criteria_evaluated


def test_one_session_in_a_hundred_over_the_tolerance_does_not_reach_the_p99() -> None:
    """It is a pooled p99 and not a maximum, so a single late frame is not a verdict."""
    sessions = [_session(index) for index in range(10)]
    sessions[0].pacing_slip_ms[0] = constants.PACING_SLIP_P99_MAX_MS * 10
    rung = _rung(sessions, batch1_wer=0.0)
    assert rung.valid is True


def test_the_pacing_tolerance_has_exactly_one_definition(monkeypatch) -> None:
    """`rung_validity` defers to `pacing_slip_validity` rather than repeating the number.

    A rung executed from its load alone, which is what the ladder does, applies the same
    tolerance as one executed against a full host record, because there is one place the
    constant is read.

    The host record's pressure thresholds are still unfrozen, so `rung_validity` returns
    `PSI_THRESHOLD_UNFROZEN` before it reaches anything else and the shared tolerance
    cannot be observed through it. The patch here stands in for the day they are frozen;
    `test_an_unfrozen_pressure_threshold_makes_the_rung_invalid` covers the short circuit
    itself.
    """
    assert pacing_slip_validity(constants.PACING_SLIP_P99_MAX_MS) is None
    just_over = constants.PACING_SLIP_P99_MAX_MS + 1e-9
    assert pacing_slip_validity(just_over) is InvalidReason.PACING_SLIP
    from test_ladder import _counters

    monkeypatch.setattr(constants, "PSI_CPU_SOME_MAX_PCT", 100.0)
    monkeypatch.setattr(constants, "PSI_CPU_FULL_MAX_PCT", 100.0)
    assert (
        rung_validity(_counters(), fake_gpu_facts(), constants.PACING_SLIP_P99_MAX_MS + 1.0)
        is InvalidReason.PACING_SLIP
    )
    assert rung_validity(_counters(), fake_gpu_facts(), constants.PACING_SLIP_P99_MAX_MS) is None


def test_a_warm_up_that_never_converged_establishes_nothing_and_is_not_invalid() -> None:
    rung = _rung([_session(index) for index in range(2)], batch1_wer=0.0, warm_up_converged=False)
    assert rung.first_failing_criterion is Criterion.UNSTABLE
    assert rung.criteria_evaluated == ()
    assert rung.valid is True
    assert rung.invalid_reason is None


def test_a_reference_is_refused_unless_it_describes_this_run() -> None:
    """The criterion is defined at one `(checkpoint, chunk, corpus_id, dtype)` and no other."""
    reference = Batch1Reference(
        checkpoint="ck", chunk_ms=160, corpus_id="sha256:abc", dtype="fp32_tf32", wer=0.05
    )
    assert reference.describes(
        checkpoint="ck", chunk_ms=160, corpus_id="sha256:abc", dtype="fp32_tf32"
    )
    for wrong in (
        {"checkpoint": "other"},
        {"chunk_ms": 560},
        {"corpus_id": "sha256:def"},
        {"dtype": "bf16"},
    ):
        coordinates = {
            "checkpoint": "ck",
            "chunk_ms": 160,
            "corpus_id": "sha256:abc",
            "dtype": "fp32_tf32",
        }
        coordinates.update(wrong)
        assert not reference.describes(**coordinates)  # type: ignore[arg-type]


async def test_the_canonical_framing_the_harness_ships_is_not_invalidated_by_its_own_jitter(
    tmp_path,
) -> None:
    """The gate on the configuration the project actually publishes, end to end.

    Until 2026-09-11 a load sending the frozen 20 ms frames drew a jitter offset for
    each frame's deadline while sleeping to the unjittered one, so the pooled
    pacing-slip p99 was the jitter constant and every canonical rung came back invalid
    for pacing whatever the box did. The jitter is now on the wire and the slip is
    graded against the deadline the sender slept to, so a rung's pacing validity is
    the generator's own lateness on this box: against the null server a one-stream
    rung is not invalid for pacing, and the ladder gets to evaluate its criteria.
    """
    from test_ladder import FAST_RUNG
    from test_ladder import SEED as LADDER_SEED
    from test_pace import make_manifest, make_wav
    from verbatim_bench.cli import main

    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    out_dir = tmp_path / "ladder-out"
    canonical_framing = [arg for arg in FAST_RUNG if arg not in ("--frame-ms", "160")]
    assert "--frame-ms" not in canonical_framing
    async with NullServer(NullServerConfig(partial_delay_ms=20.0)) as server:
        argv = [
            "ladder",
            "--endpoint",
            server.endpoint,
            "--manifest",
            str(manifest),
            "--arm",
            "a",
            "--n0",
            "1",
            "--seeds",
            str(LADDER_SEED),
            "--out",
            str(out_dir),
            *canonical_framing,
        ]
        await asyncio.get_running_loop().run_in_executor(None, main, argv)
    payload = json.loads((out_dir / "ladder.json").read_text(encoding="utf-8"))
    assert payload["config"]["frame_ms"] == constants.FRAME_MS
    assert payload["rungs"]
    for rung in payload["rungs"]:
        assert rung["invalid_reason"] != InvalidReason.PACING_SLIP.value


async def test_a_ladder_given_a_batch1_reference_evaluates_word_error_rate(tmp_path) -> None:
    """The whole path: a reference document on disk reaches the rung's criteria list."""
    from test_ladder import FAST_RUNG
    from test_ladder import SEED as LADDER_SEED
    from test_pace import make_manifest, make_wav
    from verbatim_bench.cli import main
    from verbatim_bench.corpus import manifest_corpus_id

    wav = tmp_path / "utt.wav"
    make_wav(wav, 1.0)
    manifest = make_manifest(tmp_path / "m.jsonl", [wav], ["utt-0"])
    out_dir = tmp_path / "ladder-out"
    reference_path = tmp_path / "batch1.json"
    document = {
        "checkpoint": "null-server",
        "chunk_ms": 160,
        "corpus_id": manifest_corpus_id(manifest),
        "dtype": "fp32_tf32",
        # The null server answers with its own fixed text, so the corpus WER of this run
        # is 1.0 against the manifest's reference. A reference of 1.0 puts the rung on it.
        "wer": 1.0,
    }
    reference_path.write_text(json.dumps(document), encoding="utf-8")
    argv_tail = [
        "--wer-batch1",
        str(reference_path),
        "--checkpoint",
        "null-server",
        "--dtype",
        "fp32_tf32",
    ]
    async with NullServer(NullServerConfig(partial_delay_ms=20.0)) as server:
        argv = [
            "ladder",
            "--endpoint",
            server.endpoint,
            "--manifest",
            str(manifest),
            "--arm",
            "a",
            "--n0",
            "1",
            "--seeds",
            str(LADDER_SEED),
            "--out",
            str(out_dir),
            *FAST_RUNG,
            *argv_tail,
        ]
        rc = await asyncio.get_running_loop().run_in_executor(None, main, argv)
    assert rc == 0
    payload = json.loads((out_dir / "ladder.json").read_text(encoding="utf-8"))
    assert payload["config"]["wer_batch1"] == document
    assert payload["rungs"]
    for rung in payload["rungs"]:
        assert "wer" in rung["criteria_evaluated"]
        assert rung["wer_vs_batch1"] == pytest.approx(0.0)


def test_a_reference_that_does_not_describe_the_run_is_a_usage_error(tmp_path) -> None:
    """Not a silent skip, which would look exactly like having supplied no reference."""
    from verbatim_bench.cli import main

    manifest = tmp_path / "m.jsonl"
    manifest.write_text("", encoding="utf-8")
    reference_path = tmp_path / "batch1.json"
    reference_path.write_text(
        json.dumps(
            {
                "checkpoint": "ck",
                "chunk_ms": 560,
                "corpus_id": "sha256:not-this-corpus",
                "dtype": "fp32_tf32",
                "wer": 0.05,
            }
        ),
        encoding="utf-8",
    )
    base = [
        "ladder",
        "--no-host-record",
        "--endpoint",
        "ws://127.0.0.1:1/v1/stream",
        "--manifest",
        str(manifest),
        "--arm",
        "a",
        "--out",
        str(tmp_path / "out"),
        "--wer-batch1",
        str(reference_path),
    ]
    # The coordinates disagree, so the comparison never happens and the run says so.
    assert main([*base, "--checkpoint", "ck", "--dtype", "fp32_tf32"]) == 1
    # And a reference without the coordinates to check it against is refused outright.
    assert main(base) == 1
    assert not (Path(tmp_path / "out") / "ladder.json").exists()
