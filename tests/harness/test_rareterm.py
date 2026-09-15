# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The rare-term harness: what a phrase list buys, and what it costs.

The scoring is pure and tested here against hand-built pairs. The run is tested against
an in-process server over the CPU fake, which honours a session's phrase list by mixing
its digest into its own tokens — enough to show the arms are actually different runs and
that the record carries each one, which is what the harness has to get right. What a real
boosting tree recovers is a GPU measurement and is not asserted on CPU.

The property the tests exist to defend is that the measurement cannot be rigged. Terms are
chosen by document frequency over the references, before a session runs, so a term set
cannot be selected because the bare pass failed on it — which would make any improvement
a consequence of the selection rather than of the phrase list.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
from verbatim_bench.client import ChunkMode as BenchChunk
from verbatim_bench.invariance import Clip
from verbatim_bench.rareterm import (
    DEFAULT_BOOSTS,
    ArmReading,
    MissKind,
    RareTermReport,
    TermCount,
    TermSet,
    TermSetError,
    _alignment_ops,
    classify_misses,
    derive_terms,
    load_term_set,
    occurrences,
    run_rare_terms,
    score_transcript,
)

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import Engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.ws.server import WsServer, WsServerConfig

pytestmark = pytest.mark.cpu

CHUNK_MS = 80
BUCKET = 8


# --- scoring, pure -----------------------------------------------------------------


def test_occurrences_counts_multi_word_terms_without_overlapping() -> None:
    assert (
        occurrences(("a", "risdon", "graeme", "b", "risdon", "graeme"), ("risdon", "graeme")) == 2
    )
    assert occurrences(("na", "na", "na"), ("na", "na")) == 1  # non-overlapping
    assert occurrences(("short",), ("too", "long", "a", "term")) == 0
    assert occurrences(("anything",), ()) == 0


def test_a_term_the_audio_contained_and_the_transcript_missed_is_a_miss() -> None:
    counts = score_transcript("sir risdon graeme spoke", "sir risdongram spoke", ("risdon graeme",))
    assert counts["risdon graeme"] == TermCount(hits=0, misses=1, false_accepts=0)


def test_a_term_the_audio_did_not_contain_and_the_transcript_produced_is_a_false_accept() -> None:
    counts = score_transcript("he could not help it", "he could not metformin", ("metformin",))
    assert counts["metformin"] == TermCount(hits=0, misses=0, false_accepts=1)


def test_a_term_repeated_more_often_than_the_reference_used_it_counts_the_surplus() -> None:
    """The over-boosting shape, measured. A weight that turned one clip into
    'french french french' on a B300 would otherwise have scored a hit and said nothing
    about the other two."""
    counts = score_transcript("the french call", "french french french", ("french",))
    assert counts["french"] == TermCount(hits=1, misses=0, false_accepts=2)


def test_a_term_in_neither_contributes_to_nothing() -> None:
    """A large term set must not dilute the numbers with zeros."""
    counts = score_transcript("nothing here", "nothing here", ("metformin", "certiorari"))
    assert all(count == TermCount() for count in counts.values())


def test_matching_is_case_and_punctuation_insensitive() -> None:
    counts = score_transcript("Archy, silent.", "archy silent", ("Archy",))
    assert counts["Archy"].hits == 1


def test_recall_and_precision_are_none_rather_than_zero_when_undefined() -> None:
    """Undefined is not zero: a run with no term occurrences says nothing about recall,
    and reporting 0.0 there would award a failing score for measuring nothing."""
    assert TermCount().recall is None
    assert TermCount().precision is None
    assert TermCount(hits=3, misses=1).recall == 0.75
    assert TermCount(hits=3, false_accepts=1).precision == 0.75


# --- the term set is chosen blind to the transcripts -------------------------------


def test_terms_are_the_corpus_rare_words_by_document_frequency() -> None:
    references = [
        "the quick brownfoxes jumped",
        "the lazy brownfoxes slept",
        "unique kangaroos abound",
    ]
    # brownfoxes is long enough but appears in two utterances; at a limit of one it goes.
    assert derive_terms(references, min_chars=7, max_document_frequency=1) == ("kangaroos",)
    assert derive_terms(references, min_chars=7, max_document_frequency=2) == (
        "brownfoxes",
        "kangaroos",
    )


def test_the_derivation_never_looks_at_a_transcript() -> None:
    """The rule takes references only. A term set that could see the transcripts could be
    selected on the model's failures, and the improvement would then be guaranteed."""
    references = ["sir risdon graeme has smuggled goods"]
    first = derive_terms(references, max_document_frequency=1)
    assert first == derive_terms(references, max_document_frequency=1)
    assert "risdon" not in first  # six characters: below the length floor, whatever happened


def test_derived_terms_keep_first_appearance_order_and_respect_the_limit() -> None:
    references = ["alphabet beetroot", "crocodile"]
    assert derive_terms(references, min_chars=7, max_document_frequency=1) == (
        "alphabet",
        "beetroot",
        "crocodile",
    )
    assert derive_terms(references, min_chars=7, max_document_frequency=1, limit=2) == (
        "alphabet",
        "beetroot",
    )


def test_a_term_set_that_says_nothing_is_refused() -> None:
    with pytest.raises(TermSetError, match="no terms"):
        TermSet(name="t", terms=())
    with pytest.raises(TermSetError, match="normalises to no words"):
        TermSet(name="t", terms=("...",))
    with pytest.raises(TermSetError, match="repeats a term"):
        TermSet(name="t", terms=("one", "one"))


def test_the_term_set_digest_follows_the_terms_and_their_order() -> None:
    same = TermSet(name="t", terms=("one", "two"))
    assert same.digest == TermSet(name="other", terms=("one", "two")).digest
    assert same.digest != TermSet(name="t", terms=("two", "one")).digest


def test_a_term_set_is_read_from_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"name": "clinic", "terms": ["metformin"]}))
    assert load_term_set(path).terms == ("metformin",)
    path.write_text(json.dumps({"name": "clinic"}))
    with pytest.raises(TermSetError, match="must be a list of strings"):
        load_term_set(path)


# --- the report --------------------------------------------------------------------


def _reading(
    boost: float | None,
    hits: int,
    misses: int,
    false_accepts: int,
    *,
    moved: int = 1,
) -> ArmReading:
    return ArmReading(
        boost=boost,
        counts=TermCount(hits, misses, false_accepts),
        changed_streams=() if boost is None else tuple(f"s{i}" for i in range(moved)),
    )


def _report(*arms: ArmReading) -> RareTermReport:
    return RareTermReport(
        endpoint="ws://x",
        chunk_ms=160,
        corpus={"kind": "manifest", "utterances": 4},
        term_set=TermSet(name="t", terms=("alphabet",)),
        arms=list(arms),
    )


def test_a_run_without_a_bare_arm_supports_no_reading() -> None:
    """Every arm is read against bare. Without it there is nothing to read against, and
    a recall figure alone says nothing about what the phrase list changed."""
    assert _report(_reading(2.0, 5, 5, 0)).usable is False


def test_a_run_with_no_term_occurrences_supports_no_reading() -> None:
    """A corpus whose audio contains none of the terms cannot measure recall on them,
    and a run that reported 'no false accepts' from it would be measuring nothing."""
    report = _report(_reading(None, 0, 0, 0), _reading(2.0, 0, 0, 0))
    assert report.usable is False
    assert "no reading" in report.render()


def test_an_errored_session_voids_the_arm() -> None:
    bare = _reading(None, 5, 5, 0)
    boosted = _reading(2.0, 8, 2, 1)
    boosted.errors["clip-1"] = "connection reset"
    assert _report(bare, boosted).usable is False


def test_the_report_names_what_was_gained_and_what_it_cost() -> None:
    report = _report(_reading(None, 5, 5, 0), _reading(2.0, 8, 2, 3))
    assert report.usable is True
    rendered = report.render()
    assert "+3 term occurrences recovered" in rendered
    assert "+3 false accepts" in rendered
    document = report.to_json_dict()
    assert document["record"] == "vb-rare-terms/1"
    assert [arm["boost"] for arm in document["arms"]] == [None, 2.0]
    assert document["arms"][1]["terms"]["recall"] == 0.8


# --- the run, against a server that honours phrase lists ---------------------------


@asynccontextmanager
async def _server() -> AsyncIterator[str]:
    chunk = ChunkMode(CHUNK_MS)
    config = EngineConfig(chunk=chunk, buckets=(BUCKET,), idle_timeout_s=None)
    pipeline = FakePipelineAdapter(chunk, buckets=(BUCKET,), honour_phrases=True)
    engine = Engine(config, pipeline)
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        yield server.endpoint


def _clips(n: int = 4, chunks: int = 3) -> list[Clip]:
    out: list[Clip] = []
    samples = chunks * CHUNK_MS * 16
    for index in range(n):
        rng = np.random.default_rng([17, index])
        pcm = (rng.uniform(-0.5, 0.5, size=samples) * 32767.0).astype("<i2").tobytes()
        out.append(
            Clip(
                stream_id=f"clip-{index:03d}",
                pcm=pcm,
                duration_s=samples / 16000,
                text="the appellant received hydrochlorothiazide",
            )
        )
    return out


@pytest.mark.asyncio
async def test_each_weight_is_its_own_pass_and_the_record_carries_them_all() -> None:
    term_set = TermSet(name="t", terms=("hydrochlorothiazide", "appellant"))
    async with _server() as endpoint:
        report = await run_rare_terms(
            endpoint,
            _clips(),
            term_set,
            chunk=BenchChunk(CHUNK_MS),
            boosts=(None, 1.0, 4.0),
            concurrency=2,
        )
    assert [arm.boost for arm in report.arms] == [None, 1.0, 4.0]
    assert all(not arm.errors for arm in report.arms)
    assert report.term_set.digest == term_set.digest
    # Every arm ran the whole corpus, and each recorded its own word error rate.
    assert all(arm.wer.reference_words > 0 for arm in report.arms)
    document = report.to_json_dict()
    assert len(document["arms"]) == 3
    assert document["term_set"]["terms"] == list(term_set.terms)


@pytest.mark.asyncio
async def test_the_bare_arm_puts_no_phrase_parameter_on_the_wire() -> None:
    """Not the list at weight zero, and not the list with the weight left off: nothing.

    A server with no phrase list and one biasing at zero are different configurations,
    and the bare arm has to be the one a deployment without a list actually runs -- which
    is also the only arm a server started without --biasing would accept rather than
    refuse. Counted at the adapter, because what the arm believes it sent is not evidence
    of what went on the wire.
    """
    chunk = ChunkMode(CHUNK_MS)
    config = EngineConfig(chunk=chunk, buckets=(BUCKET,), idle_timeout_s=None)
    pipeline = FakePipelineAdapter(chunk, buckets=(BUCKET,), honour_phrases=True)
    engine = Engine(config, pipeline)
    clips = _clips()
    term_set = TermSet(name="t", terms=("appellant",))
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        bare_only = await run_rare_terms(
            server.endpoint, clips, term_set, chunk=BenchChunk(CHUNK_MS), boosts=(None,)
        )
        assert pipeline.phrase_sessions == 0
        with_list = await run_rare_terms(
            server.endpoint, clips, term_set, chunk=BenchChunk(CHUNK_MS), boosts=(None, 2.0)
        )
    # The second sweep added exactly one boosted arm: one phrase-carrying session per
    # clip, and its bare arm carried none either.
    assert pipeline.phrase_sessions == len(clips)
    assert bare_only.arms[0].changed_streams == ()
    assert with_list.arms[0].changed_streams == ()
    assert len(with_list.arms[1].changed_streams) == len(clips)
    assert with_list.reached_the_decoder is True


@pytest.mark.asyncio
async def test_a_run_needs_at_least_one_boosted_arm() -> None:
    term_set = TermSet(name="t", terms=("appellant",))
    async with _server() as endpoint:
        report = await run_rare_terms(
            endpoint, _clips(), term_set, chunk=BenchChunk(CHUNK_MS), boosts=(None,)
        )
    assert report.usable is False


def test_the_default_sweep_starts_bare() -> None:
    assert DEFAULT_BOOSTS[0] is None
    assert all(boost is not None for boost in DEFAULT_BOOSTS[1:])


@pytest.mark.asyncio
async def test_a_server_that_ignores_phrase_lists_supports_no_reading() -> None:
    """The guard that keeps this from being a measurement of nothing. A server ignoring
    every list returns the bare arm's transcripts at every weight, so recall is identical
    everywhere and the run reads as "boosting did nothing" when the truth is "nothing was
    boosted". Those are different findings and only one is about the weight."""
    chunk = ChunkMode(CHUNK_MS)
    config = EngineConfig(chunk=chunk, buckets=(BUCKET,), idle_timeout_s=None)
    # honour_phrases defaults to False: exactly a decoder built without the biasing arena.
    engine = Engine(config, FakePipelineAdapter(chunk, buckets=(BUCKET,)))
    term_set = TermSet(name="t", terms=("appellant",))
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        report = await run_rare_terms(
            server.endpoint,
            _clips(),
            term_set,
            chunk=BenchChunk(CHUNK_MS),
            boosts=(None, 2.0, 4.0),
            concurrency=2,
        )
    assert report.reached_the_decoder is False
    assert report.usable is False
    assert "NO WEIGHT CHANGED ANY TRANSCRIPT" in report.render()


def test_a_report_whose_weights_moved_nothing_is_flagged_even_with_terms_present() -> None:
    bare = _reading(None, 5, 5, 0)
    boosted = _reading(2.0, 5, 5, 0, moved=0)  # identical counts, nothing moved
    report = _report(bare, boosted)
    assert report.reached_the_decoder is False
    assert report.usable is False


# --- how a residual miss was lost, which is an architecture question -----------------


def test_a_respelled_word_is_a_substitution() -> None:
    """Still within a weight's reach: the model chose to emit something there."""
    assert classify_misses("sir risdon graeme has", "sir risdongram has", ("graeme",)) == {
        MissKind.SUBSTITUTION.value: 1,
        MissKind.DELETION.value: 0,
        "unclassified": 0,
    }


def test_a_word_that_never_came_back_is_a_deletion() -> None:
    """Out of reach of any weight. Greedy boosting takes the blank-versus-emit decision
    from the unbiased argmax, so it can respell an emitted token and can never turn a
    blank into an emission; only a beam search recovers these."""
    assert classify_misses("the weevilly biscuit", "the biscuit", ("weevilly",)) == {
        MissKind.SUBSTITUTION.value: 0,
        MissKind.DELETION.value: 1,
        "unclassified": 0,
    }


def test_a_multi_word_term_is_left_unclassified_rather_than_guessed() -> None:
    """It can be part substituted and part deleted, and forcing it into one bucket would
    invent a fact. The totals still add up, so a partial count is never read as complete."""
    assert classify_misses("sir risdon graeme has", "sir risdongram has", ("risdon graeme",)) == {
        MissKind.SUBSTITUTION.value: 0,
        MissKind.DELETION.value: 0,
        "unclassified": 1,
    }


def test_a_term_that_came_back_is_classified_as_nothing() -> None:
    counts = classify_misses("the biscuit", "the biscuit", ("biscuit",))
    assert sum(counts.values()) == 0


def test_an_insertion_consumes_no_reference_word() -> None:
    """One op per reference word, whatever the hypothesis did around it."""
    ops = _alignment_ops(["a", "b"], ["a", "x", "b"])
    assert len(ops) == 2
    assert ops == ["match", "match"]


def test_ties_resolve_to_substitution_which_under_reports_deletions() -> None:
    """The conservative direction for the question being asked: it cannot manufacture
    the evidence that a weight has run out of room."""
    ops = _alignment_ops(["a"], ["z"])
    assert ops == [MissKind.SUBSTITUTION.value]


@pytest.mark.asyncio
async def test_the_record_carries_how_each_arm_lost_what_it_missed() -> None:
    term_set = TermSet(name="t", terms=("appellant", "hydrochlorothiazide"))
    async with _server() as endpoint:
        report = await run_rare_terms(
            endpoint, _clips(), term_set, chunk=BenchChunk(CHUNK_MS), boosts=(None, 2.0)
        )
    for arm in report.arms:
        assert set(arm.misses_by_kind) >= {
            MissKind.SUBSTITUTION.value,
            MissKind.DELETION.value,
        }
        # Every missed occurrence is accounted for under exactly one kind.
        assert sum(arm.misses_by_kind.values()) == arm.counts.misses
    assert "residual misses" in report.render()
    assert "beam-search question" in report.render()
    assert "misses_by_kind" in report.to_json_dict()["arms"][0]


def test_every_missed_occurrence_gets_exactly_one_kind() -> None:
    """The invariant is structural rather than guarded: `missed` is positive only when
    the hypothesis holds strictly fewer of the word than the reference does, so every
    unmatched reference position is available to explain one. A reconciliation branch
    stood here until a mutation proved nothing could reach it."""
    cases = [
        ("weevilly a weevilly b", "a b", ("weevilly",), 2),
        ("weevilly a weevilly b", "weevilly a b", ("weevilly",), 1),
        ("a weevilly b", "a woollen b", ("weevilly",), 1),
        ("a b", "a b", ("weevilly",), 0),
    ]
    for reference, transcript, terms, expected_missed in cases:
        counts = score_transcript(reference, transcript, terms)
        kinds = classify_misses(reference, transcript, terms)
        missed = sum(c.misses for c in counts.values())
        assert missed == expected_missed, (reference, transcript)
        assert sum(kinds.values()) == missed, (reference, transcript, kinds)


def test_the_kinds_always_account_for_every_miss_exhaustively() -> None:
    """The invariant behind removing the reconciliation branch, checked rather than argued.

    `classify_misses` drops a missed occurrence silently if the aligner and the occurrence
    counter ever disagree about how many there were. The branch that would have caught that
    was removed because a mutation proved it unreachable, and the argument for why was
    prose. This exhausts every (reference, hypothesis, term) over a three-letter alphabet
    up to length four, which is where a disagreement between a greedy non-overlapping count
    and an optimal edit alignment would show if it showed anywhere.
    """
    alphabet = "abc"
    checked = 0
    for reference_length in range(1, 5):
        for hypothesis_length in range(0, 5):
            for reference in itertools.product(alphabet, repeat=reference_length):
                for hypothesis in itertools.product(alphabet, repeat=hypothesis_length):
                    reference_text = " ".join(reference)
                    hypothesis_text = " ".join(hypothesis)
                    for term in set(reference):
                        counts = score_transcript(reference_text, hypothesis_text, (term,))
                        kinds = classify_misses(reference_text, hypothesis_text, (term,))
                        assert sum(kinds.values()) == counts[term].misses, (
                            reference_text,
                            hypothesis_text,
                            term,
                            kinds,
                        )
                        checked += 1
    assert checked > 5000, checked
