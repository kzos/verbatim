# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The gate with per-session phrase lists, against an in-process server over the CPU fake.

The arrangement mirrors ``test_invariance.py``: the fake is invariant by construction, so
what is under test here is whether the biasing arm can fail. Two servers stand in for the
two cases that matter.

``honour_phrases=True`` mixes a stream's own phrase digest into its own tokens, which is
what a boosting tree does to a transcript. The gate should pass, the positive control
should show a list moving a transcript, and half the corpus should be recorded as biased.

The default fake ignores phrase lists entirely -- exactly what a server built without
per-stream biasing does, and what NeMo does when the decoder has no arena. Every level
then agrees, and the gate must report **no verdict** rather than "invariant": a passing
digest over a server that ignored the vocabulary is a statement about biasing being off.
That is the one result this file exists to pin.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from verbatim_bench.client import ChunkMode as BenchChunk
from verbatim_bench.invariance import (
    EXIT_INVARIANT,
    EXIT_NO_VERDICT,
    Clip,
    Level,
    run_gate,
)
from verbatim_bench.phrases import (
    PhraseBook,
    PhraseBookError,
    assign,
    load_phrase_book,
    missed_words,
    rare_words,
)
from verbatim_bench.serverfacts import ArmContradiction, ServerFacts, check_arm

from verbatim.config import ChunkMode, EngineConfig
from verbatim.engine import Engine
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.protocols.ws.server import WsServer, WsServerConfig

pytestmark = pytest.mark.cpu

CHUNK_MS = 80
BUCKET = 8
LEVELS = (Level("1", 1), Level("32a", 4), Level("32b", 4), Level("max", BUCKET))

BOOK = PhraseBook(
    name="test-book",
    lists=(
        ("medical", ("metformin", "hydrochlorothiazide", "E11.9")),
        ("legal", ("410 U.S. 113", "certiorari", "appellant")),
    ),
    boost=2.0,
    unrelated=("brontosaurus", "quokka"),
)


@asynccontextmanager
async def _server(*, honour_phrases: bool) -> AsyncIterator[str]:
    chunk = ChunkMode(CHUNK_MS)
    config = EngineConfig(chunk=chunk, buckets=(BUCKET,), idle_timeout_s=None)
    pipeline = FakePipelineAdapter(chunk, buckets=(BUCKET,), honour_phrases=honour_phrases)
    engine = Engine(config, pipeline)
    async with engine, WsServer(engine, WsServerConfig(port=0)) as server:
        yield server.endpoint


def _clips(n: int = BUCKET, chunks: int = 3) -> list[Clip]:
    """Clips with references, so the positive control has words to boost."""
    import numpy as np

    out: list[Clip] = []
    samples = chunks * CHUNK_MS * 16
    for index in range(n):
        rng = np.random.default_rng([31, index])
        pcm = (rng.uniform(-0.5, 0.5, size=samples) * 32767.0).astype("<i2").tobytes()
        out.append(
            Clip(
                stream_id=f"clip-{index:03d}",
                pcm=pcm,
                duration_s=samples / 16000,
                text="the appellant received hydrochlorothiazide after certiorari",
            )
        )
    return out


# --- the phrase book, pure ---------------------------------------------------------


def test_the_book_digest_follows_the_lists_the_weight_and_the_order() -> None:
    same = PhraseBook(name="b", lists=(("a", ("one", "two")),), boost=1.0)
    assert same.digest == PhraseBook(name="b", lists=(("a", ("one", "two")),), boost=1.0).digest
    # Order is an input: NeMo builds the tree in the order sent.
    assert same.digest != PhraseBook(name="b", lists=(("a", ("two", "one")),), boost=1.0).digest
    assert same.digest != PhraseBook(name="b", lists=(("a", ("one", "two")),), boost=2.0).digest


def test_an_empty_or_unnamed_list_is_refused() -> None:
    with pytest.raises(PhraseBookError, match="no lists"):
        PhraseBook(name="b", lists=())
    with pytest.raises(PhraseBookError, match="is empty"):
        PhraseBook(name="b", lists=(("a", ()),))
    with pytest.raises(PhraseBookError, match="empty phrase"):
        PhraseBook(name="b", lists=(("a", ("ok", "  ")),))


def test_a_book_is_read_from_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "book.json"
    path.write_text(
        json.dumps(
            {
                "name": "clinic",
                "boost": 2.5,
                "lists": {"drugs": ["metformin"]},
                "unrelated": ["quokka"],
            }
        )
    )
    book = load_phrase_book(path)
    assert book.name == "clinic"
    assert book.boost == 2.5
    assert book.lists == (("drugs", ("metformin",)),)
    assert book.unrelated == ("quokka",)


def test_a_malformed_book_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "book.json"
    path.write_text('{"name": "x"}')
    with pytest.raises(PhraseBookError, match="no 'lists' object"):
        load_phrase_book(path)


def test_half_the_corpus_carries_a_list_and_they_interleave() -> None:
    ids = [f"s{index:02d}" for index in range(12)]
    assigned = assign(ids, BOOK)
    assert len(assigned) == 6
    # Biased and unbiased rows alternate, so a batch holds both kinds side by side.
    carried = [stream_id in assigned for stream_id in ids]
    assert carried[:4] == [True, True, False, False]
    # And it is a pure function of position: the same corpus assigns the same lists.
    assert assign(ids, BOOK) == assigned


def test_rare_words_takes_the_longer_reference_words_in_order() -> None:
    assert rare_words("the patient took metformin, then hydrochlorothiazide.") == (
        "patient",
        "metformin",
        "hydrochlorothiazide",
    )
    # Punctuation is stripped, duplicates drop, and short words never qualify.
    assert rare_words("appellant, appellant the a") == ("appellant",)
    assert rare_words("a b c") == ()
    assert rare_words("alphabet beta gamma delta", limit=2) == ("alphabet",)


def test_the_positive_control_boosts_only_what_the_bare_pass_missed() -> None:
    """The control's whole design. Boosting words the model already produced changes
    nothing on a working server, so a control built that way measures its own choice of
    words. Measured on a B300, 2026-09-14: three clips, every boosted word already in the
    bare transcript, zero change; the words the pass had MISSED moved six of twelve."""
    reference = "that my father sir risdon graeme has smuggled goods here"
    bare = "that my father sir risdongram has smuggled goods here"
    assert missed_words(reference, bare) == ("risdon", "graeme")
    # A clip the model got right has nothing to prove and yields no list at all.
    assert missed_words(reference, reference) == ()
    # Case and punctuation are typesetting, not recognition.
    assert missed_words("Archy, silent.", "archy silent") == ()
    assert missed_words("a b", "", limit=1) == ("a",)


# --- the arm the server can fail ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_server_that_ignores_phrase_lists_gets_no_verdict_not_a_pass() -> None:
    """The one that matters. Every level agrees, because the server ignored every list;
    reporting that as "invariant" would be a claim about a feature that was never on."""
    async with _server(honour_phrases=False) as endpoint:
        report = await run_gate(
            endpoint, _clips(), LEVELS, chunk=BenchChunk(CHUNK_MS), book=BOOK, control_clips=2
        )
    assert report.equal is True  # the digests do agree...
    assert report.verdict == "uncontrolled"  # ...and that is exactly why it proves nothing
    assert report.exit_code == EXIT_NO_VERDICT
    assert report.controls.changed == 0
    assert report.controls.proved_biasing is False
    assert "positive control" in report.render()


@pytest.mark.asyncio
async def test_a_server_that_honours_phrase_lists_is_invariant_and_controlled() -> None:
    async with _server(honour_phrases=True) as endpoint:
        report = await run_gate(
            endpoint, _clips(), LEVELS, chunk=BenchChunk(CHUNK_MS), book=BOOK, control_clips=2
        )
    assert report.verdict == "invariant"
    assert report.exit_code == EXIT_INVARIANT
    assert report.controls.proved_biasing is True
    assert report.controls.changed >= 1
    # Half the corpus carried a list, at every level, and the record says so.
    assert [run.biased_streams for run in report.runs] == [BUCKET // 2] * 4
    assert report.biasing is not None
    assert report.biasing["name"] == "test-book"
    assert report.biasing["digest"] == BOOK.digest


@pytest.mark.asyncio
async def test_the_negative_control_reports_its_exposure_not_just_its_result() -> None:
    """Its power is entirely exposure, and a control that says "nothing inserted" without
    saying how hard it looked reads as more reassurance than it earned. At weight 2.0 one
    clip and six terms reported zero while a corpus-wide sweep at the same weight found
    105 false accepts."""
    async with _server(honour_phrases=True) as endpoint:
        report = await run_gate(
            endpoint, _clips(), LEVELS, chunk=BenchChunk(CHUNK_MS), book=BOOK, control_clips=3
        )
    controls = report.controls
    assert controls.negative_stream
    assert controls.negative_clips == 3  # every control clip, not only the last
    assert controls.inserted == ()
    document = report.to_json_dict()["controls"]
    assert document["negative_exposure"] == 3 * len(BOOK.unrelated)
    rendered = report.render()
    assert "3 clip(s) gained no word" in rendered
    assert "smoke test" in rendered


@pytest.mark.asyncio
async def test_a_run_without_a_book_carries_no_biasing_block_at_all() -> None:
    async with _server(honour_phrases=True) as endpoint:
        report = await run_gate(endpoint, _clips(), LEVELS, chunk=BenchChunk(CHUNK_MS))
    assert report.verdict == "invariant"
    assert report.biasing is None
    assert report.to_json_dict()["controls"] is None
    assert all(run.biased_streams == 0 for run in report.runs)


# --- the arm label -----------------------------------------------------------------


def _facts(biasing: bool | None) -> ServerFacts:
    return ServerFacts(
        ready=True,
        model="m",
        pipeline="cache_aware_rnnt",
        chunk_ms=160,
        precision="bfloat16",
        execution="eager",
        tick_id=10,
        biasing=biasing,
    )


def test_sending_phrase_lists_to_a_server_without_biasing_is_a_contradiction() -> None:
    with pytest.raises(ArmContradiction, match="different arms"):
        check_arm(_facts(False), arm="invariance", declared_biasing=True)


def test_withholding_phrase_lists_from_a_biasing_server_is_also_a_contradiction() -> None:
    """Biasing changes the decode for every row, not only the biased ones, so a run with
    no phrase lists against a biasing server is not the no-biasing arm either."""
    with pytest.raises(ArmContradiction, match="different arms"):
        check_arm(_facts(True), arm="invariance", declared_biasing=False)


def test_a_server_too_old_to_report_biasing_gets_no_opinion() -> None:
    check_arm(_facts(None), arm="invariance", declared_biasing=True)


def test_matching_arms_pass() -> None:
    check_arm(_facts(True), arm="invariance", declared_biasing=True)
    check_arm(_facts(False), arm="invariance", declared_biasing=False)
