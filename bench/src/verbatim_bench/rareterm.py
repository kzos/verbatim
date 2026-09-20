# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""What a phrase list buys, and what it costs, measured separately from word error rate.

Corpus word error rate is the wrong instrument for context biasing and moves the wrong
way under it. A boosting tree exists to recover a handful of rare tokens per transcript;
recovering all of them moves a corpus WER by a fraction of a point, and an over-boosted
list that inserts its own words everywhere can move it by more, in the same direction as
an improvement. The two questions have to be asked separately:

* **recall on the terms** -- of the term occurrences the audio actually contains, how
  many came back;
* **false accepts** -- how many times a term came back from audio that did not contain
  it, which is the whole cost of the weight.

They trade off against each other, so the harness sweeps the weight and reports the curve
rather than a number. Word error rate rides along as a guard: a weight that improves term
recall while wrecking the transcript around it has not helped anybody.

**The term set is chosen blind to the transcripts.** Terms are the rare words of the
corpus's own references -- rare by document frequency, not by whether the model got them
wrong -- because selecting terms the bare pass already failed would guarantee an
improvement and measure nothing. The same list goes to every session, which is what a
deployment does: a customer sends their vocabulary, not the answers.

Nothing here is medical or legal audio, which this project does not have. LibriSpeech's
rare words are proper nouns -- ``raystoke``, ``risdon graeme``, ``archy`` -- and those
fail for the reason a drug name fails: a rare token with a flat acoustic posterior sitting
next to a common one. What this measures is that mechanism. A row from it says
"LibriSpeech test-other proper nouns", never "medical terms".
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from verbatim_bench import constants
from verbatim_bench.client import ChunkMode, SessionResult, run_session
from verbatim_bench.corpus import Utterance
from verbatim_bench.wer import WerCount, normalise, stream_wer_count

__all__ = [
    "DEFAULT_BOOSTS",
    "DEFAULT_MAX_DOCUMENT_FREQUENCY",
    "DEFAULT_MIN_TERM_CHARS",
    "ArmReading",
    "MissKind",
    "RareTermReport",
    "TermCount",
    "TermSet",
    "TermSetError",
    "classify_misses",
    "classify_misses_by_term",
    "derive_terms",
    "load_dictionary",
    "load_term_set",
    "occurrences",
    "run_rare_terms",
    "score_transcript",
]

#: A term must be at least this long. Shorter words are common enough that boosting them
#: measures the model's grip on function words, which is not what a phrase list is for.
DEFAULT_MIN_TERM_CHARS: Final = 7
#: A term must appear in at most this many of the corpus's utterances. Document frequency
#: is the rarity rule because it is blind to the transcripts: it can be computed before a
#: single session runs, so the term set cannot be selected on the outcome.
DEFAULT_MAX_DOCUMENT_FREQUENCY: Final = 2

#: Word lists tried in order when a caller asks for the out-of-dictionary rule and names
#: no list. Which one was used is recorded, because it is part of the selection rule and
#: a term set nobody can reconstruct is not a measurement anyone can repeat.
DEFAULT_DICTIONARIES: Final = (
    "/usr/share/dict/british-english",
    "/usr/share/dict/american-english",
    "/usr/share/dict/words",
)
#: The weights swept by default. ``None`` is the bare arm and must be first: every other
#: arm is read against it.
DEFAULT_BOOSTS: Final = (None, 1.0, 2.0, 4.0)


class TermSetError(ValueError):
    """The term set could not be read, or says something it cannot mean."""


def _words(text: str) -> tuple[str, ...]:
    """The harness's pinned normalisation, so a term and a word error rate agree on what
    a word is. Reused rather than re-implemented: two normalisers would drift."""
    return tuple(normalise(text))


@dataclass(frozen=True, slots=True)
class TermSet:
    """The terms a run measures recall on, and the phrase list it sends.

    ``terms`` are the surface forms; matching is done on their normalised words, so a
    term matches a transcript that spells it with different punctuation or casing. A term
    that normalises to nothing is refused rather than dropped: it would be counted in no
    denominator and would silently shrink the measurement.
    """

    name: str
    terms: tuple[str, ...]
    #: How these terms were chosen, recorded so the set can be rebuilt. A term set nobody
    #: can reconstruct is not a measurement anyone can repeat, and the rule decides what
    #: the run is even about: rarity alone selects ordinary vocabulary, while the
    #: out-of-dictionary rule selects the names the model may never have seen.
    rule: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise TermSetError("a term set needs a name; it goes on the row")
        if not self.terms:
            raise TermSetError(f"term set {self.name!r} has no terms")
        for term in self.terms:
            if not _words(term):
                raise TermSetError(f"term {term!r} in {self.name!r} normalises to no words at all")
        if len(set(self.terms)) != len(self.terms):
            raise TermSetError(f"term set {self.name!r} repeats a term")

    @property
    def digest(self) -> str:
        """The set's identity, in the order sent: the tree is built in that order."""
        hasher = hashlib.sha256()
        hasher.update(b"vb-term-set/1\n")
        for term in self.terms:
            hasher.update(f"{len(term)}:{term}\n".encode())
        return hasher.hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "rule": dict(self.rule),
            "terms": list(self.terms),
        }


def load_term_set(path: Path) -> TermSet:
    """Read ``{"name": ..., "terms": [...]}``."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TermSetError(f"cannot read term set {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise TermSetError(f"term set {path} must be a JSON object")
    terms = body.get("terms")
    if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
        raise TermSetError(f"'terms' in {path} must be a list of strings")
    return TermSet(name=str(body.get("name") or Path(path).stem), terms=tuple(terms))


def load_dictionary(path: str | Path | None = None) -> tuple[frozenset[str], str]:
    """An English word list and the path it came from.

    Used to tell an ordinary rare word from a name. Possessive forms are folded so
    ``sarah's`` does not make ``sarah`` look like a dictionary word.
    """
    candidates = [Path(path)] if path is not None else [Path(p) for p in DEFAULT_DICTIONARIES]
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        words = {
            line.strip().lower().removesuffix("'s") for line in text.splitlines() if line.strip()
        }
        if words:
            return frozenset(words), str(candidate)
    tried = ", ".join(str(c) for c in candidates)
    raise TermSetError(
        f"no usable word list: tried {tried}. The out-of-dictionary rule needs one, and "
        "guessing which words are names would make the term set unreproducible"
    )


def derive_terms(
    references: Sequence[str],
    *,
    min_chars: int = DEFAULT_MIN_TERM_CHARS,
    max_document_frequency: int = DEFAULT_MAX_DOCUMENT_FREQUENCY,
    limit: int = 256,
    dictionary: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """The corpus's own rare words, in first-appearance order, blind to any transcript.

    Rarity is document frequency: a word appearing in at most ``max_document_frequency``
    of the utterances. That rule can be evaluated before a single session runs, which is
    the point -- a term set chosen because the model got those words wrong would make the
    measurement circular, and the improvement would be guaranteed by construction.

    ``dictionary`` narrows it to words that are **not** in an English word list, which is
    the closest blind proxy available here for "a name the model has never seen". It
    matters more than it sounds. Rarity alone selects ordinary vocabulary: of 256 terms
    derived from LibriSpeech test-other on 2026-09-15, 245 were words like ``obliged``,
    ``frightened`` and ``morning``, which the model already knows. Measured over those,
    boosting moved recall 0.925 to 0.950. Measured over the 11 that were not in the word
    list -- ``bassorah``, ``comorin``, ``shahrazad``, ``orficer`` -- it moved 0.583 to
    0.833, which is a different feature. Aggregating the two hides the second inside the
    first.

    Fragmentation was tried as the proxy first and abandoned: this checkpoint's tokenizer
    has a 1024-piece vocabulary, so every word is spelled from fragments (``obliged`` is
    four pieces, ``baghdad`` five) and there is no out-of-vocabulary at the tokenizer
    level at all. That is worth knowing on its own -- an unseen name is a language-model
    problem here, not a vocabulary one, and a boosting tree over subword pieces can spell
    anything the model never saw.
    """
    if min_chars < 1:
        raise TermSetError(f"min_chars must be >= 1, got {min_chars!r}")
    if max_document_frequency < 1:
        raise TermSetError(f"max_document_frequency must be >= 1, got {max_document_frequency!r}")
    document_frequency: dict[str, int] = {}
    order: list[str] = []
    for reference in references:
        for word in set(_words(reference)):
            document_frequency[word] = document_frequency.get(word, 0) + 1
    for reference in references:
        for word in _words(reference):
            if len(word) < min_chars or word in order:
                continue
            if document_frequency[word] > max_document_frequency:
                continue
            if dictionary is not None and word in dictionary:
                continue
            order.append(word)
            if len(order) >= limit:
                return tuple(order)
    return tuple(order)


class MissKind(StrEnum):
    """How a reference word failed to come back.

    The distinction decides an architectural question rather than a tuning one. Greedy
    boosting takes the blank-versus-emit decision from the *unbiased* argmax, so it can
    respell a token the model already chose to emit and can never turn a blank into an
    emission. A residual miss that is a SUBSTITUTION is therefore something a weight can
    still reach; one that is a DELETION is not, at any weight, and only a beam search
    would recover it. A harness that reported "16 occurrences still missed" without
    saying which kind would leave that decision unanswerable.
    """

    SUBSTITUTION = "substitution"
    DELETION = "deletion"


def _alignment_ops(reference: Sequence[str], hypothesis: Sequence[str]) -> list[str]:
    """One edit op per reference word: "match", "substitution" or "deletion".

    A full matrix with a backtrace, unlike ``wer.edit_distance``, which keeps two rows
    because it only needs the count. Utterances are short, so the quadratic term is over
    one utterance at a time either way. Insertions consume a hypothesis word and no
    reference word, so they appear in no entry of the returned list, which is exactly one
    entry per reference word.

    Ties are resolved substitution, then deletion, then insertion. The order matters only
    for how a miss is labelled, and preferring substitution is the conservative direction
    for the question being asked: it under-reports deletions, so it cannot manufacture
    the evidence that a weight has run out of room.
    """
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
    for j in range(columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            cost[i][j] = min(
                cost[i - 1][j - 1] + (0 if same else 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    ops: list[str] = []
    i, j = rows, columns
    while i > 0:
        same = j > 0 and reference[i - 1] == hypothesis[j - 1]
        if j > 0 and cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
            ops.append("match" if same else MissKind.SUBSTITUTION.value)
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            ops.append(MissKind.DELETION.value)
            i -= 1
        else:
            j -= 1  # an insertion: it consumes no reference word
    ops.reverse()
    return ops


def classify_misses(reference: str, transcript: str, terms: Sequence[str]) -> dict[str, int]:
    """How the term occurrences this transcript missed were lost, pooled over the terms."""
    pooled: dict[str, int] = {
        MissKind.SUBSTITUTION.value: 0,
        MissKind.DELETION.value: 0,
        "unclassified": 0,
    }
    for kinds in classify_misses_by_term(reference, transcript, terms).values():
        for kind, count in kinds.items():
            pooled[kind] = pooled.get(kind, 0) + count
    return pooled


def classify_misses_by_term(
    reference: str, transcript: str, terms: Sequence[str]
) -> dict[str, dict[str, int]]:
    """How each term's missed occurrences were lost: by kind, per term.

    Per term rather than only pooled, because the question the split answers is asked of
    a subset. Whether a residual is reachable at any weight matters most for the names the
    model has never seen, and a pooled count over a term set that is 96 per cent ordinary
    vocabulary cannot be asked about them.

    Only single-word terms are classified. A multi-word term can be part substituted and
    part deleted, and forcing it into one bucket would invent a fact; those are counted
    under ``"unclassified"`` so the totals still add up and nobody reads a partial count
    as a complete one.
    """
    reference_words = _words(reference)
    hypothesis_words = _words(transcript)
    ops = _alignment_ops(reference_words, hypothesis_words)
    out: dict[str, dict[str, int]] = {}
    for term in terms:
        kinds = out.setdefault(
            term,
            {MissKind.SUBSTITUTION.value: 0, MissKind.DELETION.value: 0, "unclassified": 0},
        )
        needle = _words(term)
        expected = occurrences(reference_words, needle)
        produced = occurrences(hypothesis_words, needle)
        missed = expected - min(expected, produced)
        if missed <= 0:
            continue
        if len(needle) != 1:
            kinds["unclassified"] += missed
            continue
        # The positions this term sits at in the reference, in order; the first `missed`
        # of them that are not matches are the ones that did not come back.
        lost = [
            ops[index]
            for index, word in enumerate(reference_words)
            if word == needle[0] and ops[index] != "match"
        ]
        # `len(lost) >= missed` always, so there is no residue to bucket and no guard
        # here: `missed` is positive only when the hypothesis holds strictly fewer of the
        # word than the reference does, and every reference position that could not be
        # paired with one is a non-match, so `lost` has at least that many entries. A
        # reconciliation branch stood here until a mutation showed nothing could reach it
        # -- and an unreachable guard is the shape this project keeps finding.
        for kind in lost[:missed]:
            kinds[kind] = kinds.get(kind, 0) + 1
    return out


def occurrences(haystack_words: Sequence[str], needle_words: Sequence[str]) -> int:
    """Non-overlapping occurrences of one normalised word sequence inside another."""
    if not needle_words or len(needle_words) > len(haystack_words):
        return 0
    found = 0
    index = 0
    span = len(needle_words)
    while index <= len(haystack_words) - span:
        if tuple(haystack_words[index : index + span]) == tuple(needle_words):
            found += 1
            index += span
        else:
            index += 1
    return found


@dataclass(frozen=True, slots=True)
class TermCount:
    """One arm's tally over term-utterance pairs.

    ``hits`` and ``misses`` are occurrences the audio contained; ``false_accepts`` are
    occurrences it did not. A term the reference never used and the transcript never
    produced contributes to none of the three, so a large term set does not dilute the
    numbers with zeros.
    """

    hits: int = 0
    misses: int = 0
    false_accepts: int = 0

    def __add__(self, other: TermCount) -> TermCount:
        return TermCount(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_accepts + other.false_accepts,
        )

    @property
    def expected(self) -> int:
        return self.hits + self.misses

    @property
    def recall(self) -> float | None:
        """Of the occurrences the audio contained, the fraction that came back."""
        return None if self.expected == 0 else self.hits / self.expected

    @property
    def precision(self) -> float | None:
        """Of the occurrences produced, the fraction the audio actually contained."""
        produced = self.hits + self.false_accepts
        return None if produced == 0 else self.hits / produced

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "false_accepts": self.false_accepts,
            "expected": self.expected,
            "recall": self.recall,
            "precision": self.precision,
        }


def score_transcript(reference: str, transcript: str, terms: Sequence[str]) -> dict[str, TermCount]:
    """One utterance's per-term tally.

    A term found more often than the reference used it contributes the surplus as false
    accepts, which is how an over-boosted list that repeats a word shows up: the run on
    2026-09-14 that produced "french french french" for one clip would otherwise have
    scored a hit and said nothing about the other two.
    """
    reference_words = _words(reference)
    transcript_words = _words(transcript)
    out: dict[str, TermCount] = {}
    for term in terms:
        needle = _words(term)
        expected = occurrences(reference_words, needle)
        produced = occurrences(transcript_words, needle)
        hits = min(expected, produced)
        out[term] = TermCount(
            hits=hits,
            misses=expected - hits,
            false_accepts=max(0, produced - expected),
        )
    return out


@dataclass(slots=True)
class ArmReading:
    """One weight's pass over the corpus. ``boost`` is None for the bare arm."""

    boost: float | None
    counts: TermCount = field(default_factory=TermCount)
    wer: WerCount = field(default_factory=lambda: WerCount(0, 0))
    per_term: dict[str, TermCount] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    #: Streams whose transcript differs from the bare arm's. Empty on the bare arm, and
    #: empty on a boosted arm means the phrase list changed nothing at all -- which is
    #: what a server that ignored it looks like, and is indistinguishable from a weight
    #: too low to act unless it is named.
    changed_streams: tuple[str, ...] = ()
    #: How this arm's residual misses were lost, by kind. A substitution is still within
    #: a weight's reach; a deletion is not, at any weight, because greedy boosting takes
    #: the blank-versus-emit decision from the unbiased argmax. See ``MissKind``.
    misses_by_kind: dict[str, int] = field(default_factory=dict)
    #: The same split per term, so the question can be asked of a subset. Whether a
    #: residual is reachable at any weight matters most for the names the model has never
    #: seen, and a pooled count over a term set that is mostly ordinary vocabulary cannot
    #: be asked about them.
    miss_kinds_by_term: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "bare" if self.boost is None else f"boost {self.boost:g}"

    @property
    def terms_recovered(self) -> tuple[str, ...]:
        """Terms with at least one hit, in term order. Named so a reader can see what
        moved rather than only that something did."""
        return tuple(term for term, count in self.per_term.items() if count.hits)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "boost": self.boost,
            "terms": self.counts.to_json_dict(),
            "changed_streams": len(self.changed_streams),
            "misses_by_kind": dict(self.misses_by_kind),
            "wer": self.wer.wer,
            "wer_reference_words": self.wer.reference_words,
            "wer_errors": self.wer.errors,
            "errors": len(self.errors),
            "first_error": next(iter(self.errors.values()), None),
            "wall_clock_s": self.wall_clock_s,
            "per_term": {
                term: {
                    **count.to_json_dict(),
                    "miss_kinds": self.miss_kinds_by_term.get(term, {}),
                }
                for term, count in sorted(self.per_term.items())
                if count.expected or count.false_accepts
            },
        }


@dataclass(slots=True)
class RareTermReport:
    """Every arm, and the reading each supports."""

    endpoint: str
    chunk_ms: int
    corpus: dict[str, Any]
    term_set: TermSet
    arms: list[ArmReading]
    wall_clock_s: float = 0.0

    @property
    def bare(self) -> ArmReading | None:
        return next((arm for arm in self.arms if arm.boost is None), None)

    @property
    def boosted(self) -> list[ArmReading]:
        return [arm for arm in self.arms if arm.boost is not None]

    @property
    def reached_the_decoder(self) -> bool:
        """Whether any weight changed any transcript at all.

        The same guard the invariance gate's positive control applies, for the same
        reason. A server that ignored every phrase list returns exactly the bare arm's
        transcripts at every weight, so recall is identical everywhere and the run reads
        as "boosting did nothing" when the truth is "nothing was boosted". Those are
        different findings and only one of them is about the weight.
        """
        return any(arm.changed_streams for arm in self.boosted)

    @property
    def usable(self) -> bool:
        """Whether the run can say anything: a bare arm to read against, at least one
        boosted arm, no errored sessions, term occurrences to divide by, and a phrase
        list that demonstrably reached the decoder."""
        bare = self.bare
        if bare is None or not self.boosted:
            return False
        if any(arm.errors for arm in self.arms):
            return False
        if bare.counts.expected == 0:
            return False
        return self.reached_the_decoder

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record": "vb-rare-terms/1",
            "endpoint": self.endpoint,
            "chunk_ms": self.chunk_ms,
            "corpus": dict(self.corpus),
            "term_set": self.term_set.to_json_dict(),
            "arms": [arm.to_json_dict() for arm in self.arms],
            "usable": self.usable,
            "reached_the_decoder": self.reached_the_decoder,
            "wall_clock_s": self.wall_clock_s,
        }

    def render(self) -> str:
        lines = [
            f"rare-term recall: {self.endpoint}, chunk {self.chunk_ms} ms, corpus "
            f"{self.corpus.get('kind', '?')} ({self.corpus.get('utterances', '?')} utterances)",
            f"  term set {self.term_set.name} ({self.term_set.digest[:12]}), "
            f"{len(self.term_set.terms)} terms, sent whole to every session",
            "",
            f"  {'arm':<12} {'recall':>8} {'hits':>6} {'miss':>6} "
            f"{'false+':>7} {'precision':>10} {'WER':>8} {'moved':>7}",
        ]
        for arm in self.arms:
            counts = arm.counts
            recall = "-" if counts.recall is None else f"{counts.recall:.3f}"
            precision = "-" if counts.precision is None else f"{counts.precision:.3f}"
            wer = "-" if arm.wer.wer is None else f"{arm.wer.wer:.4f}"
            moved = "-" if arm.boost is None else str(len(arm.changed_streams))
            lines.append(
                f"  {arm.label:<12} {recall:>8} {counts.hits:>6} {counts.misses:>6} "
                f"{counts.false_accepts:>7} {precision:>10} {wer:>8} {moved:>7}"
            )
        bare = self.bare
        if bare is not None and bare.counts.recall is not None:
            for arm in self.arms:
                if arm.boost is None or arm.counts.recall is None:
                    continue
                gained = arm.counts.hits - bare.counts.hits
                cost = arm.counts.false_accepts - bare.counts.false_accepts
                lines.append(
                    f"  {arm.label}: {gained:+d} term occurrences recovered, "
                    f"{cost:+d} false accepts against bare"
                )
        for arm in self.arms:
            if not arm.misses_by_kind:
                continue
            deletions = arm.misses_by_kind.get(MissKind.DELETION.value, 0)
            substitutions = arm.misses_by_kind.get(MissKind.SUBSTITUTION.value, 0)
            unclassified = arm.misses_by_kind.get("unclassified", 0)
            lines.append(
                f"  {arm.label} residual misses: {deletions} deletion(s), "
                f"{substitutions} substitution(s), {unclassified} unclassified"
            )
        lines.append(
            "  a deletion is out of reach of any weight -- greedy boosting takes the "
            "blank-versus-emit decision from the unbiased argmax -- so a residual "
            "dominated by deletions is the beam-search question, not a tuning one"
        )
        for arm in self.arms:
            if arm.errors:
                lines.append(
                    f"*** {arm.label}: {len(arm.errors)} errored session(s), first: "
                    f"{next(iter(arm.errors.values()))}"
                )
        if self.boosted and not self.reached_the_decoder:
            lines.append(
                "*** NO WEIGHT CHANGED ANY TRANSCRIPT. A server that ignored every phrase "
                "list looks exactly like this, and so does a weight too low to act; the "
                "two are different findings and this run cannot tell them apart"
            )
        if not self.usable:
            lines.append(
                "FINAL: no reading. A run needs a bare arm, a boosted arm, no errored "
                "sessions, at least one term occurrence in the audio, and a phrase list "
                "that demonstrably reached the decoder"
            )
        return "\n".join(lines)


async def run_rare_terms(
    endpoint: str,
    clips: Sequence[Any],
    term_set: TermSet,
    *,
    chunk: ChunkMode,
    boosts: Sequence[float | None] = DEFAULT_BOOSTS,
    concurrency: int = 16,
    lang: str = "en-US",
    frame_ms: int = constants.FRAME_MS,
    seed: int = 20260914,
    corpus: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RareTermReport:
    """One pass over the corpus per weight, each session carrying the whole term set.

    The bare arm sends no phrases at all rather than sending the list at weight zero: a
    server without biasing and a server biasing at zero are different configurations, and
    the bare arm has to be the one every deployment without a phrase list actually runs.
    """
    import asyncio

    if not boosts:
        raise TermSetError("at least one arm is required")
    if concurrency < 1:
        raise TermSetError(f"concurrency must be >= 1, got {concurrency!r}")
    started = clock()
    arms: list[ArmReading] = []
    # The bare arm's transcripts, so every boosted arm can name the streams it moved.
    # Kept only for the run: the record carries the count and the per-term tallies, not
    # 256 transcripts.
    bare_transcripts: dict[str, str] = {}
    for arm_index, boost in enumerate(boosts):
        gate = asyncio.Semaphore(concurrency)

        async def one(
            index: int,
            clip: Any,
            boost: float | None = boost,
            gate: asyncio.Semaphore = gate,
            arm_index: int = arm_index,
        ) -> SessionResult:
            # Every loop variable this closure reads is bound here, not captured. The
            # arms run one after another so late binding would work today; it would stop
            # working the moment two arms overlapped, and silently.
            async with gate:
                utterance = Utterance(
                    stream_id=clip.stream_id,
                    audio_path=Path(clip.stream_id),
                    duration_s=clip.duration_s,
                    text=clip.text,
                )
                return await run_session(
                    endpoint,
                    session_id=f"rare-{arm_index}-{index:04d}",
                    utterance=utterance,
                    pcm=clip.pcm,
                    chunk=chunk,
                    start_delay_s=0.0,
                    words=False,
                    lang=lang,
                    frame_ms=frame_ms,
                    # The same framing schedule in every arm, so a difference between
                    # arms is the weight and not the arrival pattern.
                    frame_seed=seed * 100_000 + index,
                    phrases=() if boost is None else term_set.terms,
                    boost=boost,
                )

        arm_started = clock()
        results = await asyncio.gather(*(one(i, clip) for i, clip in enumerate(clips)))
        reading = ArmReading(boost=boost, wall_clock_s=clock() - arm_started)
        totals: dict[str, TermCount] = {term: TermCount() for term in term_set.terms}
        by_kind: dict[str, int] = {}
        by_term: dict[str, dict[str, int]] = {}
        transcripts: dict[str, str] = {}
        for clip, result in zip(clips, results, strict=True):
            if result.error is not None:
                reading.errors[result.stream_id] = result.error
                continue
            transcripts[result.stream_id] = result.final_text
            for term, count in score_transcript(
                clip.text, result.final_text, term_set.terms
            ).items():
                totals[term] = totals[term] + count
            for term, kinds in classify_misses_by_term(
                clip.text, result.final_text, term_set.terms
            ).items():
                bucket = by_term.setdefault(term, {})
                for kind, n in kinds.items():
                    bucket[kind] = bucket.get(kind, 0) + n
                    by_kind[kind] = by_kind.get(kind, 0) + n
            reading.wer = reading.wer + stream_wer_count(clip.text, result.final_text)
        reading.per_term = totals
        reading.misses_by_kind = by_kind
        reading.miss_kinds_by_term = by_term
        for count in totals.values():
            reading.counts = reading.counts + count
        if boost is None:
            bare_transcripts = transcripts
        else:
            reading.changed_streams = tuple(
                stream_id
                for stream_id, text in transcripts.items()
                if bare_transcripts.get(stream_id, text) != text
            )
        arms.append(reading)
    return RareTermReport(
        endpoint=endpoint,
        chunk_ms=chunk.ms,
        corpus=dict(corpus or {"kind": "clips", "utterances": len(clips)}),
        term_set=term_set,
        arms=arms,
        wall_clock_s=clock() - started,
    )
