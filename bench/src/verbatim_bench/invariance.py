# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The batch-invariance gate: one corpus at concurrency 1 / 32a / 32b / max, diffed.

The server is named for a property; this is the instrument that tests it for it. Each
level replays every utterance of one corpus through a running server with at most
``concurrency`` streams in flight, and keeps every stream's final text and word timings
exactly as the wire carried them. Each level's finals are canonicalised and digested
(``canonical.py``); the verdict is ``equal`` only if all four digests are the same.
Nothing is compared against a reference transcript: same audio, different batch
composition, and the only question is whether the output changed.

Two levels at 32 separate the two things a difference can mean. A level against 1 is
batch dependence. 32b against 32a is run-to-run nondeterminism at the same concurrency,
which is a different defect and is reported under its own name.

The report says which streams diverged and where: the first differing word or timing of
each, on both sides, in the style of the probes whose findings were usable for exactly
that reason.

Guards. A gate over a server that transcribes nothing passes for free, so a run in which
every final is empty is refused as vacuous rather than reported as invariant. A level in
which any session errored carries no digest, and a run with such a level has no verdict.
A corpus smaller than the highest level cannot reach it and is refused before the first
byte is sent. The fake pipeline is invariant by construction, so the gate's own tests
run it with a deliberate leak and pin that the gate goes red.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from verbatim_bench import constants
from verbatim_bench.canonical import FinalRecord, finals_digest
from verbatim_bench.client import ChunkMode, SessionResult, run_session
from verbatim_bench.corpus import Utterance, load_manifest, read_pcm16, validate_pcm_sha256
from verbatim_bench.results import percentile

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_SEED",
    "DEFAULT_SYNTHETIC_S",
    "EXIT_DIVERGENT",
    "EXIT_INVARIANT",
    "EXIT_NO_VERDICT",
    "PAIRS",
    "SLOTS",
    "Clip",
    "Divergence",
    "GateRefusal",
    "GateReport",
    "Level",
    "LevelRun",
    "assess",
    "check_levels",
    "default_levels",
    "diff_records",
    "manifest_corpus",
    "run_gate",
    "run_level",
    "synthetic_corpus",
]

SAMPLE_RATE_HZ: Final = 16000

#: The four row fields, in order: `hash_1`, `hash_32a`, `hash_32b`, `hash_max`.
SLOTS: Final = ("1", "32a", "32b", "max")

#: Which levels are diffed against which. Against "1" is batch dependence; 32b against
#: 32a is run-to-run nondeterminism at one concurrency.
PAIRS: Final = (("32a", "1"), ("32b", "1"), ("max", "1"), ("32b", "32a"))

#: The maximum level until a capacity search names the server's ceiling: the largest
#: ceiling batch size the ladder can name. Pass the server's own `--ceiling` once measured.
DEFAULT_MAX_CONCURRENCY: Final = max(constants.CEILING_BATCH_SIZES)

DEFAULT_SEED: Final = 20260914
DEFAULT_SYNTHETIC_S: Final = 0.4

EXIT_INVARIANT: Final = 0
EXIT_DIVERGENT: Final = 1
EXIT_NO_VERDICT: Final = 2

_CLASSIFICATION: Final = {"1": "batch dependence", "32a": "run-to-run at one concurrency"}


class ChurnGate:
    """Admission on a triangle wave, so occupancy swings instead of sitting pinned.

    The constant levels use ``asyncio.Semaphore(N)``: a clip releases its permit and the
    next clip takes it immediately, so occupancy holds at ``N`` for the whole run and only
    drains at the tail. That tests whether a transcript depends on how busy the server was
    *for the run*. It does not test whether a transcript depends on how busy the server was
    *during that session*, because no session ever sees its neighbour count move far.

    Here the ceiling walks 1 -> N -> 1 over ``period_s`` and repeats, so an utterance long
    enough to span a period lives through the full swing. Under fixed-shape padding this
    should be inert: the steady batch is the bucket at every occupancy, so the encoder sees
    the same shape throughout. "Should be" is the phrase this project keeps being caught by,
    which is the reason to run it.

    The wave is a pure function of elapsed time, so a re-run with the same corpus and the
    same period admits on the same schedule.
    """

    def __init__(
        self,
        concurrency: int,
        period_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if concurrency < 1:
            raise GateRefusal(f"churn concurrency must be >= 1, got {concurrency!r}")
        if period_s <= 0:
            raise GateRefusal(f"churn period must be positive, got {period_s!r}")
        self._concurrency = concurrency
        self._period_s = period_s
        self._clock = clock
        self._started = clock()
        self._in_flight = 0
        self._condition = asyncio.Condition()
        self._observed: list[int] = []
        self._by_stream: dict[str, int] = {}

    @property
    def observed_occupancy(self) -> tuple[int, ...]:
        """The in-flight count at each admission, so a run can show the wave happened
        rather than assert that it was configured."""
        return tuple(self._observed)

    @property
    def occupancy_by_stream(self) -> dict[str, int]:
        """Occupancy when each stream was admitted, so a diverging stream can be asked
        whether it started at the trough of the wave or its crest. Without this the churn
        arm can report that it churned and not that any particular transcript met the
        conditions it is being blamed on."""
        return dict(self._by_stream)

    def admit_as(self, stream_id: str) -> _ChurnAdmission:
        """Admission that remembers which stream it let in."""
        return _ChurnAdmission(self, stream_id)

    def ceiling_at(self, elapsed_s: float) -> int:
        """The triangle wave: 1 at the period's edges, ``concurrency`` at its middle."""
        if self._concurrency == 1:
            return 1
        phase = (elapsed_s % self._period_s) / self._period_s
        rising = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)
        return max(1, min(self._concurrency, 1 + round(rising * (self._concurrency - 1))))

    async def __aenter__(self) -> ChurnGate:
        async with self._condition:
            while self._in_flight >= self.ceiling_at(self._clock() - self._started):
                # Wake on the next release, and independently on a slice of the period so
                # a rising ceiling admits even when nothing has finished.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._condition.wait(), timeout=self._period_s / 20.0)
            self._in_flight += 1
            self._observed.append(self._in_flight)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        async with self._condition:
            self._in_flight -= 1
            self._condition.notify()


class _ChurnAdmission:
    """One stream's turn through a :class:`ChurnGate`, recording the occupancy it met."""

    def __init__(self, gate: ChurnGate, stream_id: str) -> None:
        self._gate = gate
        self._stream_id = stream_id

    async def __aenter__(self) -> _ChurnAdmission:
        await self._gate.__aenter__()
        self._gate._by_stream[self._stream_id] = self._gate._observed[-1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._gate.__aexit__(*exc)


class GateRefusal(ValueError):
    """The gate declined to run: it could not have produced a meaningful verdict."""


@dataclass(frozen=True, slots=True)
class Level:
    """One run of the corpus. `slot` names the row field the digest fills (one of
    `SLOTS`); `concurrency` is how many streams were in flight."""

    slot: str
    concurrency: int
    #: When set, this level admits on a triangle wave between 1 and ``concurrency`` with
    #: this period in seconds, instead of holding a constant number in flight. A session
    #: then lives through a large swing in how many neighbours it has, which the constant
    #: levels never produce: their semaphore refills the instant a clip finishes, so
    #: occupancy sits pinned at ``concurrency`` until the tail drains.
    churn_period_s: float | None = None

    def __post_init__(self) -> None:
        if self.slot not in SLOTS:
            raise GateRefusal(f"invalid level slot {self.slot!r}: must be one of {SLOTS}")
        if self.concurrency < 1:
            raise GateRefusal(f"invalid concurrency {self.concurrency!r} for level {self.slot}")
        if self.churn_period_s is not None and self.churn_period_s <= 0:
            raise GateRefusal(f"invalid churn period {self.churn_period_s!r} for level {self.slot}")


@dataclass(frozen=True, slots=True)
class Clip:
    """One utterance of the corpus: 16 kHz mono PCM16 little-endian bytes. `text` is the
    reference transcript when the corpus has one; the gate never compares against it."""

    stream_id: str
    pcm: bytes
    duration_s: float
    text: str = ""


@dataclass(slots=True)
class LevelRun:
    """What one level produced: finals by stream id (errored sessions excluded), the
    errors by stream id, and the level's digest, which is None when anything errored."""

    level: Level
    finals: dict[str, FinalRecord]
    errors: dict[str, str] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    pacing_slip_p99_ms: float | None = None
    #: Occupancy each stream met when it was admitted. Empty on a constant level, where
    #: it would be the level's own concurrency for every stream and say nothing.
    occupancy_by_stream: dict[str, int] = field(default_factory=dict)

    @property
    def digest(self) -> str | None:
        if self.errors:
            return None
        return finals_digest(self.finals.values())


@dataclass(frozen=True, slots=True)
class Divergence:
    """One stream whose output at `level` differs from its output at `against`. `kind`
    is "text", "words" or "missing"; `index` is the first differing word, None when the
    texts differ only in whitespace; `left` is what `against` had there, `right` what
    `level` had."""

    stream_id: str
    level: str
    against: str
    kind: str
    index: int | None
    left: str
    right: str

    @property
    def classification(self) -> str:
        return _CLASSIFICATION.get(self.against, "batch dependence")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "level": self.level,
            "against": self.against,
            "classification": self.classification,
            "kind": self.kind,
            "index": self.index,
            "left": self.left,
            "right": self.right,
        }


def default_levels(
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY, *, churn_period_s: float | None = None
) -> tuple[Level, ...]:
    """1 / 32a / 32b / max, with `max` the caller's.

    ``churn_period_s`` churns the ``max`` level rather than adding a fifth: the row fields
    are ``hash_1``/``hash_32a``/``hash_32b``/``hash_max`` and the frozen methodology
    forbids adding one after a run. It lands in the right place anyway -- ``max`` against
    ``1`` is already classified as batch dependence, and "does a transcript survive the
    neighbour count moving under it" is the same question asked of a harder arrival
    pattern. The record names the period, so a churned run is never mistaken for a
    constant one.
    """
    return (
        Level("1", 1),
        Level("32a", 32),
        Level("32b", 32),
        Level("max", max_concurrency, churn_period_s=churn_period_s),
    )


def check_levels(levels: Sequence[Level], corpus_size: int) -> None:
    """Refuse a run whose levels are not the four slots in order or whose corpus cannot
    reach its highest level: a level of 32 over a corpus of 8 is a level of 8 with a
    misleading name."""
    if tuple(level.slot for level in levels) != SLOTS:
        raise GateRefusal(
            f"levels must fill the slots {SLOTS} in order, got {[x.slot for x in levels]}"
        )
    highest = max(level.concurrency for level in levels)
    if corpus_size < highest:
        raise GateRefusal(
            f"a corpus of {corpus_size} utterances cannot reach concurrency {highest}: "
            "pass a larger corpus or a smaller --max"
        )


def synthetic_corpus(
    n: int, *, duration_s: float = DEFAULT_SYNTHETIC_S, seed: int = DEFAULT_SEED
) -> list[Clip]:
    """`n` clips of seeded uniform noise at 16 kHz, half full scale, reproducible from
    the seed. Enough for the fake, which hashes bytes, and for the question the gate
    asks of a real model, which is only whether the same bytes produce the same output;
    for a transcript anyone would read, pass a manifest."""
    if n < 1:
        raise GateRefusal(f"a synthetic corpus needs at least one clip, got {n}")
    if duration_s <= 0:
        raise GateRefusal(f"invalid clip duration {duration_s!r} s")
    samples = round(duration_s * SAMPLE_RATE_HZ)
    clips: list[Clip] = []
    for index in range(n):
        rng = np.random.default_rng([seed, index])
        pcm = (rng.uniform(-0.5, 0.5, size=samples) * 32767.0).astype("<i2").tobytes()
        clips.append(Clip(f"synthetic-{index:04d}", pcm, samples / SAMPLE_RATE_HZ))
    return clips


def manifest_corpus(path: Path) -> list[Clip]:
    """Every utterance of a NeMo-compatible manifest, decoded and checked against its
    recorded sha256 where the manifest carries one. Order preserved."""
    clips: list[Clip] = []
    for utterance in load_manifest(path):
        pcm = read_pcm16(utterance.audio_path)
        validate_pcm_sha256(pcm, utterance.sha256, path=utterance.audio_path)
        clips.append(Clip(utterance.stream_id, pcm, utterance.duration_s, utterance.text))
    return clips


def _word_at(words: Sequence[tuple[str, int, int]], index: int) -> str:
    return repr(words[index]) if index < len(words) else "<end>"


def diff_records(left: FinalRecord, right: FinalRecord) -> tuple[str, int | None, str, str] | None:
    """The first difference between two finals of one stream, text before timings:
    `(kind, index, left_at, right_at)`, or None when they are identical byte for byte."""
    if left.text != right.text:
        left_words, right_words = left.text.split(), right.text.split()
        if left_words == right_words:
            return ("text", None, repr(left.text), repr(right.text))
        index = next(
            (i for i, (a, b) in enumerate(zip(left_words, right_words, strict=False)) if a != b),
            min(len(left_words), len(right_words)),
        )
        left_at = left_words[index] if index < len(left_words) else "<end>"
        right_at = right_words[index] if index < len(right_words) else "<end>"
        return ("text", index, left_at, right_at)
    if left.words != right.words:
        index = next(
            (i for i, (a, b) in enumerate(zip(left.words, right.words, strict=False)) if a != b),
            min(len(left.words), len(right.words)),
        )
        return ("words", index, _word_at(left.words, index), _word_at(right.words, index))
    return None


def _divergences(runs: Mapping[str, LevelRun]) -> list[Divergence]:
    out: list[Divergence] = []
    for level, against in PAIRS:
        if level not in runs or against not in runs:
            continue
        this, base = runs[level], runs[against]
        if this.errors or base.errors:
            continue
        for stream_id in sorted(set(base.finals) | set(this.finals), key=lambda s: s.encode()):
            left, right = base.finals.get(stream_id), this.finals.get(stream_id)
            if left is None or right is None:
                out.append(
                    Divergence(
                        stream_id,
                        level,
                        against,
                        "missing",
                        None,
                        "<present>" if left is not None else "<absent>",
                        "<present>" if right is not None else "<absent>",
                    )
                )
                continue
            found = diff_records(left, right)
            if found is not None:
                kind, index, left_at, right_at = found
                out.append(Divergence(stream_id, level, against, kind, index, left_at, right_at))
    return out


@dataclass(slots=True)
class GateReport:
    """The verdict and everything it rests on. `digests` maps each slot to its level's
    digest or None; `equal` is True only if all four are present and the same."""

    endpoint: str
    chunk_ms: int
    corpus: dict[str, Any]
    runs: list[LevelRun]
    divergences: list[Divergence]
    verdict: str  # "invariant" | "divergent" | "incomplete" | "vacuous"
    timings_present: bool
    wall_clock_s: float

    @property
    def digests(self) -> dict[str, str | None]:
        return {run.level.slot: run.digest for run in self.runs}

    @property
    def equal(self) -> bool | None:
        values = [self.digests.get(slot) for slot in SLOTS]
        if any(value is None for value in values):
            return None
        return len(set(values)) == 1

    @property
    def exit_code(self) -> int:
        if self.verdict == "invariant":
            return EXIT_INVARIANT
        if self.verdict == "divergent":
            return EXIT_DIVERGENT
        return EXIT_NO_VERDICT

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "record": "vb-invariance/1",
            "endpoint": self.endpoint,
            "chunk_ms": self.chunk_ms,
            "corpus": dict(self.corpus),
            "levels": [
                {
                    "slot": run.level.slot,
                    "concurrency": run.level.concurrency,
                    # Null on a constant level. Named here so a churned digest is never
                    # read as a constant-occupancy one: they answer different questions.
                    "churn_period_s": run.level.churn_period_s,
                    # The wave, shown rather than asserted: how many streams met each
                    # occupancy when they were admitted, and the occupancy each stream
                    # met. A churn arm that reports no swing here did not churn.
                    "admission_occupancy_histogram": (
                        {
                            str(value): sum(
                                1 for v in run.occupancy_by_stream.values() if v == value
                            )
                            for value in sorted(set(run.occupancy_by_stream.values()))
                        }
                        or None
                    ),
                    "admission_occupancy_by_stream": run.occupancy_by_stream or None,
                    "streams": len(run.finals),
                    "errors": len(run.errors),
                    "first_error": next(iter(run.errors.values()), None),
                    "digest": run.digest,
                    "wall_clock_s": run.wall_clock_s,
                    "pacing_slip_p99_ms": run.pacing_slip_p99_ms,
                }
                for run in self.runs
            ],
            # The row's block, verbatim: paste into `result.invariance`.
            "invariance": {
                **{f"hash_{slot}": self.digests.get(slot) for slot in SLOTS},
                "equal": self.equal,
            },
            "timings_present": self.timings_present,
            "divergences": [d.to_json_dict() for d in self.divergences],
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "wall_clock_s": self.wall_clock_s,
        }

    def render(self) -> str:
        """The probes' style: every divergence named, then one FINAL line."""
        corpus = self.corpus
        lines = [
            f"batch-invariance gate: {self.endpoint}, chunk {self.chunk_ms} ms, corpus "
            f"{corpus.get('kind', '?')} ({corpus.get('utterances', '?')} utterances)"
        ]
        for run in self.runs:
            digest = run.digest[:12] if run.digest else "-"
            slip = f"{run.pacing_slip_p99_ms:.1f}" if run.pacing_slip_p99_ms is not None else "-"
            lines.append(
                f"  level {run.level.slot:<4} concurrency {run.level.concurrency:>4}: "
                f"{len(run.finals)} streams, {len(run.errors)} errors, {run.wall_clock_s:.1f} s, "
                f"slip p99 {slip} ms, digest {digest}"
            )
        for number, d in enumerate(self.divergences, start=1):
            where = f"index {d.index}" if d.index is not None else "whitespace only"
            lines.append(
                f"*** DIVERGENCE #{number} stream {d.stream_id}: {d.level} vs {d.against} "
                f"({d.classification}), {d.kind} differ at {where}"
            )
            lines.append(f"      {d.against:<4}: {d.left}")
            lines.append(f"      {d.level:<4}: {d.right}")
        streams = max((len(run.finals) for run in self.runs), default=0)
        if self.verdict == "invariant":
            channels = (
                "text and word timings"
                if self.timings_present
                else "text only; no word timings were emitted"
            )
            lines.append(
                f"FINAL: invariant at 1 / 32a / 32b / max over {streams} streams "
                f"({channels} compared)"
            )
        elif self.verdict == "divergent":
            # Per comparison, not unioned. The union over every level compared against 1
            # is dominated by whichever level diverges most, so a change in one arm can
            # move the union the other way -- which misread once, on 2026-09-14, as a
            # prediction failing when the arm under test had moved as predicted.
            rerun = {d.stream_id for d in self.divergences if d.against == "32a"}

            def against_one(slot: str) -> int:
                return len(
                    {d.stream_id for d in self.divergences if d.against == "1" and d.level == slot}
                )

            per_level = ", ".join(f"{slot} {against_one(slot)}" for slot in SLOTS if slot != "1")
            batch = {d.stream_id for d in self.divergences if d.against == "1"}
            lines.append(
                f"FINAL: divergent. Streams differing from concurrency 1, by level: "
                f"{per_level} (of {streams}); {len(batch)} distinct streams in total. "
                f"{len(rerun)} differ between 32a and 32b"
            )
        elif self.verdict == "incomplete":
            failed = next(run for run in self.runs if run.errors)
            first = next(iter(failed.errors.values()))
            lines.append(
                f"FINAL: no verdict: level {failed.level.slot} had {len(failed.errors)} errored "
                f"sessions (first: {first})"
            )
        else:
            lines.append(
                "FINAL: no verdict: every final was empty; a server that transcribes nothing "
                "passes for free, so this run proves nothing"
            )
        return "\n".join(lines)


def assess(
    runs: Sequence[LevelRun],
    *,
    endpoint: str = "",
    chunk_ms: int = 0,
    corpus: Mapping[str, Any] | None = None,
    wall_clock_s: float = 0.0,
) -> GateReport:
    """The verdict over completed level runs. Pure: no server, no clock."""
    by_slot = {run.level.slot: run for run in runs}
    records = [record for run in runs for record in run.finals.values()]
    timings_present = any(record.words for record in records)
    if any(run.errors for run in runs):
        verdict = "incomplete"
    elif all(record.text == "" for record in records):
        verdict = "vacuous"
    else:
        digests = [by_slot[slot].digest for slot in SLOTS if slot in by_slot]
        verdict = (
            "invariant" if len(digests) == len(SLOTS) and len(set(digests)) == 1 else "divergent"
        )
    return GateReport(
        endpoint=endpoint,
        chunk_ms=chunk_ms,
        corpus=dict(corpus or {}),
        runs=list(runs),
        divergences=_divergences(by_slot) if verdict in ("invariant", "divergent") else [],
        verdict=verdict,
        timings_present=timings_present,
        wall_clock_s=wall_clock_s,
    )


async def run_level(
    endpoint: str,
    clips: Sequence[Clip],
    level: Level,
    *,
    chunk: ChunkMode,
    lang: str = "en-US",
    frame_ms: int = constants.FRAME_MS,
    seed: int = DEFAULT_SEED,
    clock: Callable[[], float] = time.monotonic,
) -> LevelRun:
    """Every clip once, in corpus order, at most `level.concurrency` in flight, each on
    its own connection at real-time pace with canonical framing. The frame jitter is
    seeded per level and clip, so 32a and 32b arrive at different phases on purpose."""
    gate: asyncio.Semaphore | ChurnGate = (
        ChurnGate(level.concurrency, level.churn_period_s, clock=clock)
        if level.churn_period_s is not None
        else asyncio.Semaphore(level.concurrency)
    )
    level_index = SLOTS.index(level.slot)

    async def one(index: int, clip: Clip) -> SessionResult:
        admission = gate.admit_as(clip.stream_id) if isinstance(gate, ChurnGate) else gate
        async with admission:
            utterance = Utterance(
                stream_id=clip.stream_id,
                audio_path=Path(clip.stream_id),
                duration_s=clip.duration_s,
                text=clip.text,
            )
            return await run_session(
                endpoint,
                session_id=f"{level.slot}-{index:04d}",
                utterance=utterance,
                pcm=clip.pcm,
                chunk=chunk,
                start_delay_s=0.0,
                words=True,
                lang=lang,
                frame_ms=frame_ms,
                frame_seed=(seed * 4 + level_index) * 100_000 + index,
            )

    started = clock()
    results = await asyncio.gather(*(one(index, clip) for index, clip in enumerate(clips)))
    wall_clock_s = clock() - started
    finals: dict[str, FinalRecord] = {}
    errors: dict[str, str] = {}
    slips: list[float] = []
    for result in results:
        slips.extend(result.pacing_slip_ms)
        if result.error is not None:
            errors[result.stream_id] = result.error
            continue
        finals[result.stream_id] = FinalRecord(
            stream_id=result.stream_id, text=result.final_text, words=tuple(result.words)
        )
    return LevelRun(
        level=level,
        finals=finals,
        errors=errors,
        wall_clock_s=wall_clock_s,
        pacing_slip_p99_ms=percentile(slips, 99) if slips else None,
        occupancy_by_stream=(gate.occupancy_by_stream if isinstance(gate, ChurnGate) else {}),
    )


async def run_gate(
    endpoint: str,
    clips: Sequence[Clip],
    levels: Sequence[Level] | None = None,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    churn_period_s: float | None = None,
    chunk: ChunkMode | None = None,
    lang: str = "en-US",
    frame_ms: int = constants.FRAME_MS,
    seed: int = DEFAULT_SEED,
    corpus: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> GateReport:
    """The four levels in order, then the verdict. Refuses before sending a byte if the
    corpus cannot reach the highest level."""
    chosen = (
        tuple(levels)
        if levels is not None
        else default_levels(max_concurrency, churn_period_s=churn_period_s)
    )
    check_levels(chosen, len(clips))
    selected_chunk = chunk if chunk is not None else ChunkMode(160)
    started = clock()
    runs = [
        await run_level(
            endpoint,
            clips,
            level,
            chunk=selected_chunk,
            lang=lang,
            frame_ms=frame_ms,
            seed=seed,
            clock=clock,
        )
        for level in chosen
    ]
    return assess(
        runs,
        endpoint=endpoint,
        chunk_ms=selected_chunk.ms,
        corpus=corpus or {"kind": "clips", "utterances": len(clips)},
        wall_clock_s=clock() - started,
    )
