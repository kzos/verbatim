# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``verbatim-bench run|row|validate|invariance|wer|env|verify|replay|record``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from verbatim_bench import constants
from verbatim_bench import invariance as invariance_gate
from verbatim_bench import rareterm as rare_terms
from verbatim_bench.client import ChunkMode
from verbatim_bench.corpus import manifest_corpus_id
from verbatim_bench.pace import DEFAULT_RAMP_S, LoadSpec, run_load
from verbatim_bench.phrases import PhraseBookError, load_phrase_book
from verbatim_bench.results import write_results
from verbatim_bench.serverfacts import ArmContradiction, check_arm, read_server_facts
from verbatim_bench.wer import Batch1Reference, ReferenceError, load_batch1_reference

if TYPE_CHECKING:
    from verbatim_bench.ladder import RungPlan
    from verbatim_bench.results import RunResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verbatim-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="paced real-time replay against a WebSocket server")
    run.add_argument("--endpoint", required=True)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--sessions", type=int, default=1)
    run.add_argument("--chunk", default="160ms")
    run.add_argument("--profile", choices=("uniform", "bursty"), default="uniform")
    run.add_argument("--seed", type=int, default=20260914)
    run.add_argument("--ramp-s", type=float, default=DEFAULT_RAMP_S)
    run.add_argument("--words", action="store_true")
    run.add_argument("--lang", default="en-US")
    run.add_argument("--arm", default="unknown")
    run.add_argument("--x-ms", type=int, default=constants.X_MS)
    run.add_argument("--out", type=Path, default=Path("results"))
    verify = sub.add_parser("verify", help="recompute a results file from its raw samples")
    verify.add_argument("path", type=Path, help="results.json file or a directory holding it")
    verify.add_argument("--json", action="store_true", help="emit the report as JSON")
    verify.add_argument("--strict", action="store_true", help="promote warnings to errors")
    env = sub.add_parser("env", help="collect the machine-read environment record")
    env.add_argument("--gpu-index", type=int, default=0)
    env.add_argument("--from-smi-xml", type=Path, default=None)
    env.add_argument("--json", action="store_true", help="print the record as JSON")
    env.add_argument("--out", type=Path, default=None)
    ladder = sub.add_parser("ladder", help="search the sustained stream count N")
    ladder.add_argument("--endpoint", required=True)
    ladder.add_argument("--manifest", required=True, type=Path)
    ladder.add_argument("--arm", required=True)
    ladder.add_argument("--n0", type=int, default=None)
    ladder.add_argument("--ceiling", type=float, default=None)
    ladder.add_argument("--seeds", type=str, default=",".join(str(s) for s in constants.SEEDS))
    ladder.add_argument("--warm-up-s", type=float, default=constants.WARM_UP_S)
    ladder.add_argument("--window-s", type=float, default=constants.WINDOW_S)
    ladder.add_argument("--warm-up-reading-s", type=float, default=constants.WARM_UP_READING_S)
    ladder.add_argument("--warm-up-convergence", type=float, default=constants.WARM_UP_CONVERGENCE)
    ladder.add_argument("--warm-up-cap-s", type=float, default=constants.WARM_UP_CAP_S)
    ladder.add_argument("--ramp-s", type=float, default=DEFAULT_RAMP_S)
    ladder.add_argument(
        "--frame-ms",
        type=int,
        default=constants.FRAME_MS,
        help=(
            "transport framing, frozen at 20 ms. The ladder could not express this at "
            "all and silently took the default; anything else is non-canonical framing"
        ),
    )
    ladder.add_argument(
        "--wer-batch1",
        type=Path,
        default=None,
        help=(
            "JSON document holding the batch-1 corpus WER and the checkpoint, chunk, "
            "corpus and dtype it was measured at. Without it a rung does not evaluate "
            "word error rate at all"
        ),
    )
    ladder.add_argument(
        "--checkpoint",
        default=None,
        help="what the server under test is serving, declared; required by --wer-batch1",
    )
    ladder.add_argument(
        "--dtype",
        default=None,
        help="the compute dtype class of the server under test; required by --wer-batch1",
    )
    ladder.add_argument("--out", type=Path, required=True)
    ladder.add_argument(
        "--host-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "sample the host and the GPU across every rung's window and judge validity "
            "over the whole record (METHODOLOGY section 7), which is also the only way a "
            "rung evaluates thermal (section 56). The default since the pressure "
            "thresholds were calibrated (DR-0007); needs --server-pid. --no-host-record "
            "runs the ladder from its load alone: the pacing threshold only, and no rung "
            "can pass because thermal is never evaluated"
        ),
    )
    ladder.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="the server's own process id, so its compute process is not counted foreign",
    )
    ladder.add_argument(
        "--processes",
        type=int,
        default=1,
        help=(
            "spread each rung's streams across this many load processes, each running "
            "the same event loop, and combine them into one rung. One event loop loses "
            "its own send schedule well below the concurrency the MVP bar asks for, so a "
            "ladder run at 1 measures the generator above a few tens of streams. The "
            "rung is the same rung whatever this is: the same seed puts the same session "
            "in the same slot, the window opens at one instant in every process, and the "
            "host record is taken once for the box"
        ),
    )
    ladder.add_argument("--gpu-index", type=int, default=0)
    ladder.add_argument(
        "--gpu-sample-interval-s",
        type=float,
        default=None,
        help="seconds between GPU samples across the window (default 1.0)",
    )
    ladder.add_argument(
        "--from-smi-xml",
        type=Path,
        default=None,
        help="sample a saved `nvidia-smi -q -x` document instead of the live command",
    )

    cal = sub.add_parser(
        "calibrate-psi",
        help=(
            "the calibration METHODOLOGY section 7 asks for: the pressure thresholds, "
            "from null-floor windows on this box, with provenance; refuses on a busy box"
        ),
    )
    cal.add_argument("--manifest", required=True, type=Path)
    cal.add_argument("--out", required=True, type=Path)
    cal.add_argument(
        "--endpoint",
        default=None,
        help=(
            "calibrate against the server under test at this WebSocket endpoint, the "
            "complete measurement at the reference concurrency (six streams), instead of "
            "the null floor; needs --server-pid"
        ),
    )
    cal.add_argument(
        "--ns",
        type=str,
        default=None,
        help=(
            "comma-separated concurrencies to drive the floor at; the default is the frozen "
            "document's own for the null floor (16,32,128) and the reference (6) with "
            "--endpoint"
        ),
    )
    cal.add_argument("--seeds", type=str, default=",".join(str(s) for s in constants.SEEDS))
    cal.add_argument("--window-s", type=float, default=constants.WINDOW_S)
    cal.add_argument(
        "--warm-up-s",
        type=float,
        default=None,
        help=(
            "run the convergence warm-up before each window; the default is none, because "
            "the null floor is at steady state the moment every stream is live"
        ),
    )
    cal.add_argument("--warm-up-reading-s", type=float, default=constants.WARM_UP_READING_S)
    cal.add_argument("--warm-up-convergence", type=float, default=constants.WARM_UP_CONVERGENCE)
    cal.add_argument("--warm-up-cap-s", type=float, default=constants.WARM_UP_CAP_S)
    cal.add_argument("--ramp-s", type=float, default=DEFAULT_RAMP_S)
    cal.add_argument("--frame-ms", type=int, default=constants.FRAME_MS)
    cal.add_argument("--interval-s", type=float, default=1.0, help="seconds between samples")
    cal.add_argument(
        "--quiet-s",
        type=float,
        default=None,
        help="quiet observation before the runs (default: the estimator's 60 s horizon)",
    )
    cal.add_argument("--gpu-index", type=int, default=0)
    cal.add_argument("--server-pid", type=int, default=None)
    cal.add_argument("--from-smi-xml", type=Path, default=None)
    cal.add_argument("--no-gpu", action="store_true", help="a box with no GPU record to sample")
    gate = sub.add_parser(
        "invariance",
        help=(
            "the batch-invariance gate: one corpus through a running server at concurrency "
            "1 / 32a / 32b / max, finals diffed against each other; exit 0 invariant, "
            "1 divergent, 2 no verdict"
        ),
    )
    gate.add_argument("--endpoint", required=True, help="ws://host:port/v1/stream")
    corpus = gate.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--manifest", type=Path, help="NeMo-compatible JSONL, 16 kHz mono")
    corpus.add_argument(
        "--synthetic", type=int, metavar="N", help="N clips of seeded noise instead of a manifest"
    )
    gate.add_argument(
        "--synthetic-s",
        type=float,
        default=invariance_gate.DEFAULT_SYNTHETIC_S,
        metavar="S",
        help=f"synthetic clip duration (default {invariance_gate.DEFAULT_SYNTHETIC_S})",
    )
    gate.add_argument("--seed", type=int, default=invariance_gate.DEFAULT_SEED)
    gate.add_argument("--chunk", default="160ms")
    gate.add_argument(
        "--churn-period-s",
        type=float,
        default=None,
        metavar="S",
        help=(
            "churn the max level: admit on a triangle wave between 1 and --max with this "
            "period, instead of holding a constant number in flight. The constant levels "
            "refill the instant a clip finishes, so occupancy sits pinned and no session "
            "sees its neighbour count move far; this asks whether a transcript survives "
            "that count moving underneath it. Recorded on the level, so a churned digest "
            "is never read as a constant-occupancy one"
        ),
    )
    gate.add_argument(
        "--max",
        type=int,
        default=invariance_gate.DEFAULT_MAX_CONCURRENCY,
        metavar="N",
        help=(
            "the max level: the server's ceiling once a capacity search has named it; "
            f"until then the largest ceiling batch size, {invariance_gate.DEFAULT_MAX_CONCURRENCY}"
        ),
    )
    gate.add_argument("--lang", default="en-US")
    gate.add_argument(
        "--phrases",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "a phrase book: run the gate with per-session phrase lists. Half the corpus "
            "carries a list and half carries none, interleaved, and a stream keeps the "
            "SAME list at every level so batch composition stays the only thing that "
            "changed. Two controls run first at concurrency 1: a positive control that "
            "must show a list changing a transcript (without it a server ignoring every "
            "list would pass), and a negative control that records any word an unrelated "
            "list inserted. Needs a server started with --biasing, and the record says so"
        ),
    )
    gate.add_argument(
        "--control-clips",
        type=int,
        default=invariance_gate.DEFAULT_CONTROL_CLIPS,
        metavar="N",
        help=(
            "clips the positive control runs, each twice at concurrency 1 "
            f"(default {invariance_gate.DEFAULT_CONTROL_CLIPS})"
        ),
    )
    gate.add_argument(
        "--out", type=Path, default=None, help="write the vb-invariance/1 record here"
    )

    rare = sub.add_parser(
        "rare-terms",
        help=(
            "what a phrase list buys and what it costs: term recall and false accepts "
            "against the bare arm, swept over the boosting weight"
        ),
        description=(
            "Run the corpus once per weight, every session carrying the whole term set, "
            "and report recall on the terms the audio contains separately from the false "
            "accepts on the ones it does not. Corpus word error rate is the wrong "
            "instrument for biasing and rides along only as a guard. The term set is "
            "chosen blind to the transcripts -- the corpus's own rare words by document "
            "frequency -- because selecting terms the bare pass already failed would "
            "guarantee the improvement and measure nothing. Needs a server with --biasing."
        ),
    )
    rare.add_argument("--endpoint", required=True, help="ws://host:port/v1/stream")
    rare.add_argument("--manifest", required=True, type=Path, help="NeMo-compatible JSONL")
    rare.add_argument(
        "--terms",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            'a term set as {"name": ..., "terms": [...]}. Without it the terms are '
            "derived from the corpus's own references by document frequency"
        ),
    )
    rare.add_argument(
        "--min-term-chars",
        type=int,
        default=rare_terms.DEFAULT_MIN_TERM_CHARS,
        metavar="N",
        help=f"shortest derived term (default {rare_terms.DEFAULT_MIN_TERM_CHARS})",
    )
    rare.add_argument(
        "--max-document-frequency",
        type=int,
        default=rare_terms.DEFAULT_MAX_DOCUMENT_FREQUENCY,
        metavar="N",
        help=(
            "a derived term appears in at most this many utterances (default "
            f"{rare_terms.DEFAULT_MAX_DOCUMENT_FREQUENCY})"
        ),
    )
    rare.add_argument(
        "--term-rule",
        choices=("rare", "unseen"),
        default="rare",
        help=(
            "'rare' takes the corpus's rare words by document frequency; 'unseen' keeps "
            "only those NOT in an English word list, which is the closest blind proxy here "
            "for a name the model may never have been trained on. It matters: of 256 terms "
            "derived by rarity alone on 2026-09-15, 245 were ordinary words like 'obliged' "
            "and 'morning', where boosting moved recall 0.925 to 0.950; over the 11 that "
            "were not in the word list it moved 0.583 to 0.833. Aggregating hides the second"
        ),
    )
    rare.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "the word list --term-rule unseen subtracts, one word per line; the default "
            "tries the system lists. Which one was used is recorded, because it is part of "
            "the selection rule"
        ),
    )
    rare.add_argument(
        "--boosts",
        default=",".join("bare" if b is None else f"{b:g}" for b in rare_terms.DEFAULT_BOOSTS),
        help="comma-separated weights to sweep; 'bare' sends no phrase list at all",
    )
    rare.add_argument("--concurrency", type=int, default=16)
    rare.add_argument(
        "--limit", type=int, default=None, metavar="N", help="use only the first N utterances"
    )
    rare.add_argument("--chunk", default="160ms")
    rare.add_argument("--lang", default="en-US")
    rare.add_argument("--seed", type=int, default=invariance_gate.DEFAULT_SEED)
    rare.add_argument(
        "--out", type=Path, default=None, help="write the vb-rare-terms/1 record here"
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    try:
        chunk = ChunkMode.parse(args.chunk)
    except ValueError as exc:
        print(f"verbatim-bench: {exc}")
        return 1
    if args.sessions < 1:
        print("verbatim-bench: --sessions must be >= 1")
        return 1
    spec = LoadSpec(
        endpoint=args.endpoint,
        manifest=args.manifest,
        sessions=args.sessions,
        chunk=chunk,
        profile=args.profile,
        seed=args.seed,
        ramp_s=args.ramp_s,
        words=args.words,
        lang=args.lang,
        arm=args.arm,
        x_ms=args.x_ms,
    )
    try:
        result = await run_load(spec)
    except (ValueError, OSError) as exc:
        print(f"verbatim-bench: {exc}")
        return 1
    out_path = write_results(result, args.out)
    data = result.to_json_dict()["result"]["latency_ms"]

    def summary(name: str) -> str:
        block = data[name]
        return (
            f"{name} p50={block['p50']:.2f}ms p95={block['p95']:.2f}ms "
            f"p99={block['p99']:.2f}ms n={block['n']}"
        )

    print(summary("first_partial"))
    print(summary("partial"))
    print(summary("final"))
    print(str(out_path))
    failed = result.to_json_dict()["result"]["sessions_failed"]
    return 2 if failed > 0 else 0


def _verify(args: argparse.Namespace) -> int:
    from verbatim_bench.verify import Finding, Level, VerifyReport, verify_file

    try:
        report = verify_file(args.path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"verbatim-bench: cannot verify {args.path}: {exc}")
        return 1
    findings = list(report.findings)
    if args.strict:
        findings = [
            Finding(Level.ERROR, finding.code, finding.message, finding.path)
            if finding.level is Level.WARNING
            else finding
            for finding in findings
        ]
        report = VerifyReport(findings=tuple(findings))
    if args.json:
        payload = {
            "ok": report.ok,
            "findings": [
                {
                    "level": finding.level.value,
                    "code": finding.code,
                    "message": finding.message,
                    "path": finding.path,
                }
                for finding in findings
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        for finding in findings:
            line = f"{finding.level.value.upper()} {finding.code}"
            if finding.path:
                line += f" {finding.path}: {finding.message}"
            else:
                line += f": {finding.message}"
            print(line)
        if report.ok:
            print("OK")
        else:
            n_errors = sum(1 for finding in findings if finding.level is Level.ERROR)
            n_warnings = sum(1 for finding in findings if finding.level is Level.WARNING)
            print(f"FAILED: {n_errors} error(s), {n_warnings} warning(s)")
    return 0 if report.ok else 2


def _env(args: argparse.Namespace) -> int:
    from verbatim_bench.env import (
        EnvironmentRefusal,
        SmiProbe,
        collect,
    )

    try:
        if args.from_smi_xml is not None:
            text = Path(args.from_smi_xml).read_text(encoding="utf-8")
            probe = SmiProbe.from_xml(text)
        else:
            probe = SmiProbe.from_command()
        repo_root = Path.cwd()
        record = collect(gpu=probe, gpu_index=args.gpu_index, repo_root=repo_root)
    except EnvironmentRefusal as exc:
        print(f"verbatim-bench env refused: {exc}")
        return 2
    except (OSError, ValueError) as exc:
        print(f"verbatim-bench: cannot collect environment: {exc}")
        return 1
    payload = record.to_json_dict()
    if args.json or args.out is None:
        print(json.dumps(payload, indent=2))
    if args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "env.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def _overrides_frozen_durations(args: argparse.Namespace) -> bool:
    """Whether the operator typed anything other than the frozen rung durations.

    A warning only. It cannot certify a run: every rung this harness produced on
    2026-09-11 passed this check and none of them ran a measurement window.
    """
    return (
        float(args.warm_up_s) != float(constants.WARM_UP_S)
        or float(args.window_s) != float(constants.WINDOW_S)
        or float(args.warm_up_reading_s) != float(constants.WARM_UP_READING_S)
        or float(args.warm_up_convergence) != float(constants.WARM_UP_CONVERGENCE)
        or float(args.warm_up_cap_s) != float(constants.WARM_UP_CAP_S)
    )


def ladder_load_spec(args: argparse.Namespace, plan: RungPlan, chunk: ChunkMode) -> LoadSpec:
    """The load one rung runs: the plan's warm-up and window, and a real ramp to N.

    The ramp is not forced to zero. All `n` streams have to be live before the warm-up
    clock starts, and the load generator establishes that by waiting for every slot to
    open its first session, so a stagger costs the rung nothing and a zero-length one
    buys it nothing but a thundering herd.
    """
    return LoadSpec(
        endpoint=args.endpoint,
        manifest=Path(args.manifest),
        sessions=plan.n,
        chunk=chunk,
        profile="uniform",
        seed=plan.seed,
        arm=args.arm,
        ramp_s=float(args.ramp_s),
        frame_ms=int(args.frame_ms),
        window_s=float(plan.window_s),
        warm_up_s=float(plan.warm_up_s),
        warm_up_reading_s=float(args.warm_up_reading_s),
        warm_up_convergence=float(args.warm_up_convergence),
        warm_up_cap_s=float(args.warm_up_cap_s),
    )


def _resolve_wer_reference(args: argparse.Namespace, chunk: ChunkMode) -> Batch1Reference | None:
    """Return the batch-1 reference this ladder may compare itself against, or `None`.

    A reference is optional, and without one the rung leaves word error rate out of the
    criteria it evaluated. A reference that was supplied and cannot be used is a usage
    error rather than a quiet skip: silently dropping the criterion would look exactly
    like not having asked for it, and the operator would never learn that the comparison
    they thought they had configured never happened.

    Two of the four coordinates the criterion is defined at are read from the run itself,
    the chunk and the corpus. The other two, the checkpoint and the dtype, are declared
    on the command line, because the ladder speaks to a server over a socket and never
    learns what it loaded. They are declarations and are recorded as such.

    Raises `ReferenceError` with a message fit to print.
    """
    if args.wer_batch1 is None:
        return None
    if not args.checkpoint or not args.dtype:
        raise ReferenceError("--wer-batch1 needs --checkpoint and --dtype to compare against")
    reference = load_batch1_reference(Path(args.wer_batch1))
    corpus_id = manifest_corpus_id(Path(args.manifest))
    if not reference.describes(
        checkpoint=args.checkpoint,
        chunk_ms=chunk.ms,
        corpus_id=corpus_id,
        dtype=args.dtype,
    ):
        raise ReferenceError(
            f"batch-1 reference {args.wer_batch1} was measured at "
            f"checkpoint={reference.checkpoint!r} chunk_ms={reference.chunk_ms} "
            f"corpus_id={reference.corpus_id!r} dtype={reference.dtype!r}, and this run is "
            f"checkpoint={args.checkpoint!r} chunk_ms={chunk.ms} corpus_id={corpus_id!r} "
            f"dtype={args.dtype!r}: the criterion is not defined across them"
        )
    return reference


def _ladder(args: argparse.Namespace) -> int:
    import asyncio

    from verbatim_bench.env import SmiProbe
    from verbatim_bench.hostrecord import (
        DEFAULT_INTERVAL_S,
        LiveSmiProbe,
        WindowRecorder,
        run_load_recorded,
        run_sharded_load_recorded,
    )
    from verbatim_bench.ladder import Rung, run_ladder, rung_from_run
    from verbatim_bench.multiproc import run_sharded_load
    from verbatim_bench.serverfacts import ArmContradiction, check_arm, read_server_facts

    try:
        seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    except ValueError:
        print("verbatim-bench: --seeds must be comma-separated integers")
        return 1
    if not seeds:
        print("verbatim-bench: --seeds must list at least one seed")
        return 1
    processes = int(args.processes)
    if processes < 1:
        print("verbatim-bench: --processes must be >= 1")
        return 1
    n0 = args.n0
    if n0 is None:
        from verbatim_bench.ladder import n0_for

        n0 = n0_for(args.ceiling)
    if n0 < 1:
        print("verbatim-bench: --n0 must be >= 1")
        return 1
    if _overrides_frozen_durations(args):
        # A warning about what was asked for, and nothing more. Whether the run was
        # canonical is decided per rung, from the load the generator actually executed.
        print("verbatim-bench: warning: overridden durations mark the run non-canonical")
    chunk = ChunkMode.parse(160)
    threshold_ms = chunk.ms + constants.X_MS
    try:
        reference = _resolve_wer_reference(args, chunk)
    except ReferenceError as exc:
        print(f"verbatim-bench: {exc}")
        return 1

    # Read the server before a single stream is opened, and refuse a run whose label
    # contradicts it. A row mislabelled at the top carries real numbers under the wrong
    # arm and nothing downstream can tell -- the same defect class as a warm-up that
    # recorded a capture it never made (docs/decisions/0011).
    server_facts = read_server_facts(args.endpoint)
    if server_facts is None:
        print(
            "verbatim-bench: warning: the server did not answer /readyz, so this run "
            "records no observation of what it was serving and its arm name is the only "
            "statement of it"
        )
    try:
        check_arm(
            server_facts,
            arm=str(args.arm),
            declared_dtype=args.dtype,
            declared_chunk_ms=chunk.ms,
        )
    except ArmContradiction as exc:
        print(f"verbatim-bench: refusing to run: {exc}")
        return 1

    probe = None
    interval_s = DEFAULT_INTERVAL_S
    if args.host_record:
        if args.server_pid is None:
            print(
                "verbatim-bench: the host record needs --server-pid, so the server's own "
                "compute process is told apart from a foreign one; pass it, or "
                "--no-host-record to run the ladder from its load alone"
            )
            return 1
        if args.gpu_sample_interval_s is not None:
            interval_s = float(args.gpu_sample_interval_s)
            if interval_s <= 0:
                print("verbatim-bench: --gpu-sample-interval-s must be positive")
                return 1
        if args.from_smi_xml is not None:
            probe = SmiProbe.from_xml(Path(args.from_smi_xml).read_text(encoding="utf-8"))
        else:
            probe = LiveSmiProbe()

    async def _run_unrecorded(spec: LoadSpec) -> RunResult:
        if processes == 1:
            return await run_load(spec)
        return (await run_sharded_load(spec, processes=processes)).result

    def _make_rung(plan: RungPlan) -> Rung:
        spec = ladder_load_spec(args, plan, chunk)
        host = None
        if probe is not None:
            recorder = WindowRecorder(
                gpu=probe,
                gpu_index=int(args.gpu_index),
                server_pid=int(args.server_pid),
                interval_s=interval_s,
            )
            if processes == 1:
                result = asyncio.run(run_load_recorded(spec, recorder))
            else:
                result = asyncio.run(run_sharded_load_recorded(spec, recorder, processes=processes))
            host = recorder.finish()
        else:
            result = asyncio.run(_run_unrecorded(spec))
        return rung_from_run(
            result,
            plan=plan,
            threshold_ms=threshold_ms,
            batch1_wer=reference.wer if reference is not None else None,
            host=host,
            # Recorded on the rung so an UNSTABLE verdict can be read against the
            # tolerance the readings were actually tested at, not against the default
            # someone assumes it ran with.
            warm_up_convergence=float(args.warm_up_convergence),
        )

    outcome = run_ladder(
        _make_rung,
        n0=n0,
        seeds=seeds,
        warm_up_s=float(args.warm_up_s),
        window_s=float(args.window_s),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": args.arm,
        "s": outcome.s,
        "ending_criterion": outcome.ending_criterion.value
        if outcome.ending_criterion is not None
        else None,
        "aborted": outcome.aborted,
        "abort_reason": outcome.abort_reason,
        "config": {
            # True only when every rung in this ladder ran the frozen window. A ladder
            # with no rungs ran nothing and certifies nothing.
            "canonical_window": bool(outcome.rungs)
            and all(rung.canonical_window for rung in outcome.rungs),
            "warm_up_s": float(args.warm_up_s),
            "window_s": float(args.window_s),
            "warm_up_reading_s": float(args.warm_up_reading_s),
            "warm_up_convergence": float(args.warm_up_convergence),
            "warm_up_cap_s": float(args.warm_up_cap_s),
            "ramp_s": float(args.ramp_s),
            "frame_ms": int(args.frame_ms),
            # How many load processes drove each rung. One is one event loop, which is
            # what every ladder before this ran; anything more is the same rung split
            # across that many, combined over one shared window.
            "processes": processes,
            # What the server said it was running, read from /readyz before the first
            # stream opened. Null when it did not answer, which is "no observation" and
            # not "agreed".
            "server": server_facts.to_json_dict() if server_facts is not None else None,
            # Which batch-1 reference the WER criterion was read against, or null when
            # no reference was supplied and no rung evaluated it. A rung's
            # `wer_vs_batch1` is the signed difference from this number, so the two
            # together give back the corpus WER the run measured.
            "wer_batch1": reference.to_json_dict() if reference is not None else None,
            # Whether every rung's window was sampled for its host record. Without it no
            # rung evaluates thermal and none can pass.
            "host_record": bool(args.host_record),
            "server_pid": int(args.server_pid) if args.server_pid is not None else None,
            "gpu_index": int(args.gpu_index),
            "gpu_sample_interval_s": interval_s if args.host_record else None,
        },
        "rungs": outcome.to_json_list(include_window=True),
    }
    (out_dir / "ladder.json").write_text(json.dumps(payload, indent=2) + "\n")
    if outcome.aborted:
        print(f"verbatim-bench: ladder aborted: {outcome.abort_reason}")
        return 2
    print(f"S={outcome.s} criterion={payload['ending_criterion']}")
    return 0


def _calibrate_psi(args: argparse.Namespace) -> int:
    import asyncio

    from verbatim_bench.calibrate import QUIET_OBSERVATION_S, CalibrationRefusal, calibrate
    from verbatim_bench.env import SmiProbe
    from verbatim_bench.hostrecord import LiveSmiProbe

    try:
        seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    except ValueError:
        print("verbatim-bench: --seeds must be comma-separated integers")
        return 1
    try:
        ns = (
            None
            if args.ns is None
            else tuple(int(part) for part in str(args.ns).split(",") if part.strip())
        )
    except ValueError:
        print("verbatim-bench: --ns must be comma-separated integers")
        return 1
    if not seeds or (ns is not None and (not ns or min(ns) < 1)) or args.interval_s <= 0:
        print("verbatim-bench: need at least one seed, every N >= 1 and a positive --interval-s")
        return 1
    if args.endpoint is not None and args.server_pid is None:
        print(
            "verbatim-bench: --endpoint calibrates against the server under test and needs "
            "--server-pid, so its CPU is in the record and it is not counted foreign"
        )
        return 1
    if args.no_gpu:
        probe = None
    elif args.from_smi_xml is not None:
        probe = SmiProbe.from_xml(Path(args.from_smi_xml).read_text(encoding="utf-8"))
    else:
        probe = LiveSmiProbe()
    try:
        record = asyncio.run(
            calibrate(
                manifest=Path(args.manifest),
                endpoint=args.endpoint,
                ns=ns,
                seeds=seeds,
                window_s=float(args.window_s),
                warm_up_s=None if args.warm_up_s is None else float(args.warm_up_s),
                warm_up_reading_s=float(args.warm_up_reading_s),
                warm_up_convergence=float(args.warm_up_convergence),
                warm_up_cap_s=float(args.warm_up_cap_s),
                ramp_s=float(args.ramp_s),
                frame_ms=int(args.frame_ms),
                interval_s=float(args.interval_s),
                quiet_s=QUIET_OBSERVATION_S if args.quiet_s is None else float(args.quiet_s),
                gpu=probe,
                gpu_index=int(args.gpu_index),
                server_pid=args.server_pid,
                repo_root=Path.cwd(),
            )
        )
    except CalibrationRefusal as exc:
        print(f"verbatim-bench calibrate-psi refused: {exc}")
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(
        json.dumps(record.to_json_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if not record.canonical:
        print("verbatim-bench: warning: overridden durations, seeds or framing: not canonical")
    print(record.constants_lines(), end="")
    return 0


def _invariance(args: argparse.Namespace) -> int:
    """The gate: 0 invariant, 1 divergent, 2 no verdict (refused, errored or vacuous)."""
    import asyncio

    chunk = ChunkMode.parse(args.chunk)
    book = None
    try:
        if args.phrases is not None:
            book = load_phrase_book(args.phrases)
        # What the server says it is, before a byte is sent. Biasing changes the decode
        # for every row, biased or not, so a run that sends phrase lists to a server
        # without it -- or withholds them from a server with it -- is a different arm
        # than the record would claim.
        check_arm(
            read_server_facts(args.endpoint),
            arm="invariance",
            declared_chunk_ms=chunk.ms,
            declared_biasing=book is not None,
        )
        if args.manifest is not None:
            clips = invariance_gate.manifest_corpus(args.manifest)
            corpus = {
                "kind": "manifest",
                "path": str(args.manifest),
                "id": manifest_corpus_id(args.manifest),
                "utterances": len(clips),
            }
        else:
            clips = invariance_gate.synthetic_corpus(
                args.synthetic, duration_s=args.synthetic_s, seed=args.seed
            )
            corpus = {
                "kind": "synthetic",
                "utterances": len(clips),
                "duration_s": args.synthetic_s,
                "seed": args.seed,
            }
        levels = invariance_gate.default_levels(args.max, churn_period_s=args.churn_period_s)
        invariance_gate.check_levels(levels, len(clips))
    except (invariance_gate.GateRefusal, ArmContradiction, PhraseBookError, ValueError) as exc:
        print(f"refused: {exc}")
        return invariance_gate.EXIT_NO_VERDICT
    report = asyncio.run(
        invariance_gate.run_gate(
            args.endpoint,
            clips,
            levels,
            chunk=chunk,
            lang=args.lang,
            seed=args.seed,
            corpus=corpus,
            book=book,
            control_clips=args.control_clips,
        )
    )
    print(report.render())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"record: {args.out}")
    return report.exit_code


def _parse_boosts(raw: str) -> tuple[float | None, ...]:
    """`bare,1,2,4` into arms. `bare` sends no phrase list; every other entry is a weight."""
    out: list[float | None] = []
    for piece in raw.split(","):
        token = piece.strip().lower()
        if not token:
            continue
        if token in ("bare", "none", "off"):
            out.append(None)
            continue
        try:
            out.append(float(token))
        except ValueError:
            raise rare_terms.TermSetError(
                f"invalid --boosts entry {piece!r}: expected a number or 'bare'"
            ) from None
    if not out:
        raise rare_terms.TermSetError("--boosts named no arms")
    return tuple(out)


def _rare_terms(args: argparse.Namespace) -> int:
    """Term recall against the bare arm, swept over the weight. 0 on a usable reading."""
    import asyncio

    chunk = ChunkMode.parse(args.chunk)
    try:
        boosts = _parse_boosts(args.boosts)
        # The arms differ only in the weight, so the server has to be the same server for
        # all of them, and it has to be one that serves phrase lists at all.
        check_arm(
            read_server_facts(args.endpoint),
            arm="rare-terms",
            declared_chunk_ms=chunk.ms,
            declared_biasing=any(boost is not None for boost in boosts),
        )
        clips = invariance_gate.manifest_corpus(args.manifest)
        if args.limit is not None:
            clips = clips[: args.limit]
        if not clips:
            raise rare_terms.TermSetError("the corpus is empty")
        if args.terms is not None:
            term_set = rare_terms.load_term_set(args.terms)
        else:
            dictionary = dictionary_path = None
            if args.term_rule == "unseen":
                dictionary, dictionary_path = rare_terms.load_dictionary(args.dictionary)
            derived = rare_terms.derive_terms(
                [clip.text for clip in clips],
                min_chars=args.min_term_chars,
                max_document_frequency=args.max_document_frequency,
                dictionary=dictionary,
            )
            if not derived:
                raise rare_terms.TermSetError(
                    f"no term met the {args.term_rule!r} rule: loosen --min-term-chars or "
                    "--max-document-frequency, or pass --terms"
                )
            term_set = rare_terms.TermSet(
                name=(
                    f"derived-{args.term_rule}"
                    f"-df{args.max_document_frequency}-c{args.min_term_chars}"
                ),
                terms=derived,
                # The rule travels with the set: a term set nobody can rebuild is not a
                # measurement anyone can repeat.
                rule={
                    "kind": args.term_rule,
                    "min_chars": args.min_term_chars,
                    "max_document_frequency": args.max_document_frequency,
                    "dictionary": dictionary_path,
                    "corpus_utterances": len(clips),
                },
            )
    except (rare_terms.TermSetError, ArmContradiction, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    report = asyncio.run(
        rare_terms.run_rare_terms(
            args.endpoint,
            clips,
            term_set,
            chunk=chunk,
            boosts=boosts,
            concurrency=args.concurrency,
            lang=args.lang,
            seed=args.seed,
            corpus={
                "kind": "manifest",
                "path": str(args.manifest),
                "id": manifest_corpus_id(args.manifest),
                "utterances": len(clips),
            },
        )
    )
    print(report.render())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"record: {args.out}")
    return 0 if report.usable else 2


def main(argv: list[str] | None = None) -> int:
    """Entry point: 0 on a completed run, 1 on a usage error, 2 if any session failed."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code == 2 else int(exc.code or 0)
    if args.command == "run":
        import asyncio

        return asyncio.run(_run(args))
    if args.command == "verify":
        return _verify(args)
    if args.command == "env":
        return _env(args)
    if args.command == "calibrate-psi":
        return _calibrate_psi(args)
    if args.command == "ladder":
        return _ladder(args)
    if args.command == "invariance":
        return _invariance(args)
    if args.command == "rare-terms":
        return _rare_terms(args)
    parser.print_usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
