# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``verbatim-bench run|row|validate|invariance|wer|env|verify|replay|record``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from verbatim_bench import constants
from verbatim_bench.client import ChunkMode
from verbatim_bench.corpus import manifest_corpus_id
from verbatim_bench.pace import DEFAULT_RAMP_S, LoadSpec, run_load
from verbatim_bench.results import write_results
from verbatim_bench.wer import Batch1Reference, ReferenceError, load_batch1_reference

if TYPE_CHECKING:
    from verbatim_bench.ladder import RungPlan


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
        action="store_true",
        help=(
            "sample the host and the GPU across every rung's window and judge validity "
            "over the whole record (METHODOLOGY section 7), which is also the only way a "
            "rung evaluates thermal (section 56). Needs --server-pid. While the pressure "
            "thresholds are unfrozen every sampled rung is invalid, as the document says"
        ),
    )
    ladder.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="the server's own process id, so its compute process is not counted foreign",
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
        "--n",
        type=int,
        default=constants.LADDER_N0_WITHOUT_CEILING,
        help="streams per null-floor window (default: the ladder's start without a ceiling)",
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
    cal.add_argument("--quiet-s", type=float, default=None, help="quiet observation (default 10)")
    cal.add_argument("--gpu-index", type=int, default=0)
    cal.add_argument("--server-pid", type=int, default=None)
    cal.add_argument("--from-smi-xml", type=Path, default=None)
    cal.add_argument("--no-gpu", action="store_true", help="a box with no GPU record to sample")
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
    )
    from verbatim_bench.ladder import Rung, run_ladder, rung_from_run

    try:
        seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    except ValueError:
        print("verbatim-bench: --seeds must be comma-separated integers")
        return 1
    if not seeds:
        print("verbatim-bench: --seeds must list at least one seed")
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

    probe = None
    interval_s = DEFAULT_INTERVAL_S
    if args.host_record:
        if args.server_pid is None:
            print(
                "verbatim-bench: --host-record needs --server-pid: the server's own compute "
                "process must be told apart from a foreign one"
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
            result = asyncio.run(run_load_recorded(spec, recorder))
            host = recorder.finish()
        else:
            result = asyncio.run(run_load(spec))
        return rung_from_run(
            result,
            plan=plan,
            threshold_ms=threshold_ms,
            batch1_wer=reference.wer if reference is not None else None,
            host=host,
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
    if not seeds or args.n < 1 or args.interval_s <= 0:
        print("verbatim-bench: need at least one seed, --n >= 1 and a positive --interval-s")
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
                n=int(args.n),
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
    parser.print_usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
