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
from verbatim_bench.pace import (
    DEFAULT_RAMP_S,
    LoadSpec,
    executed_canonical_window,
    run_load,
    window_partial_samples,
)
from verbatim_bench.results import write_results

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
    ladder.add_argument("--out", type=Path, required=True)
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
        window_s=float(plan.window_s),
        warm_up_s=float(plan.warm_up_s),
        warm_up_reading_s=float(args.warm_up_reading_s),
        warm_up_convergence=float(args.warm_up_convergence),
        warm_up_cap_s=float(args.warm_up_cap_s),
    )


def _ladder(args: argparse.Namespace) -> int:
    import asyncio

    from verbatim_bench.ladder import Criterion, Rung, run_ladder
    from verbatim_bench.results import percentile

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

    def _make_rung(plan: RungPlan) -> Rung:
        result = asyncio.run(run_load(ladder_load_spec(args, plan, chunk)))
        refused = sum(1 for session in result.sessions if session.error is not None)
        warm_up_s = result.warm_up_length_s
        # What the load did, which is what the rung reports: the warm-up and window it
        # ran, the wall clock it took, and whether that was the frozen window.
        ran: dict[str, object] = {
            "n": plan.n,
            "seed": plan.seed,
            "wer_vs_batch1": None,
            "valid": True,
            "invalid_reason": None,
            "warm_up_s": float(warm_up_s if warm_up_s is not None else args.warm_up_s),
            "sessions_refused": int(refused),
            "sessions_dropped": 0,
            "sessions_without_final": 0,
            "canonical_window": executed_canonical_window(result),
            "window_s": float(plan.window_s),
            "wall_clock_s": float(result.wall_clock_s),
        }
        if result.warm_up_converged is False:
            # The frozen document: failure to converge by the cap fails the rung as
            # unstable. No window opened, so this rung established no criterion at all
            # and has no percentile to report.
            return Rung(
                p95_ms=float("inf"),
                first_failing_criterion=Criterion.UNSTABLE,
                criteria_evaluated=(),
                **ran,  # type: ignore[arg-type]
            )
        samples = window_partial_samples(result)
        p95 = percentile(samples, 95) if samples else float("inf")
        # Only two of the criteria a rung must meet are established here: latency, and
        # then only when the window produced samples to take a p95 of, and the refused
        # half of integrity, which is counted for every session. Word error rate, drops,
        # missing finals and throttle events are not evaluated, so they are not listed
        # and this rung cannot report a pass for them.
        criteria: list[Criterion] = []
        if samples:
            criteria.append(Criterion.LATENCY)
        criteria.append(Criterion.INTEGRITY_REFUSED)
        criterion: Criterion | None = None
        if refused:
            criterion = Criterion.INTEGRITY_REFUSED
        elif samples and p95 > threshold_ms:
            criterion = Criterion.LATENCY
        return Rung(
            p95_ms=float(p95),
            first_failing_criterion=criterion,
            criteria_evaluated=tuple(criteria),
            **ran,  # type: ignore[arg-type]
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
        },
        "rungs": outcome.to_json_list(include_window=True),
    }
    (out_dir / "ladder.json").write_text(json.dumps(payload, indent=2) + "\n")
    if outcome.aborted:
        print(f"verbatim-bench: ladder aborted: {outcome.abort_reason}")
        return 2
    print(f"S={outcome.s} criterion={payload['ending_criterion']}")
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
    if args.command == "ladder":
        return _ladder(args)
    parser.print_usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
