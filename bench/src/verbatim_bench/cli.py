# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``verbatim-bench run|row|validate|invariance|wer|env|verify|replay|record``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verbatim_bench import constants
from verbatim_bench.client import ChunkMode
from verbatim_bench.pace import LoadSpec, run_load
from verbatim_bench.results import write_results


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
    run.add_argument("--ramp-s", type=float, default=60.0)
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


def _ladder(args: argparse.Namespace) -> int:
    import asyncio

    from verbatim_bench import constants as _constants
    from verbatim_bench.ladder import Rung, RungPlan, run_ladder
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
    canonical = float(args.warm_up_s) == float(_constants.WARM_UP_S) and float(
        args.window_s
    ) == float(_constants.WINDOW_S)
    if not canonical:
        print("verbatim-bench: warning: overridden durations mark the run non-canonical")
    chunk = ChunkMode.parse(160)
    threshold_ms = chunk.ms + _constants.X_MS

    def _make_rung(plan: RungPlan) -> Rung:
        from verbatim_bench.ladder import Criterion

        async def _once() -> tuple[list[float], int, object]:
            spec = LoadSpec(
                endpoint=args.endpoint,
                manifest=Path(args.manifest),
                sessions=plan.n,
                chunk=chunk,
                profile="uniform",
                seed=plan.seed,
                ramp_s=0.0,
            )
            result = await run_load(spec)
            samples = [v for s in result.sessions for v in s.partial_ms]
            refused = sum(1 for s in result.sessions if s.error is not None)
            return samples, refused, result

        samples, refused, _ = asyncio.run(_once())
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
            n=plan.n,
            seed=plan.seed,
            p95_ms=float(p95),
            wer_vs_batch1=None,
            first_failing_criterion=criterion,
            criteria_evaluated=tuple(criteria),
            valid=True,
            invalid_reason=None,
            warm_up_s=float(args.warm_up_s),
            sessions_refused=int(refused),
            sessions_dropped=0,
            sessions_without_final=0,
            canonical_window=bool(canonical),
        )

    from verbatim_bench.ladder import n0_for as _n0_for  # noqa: F401

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
        "config": {"canonical_window": bool(canonical)},
        "rungs": outcome.to_json_list(),
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
