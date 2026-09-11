# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The server's entry point: ``verbatim serve`` and ``verbatim doctor``.

The harness has its own CLI (``verbatim-bench``, under ``bench/``); the two never
share a process and the harness never imports this package.

``doctor`` is load-bearing rather than decorative: it reports which NeMo track is
installed and whether it carries the graphed streaming encoder step (NeMo PR
#15863). ``serve`` runs the same check first and refuses the graph path when the
step is absent, rather than silently running eager: ``--eager`` runs eager and
says so, so a benchmark row can never look graphed without being graphed.

Exit codes: 0 ran and stopped cleanly (or, for ``doctor``, the graph path is
available); 2 the operator's input or configuration was refused, with the reason;
3 the runtime cannot do what was asked (no NeMo, no CUDA device, or the graph path
without ``--eager``); 1 something unexpected, with ``--traceback`` for the rest.

When the pipeline refuses to build, the operator gets one message naming what was
asked, the checkpoint, the chunk mode and its ``att_context_size``, the slots, and
NeMo's own words, and exit code 2. The stack trace is there under ``--traceback``.
``demo`` and ``invariance`` are later rounds.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from verbatim import __version__
from verbatim.config import VALID_CHUNK_MS, ChunkMode
from verbatim.core.errors import VerbatimError
from verbatim.pipelines import registry
from verbatim.pipelines.base import PipelineAdapter
from verbatim.pipelines.cache_aware_rnnt import NeMoBoundary
from verbatim.pipelines.nemo_runtime import (
    NeMoPipelineSpec,
    PipelineBuildError,
    RuntimeReport,
    att_context_size,
    build_pipeline,
    inspect_runtime,
)
from verbatim.scheduler.graph_budget import ConfigError
from verbatim.serve import Endpoints, ServeSettings, engine_config, run_server

__all__ = ["EXIT_CONFIG", "EXIT_OK", "EXIT_RUNTIME", "EXIT_UNEXPECTED", "Hooks", "main"]

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_CONFIG = 2
EXIT_RUNTIME = 3


def build_boundary(spec: NeMoPipelineSpec) -> NeMoBoundary:
    """Build NeMo's pipeline for ``spec`` and bind it: the model load, then the seam
    the adapter drives it through."""
    return NeMoBoundary.from_pipeline(build_pipeline(spec))


@dataclass(frozen=True, slots=True)
class Hooks:
    """The seams ``main`` reaches the outside world through, injectable for tests:
    the runtime probe, the NeMo build (returning the bound boundary), the server, and
    where output goes."""

    inspect_runtime: Callable[[], RuntimeReport] = inspect_runtime
    build_boundary: Callable[[NeMoPipelineSpec], NeMoBoundary] = build_boundary
    run_server: Callable[..., Any] = run_server
    on_ready: Callable[[Endpoints, Callable[[], None]], None] | None = None
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)


class _Refused(Exception):
    """A refusal with an exit code and an operator-facing message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _chunk_arg(value: str) -> ChunkMode:
    text = value.strip().lower().removesuffix("ms")
    valid = ", ".join(f"{v}ms" for v in VALID_CHUNK_MS)
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"invalid chunk {value!r}: expected one of {valid}")
    ms = int(text)
    if ms not in VALID_CHUNK_MS:
        raise argparse.ArgumentTypeError(f"invalid chunk {value!r}: expected one of {valid}")
    return ChunkMode(ms)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _port(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a port 0..65535, got {value!r}") from exc
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError(f"expected a port 0..65535, got {value!r}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verbatim",
        description=(
            "Multi-tenant tick-scheduled streaming ASR server over NeMo's cache-aware pipeline."
        ),
    )
    parser.add_argument("--version", action="version", version=f"verbatim {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    serve = commands.add_parser(
        "serve",
        help="serve one checkpoint at one chunk mode on both wires",
        description=(
            "Serve one checkpoint at one chunk mode over the Riva-compatible gRPC subset and "
            "the WebSocket demo protocol, through one engine. The bucket is never assumed: "
            "pass --ceiling from a measured row, or --bucket to name an uncalibrated one for "
            "the ladder run."
        ),
    )
    serve.add_argument("model", help="checkpoint name (Hugging Face) or a .nemo path")
    serve.add_argument(
        "--chunk",
        required=True,
        type=_chunk_arg,
        metavar="MS",
        help="chunk mode, one of " + ", ".join(f"{v}ms" for v in VALID_CHUNK_MS),
    )
    sizing = serve.add_mutually_exclusive_group(required=True)
    sizing.add_argument(
        "--ceiling",
        type=_positive_int,
        metavar="N",
        help="the calibrated ceiling from a measured row for this GPU and chunk mode",
    )
    sizing.add_argument(
        "--bucket",
        type=_positive_int,
        metavar="N",
        help="an uncalibrated bucket, named as such, for the ladder run that measures the ceiling",
    )
    serve.add_argument("--host", default="0.0.0.0", help="bind address for both wires")
    serve.add_argument(
        "--grpc-port", type=_port, default=50051, help="Riva gRPC port (0: ephemeral)"
    )
    serve.add_argument("--ws-port", type=_port, default=8080, help="WebSocket port (0: ephemeral)")
    serve.add_argument(
        "--idle-timeout",
        type=float,
        default=30.0,
        metavar="S",
        help="close a session that has sent no audio for this many seconds (DEADLINE_EXCEEDED)",
    )
    serve.add_argument(
        "--ring-seconds",
        type=float,
        default=3.0,
        metavar="S",
        help="per-session ring buffer; a full ring is back-pressure, never dropped audio",
    )
    serve.add_argument(
        "--stop-history-eou-ms",
        type=int,
        default=800,
        metavar="MS",
        help="end-of-utterance silence for NeMo's endpointer; 0 disables it",
    )
    serve.add_argument(
        "--pipeline",
        choices=("cache_aware_rnnt", "fake"),
        default="cache_aware_rnnt",
        help="cache_aware_rnnt loads the checkpoint; fake serves scripted transcripts, loudly",
    )
    serve.add_argument(
        "--eager",
        action="store_true",
        help="run the encoder step eager when the installed NeMo lacks the graphed step; "
        "without it, serve refuses rather than degrades",
    )
    serve.add_argument(
        "--att-context-left",
        type=int,
        metavar="N",
        help="left attention context for a checkpoint whose family is not known here",
    )
    serve.add_argument("--language-code", default="en-US", help="default language code on the wire")
    serve.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="NeMo compute precision",
    )
    serve.add_argument("--device-id", type=int, default=0, help="CUDA device index")
    serve.add_argument("--traceback", action="store_true", help="show stack traces on failure")

    doctor = commands.add_parser(
        "doctor",
        help="report the installed NeMo track and whether the graph path is available",
        description=(
            "Report which NeMo track is installed and whether it carries the graphed "
            "streaming encoder step. Exit 0 when the graph path is available, 3 when the "
            "runtime can only run eager or cannot run at all."
        ),
    )
    doctor.add_argument("--traceback", action="store_true", help="show stack traces on failure")
    return parser


def _settings(args: argparse.Namespace) -> ServeSettings:
    try:
        return ServeSettings(
            model=args.model,
            chunk=args.chunk,
            ceiling=args.ceiling,
            bucket=args.bucket,
            host=args.host,
            ws_port=args.ws_port,
            grpc_port=args.grpc_port,
            idle_timeout_s=args.idle_timeout,
            ring_seconds=args.ring_seconds,
            stop_history_eou_ms=args.stop_history_eou_ms,
            pipeline=args.pipeline,
            eager=args.eager,
            att_context_left=args.att_context_left,
            language_code=args.language_code,
            compute_dtype=args.compute_dtype,
            device_id=args.device_id,
        )
    except ConfigError as exc:
        raise _Refused(EXIT_CONFIG, str(exc)) from exc


def _build_adapter(settings: ServeSettings, hooks: Hooks) -> tuple[PipelineAdapter, list[str]]:
    """The pipeline adapter for these settings, and the banner lines that describe it.

    The order is deliberate: the runtime is inspected and the graph path decided
    before any model is loaded, so a refusal costs nothing, and the config guards
    run before the checkpoint download too.
    """
    config = engine_config(settings)
    if settings.pipeline == "fake":
        try:
            adapter = registry.build_for(config)
        except (ConfigError, VerbatimError) as exc:
            raise _Refused(EXIT_CONFIG, str(exc)) from exc
        return adapter, [
            "pipeline     FAKE: scripted transcripts, nothing is recognised; for wiring "
            "and harness checks only",
        ]

    report = hooks.inspect_runtime()
    if not report.usable:
        raise _Refused(
            EXIT_RUNTIME,
            "the runtime cannot serve a checkpoint:\n  "
            + "\n  ".join(report.lines())
            + "\ninstall the model runtime with pip install 'verbatim[nemo]' on a machine with "
            "a CUDA device; --pipeline fake serves scripted transcripts without one",
        )
    if not report.graph_step and not settings.eager:
        raise _Refused(
            EXIT_RUNTIME,
            "the installed NeMo lacks the graphed streaming encoder step (NeMo PR #15863), "
            "so the graph path is not available; refusing rather than running eager "
            "unasked. Pass --eager to run the encoder step eager and record it in the row.\n  "
            + "\n  ".join(report.lines()),
        )
    use_graphs = report.graph_step and not settings.eager
    try:
        att_context = att_context_size(
            settings.model, settings.chunk, left=settings.att_context_left
        )
        spec = NeMoPipelineSpec(
            model=settings.model,
            chunk=settings.chunk,
            att_context=(att_context[0], att_context[1]),
            num_slots=config.num_slots,
            batch_size=max(config.buckets or (1,)),
            stop_history_eou_ms=settings.stop_history_eou_ms,
            use_cuda_graphs=use_graphs,
            compute_dtype=settings.compute_dtype,
            device_id=settings.device_id,
        )
    except ConfigError as exc:
        raise _Refused(EXIT_CONFIG, str(exc)) from exc
    try:
        boundary = hooks.build_boundary(spec)
    except PipelineBuildError as exc:
        raise _Refused(EXIT_CONFIG, str(exc)) from exc
    try:
        adapter = registry.build_for(
            config, boundary=boundary, language_code=settings.language_code
        )
    except (ConfigError, VerbatimError) as exc:
        raise _Refused(
            EXIT_CONFIG,
            f"the built pipeline does not fit the scheduler configuration: {exc}",
        ) from exc
    graphs = (
        "graph path (CUDA graphs, NeMo PR #15863)"
        if use_graphs
        else "EAGER encoder step, by --eager"
    )
    return adapter, [
        f"pipeline     NeMo cache-aware RNNT, {settings.compute_dtype} on cuda:{settings.device_id}"
        + (f" ({report.device_name})" if report.device_name else ""),
        f"nemo         {report.nemo_version}, torch {report.torch_version}",
        f"chunk mode   {settings.chunk.ms} ms (att_context_size {att_context})",
        f"graphs       {graphs}",
    ]


def _precision_lines(settings: ServeSettings) -> list[str]:
    """Decision record 0003, item 4: the operator reads at startup what the chosen
    precision does to batch invariance, with the counts and the record, instead of
    finding it in a document. The counts are the record's (evidence section 21),
    quoted here, not measured here; the fake pipeline has no precision to speak of.
    """
    if settings.pipeline == "fake":
        return []
    measured = (
        "batch invariance MEASURED TO FAIL at bfloat16 on this server's own code path: "
        "287 of 2,939 utterances diverged on an A6000 and 264 on a B300 (float32: 6 and 1), "
        "and equalising row lengths made it worse"
    )
    if settings.compute_dtype == "float32":
        return [
            "precision    float32: the precision at which batch invariance was measured, "
            "6 of 2,939 utterances divergent on an A6000 and 1 on a B300; docs/decisions/0003"
        ]
    if settings.compute_dtype == "bfloat16":
        head = f"precision    bfloat16 (the default): {measured}."
    else:
        head = f"precision    {settings.compute_dtype}: not measured at this precision; {measured}."
    return [
        head,
        "             A row from this run must cite docs/decisions/0003; pass "
        "--compute-dtype float32 for the property this server is named for.",
    ]


def _banner(settings: ServeSettings, adapter_lines: list[str]) -> list[str]:
    config = engine_config(settings)
    if settings.calibrated:
        admission = f"ceiling {settings.ceiling} streams, calibrated (from a measured row)"
    else:
        admission = f"bucket {settings.bucket} streams, UNCALIBRATED (named for the ladder run)"
    return [
        f"checkpoint   {settings.model}",
        *adapter_lines,
        *_precision_lines(settings),
        f"admission    {admission}",
        f"slots        {config.num_slots} NeMo slots: bucket {max(config.buckets or (0,))}, "
        f"{config.effective_pad} steady pads, {config.edge_pad_rows} edge pads, "
        f"{config.drain_margin} margin",
        f"sessions     idle deadline {settings.idle_timeout_s:g} s, "
        f"ring {settings.ring_seconds:g} s, endpointing {settings.stop_history_eou_ms} ms",
    ]


def _serve(args: argparse.Namespace, hooks: Hooks) -> int:
    settings = _settings(args)
    adapter, adapter_lines = _build_adapter(settings, hooks)
    for line in _banner(settings, adapter_lines):
        print(f"[verbatim] {line}", file=hooks.stdout, flush=True)

    async def main() -> None:
        loop = asyncio.get_running_loop()
        shutdown = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, shutdown.set)

        def ready(endpoints: Endpoints) -> None:
            print(f"[verbatim] riva grpc    {endpoints.grpc_target}", file=hooks.stdout)
            print(f"[verbatim] websocket    {endpoints.ws_endpoint}", file=hooks.stdout)
            print("[verbatim] ready", file=hooks.stdout, flush=True)
            if hooks.on_ready is not None:
                hooks.on_ready(endpoints, lambda: loop.call_soon_threadsafe(shutdown.set))

        await hooks.run_server(settings, adapter, shutdown=shutdown, on_ready=ready)
        print("[verbatim] stopped", file=hooks.stdout, flush=True)

    asyncio.run(main())
    return EXIT_OK


def _doctor(hooks: Hooks) -> int:
    report = hooks.inspect_runtime()
    for line in report.lines():
        print(f"[verbatim doctor] {line}", file=hooks.stdout)
    if not report.usable:
        print("[verbatim doctor] verdict: cannot serve a checkpoint", file=hooks.stdout, flush=True)
        return EXIT_RUNTIME
    if not report.graph_step:
        print(
            "[verbatim doctor] verdict: eager only; serve refuses the graph path without --eager",
            file=hooks.stdout,
            flush=True,
        )
        return EXIT_RUNTIME
    print("[verbatim doctor] verdict: graph path available", file=hooks.stdout, flush=True)
    return EXIT_OK


def main(argv: Sequence[str] | None = None, *, hooks: Hooks | None = None) -> int:
    """Parse and run. Returns the exit code rather than calling ``sys.exit``, so the
    console script and the tests share one path."""
    hooks = hooks if hooks is not None else Hooks()
    parser = _parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse's own usage errors and --help/--version
        return int(exc.code) if isinstance(exc.code, int) else EXIT_CONFIG
    if args.command is None:
        parser.print_help(file=hooks.stderr)
        return EXIT_CONFIG
    try:
        if args.command == "doctor":
            return _doctor(hooks)
        return _serve(args, hooks)
    except _Refused as exc:
        print(f"verbatim {args.command}: {exc}", file=hooks.stderr, flush=True)
        if args.traceback and exc.__cause__ is not None:
            traceback.print_exception(exc.__cause__, file=hooks.stderr)
        return exc.code
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        print(
            f"verbatim {args.command}: unexpected {type(exc).__name__}: {exc} "
            "(--traceback for the rest)",
            file=hooks.stderr,
            flush=True,
        )
        if args.traceback:
            traceback.print_exception(exc, file=hooks.stderr)
        return EXIT_UNEXPECTED


def console_main() -> None:
    """The console-script entry point."""
    sys.exit(main())


if __name__ == "__main__":  # python -m verbatim.cli
    console_main()
