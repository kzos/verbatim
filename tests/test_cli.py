# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""`verbatim serve` and `verbatim doctor`, driven through `main` with the outside
world injected: the runtime probe, the NeMo build, and the server.

The fake pipeline serves end to end here, in a worker thread, with a real client on
the port it bound. The NeMo path is driven with a NeMo-shaped fake pipeline, so the
whole of `serve` runs except the model load, which is the one thing a CPU cannot do.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
from collections.abc import Callable
from typing import Any

import pytest
from websockets.asyncio.client import connect

from verbatim.cli import EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME, EXIT_UNEXPECTED, Hooks, main
from verbatim.config import ChunkMode
from verbatim.pipelines.cache_aware_rnnt import CacheAwareRNNTAdapter
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.pipelines.nemo_fake import FakeCacheAwarePipeline, boundary_for
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, PipelineBuildError, RuntimeReport
from verbatim.serve import Endpoints, ServeSettings

pytestmark = pytest.mark.cpu

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

PR_REPORT = RuntimeReport(
    nemo_version="3.1.0",
    torch_version="2.11.0",
    cuda_available=True,
    device_name="NVIDIA RTX A6000",
    inference_package=True,
    graph_step=True,
)
RELEASED_REPORT = RuntimeReport(
    nemo_version="3.0.0",
    torch_version="2.11.0",
    cuda_available=True,
    device_name="NVIDIA RTX A6000",
    inference_package=True,
    graph_step=False,
    notes=("the installed NeMo predates PR #15863: pass --eager",),
)
NO_NEMO_REPORT = RuntimeReport(
    nemo_version=None,
    torch_version=None,
    cuda_available=False,
    device_name=None,
    inference_package=False,
    graph_step=False,
    notes=("nemo: ImportError: No module named 'nemo'",),
)


class _Captured:
    """What the injected seams saw."""

    def __init__(self) -> None:
        self.settings: ServeSettings | None = None
        self.adapter: Any = None
        self.specs: list[NeMoPipelineSpec] = []
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    @property
    def out(self) -> str:
        return self.stdout.getvalue()

    @property
    def err(self) -> str:
        return self.stderr.getvalue()


def _hooks(
    captured: _Captured,
    *,
    report: RuntimeReport = RELEASED_REPORT,
    build: Callable[[NeMoPipelineSpec], Any] | None = None,
    run_server: Callable[..., Any] | None = None,
    on_ready: Callable[[Endpoints, Callable[[], None]], None] | None = None,
) -> Hooks:
    async def no_server(
        settings: ServeSettings, adapter: Any, *, shutdown: Any, on_ready: Any
    ) -> None:
        captured.settings = settings
        captured.adapter = adapter

    def default_build(spec: NeMoPipelineSpec) -> Any:
        captured.specs.append(spec)
        return boundary_for(
            FakeCacheAwarePipeline(spec.chunk.ms, num_slots=max(64, spec.num_slots))
        )

    def recording_build(spec: NeMoPipelineSpec) -> Any:
        captured.specs.append(spec)
        assert build is not None
        return build(spec)

    return Hooks(
        inspect_runtime=lambda: report,
        build_boundary=recording_build if build is not None else default_build,
        run_server=run_server if run_server is not None else no_server,
        on_ready=on_ready,
        stdout=captured.stdout,
        stderr=captured.stderr,
    )


FAKE = ["serve", MODEL, "--chunk", "160ms", "--bucket", "8", "--pipeline", "fake"]
NEMO = ["serve", MODEL, "--chunk", "160ms", "--bucket", "8"]


def test_no_command_is_usage_and_exit_2() -> None:
    captured = _Captured()
    assert main([], hooks=_hooks(captured)) == EXIT_CONFIG
    assert "serve" in captured.err and "doctor" in captured.err


@pytest.mark.parametrize("chunk", ["160ms", "160", "160MS"])
def test_chunk_accepts_a_suffix_or_a_bare_number(chunk: str) -> None:
    captured = _Captured()
    argv = ["serve", MODEL, "--chunk", chunk, "--bucket", "8", "--pipeline", "fake"]
    assert main(argv, hooks=_hooks(captured)) == EXIT_OK
    assert captured.settings is not None
    assert captured.settings.chunk == ChunkMode(160)


def test_an_unserved_chunk_lists_the_valid_modes(capsys: pytest.CaptureFixture[str]) -> None:
    captured = _Captured()
    argv = ["serve", MODEL, "--chunk", "100ms", "--bucket", "8"]
    assert main(argv, hooks=_hooks(captured)) == EXIT_CONFIG
    assert "80ms, 160ms, 560ms, 1120ms" in capsys.readouterr().err


def test_a_ceiling_or_a_bucket_is_required_and_they_exclude_each_other(
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = _Captured()
    assert main(["serve", MODEL, "--chunk", "160ms"], hooks=_hooks(captured)) == EXIT_CONFIG
    assert "--ceiling" in capsys.readouterr().err
    argv = ["serve", MODEL, "--chunk", "160ms", "--ceiling", "32", "--bucket", "8"]
    assert main(argv, hooks=_hooks(captured)) == EXIT_CONFIG
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3", "x"])
def test_a_bucket_must_be_a_positive_integer(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = _Captured()
    argv = ["serve", MODEL, "--chunk", "160ms", "--bucket", value]
    assert main(argv, hooks=_hooks(captured)) == EXIT_CONFIG
    assert "positive integer" in capsys.readouterr().err


def test_a_zero_idle_timeout_is_refused_with_the_reason() -> None:
    captured = _Captured()
    assert main([*FAKE, "--idle-timeout", "0"], hooks=_hooks(captured)) == EXIT_CONFIG
    assert "deadline" in captured.err


def test_the_same_port_for_both_wires_is_refused() -> None:
    captured = _Captured()
    argv = [*FAKE, "--ws-port", "9000", "--grpc-port", "9000"]
    assert main(argv, hooks=_hooks(captured)) == EXIT_CONFIG
    assert "both 9000" in captured.err


def test_the_fake_pipeline_is_served_loudly_and_never_probes_the_runtime() -> None:
    captured = _Captured()
    hooks = _hooks(captured, report=NO_NEMO_REPORT)
    assert main(FAKE, hooks=hooks) == EXIT_OK
    assert isinstance(captured.adapter, FakePipelineAdapter)
    assert "FAKE" in captured.out
    assert "UNCALIBRATED" in captured.out
    assert captured.specs == []


def test_the_banner_says_calibrated_for_a_ceiling() -> None:
    captured = _Captured()
    argv = ["serve", MODEL, "--chunk", "160ms", "--ceiling", "24", "--pipeline", "fake"]
    assert main(argv, hooks=_hooks(captured)) == EXIT_OK
    assert "ceiling 24 streams, calibrated" in captured.out
    assert captured.settings is not None
    assert captured.settings.ceiling == 24


def test_serve_refuses_the_graph_path_on_a_released_track_without_eager() -> None:
    captured = _Captured()
    assert main(NEMO, hooks=_hooks(captured, report=RELEASED_REPORT)) == EXIT_RUNTIME
    assert "PR #15863" in captured.err
    assert "--eager" in captured.err
    assert captured.specs == [], "no model may be loaded before the refusal"
    assert captured.settings is None


def test_serve_refuses_without_a_usable_runtime() -> None:
    captured = _Captured()
    assert main([*NEMO, "--eager"], hooks=_hooks(captured, report=NO_NEMO_REPORT)) == EXIT_RUNTIME
    assert "verbatim[nemo]" in captured.err
    assert "--pipeline fake" in captured.err
    assert captured.specs == []


def test_eager_serves_a_nemo_shaped_pipeline_through_the_real_adapter() -> None:
    captured = _Captured()
    assert main([*NEMO, "--eager"], hooks=_hooks(captured, report=RELEASED_REPORT)) == EXIT_OK
    assert isinstance(captured.adapter, CacheAwareRNNTAdapter)
    spec = captured.specs[0]
    assert spec.use_cuda_graphs is False
    assert spec.att_context == (56, 1)
    assert spec.batch_size == 8
    assert spec.num_slots == 8 + 8 + 7 + 8
    assert spec.stop_history_eou_ms == 800
    assert "EAGER" in captured.out
    assert "att_context_size [56, 1]" in captured.out


def test_the_pr_track_takes_the_graph_path_without_being_asked() -> None:
    captured = _Captured()
    assert main(NEMO, hooks=_hooks(captured, report=PR_REPORT)) == EXIT_OK
    assert captured.specs[0].use_cuda_graphs is True
    assert "graph path" in captured.out


def test_eager_on_the_pr_track_is_honoured_and_said() -> None:
    captured = _Captured()
    assert main([*NEMO, "--eager"], hooks=_hooks(captured, report=PR_REPORT)) == EXIT_OK
    assert captured.specs[0].use_cuda_graphs is False
    assert "EAGER" in captured.out


def test_a_pipeline_that_refuses_to_build_is_one_message_and_exit_2() -> None:
    captured = _Captured()

    def refuse(spec: NeMoPipelineSpec) -> Any:
        raise PipelineBuildError(
            "NeMo could not build the pipeline for checkpoint 'x', chunk 160 ms "
            "(att_context_size [56, 1]): RuntimeError: Failed to load model"
        ) from RuntimeError("Failed to load model")

    assert main([*NEMO, "--eager"], hooks=_hooks(captured, build=refuse)) == EXIT_CONFIG
    assert "verbatim serve: NeMo could not build the pipeline" in captured.err
    assert "Failed to load model" in captured.err
    assert "Traceback" not in captured.err
    assert captured.settings is None

    with_trace = _Captured()
    argv = [*NEMO, "--eager", "--traceback"]
    assert main(argv, hooks=_hooks(with_trace, build=refuse)) == EXIT_CONFIG
    assert "Traceback" in with_trace.err
    assert "PipelineBuildError" in with_trace.err


def test_a_built_pipeline_that_does_not_fit_the_config_is_refused_by_name() -> None:
    captured = _Captured()

    def wrong_chunk(spec: NeMoPipelineSpec) -> Any:
        return boundary_for(FakeCacheAwarePipeline(560, num_slots=64))

    assert main([*NEMO, "--eager"], hooks=_hooks(captured, build=wrong_chunk)) == EXIT_CONFIG
    assert "does not fit the scheduler configuration" in captured.err
    assert "does not match" in captured.err


def test_an_unknown_checkpoint_needs_an_explicit_left_context() -> None:
    captured = _Captured()
    argv = ["serve", "someone/other-ckpt", "--chunk", "560ms", "--bucket", "4", "--eager"]
    assert main(argv, hooks=_hooks(captured)) == EXIT_CONFIG
    assert "--att-context-left" in captured.err
    assert captured.specs == []

    explicit = _Captured()
    assert main([*argv, "--att-context-left", "64"], hooks=_hooks(explicit)) == EXIT_OK
    assert explicit.specs[0].att_context == (64, 6)


def test_doctor_verdicts() -> None:
    pr = _Captured()
    assert main(["doctor"], hooks=_hooks(pr, report=PR_REPORT)) == EXIT_OK
    assert "graph path available" in pr.out
    released = _Captured()
    assert main(["doctor"], hooks=_hooks(released, report=RELEASED_REPORT)) == EXIT_RUNTIME
    assert "eager only" in released.out
    assert "ABSENT" in released.out
    none = _Captured()
    assert main(["doctor"], hooks=_hooks(none, report=NO_NEMO_REPORT)) == EXIT_RUNTIME
    assert "cannot serve" in none.out
    assert "not installed" in none.out


def test_an_unexpected_failure_is_exit_1_with_a_hint() -> None:
    captured = _Captured()

    async def explode(settings: Any, adapter: Any, *, shutdown: Any, on_ready: Any) -> None:
        raise ValueError("boom")

    assert main(FAKE, hooks=_hooks(captured, run_server=explode)) == EXIT_UNEXPECTED
    assert "unexpected ValueError: boom" in captured.err
    assert "--traceback" in captured.err
    assert "Traceback" not in captured.err


def test_serve_runs_the_fake_pipeline_end_to_end_until_told_to_stop() -> None:
    """The real `run_server`, in a worker thread, on ephemeral ports: a client gets a
    session frame, the shutdown request is honoured, and the exit is clean."""
    captured = _Captured()
    ready = threading.Event()
    endpoints: list[Endpoints] = []
    stoppers: list[Callable[[], None]] = []

    def on_ready(found: Endpoints, request_shutdown: Callable[[], None]) -> None:
        endpoints.append(found)
        stoppers.append(request_shutdown)
        ready.set()

    hooks = Hooks(
        inspect_runtime=lambda: NO_NEMO_REPORT,
        on_ready=on_ready,
        stdout=captured.stdout,
        stderr=captured.stderr,
    )
    argv = [*FAKE, "--host", "127.0.0.1", "--ws-port", "0", "--grpc-port", "0"]
    outcome: dict[str, int] = {}
    worker = threading.Thread(target=lambda: outcome.update(rc=main(argv, hooks=hooks)))
    worker.start()
    try:
        assert ready.wait(10.0), "the server never reported ready"
        found = endpoints[0]
        assert found.ws_port != 0 and found.grpc_port != 0
        assert found.ws_endpoint.endswith("/v1/stream")

        async def one_session() -> dict:
            async with connect(found.ws_endpoint) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                assert isinstance(raw, str)
                await ws.send(b"\x00" * 5120)
                partial = await asyncio.wait_for(ws.recv(), timeout=5.0)
                assert isinstance(partial, str)
                return {"session": json.loads(raw), "partial": json.loads(partial)}

        frames = asyncio.run(one_session())
        assert frames["session"]["type"] == "session"
        assert frames["session"]["chunk_ms"] == 160
        assert frames["partial"]["type"] == "partial"
    finally:
        for stop in stoppers:
            stop()
        worker.join(10.0)
    assert not worker.is_alive(), "serve did not stop when asked"
    assert outcome["rc"] == EXIT_OK
    assert "[verbatim] ready" in captured.out
    assert "[verbatim] stopped" in captured.out
    assert f"riva grpc    127.0.0.1:{found.grpc_port}" in captured.out
