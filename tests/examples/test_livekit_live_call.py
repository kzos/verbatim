# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""A live call from LiveKit's own NVIDIA plugin, unchanged, to a Verbatim server.

This is the proof the day-60 gate names, taken on the CPU fake: the plugin sends what it
sends, Verbatim answers what it answers, and the scripted transcript comes back through
the plugin's own events. Two processes, as in deployment, and of necessity: the plugin's
Riva client pins one protobuf runtime and Verbatim's vendored stubs need a newer one, so
they cannot share an interpreter. The server runs from the project's own environment,
named by ``VERBATIM_SERVE_PYTHON``; the plugin runs wherever pytest runs:

    VERBATIM_SERVE_PYTHON=.venv/bin/python uv run --no-project --python 3.12 \\
        --with livekit-plugins-nvidia --with livekit-agents \\
        --with pytest --with pytest-asyncio python -m pytest -q tests/examples

It is skipped where the plugin is not installed, so the default suite never depends on a
framework package.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        importlib.util.find_spec("livekit") is None
        or importlib.util.find_spec("livekit.plugins.nvidia") is None,
        reason="livekit-plugins-nvidia is not installed; see the module docstring",
    ),
]

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "livekit"))

MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@asynccontextmanager
async def _fake_server() -> AsyncIterator[str]:
    """`verbatim serve --pipeline fake` from the project's environment, on free ports."""
    python = os.environ.get("VERBATIM_SERVE_PYTHON", sys.executable)
    grpc_port, ws_port = _free_port(), _free_port()
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.Popen(
        [
            python,
            "-m",
            "verbatim",
            "serve",
            MODEL,
            "--chunk",
            "160ms",
            "--pipeline",
            "fake",
            "--bucket",
            "8",
            "--host",
            "127.0.0.1",
            "--grpc-port",
            str(grpc_port),
            "--ws-port",
            str(ws_port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30.0
        banner: list[str] = []
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            banner.append(line.rstrip())
            if "ready" in line:
                break
        if not any("ready" in line for line in banner):
            pytest.skip(
                "could not start `verbatim serve` from "
                f"VERBATIM_SERVE_PYTHON={python!r}: {' | '.join(banner[-3:])}"
            )
        yield f"127.0.0.1:{grpc_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()


async def test_the_livekit_plugin_completes_a_call_with_zero_plugin_code() -> None:
    """The fake serves one token per chunk, a stable hash of the chunk's own samples, and
    a final carrying them all: nine chunks of one signal are nine copies of one token."""
    from verbatim_livekit_stt import run_call

    chunk_bytes = 160 * 16000 // 1000 * 2
    pcm = b"\x00" * (chunk_bytes * 9)
    async with _fake_server() as target:
        result = await run_call(pcm, server=target, model=MODEL, real_time=False)
    assert result.error is None, result.error
    assert result.finals, "no final transcript reached the plugin"
    words = result.transcript.split()
    assert len(words) == 9
    assert len(set(words)) == 1
    assert re.fullmatch(r"w[0-9a-f]{8}", words[0])
    assert result.interim, "no interim transcript reached the plugin"
    assert len(result.interim[-1].split()) in (8, 9)


async def test_a_model_name_verbatim_does_not_serve_is_refused_not_substituted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NOT_FOUND on the wire: the plugin logs it from its recognition thread, closes the
    stream, and hands the caller no transcript at all rather than a substituted one."""
    from verbatim_livekit_stt import run_call

    pcm = b"\x00" * (160 * 16000 // 1000 * 2 * 2)
    with caplog.at_level(logging.ERROR, logger="livekit.plugins.nvidia.stt"):
        async with _fake_server() as target:
            result = await run_call(
                pcm, server=target, model="parakeet-not-served", real_time=False
            )
    assert result.finals == []
    assert result.interim == []
    logged = " ".join(
        f"{record.getMessage()} {record.exc_text or ''}"
        for record in caplog.records
        if record.name.startswith("livekit.plugins.nvidia")
    )
    assert "NOT_FOUND" in logged or "not found" in logged.lower(), logged[:400]
    assert "parakeet-not-served" in logged


def test_the_recipe_needs_no_plugin_code() -> None:
    """The recipe imports the plugin and configures it; it defines no subclass of it."""
    source = (REPO / "examples" / "livekit" / "verbatim_livekit_stt.py").read_text()
    assert "nvidia.STT(" in source
    assert "class " not in source.split("def run_call")[1].split("def _read_wav")[0]
