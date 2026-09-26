# SPDX-License-Identifier: Apache-2.0
"""CPU checks for scripts/step1_gate_runbook.sh. No GPU: the server tests use `--pipeline fake`,
nvidia-smi is a stub first on PATH, and the HF cache is a temporary directory.

Run explicitly (it is outside the suite's testpaths on purpose), with this checkout's code
first on the path:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:bench/src "$RUNBOOK_VENV/bin/python" \\
        -m pytest scripts/test_step1_gate_runbook.py

Environment: RUNBOOK_VENV names the Python environment the runbook runs (default: the one
running pytest). RUNBOOK_CORPUS_ROOT names the local corpus root; the one test that reads
the real 256-clip manifest skips without it. Every runbook these tests start gets its own
values of the five variables a run needs.

What each part proves:

* The five variables the header says a production run needs are the ones it refuses to
  run without, each of them unset or empty, and --help prints that header with none set.
* The dry run's command lines parse against the real `verbatim serve` and
  `verbatim-bench invariance` parsers, every `--flag` is an exact option string (argparse
  would otherwise accept an abbreviation), the bucket and the max level are the operator's
  parameters and nothing else, and the rest are the published churn record's values.
* The server's own code, run on the CPU with the model build stubbed, builds the NeMo spec
  the arm asks for, and the runbook's banner check passes exactly the banner that code
  prints for the arm and fails every altered one.
* The facts check (contracts C5 and C7) refuses every /readyz and /admission that is not
  the arm, what the built model reports under "observed" and where the server imported its
  code from ("code") included, and the expectation the runbook itself runs it with
  (--facts-expect) is the arm's: a NeMo-shaped /readyz of the arm passes it and one with
  any other observation, or another checkout's code, does not.
* The embedded record, finals, WER, smoke, monitor and summary helpers reproduce the
  published records' own readings, each of their checks fails on the input it exists for,
  and each exit code the summary can return is reached by the case that should reach it.
  The monitor's two timeouts and its sleep are observed doing what they were given: a
  hanging nvidia-smi, a slow /admission and the spacing of its polls.
* Test mode runs the card, placement, checkpoint, code and git checks for real, against
  stubs, and each refusal exits 2. The stub nvidia-smi answers a per-card query for card 3
  alone, so a query of any other card fails where it is made.
* Rehearsals over the fake run the whole mechanism end to end, and the failures before,
  between, during and after arms (a server whose built encoder, banner or derived spec is
  not the arm, a server importing another checkout's code with this repository's
  PYTHONPATH, a later arm dying, the code changing, a new untracked file under bench/src/,
  a co-tenant at a single poll, an nvidia-smi that answers nothing while the gate runs, a
  monitor killed partway through the gate or still polling after it, a server gone, an
  nvidia-smi or /admission that does not answer, or a degraded server, each seen only by
  the reading after the gate, a gate exit 1 or 2 its record does not state, a level-1 WER
  below, at and above the bound, a timeout) end the way the header says. The rehearsal
  corpus sits under RUNBOOK_CORPUS_ROOT as the real one does, so its local id and its
  published-form id differ, and the record must carry the local one.

Every temporary file and directory is made under one directory, removed when the tests end,
by SIGTERM too: a runbook still running is sent SIGTERM first, so its own cleanup stops the
server it started.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # nothing in this file may reach a GPU

import atexit
import contextlib
import hashlib
import http.server
import itertools
import json
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from verbatim_bench import cli as bench_cli
from verbatim_bench import invariance
from verbatim_bench.canonical import FinalRecord

from verbatim import cli as serve_cli
from verbatim.audio.pcm import decode_pcm16
from verbatim.config import ChunkMode, EngineConfig
from verbatim.core.types import PcmFrame
from verbatim.pipelines.fake import FakePipelineAdapter
from verbatim.pipelines.nemo_runtime import att_context_size
from verbatim.serve import engine_config

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
SCRIPT = HERE / "step1_gate_runbook.sh"
ROWS = WORKTREE / "rows" / "exploratory"
CHURN_RECORD = ROWS / "invariance-churn-b300-2026-09-14.json"
#: The environment the runbook runs: RUNBOOK_VENV, else the one running these tests.
VENV = Path(os.environ.get("RUNBOOK_VENV") or Path(sys.executable).parent.parent)
VENV_BIN = VENV / "bin"
#: Where the runbook reads the corpus by default: the published runs' location.
PUBLISHED_ROOT = "/opt/verbatim/corpus"
MANIFEST_REL = "librispeech-test-other-256/librispeech-test-other-256.jsonl"
DEFAULT_MANIFEST = Path(PUBLISHED_ROOT) / MANIFEST_REL
LOCAL_CORPUS_ROOT = os.environ.get("RUNBOOK_CORPUS_ROOT")
ARMS = ("fixed-churn", "ragged-churn", "fixed-const", "ragged-const")
MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
NEMO_FILE = "nemotron-speech-streaming-en-0.6b.nemo"
CARD_UUID = "GPU-b43f9262-f250-444a-bfbf-461dd3500f1e"
PUBLISHED_CORPUS_ID = "sha256:6a142a960379d48ed31d5ec8c3bc07dbdfcd34ef2b6db041a888b07bdc0003be"
#: The server's own tick budget at 160 ms, from its own config code.
BUDGET_MS = EngineConfig(chunk=ChunkMode(160), buckets=(128,)).budget_ms
PYTHONPATH = f"{WORKTREE}/src:{WORKTREE}/bench/src"

#: Every temporary file and directory these tests make is under this one, which is removed
#: when the process that imported this module exits.
_TMP = Path(tempfile.mkdtemp(prefix="runbook-test-"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


@contextlib.contextmanager
def _scratch_directory(name: str) -> Iterator[Path]:
    """A directory under _TMP, removed with everything in it when the block ends."""
    path = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=_TMP))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Each test's own directory, removed when the test ends. It replaces pytest's
    tmp_path, which keeps the directories of the last runs."""
    with _scratch_directory(re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)[:60]) as path:
        yield path


needs_venv = pytest.mark.skipif(
    not (VENV_BIN / "verbatim").is_file() or not (VENV_BIN / "verbatim-bench").is_file(),
    reason=f"no verbatim console scripts under {VENV_BIN}; set RUNBOOK_VENV",
)


def _shim(name: str, body: str, filename: str = "nvidia-smi") -> Path:
    directory = _TMP / name
    if not directory.exists():
        directory.mkdir(parents=True)
        shim = directory / filename
        shim.write_text(body)
        shim.chmod(0o755)
    return directory


def _no_gpu_shim() -> Path:
    """An `nvidia-smi` that always fails, first on PATH for production-mode runs here. If a
    refusal under test ever stopped refusing, the runbook would die at its first card query
    instead of reaching a card."""
    return _shim("fail", "#!/bin/sh\necho 'nvidia-smi is disabled in this test' >&2\nexit 1\n")


#: Stands in for nvidia-smi in test mode. STUB_SMI_MODE picks what card 3 holds:
#:   idle          the server, once it is running (the normal case)
#:   busy          a foreign process, always
#:   foreign       the server and a foreign process, once the server is running
#:   absent        nothing, even with the server running
#:   late          the server, plus a foreign process once a gate has started
#:   transient     the server, plus a foreign process while a gate runs and no record exists
#:   blip          the server, plus a foreign process at exactly one query while a gate runs
#:                 and no record exists (the monitor's first poll)
#:   blind         the server; while a gate runs and no record exists, the compute-apps
#:                 query fails (exit 9), as a hung or broken driver would
#:   killmon       the server; while a gate runs and no record exists, the second query the
#:                 runbook's monitor makes SIGKILLs the monitor (the process asking)
#:   blind_after   the server; once a record exists, every query not from the monitor (the
#:                 runbook's own reading after the gate) fails (exit 9)
#:   vanish        the server only until a gate has started
#:   swap          the server until a gate has started, then a foreign process instead
#:   warn          idle; once a gate has started, NeMo's look-ahead warning is appended to
#:                 every server.log (as if the model build logged it mid-run)
#:   edit          idle; once a gate has started, a line is appended to STUB_SMI_EDIT
#:   edit_between  idle; once an arm has finished, a line is appended to STUB_SMI_EDIT
#:   commit        idle; once a gate has started, an empty commit lands in STUB_SMI_COMMIT
#: The server is found by the server.pid files the runbook writes under STUB_SMI_ROOT.
#: Only card 3 exists: a compute-apps or uuid query without `-i 3` fails (exit 6, as
#: nvidia-smi does for a card it cannot find), so a query of any other card is seen. When
#: STUB_SMI_LOG is set, each call appends "monitor ARGS" or "runbook ARGS" to it, by
#: whether the process asking is the runbook's monitor.
STUB_SMI = r"""#!/usr/bin/env bash
# runbook-test-stub: never calls the real nvidia-smi
mode="${STUB_SMI_MODE:-idle}"
uuid="${STUB_SMI_UUID:-GPU-b43f9262-f250-444a-bfbf-461dd3500f1e}"
root="${STUB_SMI_ROOT:-/nonexistent}"
foreign="999999, python3, 1024"
index="" previous=""
for arg in "$@"; do
    [[ "$previous" == -i ]] && index="$arg"
    previous="$arg"
done
who=runbook
tr '\0' '\n' < "/proc/$PPID/cmdline" 2>/dev/null | grep -qx monitor && who=monitor
[[ -n "${STUB_SMI_LOG:-}" ]] && echo "$who $*" >> "$STUB_SMI_LOG"
case " $* " in
    *" --query-compute-apps="*|*" --query-gpu=uuid "*)
        if [[ "$index" != 3 ]]; then
            echo "No devices were found (runbook-test-stub: -i '$index'; only card 3 exists)" >&2
            exit 6
        fi ;;
esac
server=""
while IFS= read -r f; do
    p=$(cat "$f" 2>/dev/null) || continue
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then server="$p"; fi
done < <(find "$root" -name server.pid 2>/dev/null)
gate=0; [[ -n "$(find "$root" -name gate.log 2>/dev/null)" ]] && gate=1
record=0; [[ -n "$(find "$root" -name 'invariance-*.json' 2>/dev/null)" ]] && record=1
finished=0; [[ -n "$(find "$root" -name arm.json 2>/dev/null)" ]] && finished=1
once() { [[ -e "$root/.stub-$1" ]] && return 1; : > "$root/.stub-$1"; }
if (( gate )); then
    case "$mode" in
        warn) warning="[NeMo W conformer_encoder] att_context_size=[70, 1] is not among"
              warning+=" the list of the supported look-aheads: [<placeholder>]"
              once warn && for f in $(find "$root" -name server.log); do
                  echo "$warning" >> "$f"
              done ;;
        edit) once edit && echo "# changed while a gate ran" >> "$STUB_SMI_EDIT" ;;
        commit) once commit && git -C "$STUB_SMI_COMMIT" -c user.name=stub \
                    -c user.email=stub@example.invalid -c commit.gpgsign=false \
                    commit -q --allow-empty -m "moved while a gate ran" ;;
    esac
fi
if (( finished )) && [[ "$mode" == edit_between ]]; then
    once edit && echo "# changed between arms" >> "$STUB_SMI_EDIT"
fi
case " $* " in
    *" --query-compute-apps="*)
        case "$mode" in
            busy) echo "$foreign"; [[ -n "$server" ]] && echo "$server, verbatim, 2048" ;;
            foreign) [[ -n "$server" ]] && { echo "$server, verbatim, 2048"; echo "$foreign"; } ;;
            absent) ;;
            late) [[ -n "$server" ]] && echo "$server, verbatim, 2048"
                  (( gate )) && echo "$foreign" ;;
            transient) [[ -n "$server" ]] && echo "$server, verbatim, 2048"
                       (( gate && ! record )) && echo "$foreign" ;;
            blip) [[ -n "$server" ]] && echo "$server, verbatim, 2048"
                  (( gate && ! record )) && once blip && echo "$foreign" ;;
            blind) if (( gate && ! record )); then
                       echo "NVIDIA-SMI has failed (runbook-test-stub: blind during the gate)" >&2
                       exit 9
                   fi
                   [[ -n "$server" ]] && echo "$server, verbatim, 2048" ;;
            blind_after) asker="/proc/$PPID/cmdline"
                         if (( record )) && ! tr '\0' '\n' < "$asker" | grep -qx monitor; then
                             echo "NVIDIA-SMI has failed (runbook-test-stub: blind after)" >&2
                             exit 9
                         fi
                         [[ -n "$server" ]] && echo "$server, verbatim, 2048" ;;
            killmon) asker="/proc/$PPID/cmdline"
                     if (( gate && ! record )) && tr '\0' '\n' < "$asker" | grep -qx monitor; then
                         polls=$(( $(cat "$root/.stub-killmon" 2>/dev/null || echo 0) + 1 ))
                         echo "$polls" > "$root/.stub-killmon"
                         (( polls == 2 )) && kill -KILL "$PPID"
                     fi
                     [[ -n "$server" ]] && echo "$server, verbatim, 2048" ;;
            vanish) [[ -n "$server" ]] && (( ! gate )) && echo "$server, verbatim, 2048" ;;
            swap) [[ -n "$server" ]] && (( ! gate )) && echo "$server, verbatim, 2048"
                  (( gate )) && echo "$foreign" ;;
            *) [[ -n "$server" ]] && echo "$server, verbatim, 2048" ;;
        esac ;;
    *" --query-gpu=uuid "*) echo "$uuid" ;;
    *" --query-gpu=index,uuid"*) echo "3, $uuid, stub-name, stub-bus, stub-driver" ;;
    *) echo "stub nvidia-smi: unexpected arguments: $*" >&2; exit 1 ;;
esac
exit 0
"""

#: Stands in for `ss` in one test: once a server has started under STUB_SMI_ROOT, the
#: listener on the port belongs to someone else; before that, the real `ss` answers.
STUB_SS = r"""#!/usr/bin/env bash
# runbook-test-stub for ss
root="${STUB_SMI_ROOT:-/nonexistent}"
if [[ -n "$(find "$root" -name server.pid 2>/dev/null)" ]]; then
    echo 'LISTEN 0 128 127.0.0.1:0 0.0.0.0:* users:(("squatter",pid=1,fd=3))'
    exit 0
fi
exec "$STUB_SS_REAL" "$@"
"""


def _stub_smi() -> Path:
    return _shim("stub", STUB_SMI)


#: The five variables the runbook's header says a production run needs, in its order.
REQUIRED_ENV = (
    "RUNBOOK_VENV",
    "RUNBOOK_CORPUS_ROOT",
    "RUNBOOK_MEASUREMENT_DIR",
    "RUNBOOK_BUCKET",
    "RUNBOOK_MAX_LEVEL",
)


def _clean_env(
    *,
    bucket: str | None = "128",
    max_level: str | None = "42",
    smi: Path | None = None,
    unset: tuple[str, ...] = (),
    **extra: str,
) -> dict[str, str]:
    """The five required variables set (the corpus root to the published runs' location, the
    measurement directory under _TMP), minus those named in `unset`, then `extra`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RUNBOOK_", "STUB_SMI_"))}
    env.pop("HF_HUB_CACHE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PATH"] = f"{smi or _no_gpu_shim()}{os.pathsep}{env.get('PATH', '')}"
    env["RUNBOOK_VENV"] = str(VENV)
    env["RUNBOOK_CORPUS_ROOT"] = PUBLISHED_ROOT
    env["RUNBOOK_MEASUREMENT_DIR"] = str(_TMP / "measurement")
    # Not the PYTHONPATH the runbook must set for itself (same code, other order), so a
    # runbook that inherited the caller's instead is seen.
    env["PYTHONPATH"] = f"{WORKTREE}/bench/src:{WORKTREE}/src"
    if bucket is not None:
        env["RUNBOOK_BUCKET"] = bucket
    if max_level is not None:
        env["RUNBOOK_MAX_LEVEL"] = max_level
    for name in unset:
        env.pop(name, None)
    env.update(extra)
    return env


#: The runbooks these tests have started and not yet seen exit.
_RUNNING: set[subprocess.Popen] = set()
#: How long a runbook sent SIGTERM is given to stop what it started: its stop_server waits
#: up to 60 s after SIGTERM and 30 s after SIGKILL.
_RUNBOOK_STOP_GRACE_S = 100.0


def _reaped(proc: subprocess.Popen) -> bool:
    """Whether the runbook has exited, without Popen.wait: this also runs in a signal
    handler, which may interrupt a Popen.wait holding its lock."""
    if proc.returncode is not None:
        return True
    try:
        pid, status = os.waitpid(proc.pid, os.WNOHANG)
    except ChildProcessError:  # reaped already, by the Popen.wait this interrupted
        return True
    if pid == 0:
        return False
    proc.returncode = os.waitstatus_to_exitcode(status)
    return True


def _stop_runbooks(procs: list[subprocess.Popen], grace: float = _RUNBOOK_STOP_GRACE_S) -> None:
    """SIGTERM each runbook, so its own cleanup trap stops the gate, monitor and server it
    started; SIGKILL the whole session of any still running after `grace` seconds."""
    for proc in procs:
        with contextlib.suppress(ProcessLookupError):
            os.kill(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    pending = [proc for proc in procs if not _reaped(proc)]
    while pending and time.monotonic() < deadline:
        time.sleep(0.1)
        pending = [proc for proc in pending if not _reaped(proc)]
    for proc in pending:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while not _reaped(proc) and time.monotonic() < deadline:
            time.sleep(0.05)


def _on_sigterm(signum: int, frame) -> None:
    """SIGTERM ends these tests as it would without this handler (the same signal, so the
    same exit status), but first stops every runbook still running and removes _TMP: an
    atexit handler does not run when a signal ends the process."""
    _stop_runbooks(list(_RUNNING))
    shutil.rmtree(_TMP, ignore_errors=True)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


if (
    threading.current_thread() is threading.main_thread()
    and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
):
    signal.signal(signal.SIGTERM, _on_sigterm)


def _run(
    *args: str, env: dict[str, str], timeout: float = 60.0, script: Path = SCRIPT
) -> subprocess.CompletedProcess:
    """The runbook, in a session of its own, its output in unlinked files (not pipes: a
    runbook still cleaning up must never block on output nobody reads). Whatever ends the
    wait early (the timeout, an interrupt) stops it with _stop_runbooks before passing on."""
    with (
        tempfile.TemporaryFile(dir=_TMP) as out,
        tempfile.TemporaryFile(dir=_TMP) as err,
    ):
        proc = subprocess.Popen(
            ["bash", str(script), *args],
            env=env,
            stdout=out,
            stderr=err,
            cwd=str(WORKTREE),
            start_new_session=True,
        )
        _RUNNING.add(proc)
        try:
            returncode = proc.wait(timeout=timeout)
        except BaseException:
            _stop_runbooks([proc])
            raise
        finally:
            _RUNNING.discard(proc)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(
            proc.args,
            returncode,
            out.read().decode("utf-8", errors="replace"),
            err.read().decode("utf-8", errors="replace"),
        )


def _options(parser, name: str) -> set[str]:
    sub = next(a for a in parser._subparsers._group_actions).choices[name]
    return set(sub._option_string_actions)


def _helper_source() -> str:
    text = SCRIPT.read_text()
    start = text.index("read -r -d '' HELPER <<'PYEOF'")
    body = text[text.index("\n", start) + 1 :]
    return body[: body.index("\nPYEOF\n")]


def _helper(
    *args: str, timeout: float = 60.0, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _helper_source(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write(doc: dict) -> str:
    path = Path(tempfile.mkstemp(dir=_TMP, suffix=".json")[1])
    path.write_text(json.dumps(doc))
    return str(path)


# --- these tests clean up after themselves -------------------------------------------


def test_a_tests_directory_is_removed_when_it_ends() -> None:
    with _scratch_directory("self-check") as path:
        (path / "a" / "b").mkdir(parents=True)
        (path / "a" / "b" / "file").write_text("x")
    assert not path.exists()


def test_a_run_of_these_tests_leaves_no_temporary_directory_behind(tmp_path: Path) -> None:
    """A child pytest over two tests that write under _TMP and under tmp_path, with TMPDIR
    an empty directory of its own: the directory is empty again when the child exits."""
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).resolve()),
            "-k",
            "test_an_empty_log_fails_the_banner_check_for_both_arms or "
            "test_the_monitor_summary_counts_a_truncated_line_instead_of_failing",
        ],
        env={**os.environ, "TMPDIR": str(tmpdir), "PYTHONPATH": PYTHONPATH},
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(WORKTREE),
    )
    assert done.returncode == 0 and "2 passed" in done.stdout, done.stdout[-2000:] + done.stderr
    assert list(tmpdir.iterdir()) == []


def _gone(pid: int, within: float = 10.0) -> bool:
    """Whether `pid` has exited (a zombie counts) within `within` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


#: Stands in for the runbook: a child it must stop, stopped by its TERM trap, as the
#: runbook's cleanup stops its server.
TRAPS_TERM = """
echo $$ > "$1/runbook.pid"
sleep 300 &
echo $! > "$1/child.pid"
trap 'kill $!; wait; echo stopped > "$1/stopped"; exit 143' TERM
wait
"""

#: ... and one that ignores SIGTERM, its child too (an ignored signal stays ignored).
IGNORES_TERM = """
trap '' TERM
echo $$ > "$1/runbook.pid"
sleep 300 &
echo $! > "$1/child.pid"
wait
"""


def _kill_own(root: Path) -> None:
    """SIGKILL what a stand-in runbook under `root` started and left running: its shell
    and its child, each only while its command line is still the one this test started
    (so a pid reused since is never touched)."""
    for name, mark in (("runbook.pid", str(root).encode()), ("child.pid", b"sleep\x00300")):
        with contextlib.suppress(OSError, ValueError):
            pid = int((root / name).read_text())
            if mark in Path(f"/proc/{pid}/cmdline").read_bytes():
                os.kill(pid, signal.SIGKILL)


def _child_pid(root: Path) -> int:
    deadline = time.monotonic() + 10
    while not (root / "child.pid").is_file() or not (root / "child.pid").read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return int((root / "child.pid").read_text())


def test_a_runbook_the_test_stops_waiting_for_is_sent_sigterm_first(tmp_path: Path) -> None:
    """When the wait ends early (here the timeout), the runbook gets SIGTERM, so its own
    trap stops what it started, before the exception passes on."""
    script = tmp_path / "runbook.sh"
    script.write_text(TRAPS_TERM)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run(str(tmp_path), env=dict(os.environ), timeout=2, script=script)
        assert (tmp_path / "stopped").read_text() == "stopped\n"
        assert _gone(_child_pid(tmp_path)) and not _RUNNING
    finally:  # whatever the outcome, nothing this test started is left running
        _kill_own(tmp_path)


def test_a_runbook_that_ignores_sigterm_is_killed_with_its_session(tmp_path: Path) -> None:
    script = tmp_path / "runbook.sh"
    script.write_text(IGNORES_TERM)
    proc = subprocess.Popen(["bash", str(script), str(tmp_path)], start_new_session=True)
    try:
        child = _child_pid(tmp_path)
        _stop_runbooks([proc], grace=1)
        assert proc.returncode == -signal.SIGKILL and _gone(child)
    finally:
        _kill_own(tmp_path)
        proc.wait()


def _processes_with(marker: bytes) -> list[int]:
    """Every process, this one aside, whose environment holds `marker`."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        with contextlib.suppress(OSError):
            if marker in (entry / "environ").read_bytes():
                found.append(int(entry.name))
    return found


def _stop_leftovers(marker: bytes) -> None:
    """This test's own leftovers only: every process carries its child pytest's TMPDIR."""
    for pid in _processes_with(marker):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while _processes_with(marker) and time.monotonic() < deadline:
        time.sleep(0.2)
    for pid in _processes_with(marker):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@needs_venv
def test_sigterm_mid_gate_stops_the_runbook_and_leaves_nothing_behind(tmp_path: Path) -> None:
    """A child pytest, TMPDIR an empty directory of its own, is sent SIGTERM while one of
    its rehearsals is inside the gate (server, monitor and gate all running). It exits
    by that signal; no process it started is left, the server included; TMPDIR is empty."""
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    marker = f"TMPDIR={tmpdir}\0".encode()
    log = tmp_path / "child.log"
    child: subprocess.Popen | None = None
    try:
        with log.open("wb") as out:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"{Path(__file__).resolve()}::test_a_server_emitting_wrong_text_gets_no_verdict",
                ],
                env={**os.environ, "TMPDIR": str(tmpdir), "PYTHONPATH": PYTHONPATH},
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=str(WORKTREE),
            )
        # _TMP (runbook-test-*), then the test's own directory, then its run directory
        arm = "*/out/gate-nemotron-card3-rehearsal/fixed-const"
        deadline = time.monotonic() + 150
        while not list(tmpdir.glob(f"runbook-test-*/{arm}/gate.log")):
            assert child.poll() is None and time.monotonic() < deadline, log.read_text()[-3000:]
            time.sleep(0.1)
        server = int(next(tmpdir.glob(f"runbook-test-*/{arm}/server.pid")).read_text())
        started = _processes_with(marker)
        assert server in started and child.pid in started, (server, started)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=_RUNBOOK_STOP_GRACE_S + 60) == -signal.SIGTERM
        assert _processes_with(marker) == [], log.read_text()[-3000:]
        with pytest.raises(ProcessLookupError):
            os.kill(server, 0)
        assert list(tmpdir.iterdir()) == []
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        _stop_leftovers(marker)


# --- the command lines -------------------------------------------------------------


def _dry_run(bucket: str = "128", max_level: str = "42") -> list[dict]:
    done = _run("--dry-run", env=_clean_env(bucket=bucket, max_level=max_level))
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in done.stdout.splitlines() if line.strip()]


def test_the_dry_run_names_all_four_arms_in_order() -> None:
    assert [line["arm"] for line in _dry_run()] == list(ARMS)


def test_every_serve_flag_exists_exactly_and_the_settings_are_the_arm_asked_for() -> None:
    shared = json.loads(CHURN_RECORD.read_text())["shared"]
    options = _options(serve_cli._parser(), "serve")
    for line in _dry_run():
        argv = line["serve"]
        assert argv[0] == str(VENV_BIN / "verbatim")
        flags = [token for token in argv[1:] if token.startswith("--")]
        assert flags and all(flag in options for flag in flags), set(flags) - options
        args = serve_cli._parser().parse_args(argv[1:])
        assert args.command == "serve"
        assert args.model == MODEL
        assert args.chunk.ms == shared["chunk_ms"] == 160
        assert args.bucket == 128 and args.ceiling is None  # the parameter passed above
        assert args.compute_dtype == shared["compute_dtype"] == "bfloat16"
        assert args.eager is True and shared["execution"] == "eager"
        assert args.padding == line["arm"].split("-")[0]
        assert args.pipeline == "cache_aware_rnnt"
        assert args.biasing is False and args.decoder_graphs is False
        assert args.word_confidence == "off"
        assert args.att_context_left == 70
        assert att_context_size(args.model, args.chunk, left=args.att_context_left) == [70, 1]
        assert (args.host, args.ws_port, args.grpc_port, args.device_id) == (
            "127.0.0.1",
            8765,
            0,
            0,
        )
        settings = serve_cli._settings(args)  # the server's own validation
        config = engine_config(settings)
        assert config.padding == args.padding and config.buckets == (128,)
        assert "CUDA_VISIBLE_DEVICES=3" in line["env"]
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in line["env"]
        assert "HF_HUB_OFFLINE=1" in line["env"]
        assert any(entry.startswith("HF_HUB_CACHE=/") for entry in line["env"])
        # The server runs this checkout's code, whatever the environment has installed.
        assert f"PYTHONPATH={PYTHONPATH}" in line["env"]


def test_the_bucket_and_the_max_level_are_the_operators_parameters() -> None:
    """Nothing in the runbook pins them: whatever the step profile names is what runs."""
    for bucket, max_level in (("64", "40"), ("128", "128"), ("48", "48")):
        for line in _dry_run(bucket, max_level):
            serve = serve_cli._parser().parse_args(line["serve"][1:])
            gate = bench_cli._build_parser().parse_args(line["gate"][1:])
            assert serve.bucket == int(bucket)
            assert gate.max == int(max_level)


def test_every_gate_flag_exists_exactly_and_the_levels_are_the_published_churn() -> None:
    shared = json.loads(CHURN_RECORD.read_text())["shared"]
    options = _options(bench_cli._build_parser(), "invariance")
    for line in _dry_run():
        argv = line["gate"]
        assert argv[0] == str(VENV_BIN / "verbatim-bench")
        flags = [token for token in argv[1:] if token.startswith("--")]
        assert flags and all(flag in options for flag in flags), set(flags) - options
        args = bench_cli._build_parser().parse_args(argv[1:])
        assert args.command == "invariance"
        assert args.endpoint == "ws://127.0.0.1:8765/v1/stream"
        assert args.manifest == DEFAULT_MANIFEST
        assert args.synthetic is None and args.phrases is None
        assert args.max == 42
        assert args.seed == invariance.DEFAULT_SEED
        assert args.out == Path(line["record"])
        assert args.finals_out == Path(line["finals"]) and args.finals_out != args.out
        churned = line["arm"].endswith("-churn")
        assert args.churn_period_s == (shared["churn_period_s"] if churned else None)
        levels = invariance.default_levels(args.max, churn_period_s=args.churn_period_s)
        assert [(lv.slot, lv.concurrency) for lv in levels] == [
            ("1", 1),
            ("32a", 32),
            ("32b", 32),
            ("max", 42),
        ]
        assert levels[-1].churn_period_s == (20.0 if churned else None)


def test_the_locations_come_from_the_environment(tmp_path: Path) -> None:
    root = tmp_path / "corpus-root"
    done = _run(
        "--dry-run",
        "fixed-churn",
        env=_clean_env(
            RUNBOOK_CORPUS_ROOT=str(root) + "/",
            RUNBOOK_MEASUREMENT_DIR=str(tmp_path / "measure"),
        ),
    )
    assert done.returncode == 0, done.stderr
    line = json.loads(done.stdout)
    gate = bench_cli._build_parser().parse_args(line["gate"][1:])
    assert gate.manifest == root / MANIFEST_REL
    assert line["record"].startswith(str(tmp_path / "measure") + "/gate-nemotron-card3-")


@pytest.mark.parametrize(
    ("bucket", "max_level", "message"),
    [
        (None, "42", "set RUNBOOK_BUCKET: a run needs all of"),
        ("128", None, "set RUNBOOK_MAX_LEVEL: a run needs all of"),
        ("12x", "42", "positive integer"),
        ("128", "0", "positive integer"),
        ("31", "31", "below 32"),
        ("64", "65", "exceeds RUNBOOK_BUCKET"),
        ("512", "257", "exceeds the corpus"),
    ],
)
def test_production_refuses_a_missing_or_impossible_bucket_or_max_level(
    bucket: str | None, max_level: str | None, message: str
) -> None:
    done = _run("--dry-run", env=_clean_env(bucket=bucket, max_level=max_level))
    assert done.returncode == 2 and message in done.stderr, done.stderr
    assert not done.stdout


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
@pytest.mark.parametrize("name", REQUIRED_ENV)
def test_every_variable_a_production_run_needs_is_refused_when_unset(
    name: str, value: str | None
) -> None:
    """Each of the five, unset or empty, stops the runbook before it does anything: a run of
    an arm, a dry run, and the modes that print what a run would use."""
    env = _clean_env(unset=(name,)) if value is None else _clean_env(**{name: value})
    for args in (("fixed-churn",), ("--dry-run",), ("--facts-expect",), ("--derive-spec", "fixed")):
        done = _run(*args, env=env)
        assert done.returncode == 2, (args, done.stdout, done.stderr)
        assert f"FATAL: set {name}: a run needs all of" in done.stderr, (args, done.stderr)
        assert not done.stdout, (args, done.stdout)


def test_the_header_states_the_environment_a_production_run_needs() -> None:
    """The header's first block, which --help prints with no environment at all, names
    exactly the five variables the runbook refuses to run without, in its order, with the
    step-1 run's bucket and max level; with all five unset the refusal names the same five."""
    done = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0 and not done.stderr, done.stderr
    block = done.stdout.split("# THE ENVIRONMENT A PRODUCTION RUN NEEDS.", 1)[1].split("\n#\n", 1)[
        0
    ]
    assert tuple(re.findall(r"\b(RUNBOOK_[A-Z_]+)=", block)) == REQUIRED_ENV, block
    assert "RUNBOOK_BUCKET=128 RUNBOOK_MAX_LEVEL=128" in block, block
    refused = _run("--dry-run", env=_clean_env(unset=REQUIRED_ENV))
    assert refused.returncode == 2 and f"FATAL: set {' '.join(REQUIRED_ENV)}: " in refused.stderr, (
        refused.stderr
    )


def test_the_old_parameter_name_is_refused_rather_than_ignored() -> None:
    done = _run("--dry-run", env=_clean_env(RUNBOOK_MAX="42"))
    assert done.returncode == 2 and "RUNBOOK_MAX_LEVEL" in done.stderr


def test_a_missing_python_environment_is_refused(tmp_path: Path) -> None:
    done = _run("--dry-run", env=_clean_env(RUNBOOK_VENV=str(tmp_path / "no-venv")))
    assert done.returncode == 2 and "no Python at" in done.stderr, done.stderr


@needs_venv
def test_the_console_scripts_are_the_entry_points_the_comments_cite() -> None:
    assert "from verbatim.cli import console_main" in (VENV_BIN / "verbatim").read_text()
    assert "from verbatim_bench.cli import main" in (VENV_BIN / "verbatim-bench").read_text()


def test_the_runbook_runs_only_from_the_repository_it_tests(tmp_path: Path) -> None:
    """A copy outside any git repository, and a copy inside one but not at scripts/, are
    both refused: the repository the runbook is in is the code it tests."""
    loose = tmp_path / "loose" / SCRIPT.name
    loose.parent.mkdir()
    shutil.copy2(SCRIPT, loose)
    done = _run("--dry-run", env=_clean_env(), script=loose)
    assert done.returncode == 2 and "not inside a git repository" in done.stderr, done.stderr
    elsewhere = _git_repo(tmp_path / "repo") / "tools" / SCRIPT.name
    elsewhere.parent.mkdir()
    shutil.copy2(SCRIPT, elsewhere)
    done = _run("--dry-run", env=_clean_env(), script=elsewhere)
    assert done.returncode == 2 and "is not scripts/" in done.stderr, done.stderr


def test_no_machine_paths_in_the_public_files() -> None:
    """The runbook and its tests go into the public repository: locations come from the
    environment, never from this machine."""
    patterns = ("/" + "elm/", "/tmp/" + "claude", "/" + "home/")
    for path in (
        SCRIPT,
        Path(__file__),
        WORKTREE / "bench" / "src" / "verbatim_bench" / "cli.py",
        WORKTREE / "bench" / "src" / "verbatim_bench" / "invariance.py",
        WORKTREE / "tests" / "test_invariance_finals_out.py",
    ):
        text = path.read_text()
        found = [p for p in patterns if p in text]
        assert not found, (path, found)


# --- what the server's own code builds, and the banner check ----------------------


def _derive(padding: str, bucket: str = "128") -> dict:
    done = _run("--derive-spec", padding, env=_clean_env(bucket=bucket))
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("padding", ["fixed", "ragged"])
def test_the_servers_own_code_builds_the_arm_asked_for(padding: str) -> None:
    doc = _derive(padding, bucket="96")
    spec = doc["spec"]
    assert spec["model"] == MODEL
    assert spec["att_context"] == [70, 1]
    assert spec["compute_dtype"] == "bfloat16"
    assert (
        spec["use_cuda_graphs"] is False
    )  # the runtime said the graph step exists: --eager did this
    assert spec["use_cuda_graph_decoder"] is False
    assert spec["enable_per_stream_biasing"] is False
    assert spec["word_confidence"] == "off"
    assert spec["batch_size"] == 96
    # What the server hands NeMo's pipeline config: the spec's own default, since serve
    # passes none. Recorded in every arm and in summary.json.
    assert spec["matmul_precision"] == "high"
    assert doc["execution"] == "eager"
    assert doc["budget_ms"] == BUDGET_MS
    assert doc["derived_from"][doc["derived_from"].index("--padding") + 1] == padding
    ok = _helper("spec_check", _write(doc), _spec_expect(96))
    assert ok.returncode == 0, ok.stdout


def test_without_eager_the_servers_code_takes_the_graph_path_and_the_check_refuses_it() -> None:
    """The derivation reports the graph step present, as the installed NeMo does, so it is
    --eager and nothing else that keeps the step eager: drop it and the spec says graphs."""
    argv = [token for token in _dry_run()[0]["serve"][1:] if token != "--eager"]
    done = _helper("derive_spec", *argv)
    assert done.returncode == 0, done.stdout + done.stderr
    doc = json.loads(done.stdout)
    assert doc["spec"]["use_cuda_graphs"] is True and doc["execution"] == "graph path"
    refused = _helper("spec_check", _write(doc), _spec_expect(128))
    assert refused.returncode == 1
    assert "use_cuda_graphs" in refused.stdout and "execution" in refused.stdout


def _spec_expect(bucket: int) -> str:
    return json.dumps(
        {
            "model": MODEL,
            "att_context": [70, 1],
            "compute_dtype": "bfloat16",
            "use_cuda_graphs": False,
            "use_cuda_graph_decoder": False,
            "enable_per_stream_biasing": False,
            "batch_size": bucket,
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("att_context", [70, 13]),
        ("use_cuda_graphs", True),
        ("use_cuda_graph_decoder", True),
        ("enable_per_stream_biasing", True),
        ("compute_dtype", "float32"),
        ("batch_size", 127),
        ("matmul_precision", None),
        ("word_confidence", "paper-best"),
    ],
)
def test_the_spec_check_refuses_a_spec_that_is_not_the_arm(field: str, value) -> None:
    doc = _derive("fixed")
    doc["spec"][field] = value
    done = _helper("spec_check", _write(doc), _spec_expect(128))
    assert done.returncode == 1 and field in done.stdout, done.stdout


def test_the_spec_check_refuses_a_graph_path_and_a_missing_budget() -> None:
    for key, value in (("execution", "graph path"), ("budget_ms", None), ("spec", None)):
        doc = _derive("fixed")
        doc[key] = value
        assert _helper("spec_check", _write(doc), _spec_expect(128)).returncode == 1, key


def _banner_file(padding: str, *, edit=None, extra: str = "") -> str:
    lines = _derive(padding)["banner"]
    text = "\n".join(lines) + "\n"
    if edit is not None:
        old, new = edit
        assert old in text
        text = text.replace(old, new)
    path = Path(tempfile.mkstemp(dir=_TMP, suffix=".log")[1])
    path.write_text(text + extra)
    return str(path)


def _check_banner(padding: str, log: str) -> subprocess.CompletedProcess:
    return _run("--check-banner", padding, log, env=_clean_env())


@pytest.mark.parametrize("padding", ["fixed", "ragged"])
def test_the_banner_the_server_prints_for_the_arm_passes_its_check(padding: str) -> None:
    done = _check_banner(padding, _banner_file(padding))
    assert done.returncode == 0, done.stdout + done.stderr


#: The shape of NeMo's warning (conformer_encoder.py:1000-1004); the list is not the model's.
LOOKAHEAD_WARNING = (
    "[NeMo W conformer_encoder] att_context_size=[70, 1] is not among the list of the "
    "supported look-aheads: [<placeholder>]\n"
)


@pytest.mark.parametrize(
    ("padding", "log_padding", "edit", "extra", "message"),
    [
        ("fixed", "ragged", None, "", "says RAGGED"),
        ("ragged", "fixed", None, "", "does not say RAGGED"),
        ("fixed", "fixed", None, LOOKAHEAD_WARNING, "not one this checkpoint supports"),
        ("fixed", "fixed", ("[70, 1]", "[70, 13]"), "", "att_context_size [70, 1]"),
        ("fixed", "fixed", ("EAGER encoder step", "graph path"), "", "not eager"),
        ("fixed", "fixed", ("bucket 128 streams", "bucket 64 streams"), "", "admission"),
        ("fixed", "fixed", ("NeMo slots: bucket 128", "NeMo slots: bucket 64"), "", "NeMo slots"),
        ("fixed", "fixed", ("precision    bfloat16", "precision    float16"), "", "precision"),
        ("fixed", "fixed", (MODEL, "nvidia/other"), "", "checkpoint line"),
        (
            "fixed",
            "fixed",
            None,
            "[verbatim] decoder      CUDA graphs ON for the RNNT decoder\n",
            "decoder",
        ),
        ("fixed", "fixed", None, "[verbatim] biasing      ON: sessions may carry\n", "biasing"),
    ],
)
def test_the_banner_check_fails_every_banner_that_is_not_the_arm(
    padding: str, log_padding: str, edit, extra: str, message: str
) -> None:
    done = _check_banner(padding, _banner_file(log_padding, edit=edit, extra=extra))
    assert done.returncode == 2 and message in done.stdout, done.stdout + done.stderr


def test_an_empty_log_fails_the_banner_check_for_both_arms() -> None:
    empty = Path(tempfile.mkstemp(dir=_TMP, suffix=".log")[1])
    for padding in ("fixed", "ragged"):
        assert _check_banner(padding, str(empty)).returncode == 2


# --- the facts check: /readyz and /admission, contract C5 --------------------------


#: Where a server running this checkout's code says it imported it from (contract C7).
SERVER_CODE = {
    "verbatim_path": str((WORKTREE / "src" / "verbatim").resolve()),
    "bench_path": str((WORKTREE / "bench" / "src" / "verbatim_bench").resolve()),
}


def _readyz(**over) -> dict:
    """/readyz as a NeMo server of the fixed arm answers it: the settings keys, then
    word_confidence, what the built model and decoder are, and where its code comes from."""
    doc = {
        "ready": True,
        "reason": None,
        "model": MODEL,
        "pipeline": "cache_aware_rnnt",
        "chunk_ms": 160,
        "precision": "bfloat16",
        "execution": "eager",
        "biasing": False,
        "word_confidence": "off",
        "observed": {
            "att_context_size": [70, 1],
            "decoder_step_confidence": False,
            "decoder_graphs": False,
        },
        "code": dict(SERVER_CODE),
    }
    doc.update(over)
    return doc


def _facts_expect(pipeline: str = "cache_aware_rnnt") -> str:
    fake = pipeline == "fake"
    return json.dumps(
        {
            "model": MODEL,
            "pipeline": pipeline,
            "chunk_ms": 160,
            "precision": "none" if fake else "bfloat16",
            "execution": "fake" if fake else "eager",
            "biasing": False,
            "word_confidence": "off",
            "bucket": 128,
            "observed": {
                "att_context_size": [70, 1],
                "decoder_step_confidence": False,
                "decoder_graphs": False,
            },
            "code": SERVER_CODE,
        }
    )


def _facts(readyz: dict, admission: dict | None = None, *, pipeline: str = "cache_aware_rnnt"):
    return _helper(
        "facts", _write(readyz), _write(admission or {"bucket": 128}), _facts_expect(pipeline)
    )


def _observed(**over) -> dict:
    return {**_readyz()["observed"], **over}


def test_the_facts_check_passes_the_server_the_arm_asked_for() -> None:
    assert _facts(_readyz()).returncode == 0
    # The fake has no encoder and no decoder: it reports null, and only it may.
    fake = _readyz(
        pipeline="fake",
        precision="none",
        execution="fake",
        observed=dict.fromkeys(("att_context_size", "decoder_step_confidence", "decoder_graphs")),
    )
    assert _facts(fake, pipeline="fake").returncode == 0
    # verbatim_bench need not be importable in the server: null there is not a mismatch.
    assert _facts(_readyz(code={**SERVER_CODE, "bench_path": None})).returncode == 0


@pytest.mark.parametrize(
    ("readyz", "admission", "message"),
    [
        (_readyz(ready=False), None, "ready=False"),
        (_readyz(model="nvidia/other"), None, "/readyz model="),
        (_readyz(pipeline="cache_aware_ctc"), None, "/readyz pipeline="),
        (_readyz(chunk_ms=80), None, "/readyz chunk_ms="),
        (_readyz(precision="float32"), None, "/readyz precision="),
        (_readyz(execution="graph path"), None, "/readyz execution="),
        (_readyz(biasing=True), None, "/readyz biasing="),
        (_readyz(word_confidence="paper-best"), None, "/readyz word_confidence="),
        (_readyz(word_confidence=None), None, "/readyz word_confidence="),
        ({k: v for k, v in _readyz().items() if k != "observed"}, None, "observed=None"),
        (_readyz(observed=_observed(att_context_size=[70, 13])), None, "att_context_size=[70, 13]"),
        (_readyz(observed=_observed(att_context_size=None)), None, "not observed"),
        (_readyz(observed=_observed(decoder_graphs=True)), None, "decoder_graphs=True"),
        (_readyz(observed=_observed(decoder_graphs=None)), None, "decoder_graphs=None"),
        (
            _readyz(observed=_observed(decoder_step_confidence=True)),
            None,
            "decoder_step_confidence=True",
        ),
        (
            _readyz(observed=_observed(decoder_step_confidence=None)),
            None,
            "decoder_step_confidence=None",
        ),
        (_readyz(), {"bucket": 64}, "/admission bucket=64"),
        # Contract C7: where the server imported its code from.
        ({k: v for k, v in _readyz().items() if k != "code"}, None, "/readyz code=None"),
        (_readyz(code=None), None, "/readyz code=None"),
        (
            _readyz(code={**SERVER_CODE, "verbatim_path": "/other/checkout/src/verbatim"}),
            None,
            "/readyz code.verbatim_path='/other/checkout/src/verbatim', expected "
            f"{SERVER_CODE['verbatim_path']!r}: the server does not run this repository's code",
        ),
        (
            _readyz(code={**SERVER_CODE, "verbatim_path": None}),
            None,
            "/readyz code.verbatim_path=None",
        ),
        (
            _readyz(code={**SERVER_CODE, "bench_path": "/other/checkout/bench/src/verbatim_bench"}),
            None,
            "/readyz code.bench_path='/other/checkout/bench/src/verbatim_bench', expected "
            f"{SERVER_CODE['bench_path']!r} or null",
        ),
    ],
)
def test_the_facts_check_refuses_every_server_that_is_not_the_arm(
    readyz: dict, admission: dict | None, message: str
) -> None:
    done = _facts(readyz, admission)
    assert done.returncode == 1 and message in done.stdout, done.stdout + done.stderr


def test_the_fake_may_report_null_but_not_a_wrong_observation() -> None:
    base = {"pipeline": "fake", "precision": "none", "execution": "fake"}
    for observed in (
        _observed(att_context_size=[70, 13]),
        _observed(decoder_graphs=True),
        _observed(decoder_step_confidence=True),
    ):
        done = _facts(_readyz(observed=observed, **base), pipeline="fake")
        assert done.returncode == 1 and "observed." in done.stdout, done.stdout
    # A NeMo server passing itself off as the fake's null is still refused in production.
    nulls = dict.fromkeys(("att_context_size", "decoder_step_confidence", "decoder_graphs"))
    assert _facts(_readyz(observed=nulls)).returncode == 1
    # The fake runs the same server code: its "code" is checked like any server's.
    done = _facts(_readyz(code=None, **base), pipeline="fake")
    assert done.returncode == 1 and "/readyz code=None" in done.stdout, done.stdout


def _runbook_facts_expect(bucket: str = "128") -> dict:
    """The expectation require_facts checks every production server against, printed by
    the runbook itself (--facts-expect), not written here."""
    done = _run("--facts-expect", env=_clean_env(bucket=bucket, max_level=bucket))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_runbooks_own_facts_expectation_is_contract_c5_for_this_arm() -> None:
    """What the runbook requires of /readyz is an eager, bare arm at 160 ms: the attention
    context the server's own code derives for it, no step confidence, no decoder graphs,
    word confidence off. The helper tests above use the same expectation, and a NeMo
    server built as the arm asks passes it; one built any other way does not."""
    doc = _runbook_facts_expect()
    assert doc["observed"] == {
        "att_context_size": att_context_size(MODEL, ChunkMode(160), left=70),
        "decoder_step_confidence": False,
        "decoder_graphs": False,
    }
    assert doc["word_confidence"] == "off"
    # Contract C7: this checkout's own packages, resolved.
    assert doc["code"] == SERVER_CODE
    assert doc == json.loads(_facts_expect())
    admission = _write({"bucket": 128})
    done = _helper("facts", _write(_readyz()), admission, json.dumps(doc))
    assert done.returncode == 0, done.stdout
    for readyz, message in (
        (_readyz(observed=_observed(att_context_size=[70, 13])), "att_context_size=[70, 13]"),
        (_readyz(observed=_observed(decoder_graphs=True)), "decoder_graphs=True"),
        (_readyz(observed=_observed(decoder_step_confidence=True)), "decoder_step_confidence=True"),
        (_readyz(word_confidence="paper-best"), "word_confidence='paper-best'"),
        (_readyz(code={**SERVER_CODE, "verbatim_path": "/x/src/verbatim"}), "code.verbatim_path"),
    ):
        done = _helper("facts", _write(readyz), admission, json.dumps(doc))
        assert done.returncode == 1 and message in done.stdout, done.stdout


def test_the_nemo_shaped_readyz_is_what_the_servers_reader_reports_for_this_arm() -> None:
    """The NeMo-shaped /readyz above, under "observed" and word_confidence, is what the
    server's own reader (src/verbatim/pipelines/observed.py) reports for a pipeline holding
    what this arm asks NeMo to build: the encoder's context [70, 1], an RNNT decoding
    computer with no step confidence and no CUDA-graphs mode, a greedy decoding
    configuration without frame confidence. These are attribute stand-ins, not NeMo:
    whether the installed NeMo builds exactly this is read off the GPU run's /readyz."""
    from types import SimpleNamespace as NS

    from verbatim.pipelines.observed import configured_word_confidence, observe

    pipeline = NS(
        asr_model=NS(
            asr_model=NS(encoder=NS(att_context_size=[70, 1])),
            decoding_cfg=NS(greedy=NS(preserve_frame_confidence=False)),
        ),
        decoding_computer=NS(preserve_step_confidence=False, cuda_graphs_mode=None),
    )
    # The pipeline stand-in has no decoding object, so its word flag reads as None; the
    # readyz fixture predates that fact.
    assert observe(pipeline).to_json_dict() == {
        **_readyz()["observed"],
        "decoder_word_confidence": None,
    }
    assert configured_word_confidence(NS(_boundary=NS(pipeline=pipeline))) == "off"
    assert _readyz()["word_confidence"] == "off"


# --- where the code comes from ---------------------------------------------------------


def _fake_package(root: Path) -> Path:
    """A directory holding a package called verbatim that is not this checkout's."""
    (root / "verbatim").mkdir(parents=True)
    (root / "verbatim" / "__init__.py").write_text("")
    return root


def test_code_origin_requires_both_packages_from_the_repository(tmp_path: Path) -> None:
    ok = _helper("code_origin", str(WORKTREE), env={**os.environ, "PYTHONPATH": PYTHONPATH})
    assert ok.returncode == 0, ok.stdout + ok.stderr
    doc = json.loads(ok.stdout)
    assert doc["verbatim_from_repo"] and doc["verbatim_bench_from_repo"]
    assert doc["verbatim"] == f"{WORKTREE}/src/verbatim/__init__.py"
    elsewhere = _fake_package(tmp_path / "elsewhere")
    bad = _helper(
        "code_origin",
        str(WORKTREE),
        env={**os.environ, "PYTHONPATH": f"{elsewhere}:{PYTHONPATH}"},
    )
    assert bad.returncode == 1 and json.loads(bad.stdout)["verbatim_from_repo"] is False
    # And a repository that is not the one the packages come from.
    assert (
        _helper(
            "code_origin", str(tmp_path), env={**os.environ, "PYTHONPATH": PYTHONPATH}
        ).returncode
        == 1
    )


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _wrapped_venv(
    root: Path,
    *,
    python_path_prefix: str = "",
    server_path_suffix: str = "",
    bench_exit: int | None = None,
    server_prelude: str = "",
    server_stdout_discarded: bool = False,
    helper_prelude: dict[str, str] | None = None,
) -> Path:
    """A stand-in for RUNBOOK_VENV; everything not named is the real environment's.

    python_path_prefix       its python puts this first on PYTHONPATH
    server_path_suffix       its `verbatim` appends this to PYTHONPATH
    bench_exit               its `verbatim-bench` runs the real one, then exits this
    server_prelude           its `verbatim` runs this Python in the server process first
    server_stdout_discarded  its `verbatim` sends the server's stdout, the banner, nowhere
    helper_prelude           {helper command: Python its python runs before the runbook's
                             embedded helper, for that command only}
    Every wrapper execs, so the server keeps the pid the runbook started."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    # A script, not a symlink: Python finds its environment next to the path it was run by.
    prefix = f'PYTHONPATH="{python_path_prefix}:$PYTHONPATH" ' if python_path_prefix else ""
    intercepts = ""
    for command, code in (helper_prelude or {}).items():
        (bin_dir / f"prelude-{command}.py").write_text(code)
        intercepts += (
            f'if [ "$1" = -c ] && [ "$3" = {command} ]; then\n'
            f'    code="$2"; shift 2\n'
            f'    {prefix}exec "{sys.executable}" -c "$(cat "{bin_dir}/prelude-{command}.py")\n'
            f'$code" "$@"\n'
            "fi\n"
        )
    _executable(
        bin_dir / "python", f'#!/bin/sh\n{intercepts}{prefix}exec "{sys.executable}" "$@"\n'
    )
    if server_prelude:
        entry = bin_dir / "verbatim-entry.py"
        entry.write_text(
            f"{server_prelude}\nimport sys\nfrom verbatim.cli import console_main\n"
            "sys.exit(console_main())\n"
        )
        _executable(
            bin_dir / "verbatim", f'#!/bin/sh\nexec "{VENV_BIN / "python"}" "{entry}" "$@"\n'
        )
    elif server_stdout_discarded:
        _executable(
            bin_dir / "verbatim", f'#!/bin/sh\nexec "{VENV_BIN / "verbatim"}" "$@" > /dev/null\n'
        )
    elif server_path_suffix:
        _executable(
            bin_dir / "verbatim",
            f'#!/bin/sh\nPYTHONPATH="$PYTHONPATH:{server_path_suffix}" '
            f'exec "{VENV_BIN / "verbatim"}" "$@"\n',
        )
    else:
        (bin_dir / "verbatim").symlink_to(VENV_BIN / "verbatim")
    if bench_exit is not None:
        (bin_dir / "verbatim-bench").write_text(
            f'#!/bin/sh\n"{VENV_BIN / "verbatim-bench"}" "$@"\nexit {bench_exit}\n'
        )
        (bin_dir / "verbatim-bench").chmod(0o755)
    else:
        (bin_dir / "verbatim-bench").symlink_to(VENV_BIN / "verbatim-bench")
    return root


@needs_venv
def test_preflight_refuses_code_imported_from_elsewhere(tmp_path: Path) -> None:
    elsewhere = _fake_package(tmp_path / "elsewhere")
    env = _rehearsal_env(tmp_path, _free_port(), 4)
    env["RUNBOOK_VENV"] = str(_wrapped_venv(tmp_path / "venv", python_path_prefix=str(elsewhere)))
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2 and "do not import from" in done.stderr, done.stderr
    assert not (_run_dir(tmp_path) / "fixed-churn").exists()


# --- the embedded helpers against the published records ----------------------------


@pytest.mark.skipif(
    not LOCAL_CORPUS_ROOT or not (Path(LOCAL_CORPUS_ROOT) / MANIFEST_REL).is_file(),
    reason="set RUNBOOK_CORPUS_ROOT to the local corpus root to check the real manifest",
)
def test_the_local_corpus_is_the_published_one_with_its_prefix_moved() -> None:
    root = str(LOCAL_CORPUS_ROOT).rstrip("/")
    done = _helper("corpus", f"{root}/{MANIFEST_REL}", root + "/", PUBLISHED_ROOT + "/")
    doc = json.loads(done.stdout)
    published = json.loads(CHURN_RECORD.read_text())["shared"]["corpus_id"]
    assert doc["published_form_id"] == published == PUBLISHED_CORPUS_ID
    assert doc["utterances"] == doc["local_prefix_occurrences"] == 256
    path = _write(doc)
    assert _helper("corpus_check", path, PUBLISHED_CORPUS_ID, "256").returncode == 0
    assert _helper("corpus_check", path, "sha256:0", "256").returncode == 1
    assert _helper("corpus_check", path, PUBLISHED_CORPUS_ID, "255").returncode == 1


def _corpus_doc(**over) -> dict:
    doc = {
        "local_id": "sha256:local",
        "published_form_id": PUBLISHED_CORPUS_ID,
        "utterances": 256,
        "local_prefix_occurrences": 256,
        "every_audio_path_under_local_prefix": True,
    }
    doc.update(over)
    return doc


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"published_form_id": "sha256:0"}, "not the published"),
        ({"utterances": 255}, "255 utterances"),
        ({"local_prefix_occurrences": 257}, "prefix seen 257 times"),
        ({"every_audio_path_under_local_prefix": False}, "outside the local prefix"),
    ],
)
def test_the_corpus_check_refuses_a_manifest_that_is_not_the_published_one(
    over: dict, message: str
) -> None:
    assert (
        _helper("corpus_check", _write(_corpus_doc()), PUBLISHED_CORPUS_ID, "256").returncode == 0
    )
    done = _helper("corpus_check", _write(_corpus_doc(**over)), PUBLISHED_CORPUS_ID, "256")
    assert done.returncode == 1 and message in done.stdout, done.stdout


def _expect(**overrides) -> str:
    base = {
        "max": 42,
        "churn_period_s": 20.0,
        "chunk_ms": 160,
        "utterances": 256,
        "corpus_id": PUBLISHED_CORPUS_ID,
        "gate_exit": 0,
    }
    base.update(overrides)
    return json.dumps(base)


DECGRAPH = ROWS / "invariance-decgraph-b300-2026-09-16.json"


def test_a_published_churned_record_passes_the_record_check() -> None:
    done = _helper("record", str(DECGRAPH), _expect())
    assert done.returncode == 0, done.stdout
    doc = json.loads(done.stdout)
    assert doc["verdict"] == "invariant" and doc["distinct_digests"] == 1
    assert doc["max_level_admission_occupancy_range"][1] > 1
    assert doc["gate_exit_checked"] == 0


def test_the_record_check_fails_when_the_record_is_not_the_arm_asked_for(tmp_path: Path) -> None:
    assert _helper("record", str(DECGRAPH), _expect(churn_period_s=None)).returncode == 1
    assert _helper("record", str(DECGRAPH), _expect(max=38)).returncode == 1
    assert _helper("record", str(DECGRAPH), _expect(gate_exit=1)).returncode == 1
    missing = _helper("record", str(tmp_path / "absent.json"), _expect())
    assert missing.returncode == 1 and json.loads(missing.stdout)["record_present"] is False


def _set(path: str, value):
    def edit(rec: dict) -> None:
        *keys, last = path.split(".")
        target = rec
        for key in keys:
            target = target[int(key)] if isinstance(target, list) else target[key]
        if isinstance(target, list):
            target[int(last)] = value
        else:
            target[last] = value

    return edit


def _drop(path: str):
    def edit(rec: dict) -> None:
        *keys, last = path.split(".")
        target = rec
        for key in keys:
            target = target[int(key)] if isinstance(target, list) else target[key]
        del target[last]

    return edit


def _constant_with_histogram(rec: dict) -> None:
    rec["levels"][-1]["churn_period_s"] = None


def _swap_levels(rec: dict) -> None:
    rec["levels"][1], rec["levels"][2] = rec["levels"][2], rec["levels"][1]


def _divergent_without_divergences(rec: dict) -> None:
    rec["verdict"] = "divergent"
    rec["divergences"] = []


def _divergent_without_a_digest(rec: dict) -> None:
    rec["verdict"] = "divergent"
    rec["divergences"] = [{"stream_id": "s0", "level": "max", "against": "1", "kind": "text"}]
    rec["levels"][0]["digest"] = None


@pytest.mark.parametrize(
    ("edit", "expect", "message"),
    [
        (_set("record", "vb-other/1"), {}, "record kind"),
        (_swap_levels, {}, "slots"),
        (_set("levels.1.concurrency", 31), {}, "concurrency"),
        (_set("levels.0.churn_period_s", 20.0), {}, "level 1 is churned"),
        (_set("levels.2.churn_period_s", 20.0), {}, "level 32b is churned"),
        # Churned, but at another period than the arm's.
        (_set("levels.3.churn_period_s", 5.0), {}, "max level churn_period_s=5.0, expected 20.0"),
        (_set("levels.3.admission_occupancy_histogram", {"1": 99}), {}, "never admitted more"),
        (_drop("levels.3.admission_occupancy_histogram"), {}, "recorded no admission occupancy"),
        (_constant_with_histogram, {"churn_period_s": None}, "carries an admission histogram"),
        (_set("chunk_ms", 80), {}, "chunk_ms"),
        (_set("corpus.id", "sha256:other"), {}, "corpus id"),
        (_set("corpus.utterances", 255), {}, "corpus utterances"),
        (_set("biasing", {"phrases": 1}), {}, "carries biasing"),
        (_set("controls", {"ran": True}), {}, "carries biasing"),
        (_set("levels.1.digest", "sha256:another"), {}, "invariant with more than one digest"),
        (_divergent_without_divergences, {}, "divergent with no divergence listed"),
        # bench gives a verdict only over four digested levels (invariance.py assess).
        (_set("levels.3.digest", None), {}, "verdict invariant with no digest at level(s) ['max']"),
        (_divergent_without_a_digest, {}, "verdict divergent with no digest at level(s) ['1']"),
        (_set("exit_code", 1), {}, "the gate exited 0 and its record says 1"),
    ],
)
def test_every_record_check_fails_on_the_record_it_exists_for(
    tmp_path: Path, edit, expect: dict, message: str
) -> None:
    rec = json.loads(DECGRAPH.read_text())
    edit(rec)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(rec))
    done = _helper("record", str(path), _expect(**expect))
    assert done.returncode == 1 and message in done.stdout, done.stdout


def test_a_record_that_compared_the_text_alone_is_refused(tmp_path: Path) -> None:
    rec = json.loads(DECGRAPH.read_text())
    for value in (False, None):
        rec["timings_present"] = value
        path = tmp_path / "text-only.json"
        path.write_text(json.dumps(rec))
        done = _helper("record", str(path), _expect())
        assert done.returncode == 1 and "timings_present" in done.stdout


def test_per_comparison_counts_reproduce_the_published_ragged_reading(tmp_path: Path) -> None:
    """The churn record reads ragged at constant occupancy as 118 / 118 / 66 streams
    differing from concurrency 1, and DR-0014 as 128 of 256 in total."""
    raw = json.loads((ROWS / "invariance-control-arm-b300-2026-09-14.json").read_text())["raw"][
        "ragged"
    ]
    path = tmp_path / "ragged.json"
    path.write_text(json.dumps(raw))
    done = _helper("record", str(path), _expect(max=38, churn_period_s=None, gate_exit=1))
    assert done.returncode == 0, done.stdout
    doc = json.loads(done.stdout)
    assert doc["verdict"] == "divergent"
    assert doc["streams_differing"] == {
        "32a_vs_1": 118,
        "32b_vs_1": 118,
        "max_vs_1": 66,
        "32b_vs_32a": 0,
    }
    assert doc["streams_differing_from_1_any_level"] == 128


GATE_TEXTS = {f"s{i}": f"word{i} two" for i in range(4)}


def _finals(texts: dict[str, str]) -> dict[str, FinalRecord]:
    return {
        sid: FinalRecord(
            sid, text, tuple((w, 80 * k, 80 * (k + 1)) for k, w in enumerate(text.split()))
        )
        for sid, text in texts.items()
    }


def _gate_pair(
    tmp_path: Path,
    *,
    manifest_texts: dict | None = None,
    level_texts: dict[str, dict[str, str]] | None = None,
) -> tuple[Path, Path, str]:
    """A record and the finals file beside it, as `verbatim-bench invariance` writes them,
    and a manifest holding each stream's reference text. Every level transcribes
    GATE_TEXTS unless `level_texts` gives a slot other text; the record's exit code is the
    one the gate computes for what the levels say."""
    texts = {slot: GATE_TEXTS for slot in ("1", "32a", "32b", "max")} | (level_texts or {})
    concurrency = {"1": 1, "32a": 32, "32b": 32, "max": 42}
    runs = [
        invariance.LevelRun(invariance.Level(slot, concurrency[slot]), _finals(texts[slot]))
        for slot in ("1", "32a", "32b", "max")
    ]
    report = invariance.assess(runs, chunk_ms=160, corpus={"id": "sha256:x", "utterances": 4})
    tmp_path.mkdir(parents=True, exist_ok=True)
    rec, fin = tmp_path / "invariance.json", tmp_path / "finals.json"
    rec.write_text(json.dumps(report.to_json_dict()))
    fin.write_text(json.dumps(report.finals_json_dict()))
    manifest = _text_manifest(tmp_path / "refs", manifest_texts or GATE_TEXTS)
    expect = _expect(
        churn_period_s=None,
        utterances=4,
        corpus_id="sha256:x",
        finals_path=str(fin),
        manifest=str(manifest),
        level1_wer_max=0.5,
        gate_exit=report.exit_code,
    )
    return rec, fin, expect


def _text_manifest(root: Path, texts: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for sid, text in texts.items():
        audio = root / f"{sid}.wav"
        audio.write_bytes(b"")
        lines.append(
            json.dumps(
                {"audio_filepath": str(audio), "duration": 1.0, "text": text, "stream_id": sid}
            )
        )
    path = root / "refs.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_finals_file_must_be_the_run_the_record_describes(tmp_path: Path) -> None:
    rec, fin, expect = _gate_pair(tmp_path)
    done = _helper("record", str(rec), expect)
    assert done.returncode == 0, done.stdout
    assert json.loads(done.stdout)["finals_digest_match"] == dict.fromkeys(
        ("1", "32a", "32b", "max"), True
    )
    original = fin.read_text()
    doc = json.loads(original)
    doc["levels"][3]["finals"][0]["words"][1][2] = 161  # one timing, one millisecond
    fin.write_text(json.dumps(doc))
    done = _helper("record", str(rec), expect)
    assert done.returncode == 1 and "does not re-digest" in done.stdout
    assert json.loads(done.stdout)["finals_digest_match"]["max"] is False
    # A level missing from the finals file is not "nothing to compare".
    doc = json.loads(original)
    doc["levels"] = doc["levels"][:3]
    fin.write_text(json.dumps(doc))
    done = _helper("record", str(rec), expect)
    assert done.returncode == 1 and "does not re-digest" in done.stdout
    assert json.loads(done.stdout)["finals_digest_match"]["max"] is False
    doc = json.loads(original)
    doc["record"] = "vb-invariance-finals/0"
    fin.write_text(json.dumps(doc))
    done = _helper("record", str(rec), expect)
    assert done.returncode == 1 and "finals kind" in done.stdout
    fin.unlink()
    done = _helper("record", str(rec), expect)
    assert done.returncode == 1 and "no finals file" in done.stdout
    assert "could not be scored" in done.stdout


def test_level_one_is_scored_against_the_reference(tmp_path: Path) -> None:
    rec, _, expect = _gate_pair(tmp_path / "same")
    done = _helper("record", str(rec), expect)
    assert done.returncode == 0, done.stdout
    wer = json.loads(done.stdout)["level1_wer"]
    assert wer["wer"] == 0.0 and wer["reference_words"] == 8 and wer["streams_with_a_final"] == 4
    # The gate's pinned normaliser: case and punctuation are not errors.
    shouted = {sid: text.upper().replace(" ", ", ") + "." for sid, text in GATE_TEXTS.items()}
    rec, _, expect = _gate_pair(tmp_path / "punctuated", manifest_texts=shouted)
    done = _helper("record", str(rec), expect)
    assert done.returncode == 0 and json.loads(done.stdout)["level1_wer"]["wer"] == 0.0
    # A stream with no level-1 final counts its whole reference as deletions.
    extra = {**GATE_TEXTS, "s9": "three more words"}
    rec, _, expect = _gate_pair(tmp_path / "extra", manifest_texts=extra)
    done = _helper("record", str(rec), expect)
    wer = json.loads(done.stdout)["level1_wer"]
    assert (wer["errors"], wer["reference_words"], wer["streams_with_a_final"]) == (3, 11, 4)
    # Consistent, non-empty, wrong text: invariant by digest, refused here.
    wrong = {sid: "alpha beta" for sid in GATE_TEXTS}
    rec, _, expect = _gate_pair(tmp_path / "wrong", manifest_texts=wrong)
    done = _helper("record", str(rec), expect)
    doc = json.loads(done.stdout)
    assert doc["verdict"] == "invariant"
    assert done.returncode == 1 and "over the sanity bound 0.5" in done.stdout, done.stdout
    assert doc["level1_wer"]["wer"] == 1.0


def test_level_one_and_no_other_level_is_scored(tmp_path: Path) -> None:
    """Two records that diverge at one level only. Level 1 right and max wrong is within
    the bound: the divergence is the gate's to report, not the WER check's. Level 1 wrong
    and every other level right is over it."""
    wrong = {sid: "alpha beta" for sid in GATE_TEXTS}
    rec, _, expect = _gate_pair(tmp_path / "max-wrong", level_texts={"max": wrong})
    done = _helper("record", str(rec), expect)
    doc = json.loads(done.stdout)
    assert done.returncode == 0, done.stdout
    assert doc["verdict"] == "divergent" and doc["streams_differing"]["max_vs_1"] == 4
    assert doc["level1_wer"]["wer"] == 0.0
    rec, _, expect = _gate_pair(tmp_path / "one-wrong", level_texts={"1": wrong})
    done = _helper("record", str(rec), expect)
    doc = json.loads(done.stdout)
    assert doc["verdict"] == "divergent" and doc["level1_wer"]["wer"] == 1.0
    assert done.returncode == 1 and "level-1 WER 1.0000 is over the sanity bound" in done.stdout


# --- the summary: every exit code, from the case that should produce it --------------

RAGGED_DIGESTS = {"1": "one", "32a": "r32a", "32b": "r32b", "max": "rmax"}
RAGGED_DIFF = {"32a_vs_1": 5, "32b_vs_1": 6, "max_vs_1": 3, "32b_vs_32a": 0}


def _arm_doc(
    root: Path,
    arm: str,
    verdict: str | None,
    *,
    digests=None,
    differing=None,
    bucket: int = 128,
    max_level: str = "42",
    operational: list[str] | None = None,
    invalidating: list[str] | None = None,
    runbook_sha: str | None = "same",
    git_head: str | None = "head",
    versions: dict | None = None,
) -> None:
    (root / arm).mkdir(parents=True)
    ragged = arm.startswith("ragged")
    present = verdict is not None
    summary = {
        "record_present": present,
        "problems": [] if present else ["no record was written"],
        "verdict": verdict,
        "digests": digests
        or (
            dict(RAGGED_DIGESTS)
            if ragged and verdict == "divergent"
            else dict.fromkeys(("1", "32a", "32b", "max"), "one")
        ),
        # As a real record summary carries them: every comparison counted, 0 when none.
        "streams_differing": differing
        if differing is not None
        else (dict(RAGGED_DIFF) if verdict == "divergent" else dict.fromkeys(RAGGED_DIFF, 0)),
        "levels": [],
    }
    (root / arm / "arm.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "padding": arm.split("-")[0],
                "occupancy": arm.split("-")[1],
                "record_summary": summary,
                "operational_problems": operational or [],
                "invalidating_problems": invalidating or [],
                "serve_spec": {"spec": {"matmul_precision": "high", "batch_size": bucket}},
                "gate_argv": ["verbatim-bench", "invariance", "--max", max_level, "--out", "x"],
                "runbook": {"sha256": runbook_sha},
                "git_head": git_head,
                "code": {
                    "verbatim": "x",
                    **(versions if versions is not None else {"version_verbatim": "0.1"}),
                },
            }
        )
    )


CLEAN = {
    "fixed-churn": ("invariant",),
    "ragged-churn": ("divergent",),
    "fixed-const": ("invariant",),
    "ragged-const": ("divergent",),
}
OP_PROBLEM = "no /admission reading after the gate"
CODE_CHANGED = "the code under test changed during the arm: HEAD is b, not the a preflight recorded"


def _case(**arms) -> dict:
    out = dict(CLEAN)
    for name, value in arms.items():
        name = name.replace("_", "-")
        if value is None:
            out.pop(name)
        else:
            out[name] = value
    return out


def _write_arms(root: Path, arms: dict) -> None:
    for arm, spec in arms.items():
        extra = spec[3] if len(spec) > 3 else {}
        _arm_doc(
            root,
            arm,
            spec[0],
            digests=spec[1] if len(spec) > 1 else None,
            differing=spec[2] if len(spec) > 2 else None,
            **extra,
        )


@pytest.mark.parametrize(
    ("arms", "stopped", "code", "reading"),
    [
        (CLEAN, "", 0, "controlled by its ragged twin"),
        # A fixed divergence is exit 1 even when another arm has no verdict.
        (
            _case(fixed_churn=("divergent", None, {"max_vs_1": 2}), ragged_churn=(None,)),
            "",
            1,
            "no verdict for ['ragged-churn']",
        ),
        # ... and when a later arm stopped the run.
        (
            _case(fixed_churn=("divergent", None, {"max_vs_1": 2}), ragged_const=None),
            "arm ragged-const: card 3 is in use",
            1,
            "the run stopped before it finished: arm ragged-const",
        ),
        # ... and beside an operational problem that does not touch the record, named.
        (
            _case(fixed_churn=("divergent", None, {"max_vs_1": 2}, {"operational": [OP_PROBLEM]})),
            "",
            1,
            "fixed-churn also had operational problems, none of which touches its record: "
            + OP_PROBLEM,
        ),
        # A problem that says the record is not this configuration's takes the 1 away, and
        # the reading still names the divergence it recorded.
        (
            _case(
                fixed_churn=("divergent", None, {"max_vs_1": 2}, {"invalidating": [CODE_CHANGED]})
            ),
            "",
            2,
            f"fixed-churn's record says divergent, but: {CODE_CHANGED}",
        ),
        # A ragged arm with an operational problem controls nothing.
        (
            _case(ragged_churn=("divergent", None, None, {"operational": [OP_PROBLEM]})),
            "",
            2,
            "no verdict for ['ragged-churn']",
        ),
        # A run that stopped with nothing divergent is 2, whatever finished.
        (
            _case(ragged_const=None),
            "arm ragged-const: card 3 is in use",
            2,
            "not run ['ragged-const']",
        ),
        # The churned pair's own control was clean.
        (
            _case(ragged_churn=("invariant",)),
            "",
            3,
            "fixed-churn max vs 1 (twin ragged-churn shows 0",
        ),
        # The churned twin diverged at 32 only.
        (
            _case(ragged_churn=("divergent", None, {"32a_vs_1": 9, "32b_vs_1": 9, "max_vs_1": 0})),
            "",
            3,
            "fixed-churn max vs 1",
        ),
        # Both fixed arms invariant, disagreeing at every level.
        (
            _case(fixed_const=("invariant", dict.fromkeys(("1", "32a", "32b", "max"), "two"))),
            "",
            4,
            "differ at level max",
        ),
        (
            _case(ragged_const=("divergent", {**RAGGED_DIGESTS, "1": "other"})),
            "",
            4,
            "ragged-churn and ragged-const differ at level 1",
        ),
        # The ragged 32a/32b comparison across a restart is information only.
        (
            _case(ragged_const=("divergent", {**RAGGED_DIGESTS, "32a": "moved", "32b": "too"})),
            "",
            0,
            "controlled",
        ),
        (_case(ragged_const=(None,)), "", 2, "no verdict for ['ragged-const']"),
        (
            _case(fixed_const=None, ragged_const=None),
            "",
            3,
            "not run: ['fixed-const', 'ragged-const']",
        ),
        (_case(fixed_const=None), "", 3, "not run: ['fixed-const']"),
        # All four arms ran with a verdict, but a level the restart comparison needs has no
        # digest in one of them: that comparison was not made, so no 0. (The record check
        # refuses such a record; this is the summary's own guard, behind it.)
        (
            _case(fixed_const=("invariant", {"1": "one", "32a": "one", "32b": "one", "max": None})),
            "",
            3,
            "restart comparison not made at: ['fixed level max']",
        ),
        (
            _case(ragged_const=("divergent", {**RAGGED_DIGESTS, "1": None})),
            "",
            3,
            "restart comparison not made at: ['ragged level 1']",
        ),
        (_case(fixed_churn=None, fixed_const=None), "", 3, "no fixed arm"),
        (
            _case(
                fixed_const=(
                    "divergent",
                    {"1": "one", "32a": "one", "32b": "one", "max": "x"},
                    {"max_vs_1": 1},
                ),
                fixed_churn=("invariant", dict.fromkeys(("1", "32a", "32b", "max"), "two")),
            ),
            "",
            1,
            "also: fixed-churn and fixed-const differ at level 1",
        ),
    ],
)
def test_every_summary_exit_code_is_reached_by_its_case(
    tmp_path: Path, arms: dict, stopped: str, code: int, reading: str
) -> None:
    _write_arms(tmp_path, arms)
    done = _helper("summary", str(tmp_path), *([stopped] if stopped else []))
    assert done.returncode == code, done.stdout[-2000:] + done.stderr[-2000:]
    doc = json.loads((tmp_path / "summary.json").read_text())
    assert doc["exit_code"] == code and reading in doc["reading"], doc["reading"]
    assert doc["scope"].startswith("EAGER ONLY")
    assert doc["stopped"] == (stopped or None)
    assert "EAGER ONLY" in done.stdout.splitlines()[-1]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"bucket": 96}, "ran at different settings"),
        ({"max_level": "40"}, "ran at different settings"),
        ({"runbook_sha": "other"}, "ran at different settings"),
        ({"git_head": "other"}, "ran at different settings"),
        ({"versions": {"version_verbatim": "0.2"}}, "ran at different settings"),
        ({"git_head": None}, "did not record their settings"),
        ({"runbook_sha": None}, "did not record their settings"),
        ({"versions": {}}, "did not record their settings"),
    ],
)
def test_arms_run_at_different_settings_are_not_read_against_each_other(
    tmp_path: Path, override: dict, message: str
) -> None:
    """A continued run directory whose ragged-const ran at another bucket or max level, or
    with another runbook, git HEAD or package set: its control and its restart comparison
    would be between different experiments."""
    for arm, spec in CLEAN.items():
        _arm_doc(tmp_path, arm, spec[0], **(override if arm == "ragged-const" else {}))
    done = _helper("summary", str(tmp_path))
    assert done.returncode == 2, done.stdout[-2000:]
    doc = json.loads((tmp_path / "summary.json").read_text())
    assert message in doc["reading"], doc["reading"]
    assert doc["settings_by_arm"]["fixed-churn"] == {
        "bucket": 128,
        "max_level": "42",
        "runbook_sha256": "same",
        "git_head": "head",
        "versions": {"version_verbatim": "0.1"},
    }


def test_an_uncontrolled_pass_names_every_uncontrolled_comparison(tmp_path: Path) -> None:
    _write_arms(
        tmp_path,
        _case(
            ragged_churn=("invariant",),
            ragged_const=("divergent", None, {"32a_vs_1": 4, "32b_vs_1": 0, "max_vs_1": 2}),
        ),
    )
    done = _helper("summary", str(tmp_path))
    assert done.returncode == 3
    reading = json.loads((tmp_path / "summary.json").read_text())["reading"]
    for named in (
        "fixed-churn 32a vs 1",
        "fixed-churn 32b vs 1",
        "fixed-churn max vs 1",
        "fixed-const 32b vs 1",
    ):
        assert named in reading
    for controlled in ("fixed-const 32a vs 1", "fixed-const max vs 1"):
        assert controlled not in reading


def test_a_helper_that_raises_exits_2_never_1(tmp_path: Path) -> None:
    """Exit 1 means a fixed arm diverged; a summary that crashes on a malformed arm file must
    not say so."""
    (tmp_path / "fixed-churn").mkdir()
    (tmp_path / "fixed-churn" / "arm.json").write_text('{"arm": "fixed-churn"')  # truncated
    done = _helper("summary", str(tmp_path))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "exit 2 (no verdict), never 1" in done.stderr
    assert _helper("facts", str(tmp_path / "absent.json"), "x", "{}").returncode == 2


# --- the smoke's decision, and the monitor ------------------------------------------


def _reading(**over) -> dict:
    base = {
        "p95_tick_ms": BUDGET_MS / 4,
        "consecutive_overruns": 0,
        "degradation_level": 0,
        "refused_total": 0,
    }
    base.update(over)
    return base


def _smoke_doc() -> dict:
    return {
        "solo_target_s": 60.0,
        "burst_concurrency": 4,
        "phases": {
            "solo": {"sessions": 3, "errors": 0, "first_error": None},
            "burst": {"sessions": 4, "errors": 0, "first_error": None},
        },
        "admission": {"before": _reading(), "end_of_solo": _reading(), "end_of_burst": _reading()},
        "samples": [
            {"t": 0.0, "phase": "solo", "admission": _reading()},
            {"t": 1.0, "phase": "burst", "admission": _reading()},
        ],
    }


def _smoke_check(doc: dict) -> subprocess.CompletedProcess:
    return _helper("smoke_check", _write(doc), str(BUDGET_MS))


def test_a_clean_smoke_lets_the_gate_run() -> None:
    done = _smoke_check(_smoke_doc())
    assert done.returncode == 0, done.stdout
    assert json.loads(done.stdout)["max_p95_tick_ms"] == BUDGET_MS / 4


def test_a_slow_first_tick_in_the_solo_phase_does_not_stop_the_run() -> None:
    """A fresh server's first ticks are read only while they are still in the p95 window;
    the gate runs after them."""
    doc = _smoke_doc()
    doc["samples"][0]["admission"] = _reading(p95_tick_ms=BUDGET_MS * 2)
    assert _smoke_check(doc).returncode == 0


def test_the_smokes_solo_phase_outlasts_the_p95_window() -> None:
    """The solo phase is there so that the p95 read at its end no longer holds a fresh
    server's first ticks: its default length must be longer than WINDOW_TICKS ticks of the
    runbook's chunk, both read from where they are set, and the comment must say so."""
    from verbatim.scheduler.admission import WINDOW_TICKS

    text = SCRIPT.read_text()
    chunk_ms = int(re.search(r"^readonly CHUNK_MS=(\d+)$", text, re.M).group(1))
    default = re.search(r'^SMOKE_SOLO_S="\$\{RUNBOOK_SMOKE_SOLO_S:-([0-9.]+)\}"$', text, re.M)
    window_s = WINDOW_TICKS * chunk_ms / 1000
    assert float(default.group(1)) > window_s, (default.group(1), window_s)
    assert f"{WINDOW_TICKS}-tick p95 window" in text and f"{window_s:g} s at {chunk_ms} ms" in text


def _mutate(doc: dict, where: str, value) -> dict:
    if where == "burst_sample_p95":
        doc["samples"][1]["admission"]["p95_tick_ms"] = value
    elif where == "end_of_solo_p95":
        doc["admission"]["end_of_solo"]["p95_tick_ms"] = value
    elif where == "end_of_burst_p95":
        doc["admission"]["end_of_burst"]["p95_tick_ms"] = value
    elif where == "refused":
        doc["admission"]["end_of_burst"]["refused_total"] = value
    elif where == "degradation":
        doc["samples"][0]["admission"]["degradation_level"] = value
    elif where == "errors":
        doc["phases"]["burst"]["errors"] = value
    elif where == "short_burst":
        doc["phases"]["burst"]["sessions"] = value
    elif where == "unread":
        doc["admission"] = {k: {"error": "URLError"} for k in doc["admission"]}
        doc["samples"] = [{"t": 0.0, "phase": "burst", "admission": {"error": "URLError"}}]
    return doc


@pytest.mark.parametrize(
    ("where", "value", "message"),
    [
        ("burst_sample_p95", BUDGET_MS, "p95_tick_ms reached"),  # the budget itself is over
        ("end_of_solo_p95", BUDGET_MS + 1, "p95_tick_ms reached"),
        ("end_of_burst_p95", BUDGET_MS + 1, "p95_tick_ms reached"),
        ("refused", 1, "refused_total rose from 0 to 1"),
        ("degradation", 1, "degradation_level reached 1"),
        ("errors", 2, "burst sessions errored"),
        ("short_burst", 3, "the corpus is too small"),
        ("unread", None, "no p95_tick_ms reading"),
    ],
)
def test_a_smoke_the_gate_would_lose_its_verdict_on_stops_the_run(
    where: str, value, message: str
) -> None:
    done = _smoke_check(_mutate(_smoke_doc(), where, value))
    assert done.returncode == 1 and message in done.stdout, done.stdout


class _Admission(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """The stub's state, after its `delay` seconds (0 unless a test sets it)."""
        time.sleep(getattr(self.server, "delay", 0.0))
        body = json.dumps(self.server.state).encode()  # type: ignore[attr-defined]
        # A reader that timed out has closed the connection by now.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def admission_stub():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Admission)
    server.state = _reading()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _card_shim(root: Path, *lines: str, body: str | None = None) -> Path:
    """An nvidia-smi whose compute-apps listing is `lines`, or whose script is `body`."""
    root.mkdir(parents=True, exist_ok=True)
    shim = root / "nvidia-smi"
    shim.write_text(body or "#!/bin/sh\n" + "".join(f"echo '{line}'\n" for line in lines))
    shim.chmod(0o755)
    return root


#: An nvidia-smi that never answers (exec, so the timeout's kill reaches the sleep itself).
HANGING_SMI = "#!/bin/sh\nexec sleep 30\n"


def _monitor(
    server,
    gate: subprocess.Popen,
    tmp_path: Path,
    *card_lines: str,
    every: str = "0.05",
    smi_body: str | None = None,
) -> subprocess.Popen:
    """The runbook's monitor, watching `gate`, with a 2 s nvidia-smi timeout and a 1 s
    /admission timeout, sleeping `every` between polls."""
    url = f"http://127.0.0.1:{server.server_address[1]}/admission"
    shim = _card_shim(tmp_path / "smi", *(card_lines or ("4242, verbatim, 2048",)), body=smi_body)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _helper_source(),
            "monitor",
            url,
            str(tmp_path / "m.jsonl"),
            str(gate.pid),
            every,
            "0",
            str(tmp_path / "abort.txt"),
            "4242",
            "3",
            "2",  # the nvidia-smi timeout
            "1",  # the /admission timeout
        ],
        env={**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}"},
    )


def _watch(monitor: subprocess.Popen, gate: subprocess.Popen, within: float = 30) -> None:
    """Wait for the gate, then for the monitor to see it gone and exit 0 by itself."""
    try:
        assert gate.wait(timeout=within) == 0
        assert monitor.wait(timeout=within) == 0
    finally:
        for proc in (gate, monitor):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def _monitor_summary(
    path: Path,
    start: float,
    end: float,
    *,
    every: str = "0.05",
    card: str = "2",
    read: str = "1",
    exit_status: str = "0",
) -> dict:
    """monitor_summary over `path` for a gate from `start` to `end` (by default with the
    sleep and timeouts _monitor runs with)."""
    done = _helper(
        "monitor_summary", str(path), "3", every, card, read, str(start), str(end), exit_status
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"refused_total": 1}, "refused_total rose from 0 to 1"),
        ({"degradation_level": 1}, "degradation_level 1"),
    ],
)
def test_the_monitor_stops_the_gate_as_soon_as_the_verdict_is_lost(
    admission_stub, tmp_path: Path, change: dict, message: str
) -> None:
    gate = subprocess.Popen(["sleep", "60"])
    monitor = _monitor(admission_stub, gate, tmp_path)
    try:
        time.sleep(0.4)
        assert gate.poll() is None  # a clean reading leaves the gate alone
        admission_stub.state = _reading(**change)
        assert gate.wait(timeout=10) == -15
        assert monitor.wait(timeout=10) == 0
    finally:
        for proc in (gate, monitor):
            if proc.poll() is None:
                proc.kill()
    assert message in (tmp_path / "abort.txt").read_text()
    lines = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert lines[0]["monitor"]["poll_sleep_s"] == 0.05
    assert len([line for line in lines if "card" in line]) >= 2


def test_the_monitor_leaves_a_clean_gate_alone_and_stops_with_it(
    admission_stub, tmp_path: Path
) -> None:
    start = time.time()
    gate = subprocess.Popen(["sleep", "0.6"])
    monitor = _monitor(admission_stub, gate, tmp_path)
    try:
        assert gate.wait(timeout=10) == 0
        end = time.time()
        assert monitor.wait(timeout=10) == 0  # it saw the gate go, and stopped by itself
    finally:
        if monitor.poll() is None:
            monitor.kill()
    assert not (tmp_path / "abort.txt").exists()
    summary = _monitor_summary(tmp_path / "m.jsonl", start, end)
    assert summary["readings"] >= 1 and summary["card_polls"] == summary["readings"]
    assert summary["card_polls_with_foreign"] == 0 and summary["card_polls_without_server"] == 0
    assert summary["timestamped_polls"] == summary["card_polls"]
    assert summary["unwatched_stretches_over_bound"] == [] and summary["monitor_exit"] == 0
    assert summary["problems"] == []


def test_the_monitor_records_what_else_held_the_card(admission_stub, tmp_path: Path) -> None:
    gate = subprocess.Popen(["sleep", "0.6"])
    monitor = _monitor(
        admission_stub, gate, tmp_path, "4242, verbatim, 2048", "999999, python3, 1024"
    )
    try:
        assert gate.wait(timeout=10) == 0  # recorded, not acted on
        assert monitor.wait(timeout=10) == 0
    finally:
        if monitor.poll() is None:
            monitor.kill()
    summary = _monitor_summary(tmp_path / "m.jsonl", time.time() - 0.6, time.time())
    assert summary["card_polls_with_foreign"] == summary["card_polls"] >= 1
    assert summary["card_foreign_seen"] == ["999999, python3, 1024"]
    assert [p for p in summary["problems"] if "a process other than the server on card 3" in p]


def test_the_monitor_applies_the_nvidia_smi_timeout_it_was_given(
    admission_stub, tmp_path: Path
) -> None:
    """An nvidia-smi that never answers: every poll's card query fails at the 2 s nvidia-smi
    timeout, not at the 1 s /admission one and not never, as the error text and the time the
    monitor measured around the query both say."""
    start = time.time()
    gate = subprocess.Popen(["sleep", "3"])
    _watch(_monitor(admission_stub, gate, tmp_path, smi_body=HANGING_SMI), gate)
    summary = _monitor_summary(tmp_path / "m.jsonl", start, time.time())
    assert summary["card_polls"] >= 1 and summary["card_poll_errors"] == summary["card_polls"]
    assert summary["first_card_poll_error"].endswith("timed out after 2.0 seconds"), summary
    assert 1.9 <= summary["longest_card_query_s"] < 3.0, summary
    assert summary["readings"] == summary["card_polls"]  # /admission answered at once


def test_the_monitor_applies_the_admission_timeout_it_was_given(
    admission_stub, tmp_path: Path
) -> None:
    """An /admission that answers after 3 s: the monitor's 1 s read timeout fails every read
    at about 1 s. With the 5 s default the read would succeed; with the 2 s nvidia-smi
    timeout in its place it would fail at 2 s."""
    admission_stub.delay = 3.0
    start = time.time()
    gate = subprocess.Popen(["sleep", "2.5"])
    _watch(_monitor(admission_stub, gate, tmp_path), gate)
    summary = _monitor_summary(tmp_path / "m.jsonl", start, time.time())
    assert summary["read_errors"] >= 1 and summary["readings"] == 0, summary
    assert "timed out" in summary["first_read_error"], summary
    assert 0.9 <= summary["longest_admission_read_s"] < 1.6, summary


def test_the_monitor_sleeps_between_polls_what_it_was_given(admission_stub, tmp_path: Path) -> None:
    """Asked to sleep 0.5 s between polls, with both reads answering at once, the monitor's
    polls are written 0.5 s apart and not much more, as their own timestamps say."""
    start = time.time()
    gate = subprocess.Popen(["sleep", "2.8"])
    _watch(_monitor(admission_stub, gate, tmp_path, every="0.5"), gate)
    lines = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    stamps = [line["t"] for line in lines if "card" in line]
    spacings = [b - a for a, b in itertools.pairwise(stamps)]
    assert len(spacings) >= 3 and min(spacings) >= 0.5, spacings
    summary = _monitor_summary(tmp_path / "m.jsonl", start, time.time(), every="0.5")
    assert summary["poll_spacing_median_s"] < 0.75, (summary, spacings)
    assert summary["longest_admission_read_s"] < 0.5 and summary["longest_card_query_s"] < 0.5


def test_the_monitor_summary_counts_a_truncated_line_instead_of_failing(tmp_path: Path) -> None:
    """The monitor is stopped with SIGTERM, so its last line can be cut short."""
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"t": 1, "admission": _reading()}) + '\n{"t": 2, "admiss')
    doc = _monitor_summary(path, 0.5, 2.5, every="30")
    assert doc["readings"] == 1 and doc["unparsable_lines"] == 1


SERVER_POLL = {"server_seen": True, "foreign": []}
FOREIGN_POLL = {"server_seen": True, "foreign": ["999999, python3, 1024"]}
ABSENT_POLL = {"server_seen": False, "foreign": []}
FAILED_POLL = {"error": "nvidia-smi exited 9: NVIDIA-SMI has failed"}
FAILED_READ = {"error": "URLError: <urlopen error [Errno 111] Connection refused>"}
CLEAN_ENTRY = (_reading(), SERVER_POLL)


@pytest.mark.parametrize(
    ("entries", "messages"),
    [
        ([CLEAN_ENTRY] * 5, []),
        # An /admission read that fails now and then is counted, not acted on:
        # refused_total is cumulative and is read again after the gate.
        ([CLEAN_ENTRY] * 4 + [(FAILED_READ, SERVER_POLL)], []),
        # One poll in five is enough, whichever it is.
        ([CLEAN_ENTRY] * 2 + [(_reading(), FOREIGN_POLL)] + [CLEAN_ENTRY] * 2, ["at 1 of 5 polls"]),
        (
            [CLEAN_ENTRY] * 4 + [(_reading(), ABSENT_POLL)],
            ["did not show the server on card 3 at 1 of 5"],
        ),
        ([(_reading(), FAILED_POLL)] + [CLEAN_ENTRY] * 4, ["nvidia-smi failed at 1 of 5 polls"]),
        ([(_reading(), FAILED_POLL)] * 5, ["nvidia-smi failed at 5 of 5 polls"]),
        ([(_reading(), None)] * 5, ["made no nvidia-smi poll of card 3"]),
        ([(FAILED_READ, SERVER_POLL)] * 5, ["no /admission reading during the gate (5 failed"]),
        # No monitor log at all: nothing was observed, and nothing observed is not clean.
        (
            None,
            [
                "made no nvidia-smi poll of card 3",
                "the monitor wrote no line saying what it parsed",
                "no /admission reading during the gate (0",
            ],
        ),
    ],
    ids=[
        "clean",
        "one-read-failed",
        "one-foreign",
        "one-absent",
        "one-poll-failed",
        "every-poll-failed",
        "no-polls",
        "no-readings",
        "no-log",
    ],
)
def test_what_the_monitor_did_not_see_is_a_problem_too(
    tmp_path: Path, entries: list | None, messages: list[str]
) -> None:
    path = tmp_path / "m.jsonl"
    if entries is not None:
        path.write_text(
            _monitor_line(30, 2, 1)
            + "".join(
                json.dumps({"t": i, "admission": a, **({"card": c} if c is not None else {})})
                + "\n"
                for i, (a, c) in enumerate(entries)
            )
        )
    problems = _monitor_summary(path, 0, 4, every="30")["problems"]
    assert len(problems) == len(messages), problems
    for message, problem in zip(messages, problems, strict=True):
        assert message in problem, problems


def _monitor_line(every: float, card: float, read: float) -> str:
    """The first line the monitor writes: the arguments it parsed."""
    ran = {"poll_sleep_s": every, "card_query_timeout_s": card, "admission_read_timeout_s": read}
    return json.dumps({"t": 0.0, "monitor": ran}) + "\n"


def _polls_at(
    path: Path, stamps: list[float], ran: tuple[float, float, float] | None = (1.0, 1.0, 1.0)
) -> Path:
    """A monitor log whose polls were written at `stamps`, after the line saying what the
    monitor parsed (none when `ran` is None)."""
    path.write_text(
        (_monitor_line(*ran) if ran is not None else "")
        + "".join(
            json.dumps({"t": t, "admission": _reading(), "card": SERVER_POLL}) + "\n"
            for t in stamps
        )
    )
    return path


#: With a 1 s sleep between polls and 1 s timeouts, the bound is 3 x 1 + 1 + 1 = 5 s.
UNWATCHED = "the monitor observed nothing for"


@pytest.mark.parametrize(
    ("stamps", "end", "exit_status", "messages"),
    [
        ([0.5, 5.0, 9.5], 10, "0", []),
        # A stretch of exactly the bound is allowed; only a longer one is not.
        ([5.0, 10.0], 15, "0", []),
        # A poll written just after the gate ended covers up to the end, no further.
        ([0.5, 4.0, 8.5], 8, "0", []),
        ([5.5, 6.0, 7.0], 8, "0", [f"{UNWATCHED} 5.5 s of the gate, from 0.0 s to 5.5 s"]),
        ([0.5, 1.0, 7.0, 8.0], 8, "0", [f"{UNWATCHED} 6.0 s of the gate, from 1.0 s to 7.0 s"]),
        ([0.5, 1.0, 2.0], 8, "0", [f"{UNWATCHED} 6.0 s of the gate, from 2.0 s to 8.0 s"]),
        (
            [0.5, 7.0, 7.5],
            15,
            "0",
            [f"{UNWATCHED} 7.5 s of the gate, from 7.5 s to 15.0 s after it started (2 stretch"],
        ),
        (
            [],
            8,
            "0",
            [
                "made no nvidia-smi poll of card 3",
                f"{UNWATCHED} 8.0 s of the gate, from 0.0 s to 8.0 s",
                "no /admission reading during the gate",
            ],
        ),
        ([0.5, 4.0, 7.5], 8, "137", ["the monitor's exit status is 137, not 0"]),
        ([0.5, 4.0, 7.5], 8, "", ["the monitor's exit status is not recorded, not 0"]),
    ],
    ids=[
        "covered",
        "exactly-the-bound",
        "poll-written-after-the-end",
        "late-first-poll",
        "gap-mid-gate",
        "early-last-poll",
        "two-gaps",
        "no-poll-in-a-long-gate",
        "monitor-killed",
        "exit-not-recorded",
    ],
)
def test_how_much_of_the_gate_the_polls_covered_is_judged_from_their_timestamps(
    tmp_path: Path, stamps: list[float], end: float, exit_status: str, messages: list[str]
) -> None:
    """The gate runs from 0 to `end`. The polls' own timestamps, not the spacing asked for,
    say what was watched: any stretch longer than the bound with no poll in it, the one
    before the first poll and the one after the last included, takes the verdict away, and
    so does a monitor that did not exit 0 by itself."""
    path = _polls_at(tmp_path / "m.jsonl", stamps)
    doc = _monitor_summary(path, 0, end, every="1", card="1", read="1", exit_status=exit_status)
    assert doc["unwatched_bound_s"] == 5.0
    problems = doc["problems"]
    assert len(problems) == len(messages), problems
    for message, problem in zip(messages, problems, strict=True):
        assert message in problem, problems


def test_a_monitor_the_runbook_had_to_stop_says_so(tmp_path: Path) -> None:
    path = _polls_at(tmp_path / "m.jsonl", [0.5, 4.0, 7.5])
    done = _helper(
        "monitor_summary",
        str(path),
        "3",
        "1",
        "1",
        "1",
        "0",
        "8",
        "143",
        "still running 5 s after the gate ended; stopped by the runbook",
    )
    doc = json.loads(done.stdout)
    assert doc["monitor_exit"] == 143 and doc["monitor_end"].startswith("still running")
    assert doc["problems"] == [
        "the monitor's exit status is 143 (still running 5 s after the gate ended; stopped by the "
        "runbook), not 0: it did not run to the gate's end and stop by itself, so what it did not "
        "write was not observed"
    ]


@pytest.mark.parametrize(
    ("ran", "problems"),
    [
        ((1.0, 2.0, 1.0), []),
        # The runbook's two timeouts swapped on their way to the monitor.
        (
            (1.0, 1.0, 2.0),
            [
                "the monitor parsed a 1.0 s sleep between polls, a 1.0 s nvidia-smi timeout and "
                "a 2.0 s /admission timeout, not the 1 s, 2 s and 1 s the runbook asked for"
            ],
        ),
        (None, ["the monitor wrote no line saying what it parsed"]),
    ],
    ids=["as-asked", "timeouts-swapped", "not-written"],
)
def test_what_the_monitor_parsed_is_read_from_its_own_first_line(
    tmp_path: Path, ran: tuple | None, problems: list[str]
) -> None:
    """The sleep and timeouts the summary reports as parsed are the monitor's, as it wrote
    them; the ones the runbook asked for are reported beside them and set the unwatched
    bound. What the timeouts and the sleep did is measured, and tested, above."""
    path = _polls_at(tmp_path / "m.jsonl", [0.5, 4.0, 7.5], ran=ran)
    doc = _monitor_summary(path, 0, 8, every="1", card="2", read="1")
    assert [
        doc["poll_sleep_parsed_s"],
        doc["card_query_timeout_parsed_s"],
        doc["admission_read_timeout_parsed_s"],
    ] == (list(ran) if ran is not None else [None, None, None])
    assert [
        doc["poll_sleep_requested_s"],
        doc["card_query_timeout_requested_s"],
        doc["admission_read_timeout_requested_s"],
    ] == [1.0, 2.0, 1.0]
    assert doc["unwatched_bound_s"] == 6.0 and doc["timestamped_polls"] == 3
    assert len(doc["problems"]) == len(problems), doc["problems"]
    for message, problem in zip(problems, doc["problems"], strict=True):
        assert problem.startswith(message), doc["problems"]


def test_the_unwatched_bound_is_three_sleeps_plus_both_timeouts() -> None:
    """125 s in production (30 s between polls, 30 s nvidia-smi, 5 s /admission), 3.6 s in a
    rehearsal (0.2 s, 2 s, 1 s)."""
    for args, bound in ((("30", "30", "5"), "125"), (("0.2", "2", "1"), "3.6")):
        done = _helper("gap_bound", *args)
        assert done.returncode == 0 and done.stdout.strip() == bound, done.stdout + done.stderr
    text = SCRIPT.read_text()
    for assignment in (
        "MONITOR_EVERY_S=30\n",
        "CARD_QUERY_TIMEOUT_S=30\n",
        "ADMISSION_READ_TIMEOUT_S=5\n",
        "    MONITOR_EVERY_S=0.2\n",
        "    CARD_QUERY_TIMEOUT_S=2\n",
        "    ADMISSION_READ_TIMEOUT_S=1\n",
    ):
        assert text.count(assignment) == 1, assignment


def test_a_summary_without_its_problems_list_is_not_read_as_no_problems() -> None:
    assert _helper("lines", _write({"readings": 3}), "problems").returncode == 2
    assert _helper("lines", _write({"problems": "one"}), "problems").returncode == 2
    done = _helper("lines", _write({"problems": ["a\nb", "c"]}), "problems")
    assert done.returncode == 0 and done.stdout == "a b\nc\n"
    done = _helper("lines", _write({"problems": []}), "problems")
    assert done.returncode == 0 and done.stdout == ""


# --- refusals before anything starts -------------------------------------------------


def test_production_mode_refuses_the_test_overrides(tmp_path: Path) -> None:
    for var in (
        "RUNBOOK_OUT_DIR",
        "RUNBOOK_MANIFEST",
        "RUNBOOK_CHURN_S",
        "RUNBOOK_TEST_GIT_DIR",
        "RUNBOOK_TEST_SMOKE_BURST",
    ):
        done = _run("fixed-churn", env=_clean_env(**{var: str(tmp_path)}))
        assert done.returncode == 2 and "test-mode override" in done.stderr, (var, done.stderr)
    assert not any(tmp_path.iterdir())


def test_test_mode_may_not_write_into_the_measurement_directory(tmp_path: Path) -> None:
    measure = tmp_path / "measure"
    for out in (measure / "x", measure):
        env = _clean_env(
            RUNBOOK_TEST_FAKE="1",
            RUNBOOK_OUT_DIR=str(out),
            RUNBOOK_MANIFEST=str(tmp_path / "m"),
            RUNBOOK_MEASUREMENT_DIR=str(measure),
        )
        done = _run("fixed-churn", env=env)
        assert done.returncode == 2 and "may not write" in done.stderr, done.stderr
        assert not out.exists()


def test_an_unknown_arm_is_refused() -> None:
    done = _run("--dry-run", "fixed-maybe", env=_clean_env())
    assert done.returncode == 2 and "unknown arm" in done.stderr


@needs_venv
def test_production_refuses_a_corpus_that_is_not_the_published_one(tmp_path: Path) -> None:
    """Production mode, up to the corpus check: the manifest under RUNBOOK_CORPUS_ROOT is not
    the published corpus with its prefix moved. nvidia-smi is the failing shim, so nothing
    past preflight could run anyway."""
    root = tmp_path / "corpus"
    manifest = root / MANIFEST_REL
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": f"{root}/librispeech-test-other-256/audio/x.wav",
                "duration": 1.0,
                "text": "x",
                "stream_id": "x",
            }
        )
        + "\n"
    )
    env = _clean_env(RUNBOOK_CORPUS_ROOT=str(root), RUNBOOK_MEASUREMENT_DIR=str(tmp_path / "m"))
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2 and "the corpus is not the published one" in done.stderr, (
        done.stderr
    )
    assert "not the published" in done.stderr and "1 utterances" in done.stderr


# --- test mode: the card, checkpoint, code and git checks run for real, against stubs ----


def _fake_text(pcm: np.ndarray) -> str:
    """What `--pipeline fake` transcribes a clip of exactly one 160 ms chunk as: one token
    hashed from the chunk's samples, by the fake's own code, from the samples the server
    decodes (src/verbatim/pipelines/fake.py _token, src/verbatim/audio/pcm.py)."""
    samples, _ = decode_pcm16(pcm.tobytes())
    frame = PcmFrame(
        stream_id=1, samples=samples, is_first=True, is_last=True, valid_samples=len(samples)
    )
    return FakePipelineAdapter._token(frame)


def _tiny_manifest(root: Path, n: int, *, text: str | None = None, wrong: int = 0) -> Path:
    """`n` clips of 0.16 s of seeded noise. Each clip's reference text is what the fake
    transcribes it as, one word, so level 1 scores WER 0; `text` replaces every reference,
    and `wrong` replaces the first `wrong` with a word the fake never says, so level 1
    scores WER wrong / n."""
    audio = root / "audio"
    audio.mkdir(parents=True)
    lines = []
    for index in range(n):
        rng = np.random.default_rng([11, index])
        pcm = (rng.uniform(-0.5, 0.5, size=2560) * 32767.0).astype("<i2")  # 0.16 s
        path = audio / f"clip-{index:03d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        lines.append(
            json.dumps(
                {
                    "audio_filepath": str(path),
                    "duration": 0.16,
                    "text": text
                    if text is not None
                    else "noise"
                    if index < wrong
                    else _fake_text(pcm),
                    "stream_id": f"clip-{index:03d}",
                }
            )
        )
    manifest = root / "tiny.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


CHECKPOINT_BYTES = b"a stand-in checkpoint for the runbook's CPU tests\n"


def _hf_cache(
    root: Path,
    *,
    ref: str = REVISION,
    blob_name: str | None = None,
    snapshot: bool = True,
    xet: bool = False,
) -> Path:
    """The layout huggingface_hub keeps: refs/main names the revision, the snapshot's file
    is a symlink to a blob named for the sha256 of its bytes. With ``xet``, that blob is
    itself a link into the cache-wide content-addressed store named by another hash, as
    huggingface_hub 1.x lays out Xet downloads
    (<repo>/blobs/<sha256> -> ../../blobs/ce/<xet hash>, i.e. <cache>/blobs/ce/<xet hash>)."""
    repo = root / "models--nvidia--nemotron-speech-streaming-en-0.6b"
    name = blob_name or hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    (repo / "blobs").mkdir(parents=True)
    if xet:
        xet_name = hashlib.sha256(b"xet:" + CHECKPOINT_BYTES).hexdigest()
        (root / "blobs" / "ce").mkdir(parents=True)
        (root / "blobs" / "ce" / xet_name).write_bytes(CHECKPOINT_BYTES)
        (repo / "blobs" / name).symlink_to(f"../../blobs/ce/{xet_name}")
    else:
        (repo / "blobs" / name).write_bytes(CHECKPOINT_BYTES)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(ref)
    snapshot_dir = repo / "snapshots" / REVISION
    snapshot_dir.mkdir(parents=True)
    if snapshot:
        (snapshot_dir / NEMO_FILE).symlink_to(f"../../blobs/{name}")
    return root


GIT = ("-c", "user.name=runbook-test", "-c", "user.email=runbook-test@example.invalid")


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *GIT, "-c", "commit.gpgsign=false", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _git_repo(root: Path, *, dirty: str | None = None) -> Path:
    """A repository standing in for this one: src/ and bench/src/ committed, plus an
    untracked file outside them, which must not matter."""
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "bench" / "src").mkdir(parents=True)
    (root / "bench" / "src" / "b.py").write_text("y = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    (root / "notes.txt").write_text("untracked, outside src\n")
    if dirty is not None:
        target = root / dirty
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")
    return root


def _repo_copy(root: Path) -> Path:
    """A git repository holding a committed copy of the runbook at scripts/, with src/ and
    bench/src/ linked to this checkout's, so the copy tests this checkout's code."""
    root.mkdir(parents=True)
    (root / "scripts").mkdir()
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    (root / "src").symlink_to(WORKTREE / "src")
    (root / "bench").mkdir()
    (root / "bench" / "src").symlink_to(WORKTREE / "bench" / "src")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "the runbook")
    return root


def _rehearsal_env(
    tmp_path: Path,
    port: int,
    n: int,
    *,
    bucket: str = "64",
    max_level: str | None = None,
    smi_mode: str = "idle",
    text: str | None = None,
    wrong: int = 0,
    **extra: str,
) -> dict[str, str]:
    """A test-mode run over `n` tiny clips. The corpus root is the tiny manifest's own, as
    RUNBOOK_CORPUS_ROOT is the real one's in production: every audio path is under it, so
    the manifest's local id and its published-form id differ, as they do on the real
    corpus."""
    env = _clean_env(
        bucket=bucket,
        max_level=max_level or str(n),
        smi=_stub_smi(),
        RUNBOOK_TEST_FAKE="1",
        RUNBOOK_OUT_DIR=str(tmp_path / "out"),
        RUNBOOK_MANIFEST=str(_tiny_manifest(tmp_path / "corpus", n, text=text, wrong=wrong)),
        RUNBOOK_CORPUS_ROOT=str(tmp_path / "corpus"),
        RUNBOOK_CHURN_S="1",
        RUNBOOK_WS_PORT=str(port),
        RUNBOOK_RUN_ID="rehearsal",
        RUNBOOK_SMOKE_SOLO_S="0.5",
        RUNBOOK_CARD_IDLE_WAIT_S="0",
        RUNBOOK_TEST_GIT_DIR=str(_git_repo(tmp_path / "repo")),
        HF_HUB_CACHE=str(_hf_cache(tmp_path / "hf")),
        STUB_SMI_MODE=smi_mode,
        STUB_SMI_ROOT=str(tmp_path / "out"),
        STUB_SMI_LOG=str(tmp_path / "nvidia-smi-calls.log"),
    )
    env.update(extra)
    return env


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "out" / "gate-nemotron-card3-rehearsal"


def _port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _arm(tmp_path: Path, arm: str) -> dict:
    return json.loads((_run_dir(tmp_path) / arm / "arm.json").read_text())


def _summary(tmp_path: Path) -> dict:
    return json.loads((_run_dir(tmp_path) / "summary.json").read_text())


COMPUTE_APPS = (
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
)


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (("-i", "3", *COMPUTE_APPS), 0),
        (("-i", "0", *COMPUTE_APPS), 6),
        (COMPUTE_APPS, 6),
        (("-i", "3", "--query-gpu=uuid", "--format=csv,noheader"), 0),
        (("-i", "2", "--query-gpu=uuid", "--format=csv,noheader"), 6),
        (("--query-gpu=index,uuid,name,pci.bus_id,driver_version", "--format=csv,noheader"), 0),
    ],
    ids=["apps-card-3", "apps-card-0", "apps-no-card", "uuid-card-3", "uuid-card-2", "gpu-list"],
)
def test_the_stub_nvidia_smi_answers_for_card_3_alone(
    tmp_path: Path, args: tuple, code: int
) -> None:
    """The stub the rehearsals run with: a per-card query of any card but 3 fails, so a
    runbook or monitor asking about another card is seen; each call is logged with it."""
    log = tmp_path / "calls.log"
    done = subprocess.run(
        [str(_stub_smi() / "nvidia-smi"), *args],
        env={**os.environ, "STUB_SMI_ROOT": str(tmp_path), "STUB_SMI_LOG": str(log)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == code, done.stdout + done.stderr
    if code == 6:
        assert "only card 3 exists" in done.stderr and not done.stdout
    # "runbook" or "monitor", by the process asking (here pytest, which is not the monitor).
    assert log.read_text().split(" ", 1)[1] == " ".join(args) + "\n"


@needs_venv
def test_test_mode_refuses_to_run_without_the_nvidia_smi_stub(tmp_path: Path) -> None:
    env = _rehearsal_env(tmp_path, _free_port(), 4)
    env["PATH"] = f"{_no_gpu_shim()}{os.pathsep}{os.environ['PATH']}"
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2 and "stub nvidia-smi" in done.stderr, done.stderr


@needs_venv
def test_an_unexpected_failure_is_exit_2_never_1(tmp_path: Path) -> None:
    """A command failing where the runbook expected nothing to fail: `set -e` alone would
    exit with its status, here 1, the code for a fixed-arm divergence."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    env = _rehearsal_env(tmp_path, _free_port(), 4, RUNBOOK_OUT_DIR=str(blocker / "out"))
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2 and "unexpected failure (exit 1)" in done.stderr, done.stderr
    env = _rehearsal_env(tmp_path / "b", _free_port(), 4)
    env["RUNBOOK_MANIFEST"] = str(tmp_path / "no-such-manifest.jsonl")
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2 and "FileNotFoundError" in done.stderr, done.stderr


def _break_uuid(tmp_path: Path, env: dict) -> None:
    env["STUB_SMI_UUID"] = "GPU-00000000-0000-0000-0000-000000000000"


def _break_refs(tmp_path: Path, env: dict) -> None:
    shutil.rmtree(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(_hf_cache(tmp_path / "hf", ref="0" * 40))


def _break_blob(tmp_path: Path, env: dict) -> None:
    shutil.rmtree(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(_hf_cache(tmp_path / "hf", blob_name="ab" * 32))


def _break_xet_blob(tmp_path: Path, env: dict) -> None:
    """The Xet layout with the first link's name wrong: the bytes do not hash to it."""
    shutil.rmtree(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(_hf_cache(tmp_path / "hf", blob_name="cd" * 32, xet=True))


@needs_venv
def test_preflight_accepts_a_checkpoint_stored_the_xet_way(tmp_path: Path) -> None:
    """blobs/<sha256> -> blobs/ce/<xet hash>, as huggingface_hub 1.x stores Xet downloads:
    the first link's name is the sha256 and must pass. The wrong card then stops preflight
    at the card check, which comes after the checkpoint check."""
    env = _rehearsal_env(tmp_path, _free_port(), 4)
    shutil.rmtree(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(_hf_cache(tmp_path / "hf", xet=True))
    _break_uuid(tmp_path, env)
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2, done.stderr
    assert "not to its blob name" not in done.stderr, done.stderr
    assert "not GPU-b43f9262-f250-444a-bfbf-461dd3500f1e" in done.stderr, done.stderr


def _break_resolution(tmp_path: Path, env: dict) -> None:
    """refs/main is right, but the pinned snapshot has no file: huggingface_hub resolves
    nothing, and NeMo would fetch or fail."""
    shutil.rmtree(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(_hf_cache(tmp_path / "hf", snapshot=False))


def _break_git(tmp_path: Path, env: dict) -> None:
    """An untracked file under bench/src/, with git configured to hide untracked files:
    the refusal must see it anyway."""
    shutil.rmtree(tmp_path / "repo")
    env["RUNBOOK_TEST_GIT_DIR"] = str(_git_repo(tmp_path / "repo", dirty="bench/src/new.py"))
    env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="status.showUntrackedFiles",
        GIT_CONFIG_VALUE_0="no",
    )


@needs_venv
@pytest.mark.parametrize(
    ("breaker", "message"),
    [
        (_break_uuid, "not GPU-b43f9262-f250-444a-bfbf-461dd3500f1e"),
        (_break_refs, "refs/main"),
        (_break_blob, "not to its blob name"),
        (_break_xet_blob, "not to its blob name"),
        (_break_resolution, "huggingface_hub resolves"),
        (_break_git, "uncommitted changes under src/ or bench/src/"),
    ],
    ids=[
        "wrong-uuid",
        "wrong-refs-main",
        "blob-hash-mismatch",
        "xet-blob-hash-mismatch",
        "unresolved",
        "dirty-bench-src",
    ],
)
def test_preflight_refuses_the_wrong_card_checkpoint_or_code(
    tmp_path: Path, breaker, message: str
) -> None:
    env = _rehearsal_env(tmp_path, _free_port(), 4)
    breaker(tmp_path, env)
    done = _run("fixed-churn", env=env)
    assert done.returncode == 2 and message in done.stderr, done.stderr
    assert not (_run_dir(tmp_path) / "fixed-churn").exists()  # refused before any arm


def _preflight_runbook(tmp_path: Path) -> dict:
    return json.loads(next(_run_dir(tmp_path).glob("preflight-*/runbook.json")).read_text())


@needs_venv
def test_preflight_records_whether_head_holds_the_runbook_that_ran(tmp_path: Path) -> None:
    """Run from a repository whose HEAD holds this exact runbook, then from one whose
    working copy differs from HEAD. The card check stops both at the end of preflight."""
    copy = _repo_copy(tmp_path / "held" / "copy") / "scripts" / SCRIPT.name
    env = _rehearsal_env(tmp_path / "held", _free_port(), 4)
    _break_uuid(tmp_path, env)
    done = _run("fixed-churn", env=env, script=copy)
    assert done.returncode == 2 and "not GPU-b43f9262" in done.stderr, done.stderr
    doc = _preflight_runbook(tmp_path / "held")
    assert doc["committed_at_head"] is True
    assert doc["sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()

    copy = _repo_copy(tmp_path / "edited" / "copy") / "scripts" / SCRIPT.name
    copy.write_text(copy.read_text() + "# an uncommitted edit\n")
    env = _rehearsal_env(tmp_path / "edited", _free_port(), 4)
    _break_uuid(tmp_path, env)
    done = _run("fixed-churn", env=env, script=copy)
    assert done.returncode == 2, done.stderr
    doc = _preflight_runbook(tmp_path / "edited")
    assert doc["committed_at_head"] is False
    assert doc["sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()


@needs_venv
@pytest.mark.parametrize(
    ("mode", "message", "server_started"),
    [
        ("busy", "is in use; refusing to start", False),
        ("foreign", "a process other than the server", True),
        ("absent", "does not show the server", True),
    ],
)
def test_the_card_must_hold_the_server_and_nothing_else(
    tmp_path: Path, mode: str, message: str, server_started: bool
) -> None:
    port = _free_port()
    done = _run("fixed-churn", env=_rehearsal_env(tmp_path, port, 34, smi_mode=mode), timeout=120)
    assert done.returncode == 2 and message in done.stderr, done.stderr[-3000:]
    arm = _run_dir(tmp_path) / "fixed-churn"
    assert (arm / "server.log").exists() is server_started
    assert not (arm / "gate.log").exists()
    assert _port_free(port)  # the server it started was stopped


@needs_venv
@pytest.mark.skipif(shutil.which("ss") is None, reason="needs ss")
def test_a_listener_that_is_not_the_server_is_refused(tmp_path: Path) -> None:
    port = _free_port()
    ss_dir = _shim("ss-stub", STUB_SS, filename="ss")
    env = _rehearsal_env(tmp_path, port, 34, STUB_SS_REAL=shutil.which("ss"))
    env["PATH"] = f"{ss_dir}{os.pathsep}{env['PATH']}"
    done = _run("fixed-churn", env=env, timeout=120)
    assert done.returncode == 2 and "does not belong to the server just started" in done.stderr, (
        done.stderr[-3000:]
    )
    assert not (_run_dir(tmp_path) / "fixed-churn" / "gate.log").exists()
    assert _port_free(port)


@needs_venv
def test_a_server_not_running_with_this_repositorys_path_is_refused(tmp_path: Path) -> None:
    env = _rehearsal_env(tmp_path, _free_port(), 34)
    env["RUNBOOK_VENV"] = str(_wrapped_venv(tmp_path / "venv", server_path_suffix="/nonexistent"))
    done = _run("fixed-churn", env=env, timeout=120)
    assert done.returncode == 2 and "the server runs with PYTHONPATH" in done.stderr, done.stderr[
        -3000:
    ]
    environ = json.loads((_run_dir(tmp_path) / "fixed-churn" / "server-environ.json").read_text())
    assert environ["environ"]["PYTHONPATH"] == f"{PYTHONPATH}:/nonexistent"


# --- rehearsals over the fake ------------------------------------------------------------


@needs_venv
def test_a_port_already_accepting_is_refused_before_a_server_starts(tmp_path: Path) -> None:
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        done = _run("fixed-churn", env=_rehearsal_env(tmp_path, port, 34))
    assert done.returncode == 2 and "already accepts" in done.stderr, done.stderr
    assert not (_run_dir(tmp_path) / "fixed-churn" / "server.log").exists()


@needs_venv
def test_a_gate_that_writes_no_record_is_no_verdict(tmp_path: Path) -> None:
    """A corpus that cannot reach the max level is refused by the gate before a byte is
    sent: exit 2, no record, no finals, and the runbook must read that as no verdict."""
    env = _rehearsal_env(tmp_path, _free_port(), 34, max_level="40", RUNBOOK_TEST_SMOKE_BURST="4")
    done = _run("fixed-const", env=env, timeout=120)
    assert done.returncode == 2, done.stdout + done.stderr
    arm = _arm(tmp_path, "fixed-const")
    assert arm["record_summary"]["record_present"] is False
    assert arm["gate_exit"] == 2
    summary = _summary(tmp_path)
    assert summary["exit_code"] == 2 and "no verdict for ['fixed-const']" in summary["reading"]


@needs_venv
def test_a_smoke_that_sees_refusals_stops_the_run_before_the_gate(tmp_path: Path) -> None:
    """Bucket 8 cannot hold a burst of 34: the server refuses, the smoke reads it on
    /admission, and the long gate never starts."""
    port = _free_port()
    done = _run("fixed-churn", env=_rehearsal_env(tmp_path, port, 34, bucket="8"), timeout=120)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "stopping before the long gate" in done.stderr and "refused_total rose" in done.stderr
    arm = _run_dir(tmp_path) / "fixed-churn"
    assert json.loads((arm / "smoke.json").read_text())["phases"]["burst"]["errors"] > 0
    assert not (arm / "gate.log").exists()
    assert _port_free(port)


@needs_venv
def test_refusals_during_the_gate_take_the_verdict_away(tmp_path: Path) -> None:
    """A smoke of 4 fits bucket 8; the gate's 32a level does not. The monitor stops the
    gate once it sees the refusal (level 1 alone gives it seconds to start, and three
    levels of refusals follow), and the reading after the gate says so again."""
    env = _rehearsal_env(tmp_path, _free_port(), 34, bucket="8", RUNBOOK_TEST_SMOKE_BURST="4")
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    problems = arm["operational_problems"]
    assert any(
        p.startswith("the monitor stopped the gate: refused_total rose") for p in problems
    ), problems
    assert any("refused_total rose during the gate" in p for p in problems), problems
    assert arm["gate_exit"] != 0 and arm["smoke"]["problems"] == []


@needs_venv
@pytest.mark.parametrize(
    ("mode", "message", "where"),
    [
        # Seen by the reading after the gate (and by the monitor's polls during it).
        ("late", "on card 3 after the gate", "operational_problems"),
        # Seen only while the gate ran: the monitor's polls are what catch it.
        ("transient", "polls during the gate", "operational_problems"),
        # ... at a single one of them.
        ("blip", "at 1 of", "operational_problems"),
        # nvidia-smi answered nothing while the gate ran: what held the card is unobserved.
        ("blind", "nvidia-smi failed at", "operational_problems"),
        ("vanish", "no longer shows the server", "operational_problems"),
        # Two things wrong at every poll: each is a problem of its own.
        ("swap", "did not show the server on card 3 at", "operational_problems"),
        ("warn", "not a supported look-ahead", "invalidating_problems"),
    ],
)
def test_what_happens_on_the_card_during_the_gate_takes_the_verdict_away(
    tmp_path: Path, mode: str, message: str, where: str
) -> None:
    done = _run(
        "fixed-const", env=_rehearsal_env(tmp_path, _free_port(), 34, smi_mode=mode), timeout=180
    )
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    assert any(message in p for p in arm[where]), arm
    assert arm["record_summary"]["verdict"] == "invariant"  # the gate itself passed
    assert "no verdict for ['fixed-const']" in _summary(tmp_path)["reading"]
    # Every problem the monitor's polls showed is one of the arm's.
    monitor_problems = arm["monitor_summary"]["problems"]
    assert all(p in arm["operational_problems"] for p in monitor_problems), arm
    if mode == "swap":
        assert len(monitor_problems) == 2, monitor_problems
    if mode in ("transient", "blip", "blind"):
        assert not any("after the gate" in p for p in arm["operational_problems"])
    if mode == "blip":
        assert arm["monitor_summary"]["card_polls_with_foreign"] == 1
        assert arm["monitor_summary"]["card_polls"] > 1
    if mode == "blind":
        assert arm["monitor_summary"]["card_poll_errors"] >= 1
        assert arm["monitor_summary"]["card_polls_with_foreign"] == 0


@needs_venv
def test_a_gate_that_hits_the_time_limit_takes_the_verdict_away(tmp_path: Path) -> None:
    env = _rehearsal_env(tmp_path, _free_port(), 34, RUNBOOK_GATE_TIMEOUT_S="1")
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    assert arm["gate_exit"] == 124
    assert "the gate hit the 1 s runbook limit" in arm["operational_problems"], arm


@needs_venv
def test_a_server_emitting_wrong_text_gets_no_verdict(tmp_path: Path) -> None:
    """Every reference says "noise": the fake's text is the same at every level, so the
    digests agree, and only the level-1 WER sees that it is not a transcript."""
    env = _rehearsal_env(tmp_path, _free_port(), 34, text="noise")
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    rs = _arm(tmp_path, "fixed-const")["record_summary"]
    assert rs["verdict"] == "invariant" and rs["level1_wer"]["wer"] == 1.0
    assert any("over the sanity bound" in p for p in rs["problems"]), rs["problems"]


@needs_venv
@pytest.mark.parametrize(("wrong", "code"), [(16, 3), (17, 3), (18, 2)])
def test_the_level_one_wer_bound_is_one_half(tmp_path: Path, wrong: int, code: int) -> None:
    """Of 34 one-word references, `wrong` are a word the fake never says: level-1 WER 16/34
    (0.47) and 17/34 (exactly 0.5: the bound itself is not over it) keep the arm's verdict
    (exit 3: its twin did not run), 18/34 (0.53) takes it away. The bound is the runbook's
    own, read end to end, not one this test passes in."""
    env = _rehearsal_env(tmp_path, _free_port(), 34, wrong=wrong)
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == code, done.stdout[-3000:] + done.stderr[-3000:]
    rs = _arm(tmp_path, "fixed-const")["record_summary"]
    assert rs["verdict"] == "invariant"
    assert rs["level1_wer"]["wer"] == round(wrong / 34, 6) and rs["level1_wer"]["bound"] == 0.5
    over = [p for p in rs["problems"] if "is over the sanity bound 0.5" in p]
    assert len(over) == (1 if wrong == 18 else 0) and len(rs["problems"]) == len(over), rs


@needs_venv
@pytest.mark.parametrize("bench_exit", [1, 2])
def test_a_gate_exit_its_record_does_not_state_takes_the_verdict_away(
    tmp_path: Path, bench_exit: int
) -> None:
    """verbatim-bench writes a whole, valid, invariant record and then exits 1 (divergent,
    or an argparse usage error) or 2 (no verdict): the record's exit_code 0 is not what the
    process returned, so its verdict is not read."""
    env = _rehearsal_env(tmp_path, _free_port(), 34)
    env["RUNBOOK_VENV"] = str(_wrapped_venv(tmp_path / "venv", bench_exit=bench_exit))
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    rs = arm["record_summary"]
    assert arm["gate_exit"] == bench_exit and rs["gate_exit_checked"] == bench_exit
    assert rs["verdict"] == "invariant" and rs["exit_code_in_record"] == 0
    message = f"the gate exited {bench_exit} and its record says 0"
    assert message in rs["problems"], rs["problems"]
    assert "no verdict for ['fixed-const']" in _summary(tmp_path)["reading"]


@needs_venv
def test_a_monitor_killed_partway_through_the_gate_takes_the_verdict_away(tmp_path: Path) -> None:
    """The stub nvidia-smi SIGKILLs the monitor at its second poll of the gate, so its polls
    cover the gate's first moments only. The gate itself passes. What the polls did not
    cover, and the monitor's own exit status, are each a problem of the arm."""
    env = _rehearsal_env(tmp_path, _free_port(), 34, smi_mode="killmon")
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    monitor = arm["monitor_summary"]
    gate_s = arm["t_gate_end"] - arm["t_gate_start"]
    assert monitor["unwatched_bound_s"] == 3.6 and gate_s > 3.6 + 2, gate_s  # the test's premise
    assert arm["monitor_exit"] == 137 and monitor["monitor_exit"] == 137
    assert 1 <= monitor["timestamped_polls"] <= 2, monitor
    # The last stretch runs from the last poll the monitor wrote to the gate's end.
    assert monitor["unwatched_stretches_over_bound"][-1][1] == round(gate_s, 3), monitor
    problems = arm["operational_problems"]
    assert any(p.startswith("the monitor observed nothing for") for p in problems), problems
    assert any(p.startswith("the monitor's exit status is 137, not 0") for p in problems), problems
    assert arm["record_summary"]["verdict"] == "invariant"
    assert "no verdict for ['fixed-const']" in _summary(tmp_path)["reading"]


#: Run before the helper's monitor: the gate never looks gone, so the monitor never stops.
MONITOR_BLIND_TO_THE_GATE_ENDING = """
import os as _os
_kill = _os.kill
_os.kill = lambda pid, sig: None if sig == 0 else _kill(pid, sig)
"""


@needs_venv
def test_a_monitor_still_polling_after_the_gate_is_stopped_and_says_so(tmp_path: Path) -> None:
    """A monitor that does not see the gate end is waited for as long as the unwatched
    bound, then stopped by the runbook; its exit status says so and takes the verdict away."""
    env = _rehearsal_env(tmp_path, _free_port(), 34)
    env["RUNBOOK_VENV"] = str(
        _wrapped_venv(
            tmp_path / "venv", helper_prelude={"monitor": MONITOR_BLIND_TO_THE_GATE_ENDING}
        )
    )
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    monitor = arm["monitor_summary"]
    assert arm["monitor_exit"] == 143 and monitor["monitor_exit"] == 143
    assert (
        monitor["monitor_end"] == "still running 3.6 s after the gate ended; stopped by the runbook"
    )
    assert monitor["unwatched_stretches_over_bound"] == []  # it did watch the whole gate
    assert [p for p in arm["operational_problems"] if p.startswith("the monitor")] == [
        "the monitor's exit status is 143 (still running 3.6 s after the gate ended; stopped by "
        "the runbook), not 0: it did not run to the gate's end and stop by itself, so what it "
        "did not write was not observed"
    ]
    assert arm["t_stopped"] - arm["t_gate_end"] >= 3.6


#: Run in the server process before `verbatim serve`: once the runbook has begun summarising
#: the monitor (its monitor-summary.json exists), after the gate and after the monitor's
#: last poll, the server exits.
SERVER_EXITS_AFTER_THE_MONITOR = """
import os as _os, pathlib as _pathlib, threading as _threading, time as _time
def _exit_after_the_monitor():
    root = _pathlib.Path(_os.environ["STUB_SMI_ROOT"])
    while not any(root.rglob("monitor-summary.json")):
        _time.sleep(0.01)
    _os._exit(0)
_threading.Thread(target=_exit_after_the_monitor, daemon=True).start()
"""

#: Run before the helper's `http` when it writes admission-after.json: nothing answers.
ADMISSION_UNANSWERED_AFTER_THE_GATE = """
import sys as _sys, urllib.request as _request
if _sys.argv[3:4] and _sys.argv[3].endswith("admission-after.json"):
    def _unanswered(url, timeout=None):
        raise OSError("/admission did not answer (runbook test)")
    _request.urlopen = _unanswered
"""

#: Run before the helper's `http` when it writes admission-after.json: the server's answer,
#: with degradation_level 1.
ADMISSION_DEGRADED_AFTER_THE_GATE = """
import io as _io, json as _json, sys as _sys, urllib.request as _request
if _sys.argv[3:4] and _sys.argv[3].endswith("admission-after.json"):
    _urlopen = _request.urlopen
    class _Answer(_io.BytesIO):
        status = 200
    def _degraded(url, timeout=None):
        with _urlopen(url, timeout=timeout) as answer:
            doc = _json.loads(answer.read())
        doc["degradation_level"] = 1
        return _Answer(_json.dumps(doc).encode())
    _request.urlopen = _degraded
"""


@needs_venv
@pytest.mark.parametrize(
    ("wrap", "smi_mode", "problem"),
    [
        (
            {"server_prelude": SERVER_EXITS_AFTER_THE_MONITOR},
            "idle",
            "the server was not running when the runbook read it after the gate",
        ),
        ({}, "blind_after", "nvidia-smi could not list card 3's compute processes after the gate"),
        (
            {"helper_prelude": {"http": ADMISSION_UNANSWERED_AFTER_THE_GATE}},
            "idle",
            "no /admission reading after the gate",
        ),
        (
            {"helper_prelude": {"http": ADMISSION_DEGRADED_AFTER_THE_GATE}},
            "idle",
            "degradation_level 1 after the gate: the server was holding admissions",
        ),
    ],
    ids=["server-gone", "card-unlisted", "admission-unanswered", "admission-degraded"],
)
def test_what_the_runbook_reads_after_the_gate_can_take_the_verdict_away(
    tmp_path: Path, wrap: dict, smi_mode: str, problem: str
) -> None:
    """The gate passes, its record passes every check and the monitor saw nothing wrong:
    the runbook's own reading after the gate is the only thing that sees each of these,
    and each alone takes the arm's verdict away."""
    env = _rehearsal_env(tmp_path, _free_port(), 34, smi_mode=smi_mode)
    if wrap:
        env["RUNBOOK_VENV"] = str(_wrapped_venv(tmp_path / "venv", **wrap))
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    arm = _arm(tmp_path, "fixed-const")
    assert arm["operational_problems"] == [problem], arm["operational_problems"]
    assert arm["invalidating_problems"] == [] and arm["monitor_summary"]["problems"] == [], arm
    rs = arm["record_summary"]
    assert rs["verdict"] == "invariant" and rs["problems"] == [], rs
    if problem.startswith("degradation_level"):
        assert arm["admission_after"]["degradation_level"] == 1
    assert "no verdict for ['fixed-const']" in _summary(tmp_path)["reading"]


#: Run in the server process before `verbatim serve`: its /readyz reports that the built
#: encoder holds attention context [70, 13], whatever the flags asked for.
ENCODER_BUILT_WITH_ANOTHER_CONTEXT = """
import verbatim.serve
from verbatim.pipelines.observed import ObservedFacts
verbatim.serve.observe = lambda pipeline: ObservedFacts(att_context_size=(70, 13))
"""

#: Run before the helper's derive_spec: the server's code builds a spec with decoder graphs.
SPEC_WITH_DECODER_GRAPHS = """
import verbatim.cli as _cli
_Spec = _cli.NeMoPipelineSpec
_cli.NeMoPipelineSpec = lambda **kw: _Spec(**{**kw, "use_cuda_graph_decoder": True})
"""


@needs_venv
@pytest.mark.parametrize(
    ("wrap", "message", "server_started"),
    [
        # Contract C5: the flags, the banner and the digests are the arm's; what the built
        # encoder holds is not.
        (
            {"server_prelude": ENCODER_BUILT_WITH_ANOTHER_CONTEXT},
            "FATAL: the server is not what this arm asked for: "
            "/readyz observed.att_context_size=[70, 13], expected [70, 1]",
            True,
        ),
        # The only check of the arm's padding in production (padding is not on /readyz).
        ({"server_stdout_discarded": True}, "FATAL: banner: ", True),
        # The server's own code would build something other than the arm.
        (
            {"helper_prelude": {"derive_spec": SPEC_WITH_DECODER_GRAPHS}},
            "FATAL: the server's code would not build this arm: "
            "spec use_cuda_graph_decoder=True, expected False",
            False,
        ),
    ],
    ids=["readyz-observed", "banner", "spec"],
)
def test_a_server_that_is_not_the_arm_is_refused_before_the_gate(
    tmp_path: Path, wrap: dict, message: str, server_started: bool
) -> None:
    port = _free_port()
    env = _rehearsal_env(tmp_path, port, 34)
    env["RUNBOOK_VENV"] = str(_wrapped_venv(tmp_path / "venv", **wrap))
    done = _run("fixed-const", env=env, timeout=180)
    assert done.returncode == 2 and message in done.stderr, done.stderr[-3000:]
    arm = _run_dir(tmp_path) / "fixed-const"
    assert (arm / "server.log").exists() is server_started
    assert not (arm / "gate.log").exists() and not (arm / "arm.json").exists()
    assert _port_free(port)


@needs_venv
def test_a_server_importing_another_checkouts_code_is_refused_before_the_gate(
    tmp_path: Path,
) -> None:
    """Contract C7, end to end. The server keeps this repository's PYTHONPATH, which the
    runbook reads from /proc and accepts, but before anything else it puts a copy of the
    verbatim package elsewhere ahead of it on sys.path, as a .pth file, an editable
    install's finder or a sitecustomize can. It runs, and its /readyz names the copy it
    imported: the arm is refused there, before the smoke and the gate."""
    elsewhere = tmp_path / "other-checkout" / "src"
    shutil.copytree(
        WORKTREE / "src" / "verbatim",
        elsewhere / "verbatim",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    port = _free_port()
    env = _rehearsal_env(tmp_path, port, 34)
    prelude = f"import sys as _sys\n_sys.path.insert(0, {str(elsewhere)!r})\n"
    env["RUNBOOK_VENV"] = str(_wrapped_venv(tmp_path / "venv", server_prelude=prelude))
    done = _run("fixed-const", env=env, timeout=180)
    copy = str((elsewhere / "verbatim").resolve())
    message = (
        "FATAL: the server is not what this arm asked for: "
        f"/readyz code.verbatim_path={copy!r}, expected {SERVER_CODE['verbatim_path']!r}: "
        "the server does not run this repository's code"
    )
    assert done.returncode == 2 and message in done.stderr, done.stderr[-3000:]
    arm = _run_dir(tmp_path) / "fixed-const"
    assert json.loads((arm / "readyz-before.json").read_text())["code"]["verbatim_path"] == copy
    # What the PYTHONPATH check sees is right: it alone could not tell.
    environ = json.loads((arm / "server-environ.json").read_text())["environ"]
    assert environ["PYTHONPATH"] == PYTHONPATH
    assert (arm / "facts-check.txt").read_text().splitlines() == [message.split(": ", 2)[2]]
    assert not (arm / "smoke.json").exists() and not (arm / "gate.log").exists()
    assert not (arm / "arm.json").exists() and _port_free(port)


@needs_venv
def test_a_later_arm_that_fails_still_gets_the_earlier_arms_summarised(tmp_path: Path) -> None:
    """The first arm finishes (with a co-tenant, so no verdict); the second refuses to start
    on the busy card. The run is 2, and the first arm is in summary.json."""
    done = _run(
        "fixed-const",
        "ragged-const",
        env=_rehearsal_env(tmp_path, _free_port(), 34, smi_mode="late"),
        timeout=300,
    )
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    summary = _summary(tmp_path)
    assert summary["exit_code"] == 2
    assert [row["arm"] for row in summary["arms"]] == ["fixed-const"]
    assert summary["stopped"].startswith("arm ragged-const: card 3 is in use")
    assert "the run stopped before it finished" in summary["reading"]
    run = _run_dir(tmp_path)
    assert "is in use" in (run / "ragged-const" / "fatal.txt").read_text()
    assert not (run / "ragged-const" / "server.log").exists()


@needs_venv
def test_a_fixed_divergence_found_before_a_later_failure_still_exits_1(tmp_path: Path) -> None:
    """The run directory is continued (RUNBOOK_RUN_ID) from an invocation whose fixed-churn
    arm diverged. This invocation's second arm dies; the summary still reads 1."""
    run = _run_dir(tmp_path)
    run.mkdir(parents=True)
    _arm_doc(
        run,
        "fixed-churn",
        "divergent",
        digests={"1": "one", "32a": "one", "32b": "one", "max": "x"},
        differing={"32a_vs_1": 0, "32b_vs_1": 0, "max_vs_1": 3, "32b_vs_32a": 0},
    )
    done = _run(
        "fixed-const",
        "ragged-const",
        env=_rehearsal_env(tmp_path, _free_port(), 34, smi_mode="late"),
        timeout=300,
    )
    assert done.returncode == 1, done.stdout[-3000:] + done.stderr[-3000:]
    summary = _summary(tmp_path)
    assert summary["exit_code"] == 1
    assert "DIVERGED within ['fixed-churn']" in summary["reading"]
    assert "the run stopped before it finished: arm ragged-const" in summary["reading"]


@needs_venv
@pytest.mark.parametrize(
    ("mode", "edited", "message", "first_arm_verdict"),
    [
        ("edit", "src/a.py", "uncommitted changes under src/ or bench/src/:  M src/a.py", False),
        # A file git does not track, new while the gate ran, with git set to hide untracked
        # files: a new module or package can shadow a tracked one for the next arm's server.
        (
            "edit",
            "bench/src/new_module.py",
            "uncommitted changes under src/ or bench/src/: ?? bench/src/new_module.py",
            False,
        ),
        ("commit", "src/a.py", "HEAD is ", False),
        ("edit_between", "src/a.py", "before arm ragged-const's server started", True),
    ],
    ids=["edit-tracked", "new-untracked", "commit", "edit-between-arms"],
)
def test_the_code_changing_during_a_run_stops_it(
    tmp_path: Path, mode: str, edited: str, message: str, first_arm_verdict: bool
) -> None:
    env = _rehearsal_env(tmp_path, _free_port(), 34, smi_mode=mode)
    env["STUB_SMI_EDIT"] = str(Path(env["RUNBOOK_TEST_GIT_DIR"]) / edited)
    env["STUB_SMI_COMMIT"] = env["RUNBOOK_TEST_GIT_DIR"]
    if not (Path(env["RUNBOOK_TEST_GIT_DIR"]) / edited).exists():
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="status.showUntrackedFiles",
            GIT_CONFIG_VALUE_0="no",
        )
    done = _run("fixed-const", "ragged-const", env=env, timeout=300)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    assert message in done.stderr, done.stderr[-3000:]
    run = _run_dir(tmp_path)
    assert not (run / "ragged-const" / "server.log").exists()  # the second arm never started
    summary = _summary(tmp_path)
    assert summary["exit_code"] == 2 and summary["stopped"]
    row = next(r for r in summary["arms"] if r["arm"] == "fixed-const")
    if first_arm_verdict:
        # Changed after the first arm finished: that arm stands, the next never starts.
        assert row["verdict"] == "invariant", row
    else:
        assert row["verdict"].startswith("NO VERDICT"), row
        invalidating = _arm(tmp_path, "fixed-const")["invalidating_problems"]
        assert any(message in p for p in invalidating), invalidating


@needs_venv
def test_the_runbook_changing_during_a_run_stops_it(tmp_path: Path) -> None:
    copy = _repo_copy(tmp_path / "copy") / "scripts" / SCRIPT.name
    env = _rehearsal_env(tmp_path, _free_port(), 34, smi_mode="edit", STUB_SMI_EDIT=str(copy))
    done = _run("fixed-const", env=env, timeout=180, script=copy)
    assert done.returncode == 2, done.stdout[-3000:] + done.stderr[-3000:]
    invalidating = _arm(tmp_path, "fixed-const")["invalidating_problems"]
    assert any("this runbook now hashes to" in p for p in invalidating), invalidating


@needs_venv
@pytest.mark.skipif(shutil.which("ss") is None, reason="needs ss")
def test_a_rehearsal_over_the_fake_runs_the_whole_mechanism(tmp_path: Path) -> None:
    port = _free_port()
    env = _rehearsal_env(tmp_path, port, 34, HF_TOKEN="not-a-real-token")
    done = _run(*ARMS, env=env, timeout=900)
    # The fake is invariant by construction, ragged included, so the controls show no
    # power and the runbook must say 3, naming all six comparisons, not 0.
    assert done.returncode == 3, done.stdout[-3000:] + done.stderr[-3000:]
    run = _run_dir(tmp_path)
    summary = json.loads((run / "summary.json").read_text())
    assert [row["verdict"] for row in summary["arms"]] == ["invariant"] * 4
    assert all(not row["problems"] for row in summary["arms"])
    for fixed in ("fixed-churn", "fixed-const"):
        for level in ("32a", "32b", "max"):
            assert f"{fixed} {level} vs 1" in summary["reading"]
    assert summary["cross_restart_digest_agreement"] == {
        "fixed-churn_vs_fixed-const": {"1": True, "32a": True, "32b": True, "max": True},
        "ragged-churn_vs_ragged-const": {"1": True, "32a": True, "32b": True},
    }
    assert summary["scope"].startswith("EAGER ONLY")
    assert summary["matmul_precision_by_arm"] == dict.fromkeys(ARMS, "high")
    # Each arm checked its server against the production expectation, with only the
    # settings keys the fake cannot serve changed.
    production = _runbook_facts_expect(bucket="64")
    for arm in ARMS:
        assert json.loads((run / arm / "facts-expected.json").read_text()) == {
            **production,
            "pipeline": "fake",
            "precision": "none",
            "execution": "fake",
        }
    sha = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    preflight = next(run.glob("preflight-*"))
    code = json.loads((preflight / "code.json").read_text())
    assert code["verbatim"] == f"{WORKTREE}/src/verbatim/__init__.py"
    assert code["verbatim_bench"] == f"{WORKTREE}/bench/src/verbatim_bench/__init__.py"
    head = _git(Path(env["RUNBOOK_TEST_GIT_DIR"]), "rev-parse", "HEAD")
    versions = {k: v for k, v in code.items() if k.startswith("version_")}
    assert summary["settings_by_arm"] == {
        arm: {
            "bucket": 64,
            "max_level": "34",
            "runbook_sha256": sha,
            "git_head": head,
            "versions": versions,
        }
        for arm in ARMS
    }
    assert (preflight / "runbook.sh").read_bytes() == SCRIPT.read_bytes()
    assert (preflight / "runbook.sha256").read_text().split()[0] == sha
    self_blob = _git(WORKTREE, "hash-object", str(SCRIPT))
    head_blob = subprocess.run(
        ["git", "-C", str(WORKTREE), "rev-parse", "-q", "--verify", f"HEAD:scripts/{SCRIPT.name}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    runbook = json.loads((preflight / "runbook.json").read_text())
    assert runbook["committed_at_head"] is (self_blob == head_blob)
    model = json.loads((preflight / "model.json").read_text())
    assert model["nemo_sha256"] == hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    corpus = json.loads((preflight / "corpus.json").read_text())
    # As on the real corpus: the ids differ, and the record must carry the local one.
    assert corpus["local_prefix_occurrences"] == corpus["utterances"] == 34
    assert corpus["local_id"] != corpus["published_form_id"]
    for arm in ARMS:
        doc = json.loads((run / arm / "arm.json").read_text())
        times = [
            doc[k]
            for k in ("t_start", "t_accept", "t_ready", "t_gate_start", "t_gate_end", "t_stopped")
        ]
        assert times == sorted(times)
        assert doc["gate_exit"] == 0 and doc["server_exit"] == 0
        assert doc["readyz_before"]["execution"] == "fake"
        # What the fake reports it is (contract C3): no encoder, no decoder, so null, and
        # word confidence off.
        assert doc["readyz_before"]["word_confidence"] == "off"
        assert doc["readyz_before"]["observed"] == dict.fromkeys(
            (
                "att_context_size",
                "decoder_step_confidence",
                "decoder_graphs",
                "decoder_word_confidence",
            )
        )
        assert doc["admission_before"]["bucket"] == 64
        assert doc["serve_spec"]["spec"]["att_context"] == [70, 1]
        assert doc["serve_spec"]["budget_ms"] == BUDGET_MS
        assert doc["smoke"]["problems"] == [] and doc["smoke"]["admission_readings"] >= 3
        monitor = doc["monitor_summary"]
        assert monitor["readings"] >= 1 and monitor["card_polls"] >= 1
        assert monitor["card_polls_with_foreign"] == 0 and monitor["card_polls_without_server"] == 0
        assert doc["monitor_exit"] == 0 and monitor["monitor_exit"] == 0
        assert (
            monitor["gate_start"] == doc["t_gate_start"]
            and monitor["gate_end"] == doc["t_gate_end"]
        )
        assert (
            monitor["unwatched_bound_s"] == 3.6 and monitor["unwatched_stretches_over_bound"] == []
        )
        assert monitor["longest_unwatched_s"] <= 3.6
        # What the monitor parsed, as it wrote it, is what the runbook asked for, and every
        # read it made answered well inside its timeout.
        ran = [
            monitor["poll_sleep_parsed_s"],
            monitor["card_query_timeout_parsed_s"],
            monitor["admission_read_timeout_parsed_s"],
        ]
        asked = [
            monitor["poll_sleep_requested_s"],
            monitor["card_query_timeout_requested_s"],
            monitor["admission_read_timeout_requested_s"],
        ]
        assert ran == asked == [0.2, 2.0, 1.0], monitor
        assert monitor["monitor_parsed"]["server_pid"] == str(doc["server_pid"])
        assert monitor["monitor_parsed"]["gpu_index"] == "3"
        assert monitor["longest_admission_read_s"] < 1.0 and monitor["longest_card_query_s"] < 2.0
        # Contract C7: the server said it runs this checkout's code.
        assert doc["readyz_before"]["code"] == SERVER_CODE
        assert doc["runbook"]["sha256"] == sha
        assert doc["git_head"] == head
        assert doc["operational_problems"] == [] and doc["invalidating_problems"] == []
        # The server's own environment, read from /proc: its overrides and what it inherited.
        environ = doc["server_environ"]
        assert environ["source"] == f"/proc/{doc['server_pid']}/environ"
        assert environ["environ"]["PYTHONPATH"] == PYTHONPATH
        assert environ["environ"]["HF_HUB_OFFLINE"] == "1"
        assert environ["environ"]["CUDA_VISIBLE_DEVICES"] == ""
        assert environ["environ"]["HF_TOKEN"] == "<redacted>"
        rs = doc["record_summary"]
        assert rs["timings_present"] is True
        assert rs["gate_exit_checked"] == 0
        assert rs["corpus_id"] == corpus["local_id"]
        assert rs["finals_digest_match"] == dict.fromkeys(("1", "32a", "32b", "max"), True)
        assert rs["level1_wer"]["wer"] == 0.0 and rs["level1_wer"]["streams_with_a_final"] == 34
        assert (run / f"invariance-{arm}.json").is_file() and (run / f"finals-{arm}.json").is_file()
        assert "[verbatim] stopped" in (run / arm / "server.log").read_text()
        assert (run / arm / "admission-monitor.jsonl").read_text().strip()
        smoke = json.loads((run / arm / "smoke.json").read_text())
        assert (
            smoke["phases"]["solo"]["sessions"] >= 1 and smoke["phases"]["burst"]["sessions"] == 34
        )
    # Every per-card question, the runbook's own and its monitor's, was about card 3 (the
    # stub fails any other, which would have been a problem above), and both asked.
    calls = Path(env["STUB_SMI_LOG"]).read_text().splitlines()
    apps = [line for line in calls if "--query-compute-apps=" in line]
    assert apps and all(" -i 3 " in f" {line} " for line in apps), calls
    assert {line.split()[0] for line in apps} == {"runbook", "monitor"}, calls
    churned = json.loads((run / "fixed-churn" / "record-summary.json").read_text())
    assert churned["max_level_admission_occupancy_range"][1] > 1
    const = json.loads((run / "fixed-const" / "record-summary.json").read_text())
    assert "max_level_admission_occupancy_range" not in const
    assert _port_free(port)
