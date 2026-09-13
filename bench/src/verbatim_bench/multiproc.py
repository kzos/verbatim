# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""One rung, several event loops, one measurement window.

A single event loop loses its own send schedule long before the MVP bar. Measured on
this box on 2026-09-11 against the null server, the generator's pacing slip at the 99th
percentile is 2.5 ms at 16 streams, 3.9 to 5.4 ms at 32 with one window in three over the
frozen 5.0 ms tolerance, and 33 to 38 ms at 128, while the bar is roughly 112 concurrent
streams. An instrument that gives out at a third of what it has to measure cannot show a
server clearing the bar even if the server clears it. So the sessions of one rung are
spread across several processes, each running the same event loop out of ``pace.py``.

**The processes are the easy half; the aggregation is the work.** Four things have to
hold or the split rung is a different measurement wearing the same name:

* **One window, not N windows.** Every process opens its window at the same instant on
  the same clock and closes it ``window_s`` later, because a percentile pooled over
  windows that did not overlap is a number no load ever produced. The window opens at a
  deadline: the coordinator names one instant and broadcasts it, and each process records
  that instant verbatim rather than the clock it read after waiting for it. The instant
  is on ``CLOCK_MONOTONIC``, which Linux keeps per boot and not per process -- measured
  here, and checked at run time by ``require_shared_monotonic`` rather than assumed.
* **One warm-up, not N warm-ups.** Convergence is a property of the rung, so it is
  decided here, on the rung's pooled samples, and not in whichever process happens to be
  holding a quarter of them. Each process reports the raw latencies of each reading
  interval -- raw, because the reading is the 95th percentile of the pooled sample and
  percentiles do not pool -- and the coordinator takes the rung's reading over all of
  them. Four processes each declaring themselves settled on their own quarter at four
  different moments would open the rung's window at a moment no pooled reading was ever
  taken at, and the statistic the warm-up tested would not be the statistic the window
  reports.
* **One host record, not N.** The box is one box. The record is taken once, in the
  coordinator, over the shared window. What does change is the client's own CPU: it now
  lives in the children, so the sampler is told every child pid and the validity gate
  keeps its teeth instead of reading a parent that did nothing.
* **One slot plan, however it is split.** A rung slot is named by its index in the rung,
  never by its index in a process's own share, so the ramp position, the utterance and
  the frame jitter of slot 37 are the same whether one process drives all 128 slots or
  eight drive sixteen each. Slots are dealt round-robin rather than in blocks, so every
  process holds slots from the whole length of the ramp and they all come live together;
  contiguous blocks would put one process at the front of a sixty-second ramp and another
  at the back, and their reading intervals would not line up.

The coordinator forks. ``spawn`` and ``forkserver`` both re-import the parent's
``__main__`` in the child, which under a console script or a test runner re-runs it, so
they are not usable here; Linux is the platform this project measures on and ``fork`` is
what it has. Forking a process that has threads can leave a child holding a lock nobody
will release, and CPython warns about it; the children are forked before the coordinator
starts any thread of its own, and every wait here is against an absolute deadline, so a
child that never answers costs its own share and a bounded wait rather than the rung's
wall clock. It does cost the rung: a process that returns no result held slots that
stopped sending, so the load was under N for part of its own window, and
``run_sharded_load`` raises instead of combining what is left.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final

# `_readings_agree` is imported rather than restated: the rung's convergence test and a
# single process's are the same test, and two copies of it would be two protocols.
from verbatim_bench.pace import LoadSpec, WindowHooks, _readings_agree, run_load, spec_dict
from verbatim_bench.results import RunResult, percentile

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.context import BaseContext, Process

__all__ = [
    "LIVE_GRACE_S",
    "RESULT_GRACE_S",
    "WINDOW_OPEN_LEAD_S",
    "ShardPlan",
    "ShardedRun",
    "combine_shards",
    "plan_shards",
    "require_shared_monotonic",
    "run_sharded_load",
    "shard_spec",
]

#: How far ahead of now the coordinator sets the window-open deadline once the rung has
#: settled. It is the time a child needs to be told; a child told late still records the
#: same instant and filters its own samples by it, so the lead costs the warm-up a
#: fraction of a second and buys every process the same first sample.
WINDOW_OPEN_LEAD_S: Final = 0.5
#: How long past the end of the ramp the coordinator waits for every process to report
#: its slots live, and how far past the warm-up cap a child waits for a decision that
#: should already have arrived. Both are backstops: the coordinator's own broadcast is
#: the mechanism, and these only stop a rung hanging on a process that died.
LIVE_GRACE_S: Final = 60.0
#: How long past the close of the window the coordinator waits for a process to send its
#: result. A slot in flight at the close is allowed to finish, so this has to cover one
#: whole session plus the client's own final wait.
RESULT_GRACE_S: Final = 300.0


class ShardedLoadError(RuntimeError):
    """The rung could not be run across processes at all, as opposed to running badly."""


def require_shared_monotonic() -> None:
    """Refuse to split a rung on a platform whose monotonic clock is per process.

    The whole scheme rests on one instant meaning the same thing in every process. On
    Linux ``time.monotonic()`` is ``clock_gettime(CLOCK_MONOTONIC)``, which is per boot
    and not per process, so a child's reading falls between two of the parent's. Where
    that is not what the clock is, a shared deadline is meaningless and the right answer
    is to say so rather than to pool samples from windows that only look aligned.
    """
    implementation = time.get_clock_info("monotonic").implementation
    if implementation != "clock_gettime(CLOCK_MONOTONIC)":
        raise ShardedLoadError(
            f"the monotonic clock here is {implementation!r}, which is not known to be "
            "shared between processes; a rung cannot be split across processes on it"
        )


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """One process's share of a rung: which of the rung's slots it drives."""

    index: int
    slots: tuple[int, ...]

    @property
    def sessions(self) -> int:
        return len(self.slots)


def plan_shards(sessions: int, processes: int) -> tuple[ShardPlan, ...]:
    """Deal the rung's slots round-robin across the processes that will drive them.

    Round-robin, not contiguous blocks, so every process holds slots from the whole
    length of the ramp and they all reach their last slot at about the same moment. A
    process holding the first sixteen slots of a sixty-second ramp would go live a minute
    before one holding the last sixteen, and the warm-up readings the coordinator pools
    would cover intervals a minute apart.

    More processes than streams is not an error, it is just arithmetic: a rung of three
    streams gets three processes however many were asked for, because an event loop with
    no session in it is not load.
    """
    if sessions < 1:
        raise ValueError(f"a rung needs at least one session, got {sessions}")
    if processes < 1:
        raise ValueError(f"a rung needs at least one process, got {processes}")
    used = min(processes, sessions)
    return tuple(
        ShardPlan(index=index, slots=tuple(range(index, sessions, used))) for index in range(used)
    )


def shard_spec(spec: LoadSpec, plan: ShardPlan) -> LoadSpec:
    """The load spec one process runs: its own slots, named in the rung's numbering."""
    return replace(
        spec,
        sessions=plan.sessions,
        slots=plan.slots,
        total_sessions=spec.rung_sessions,
    )


def combine_shards(
    shards: Sequence[RunResult],
    *,
    spec: LoadSpec,
    processes: int,
    readings: Sequence[float | None] = (),
    converged: bool | None = None,
) -> RunResult:
    """Reduce what every process measured to the one rung they measured together.

    Latency samples pool because the sessions pool and every sample carries the moment it
    was matched; ``window_partial_samples`` then takes the ones inside the shared window,
    whichever process they came from. Pacing slip pools the same way, off the same
    sessions, so the validity threshold is applied to the rung's whole load and not to a
    quarter of it. Session counts sum. The spec recorded is the RUNG's, so the row says
    the rung ran ``n`` streams rather than what one process's share of them was.

    The one thing that is checked rather than combined is the window. Every process must
    report the same two instants, to the bit, because they were all told one deadline and
    recorded it verbatim. Anything else means the processes measured different windows,
    and a percentile over those is a number no load produced, so it raises instead.
    """
    if not shards:
        raise ValueError("a rung combined from no processes measured nothing")
    if len(shards) != processes:
        raise ValueError(
            f"this rung was driven by {processes} processes and {len(shards)} of them "
            "returned a result. The missing ones held slots that stopped sending, so the "
            "load was below N for part of the window; combining what is left would report "
            "the rung's stream count over a load that never ran at it."
        )
    opens = {shard.window_open_s for shard in shards}
    closes = {shard.window_close_s for shard in shards}
    if len(opens) != 1 or len(closes) != 1:
        raise ValueError(
            "the processes of one rung did not measure the same window: "
            f"opens={sorted(value for value in opens if value is not None)!r} "
            f"closes={sorted(value for value in closes if value is not None)!r}. "
            "A percentile pooled over windows that did not overlap is not a percentile "
            "of anything the server did, so this is not a rung."
        )
    window_open_s = opens.pop()
    window_close_s = closes.pop()
    sessions = sorted(
        (session for shard in shards for session in shard.sessions),
        key=lambda session: session.session_id,
    )
    starts = [shard.wall_start_s for shard in shards]
    ends = [shard.wall_start_s + shard.wall_clock_s for shard in shards]
    live = [shard.all_live_at_s for shard in shards if shard.all_live_at_s is not None]
    warm_up_ends = [shard.warm_up_end_s for shard in shards if shard.warm_up_end_s is not None]
    started = [shard.started_at for shard in shards if shard.started_at is not None]
    document = spec_dict(spec)
    document["processes"] = int(processes)
    if window_open_s is not None:
        warm_up_end_s: float | None = window_open_s
    else:
        warm_up_end_s = max(warm_up_ends) if len(warm_up_ends) == len(shards) else None
    return RunResult(
        spec_dict=document,
        sessions=sessions,
        wall_clock_s=max(ends) - min(starts),
        corpus_id=shards[0].corpus_id,
        manifest_name=shards[0].manifest_name,
        started_at=min(started) if started else None,
        wall_start_s=min(starts),
        # The rung is at N when its LAST process is, which is the instant every process
        # was handed as the origin of its warm-up, so these agree by construction.
        all_live_at_s=max(live) if len(live) == len(shards) else None,
        warm_up_end_s=warm_up_end_s,
        window_open_s=window_open_s,
        window_close_s=window_close_s,
        warm_up_converged=converged,
        warm_up_readings=tuple(readings),
    )


@dataclass(frozen=True, slots=True)
class ShardedRun:
    """What the processes of one rung reported, and the rung they reported together.

    ``shards`` is kept rather than discarded so a caller -- a test above all -- can ask
    what each process measured before the combination hides the difference.
    """

    spec: LoadSpec
    processes: int
    shards: tuple[RunResult, ...]
    readings: tuple[float | None, ...] = ()
    converged: bool | None = None
    failures: tuple[str, ...] = ()

    @property
    def result(self) -> RunResult:
        """The one rung. Raises if the processes did not measure the same window."""
        return combine_shards(
            self.shards,
            spec=self.spec,
            processes=self.processes,
            readings=self.readings,
            converged=self.converged,
        )


# --------------------------------------------------------------------------------------
# The child side.
# --------------------------------------------------------------------------------------


class _WorkerGate:
    """The window gate of a process that is a share of a rung: it asks the coordinator.

    Every wait is against an absolute instant on the shared monotonic clock handed down
    by the coordinator, so a child whose coordinator has died answers instead of hanging:
    no origin means this process abandons the rung, and no decision means no window.
    """

    def __init__(
        self,
        up: Connection,
        down: Connection,
        *,
        origin_deadline_s: float,
        decision_deadline_s: float,
    ) -> None:
        self._up = up
        self._down = down
        self._origin_deadline_s = origin_deadline_s
        self._decision_deadline_s = decision_deadline_s
        self._abandoned = False

    async def all_live(self, at_s: float) -> float:
        await self._send(("live", at_s))
        message = await self._recv(self._origin_deadline_s)
        if message is not None and message[0] == "origin":
            return float(message[1])
        self._abandoned = True
        return at_s

    async def reading(self, values: Sequence[float], *, start_s: float, end_s: float) -> None:
        if self._abandoned:
            return
        await self._send(("reading", start_s, end_s, list(values)))

    async def opens_at(self) -> float | None:
        if self._abandoned:
            return None
        message = await self._recv(self._decision_deadline_s)
        if message is not None and message[0] == "open":
            return float(message[1])
        return None

    async def _send(self, message: tuple[Any, ...]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_blocking, message)

    def _send_blocking(self, message: tuple[Any, ...]) -> None:
        try:
            self._up.send(message)
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._abandoned = True

    async def _recv(self, deadline_s: float) -> tuple[Any, ...] | None:
        """Wait on the coordinator's channel without blocking a thread, until the deadline.

        On the loop rather than in an executor on purpose. A thread parked in ``recv``
        outlives the coroutine that started it: this process ends its load by cancelling
        the supervisor, and ``asyncio.run`` then waits for the default executor's threads,
        so a parked one would hold the process open until its own deadline. Everything the
        coordinator sends down is a short tuple written in one go, so a readable channel
        has a whole message on it.
        """
        loop = asyncio.get_running_loop()
        fileno = self._down.fileno()
        while True:
            remaining = deadline_s - time.monotonic()
            if remaining <= 0:
                return None
            if not self._poll_ready():
                waiter: asyncio.Future[None] = loop.create_future()
                try:
                    loop.add_reader(fileno, self._wake, waiter)
                except (OSError, ValueError):
                    return None
                try:
                    await asyncio.wait_for(waiter, timeout=min(remaining, 1.0))
                except TimeoutError:
                    continue
                finally:
                    loop.remove_reader(fileno)
            try:
                message = self._down.recv()
            except (EOFError, OSError, ValueError):
                return None
            if isinstance(message, tuple) and message:
                return message

    def _poll_ready(self) -> bool:
        try:
            return bool(self._down.poll(0))
        except (OSError, ValueError):
            return False

    @staticmethod
    def _wake(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)


def _shard_main(
    spec: LoadSpec,
    up_r: Connection,
    up_w: Connection,
    down_r: Connection,
    down_w: Connection,
    origin_deadline_s: float,
    decision_deadline_s: float,
) -> None:
    """One process's whole job: run the shared event loop over its own slots and report.

    Both ends of both pipes are inherited across the fork; the two this process does not
    use are closed here so the coordinator sees an end-of-file when this process exits
    rather than waiting on a copy of its own writer.
    """
    up_r.close()
    down_w.close()
    try:
        gate = _WorkerGate(
            up_w,
            down_r,
            origin_deadline_s=origin_deadline_s,
            decision_deadline_s=decision_deadline_s,
        )
        result = asyncio.run(run_load(spec, gate=gate))
        up_w.send(("done", result))
    except BaseException:  # a child that dies silently would hang the rung
        with contextlib.suppress(BrokenPipeError, EOFError, OSError, ValueError):
            up_w.send(("failed", traceback.format_exc(limit=8)))
    finally:
        for connection in (up_w, down_r):
            with contextlib.suppress(OSError):
                connection.close()


# --------------------------------------------------------------------------------------
# The coordinator side.
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Shard:
    plan: ShardPlan
    process: Process
    up: Connection
    down: Connection
    live_at_s: float | None = None
    readings: list[list[float]] = field(default_factory=list)
    result: RunResult | None = None
    failure: str | None = None
    closed: bool = False

    @property
    def finished(self) -> bool:
        return self.result is not None or self.failure is not None or self.closed


class _Coordinator:
    """Runs the rung's warm-up protocol over the processes and hands out one deadline."""

    def __init__(self, spec: LoadSpec, shards: Sequence[_Shard], *, lead_s: float) -> None:
        self._spec = spec
        self._shards = list(shards)
        self._lead_s = lead_s
        self._queue: asyncio.Queue[tuple[int, tuple[Any, ...]]] = asyncio.Queue()
        self._threads: list[threading.Thread] = []
        self.origin_s: float | None = None
        self.opens_at: float | None = None
        self.readings: list[float | None] = []
        self.converged: bool | None = None

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        for shard in self._shards:
            thread = threading.Thread(
                target=self._pump,
                args=(shard, loop),
                name=f"vb-shard-{shard.plan.index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _pump(self, shard: _Shard, loop: asyncio.AbstractEventLoop) -> None:
        """Read one child's messages on a thread and hand them to the loop.

        A thread rather than a reader on the loop because a child's result is a whole
        run's worth of samples and arrives in several pipe writes; a partial message must
        block this thread and never the coordinator's loop.
        """
        index = shard.plan.index
        while True:
            try:
                message = shard.up.recv()
            except (EOFError, OSError, ValueError):
                loop.call_soon_threadsafe(self._queue.put_nowait, (index, ("eof",)))
                return
            loop.call_soon_threadsafe(self._queue.put_nowait, (index, message))
            if message and message[0] in ("done", "failed"):
                return

    def _broadcast(self, message: tuple[Any, ...]) -> None:
        for shard in self._shards:
            try:
                shard.down.send(message)
            except (BrokenPipeError, EOFError, OSError, ValueError):
                continue

    def _apply(self, index: int, message: tuple[Any, ...]) -> None:
        shard = self._shards[index]
        kind = message[0]
        if kind == "live":
            shard.live_at_s = float(message[1])
        elif kind == "reading":
            shard.readings.append([float(value) for value in message[3]])
        elif kind == "done":
            shard.result = message[1]
        elif kind == "failed":
            shard.failure = str(message[1])
        elif kind == "eof":
            shard.closed = True

    async def _drain(self, deadline_s: float) -> bool:
        """Take one message, or return False when the deadline passed with none."""
        remaining = deadline_s - time.monotonic()
        if remaining <= 0:
            return False
        try:
            index, message = await asyncio.wait_for(self._queue.get(), timeout=remaining)
        except TimeoutError:
            return False
        self._apply(index, message)
        return True

    async def await_all_live(self, deadline_s: float) -> float | None:
        """The instant the rung is at N: when its last process reported its slots live."""
        while True:
            live = [shard.live_at_s for shard in self._shards]
            if all(value is not None for value in live):
                self.origin_s = max(value for value in live if value is not None)
                self._broadcast(("origin", self.origin_s))
                return self.origin_s
            if any(shard.finished for shard in self._shards):
                return None
            if not await self._drain(deadline_s):
                return None

    async def await_convergence(self, origin_s: float) -> float | None:
        """Pool each reading round across the processes and decide on the rung's series.

        The rung's k-th reading is the 95th percentile of every process's k-th interval
        together, which is the statistic the window itself reports. Two consecutive rung
        readings that agree within the frozen tolerance open the window; nothing agreeing
        by the cap opens none.
        """
        if self._spec.warm_up_s is None:
            return self._decide(origin_s)
        cap_at = origin_s + self._spec.warm_up_cap_s
        previous: float | None = None
        while True:
            rounds = min(len(shard.readings) for shard in self._shards)
            while len(self.readings) < rounds:
                index = len(self.readings)
                pooled = [value for shard in self._shards for value in shard.readings[index]]
                current = percentile(pooled, 95) if pooled else None
                self.readings.append(current)
                if _readings_agree(previous, current, self._spec.warm_up_convergence):
                    return self._decide(origin_s)
                previous = current
            if any(shard.finished for shard in self._shards):
                break
            if not await self._drain(cap_at):
                break
        self.converged = False
        self._broadcast(("never",))
        return None

    def _decide(self, origin_s: float) -> float:
        floor_at = origin_s + (self._spec.warm_up_s or 0.0)
        opens_at = max(time.monotonic() + self._lead_s, floor_at)
        self.opens_at = opens_at
        self.converged = None if self._spec.warm_up_s is None else True
        self._broadcast(("open", opens_at))
        return opens_at

    def give_up(self) -> None:
        self.converged = False if self._spec.warm_up_s is not None else None
        self._broadcast(("never",))

    async def collect(self, deadline_s: float) -> None:
        while not all(shard.finished for shard in self._shards):
            if not await self._drain(deadline_s):
                return

    def join_threads(self, timeout_s: float = 5.0) -> None:
        for thread in self._threads:
            thread.join(timeout_s)


async def run_sharded_load(
    spec: LoadSpec,
    *,
    processes: int,
    hooks: WindowHooks | None = None,
    on_children: Callable[[Sequence[int]], None] | None = None,
    window_open_lead_s: float = WINDOW_OPEN_LEAD_S,
    live_grace_s: float = LIVE_GRACE_S,
    result_grace_s: float = RESULT_GRACE_S,
) -> ShardedRun:
    """Run one rung across ``processes`` event loops and return what they measured.

    ``hooks`` are fired here, in the coordinator, at the shared window's own two instants,
    so the host record is taken once for the box over the window every process measured.
    ``on_children`` is handed the child pids as soon as they exist, which is what lets the
    sampler charge the client's CPU to the processes that actually spent it.

    One process is not a special case of the protocol, only of the plumbing: it runs the
    same event loop with its own local gate, exactly as before this module existed.
    """
    if processes < 1:
        raise ValueError(f"a rung needs at least one process, got {processes}")
    if spec.sessions < 1:
        raise ValueError(f"a rung needs at least one session, got {spec.sessions}")
    if processes == 1 or spec.sessions == 1:
        shard = await run_load(spec, hooks=hooks)
        return ShardedRun(
            spec=spec,
            processes=1,
            shards=(shard,),
            readings=tuple(shard.warm_up_readings),
            converged=shard.warm_up_converged,
        )
    if spec.window_s is None:
        raise ValueError(
            "a load with no measurement window has no window to share, so splitting it "
            "across processes would only spread slots; run it in one"
        )
    require_shared_monotonic()
    plans = plan_shards(spec.sessions, processes)
    context = _fork_context()
    started_at_s = time.monotonic()
    origin_deadline_s = started_at_s + spec.ramp_s + live_grace_s
    decision_deadline_s = origin_deadline_s + spec.warm_up_cap_s + live_grace_s
    shards = _fork_shards(
        spec,
        plans,
        context=context,
        origin_deadline_s=origin_deadline_s,
        decision_deadline_s=decision_deadline_s,
    )
    if on_children is not None:
        on_children([shard.process.pid for shard in shards if shard.process.pid is not None])
    coordinator = _Coordinator(spec, shards, lead_s=window_open_lead_s)
    try:
        coordinator.start()
        origin_s = await coordinator.await_all_live(origin_deadline_s)
        opens_at: float | None = None
        if origin_s is None:
            coordinator.give_up()
        else:
            opens_at = await coordinator.await_convergence(origin_s)
        if opens_at is not None:
            await _hold_the_window(opens_at, spec.window_s, hooks)
            deadline_s = opens_at + spec.window_s + result_grace_s
        else:
            deadline_s = time.monotonic() + result_grace_s
        await coordinator.collect(deadline_s)
    finally:
        # Processes first: a reader thread ends when its child's end of the pipe closes,
        # which is when the child exits. Then the threads, then the pipes, so no thread is
        # ever reading a connection this closed under it.
        _stop_processes(shards)
        coordinator.join_threads()
        _close_pipes(shards)
    results = tuple(shard.result for shard in shards if shard.result is not None)
    failures = tuple(
        shard.failure
        if shard.failure is not None
        else f"process {shard.plan.index} ended without a result"
        for shard in shards
        if shard.result is None
    )
    if failures:
        # Not a rung that ran badly: a rung that ran at less than N for part of its own
        # window and cannot say for how long. Every wait above is bounded so this is
        # reached rather than hung, and it is raised rather than combined so that a load
        # short of the concurrency it names never reaches a row.
        raise ShardedLoadError(
            f"{len(failures)} of this rung's {len(shards)} load processes returned no "
            "result, so it did not hold its stream count: " + "; ".join(failures)
        )
    return ShardedRun(
        spec=spec,
        processes=len(shards),
        shards=results,
        readings=tuple(coordinator.readings),
        converged=coordinator.converged,
        failures=failures,
    )


def _fork_context() -> BaseContext:
    if "fork" not in mp.get_all_start_methods():
        raise ShardedLoadError(
            "this platform has no fork start method, and spawn and forkserver both "
            "re-import the parent's __main__ in the child, which re-runs a console "
            "script or a test runner; a rung cannot be split here"
        )
    return mp.get_context("fork")


def _fork_shards(
    spec: LoadSpec,
    plans: Sequence[ShardPlan],
    *,
    context: BaseContext,
    origin_deadline_s: float,
    decision_deadline_s: float,
) -> list[_Shard]:
    """Fork every child before the coordinator starts a thread of its own.

    Order matters: a fork copies only the forking thread, so forking first keeps the
    children out of any lock this process's own reader threads might be holding.
    """
    shards: list[_Shard] = []
    for plan in plans:
        up_r, up_w = context.Pipe(duplex=False)
        down_r, down_w = context.Pipe(duplex=False)
        process = context.Process(
            target=_shard_main,
            args=(
                shard_spec(spec, plan),
                up_r,
                up_w,
                down_r,
                down_w,
                origin_deadline_s,
                decision_deadline_s,
            ),
            name=f"vb-load-{plan.index}",
            daemon=True,
        )
        process.start()
        up_w.close()
        down_r.close()
        shards.append(_Shard(plan=plan, process=process, up=up_r, down=down_w))
    return shards


async def _hold_the_window(opens_at: float, window_s: float, hooks: WindowHooks | None) -> None:
    """Open and close the box's own record on the same two instants the children use."""
    lead = opens_at - time.monotonic()
    if lead > 0:
        await asyncio.sleep(lead)
    if hooks is not None:
        hooks.on_open()
    remaining = opens_at + window_s - time.monotonic()
    if remaining > 0:
        await asyncio.sleep(remaining)
    if hooks is not None:
        hooks.on_close()


def _stop_processes(shards: Sequence[_Shard], *, grace_s: float = 30.0) -> None:
    """Every child is joined, and one that will not go is killed. Bounded either way."""
    deadline = time.monotonic() + grace_s
    for shard in shards:
        shard.process.join(max(0.0, deadline - time.monotonic()))
    for shard in shards:
        if shard.process.is_alive():
            shard.process.terminate()
            shard.process.join(5.0)


def _close_pipes(shards: Sequence[_Shard]) -> None:
    for shard in shards:
        for connection in (shard.up, shard.down):
            with contextlib.suppress(OSError):
                connection.close()
        with contextlib.suppress(ValueError, OSError):
            shard.process.close()
