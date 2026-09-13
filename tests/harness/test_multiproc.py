# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""One rung split across processes is the same rung, or it is not a rung.

Everything here runs against the null server on the box the suite runs on: no GPU, no
model, and every duration injected, so a three-phase rung across several real processes
costs seconds instead of the frozen four minutes.

The four things a split rung has to preserve, which is what these tests are for: the same
seed puts the same session in the same slot however the rung is divided; the samples of
every process reach the rung's percentile; the pacing slip and the session counts pool
and sum over the whole load and not over one share of it; and the window is one window,
opened at one instant in every process. The last is the one a mutation is aimed at.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import replace
from pathlib import Path

import pytest
from test_pace import make_manifest, make_wav
from verbatim_bench.client import ChunkMode, SessionResult
from verbatim_bench.env import FakeGpuProbe, fake_gpu_facts
from verbatim_bench.hostrecord import WindowRecorder, run_sharded_load_recorded
from verbatim_bench.multiproc import (
    ShardedRun,
    combine_shards,
    plan_shards,
    run_sharded_load,
    shard_spec,
)
from verbatim_bench.nullserver import NullServer, NullServerConfig
from verbatim_bench.pace import LoadSpec, plan_start_offsets, window_partial_samples
from verbatim_bench.results import RunResult

pytestmark = pytest.mark.cpu

#: A ramp, warm-up and window with the shape of the frozen ones and none of their length.
FAST: dict[str, float] = {
    "ramp_s": 0.1,
    "warm_up_s": 0.4,
    "warm_up_reading_s": 0.4,
    "warm_up_cap_s": 10.0,
    "window_s": 1.0,
}
#: Bounds for the coordinator, small enough that a broken rung fails this suite in
#: seconds rather than sitting on the default backstops for minutes.
BOUNDS: dict[str, float] = {
    "window_open_lead_s": 0.2,
    "live_grace_s": 20.0,
    "result_grace_s": 40.0,
}


def _spec(endpoint: str, manifest: Path, *, sessions: int, seed: int = 11) -> LoadSpec:
    return LoadSpec(
        endpoint=endpoint,
        manifest=manifest,
        sessions=sessions,
        chunk=ChunkMode.parse(160),
        seed=seed,
        frame_ms=20,
        **FAST,  # type: ignore[arg-type]
    )


def _corpus(tmp_path: Path, entries: int = 3) -> Path:
    wavs = []
    for index in range(entries):
        wav = tmp_path / f"utt{index}.wav"
        make_wav(wav, 1.0)
        wavs.append(wav)
    return make_manifest(tmp_path / "m.jsonl", wavs, [f"e{index}" for index in range(entries)])


async def _run(manifest: Path, *, sessions: int, processes: int, seed: int = 11) -> ShardedRun:
    async with NullServer(NullServerConfig(partial_delay_ms=20.0)) as server:
        spec = _spec(server.endpoint, manifest, sessions=sessions, seed=seed)
        return await run_sharded_load(spec, processes=processes, **BOUNDS)  # type: ignore[arg-type]


def _slot_of(session_id: str) -> int:
    return int(session_id.split("-")[0][1:])


def _first_sessions(result: RunResult) -> set[tuple[str, str]]:
    """The session each slot opened first: the seeded slot assignment, with the ordinals
    that closed-loop replacement produced from the clock left out of it."""
    return {
        (session.session_id, session.stream_id)
        for session in result.sessions
        if session.session_id.endswith("-0000")
    }


# --------------------------------------------------------------------------------------
# The slot plan: the same seed, the same slots, however many processes.
# --------------------------------------------------------------------------------------


def test_shards_deal_every_rung_slot_to_exactly_one_process() -> None:
    for processes in range(1, 8):
        plans = plan_shards(20, processes)
        dealt = [slot for plan in plans for slot in plan.slots]
        assert sorted(dealt) == list(range(20)), processes
        assert sum(plan.sessions for plan in plans) == 20


def test_more_processes_than_streams_gets_one_process_per_stream() -> None:
    plans = plan_shards(3, 16)
    assert len(plans) == 3
    assert [plan.slots for plan in plans] == [(0,), (1,), (2,)]


def test_the_start_schedule_is_the_same_however_many_processes_split_it(tmp_path: Path) -> None:
    """A rung's session-to-slot assignment is a function of its seed, not of its plumbing.

    If splitting a rung re-seeded per process, or numbered slots inside each process, the
    same rung at the same seed would put different sessions in different places at
    different moments and the ladder would stop being repeatable.
    """
    rung = _spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=12, seed=4242)
    whole = plan_start_offsets(rung)
    assert len(whole) == 12
    for processes in range(1, 7):
        rebuilt: dict[int, float] = {}
        for plan in plan_shards(rung.sessions, processes):
            spec = shard_spec(rung, plan)
            offsets = plan_start_offsets(spec)
            assert len(offsets) == plan.sessions
            assert spec.slot_indices == plan.slots
            for slot, offset in zip(plan.slots, offsets, strict=True):
                rebuilt[slot] = offset
        assert [rebuilt[slot] for slot in range(12)] == whole, processes


def test_a_shard_spec_refuses_slots_that_do_not_describe_a_share_of_a_rung(
    tmp_path: Path,
) -> None:
    base = _spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=4)
    with pytest.raises(ValueError, match="total_sessions"):
        replace(base, slots=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="entries but sessions"):
        replace(base, slots=(0, 1), total_sessions=4)
    with pytest.raises(ValueError, match="repeats"):
        replace(base, sessions=2, slots=(1, 1), total_sessions=4)
    with pytest.raises(ValueError, match="leaves the rung"):
        replace(base, sessions=2, slots=(0, 9), total_sessions=4)


async def test_a_split_rung_runs_the_same_sessions_as_a_single_process_rung(
    tmp_path: Path,
) -> None:
    """Same seed, same concurrency, same session in the same slot: split or not."""
    manifest = _corpus(tmp_path)
    single = await _run(manifest, sessions=6, processes=1, seed=777)
    split = await _run(manifest, sessions=6, processes=3, seed=777)
    assert len(single.shards) == 1
    assert len(split.shards) == 3
    assigned = _first_sessions(single.result)
    assert len(assigned) == 6
    assert _first_sessions(split.result) == assigned
    # And the slots really were spread across the processes rather than all run in one.
    per_process = [{_slot_of(s.session_id) for s in shard.sessions} for shard in split.shards]
    assert sorted(slot for group in per_process for slot in group) == list(range(6))


# --------------------------------------------------------------------------------------
# The window: one window, in every process.
# --------------------------------------------------------------------------------------


async def test_every_process_opens_the_same_window(tmp_path: Path) -> None:
    """The guard this change exists for.

    The rung's percentile is taken over the samples of every process pooled. If each
    process opened its own window when its own quarter of the streams looked settled, the
    pool would mix samples from windows that did not overlap and the percentile would
    describe a load nobody ran. So the window opens at one instant, named once by the
    coordinator and recorded verbatim by every process, and this asserts exactly that:
    one instant, one close, in all of them, and it is the rung's.
    """
    run = await _run(_corpus(tmp_path), sessions=6, processes=3)
    assert len(run.shards) == 3
    assert run.converged is True
    opens = {shard.window_open_s for shard in run.shards}
    closes = {shard.window_close_s for shard in run.shards}
    assert len(opens) == 1, f"the processes opened different windows: {sorted(opens)!r}"
    assert len(closes) == 1, f"the processes closed different windows: {sorted(closes)!r}"
    open_s = opens.pop()
    close_s = closes.pop()
    assert open_s is not None and close_s is not None
    assert close_s - open_s == pytest.approx(FAST["window_s"], abs=1e-9)
    rung = run.result
    assert rung.window_open_s == open_s
    assert rung.window_close_s == close_s
    # The warm-up is the rung's too: every process was handed the same origin, so they
    # all report the same warm-up length rather than each timing its own.
    assert len({shard.all_live_at_s for shard in run.shards}) == 1


async def test_the_rung_keeps_the_pooled_warm_up_readings_not_one_process_s(
    tmp_path: Path,
) -> None:
    run = await _run(_corpus(tmp_path), sessions=6, processes=3)
    assert run.converged is True
    assert len(run.readings) >= 2
    assert run.result.warm_up_readings == tuple(run.readings)
    # Every process took its own readings as well, and they are its own record.
    assert all(shard.warm_up_readings for shard in run.shards)


# --------------------------------------------------------------------------------------
# The aggregation: samples pool, slip pools, counts sum.
# --------------------------------------------------------------------------------------


async def test_samples_from_every_process_reach_the_rung(tmp_path: Path) -> None:
    run = await _run(_corpus(tmp_path), sessions=6, processes=3)
    rung = run.result
    per_process = [window_partial_samples(shard) for shard in run.shards]
    assert all(samples for samples in per_process), (
        f"a process contributed no window sample at all: {[len(s) for s in per_process]}"
    )
    pooled = window_partial_samples(rung)
    assert len(pooled) == sum(len(samples) for samples in per_process)
    assert sorted(pooled) == sorted(value for samples in per_process for value in samples)
    # Every slot of the rung is represented in the window, not merely every process.
    open_s, close_s = rung.window_open_s, rung.window_close_s
    assert open_s is not None and close_s is not None
    slots_in_window = {
        _slot_of(session.session_id)
        for session in rung.sessions
        if any(open_s <= recv <= close_s for recv in session.partial_recv_s)
    }
    assert slots_in_window == set(range(6))


def _session(session_id: str, *, slips: list[float], error: str | None = None) -> SessionResult:
    return SessionResult(
        session_id=session_id,
        stream_id="e0",
        server_session_id=None if error == "refused" else "srv",
        pacing_slip_ms=slips,
        finals_received=0 if error else 1,
        error=None if error is None else "boom",
    )


def _shard(sessions: list[SessionResult], *, open_s: float = 100.0, wall_start: float = 90.0):
    return RunResult(
        spec_dict={},
        sessions=sessions,
        wall_clock_s=120.0,
        corpus_id="sha256:deadbeef",
        manifest_name="m.jsonl",
        wall_start_s=wall_start,
        all_live_at_s=95.0,
        warm_up_end_s=open_s,
        window_open_s=open_s,
        window_close_s=open_s + 180.0,
        warm_up_converged=True,
    )


def test_the_rung_pools_pacing_slip_and_sums_session_counts(tmp_path: Path) -> None:
    rung = _spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=4)
    left = _shard([_session("s0000-0000", slips=[1.0, 9.0]), _session("s0002-0000", slips=[2.0])])
    right = _shard(
        [
            _session("s0001-0000", slips=[3.0]),
            _session("s0003-0000", slips=[4.0], error="refused"),
        ],
        wall_start=91.0,
    )
    combined = combine_shards([left, right], spec=rung, processes=2, converged=True)
    assert [session.session_id for session in combined.sessions] == [
        "s0000-0000",
        "s0001-0000",
        "s0002-0000",
        "s0003-0000",
    ]
    slips = [slip for session in combined.sessions for slip in session.pacing_slip_ms]
    assert sorted(slips) == [1.0, 2.0, 3.0, 4.0, 9.0]
    document = combined.to_json_dict()["result"]
    assert document["sessions_started"] == 4
    assert document["sessions_failed"] == 1
    # The rung's spec is the rung's, not one process's share of it.
    assert combined.spec_dict["sessions"] == 4
    assert combined.spec_dict["processes"] == 2
    assert combined.spec_dict["slots"] is None
    # One wall clock over the whole rung: first process to start, last to stop.
    assert combined.wall_start_s == 90.0
    assert combined.wall_clock_s == pytest.approx(121.0)


def test_a_rung_missing_one_of_its_processes_is_not_a_rung(tmp_path: Path) -> None:
    """A process that died held slots that stopped sending, so the load fell below N for
    part of the window and there is no way to say for how long."""
    rung = _spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=4)
    survivor = _shard([_session("s0000-0000", slips=[1.0])])
    with pytest.raises(ValueError, match="returned a result"):
        combine_shards([survivor], spec=rung, processes=2, converged=True)


def test_shards_that_measured_different_windows_are_not_a_rung(tmp_path: Path) -> None:
    rung = _spec("ws://127.0.0.1:1/v1/stream", tmp_path / "m.jsonl", sessions=2)
    left = _shard([_session("s0000-0000", slips=[1.0])], open_s=100.0)
    right = _shard([_session("s0001-0000", slips=[1.0])], open_s=100.5)
    with pytest.raises(ValueError, match="did not measure the same window"):
        combine_shards([left, right], spec=rung, processes=2, converged=True)


# --------------------------------------------------------------------------------------
# The host record: the box's, taken once, charged to the processes that spent the CPU.
# --------------------------------------------------------------------------------------


async def test_the_host_record_is_taken_once_for_the_box_and_names_every_load_process(
    tmp_path: Path,
) -> None:
    manifest = _corpus(tmp_path)
    probe = FakeGpuProbe((fake_gpu_facts(throttle_reasons={"clocks_throttle_reason_gpu_idle": 1}),))
    recorder = WindowRecorder(gpu=probe, client_pid=os.getpid(), interval_s=0.1)
    async with NullServer(NullServerConfig(partial_delay_ms=20.0)) as server:
        spec = _spec(server.endpoint, manifest, sessions=6)
        result = await run_sharded_load_recorded(spec, recorder, processes=3, **BOUNDS)
    host = recorder.finish()
    assert host is not None
    # One record, and it spans the window every process measured.
    assert host.open_s == pytest.approx(result.window_open_s, abs=0.2)
    assert host.close_s == pytest.approx(result.window_close_s, abs=0.2)
    assert host.gpu.samples >= 3
    # The client's CPU is charged to the processes that spent it: this one and its three.
    assert len(recorder.client_pids) == 4
    assert os.getpid() in recorder.client_pids
    assert host.counters.client_processes == 4


# --------------------------------------------------------------------------------------
# The ladder: the rung the operator asked for, across the processes they asked for.
# --------------------------------------------------------------------------------------


async def test_the_ladder_runs_each_rung_across_the_processes_it_was_given(
    tmp_path: Path,
) -> None:
    """`--processes` has to reach the rung executor, and the artifact has to say so.

    A rung that quietly ran in one process while the ladder recorded four would report a
    generator-bound percentile under a heading claiming it was not.
    """
    from test_ladder import FAST_RUNG, _run_fast_ladder

    document = await _run_fast_ladder(
        tmp_path,
        NullServerConfig(partial_delay_ms=20.0),
        [*FAST_RUNG, "--processes", "3", "--n0", "6"],
    )
    assert document["config"]["processes"] == 3
    rungs = document["rungs"]
    assert rungs and rungs[0]["n"] == 6
    # The rung ran its window and reported the whole load, not one process's third of it.
    assert rungs[0]["wall_clock_s"] > 1.5


# --------------------------------------------------------------------------------------
# The failure paths, which have to end in bounded time and say what happened.
# --------------------------------------------------------------------------------------


async def test_a_rung_no_process_can_connect_to_opens_no_window_and_ends(
    tmp_path: Path,
) -> None:
    """Nothing answers, so nothing settles, so no window opens -- inside the cap.

    The bound is the point. A coordinator that waited on processes that will never report
    a reading would hang the ladder rather than record an unstable rung.
    """
    manifest = _corpus(tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
    spec = replace(
        _spec(f"ws://127.0.0.1:{closed_port}/v1/stream", manifest, sessions=4),
        warm_up_cap_s=2.0,
    )
    started = time.monotonic()
    run = await run_sharded_load(spec, processes=2, **BOUNDS)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started
    assert elapsed < 30.0, f"the rung took {elapsed:.1f}s to give up on a dead endpoint"
    assert run.converged is False
    rung = run.result
    assert rung.window_open_s is None and rung.window_close_s is None
    assert window_partial_samples(rung) == []
    assert rung.sessions and all(session.error is not None for session in rung.sessions)


async def test_a_rung_that_lost_a_load_process_refuses_to_be_a_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that returns no result held slots that stopped sending.

    The load was then under N for part of the window and there is no way to say for how
    long, so the rung is refused rather than combined out of what is left. Without this
    the row would name a stream count no load ever held.
    """
    from verbatim_bench import multiproc

    def _die(spec, up_r, up_w, down_r, down_w, origin_deadline_s, decision_deadline_s):  # type: ignore[no-untyped-def]
        up_r.close()
        down_w.close()
        if spec.slots[0] == 0:
            up_w.send(("failed", "injected: this process never ran"))
            up_w.close()
            return
        multiproc._shard_main(
            spec, up_r, up_w, down_r, down_w, origin_deadline_s, decision_deadline_s
        )

    monkeypatch.setattr(multiproc, "_shard_main", _die)
    manifest = _corpus(tmp_path)
    async with NullServer(NullServerConfig(partial_delay_ms=20.0)) as server:
        spec = replace(_spec(server.endpoint, manifest, sessions=4), warm_up_cap_s=2.0)
        with pytest.raises(multiproc.ShardedLoadError, match="did not hold its stream count"):
            await run_sharded_load(spec, processes=2, **BOUNDS)  # type: ignore[arg-type]
