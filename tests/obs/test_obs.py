# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The observability modules on their own: counters, the latency window, the exposition."""

from __future__ import annotations

import pytest

from verbatim.core.types import TickStats
from verbatim.obs.counters import Counters
from verbatim.obs.latency import LatencySketch
from verbatim.obs.metrics import MetricsSnapshot, render

pytestmark = pytest.mark.cpu


def _stats(**overrides: float) -> TickStats:
    fields = dict(
        tick_id=1,
        steady_rows=8,
        live_rows=2,
        pad_rows=6,
        edge_batches=0,
        eager_rows=0,
        starved=0,
        step_ms=10.0,
        edge_ms=0.0,
        lateness_ms=20.0,
    )
    fields.update(overrides)
    return TickStats(**fields)  # type: ignore[arg-type]


def test_counters_fold_a_tick_by_its_cost_and_its_lateness() -> None:
    counters = Counters()
    counters.observe_tick(_stats(), budget_ms=112.0, period_ms=160.0)
    counters.observe_tick(
        _stats(step_ms=200.0, lateness_ms=100.0), budget_ms=112.0, period_ms=160.0
    )
    counters.observe_tick(
        _stats(lateness_ms=170.0, edge_batches=2), budget_ms=112.0, period_ms=160.0
    )
    assert counters.ticks_total == 3
    assert counters.ticks_over_budget_total == 1
    assert counters.ticks_late_total == 1
    assert counters.steady_steps_total == 3
    assert counters.eager_steps_total == 2


def test_a_tick_with_no_steady_rows_is_not_a_steady_step() -> None:
    counters = Counters()
    counters.observe_tick(
        _stats(steady_rows=0, live_rows=0, pad_rows=0), budget_ms=112.0, period_ms=160.0
    )
    assert counters.ticks_total == 1
    assert counters.steady_steps_total == 0


def test_counters_split_admissions_from_refusals() -> None:
    counters = Counters()
    counters.observe_admission(True)
    counters.observe_admission(False)
    counters.observe_admission(True)
    assert (counters.sessions_admitted_total, counters.sessions_refused_total) == (2, 1)


def test_the_latency_window_reports_nearest_rank_quantiles_over_what_it_kept() -> None:
    sketch = LatencySketch(window=5)
    assert sketch.count == 0 and sketch.p95 == 0.0
    for ms in (50.0, 10.0, 40.0, 20.0, 30.0):
        sketch.observe(ms)
    assert sketch.count == 5
    assert sketch.p50 == 30.0
    assert sketch.p95 == 50.0
    assert sketch.p99 == 50.0
    sketch.observe(5.0)  # the oldest, 50.0, falls out of the window
    assert sketch.count == 5
    assert sketch.p99 == 40.0


def test_the_latency_window_refuses_nonsense() -> None:
    with pytest.raises(ValueError):
        LatencySketch(window=0)
    with pytest.raises(ValueError):
        LatencySketch().quantile(1.5)


def _snapshot(**overrides: object) -> MetricsSnapshot:
    fields: dict[str, object] = dict(
        chunk_ms=160,
        period_ms=160.0,
        budget_ms=112.0,
        bucket=8,
        started=True,
        running=True,
        dead=False,
        tick_id=12,
        last_tick_age_s=0.01,
        live_sessions=2,
        slots_capacity=31,
        slots_reserved=17,
        slots_free=14,
        ceiling=None,
        calibrated_ceiling=None,
        degradation_level=0,
        consecutive_overruns=0,
        p95_tick_ms=9.5,
        eager_step_fraction=0.25,
        counters=Counters(ticks_total=12, sessions_admitted_total=2, sessions_refused_total=1),
        latency_count=12,
        partial_latency_ms=(21.0, 30.0, 33.0),
        tick_cost_ms=(9.0, 9.5, 9.9),
        last_refusal_reason="session refused: no free slot",
        results_partials_dropped_total=3,
    )
    fields.update(overrides)
    return MetricsSnapshot(**fields)  # type: ignore[arg-type]


def test_the_exposition_carries_the_process_labels_on_every_series() -> None:
    text = render(
        _snapshot(), labels={"chunk_ms": "160", "precision": "bfloat16", "execution": "eager"}
    )
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert lines, text
    for line in lines:
        assert 'chunk_ms="160",precision="bfloat16",execution="eager"' in line, line
    assert (
        'verbatim_live_sessions{chunk_ms="160",precision="bfloat16",execution="eager"} 2' in lines
    )
    assert (
        'verbatim_sessions_refused_total{chunk_ms="160",precision="bfloat16",execution="eager"} 1'
        in lines
    )
    labelled = 'chunk_ms="160",precision="bfloat16",execution="eager",quantile="0.95"'
    assert f"verbatim_partial_latency_ms{{{labelled}}} 30.0" in lines
    assert "# TYPE verbatim_ticks_total counter" in text
    dropped = "verbatim_result_partials_dropped_total"
    assert f'{dropped}{{chunk_ms="160",precision="bfloat16",execution="eager"}} 3' in lines
    assert "# TYPE verbatim_live_sessions gauge" in text
    assert text.endswith("\n")


def test_an_uncalibrated_engine_exposes_no_ceiling_series() -> None:
    assert "verbatim_ceiling" not in render(_snapshot(), labels={})
    assert "verbatim_ceiling{} 24" not in render(_snapshot(ceiling=24), labels={})
    assert "verbatim_ceiling 24" in render(_snapshot(ceiling=24), labels={})


def test_label_values_are_escaped() -> None:
    text = render(_snapshot(), labels={"model": 'a"b\\c'})
    assert 'model="a\\"b\\\\c"' in text
