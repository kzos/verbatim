# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The results record and its JSON writer.

Percentiles are nearest-rank on the sorted sample, defined once here so a later
recomputation from the raw samples reproduces the summary exactly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verbatim_bench.client import SessionResult


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on the sorted sample. Returns 0.0 for an empty input.

    Defined once here and used everywhere, so that a later recomputation from the raw
    samples reproduces the summary exactly.
    """
    if len(values) == 0:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(p / 100.0 * len(ordered)) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return float(ordered[rank])


def _block(values: Sequence[float]) -> dict[str, Any]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "n": len(values),
    }


@dataclass
class RunResult:
    spec_dict: dict[str, Any]
    sessions: list[SessionResult] = field(default_factory=list)
    wall_clock_s: float = 0.0
    corpus_id: str = ""
    manifest_name: str = ""
    started_at: datetime | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Shape this task owns jointly with the schema contract; keys are never renamed."""
        from verbatim_bench import __version__
        from verbatim_bench.canonical import stamp_checksum

        first_partials = [
            s.first_partial_ms for s in self.sessions if s.first_partial_ms is not None
        ]
        partials = [v for s in self.sessions for v in s.partial_ms]
        finals = [s.final_ms for s in self.sessions if s.final_ms is not None]
        slips = [v for s in self.sessions for v in s.pacing_slip_ms]
        completed = sum(1 for s in self.sessions if s.error is None)
        failed = sum(1 for s in self.sessions if s.error is not None)
        started_at = self.started_at or datetime.now().astimezone()
        doc: dict[str, Any] = {
            "schema": "vb-results/1",
            "harness": {"version": __version__, "git": None, "client_floor_p95_ms": None},
            "arm": {
                "name": self.spec_dict.get("arm", "unknown"),
                "version": None,
                "container_digest": None,
                "surface": "websocket",
            },
            "checkpoint": {"hf_id": None, "revision": None, "file_sha256": None},
            "env": {
                "die": None,
                "sm": None,
                "vram_gb": None,
                "driver": None,
                "cuda": None,
                "nemo": None,
                "torch": None,
                "cpu_sets": None,
            },
            "config": {
                "chunk_ms": self.spec_dict.get("chunk_ms"),
                "frame_ms": self.spec_dict.get("chunk_ms"),
                "corpus_id": self.corpus_id,
                "manifest": self.manifest_name,
                "seed": self.spec_dict.get("seed"),
                "sessions": self.spec_dict.get("sessions"),
                "pacing_profile": self.spec_dict.get("profile"),
                "paced_real_time": True,
                "x_ms": self.spec_dict.get("x_ms"),
                "endpoint": self.spec_dict.get("endpoint"),
                "words": self.spec_dict.get("words", False),
                "compute_dtype": None,
                "att_context": None,
            },
            "result": {
                "streams": None,
                "streams_runs": None,
                "latency_ms": {
                    "first_partial": _block(first_partials),
                    "partial": _block(partials),
                    "final": _block(finals),
                },
                "sessions_started": len(self.sessions),
                "sessions_completed": completed,
                "sessions_failed": failed,
                "chunks_sent": sum(s.chunks for s in self.sessions),
                "partials_received": sum(s.partials_received for s in self.sessions),
                "finals_received": sum(s.finals_received for s in self.sessions),
                "wall_clock_s": self.wall_clock_s,
                "pacing_slip_ms": {
                    "p50": percentile(slips, 50),
                    "p95": percentile(slips, 95),
                    "max": max(slips) if slips else 0.0,
                },
                "invariance": {
                    "hash_1": None,
                    "hash_32a": None,
                    "hash_32b": None,
                    "hash_max": None,
                    "equal": None,
                },
                "eager_step_fraction": None,
                "fraction_of_ceiling": None,
            },
            "sessions": [
                {
                    "session_id": s.session_id,
                    "stream_id": s.stream_id,
                    "server_session_id": s.server_session_id,
                    "started_at_s": s.started_at_s,
                    "chunks": s.chunks,
                    "audio_s": s.audio_s,
                    "first_partial_ms": s.first_partial_ms,
                    "final_ms": s.final_ms,
                    "partial_ms": list(s.partial_ms),
                    "partials_received": s.partials_received,
                    "final_text": s.final_text,
                    "reference_text": s.reference_text,
                    "words": [],
                    "error": s.error,
                }
                for s in self.sessions
            ],
            "date": started_at.isoformat(),
        }
        return stamp_checksum(doc)

    def to_json_dict_v2(
        self,
        *,
        arm_id: str = "a",
        arm_status: str = "measured",
        arm_reason: str | None = None,
        arm_panel: str = "main",
        arm_dtype_class: str = "fp32_tf32",
        arm_kill_rule: int | None = 1,
        harness_tag: str | None = None,
        harness_tree_clean: bool = True,
        harness_container_digest: str | None = None,
        box: dict[str, Any] | None = None,
        gpu: dict[str, Any] | None = None,
        sw: dict[str, Any] | None = None,
        host: dict[str, Any] | None = None,
        cal: dict[str, Any] | None = None,
        sensitivity: dict[str, Any] | None = None,
        ladder: list[dict[str, Any]] | None = None,
        ceiling: dict[str, Any] | None = None,
        declared: dict[str, Any] | None = None,
        session_profile: str | None = None,
        pacing_profile: str | None = None,
        canonical_window: bool = True,
    ) -> dict[str, Any]:
        """Emit the ``vb-results/2`` document: everything v1 carries plus the
        environment, ladder and comparability blocks. Never renames a v1 key."""
        from verbatim_bench import __version__, constants
        from verbatim_bench.canonical import stamp_checksum

        doc = self.to_json_dict()
        doc.pop("checksum", None)
        doc["schema"] = "vb-results/2"
        raw_profile = (
            pacing_profile
            if pacing_profile is not None
            else self.spec_dict.get("profile", "uniform")
        )
        v2_profile = "burst" if raw_profile == "bursty" else str(raw_profile)
        frame_ms = self.spec_dict.get("frame_ms", self.spec_dict.get("chunk_ms"))
        doc["config"]["frame_ms"] = frame_ms
        doc["config"]["pacing_profile"] = v2_profile
        doc["config"]["session_profile"] = (
            session_profile if session_profile is not None else constants.SESSION_PROFILE
        )
        doc["config"]["canonical_window"] = bool(canonical_window)
        doc["harness"]["tag"] = harness_tag or __version__
        doc["harness"]["tree_clean"] = bool(harness_tree_clean)
        doc["harness"]["container_digest"] = harness_container_digest
        doc["arm"]["status"] = arm_status
        doc["arm"]["reason"] = arm_reason
        doc["arm"]["panel"] = arm_panel
        doc["arm"]["dtype_class"] = arm_dtype_class
        doc["arm"]["kill_rule"] = arm_kill_rule
        doc["arm"]["arm_id"] = arm_id
        doc["checkpoint"]["converted_sha256"] = None
        doc["checkpoint"]["converter_version"] = None
        for entry, session in zip(doc["sessions"], self.sessions, strict=True):
            entry["pacing_slip_samples"] = list(session.pacing_slip_ms)
        doc["box"] = (
            dict(box)
            if box is not None
            else {
                "boot_id": None,
                "gpu_uuid": None,
                "cgroup_cpu_max": None,
                "cpuset_effective": None,
                "container_digest": None,
                "id": None,
            }
        )
        doc["gpu"] = dict(gpu) if gpu is not None else None
        doc["sw"] = dict(sw) if sw is not None else {}
        doc["host"] = dict(host) if host is not None else {}
        doc["cal"] = (
            dict(cal)
            if cal is not None
            else {"null_floor_p95_ms": None, "null_floor_client_cpu_pct": None}
        )
        doc["sensitivity"] = dict(sensitivity) if sensitivity is not None else {}
        doc["ladder"] = list(ladder) if ladder is not None else []
        doc["ceiling"] = dict(ceiling) if ceiling is not None else {}
        doc["declared"] = dict(declared) if declared is not None else {}
        return stamp_checksum(doc)


def write_results(result: RunResult, out_dir: Path) -> Path:
    """Write `<out_dir>/results.json` (UTF-8, `indent=2`, `sort_keys=False`,
    trailing newline) and return the path. Creates `out_dir` if needed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path
