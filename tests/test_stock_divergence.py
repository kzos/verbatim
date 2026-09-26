# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``probes/stock_divergence.py`` on the CPU: its comparisons, its controls, its guards, and
its whole ``main()`` driven through a fake backend.

The probe is imported, not parsed: importing it loads no torch, no NeMo, no model and no
data (checked in a subprocess below). Its pure functions are called directly. ``main`` runs
end to end on synthetic recordings through ``FakeProbePipeline``, a stand-in shaped like
``CacheAwareRNNTPipeline`` at the seams the probe uses: ``transcribe_step`` taking frames and
returning, step by step, the finals NeMo would (a leading separator after a stream's first
request, "" on a step that finalises nothing), and the decoding computer and encoder the
probe reads its observations off. Its answer for a recording depends on the batch it is in
exactly where the test says it should. Each record it produces is stamped
``"fake_pipeline": true`` by the probe itself.

What the fake cannot tell us: anything about NeMo. That the attributes ``observe_decoder``
and ``observe_att_context`` read exist on NeMo's own decoding computer and encoder is checked
against the installed NeMo in ``tests/test_word_confidence_switch.py``.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from verbatim.config import ChunkMode
from verbatim.pipelines.nemo_fake import FakeFrame, FakeRequestOptions, FakeTextSegment
from verbatim.pipelines.nemo_runtime import NeMoPipelineSpec, att_context_size

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "probes" / "stock_divergence.py"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location("stock_divergence_under_test", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass resolves its module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sd = _load_probe()


def _side(
    text: str, *timings: tuple[str, float, float, float], nemo: str | None = None
) -> tuple[str, list, str]:
    """(text, timings, nemo_text), as ``transcribe`` returns a side; NeMo's own text is the
    probe's unless given."""
    return text, list(timings), text if nemo is None else nemo


HELLO = _side("Hello world.", ("Hello", 0.0, 0.16, 0.9), ("world.", 0.16, 0.32, 0.8))

#: What the finished stock run's record says it ran: its ``model``, ``chunk_ms``,
#: ``att_context_size``, ``batch`` and ``matmul_precision``, and its ``not_a_row``. The
#: 64-target GPU replay needs the probe to build this spec and say this again.
STOCK_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
STOCK_ATT = (70, 13)
REAL_NOT_A_ROW = (
    "Stock-pipeline probe (alone vs slot 0 of a batch); not the server, not a harness row."
)
FAKE_NOT_A_ROW = "FAKE PIPELINE: a CPU test double produced this record; it measures nothing."


# --- importing the probe touches nothing -------------------------------------------------------


def test_importing_the_probe_loads_no_torch_nemo_model_or_data() -> None:
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(PROBE)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['p'] = m\n"
        "spec.loader.exec_module(m)\n"
        "heavy = sorted(n for n in ('torch', 'nemo', 'datasets', 'soundfile', 'huggingface_hub')"
        " if n in sys.modules)\n"
        "print('HEAVY', heavy)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "bench" / "src")]),
        "CUDA_VISIBLE_DEVICES": "",
    }
    env.pop("OUT", None)
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert "HEAVY []" in done.stdout, done.stdout


# --- the normalisation and the comparisons -----------------------------------------------------


def test_words_is_the_rule_the_published_counts_used() -> None:
    assert sd.words("I'm from the cutter, lying off the coast.") == [
        "i'm",
        "from",
        "the",
        "cutter",
        "lying",
        "off",
        "the",
        "coast",
    ]
    assert sd.words('Yes, sir? We--we did; "no"') == ["yes", "sir", "we", "we", "did", "no"]
    # The typographic apostrophe is NOT mapped: pinned, because the scorer documents it.
    assert sd.words("I\u2019m") == ["i", "m"]


def test_comparable_drops_the_stored_conf_and_keeps_text_start_end() -> None:
    assert sd.comparable(HELLO) == ("Hello world.", [("Hello", 0.0, 0.16), ("world.", 0.16, 0.32)])


def test_timing_of_stores_the_segments_conf_as_the_fourth_element() -> None:
    seg = SimpleNamespace(text=" world. ", start=0.123456, end=0.32, conf=0.91)
    assert sd.timing_of(seg) == ("world.", 0.1235, 0.32, 0.91)
    # The end is rounded like the start: to 4 decimals, the stored format a replay compares.
    late = SimpleNamespace(text="x", start=0.5, end=0.567891, conf=0.0)
    assert sd.timing_of(late) == ("x", 0.5, 0.5679, 0.0)
    zero = SimpleNamespace(text="a", start=0, end=1, conf=0.0)
    assert sd.timing_of(zero) == ("a", 0, 1, 0.0)


@pytest.mark.parametrize(
    ("b_side", "kind"),
    [
        (HELLO, "identical"),
        (_side("Hello world", ("Hello", 0.0, 0.16, 0.9), ("world", 0.16, 0.32, 0.8)), "text"),
        # Text differs AND timings differ: text wins.
        (_side("Hello there.", ("Hello", 0.0, 0.2, 0.9), ("there.", 0.2, 0.32, 0.8)), "text"),
        (_side("Hello world.", ("Hello", 0.0, 0.16, 0.9), ("world.", 0.24, 0.32, 0.8)), "timing"),
        (_side("Hello world.", ("Hello", 0.0, 0.16, 0.9), ("world.", 0.16, 0.40, 0.8)), "timing"),
        (_side("Hello world.", ("Hello", 0.0, 0.16, 0.9)), "timing"),
        (_side("Hello world.", ("Hello", 0.0, 0.16, 0.9), ("World.", 0.16, 0.32, 0.8)), "timing"),
        (
            _side("Hello world.", ("Hello", 0.0, 0.16, 0.1), ("world.", 0.16, 0.32, 0.8)),
            "confidence",
        ),
    ],
    ids=[
        "identical",
        "text",
        "text-and-timing",
        "start-moved",
        "end-moved",
        "word-missing",
        "timing-word-differs",
        "conf-only",
    ],
)
def test_classify_names_each_kind_of_pair(b_side: tuple[str, list], kind: str) -> None:
    assert sd.classify(HELLO, b_side) == kind
    assert sd.classify(b_side, HELLO) == kind


def test_the_repeat_comparison_reads_timings_and_not_conf() -> None:
    moved = _side("Hello world.", ("Hello", 0.0, 0.16, 0.9), ("world.", 0.24, 0.32, 0.8))
    assert sd.same_in_one_shape(HELLO, HELLO)
    assert not sd.same_in_one_shape(HELLO, moved), "same text, moved timing: not identical"
    other_conf = _side("Hello world.", ("Hello", 0.0, 0.16, 0.2), ("world.", 0.16, 0.32, 0.3))
    assert sd.same_in_one_shape(HELLO, other_conf)
    assert not sd.same_in_one_shape(HELLO, _side("Hello, world.", *HELLO[1]))


@pytest.mark.parametrize(
    ("rep", "verdict"),
    [
        ({"alone_identical": 8, "batch_identical": 8, "checked": 8}, sd.SAME_SHAPE_VERDICT),
        ({"alone_identical": 7, "batch_identical": 8, "checked": 8}, sd.NOT_DETERMINISTIC_VERDICT),
        ({"alone_identical": 8, "batch_identical": 7, "checked": 8}, sd.NOT_DETERMINISTIC_VERDICT),
        ({"alone_identical": 0, "batch_identical": 0, "checked": 0}, sd.NOT_CHECKED_VERDICT),
    ],
    ids=["identical", "alone-differs", "batch-differs", "nothing-checked"],
)
def test_the_repeat_verdict_follows_the_counts(rep: dict[str, int], verdict: str) -> None:
    assert sd.repeat_verdict(rep) == verdict
    assert sd.SAME_SHAPE_VERDICT == "run-to-run identical in the same shape"


def test_count_nonzero_conf_reads_the_conf_and_not_the_times() -> None:
    off = [("a", 0.0, 0.16, 0.0), ("b", 0.16, 0.32, 0.0)]
    assert sd.count_nonzero_conf(off) == 0
    assert sd.count_nonzero_conf([("a", 0.0, 0.0, 0.7), ("b", 0.1, 0.2, 0.0)]) == 1


def _counts(**arms: tuple[int, int]) -> dict[str, dict[str, int]]:
    return {
        arm: {"text_divergent": text, "timing_only_divergent": timing}
        for arm, (text, timing) in arms.items()
    }


def test_the_positive_control_counts_text_or_timing_in_the_two_shape_arms_only() -> None:
    every = ["ragged", "equalised", "fixed"]
    timing_only = sd.positive_control(_counts(ragged=(0, 1), equalised=(0, 0), fixed=(0, 0)), every)
    assert timing_only.startswith("present: ragged diverged"), "a moved timing is a divergence"
    text_only = sd.positive_control(_counts(ragged=(0, 0), equalised=(2, 0), fixed=(0, 0)), every)
    assert text_only.startswith("present: equalised diverged")
    both = sd.positive_control(_counts(ragged=(1, 0), equalised=(0, 3), fixed=(0, 0)), every)
    assert both.startswith("present: ragged, equalised diverged")
    fixed_only = sd.positive_control(_counts(ragged=(0, 0), equalised=(0, 0), fixed=(4, 5)), every)
    assert fixed_only.startswith("ABSENT"), "the one-shape arm is not the positive control"
    alone = sd.positive_control(_counts(equalised=(0, 2)), ["equalised"])
    assert alone.startswith("present: equalised diverged")


def test_the_confidence_stamp_is_what_the_decoder_keeps_not_what_was_asked() -> None:
    asked_off_kept_on = sd.word_confidence_stamp("off", _seen(True, False), 0)
    assert asked_off_kept_on == {
        "mode_requested": "off",
        "decoder_step_confidence": True,
        "nonzero_conf_words_on_guard_recording": 0,
    }
    asked_on_kept_off = sd.word_confidence_stamp("paper-best", _seen(False, True), 7)
    assert asked_on_kept_off["decoder_step_confidence"] is False
    assert asked_on_kept_off["nonzero_conf_words_on_guard_recording"] == 7


# --- one recording into its arm ----------------------------------------------------------------


def _one(a_side: tuple[str, list], b_side: tuple[str, list]) -> tuple[str, dict[str, Any]]:
    arm = sd.new_arm()
    kind = sd.record_pair(arm, 4, "rid-4", a_side, b_side)
    return kind, arm


def test_a_timing_only_pair_is_counted_and_listed_with_text_differs_false() -> None:
    moved = _side("Hello world.", ("Hello", 0.0, 0.16, 0.9), ("world.", 0.24, 0.32, 0.8))
    kind, arm = _one(HELLO, moved)
    assert kind == "timing"
    assert (arm["checked"], arm["text_divergent"], arm["timing_only_divergent"]) == (1, 0, 1)
    assert arm["confidence_only_divergent"] == 0
    assert arm["transcripts"] == {"rid-4": {"a": "Hello world."}}
    assert arm["divergences"] == [
        {
            "n": 4,
            "librispeech_id": "rid-4",
            "text_differs": False,
            "a": "Hello world.",
            "b": "Hello world.",
            "a_timings": HELLO[1],
            "b_timings": moved[1],
        }
    ]


def test_a_text_pair_stores_both_texts() -> None:
    other = _side("Hello word.", ("Hello", 0.0, 0.16, 0.9), ("word.", 0.16, 0.32, 0.8))
    kind, arm = _one(HELLO, other)
    assert kind == "text"
    assert (arm["text_divergent"], arm["timing_only_divergent"]) == (1, 0)
    assert arm["transcripts"]["rid-4"] == {"a": "Hello world.", "b": "Hello word."}
    assert [d["text_differs"] for d in arm["divergences"]] == [True]


def test_a_confidence_only_pair_is_counted_apart_and_is_not_a_divergence() -> None:
    other = _side("Hello world.", ("Hello", 0.0, 0.16, 0.1), ("world.", 0.16, 0.32, 0.8))
    kind, arm = _one(HELLO, other)
    assert kind == "confidence"
    assert (arm["text_divergent"], arm["timing_only_divergent"]) == (0, 0)
    assert arm["confidence_only_divergent"] == 1
    assert arm["divergences"] == []
    assert arm["transcripts"]["rid-4"] == {"a": "Hello world."}


def test_an_identical_pair_is_still_kept_in_every_recording() -> None:
    kind, arm = _one(HELLO, HELLO)
    assert kind == "identical"
    assert arm["divergences"] == [] and arm["confidence_only_divergent"] == 0
    assert (arm["nemo_text_divergent"], arm["nemo_text_divergent_hidden"]) == (0, 0)
    assert arm["every_recording"] == {
        "rid-4": {
            "a_text": "Hello world.",
            "b_text": "Hello world.",
            "a_words": [["Hello", 0.0, 0.16, 0.9], ["world.", 0.16, 0.32, 0.8]],
            "b_words": [["Hello", 0.0, 0.16, 0.9], ["world.", 0.16, 0.32, 0.8]],
            "a_nemo_text": "Hello world.",
            "b_nemo_text": "Hello world.",
        }
    }


@pytest.mark.parametrize(
    ("b_side", "kind", "nemo_counts"),
    [
        # The join hides it: the probe's text and timings agree, NeMo's own texts do not.
        (_side("Hello world.", *HELLO[1], nemo="Helloworld."), "identical", (1, 1)),
        # NeMo's texts differ and so do the probe's timings; the probe's texts still agree.
        (
            _side(
                "Hello world.",
                ("Hello", 0.0, 0.16, 0.9),
                ("world.", 0.24, 0.32, 0.8),
                nemo="Helloworld.",
            ),
            "timing",
            (1, 1),
        ),
        # Both see a text difference: not hidden.
        (_side("Hello word.", *HELLO[1], nemo="Hello word."), "text", (1, 0)),
        # The join makes a text difference NeMo does not see: counted by the join, not NeMo.
        (_side("Hel lo world.", *HELLO[1], nemo="Hello world."), "text", (0, 0)),
    ],
    ids=["hidden-by-the-join", "hidden-behind-a-timing", "both-see-it", "made-by-the-join"],
)
def test_nemos_own_text_is_counted_beside_the_join_and_classified_by_nothing(
    b_side: tuple[str, list, str], kind: str, nemo_counts: tuple[int, int]
) -> None:
    got, arm = _one(HELLO, b_side)
    assert got == kind, "NeMo's own text does not change what the pair is called"
    assert (arm["nemo_text_divergent"], arm["nemo_text_divergent_hidden"]) == nemo_counts
    entry = arm["every_recording"]["rid-4"]
    assert (entry["a_nemo_text"], entry["b_nemo_text"]) == (HELLO[2], b_side[2])
    assert (entry["a_text"], entry["b_text"]) == (HELLO[0], b_side[0])


# --- what the built decoder does, against what was asked ---------------------------------------


def _pipeline_with(computer: Any) -> Any:
    decoding = SimpleNamespace(decoding=SimpleNamespace(decoding_computer=computer))
    return SimpleNamespace(asr_model=SimpleNamespace(asr_model=SimpleNamespace(decoding=decoding)))


def test_observe_decoder_reads_the_computer_the_decode_calls() -> None:
    mode = SimpleNamespace(value="no_while_loops")
    seen = sd.observe_decoder(
        _pipeline_with(SimpleNamespace(preserve_step_confidence=True, cuda_graphs_mode=mode))
    )
    assert seen == {
        "decoder_step_confidence": True,
        "decoder_graphs": True,
        "decoder_graphs_mode": "no_while_loops",
    }
    off = sd.observe_decoder(
        _pipeline_with(SimpleNamespace(preserve_step_confidence=False, cuda_graphs_mode=None))
    )
    assert off == {
        "decoder_step_confidence": False,
        "decoder_graphs": False,
        "decoder_graphs_mode": None,
    }


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        (SimpleNamespace(), "no pipeline.asr_model;"),
        (_pipeline_with(None), "decoding.decoding_computer;"),
        (_pipeline_with(SimpleNamespace(cuda_graphs_mode=None)), "is None, not a bool"),
        (
            _pipeline_with(SimpleNamespace(preserve_step_confidence=1, cuda_graphs_mode=None)),
            "is 1, not a bool",
        ),
        (
            _pipeline_with(
                SimpleNamespace(preserve_step_confidence="False", cuda_graphs_mode=None)
            ),
            "is 'False', not a bool",
        ),
        (_pipeline_with(SimpleNamespace(preserve_step_confidence=True)), "no cuda_graphs_mode"),
    ],
    ids=[
        "no-wrapper",
        "no-computer",
        "no-confidence-flag",
        "confidence-flag-an-int",
        "confidence-flag-a-string",
        "no-graphs-mode",
    ],
)
def test_observe_decoder_refuses_what_it_cannot_read(pipeline: Any, message: str) -> None:
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED") as info:
        sd.observe_decoder(pipeline)
    assert message in str(info.value)


def _seen(step_confidence: bool, graphs: bool) -> dict[str, Any]:
    return {
        "decoder_step_confidence": step_confidence,
        "decoder_graphs": graphs,
        "decoder_graphs_mode": "full_graph" if graphs else None,
    }


@pytest.mark.parametrize("mode", ["off", "nemo-shipped", "paper-best"])
def test_the_decoder_guard_passes_what_was_asked_and_refuses_the_rest(mode: str) -> None:
    on = mode != "off"
    sd.refuse_unrequested_decoder(mode, False, _seen(on, False))
    sd.refuse_unrequested_decoder(mode, True, _seen(on, True))
    with pytest.raises(SystemExit, match="preserve_step_confidence is"):
        sd.refuse_unrequested_decoder(mode, False, _seen(not on, False))
    with pytest.raises(SystemExit, match="decoder graphs False"):
        sd.refuse_unrequested_decoder(mode, False, _seen(on, True))
    with pytest.raises(SystemExit, match="decoder graphs True"):
        sd.refuse_unrequested_decoder(mode, True, _seen(on, False))


def test_the_confidence_guard_refuses_a_count_that_contradicts_the_mode() -> None:
    sd.refuse_contradicting_confidence("off", 0)
    with pytest.raises(SystemExit, match="off is not off"):
        sd.refuse_contradicting_confidence("off", 1)
    for mode in ("nemo-shipped", "paper-best"):
        sd.refuse_contradicting_confidence(mode, 1)
        with pytest.raises(SystemExit, match="did not reach the words"):
            sd.refuse_contradicting_confidence(mode, 0)


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(SystemExit, match="unknown arms"):
        sd.settings_from({"OUT": "x", "ARMS": "ragged,fixd"}, ["probe"])


def test_the_settings_keep_their_defaults() -> None:
    s = sd.settings_from({"OUT": "x"}, ["probe"])
    assert (s.n_targets, s.batch, s.chunk_ms, s.repeat) == (2939, 32, 1120, 8)
    assert (s.dtypes, s.arms, s.matmul) == (
        ["bfloat16"],
        ["ragged", "equalised", "fixed"],
        "highest",
    )
    assert (s.word_confidence, s.left, s.model_revision) == ("off", None, "")
    assert s.model == "nvidia/nemotron-speech-streaming-en-0.6b"


def test_the_default_settings_build_the_stock_runs_spec() -> None:
    """Every field of the spec, compared whole: the defaults give what the stock run built."""
    s = sd.settings_from({"OUT": "x"}, ["probe"])
    chunk = ChunkMode(s.chunk_ms)
    att = att_context_size(s.model, chunk, left=s.left)
    assert tuple(att) == STOCK_ATT
    assert sd.spec_for(s, "bfloat16", chunk, att) == NeMoPipelineSpec(
        model=STOCK_MODEL,
        chunk=ChunkMode(1120),
        att_context=STOCK_ATT,
        num_slots=256,
        batch_size=32,
        compute_dtype="bfloat16",
        matmul_precision="highest",
        word_confidence="off",
    )
    # Past 64 rows the slot pool grows with the batch: four slots a row.
    big = sd.settings_from({"OUT": "x", "BATCH": "128"}, ["probe"])
    assert sd.spec_for(big, "float32", chunk, att).num_slots == 512


# --- the real backend's own code, without its imports ------------------------------------------


class _Dataset(list):
    """Shaped like a streaming ``datasets`` split: iterable, with ``cast_column``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self.cast: list[tuple[str, Any]] = []

    def cast_column(self, column: str, feature: Any) -> _Dataset:
        self.cast.append((column, feature))
        return self


def _bare_backend(rows: list[dict[str, Any]], rate: int) -> tuple[Any, _Dataset, list[Any]]:
    """A ``NeMoBackend`` whose ``__init__`` never ran, so nothing heavy is imported: its
    dataset loader, ``Audio`` and ``soundfile`` are stand-ins, ``load`` is its own."""
    backend = object.__new__(sd.NeMoBackend)
    dataset = _Dataset(rows)
    calls: list[Any] = []

    def load_dataset(*args: Any, **kwargs: Any) -> _Dataset:
        calls.append((args, kwargs))
        return dataset

    def read(buffer: Any, dtype: str) -> tuple[np.ndarray, int]:
        assert dtype == "float32"
        return np.frombuffer(buffer.read(), dtype=np.float32).copy(), rate

    backend._load_dataset = load_dataset
    backend._audio = lambda **kwargs: ("Audio", kwargs)
    backend._sf = SimpleNamespace(read=read)
    return backend, dataset, calls


def test_the_real_backend_loads_test_other_undecoded_and_stops_at_the_count(
    tmp_path: Path,
) -> None:
    on_disk = tmp_path / "b.flac"
    on_disk.write_bytes(np.array([1.0], np.float32).tobytes())
    rows = [
        {"id": "a", "text": "HELLO THERE", "audio": {"bytes": np.float32([0.25, -0.5]).tobytes()}},
        {"id": "b", "text": "Second", "audio": {"bytes": None, "path": str(on_disk)}},
        {"id": "c", "text": "never read", "audio": {"bytes": np.float32([9.0]).tobytes()}},
    ]
    backend, dataset, calls = _bare_backend(rows, 16000)
    pool, meta = backend.load(2)
    assert calls == [(("openslr/librispeech_asr", "other"), {"split": "test", "streaming": True})]
    assert dataset.cast == [("audio", ("Audio", {"decode": False}))]
    assert [p.tolist() for p in pool] == [[0.25, -0.5], [1.0]]
    assert all(p.dtype == np.float32 for p in pool)
    assert meta == [{"id": "a", "reference": "hello there"}, {"id": "b", "reference": "second"}]


def test_the_real_backend_refuses_audio_that_is_not_16khz() -> None:
    rows = [{"id": "r-8k", "text": "x", "audio": {"bytes": np.zeros(4, np.float32).tobytes()}}]
    backend, _, _ = _bare_backend(rows, 8000)
    with pytest.raises(SystemExit, match=r"\[guard\] r-8k is 8000 Hz, not 16000"):
        backend.load(1)


def test_the_real_backend_hands_float32_cpu_tensors_under_inference_mode() -> None:
    torch = pytest.importorskip("torch")
    backend = object.__new__(sd.NeMoBackend)
    backend._torch = torch
    piece = np.float32([0.5, -0.25, 0.0])
    tensor = backend.samples(piece)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.tolist() == [0.5, -0.25, 0.0]
    with backend.no_grad():
        assert torch.is_inference_mode_enabled()


# --- the save is atomic ------------------------------------------------------------------------


def test_a_save_that_fails_leaves_the_previous_record_whole(tmp_path: Path) -> None:
    out = str(tmp_path / "record.json")
    sd.save({"runs": {"bfloat16": {"checked": 1}}}, out)
    with pytest.raises(TypeError):
        sd.save({"runs": {"bfloat16": {"checked": 2, "bad": object()}}}, out)
    assert json.loads(Path(out).read_text()) == {"runs": {"bfloat16": {"checked": 1}}}


# --- main() end to end, through a fake backend -------------------------------------------------

#: Samples per chunk the fake pipeline reports (0.01 s at 1600 Hz), and seconds per frame it
#: stamps on the words it emits.
N = 16
STEP_S = 0.08


@dataclass
class _Stream:
    rid: int | None
    frames: int = 0
    rows: int = 0
    touched: bool = False
    pending: list[list[Any]] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    samples: list[np.ndarray] = field(default_factory=list)


class FakeProbePipeline:
    """The seam ``transcribe`` uses, with an answer that depends on the batch where told to.

    Each frame with any nonzero valid sample makes one word, ``r{rid}w{frame}``, timed from
    ``frame * STEP_S`` to ``(frame + 1) * STEP_S``. The recording id is the first sample minus
    one (silence has none).

    The words come out one step at a time, the way NeMo hands them out
    (``TranscribeStepOutput.from_state``, ``base_pipeline.py:88-100``): a step's
    ``final_transcript`` and ``final_segments`` hold only the words that step finalised, the
    first of them with a leading separator " " when the step is not the stream's first request,
    and a step that finalises nothing gives "" and no segments. This fake finalises a stream's
    words on its even frames and on its last frame, so the step after each even frame is empty
    unless it is the last: a 3-frame recording's finals are ``["r1w0", "", " r1w1 r1w2"]``.

    A recording's behaviour applies to its FIRST word, decided in its first step: ``shape_*``
    when that step held more than one row, ``content_*`` when another row in it carried sound.
    ``*_text`` changes the word, ``*_timing`` moves its end, ``*_conf`` changes its confidence.
    ``flaky_timing`` moves the first word's end on every call, so the same input twice is not
    the same answer; ``flaky_batch_timing`` does so only in a step with more than one row.

    ``bare`` names recordings whose steps carry no segments; ``conf_for`` gives a recording a
    confidence of its own. ``glue`` names recordings whose finals never carry the separator,
    the way NeMo gives a step's final none when its first word continues the last word the
    stream finalised (``concat_with_space`` False, ``state.py:350-354``): a 3-frame glued
    recording's finals are ``["r1w0", "", "r1w1 r1w2"]``. A recording whose behaviour is
    ``shape_glue`` hands out its finals that way only in a step with more than one row, so its
    words, timings and the probe's text are the same alone and batched while NeMo's own
    concatenation is not. It keeps what it was handed: every
    sample of every stream, whole frames padding included (``samples``, by stream id), the
    dtype of every frame's samples, and each stream's finals (``finals``, by stream id).
    """

    chunk_size_in_secs = 0.01
    sample_rate = 1600

    def __init__(
        self,
        behaviour: dict[int, str],
        *,
        conf: float,
        step_confidence: bool,
        graphs_mode: Any = None,
        segments: bool = True,
        bare: frozenset[int] = frozenset(),
        conf_for: dict[int, float] | None = None,
        glue: frozenset[int] = frozenset(),
    ) -> None:
        self.behaviour = behaviour
        self.conf = conf
        self.segments = segments
        self.bare = bare
        self.glue = glue
        self.conf_for = conf_for or {}
        computer = SimpleNamespace(
            preserve_step_confidence=step_confidence, cuda_graphs_mode=graphs_mode
        )
        self.asr_model = _pipeline_with(computer).asr_model
        self.active: dict[int, _Stream] = {}
        #: One list per ``transcribe`` call: (stream id, recording id, frames seen).
        self.calls: list[list[tuple[int, int | None, int]]] = []
        self.samples: dict[int, np.ndarray] = {}
        self.finals: dict[int, list[str]] = {}
        self.dtypes: set[Any] = set()
        self.flaky = 0

    def transcribe_step(self, frames: list[FakeFrame]) -> list[Any]:
        if not self.active:
            self.calls.append([])
        sound = [bool(np.any(np.asarray(f.samples)[: f.length] != 0)) for f in frames]
        outs = []
        for idx, f in enumerate(frames):
            if f.is_first:
                first = float(np.asarray(f.samples)[0])
                self.active[f.stream_id] = _Stream(rid=round(first) - 1 if first else None)
            st = self.active[f.stream_id]
            st.rows = max(st.rows, len(frames))
            st.touched |= any(s for j, s in enumerate(sound) if j != idx)
            st.samples.append(np.array(f.samples, copy=True))
            self.dtypes.add(np.asarray(f.samples).dtype)
            if sound[idx]:
                start = st.frames * STEP_S
                conf = self.conf_for.get(st.rid, self.conf) if st.rid is not None else self.conf
                word = [f"r{st.rid}w{st.frames}", start, start + STEP_S, conf]
                if st.frames == 0:
                    self._behave(st, word)
                st.pending.append(word)
            done: list[list[Any]] = []
            if st.frames % 2 == 0 or f.is_last:
                done, st.pending = st.pending, []
            st.frames += 1
            out = self._step_output(f, st, done)
            st.finals.append(out.final_transcript)
            if f.is_last:
                del self.active[f.stream_id]
                self.calls[-1].append((f.stream_id, st.rid, st.frames))
                self.samples[f.stream_id] = np.concatenate(st.samples)
                self.finals[f.stream_id] = st.finals
            outs.append(out)
        return outs

    def _step_output(self, f: FakeFrame, st: _Stream, done: list[list[Any]]) -> Any:
        if not done:
            return SimpleNamespace(stream_id=f.stream_id, final_transcript="", final_segments=[])
        glued = st.rid in self.glue or (self.behaviour.get(st.rid) == "shape_glue" and st.rows > 1)
        separator = "" if f.is_first or glued else " "
        segs = [FakeTextSegment(w[0], w[1], w[2], w[3]) for w in done]
        segs[0].text = separator + segs[0].text
        if not self.segments or st.rid in self.bare:
            segs = []
        text = separator + " ".join(w[0] for w in done)
        return SimpleNamespace(stream_id=f.stream_id, final_transcript=text, final_segments=segs)

    def _behave(self, st: _Stream, word: list[Any]) -> None:
        kind = self.behaviour.get(st.rid, "plain") if st.rid is not None else "plain"
        if kind.startswith("shape_") and st.rows > 1:
            self._apply(kind, word)
        if kind.startswith("content_") and st.touched:
            self._apply(kind, word)
        if kind == "flaky_timing" or (kind == "flaky_batch_timing" and st.rows > 1):
            word[2] += 0.001 * self.flaky
            self.flaky += 1

    @staticmethod
    def _apply(kind: str, word: list[Any]) -> None:
        if kind.endswith("_text"):
            word[0] = "changed"
        elif kind.endswith("_timing"):
            word[2] += 0.04
        elif kind.endswith("_conf"):
            word[3] = 0.5


#: ``FakeBackend.encoder`` default: the built encoder carries the context the spec asked for,
#: as NeMo's ``set_default_att_context_size`` leaves it.
AS_ASKED = object()
#: What the fake backend says about the machine: every field its own value, so a record field
#: copied from the wrong one is seen.
#: The two card readings name one card only once nvidia-smi's prefix and the case are set aside.
FAKE_PROVENANCE = {
    "machine": "FAKE: CPU test double",
    "capability": "FAKE capability",
    "driver_and_uuid": "FAKE driver",
    "gpu_uuid": "GPU-FAKE-0000-card",
    "gpu_uuid_torch": "fake-0000-card",
    "torch": "FAKE torch",
    "torch_cuda": "FAKE torch cuda",
}
FAKE_NEMO = "FAKE nemo version"


class FakeBackend:
    frame = FakeFrame
    request_options = FakeRequestOptions

    def __init__(self, pool: list[np.ndarray], meta: list[dict[str, str]], make: Any) -> None:
        self.pool, self.meta, self.make = pool, meta, make
        self.revision = "fake0revision"
        self.encoder: Any = AS_ASKED
        self.card: dict[str, Any] = {}
        self.on_build: Any = None
        self.specs: list[Any] = []
        self.pipelines: list[FakeProbePipeline] = []
        self.loaded: list[int] = []
        self.released = 0

    def resolved_revision(self, model: str) -> str:
        return self.revision

    def load(self, count: int) -> tuple[list[np.ndarray], list[dict[str, str]]]:
        self.loaded.append(count)
        return self.pool[:count], self.meta[:count]

    def provenance(self) -> dict[str, Any]:
        return {**FAKE_PROVENANCE, **self.card}

    def build(self, spec: Any) -> FakeProbePipeline:
        if self.on_build is not None:
            self.on_build()
        self.specs.append(spec)
        pipeline = self.make()
        pipeline.asr_model.asr_model.encoder = (
            SimpleNamespace(att_context_size=list(spec.att_context))
            if self.encoder is AS_ASKED
            else self.encoder
        )
        self.pipelines.append(pipeline)
        return pipeline

    def nemo_version(self) -> str:
        return FAKE_NEMO

    def samples(self, piece: np.ndarray) -> np.ndarray:
        return piece

    def no_grad(self) -> Any:
        return contextlib.nullcontext()

    def release(self) -> None:
        self.released += 1


def _frames_of(r: int) -> int:
    return 2 + r % 3


def _text_of(r: int) -> str:
    return " ".join(f"r{r}w{k}" for k in range(_frames_of(r)))


def recordings(count: int) -> tuple[list[np.ndarray], list[dict[str, str]]]:
    """Ragged synthetic recordings: 2 to 4 frames, the last one half full, first sample r + 1."""
    pool, meta = [], []
    for r in range(count):
        frames = _frames_of(r)
        audio = np.full((frames - 1) * N + N // 2, 0.5, dtype=np.float32)
        audio[0] = r + 1
        pool.append(audio)
        meta.append({"id": f"rec-{r}", "reference": _text_of(r)})
    return pool, meta


TARGETS, BATCH, REPEAT = 8, 3, 2


def _run(
    tmp_path: Path,
    behaviour: dict[int, str],
    *,
    mode: str = "off",
    conf: float | None = None,
    step_confidence: bool | None = None,
    graphs_mode: Any = None,
    segments: bool = True,
    bare: frozenset[int] = frozenset(),
    conf_for: dict[int, float] | None = None,
    glue: frozenset[int] = frozenset(),
    encoder: Any = AS_ASKED,
    targets: int = TARGETS,
    extra_env: dict[str, str] | None = None,
    meta_edit: Any = None,
    pool_edit: Any = None,
    pool_size: int = TARGETS + BATCH,
    card: dict[str, Any] | None = None,
    on_build: Any = None,
    backends: list[FakeBackend] | None = None,
) -> tuple[dict[str, Any], FakeBackend]:
    """One run of ``main`` through the fake. ``backends`` receives the backend before the run,
    so a test can look at it after a refusal."""
    on = mode != "off"
    pool, meta = recordings(pool_size)
    if meta_edit is not None:
        meta_edit(meta)
    if pool_edit is not None:
        pool_edit(pool)
    backend = FakeBackend(
        pool,
        meta,
        lambda: FakeProbePipeline(
            behaviour,
            conf=(0.9 if on else 0.0) if conf is None else conf,
            step_confidence=on if step_confidence is None else step_confidence,
            graphs_mode=graphs_mode,
            segments=segments,
            bare=bare,
            conf_for=conf_for,
            glue=glue,
        ),
    )
    backend.encoder = encoder
    backend.card = card or {}
    backend.on_build = on_build
    if backends is not None:
        backends.append(backend)
    env = {
        "OUT": str(tmp_path / "record.json"),
        "BATCH": str(BATCH),
        "REPEAT": str(REPEAT),
        "WORD_CONFIDENCE": mode,
        "CUDA_VISIBLE_DEVICES": "",
        **(extra_env or {}),
    }
    record = sd.main(["stock_divergence.py", str(targets)], env, backend)
    return record, backend


EVERY_KIND = {
    2: "shape_text",
    3: "shape_timing",
    4: "shape_conf",
    5: "content_text",
    6: "content_timing",
    7: "content_conf",
}


def test_main_counts_every_kind_in_every_arm_and_keeps_every_recording(tmp_path: Path) -> None:
    record, backend = _run(tmp_path, EVERY_KIND, mode="nemo-shipped")
    on_disk = json.loads((tmp_path / "record.json").read_text())
    assert on_disk == json.loads(json.dumps(record)), "the saved record is the returned one"
    assert not (tmp_path / "record.json.tmp").exists()

    assert next(iter(on_disk)) == "fake_pipeline" and on_disk["fake_pipeline"] is True
    assert on_disk["not_a_row"] == FAKE_NOT_A_ROW
    run = on_disk["runs"]["bfloat16"]
    counts = {
        arm: (
            a["checked"],
            a["text_divergent"],
            a["timing_only_divergent"],
            a["confidence_only_divergent"],
        )
        for arm, a in run["arms"].items()
    }
    # shape_* diverge wherever the two sides are different shapes; content_* wherever the
    # neighbours' sound differs, which is every arm, fixed included.
    assert counts == {
        "ragged": (8, 2, 2, 2),
        "equalised": (8, 2, 2, 2),
        "fixed": (8, 1, 1, 1),
    }
    ragged = run["arms"]["ragged"]
    # "n" is the recording's index among the targets, which is what the replay compares with
    # the stock record: rec-r is target r.
    listed = [(d["n"], d["librispeech_id"], d["text_differs"]) for d in ragged["divergences"]]
    assert listed == [
        (2, "rec-2", True),
        (3, "rec-3", False),
        (5, "rec-5", True),
        (6, "rec-6", False),
    ]
    assert [
        (d["n"], d["librispeech_id"], d["text_differs"])
        for d in run["arms"]["equalised"]["divergences"]
    ] == listed
    fixed = run["arms"]["fixed"]
    assert [(d["n"], d["librispeech_id"], d["text_differs"]) for d in fixed["divergences"]] == [
        (5, "rec-5", True),
        (6, "rec-6", False),
    ]
    ids = [f"rec-{r}" for r in range(TARGETS)]
    for arm in run["arms"].values():
        assert list(arm["every_recording"]) == ids
        assert list(arm["transcripts"]) == ids
        for rid, entry in arm["every_recording"].items():
            stored = arm["transcripts"][rid]
            assert entry["a_text"] == stored["a"]
            assert entry["b_text"] == stored.get("b", stored["a"])
            assert all(len(w) == 4 for w in entry["a_words"] + entry["b_words"])
    # The confidences are the pipeline's, stored per word, for a recording that did not diverge.
    shape_conf = ragged["every_recording"]["rec-4"]
    assert [w[3] for w in shape_conf["a_words"]] == [0.9, 0.9, 0.9]
    assert [w[3] for w in shape_conf["b_words"]] == [0.5, 0.9, 0.9]
    assert ragged["every_recording"]["rec-0"]["a_words"] == [
        ["r0w0", 0.0, 0.08, 0.9],
        ["r0w1", 0.08, 0.16, 0.9],
    ]

    assert run["repeat"] == {"alone_identical": 2, "batch_identical": 2, "checked": 2}
    assert run["repeat_verdict"] == "run-to-run identical in the same shape"
    assert run["positive_control"].startswith("present: ragged, equalised diverged")
    # No recording here is glued: NeMo's own text reads as the probe's on every side, so it
    # differs exactly where the probe's text does and hides nothing.
    for arm in run["arms"].values():
        assert arm["nemo_text_divergent"] == arm["text_divergent"]
        assert arm["nemo_text_divergent_hidden"] == 0
        for entry in arm["every_recording"].values():
            assert (entry["a_nemo_text"], entry["b_nemo_text"]) == (
                entry["a_text"],
                entry["b_text"],
            )

    observed = {
        "mode_requested": "nemo-shipped",
        "decoder_step_confidence": True,
        "nonzero_conf_words_on_guard_recording": 2,
    }
    assert on_disk["word_confidence_observed"] == observed
    assert run["word_confidence_observed"] == observed
    assert on_disk["decoder_graphs_observed"] is False
    assert run["decoder_graphs_observed"] is False
    assert on_disk["word_confidence"] == "nemo-shipped"
    assert on_disk["use_cuda_graphs"] is False
    assert on_disk["timing_channel"] == {
        "segments_on_guard_recording": 2,
        "words_on_guard_recording": 2,
        "sample": [["r0w0", 0.0, 0.08, 0.9], ["r0w1", 0.08, 0.16, 0.9]],
        "nonzero_conf_segments_on_guard_recording": 2,
    }
    assert [spec.word_confidence for spec in backend.specs] == ["nemo-shipped"]
    assert backend.released == 1
    assert on_disk["references"] == {
        f"rec-{r}": recordings(TARGETS)[1][r]["reference"] for r in range(TARGETS)
    }


ALL_ARMS = ("ragged", "equalised", "fixed")


def _expected_calls(
    repeat: int = REPEAT, arms: tuple[str, ...] = ALL_ARMS, pool: int = TARGETS + BATCH
) -> list[list[tuple[int | None, int]]]:
    """The order of ``transcribe`` calls the probe has always made, and what each one holds:
    (recording id, frames) per row, row 0 first. None is a silent row. ``pool`` is how many
    recordings the data held: fewer than TARGETS + BATCH wraps the last targets' neighbours."""
    calls: list[list[tuple[int | None, int]]] = [[(0, _frames_of(0)), (1, _frames_of(1))]]

    def nbrs(i: int) -> list[int]:
        return [(i + 1 + k) % pool for k in range(BATCH - 1)]

    for i in range(repeat):
        alone = [(i, _frames_of(i))]
        batch = alone + [(j, _frames_of(j)) for j in nbrs(i)]
        calls += [alone, alone, batch, batch]
    for i in range(TARGETS):
        common = max(_frames_of(j) for j in [i, *nbrs(i)])
        if "ragged" in arms:
            calls += [
                [(i, _frames_of(i))],
                [(i, _frames_of(i))] + [(j, _frames_of(j)) for j in nbrs(i)],
            ]
        if "equalised" in arms or "fixed" in arms:
            calls.append([(i, common)] + [(j, common) for j in nbrs(i)])
        if "equalised" in arms:
            calls.append([(i, common)])
        if "fixed" in arms:
            calls.append([(i, common)] + [(None, common)] * (BATCH - 1))
    return calls


def test_main_makes_the_same_calls_in_the_same_order_with_the_same_rows(tmp_path: Path) -> None:
    _, backend = _run(tmp_path, {})
    (pipeline,) = backend.pipelines
    seen = [[(rid, frames) for _, rid, frames in sorted(call)] for call in pipeline.calls]
    assert seen == _expected_calls()
    # Stream ids run on from 1 across every call, in row order.
    ids = [sid for call in pipeline.calls for sid, _, _ in sorted(call)]
    assert ids == list(range(1, len(ids) + 1))


def test_main_with_confidence_off_counts_no_confidence_divergence(tmp_path: Path) -> None:
    behaviour = {2: "shape_text", 3: "shape_timing", 5: "content_text", 6: "content_timing"}
    record, backend = _run(tmp_path, behaviour)
    run = record["runs"]["bfloat16"]
    assert {arm: a["confidence_only_divergent"] for arm, a in run["arms"].items()} == {
        "ragged": 0,
        "equalised": 0,
        "fixed": 0,
    }
    assert record["word_confidence_observed"] == {
        "mode_requested": "off",
        "decoder_step_confidence": False,
        "nonzero_conf_words_on_guard_recording": 0,
    }
    assert record["word_confidence"] == "off"
    assert all(
        w[3] == 0.0
        for a in run["arms"].values()
        for e in a["every_recording"].values()
        for w in e["a_words"] + e["b_words"]
    )
    assert [spec.word_confidence for spec in backend.specs] == ["off"]


def test_main_says_absent_when_no_two_shape_arm_diverges(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {})
    run = record["runs"]["bfloat16"]
    assert all(
        (a["text_divergent"], a["timing_only_divergent"]) == (0, 0) for a in run["arms"].values()
    )
    assert run["positive_control"].startswith("ABSENT")
    assert all(len(a["every_recording"]) == TARGETS for a in run["arms"].values())


def test_main_says_not_deterministic_when_the_same_shape_twice_moves_a_timing(
    tmp_path: Path,
) -> None:
    record, _ = _run(tmp_path, {1: "flaky_timing"})
    run = record["runs"]["bfloat16"]
    # Recording 1's text is stable and its timing is not, alone and in its batch.
    assert run["repeat"] == {"alone_identical": 1, "batch_identical": 1, "checked": 2}
    assert (
        run["repeat_verdict"]
        == "NOT run-to-run deterministic: divergence counts below include noise"
    )


def test_the_repeat_control_compares_the_batched_timings_too(tmp_path: Path) -> None:
    """Recording 1's first word ends somewhere new on every BATCHED call and never alone: only
    the batch channel's timing comparison can see it, and the verdict must follow it."""
    record, _ = _run(tmp_path, {1: "flaky_batch_timing"})
    run = record["runs"]["bfloat16"]
    assert run["repeat"] == {"alone_identical": 2, "batch_identical": 1, "checked": 2}
    assert run["repeat_verdict"] == sd.NOT_DETERMINISTIC_VERDICT


def test_the_repeat_control_stops_at_the_last_target(tmp_path: Path) -> None:
    record, backend = _run(tmp_path, {}, extra_env={"REPEAT": str(TARGETS + 2)})
    run = record["runs"]["bfloat16"]
    assert run["repeat"] == {
        "alone_identical": TARGETS,
        "batch_identical": TARGETS,
        "checked": TARGETS,
    }
    assert run["repeat_verdict"] == sd.SAME_SHAPE_VERDICT
    (pipeline,) = backend.pipelines
    seen = [[(rid, frames) for _, rid, frames in sorted(call)] for call in pipeline.calls]
    assert seen == _expected_calls(repeat=TARGETS)


def test_main_with_no_repeat_does_not_call_the_run_identical(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {}, extra_env={"REPEAT": "0"})
    assert record["runs"]["bfloat16"]["repeat_verdict"].startswith("NOT checked")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "off", "step_confidence": True}, "preserve_step_confidence is True"),
        ({"mode": "paper-best", "step_confidence": False}, "preserve_step_confidence is False"),
        ({"mode": "off", "graphs_mode": "full_graph"}, "decoder graphs False"),
        ({"mode": "off", "conf": 0.7}, "off is not off"),
        ({"mode": "nemo-shipped", "conf": 0.0}, "did not reach the words"),
        # The guard recording is recording 0; recording 1 answers normally in these two.
        ({"mode": "nemo-shipped", "conf_for": {0: 0.0}}, "did not reach the words"),
        ({"segments": False}, "final_segments empty"),
        ({"bare": frozenset({0})}, "final_segments empty"),
        ({"extra_env": {"MODEL_REVISION": "abc"}}, "not the pinned abc"),
        (
            {"meta_edit": lambda meta: meta[0].update(reference="nothing like it at all")},
            "overlap 0.00",
        ),
        # One of three reference words recognised: 0.33, under the 0.5 the guard asks for.
        ({"meta_edit": lambda meta: meta[0].update(reference="r0w0 x y")}, "overlap 0.33"),
        # Recording 1 is silence: its transcript is empty while recording 0's is recognised.
        ({"pool_edit": lambda pool: pool[1].fill(0.0)}, "empty transcript or overlap 1.00"),
        (
            {"encoder": SimpleNamespace(att_context_size=[70, 1])},
            "asks for att_context_size [70, 13], but the built encoder's att_context_size is"
            " [70, 1]",
        ),
        # Only the left context differs: the look-ahead alone would pass.
        (
            {"encoder": SimpleNamespace(att_context_size=[56, 13])},
            "asks for att_context_size [70, 13], but the built encoder's att_context_size is"
            " [56, 13]",
        ),
        ({"encoder": None}, "no pipeline.asr_model.asr_model.encoder;"),
        ({"encoder": SimpleNamespace()}, "att_context_size is None, not two ints"),
    ],
    ids=[
        "confidence-on-under-off",
        "confidence-off-under-paper-best",
        "decoder-graphs-unasked",
        "nonzero-conf-under-off",
        "zero-conf-under-nemo-shipped",
        "zero-conf-on-the-guard-recording-only",
        "blind-timing-channel",
        "blind-timing-on-the-guard-recording-only",
        "wrong-revision",
        "guard-recording-not-recognised",
        "guard-overlap-below-a-half",
        "second-guard-recording-empty",
        "encoder-context-not-the-one-asked-for",
        "encoder-left-context-not-the-one-asked-for",
        "no-encoder",
        "encoder-context-unreadable",
    ],
)
def test_main_stops_on_every_guard(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SystemExit, match=r"\[guard\]") as info:
        _run(tmp_path, {}, **kwargs)
    assert message in str(info.value)
    assert not (tmp_path / "record.json").exists(), "a refused run leaves no record behind"


def test_an_injected_backend_is_stamped_fake_whatever_it_says(tmp_path: Path) -> None:
    class Disguised(FakeBackend):
        def provenance(self) -> dict[str, Any]:
            return {
                "machine": "NVIDIA RTX A6000",
                "capability": "sm_86",
                "driver_and_uuid": "550.00, GPU-0",
                "gpu_uuid": "GPU-0",
                "gpu_uuid_torch": "0",
                "torch": "2.11.0",
                "torch_cuda": "12.4",
            }

    pool, meta = recordings(TARGETS + BATCH)
    backend = Disguised(pool, meta, lambda: FakeProbePipeline({}, conf=0.0, step_confidence=False))
    env = {"OUT": str(tmp_path / "r.json"), "BATCH": str(BATCH), "REPEAT": "1"}
    record = sd.main(["probe", "4"], env, backend)
    assert record["fake_pipeline"] is True
    assert record["not_a_row"] == FAKE_NOT_A_ROW
    assert json.loads((tmp_path / "r.json").read_text())["fake_pipeline"] is True


# --- main() down the real path, with the real backend's class replaced -------------------------


def test_main_without_an_injected_backend_is_not_stamped_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one path a GPU run takes: ``main`` builds ``NeMoBackend`` itself. Here that class
    is swapped for the fake, so the record is not stamped, and must not be: the scorer
    refuses a stamped record, and a real run stamped fake would be thrown away."""
    pool, meta = recordings(TARGETS + BATCH)
    built: list[FakeBackend] = []

    def nemo_backend() -> FakeBackend:
        backend = FakeBackend(
            pool, meta, lambda: FakeProbePipeline({}, conf=0.0, step_confidence=False)
        )
        built.append(backend)
        return backend

    monkeypatch.setattr(sd, "NeMoBackend", nemo_backend)
    env = {"OUT": str(tmp_path / "record.json"), "BATCH": str(BATCH), "REPEAT": str(REPEAT)}
    with warnings.catch_warnings():  # the real path silences warnings; keep that to this test
        record = sd.main(["stock_divergence.py", str(TARGETS)], env, None)
    assert len(built) == 1, "main built its real backend itself"
    on_disk = json.loads((tmp_path / "record.json").read_text())
    for seen in (record, on_disk):
        assert "fake_pipeline" not in seen, "C1: absent unless the fake wrote the record"
        assert seen["not_a_row"] == REAL_NOT_A_ROW
    assert next(iter(on_disk)) == "question"
    assert len(on_disk["runs"]["bfloat16"]["arms"]["ragged"]["every_recording"]) == TARGETS


# --- what the GPU replay needs unchanged, and what the record says was built ------------------


def test_a_two_shape_arm_that_moves_only_a_timing_is_a_positive_control(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {3: "shape_timing"})
    run = record["runs"]["bfloat16"]
    counts = {
        arm: (a["text_divergent"], a["timing_only_divergent"]) for arm, a in run["arms"].items()
    }
    assert counts == {"ragged": (0, 1), "equalised": (0, 1), "fixed": (0, 0)}
    assert run["positive_control"] == (
        "present: ragged, equalised diverged, so the harness can see a difference on this run"
    )


def test_main_hands_build_the_stock_runs_spec_with_the_left_context_it_was_given(
    tmp_path: Path,
) -> None:
    record, backend = _run(tmp_path, {}, extra_env={"ATT_LEFT": "56"})
    assert backend.specs == [
        NeMoPipelineSpec(
            model=STOCK_MODEL,
            chunk=ChunkMode(1120),
            att_context=(56, STOCK_ATT[1]),
            num_slots=256,
            batch_size=BATCH,
            compute_dtype="bfloat16",
            matmul_precision="highest",
            word_confidence="off",
        )
    ]
    assert record["att_context_size"] == [56, STOCK_ATT[1]]
    assert (record["model"], record["chunk_ms"], record["batch"], record["matmul_precision"]) == (
        STOCK_MODEL,
        1120,
        BATCH,
        "highest",
    )


def test_main_pads_every_row_with_zeros_beyond_its_own_audio(tmp_path: Path) -> None:
    """Every sample the pipeline is handed past a row's own audio is 0.0: the tail of its
    last chunk and, in the equalised and fixed arms, the ``pad_to`` padding up to the
    longest row. Silent rows are zeros throughout."""
    _, backend = _run(tmp_path, {})
    (pipeline,) = backend.pipelines
    assert pipeline.dtypes == {np.dtype(np.float32)}
    rows = past_own_chunks = 0
    for call in pipeline.calls:
        for sid, rid, frames in call:
            seen = pipeline.samples[sid]
            own = np.zeros(0, np.float32) if rid is None else backend.pool[rid]
            assert seen.shape == (frames * N,)
            assert np.array_equal(seen[: len(own)], own)
            assert np.all(seen[len(own) :] == 0.0), f"stream {sid}: the padding is not zeros"
            rows += 1
            past_own_chunks += rid is not None and frames > _frames_of(rid)
    assert rows == sum(len(call) for call in _expected_calls())
    assert past_own_chunks > 0, "no row was padded past its own chunks; pad_to went untested"


def test_the_stamps_are_what_the_decoder_did_even_when_the_refusal_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the refusal out of the way, asked-for and built differ, and the record must
    carry what was built. A stamp copied from the spec reads False here."""
    monkeypatch.setattr(sd, "refuse_unrequested_decoder", lambda *args: None)
    record, _ = _run(tmp_path, {}, mode="off", step_confidence=True, graphs_mode="full_graph")
    for where in (record, record["runs"]["bfloat16"]):
        assert where["word_confidence_observed"]["decoder_step_confidence"] is True
        assert where["word_confidence_observed"]["mode_requested"] == "off"
        assert where["decoder_graphs_observed"] is True
    assert record["decoder_graphs_mode_observed"] == "full_graph"
    # What was asked for is still stamped as asked.
    assert (record["word_confidence"], record["use_cuda_graphs"]) == ("off", False)


def test_decoder_graphs_are_compared_with_the_decoder_flag_not_the_encoder_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_spec_for = sd.spec_for

    def with_encoder_graphs(*args: Any) -> NeMoPipelineSpec:
        return dataclasses.replace(real_spec_for(*args), use_cuda_graphs=True)

    monkeypatch.setattr(sd, "spec_for", with_encoder_graphs)
    record, backend = _run(tmp_path, {})
    assert [spec.use_cuda_graphs for spec in backend.specs] == [True]
    assert record["use_cuda_graphs"] is True
    assert record["decoder_graphs_observed"] is False

    def with_decoder_graphs(*args: Any) -> NeMoPipelineSpec:
        return dataclasses.replace(real_spec_for(*args), use_cuda_graph_decoder=True)

    monkeypatch.setattr(sd, "spec_for", with_decoder_graphs)
    decoder = tmp_path / "decoder"
    decoder.mkdir()
    record, _ = _run(decoder, {}, graphs_mode="no_while_loops")
    assert record["decoder_graphs_observed"] is True
    assert record["decoder_graphs_mode_observed"] == "no_while_loops"
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(SystemExit, match="asks for decoder graphs True"):
        _run(refused, {}, graphs_mode=None)
    assert not (refused / "record.json").exists()


def test_main_runs_each_dtype_on_its_own_pipeline_from_stream_id_one(tmp_path: Path) -> None:
    record, backend = _run(tmp_path, {3: "shape_timing"}, extra_env={"DTYPES": "bfloat16,float32"})
    assert list(record["runs"]) == ["bfloat16", "float32"]
    assert [spec.compute_dtype for spec in backend.specs] == ["bfloat16", "float32"]
    assert len(backend.pipelines) == 2 and backend.released == 2
    for pipeline in backend.pipelines:
        seen = [[(rid, frames) for _, rid, frames in sorted(call)] for call in pipeline.calls]
        assert seen == _expected_calls()
        ids = [sid for call in pipeline.calls for sid, _, _ in sorted(call)]
        assert ids == list(range(1, len(ids) + 1)), "each dtype's stream ids start again at 1"
    for run in record["runs"].values():
        assert run["arms"]["ragged"]["timing_only_divergent"] == 1
        assert all(len(a["every_recording"]) == TARGETS for a in run["arms"].values())
        assert run["repeat"]["checked"] == REPEAT


# --- how a stream's step finals become its text ------------------------------------------------


class _ScriptedSteps:
    """Hands each stream its scripted (final_transcript, final_segments) one step at a time."""

    def __init__(self, script: dict[int, list[tuple[str, list[FakeTextSegment]]]]) -> None:
        self.script = {sid: list(steps) for sid, steps in script.items()}

    def transcribe_step(self, frames: list[FakeFrame]) -> list[Any]:
        outs = []
        for f in frames:
            text, segs = self.script[f.stream_id].pop(0)
            outs.append(
                SimpleNamespace(stream_id=f.stream_id, final_transcript=text, final_segments=segs)
            )
        return outs


def test_transcribe_joins_nemo_style_step_finals_into_one_text() -> None:
    """NeMo gives each step's final a leading separator after the stream's first request, and
    "" on a step that finalises nothing (``TranscribeStepOutput.from_state``). The probe keeps
    each nonempty final stripped and joins them with one space; so did the stock run. NeMo's
    own concatenation reads the same here: every final carries its separator, and stream 2's
    first final, after an empty first step, loses its own as nothing has accumulated."""
    seg = FakeTextSegment
    script = {
        1: [
            ("I'm", [seg("I'm", 0.08, 0.4, 0.25)]),
            (" from", [seg(" from", 0.4, 0.56, 0.5)]),
            ("", []),
            (" the", [seg(" the", 0.56, 0.64, 0.125)]),
        ],
        2: [("", []), (" Yes.", [seg(" Yes.", 0.16, 0.48, 0.75)])],
    }
    backend = FakeBackend([], [], None)
    rows = [np.full(4 * N, 0.5, np.float32), np.full(2 * N, 0.5, np.float32)]
    out = sd.transcribe(backend, _ScriptedSteps(script), N, [1], rows)
    assert out == [
        (
            "I'm from the",
            [("I'm", 0.08, 0.4, 0.25), ("from", 0.4, 0.56, 0.5), ("the", 0.56, 0.64, 0.125)],
            "I'm from the",
        ),
        ("Yes.", [("Yes.", 0.16, 0.48, 0.75)], "Yes."),
    ]


def test_main_joins_each_streams_step_finals_into_the_recordings_text(tmp_path: Path) -> None:
    _, backend = _run(tmp_path, {})
    (pipeline,) = backend.pipelines
    # The fake handed out NeMo-shaped finals: separators, and empty steps in the middle.
    shapes = {
        (rid, frames, tuple(pipeline.finals[sid]))
        for call in pipeline.calls
        for sid, rid, frames in call
        if rid is not None
    }
    assert (1, 3, ("r1w0", "", " r1w1 r1w2")) in shapes
    assert (2, 4, ("r2w0", "", " r2w1 r2w2", " r2w3")) in shapes
    assert (0, 4, ("r0w0", "", " r0w1", "")) in shapes, "a row padded past its own chunks"
    on_disk = json.loads((tmp_path / "record.json").read_text())
    for arm in on_disk["runs"]["bfloat16"]["arms"].values():
        for r in range(TARGETS):
            entry = arm["every_recording"][f"rec-{r}"]
            assert entry["a_text"] == entry["b_text"] == _text_of(r)
            assert entry["a_nemo_text"] == entry["b_nemo_text"] == _text_of(r)
            assert (
                entry["a_words"]
                == entry["b_words"]
                == [
                    [f"r{r}w{k}", round(k * STEP_S, 4), round(k * STEP_S + STEP_S, 4), 0.0]
                    for k in range(_frames_of(r))
                ]
            )
            assert arm["transcripts"][f"rec-{r}"] == {"a": _text_of(r)}


#: A word split across two steps. The words and timings are the stock record's at n=327
#: (3080-5032-0015, side a of the ragged and equalised arms): "the" ends at 15.68 s, the end of
#: the 14th 1,120 ms chunk, and "y" starts there. The finals are shaped as NeMo's
#: ``TranscribeStepOutput.from_state`` hands them out: no separator on the stream's first
#: request, and none on a final whose first word continues the last one finalised
#: (``concat_with_space`` False).
SPLIT_WORD_FINALS = [
    ("so ill that", [("so", 13.84, 14.0), ("ill", 14.0, 14.08), ("that", 14.48, 14.56)]),
    (" the", [(" the", 15.6, 15.68)]),
    ("y are not", [("y", 15.68, 15.76), ("are", 15.76, 15.84), ("not", 15.84, 15.92)]),
]


def test_transcribe_space_joins_finals_even_where_nemo_gives_no_separator() -> None:
    """NeMo's own concatenation of these finals reads "they"; the probe's per-step space join
    reads "the y", as the stock record does. Pinned as it is, because the replay must reproduce
    the stock run's text; NeMo's own concatenation is returned beside it."""
    script = {
        1: [
            (text, [FakeTextSegment(w, s, e, 0.0) for w, s, e in segs])
            for text, segs in SPLIT_WORD_FINALS
        ]
    }
    assert "".join(text for text, _ in SPLIT_WORD_FINALS) == "so ill that they are not"
    backend = FakeBackend([], [], None)
    rows = [np.full(3 * N, 0.5, np.float32)]
    out = sd.transcribe(backend, _ScriptedSteps(script), N, [1], rows)
    assert out == [
        (
            "so ill that the y are not",
            [(w.strip(), s, e, 0.0) for _, segs in SPLIT_WORD_FINALS for w, s, e in segs],
            "so ill that they are not",
        )
    ]
    assert out[0][1][3:5] == [("the", 15.6, 15.68, 0.0), ("y", 15.68, 15.76, 0.0)]


def test_main_space_joins_finals_that_carry_no_separator(tmp_path: Path) -> None:
    """Through main(): recordings 1 and 2 hand out every final without the separator, as NeMo
    does for a word continued across steps. The probe's text is still each final stripped and
    joined with one space; NeMo's own concatenation would read ``r1w0r1w1 r1w2``."""
    _, backend = _run(tmp_path, {}, glue=frozenset({1, 2}))
    (pipeline,) = backend.pipelines
    shapes = {
        (rid, frames, tuple(pipeline.finals[sid]))
        for call in pipeline.calls
        for sid, rid, frames in call
        if rid is not None
    }
    assert (1, 3, ("r1w0", "", "r1w1 r1w2")) in shapes
    assert (2, 4, ("r2w0", "", "r2w1 r2w2", "r2w3")) in shapes
    on_disk = json.loads((tmp_path / "record.json").read_text())
    # NeMo's own concatenation glues each final onto the one before, the same on both sides.
    glued = {1: "r1w0r1w1 r1w2", 2: "r2w0r2w1 r2w2r2w3"}
    for arm in on_disk["runs"]["bfloat16"]["arms"].values():
        assert arm["checked"] == TARGETS and arm["divergences"] == []
        assert (arm["nemo_text_divergent"], arm["nemo_text_divergent_hidden"]) == (0, 0)
        for r in (1, 2):
            entry = arm["every_recording"][f"rec-{r}"]
            assert entry["a_text"] == entry["b_text"] == _text_of(r)
            assert entry["a_nemo_text"] == entry["b_nemo_text"] == glued[r]
            assert arm["transcripts"][f"rec-{r}"] == {"a": _text_of(r)}


def test_main_counts_a_difference_only_nemos_own_text_shows(tmp_path: Path) -> None:
    """Recording 4 hands out its finals without the separator only when batched: its words,
    timings and joined text are the same on both sides of every arm, so every count the stock
    run kept reads it as identical. NeMo's own concatenation differs wherever one side is
    batched and the other is not, and the record counts and keeps that."""
    record, _ = _run(tmp_path, {4: "shape_glue"})
    arms = json.loads((tmp_path / "record.json").read_text())["runs"]["bfloat16"]["arms"]
    assert arms == json.loads(json.dumps(record["runs"]["bfloat16"]["arms"]))
    for name in ALL_ARMS:
        arm = arms[name]
        assert (arm["text_divergent"], arm["timing_only_divergent"]) == (0, 0), name
        assert arm["divergences"] == [], name
        entry = arm["every_recording"]["rec-4"]
        assert entry["a_text"] == entry["b_text"] == _text_of(4), name
        assert entry["a_words"] == entry["b_words"], name
    for name in ("ragged", "equalised"):
        arm = arms[name]
        assert (arm["nemo_text_divergent"], arm["nemo_text_divergent_hidden"]) == (1, 1), name
        entry = arm["every_recording"]["rec-4"]
        assert (entry["a_nemo_text"], entry["b_nemo_text"]) == (
            "r4w0 r4w1 r4w2",
            "r4w0r4w1 r4w2",
        ), name
    # Fixed: both sides are batched, so both are glued alike.
    fixed = arms["fixed"]
    assert (fixed["nemo_text_divergent"], fixed["nemo_text_divergent_hidden"]) == (0, 0)
    entry = fixed["every_recording"]["rec-4"]
    assert entry["a_nemo_text"] == entry["b_nemo_text"] == "r4w0r4w1 r4w2"


# --- which side is a, which is b ---------------------------------------------------------------


def test_each_arm_puts_its_reference_side_in_a_and_the_batch_in_b(tmp_path: Path) -> None:
    """a is the recording alone (ragged), alone padded (equalised), or among silent rows
    (fixed); b is the recording among its real neighbours. The scorer reads meaning into the
    sides, so a swap must show."""
    behaviour = {2: "shape_text", 3: "shape_timing", 5: "content_text", 6: "content_timing"}
    _run(tmp_path, behaviour)
    arms = json.loads((tmp_path / "record.json").read_text())["runs"]["bfloat16"]["arms"]

    def timing(arm: dict[str, Any], rid: str) -> tuple[list[Any], list[Any]]:
        (d,) = [d for d in arm["divergences"] if d["librispeech_id"] == rid]
        e = arm["every_recording"][rid]
        assert (d["a_timings"], d["b_timings"]) == (e["a_words"], e["b_words"])
        return d["a_timings"][0], d["b_timings"][0]

    changed = {r: "changed " + _text_of(r).split(" ", 1)[1] for r in (2, 5)}
    for name in ALL_ARMS:
        arm = arms[name]
        # Content: the neighbours carry sound only on side b, in every arm.
        assert arm["transcripts"]["rec-5"] == {"a": _text_of(5), "b": changed[5]}, name
        entry = arm["every_recording"]["rec-5"]
        assert (entry["a_text"], entry["b_text"]) == (_text_of(5), changed[5]), name
        (d,) = [d for d in arm["divergences"] if d["librispeech_id"] == "rec-5"]
        assert (d["a"], d["b"]) == (_text_of(5), changed[5]), name
        assert timing(arm, "rec-6") == (["r6w0", 0.0, 0.08, 0.0], ["r6w0", 0.0, 0.12, 0.0]), name
    for name in ("ragged", "equalised"):
        # Shape: side a is one row, side b is BATCH rows.
        assert arms[name]["transcripts"]["rec-2"] == {"a": _text_of(2), "b": changed[2]}, name
        assert timing(arms[name], "rec-3") == (
            ["r3w0", 0.0, 0.08, 0.0],
            ["r3w0", 0.0, 0.12, 0.0],
        ), name
    # Fixed: both sides are BATCH rows, so a change that follows the shape is on both.
    assert arms["fixed"]["transcripts"]["rec-2"] == {"a": changed[2]}
    assert "rec-3" not in {d["librispeech_id"] for d in arms["fixed"]["divergences"]}


# --- the arms asked for, and only those --------------------------------------------------------


@pytest.mark.parametrize(
    "arms",
    [("ragged",), ("equalised",), ("fixed",), ("ragged", "fixed"), ("equalised", "fixed")],
    ids=lambda arms: "+".join(arms),
)
def test_main_runs_an_arm_subset_exactly_as_it_runs_those_arms_in_the_full_set(
    tmp_path: Path, arms: tuple[str, ...]
) -> None:
    behaviour = {2: "shape_text", 3: "shape_timing", 5: "content_text", 6: "content_timing"}
    (tmp_path / "all").mkdir()
    (tmp_path / "subset").mkdir()
    full, _ = _run(tmp_path / "all", behaviour)
    record, backend = _run(tmp_path / "subset", behaviour, extra_env={"ARMS": ",".join(arms)})
    run = record["runs"]["bfloat16"]
    assert record["arms"] == list(arms)
    assert list(run["arms"]) == list(arms)
    for arm in arms:
        assert run["arms"][arm]["checked"] == TARGETS
        assert run["arms"][arm] == full["runs"]["bfloat16"]["arms"][arm], arm
    (pipeline,) = backend.pipelines
    seen = [[(rid, frames) for _, rid, frames in sorted(call)] for call in pipeline.calls]
    assert seen == _expected_calls(arms=arms)


# --- the record and the spec follow the environment --------------------------------------------


def test_the_record_and_the_spec_follow_the_environment_not_the_defaults(tmp_path: Path) -> None:
    """Every setting away from its default, and a revision pin that is only a prefix: the spec
    handed to build and the record's fields must carry what the environment said and what the
    run resolved, not the defaults and not the pin."""
    env = {
        "MODEL": "example/other-streaming-model",
        "ATT_LEFT": "40",
        "CHUNK_MS": "560",
        "MATMUL": "high",
        "DTYPES": "float32",
        "MODEL_REVISION": "fake0",
        "NEMO_COMMIT": "nemo commit from the environment",
        "VERBATIM_COMMIT": "verbatim commit from the environment",
        "CUDA_VISIBLE_DEVICES": "fake-card",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    _, backend = _run(tmp_path, {}, targets=5, extra_env=env)
    assert backend.specs == [
        NeMoPipelineSpec(
            model="example/other-streaming-model",
            chunk=ChunkMode(560),
            att_context=(40, 6),
            num_slots=256,
            batch_size=BATCH,
            compute_dtype="float32",
            matmul_precision="high",
            word_confidence="off",
        )
    ]
    on_disk = json.loads((tmp_path / "record.json").read_text())
    expected = {
        "model": "example/other-streaming-model",
        "model_revision": "fake0revision",
        "chunk_ms": 560,
        "batch": BATCH,
        "matmul_precision": "high",
        "targets": 5,
        "arms": list(ALL_ARMS),
        "att_context_size": [40, 6],
        "att_context_size_observed": [40, 6],
        "nemo": FAKE_NEMO,
        "nemo_commit": "nemo commit from the environment",
        "verbatim_commit_from_env": "verbatim commit from the environment",
        "cuda_visible_devices": "fake-card",
        "cuda_device_order": "PCI_BUS_ID",
        "pool": 5 + BATCH,
        **FAKE_PROVENANCE,
    }
    assert {key: on_disk[key] for key in expected} == expected
    assert list(on_disk["runs"]) == ["float32"]
    assert on_disk["runs"]["float32"]["att_context_size_observed"] == [40, 6]
    assert list(on_disk["references"]) == [f"rec-{r}" for r in range(5)]
    assert all(a["checked"] == 5 for a in on_disk["runs"]["float32"]["arms"].values())


def test_the_record_says_unstated_when_the_environment_names_no_commit(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {})
    assert (record["nemo_commit"], record["verbatim_commit_from_env"]) == ("unstated", "unstated")
    assert record["cuda_device_order"] is None


# --- the attention context is read off the built encoder ---------------------------------------


def _with_encoder(encoder: Any) -> Any:
    return SimpleNamespace(asr_model=SimpleNamespace(asr_model=SimpleNamespace(encoder=encoder)))


class _ListLike:
    """Iterable and sized, like OmegaConf's ListConfig, which is what NeMo's encoder keeps."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __iter__(self) -> Any:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


@pytest.mark.parametrize(
    "value", [[70, 13], (70, 13), _ListLike([70, 13])], ids=["list", "tuple", "list-like"]
)
def test_observe_att_context_reads_two_ints_off_the_encoder(value: Any) -> None:
    seen = sd.observe_att_context(_with_encoder(SimpleNamespace(att_context_size=value)))
    assert seen == [70, 13] and type(seen) is list


@pytest.mark.parametrize(
    "value",
    [None, 7, "70,13", bytes([70, 13]), [70], [70, 13, 1], [True, 13], [70.0, 13]],
    ids=["none", "an-int", "a-string", "two-bytes", "one", "three", "a-bool", "a-float"],
)
def test_observe_att_context_refuses_what_is_not_two_ints(value: Any) -> None:
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED: .*not two ints"):
        sd.observe_att_context(_with_encoder(SimpleNamespace(att_context_size=value)))


def test_observe_att_context_refuses_a_pipeline_without_the_encoder() -> None:
    with pytest.raises(SystemExit, match=r"no pipeline\.asr_model;"):
        sd.observe_att_context(SimpleNamespace())
    with pytest.raises(SystemExit, match=r"no pipeline\.asr_model\.asr_model\.encoder;"):
        sd.observe_att_context(_with_encoder(None))


def test_the_att_context_guard_passes_the_requested_context_and_refuses_another() -> None:
    sd.refuse_unrequested_att_context((70, 13), [70, 13])
    with pytest.raises(SystemExit, match=r"asks for att_context_size \[70, 13\].* is \[70, 1\]"):
        sd.refuse_unrequested_att_context((70, 13), [70, 1])
    # Each half is compared: a left context of its own is refused as a look-ahead is.
    with pytest.raises(SystemExit, match=r"asks for att_context_size \[70, 13\].* is \[56, 13\]"):
        sd.refuse_unrequested_att_context((70, 13), [56, 13])


def test_the_att_context_stamp_is_the_encoders_even_when_the_refusal_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sd, "refuse_unrequested_att_context", lambda *args: None)
    record, backend = _run(tmp_path, {}, encoder=SimpleNamespace(att_context_size=(70, 1)))
    assert [spec.att_context for spec in backend.specs] == [STOCK_ATT]
    assert record["att_context_size"] == list(STOCK_ATT), "what was asked for"
    assert record["att_context_size_observed"] == [70, 1]
    assert record["runs"]["bfloat16"]["att_context_size_observed"] == [70, 1]


# --- the guard's threshold, the stored confidence ----------------------------------------------


def test_the_guard_accepts_an_overlap_of_exactly_a_half(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {}, meta_edit=lambda meta: meta[0].update(reference="r0w0 x"))
    assert record["runs"]["bfloat16"]["arms"]["ragged"]["checked"] == TARGETS


def test_the_stored_confidence_is_the_pipelines_to_full_precision(tmp_path: Path) -> None:
    conf = 0.1234567890123456
    seg = SimpleNamespace(text="a", start=0.0, end=0.08, conf=conf)
    assert sd.timing_of(seg)[3] == conf
    _run(tmp_path, {}, mode="nemo-shipped", conf=conf)
    on_disk = json.loads((tmp_path / "record.json").read_text())
    stored = {
        w[3]
        for arm in on_disk["runs"]["bfloat16"]["arms"].values()
        for e in arm["every_recording"].values()
        for w in e["a_words"] + e["b_words"]
    }
    assert stored == {conf}
    assert [w[3] for w in on_disk["timing_channel"]["sample"]] == [conf, conf]


# --- a replay of the first targets gives the longer run's answers ------------------------------


def _compare_stock_replay() -> Any:
    path = ROOT / "scripts" / "compare_stock_replay.py"
    spec = importlib.util.spec_from_file_location("compare_stock_replay_for_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: What the probe writes now and did not write at 6583a83.
NEWER_THAN_THE_STOCK = (
    "fake_pipeline",
    "word_confidence",
    "use_cuda_graphs",
    "word_confidence_observed",
    "decoder_graphs_observed",
    "decoder_graphs_mode_observed",
    "att_context_size_observed",
    "timing_tuple",
    "gpu_uuid",
    "gpu_uuid_torch",
    "cuda_device_order",
    "verbatim_commit",
    "tracked_files_modified",
    "verbatim_commit_from_env",
    "code",
    "pool",
)


def _as_the_stock_record(record: dict[str, Any]) -> dict[str, Any]:
    """The record in the shape the probe wrote at 6583a83: three-element timings, and no
    ``every_recording``, confidence counts or observed stamps."""
    old = {k: v for k, v in copy.deepcopy(record).items() if k not in NEWER_THAN_THE_STOCK}
    for run in old["runs"].values():
        for key in NEWER_THAN_THE_STOCK:
            run.pop(key, None)
        for arm in run["arms"].values():
            del arm["every_recording"], arm["confidence_only_divergent"]
            del arm["nemo_text_divergent"], arm["nemo_text_divergent_hidden"]
            for d in arm["divergences"]:
                d["a_timings"] = [t[:3] for t in d["a_timings"]]
                d["b_timings"] = [t[:3] for t in d["b_timings"]]
    return old


@pytest.mark.parametrize(
    ("replay_targets", "replay_behaviour", "differs"),
    [
        (5, {}, None),
        (5, {2: "shape_timing"}, "bfloat16/ragged rec-2: text_differs is True"),
        (1, {}, None),
    ],
    ids=["same-pipeline", "a-pipeline-that-answers-otherwise", "fewer-targets-than-repeat"],
)
def test_a_replay_of_the_first_targets_compares_clean_with_a_longer_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_targets: int,
    replay_behaviour: dict[int, str],
    differs: str | None,
) -> None:
    """The probe, down the path a GPU run takes, writes a run of TARGETS and a replay of its
    first five, and ``scripts/compare_stock_replay.py`` compares them as it will compare the
    GPU replay with the stock record. Recording 5 diverges in the longer run only: it is past
    the replay's last target. The neighbours of the first five are the same in both pools, so
    the same pipeline must give the same answers; a pipeline that answers otherwise must not
    compare clean. A replay of one target checks one in the repeat control where the longer
    run checked REPEAT, and still compares clean."""
    compare = _compare_stock_replay()
    pool, meta = recordings(TARGETS + BATCH)
    behaviour = {2: "shape_text", 3: "shape_timing", 5: "content_text"}
    current = [behaviour]

    def nemo_backend() -> FakeBackend:
        return FakeBackend(
            pool, meta, lambda: FakeProbePipeline(current[0], conf=0.0, step_confidence=False)
        )

    monkeypatch.setattr(sd, "NeMoBackend", nemo_backend)
    base = {"BATCH": str(BATCH), "REPEAT": str(REPEAT), "CUDA_VISIBLE_DEVICES": ""}
    with warnings.catch_warnings():
        longer = sd.main(["probe", str(TARGETS)], {**base, "OUT": str(tmp_path / "s.json")})
        current[0] = {**behaviour, **replay_behaviour}
        replay = sd.main(["probe", str(replay_targets)], {**base, "OUT": str(tmp_path / "r.json")})
    stock = _as_the_stock_record(json.loads((tmp_path / "s.json").read_text()))
    assert "fake_pipeline" not in longer and "fake_pipeline" not in replay
    assert [d["n"] for d in stock["runs"]["bfloat16"]["arms"]["ragged"]["divergences"]] == [2, 3, 5]
    assert stock["runs"]["bfloat16"]["repeat"]["checked"] == REPEAT
    assert replay["runs"]["bfloat16"]["repeat"]["checked"] == min(REPEAT, replay_targets)
    found = compare.differences(stock, json.loads((tmp_path / "r.json").read_text()))
    if differs is None:
        assert found == []
    else:
        assert any(differs in line for line in found), found


# --- a record is written once ------------------------------------------------------------------

SENTINEL = '{"another run": "its record"}'


@pytest.mark.parametrize("name", ["record.json", "record.json.tmp"], ids=["out", "out-tmp"])
def test_main_refuses_an_out_or_out_tmp_that_exists_before_anything_runs(
    tmp_path: Path, name: str
) -> None:
    (tmp_path / name).write_text(SENTINEL)
    backends: list[FakeBackend] = []
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED: .*exists; a record is written once"):
        _run(tmp_path, {}, backends=backends)
    assert (tmp_path / name).read_text() == SENTINEL, "the file already there is untouched"
    assert sorted(p.name for p in tmp_path.iterdir()) == [name], "nothing else was written"
    (backend,) = backends
    assert (backend.loaded, backend.specs) == ([], []), "refused before the data or a model"


def test_a_dangling_link_at_out_is_refused_too(tmp_path: Path) -> None:
    (tmp_path / "record.json").symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(SystemExit, match="exists; a record is written once"):
        _run(tmp_path, {})
    assert (tmp_path / "record.json").is_symlink()
    assert not (tmp_path / "nowhere.json").exists()


@pytest.mark.parametrize("name", ["record.json", "record.json.tmp"], ids=["out", "out-tmp"])
def test_a_file_that_appears_at_out_during_the_run_is_not_written_over(
    tmp_path: Path, name: str
) -> None:
    """Another run given the same OUT writes there while this one builds its model: the first
    save must neither replace it nor carry on as if the record were this run's."""
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED: .* appeared during the run"):
        _run(tmp_path, {}, on_build=lambda: (tmp_path / name).write_text(SENTINEL))
    assert (tmp_path / name).read_text() == SENTINEL
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def test_the_first_save_creates_out_and_the_later_ones_replace_it(tmp_path: Path) -> None:
    out = str(tmp_path / "record.json")
    sd.save({"checked": 1}, out, first=True)
    assert json.loads(Path(out).read_text()) == {"checked": 1}
    assert not Path(out + ".tmp").exists()
    with pytest.raises(SystemExit, match="appeared during the run"):
        sd.save({"checked": 2}, out, first=True)
    assert json.loads(Path(out).read_text()) == {"checked": 1}
    assert not Path(out + ".tmp").exists(), "a refused first save leaves no temporary behind"
    sd.save({"checked": 3}, out)
    assert json.loads(Path(out).read_text()) == {"checked": 3}
    assert not Path(out + ".tmp").exists()


def test_without_hard_links_the_first_save_still_never_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_links(src: Any, dst: Any) -> None:
        raise PermissionError("this filesystem makes no hard links")

    monkeypatch.setattr(sd.os, "link", no_links)
    out = str(tmp_path / "record.json")
    sd.save({"checked": 1}, out, first=True)
    assert json.loads(Path(out).read_text()) == {"checked": 1}
    assert not Path(out + ".tmp").exists()
    with pytest.raises(SystemExit, match="appeared during the run"):
        sd.save({"checked": 2}, out, first=True)
    assert json.loads(Path(out).read_text()) == {"checked": 1}
    assert not Path(out + ".tmp").exists()


# --- which card ran ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smi", "torch_uuid"),
    [
        ("GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab", "3f2a91c0-aaaa-bbbb-cccc-0123456789ab"),
        ("GPU-3F2A91C0-AAAA-BBBB-CCCC-0123456789AB", "3f2a91c0-aaaa-bbbb-cccc-0123456789ab"),
        ("GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab", "GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab"),
    ],
    ids=["torch-without-the-prefix", "another-case", "both-with-the-prefix"],
)
def test_the_card_check_accepts_one_card_however_it_is_printed(smi: str, torch_uuid: str) -> None:
    sd.refuse_other_card(smi, torch_uuid)


@pytest.mark.parametrize(
    ("smi", "torch_uuid", "message"),
    [
        (
            "GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab",
            "77777777-aaaa-bbbb-cccc-0123456789ab",
            "nvidia-smi names card GPU-3f2a91c0",
        ),
        # A prefix of the other is still another card.
        ("GPU-3f2a91c0", "3f2a91c0-aaaa-bbbb-cccc-0123456789ab", "nvidia-smi names card"),
        (None, "3f2a91c0-aaaa-bbbb-cccc-0123456789ab", "cannot be checked"),
        ("GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab", None, "cannot be checked"),
        ("GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab", "", "cannot be checked"),
    ],
    ids=["another-card", "a-prefix", "smi-unread", "torch-unread", "torch-empty"],
)
def test_the_card_check_refuses_another_card_or_one_it_cannot_read(
    smi: str | None, torch_uuid: str | None, message: str
) -> None:
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED") as info:
        sd.refuse_other_card(smi, torch_uuid)
    assert message in str(info.value)


@pytest.mark.parametrize(
    ("line", "uuid"),
    [
        ("550.54.14, GPU-3f2a91c0-aaaa", "GPU-3f2a91c0-aaaa"),
        ("550.54.14, GPU-3f2a91c0-aaaa\n", "GPU-3f2a91c0-aaaa"),
        ("550.54.14, MIG-3f2a91c0-aaaa", "MIG-3f2a91c0-aaaa"),
        ("unknown (FileNotFoundError(2, 'No such file'))", None),
        ("550.54.14, GPU-1\n550.54.14, GPU-2", None),
        ("550.54.14, ", None),
        ("550.54.14, [N/A]", None),
        ("", None),
    ],
    ids=[
        "one-card",
        "trailing-newline",
        "mig",
        "unread",
        "two-cards",
        "no-uuid",
        "not-a-uuid",
        "empty",
    ],
)
def test_the_nvidia_smi_uuid_is_read_from_its_one_line(line: str, uuid: str | None) -> None:
    assert sd.smi_uuid(line) == uuid


def test_the_uuid_is_compared_without_the_prefix_or_the_case() -> None:
    assert sd.normalise_uuid(" GPU-AbC-12 ") == "abc-12"
    assert sd.normalise_uuid("abc-12") == "abc-12"
    assert sd.normalise_uuid("MIG-abc") == "mig-abc", "only nvidia-smi's GPU- prefix is dropped"


def test_the_real_backend_reads_the_card_from_nvidia_smi_and_from_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[int] = []

    class _Uuid:
        def __str__(self) -> str:
            return "3f2a91c0-aaaa-bbbb-cccc-0123456789ab"

    def properties(index: int) -> Any:
        asked.append(index)
        return SimpleNamespace(uuid=_Uuid())

    cuda = SimpleNamespace(
        get_device_name=lambda index: "FAKE card",
        get_device_capability=lambda index: (8, 6),
        get_device_properties=properties,
    )
    backend = object.__new__(sd.NeMoBackend)
    backend._torch = SimpleNamespace(
        cuda=cuda, __version__="FAKE", version=SimpleNamespace(cuda="x")
    )
    line = "550.54.14, GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab"
    monkeypatch.setattr(sd, "driver_version", lambda: line)
    prov = backend.provenance()
    assert asked == [0], "torch's cuda:0, the card the run computes on"
    assert prov["driver_and_uuid"] == line
    assert prov["gpu_uuid"] == "GPU-3f2a91c0-aaaa-bbbb-cccc-0123456789ab"
    assert prov["gpu_uuid_torch"] == "3f2a91c0-aaaa-bbbb-cccc-0123456789ab"
    sd.refuse_other_card(prov["gpu_uuid"], prov["gpu_uuid_torch"])


@pytest.mark.parametrize(
    ("card", "message"),
    [
        ({"gpu_uuid_torch": "fake-1111-card"}, "nvidia-smi names card GPU-FAKE-0000-card"),
        ({"gpu_uuid": None}, "cannot be checked"),
        ({"gpu_uuid_torch": None}, "cannot be checked"),
    ],
    ids=["another-card", "smi-unread", "torch-unread"],
)
def test_main_refuses_a_card_torch_and_nvidia_smi_disagree_on(
    tmp_path: Path, card: dict[str, Any], message: str
) -> None:
    backends: list[FakeBackend] = []
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED") as info:
        _run(tmp_path, {}, card=card, backends=backends)
    assert message in str(info.value)
    assert not (tmp_path / "record.json").exists()
    (backend,) = backends
    assert (backend.loaded, backend.specs) == ([], []), "refused before the data or a model"


def test_main_stamps_both_card_readings(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {}, card={"gpu_uuid_torch": "FAKE-0000-CARD"})
    on_disk = json.loads((tmp_path / "record.json").read_text())
    for seen in (record, on_disk):
        assert (seen["gpu_uuid"], seen["gpu_uuid_torch"]) == (
            "GPU-FAKE-0000-card",
            "FAKE-0000-CARD",
        )
        assert seen["driver_and_uuid"] == "FAKE driver"


# --- which code ran ----------------------------------------------------------------------------


def _checkout(root: Path) -> tuple[Path, Path]:
    """A probe file and a verbatim package laid out as in the repo, under ``root``."""
    probe = root / "probes" / "stock_divergence.py"
    package = root / "src" / "verbatim" / "__init__.py"
    probe.parent.mkdir(parents=True)
    package.parent.mkdir(parents=True)
    probe.write_bytes(b"# a probe\n")
    package.write_text("")
    return probe, package


class _Git:
    """``run_git`` scripted: each subcommand's answer, None for a failure; every call kept."""

    def __init__(self, answers: dict[str, str | None]) -> None:
        self.answers = answers
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def __call__(self, cwd: Path, *args: str) -> str | None:
        self.calls.append((cwd, args))
        key = " ".join(a for a in args if a != "--no-optional-locks")
        return self.answers.get(key)


HEAD = "0123456789abcdef0123456789abcdef01234567"


def _answers(top: Path | str | None, status: str | None = "") -> dict[str, str | None]:
    return {
        "rev-parse --show-toplevel": None if top is None else str(top),
        "rev-parse HEAD": HEAD,
        "status --porcelain --untracked-files=no": status,
    }


def test_the_code_identity_reads_the_package_the_probe_and_the_checkout(tmp_path: Path) -> None:
    probe, package = _checkout(tmp_path)
    git = _Git(_answers(tmp_path, " M src/verbatim/serve.py\n M probes/stock_divergence.py"))
    code = sd.code_identity(probe, str(package), git)
    assert code == {
        "verbatim_path": str((tmp_path / "src" / "verbatim").resolve()),
        "checkout": str(tmp_path.resolve()),
        "probe_sha256": hashlib.sha256(b"# a probe\n").hexdigest(),
        "verbatim_commit": HEAD,
        "tracked_files_modified": True,
        "tracked_changes": [" M src/verbatim/serve.py", " M probes/stock_divergence.py"],
    }
    assert {cwd for cwd, _ in git.calls} == {tmp_path.resolve()}, "git asks the probe's checkout"
    status = [args for _, args in git.calls if "status" in args]
    assert status == [("--no-optional-locks", "status", "--porcelain", "--untracked-files=no")]
    sd.refuse_code_outside_checkout(code)


def test_the_package_path_is_resolved_before_it_is_compared(tmp_path: Path) -> None:
    """A verbatim reached through a link to this checkout is this checkout's: the path is
    resolved, so the guard compares where the files are, not how they were reached."""
    probe, _ = _checkout(tmp_path / "real")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    through_link = tmp_path / "link" / "src" / "verbatim" / "__init__.py"
    code = sd.code_identity(probe, str(through_link), _Git(_answers(tmp_path / "real")))
    assert code["verbatim_path"] == str((tmp_path / "real" / "src" / "verbatim").resolve())
    sd.refuse_code_outside_checkout(code)


def test_a_clean_checkout_says_no_tracked_file_is_modified(tmp_path: Path) -> None:
    probe, package = _checkout(tmp_path)
    code = sd.code_identity(probe, str(package), _Git(_answers(tmp_path, "")))
    assert (code["verbatim_commit"], code["tracked_files_modified"]) == (HEAD, False)
    assert code["tracked_changes"] == []


@pytest.mark.parametrize(
    ("answers", "why"),
    [
        (_answers(None), "git cannot say"),
        (_answers("/somewhere/else"), "a git above the checkout, not the checkout"),
        ({**_answers(None), "rev-parse HEAD": HEAD}, "no work tree, whatever HEAD says"),
    ],
    ids=["no-git", "another-top", "no-top"],
)
def test_the_git_state_is_null_unless_the_checkout_is_the_top_of_a_work_tree(
    tmp_path: Path, answers: dict[str, str | None], why: str
) -> None:
    probe, package = _checkout(tmp_path)
    code = sd.code_identity(probe, str(package), _Git(answers))
    assert code["verbatim_commit"] is None, why
    assert code["tracked_files_modified"] is None, why
    assert code["tracked_changes"] is None, why


def test_an_unreadable_status_leaves_the_tree_state_null_and_keeps_the_head(
    tmp_path: Path,
) -> None:
    probe, package = _checkout(tmp_path)
    code = sd.code_identity(probe, str(package), _Git(_answers(tmp_path, None)))
    assert code["verbatim_commit"] == HEAD
    assert (code["tracked_files_modified"], code["tracked_changes"]) == (None, None)


def test_the_real_git_reads_nothing_in_a_directory_that_is_not_a_work_tree_top(
    tmp_path: Path,
) -> None:
    """A git work tree whose top is ABOVE the probe's checkout: its HEAD would describe another
    tree. Real git, no commit made."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    probe, package = _checkout(tmp_path / "nested")
    assert sd.run_git(tmp_path / "nested", "rev-parse", "--show-toplevel") == str(
        tmp_path.resolve()
    )
    code = sd.code_identity(probe, str(package))
    assert (code["verbatim_commit"], code["tracked_files_modified"]) == (None, None)


def test_the_real_git_reads_this_checkouts_head_and_tree() -> None:
    """The wiring against real git, on the checkout these tests run from. A copy of the repo
    that is not a git work tree cannot check this, and skips."""
    top = sd.run_git(ROOT, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != ROOT.resolve():
        pytest.skip("this checkout is not the top of a git work tree")

    def status() -> list[str]:
        return (
            subprocess.run(
                ["git", "-C", str(ROOT), "--no-optional-locks", "status", "--porcelain", "-uno"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )

    # Others may edit this checkout meanwhile: the reading must be the tree before or after.
    before = status()
    code = sd.code_identity(PROBE, str(ROOT / "src" / "verbatim" / "__init__.py"))
    after = status()
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert code["verbatim_commit"] == head and len(head) == 40
    assert code["tracked_changes"] in (before, after)
    assert code["tracked_files_modified"] is bool(code["tracked_changes"])


def test_the_code_guard_refuses_a_verbatim_from_another_checkout(tmp_path: Path) -> None:
    probe, _ = _checkout(tmp_path / "here")
    _, elsewhere = _checkout(tmp_path / "there")
    code = sd.code_identity(probe, str(elsewhere), _Git(_answers(tmp_path / "here")))
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED: verbatim is imported from"):
        sd.refuse_code_outside_checkout(code)


def test_main_refuses_to_run_code_from_outside_its_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed ``verbatim`` of another checkout, as an editable install would give when
    PYTHONPATH is not set: refused before the data or a model is touched."""
    _, elsewhere = _checkout(tmp_path / "other-checkout")
    monkeypatch.setattr(sd, "verbatim", SimpleNamespace(__file__=str(elsewhere)))
    out = tmp_path / "out"
    out.mkdir()
    backends: list[FakeBackend] = []
    with pytest.raises(SystemExit, match=r"verbatim is imported from .*other-checkout"):
        _run(out, {}, backends=backends)
    assert list(out.iterdir()) == []
    (backend,) = backends
    assert (backend.loaded, backend.specs) == ([], [])


def test_main_stamps_the_code_it_read_and_keeps_the_environments_commit_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = _Git(_answers(ROOT.resolve(), " M src/verbatim/serve.py"))
    monkeypatch.setattr(sd, "run_git", git)
    record, _ = _run(tmp_path, {}, extra_env={"VERBATIM_COMMIT": "what the environment says"})
    on_disk = json.loads((tmp_path / "record.json").read_text())
    for seen in (record, on_disk):
        assert seen["verbatim_commit"] == HEAD, "read off the checkout, not the environment"
        assert seen["tracked_files_modified"] is True
        assert seen["verbatim_commit_from_env"] == "what the environment says"
        assert seen["code"] == {
            "verbatim_path": str((ROOT / "src" / "verbatim").resolve()),
            "checkout": str(ROOT.resolve()),
            "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
            "verbatim_commit": HEAD,
            "tracked_files_modified": True,
            "tracked_changes": [" M src/verbatim/serve.py"],
        }
    assert {cwd for cwd, _ in git.calls} == {ROOT.resolve()}


# --- the neighbours of the last targets, when the data runs out --------------------------------


@pytest.mark.parametrize("pool_size", [TARGETS, TARGETS + 1, TARGETS + BATCH - 1])
def test_the_last_targets_neighbours_wrap_to_the_first_recordings(
    tmp_path: Path, pool_size: int
) -> None:
    """The data holds fewer than TARGETS + BATCH recordings, as test-other did for the stock
    run: the neighbours of the last targets are the first recordings, in both the repeat
    control and the arms, and every target is still checked."""
    record, backend = _run(tmp_path, {}, pool_size=pool_size, extra_env={"REPEAT": str(TARGETS)})
    (pipeline,) = backend.pipelines
    seen = [[(rid, frames) for _, rid, frames in sorted(call)] for call in pipeline.calls]
    assert seen == _expected_calls(repeat=TARGETS, pool=pool_size)
    last = [
        call
        for call in seen
        if len(call) == BATCH and call[0][0] == TARGETS - 1 and call[1][0] is not None
    ]
    assert last and all(
        [rid for rid, _ in call[1:]] == [(TARGETS + k) % pool_size for k in range(BATCH - 1)]
        for call in last
    ), "the last target's neighbours are the ones after it, wrapped"
    assert record["pool"] == pool_size
    run = record["runs"]["bfloat16"]
    assert run["repeat"]["checked"] == TARGETS
    assert all(a["checked"] == TARGETS for a in run["arms"].values())
    assert all(
        list(a["every_recording"]) == [f"rec-{r}" for r in range(TARGETS)]
        for a in run["arms"].values()
    )


@pytest.mark.parametrize(
    ("pool_size", "targets", "message"),
    [
        (TARGETS - 1, TARGETS, f"holds {TARGETS - 1} recordings; {TARGETS} targets"),
        (BATCH - 1, BATCH - 1, f"holds {BATCH - 1} recordings; {BATCH - 1} targets at BATCH"),
    ],
    ids=["fewer-than-the-targets", "fewer-than-the-batch"],
)
def test_main_refuses_data_too_short_for_the_targets_or_the_batch(
    tmp_path: Path, pool_size: int, targets: int, message: str
) -> None:
    backends: list[FakeBackend] = []
    with pytest.raises(SystemExit, match=r"\[guard\] FAILED") as info:
        _run(
            tmp_path,
            {},
            pool_size=pool_size,
            targets=targets,
            extra_env={"REPEAT": "0"},
            backends=backends,
        )
    assert message in str(info.value)
    assert not (tmp_path / "record.json").exists()
    assert backends[0].specs == [], "refused before a model is built"


def test_data_as_long_as_the_batch_is_enough(tmp_path: Path) -> None:
    record, _ = _run(tmp_path, {}, pool_size=BATCH, targets=BATCH, extra_env={"REPEAT": "0"})
    assert record["pool"] == BATCH
    assert all(a["checked"] == BATCH for a in record["runs"]["bfloat16"]["arms"].values())
