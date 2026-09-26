# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""NeMo's word-level confidence aggregation raises on some transcripts, and nothing on the
streaming path reads it. ``build_pipeline`` switches it off on the built decoding object
when word confidence is on, leaves the object alone when it is off, and ``/readyz``
reports the flag it finds there.

The crash, as a streaming run met it (``nvidia/nemotron-speech-streaming-en-0.6b``, 160 ms
chunks, paper-best): ``RuntimeError: Something went wrong with word-level confidence
aggregation ... len(words): 5, len(word_confidence): 6, recognized text: `He has one son,
and'tis```, raised by ``_aggregate_token_confidence_subwords_sentencepiece`` from
``compute_confidence`` inside a streaming step.

What is held here, and how:

* The crash itself, with NeMo's own code and the checkpoint's own tokenizer, which NeMo's
  ``ASRBPEMixin`` sets up from the tokenizer files in the archive alone (no weights are
  read, no model is built). The text's tokens as the tokenizer writes them aggregate
  cleanly. The same text with one lone word-start token before the apostrophe or before
  the comma raises with the numbers above: NeMo's text drops the space before a mark, so
  "and 'tis" reads "and'tis" and "son ," reads "son,", one word fewer than the tokens
  start. The message does not say which of the two the model emitted; either reproduces
  it. These tests skip where NeMo is not installed or the checkpoint is not in the local
  Hugging Face cache; nothing is downloaded.
* After ``build_pipeline``, on NeMo's own ``RNNTBPEDecoding`` built from the configuration
  ``build_pipeline`` hands NeMo: the word aggregator is not called, a hypothesis that
  raised decodes, and the token confidences and the pipeline's own word confidences
  (NeMo's streaming greedy decoder and ``BPEDecoder``, whose per-word ``conf`` is what a
  final segment carries to ``cache_aware._words_of``) are what they were, on a toy decode
  the old path survives.
* "off" hands NeMo the configuration's bytes from before the switch existed and returns
  the object NeMo built with nothing written to it.
* ``/readyz`` reads the flag off the built object on every request.

The toy prediction and joint networks are seeded and run on one CPU thread, so two arms
compared differ in what the switch changes and not in how a reduction was split.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import tarfile
import tempfile
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from verbatim.config import ChunkMode
from verbatim.pipelines import registry
from verbatim.pipelines.nemo_fake import FakeCacheAwareRNNTPipeline, boundary_for
from verbatim.pipelines.nemo_runtime import (
    DECODING_PATH,
    WORD_CONFIDENCE_MODES,
    NeMoPipelineSpec,
    PipelineBuildError,
    build_pipeline,
    pipeline_config,
)
from verbatim.pipelines.observed import configured_word_confidence, observe
from verbatim.protocols.health import OBSERVED_KEYS
from verbatim.serve import Endpoints, ServeSettings, engine_config, run_server

REPO = "nvidia/nemotron-speech-streaming-en-0.6b"
CHECKPOINT = "nemotron-speech-streaming-en-0.6b.nemo"
#: The revision the crash was met on, as the full commit the cache is keyed by.
REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
CRASH_TEXT = "He has one son, and'tis"
ON = tuple(mode for mode in WORD_CONFIDENCE_MODES if mode != "off")
BUILDER = "nemo.collections.asr.inference.factory.pipeline_builder"
#: sha256 of ``json.dumps(pipeline_config(_spec("off")))``: the value
#: ``test_word_confidence_switch.OFF_BEFORE_THE_SWITCH["rnnt_default"]`` holds, computed
#: at 6583a83 before word confidence existed, for the same spec.
OFF_DIGEST = "482b6f46854990d9155149637ebce2c71df1c1d26449111ffd0efd3eec8578d7"

cpu = pytest.mark.cpu


def _spec(mode: str = "off") -> NeMoPipelineSpec:
    return NeMoPipelineSpec(
        model=REPO,
        chunk=ChunkMode(160),
        att_context=(70, 1),
        num_slots=31,
        batch_size=8,
        word_confidence=mode,
    )


def _importer(built: Any, seen: list[Any], omegaconf: Any = None) -> Callable[[str], Any]:
    """An ``import_module`` for ``build_pipeline`` whose NeMo builder records the
    configuration it is handed and returns ``built`` (or calls it, when callable)."""

    def build(cfg: Any) -> Any:
        seen.append(cfg)
        return built(cfg) if callable(built) else built

    modules = {
        "omegaconf": omegaconf or SimpleNamespace(OmegaConf=SimpleNamespace(create=lambda d: d)),
        BUILDER: SimpleNamespace(PipelineBuilder=SimpleNamespace(build_pipeline=build)),
    }
    return modules.__getitem__


# --- build_pipeline, without NeMo --------------------------------------------------------


class _Recorded:
    """An attribute bag that records every write made to it after it was built."""

    def __init__(self, **attrs: Any) -> None:
        object.__setattr__(self, "writes", [])
        for name, value in attrs.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        self.writes.append((name, value))
        object.__setattr__(self, name, value)


def _recorded_pipeline(flag: Any = True) -> tuple[_Recorded, list[_Recorded]]:
    """A pipeline shaped like NeMo's built one down to the model's decoding object, every
    node recording writes. The decoding object carries the three confidence flags as
    NeMo's builder leaves them with word confidence on, the word flag set to ``flag``."""
    decoding = _Recorded(
        preserve_frame_confidence=True,
        preserve_token_confidence=True,
        preserve_word_confidence=flag,
    )
    model = _Recorded(decoding=decoding)
    wrapper = _Recorded(asr_model=model)
    pipeline = _Recorded(asr_model=wrapper)
    return pipeline, [pipeline, wrapper, model, decoding]


@cpu
@pytest.mark.parametrize("flag", [False, True], ids=["nemo-false", "nemo-true"])
def test_off_hands_nemo_the_old_bytes_and_writes_nothing_to_what_it_built(flag: bool) -> None:
    """NeMo's builder leaves the word flag false under "off"; True stands for a NeMo that
    did not, so a write of False over False is not the only thing that can be seen."""
    pipeline, nodes = _recorded_pipeline(flag)
    seen: list[Any] = []
    assert build_pipeline(_spec("off"), _importer(pipeline, seen)) is pipeline
    (cfg,) = seen
    assert hashlib.sha256(json.dumps(cfg).encode()).hexdigest() == OFF_DIGEST
    assert [node.writes for node in nodes] == [[], [], [], []]
    assert nodes[-1].preserve_word_confidence is flag


@cpu
@pytest.mark.parametrize("mode", ON)
def test_on_writes_one_attribute_the_word_flag_and_nothing_else(mode: str) -> None:
    pipeline, nodes = _recorded_pipeline(True)
    assert build_pipeline(_spec(mode), _importer(pipeline, [])) is pipeline
    assert [node.writes for node in nodes] == [[], [], [], [("preserve_word_confidence", False)]]
    decoding = nodes[-1]
    assert decoding.preserve_frame_confidence is True
    assert decoding.preserve_token_confidence is True
    assert decoding.preserve_word_confidence is False


def _shaped(decoding: Any) -> Any:
    return SimpleNamespace(asr_model=SimpleNamespace(asr_model=SimpleNamespace(decoding=decoding)))


_UNSWITCHABLE = {
    "no-wrapper": SimpleNamespace(),
    "no-decoding": SimpleNamespace(asr_model=SimpleNamespace(asr_model=SimpleNamespace())),
    "no-flag": _shaped(SimpleNamespace()),
    "flag-none": _shaped(SimpleNamespace(preserve_word_confidence=None)),
    "flag-string": _shaped(SimpleNamespace(preserve_word_confidence="false")),
    "flag-int": _shaped(SimpleNamespace(preserve_word_confidence=1)),
}


@cpu
@pytest.mark.parametrize("case", sorted(_UNSWITCHABLE))
def test_on_refuses_a_pipeline_whose_switch_it_cannot_find(case: str) -> None:
    """Served anyway, such a pipeline would run until the first transcript that trips
    NeMo's aggregation. "off" never looks, so the same object is returned there."""
    pipeline = _UNSWITCHABLE[case]
    with pytest.raises(PipelineBuildError, match="boolean preserve_word_confidence"):
        build_pipeline(_spec("paper-best"), _importer(pipeline, []))
    assert build_pipeline(_spec("off"), _importer(pipeline, [])) is pipeline


# --- what /readyz observes, without NeMo --------------------------------------------------


@cpu
@pytest.mark.parametrize(
    ("flag", "reading"),
    [(False, False), (True, True), (None, None), ("false", None), (0, None)],
    ids=repr,
)
def test_observe_reads_the_decoding_objects_word_flag(flag: Any, reading: Any) -> None:
    """Only a bool is an observation; ``bool("false")`` would report True."""
    pipeline, _ = _recorded_pipeline(flag)
    facts = observe(pipeline)
    assert facts.decoder_word_confidence is reading
    assert facts.to_json_dict()["decoder_word_confidence"] is reading


@cpu
def test_a_pipeline_without_a_decoding_object_observes_none() -> None:
    assert observe(None).decoder_word_confidence is None
    assert observe(SimpleNamespace()).decoder_word_confidence is None
    assert observe(_shaped(SimpleNamespace())).decoder_word_confidence is None
    assert OBSERVED_KEYS[-1] == "decoder_word_confidence"
    assert tuple(observe(None).to_json_dict()) == OBSERVED_KEYS


class _WithDecoding(FakeCacheAwareRNNTPipeline):
    """``nemo_fake``'s RNNT pipeline with a model decoding object where NeMo's built
    pipeline keeps it."""

    def __init__(self, flag: bool) -> None:
        super().__init__(160, num_slots=64)
        decoding = SimpleNamespace(preserve_word_confidence=flag)
        self.asr_model = SimpleNamespace(asr_model=SimpleNamespace(decoding=decoding))


async def _observed_now(endpoints: Endpoints) -> Mapping[str, Any]:
    """``/readyz``'s ``observed`` object from the first 200 answer."""
    url = f"http://127.0.0.1:{endpoints.ws_port}/readyz"

    def fetch() -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    for _ in range(400):
        status, body = await asyncio.to_thread(fetch)
        if status == 200:
            observed: Mapping[str, Any] = json.loads(body)["observed"]
            return observed
        await asyncio.sleep(0.025)
    raise AssertionError(f"/readyz never answered 200: {status} {body}")


@cpu
async def test_readyz_reads_the_word_flag_off_the_built_object_on_every_request() -> None:
    settings = ServeSettings(
        model=REPO,
        chunk=ChunkMode(160),
        bucket=4,
        pipeline="cache_aware_rnnt",
        host="127.0.0.1",
        ws_port=0,
        grpc_port=0,
    )
    pipeline = _WithDecoding(False)
    adapter = registry.build_for(
        engine_config(settings),
        boundary=boundary_for(pipeline),
        language_code="en-US",
        use_cuda_graphs=False,
    )
    shutdown = asyncio.Event()
    ready: asyncio.Future[Endpoints] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        run_server(
            settings,
            adapter,
            shutdown=shutdown,
            on_ready=ready.set_result,
            execution="eager",
            runtime={},
        )
    )
    readings: list[Any] = []
    try:
        done, _ = await asyncio.wait({task, ready}, timeout=15, return_when=asyncio.FIRST_COMPLETED)
        assert ready in done, f"run_server did not come up: {task}"
        endpoints = ready.result()
        readings.append((await _observed_now(endpoints))["decoder_word_confidence"])
        pipeline.asr_model.asr_model.decoding.preserve_word_confidence = True
        readings.append((await _observed_now(endpoints))["decoder_word_confidence"])
        del pipeline.asr_model.asr_model.decoding
        readings.append((await _observed_now(endpoints))["decoder_word_confidence"])
    finally:
        shutdown.set()
        await asyncio.wait_for(task, timeout=15)
    assert readings == [False, True, None]


# --- the checkpoint and NeMo, when they are here -----------------------------------------
#
# Not marked cpu: that marker means "no NeMo". Nothing below touches a GPU:
# ``_nemo_checkpoint`` refuses to run unless CUDA_VISIBLE_DEVICES is "", and every tensor
# is made on the CPU.


def _checkpoint(lookup: Callable[..., Any] | None = None) -> Path:
    """The checkpoint's archive in the local Hugging Face cache, found by
    ``huggingface_hub``'s own cache lookup, or a skip. Nothing is downloaded."""
    if lookup is None:
        lookup = pytest.importorskip("huggingface_hub").try_to_load_from_cache
    found = lookup(REPO, CHECKPOINT, revision=REVISION)
    if not isinstance(found, str) or not os.path.isfile(found):
        pytest.skip(f"{REPO} at {REVISION[:8]} is not in the local Hugging Face cache")
    return Path(found)


def _nemo_checkpoint(
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    environ: Mapping[str, str] = os.environ,
    checkpoint: Callable[[], Path] = _checkpoint,
) -> Path:
    """The checkpoint, where NeMo is installed and the checkpoint is cached, or a skip.
    Where both are here and a GPU could be seen, a failure: these run NeMo on the CPU, and
    on a shared box the cards belong to other jobs."""
    if find_spec("nemo") is None:
        pytest.skip("NeMo is not installed")
    archive = checkpoint()
    if environ.get("CUDA_VISIBLE_DEVICES") != "":
        pytest.fail(
            "these tests run NeMo on the CPU and must not see a GPU: run them with"
            ' CUDA_VISIBLE_DEVICES=""'
        )
    return archive


@cpu
def test_the_nemo_tests_skip_without_nemo_or_the_checkpoint_and_refuse_a_visible_gpu() -> None:
    installed = lambda name: object()  # noqa: E731
    here = Path("checkpoint-under-test.nemo")
    cpu_only = {"CUDA_VISIBLE_DEVICES": ""}

    def uncached() -> Path:
        pytest.skip("not in the cache")

    def outcome(**kwargs: Any) -> Any:
        try:
            return _nemo_checkpoint(**kwargs)
        except BaseException as raised:  # skip and fail are BaseExceptions
            return type(raised)

    skip, fail = pytest.skip.Exception, pytest.fail.Exception
    assert outcome(find_spec=lambda name: None, environ=cpu_only, checkpoint=uncached) is skip
    # No checkpoint, nothing runs NeMo: a skip, even where a GPU could be seen.
    for environ in (cpu_only, {"CUDA_VISIBLE_DEVICES": "3"}, {}):
        assert outcome(find_spec=installed, environ=environ, checkpoint=uncached) is skip
    for environ in ({"CUDA_VISIBLE_DEVICES": "3"}, {}):
        assert outcome(find_spec=installed, environ=environ, checkpoint=lambda: here) is fail
    assert outcome(find_spec=installed, environ=cpu_only, checkpoint=lambda: here) == here


@cpu
def test_the_checkpoint_comes_from_the_cache_lookup_or_the_tests_skip() -> None:
    asked: list[Any] = []

    def absent(*args: Any, **kwargs: Any) -> Any:
        asked.append((args, kwargs))
        return None

    with pytest.raises(BaseException) as raised:
        _checkpoint(absent)
    assert raised.type is pytest.skip.Exception
    assert asked == [((REPO, CHECKPOINT), {"revision": REVISION})]
    with pytest.raises(BaseException) as known_absent:  # the cache's "known not to exist"
        _checkpoint(lambda *args, **kwargs: object())
    assert known_absent.type is pytest.skip.Exception
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, CHECKPOINT)
        with pytest.raises(BaseException) as missing:
            _checkpoint(lambda *args, **kwargs: path)
        assert missing.type is pytest.skip.Exception
        Path(path).write_bytes(b"")
        assert _checkpoint(lambda *args, **kwargs: path) == Path(path)


def _tokenizer(archive_path: Path) -> Any:
    """The checkpoint's tokenizer, set up by NeMo's own ``ASRBPEMixin._setup_tokenizer``
    from the tokenizer files the archive's ``model_config.yaml`` names. The files are
    extracted to a temporary directory that is gone before this returns: the tokenizer
    holds its model in memory."""
    from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin
    from omegaconf import OmegaConf

    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive_path, "r:*") as archive:
        members = {PurePosixPath(m.name).name: m for m in archive.getmembers() if m.isfile()}

        def read(name: str) -> bytes:
            handle = archive.extractfile(members[name])
            assert handle is not None, name
            return handle.read()

        config = OmegaConf.create(read("model_config.yaml").decode("utf-8"))
        files: dict[str, Path] = {}
        for key in ("model_path", "vocab_path", "spe_tokenizer_vocab"):
            name = str(config.tokenizer[key]).removeprefix("nemo:")
            files[name] = Path(tmp) / name
            files[name].write_bytes(read(name))

        class TokenizerOnly(ASRBPEMixin):
            """What the model class runs to make its tokenizer, and nothing else. A
            ``nemo:`` artifact path is the extracted file, as the model's own
            ``register_artifact`` resolves it."""

            def register_artifact(
                self, config_path: str, src: str, verify_src_exists: bool = True
            ) -> str:
                return str(files[src.removeprefix("nemo:")])

        host = TokenizerOnly()
        host._setup_tokenizer(config.tokenizer)
    return host.tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return _tokenizer(_nemo_checkpoint())


@contextlib.contextmanager
def _one_thread() -> Iterator[None]:
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


@pytest.fixture(scope="module")
def nets(tokenizer: Any) -> SimpleNamespace:
    """Seeded toy prediction and joint networks over the checkpoint's vocabulary, and an
    encoder output for two streams of 24 frames. The joint's last layer is scaled so its
    distributions are peaked enough for an entropy confidence to be far from 0 over 1,025
    classes, and blank is favoured so a frame does not always emit ``max_symbols``."""
    import torch
    from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint

    size = len(tokenizer.vocab)
    with _one_thread():
        torch.manual_seed(0)
        decoder = RNNTDecoder(prednet={"pred_hidden": 16, "pred_rnn_layers": 1}, vocab_size=size)
        joint = RNNTJoint(
            jointnet={
                "encoder_hidden": 12,
                "pred_hidden": 16,
                "joint_hidden": 24,
                "activation": "relu",
            },
            num_classes=size,
        )
        decoder.eval()
        joint.eval()
        with torch.no_grad():
            joint.joint_net[-1].weight *= 100.0
            joint.joint_net[-1].bias[size] += 30.0
        encoded = torch.randn(2, 12, 24)
    return SimpleNamespace(decoder=decoder, joint=joint, encoded=encoded)


class _NeMoBuilder:
    """Stands in for NeMo's ``PipelineBuilder`` without loading a model: NeMo's own
    ``CacheAwarePipelineBuilder.get_rnnt_decoding_cfg`` reads the configuration it is
    handed, NeMo's ``RNNTBPEDecoding`` is built from that with the checkpoint's tokenizer
    over the toy networks, and it is put where a built cache-aware RNNT pipeline keeps it:
    ``pipeline.asr_model`` (the wrapper) ``.asr_model`` (the model) ``.decoding``, with
    the decoding configuration on the wrapper. NeMo's own ``init_decoding_computer`` then
    finds the decoding computer on it."""

    def __init__(self, tokenizer: Any, nets: SimpleNamespace, *, record: bool = False) -> None:
        self.tokenizer = tokenizer
        self.nets = nets
        self.record = record
        self.writes: list[tuple[str, Any]] = []

    def build_pipeline(self, cfg: Any) -> Any:
        from nemo.collections.asr.inference.factory.cache_aware_pipeline_builder import (
            CacheAwarePipelineBuilder,
        )
        from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import (
            CacheAwareRNNTPipeline,
        )
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding

        decoding_cfg = CacheAwarePipelineBuilder.get_rnnt_decoding_cfg(cfg)
        decoding = RNNTBPEDecoding(
            decoding_cfg, decoder=self.nets.decoder, joint=self.nets.joint, tokenizer=self.tokenizer
        )
        if self.record:
            self._record(decoding)
        model = SimpleNamespace(decoding=decoding)
        pipeline = SimpleNamespace(
            asr_model=SimpleNamespace(asr_model=model, decoding_cfg=decoding_cfg)
        )
        CacheAwareRNNTPipeline.init_decoding_computer(pipeline)
        return pipeline

    def _record(self, decoding: Any) -> None:
        """Every write to NeMo's decoding object from here on, into ``self.writes``."""
        base = type(decoding)
        writes = self.writes

        def __setattr__(obj: Any, name: str, value: Any) -> None:
            writes.append((name, value))
            base.__setattr__(obj, name, value)

        decoding.__class__ = type(f"Recorded{base.__name__}", (base,), {"__setattr__": __setattr__})


def _built(mode: str, tokenizer: Any, nets: SimpleNamespace) -> Any:
    """``build_pipeline`` for ``mode`` over the stand-in, as a server gets it."""
    import omegaconf

    builder = _NeMoBuilder(tokenizer, nets)
    return build_pipeline(_spec(mode), _importer(builder.build_pipeline, [], omegaconf))


def _nemo_built(mode: str, tokenizer: Any, nets: SimpleNamespace) -> Any:
    """What NeMo's builder alone returns for the same configuration: the old path."""
    from omegaconf import OmegaConf

    builder = _NeMoBuilder(tokenizer, nets)
    return builder.build_pipeline(OmegaConf.create(pipeline_config(_spec(mode))))


def _decoding(pipeline: Any) -> Any:
    """The model's decoding object, walked here by hand rather than by ``DECODING_PATH``."""
    return pipeline.asr_model.asr_model.decoding


def _crash_ids(tokenizer: Any, before: tuple[str, ...] = ("'",)) -> list[int]:
    """The crash text's tokens as the tokenizer writes them, with a lone word-start token
    inserted before each token in ``before``."""
    vocab = tokenizer.vocab
    lone = vocab.index("▁")
    ids: list[int] = []
    for token in tokenizer.text_to_ids(CRASH_TEXT):
        if vocab[token] in before:
            ids.append(lone)
        ids.append(token)
    return ids


def _replay(decoding: Any, ids: list[int], confidences: list[float]) -> None:
    """Replace the decoding object's inner decoder with one that returns a hypothesis of
    ``ids``. Everything ``rnnt_decoder_predictions_tensor`` does after the decode itself,
    ``decode_hypothesis`` and ``compute_confidence``, stays NeMo's."""
    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    def decode(encoder_output: Any, encoded_lengths: Any, partial_hypotheses: Any = None) -> Any:
        hypothesis = Hypothesis(
            score=0.0,
            y_sequence=torch.tensor(ids),
            timestamp=torch.arange(len(ids)),
            non_blank_step_confidence_precomputed=list(confidences),
        )
        return ([hypothesis],)

    decoding.decoding = decode


def _predict(decoding: Any, frames: int) -> Any:
    import torch

    return decoding.rnnt_decoder_predictions_tensor(
        torch.zeros(1, 12, frames), torch.tensor([frames]), return_hypotheses=True
    )


def _pipeline_words(tokenizer: Any, mode: str, hyp: Any, frames: int) -> list[tuple[Any, ...]]:
    """The words NeMo's streaming pipeline makes of ``hyp``: its ``RNNTGreedyDecoder`` takes
    the tokens, timestamps and step confidences off the hypothesis
    (``inference/pipelines/cache_aware_rnnt_pipeline.py:394-400``), and its ``BPEDecoder``
    groups them into words, each word's ``conf`` the ``confidence`` block's aggregate of its
    tokens (``inference/utils/bpe_decoder.py:205``). The pipeline's supported punctuation
    leaves the apostrophe out (``asr_inference_wrapper.py:160``)."""
    from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import (
        RNNTGreedyDecoder,
    )
    from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
    from nemo.collections.asr.inference.utils.pipeline_utils import get_confidence_utils
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(pipeline_config(_spec(mode)))
    conf_func, aggregate = get_confidence_utils(cfg.confidence)
    vocabulary = list(tokenizer.tokenizer.get_vocab())
    greedy = RNNTGreedyDecoder(vocabulary=vocabulary, conf_func=conf_func)
    out, _, _ = greedy(
        global_timestamps=hyp.timestamp,
        tokens=hyp.y_sequence,
        length=frames,
        confidences=hyp.non_blank_step_confidence_precomputed,
    )
    bpe = BPEDecoder(
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        confidence_aggregator=aggregate,
        asr_supported_puncts=tokenizer.supported_punctuation - {"'"},
        word_boundary_tolerance=cfg.streaming.word_boundary_tolerance,
        token_duration_in_secs=0.08,
    )
    words, _ = bpe.group_tokens_into_words(out["tokens"], out["timesteps"], out["confidences"])
    return [(w.text, w.start, w.end, w.conf) for w in words]


def _spy(obj: Any, name: str) -> list[Any]:
    """Every call to ``obj.name`` from here on, passed through to the original."""
    calls: list[Any] = []
    original = getattr(obj, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    setattr(obj, name, spy)
    return calls


#: Three chunks of the toy encoder output, decoded with the previous step's hypotheses as
#: the cache-aware wrapper does (``cache_aware_rnnt_inference_wrapper.py:192-194``).
CHUNKS = ((0, 8), (8, 16), (16, 24))


def _stream(decoding: Any, encoded: Any) -> tuple[list[list[dict[str, Any]]], list[Any]]:
    """Every step's hypotheses, copied as they were at that step, and the last step's."""
    import torch

    steps: list[list[dict[str, Any]]] = []
    previous = None
    with _one_thread():
        for lo, hi in CHUNKS:
            hyps = decoding.rnnt_decoder_predictions_tensor(
                encoded[:, :, lo:hi].contiguous(),
                torch.full((encoded.shape[0],), hi - lo),
                return_hypotheses=True,
                partial_hypotheses=previous,
            )
            steps.append(
                [
                    {
                        "tokens": h.y_sequence.tolist(),
                        "timestamp": h.timestamp.tolist(),
                        "text": h.text,
                        "step_confidence": list(h.non_blank_step_confidence_precomputed),
                        "token_confidence": (
                            None if h.token_confidence is None else list(h.token_confidence)
                        ),
                        "word_confidence": (
                            None if h.word_confidence is None else list(h.word_confidence)
                        ),
                    }
                    for h in hyps
                ]
            )
            previous = hyps
    assert previous is not None
    return steps, previous


def test_the_checkpoints_tokenizer_writes_the_crash_text_with_no_lone_word_start(
    tokenizer: Any,
) -> None:
    """The tokens the tokenizer writes for the text, and the pieces the rest leans on: a
    lone word start exists, and no piece is a word start fused to the apostrophe."""
    ids = tokenizer.text_to_ids(CRASH_TEXT)
    assert tokenizer.ids_to_tokens(ids) == [
        "▁He",
        "▁has",
        "▁one",
        "▁s",
        "on",
        ",",
        "▁and",
        "'",
        "t",
        "is",
    ]
    assert "▁" in tokenizer.vocab
    assert "▁'" not in tokenizer.vocab and "▁," not in tokenizer.vocab
    assert tokenizer.ids_to_text(_crash_ids(tokenizer)) == "He has one son, and 'tis"


@pytest.mark.parametrize("mode", ON)
@pytest.mark.parametrize(
    ("before", "count"),
    [((), None), (("'",), 6), ((",",), 6), (("'", ","), 7)],
    ids=["as-written", "lone-before-apostrophe", "lone-before-comma", "lone-before-both"],
)
def test_nemo_raises_on_a_lone_word_start_before_a_mark_and_not_otherwise(
    tokenizer: Any, nets: SimpleNamespace, mode: str, before: tuple[str, ...], count: Any
) -> None:
    """The reproduction, on NeMo's decoding object as its builder makes it: every one of
    these decodes to the crash text, the tokens as written aggregate to five word
    confidences, and each lone word start before a mark adds one the text has no word for."""
    decoding = _decoding(_nemo_built(mode, tokenizer, nets))
    assert decoding.preserve_word_confidence is True
    ids = _crash_ids(tokenizer, before)
    assert decoding.decode_tokens_to_str_with_strip_punctuation(ids) == CRASH_TEXT
    _replay(decoding, ids, [0.5] * len(ids))
    if count is None:
        (hyp,) = _predict(decoding, len(ids))
        assert hyp.text == CRASH_TEXT and len(hyp.word_confidence) == 5
        return
    with pytest.raises(RuntimeError) as raised:
        _predict(decoding, len(ids))
    message = " ".join(str(raised.value).split())
    assert "Something went wrong with word-level confidence aggregation" in message
    assert "len(words): 5," in message
    assert f"len(word_confidence): {count}," in message
    assert "recognized text: `He has one son, and'tis`" in message
    frames = [frame.name for frame in traceback.extract_tb(raised.tb)]
    assert frames[-4:] == [
        "rnnt_decoder_predictions_tensor",
        "compute_confidence",
        "_aggregate_token_confidence",
        "_aggregate_token_confidence_subwords_sentencepiece",
    ]


@pytest.mark.parametrize("mode", ON)
def test_build_pipeline_switches_off_the_word_aggregation_and_nothing_else(
    tokenizer: Any, nets: SimpleNamespace, mode: str
) -> None:
    pipeline = _built(mode, tokenizer, nets)
    decoding = _decoding(pipeline)
    old = _decoding(_nemo_built(mode, tokenizer, nets))
    flags = ("preserve_frame_confidence", "preserve_token_confidence", "preserve_word_confidence")
    assert tuple(getattr(old, flag) for flag in flags) == (True, True, True)
    assert tuple(getattr(decoding, flag) for flag in flags) == (True, True, False)
    # NeMo's own walk to the decoding computer goes through the object the switch is on,
    # and the path build_pipeline writes through is that walk.
    assert pipeline.decoding_computer is decoding.decoding.decoding_computer
    assert pipeline.decoding_computer.preserve_step_confidence is True
    assert DECODING_PATH == ("asr_model", "asr_model", "decoding")
    # The configuration NeMo applied is untouched, so /readyz still names the mode.
    assert pipeline.asr_model.decoding_cfg.confidence_cfg.preserve_word_confidence is True
    adapter = SimpleNamespace(_boundary=SimpleNamespace(pipeline=pipeline))
    assert configured_word_confidence(adapter) == mode
    facts = observe(pipeline)
    assert (facts.decoder_step_confidence, facts.decoder_word_confidence) == (True, False)
    assert observe(_nemo_built(mode, tokenizer, nets)).decoder_word_confidence is True


@pytest.mark.parametrize("mode", ON)
def test_the_crash_hypothesis_decodes_after_build_pipeline(
    tokenizer: Any, nets: SimpleNamespace, mode: str
) -> None:
    """The same hypothesis (the lone word start before the apostrophe), through the same
    NeMo code, before and after: the old object raises, the built one returns the text and
    the token confidences. The pipeline's own words for it have "and" and "'tis" where
    NeMo's text has "and'tis", each with the aggregate of its own tokens' confidences."""
    ids = _crash_ids(tokenizer)
    confidences = [round(0.9 - 0.05 * i, 2) for i in range(len(ids))]
    old = _decoding(_nemo_built(mode, tokenizer, nets))
    _replay(old, ids, confidences)
    with pytest.raises(RuntimeError, match=r"len\(word_confidence\): 6"):
        _predict(old, len(ids))
    fixed = _decoding(_built(mode, tokenizer, nets))
    _replay(fixed, ids, confidences)
    (hyp,) = _predict(fixed, len(ids))
    assert hyp.text == CRASH_TEXT
    assert hyp.token_confidence == confidences
    assert hyp.word_confidence is None
    from nemo.collections.asr.inference.utils.pipeline_utils import get_confidence_utils
    from omegaconf import OmegaConf

    _, aggregate = get_confidence_utils(OmegaConf.create(pipeline_config(_spec(mode))).confidence)
    words = _pipeline_words(tokenizer, mode, hyp, len(ids))
    # He | has | one | s on , | and | lone-word-start ' t is
    groups = [(0, 1), (1, 2), (2, 3), (3, 6), (6, 7), (7, 11)]
    assert [w[0] for w in words] == ["He", "has", "one", "son,", "and", "'tis"]
    assert [w[3] for w in words] == [aggregate(confidences[a:b]) for a, b in groups]


@pytest.mark.parametrize("mode", ON)
def test_compute_confidence_still_runs_and_no_longer_calls_the_word_aggregator(
    tokenizer: Any, nets: SimpleNamespace, mode: str
) -> None:
    arms = {
        "old": _decoding(_nemo_built(mode, tokenizer, nets)),
        "built": _decoding(_built(mode, tokenizer, nets)),
    }
    computed = {arm: _spy(decoding, "compute_confidence") for arm, decoding in arms.items()}
    aggregated = {
        arm: _spy(decoding, "_aggregate_token_confidence") for arm, decoding in arms.items()
    }
    for decoding in arms.values():
        _stream(decoding, nets.encoded)
    streams = nets.encoded.shape[0]
    assert {arm: len(calls) for arm, calls in computed.items()} == {
        "old": len(CHUNKS),
        "built": len(CHUNKS),
    }
    assert {arm: len(calls) for arm, calls in aggregated.items()} == {
        "old": len(CHUNKS) * streams,
        "built": 0,
    }


@pytest.mark.parametrize("mode", ON)
def test_token_and_pipeline_word_confidences_are_what_they_were(
    tokenizer: Any, nets: SimpleNamespace, mode: str
) -> None:
    """A toy streaming decode the old path survives, through the old object and the built
    one: every step's tokens, timestamps, text, step and token confidences are equal, and
    so are the words and word confidences NeMo's pipeline makes of the last step. Only
    ``word_confidence`` on the hypothesis, which nothing reads, differs."""
    old_steps, old_last = _stream(_decoding(_nemo_built(mode, tokenizer, nets)), nets.encoded)
    new_steps, new_last = _stream(_decoding(_built(mode, tokenizer, nets)), nets.encoded)
    old_rows = [row for step in old_steps for row in step]
    new_rows = [row for step in new_steps for row in step]
    assert all(len(row["word_confidence"]) == len(row["text"].split()) for row in old_rows)
    assert all(row["word_confidence"] is None for row in new_rows)

    def without_words(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if k != "word_confidence"} for row in rows]

    assert without_words(new_rows) == without_words(old_rows)
    for row in new_rows:
        assert row["token_confidence"] == row["step_confidence"]
    last = new_steps[-1]
    assert all(len(row["tokens"]) > 20 for row in last), "the toy decoded too little to show"
    confidences = [c for row in last for c in row["token_confidence"]]
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert len({round(c, 3) for c in confidences}) > 10, "the confidences do not vary"

    frames = CHUNKS[-1][1]
    old_words = [_pipeline_words(tokenizer, mode, h, frames) for h in old_last]
    new_words = [_pipeline_words(tokenizer, mode, h, frames) for h in new_last]
    assert new_words == old_words
    assert all(len(words) > 5 for words in new_words)
    assert len({round(w[3], 3) for words in new_words for w in words}) > 5


def test_off_leaves_nemos_own_decoding_object_as_nemo_built_it(
    tokenizer: Any, nets: SimpleNamespace
) -> None:
    import omegaconf

    builder = _NeMoBuilder(tokenizer, nets, record=True)
    seen: list[Any] = []
    pipeline = build_pipeline(_spec("off"), _importer(builder.build_pipeline, seen, omegaconf))
    decoding = _decoding(pipeline)
    assert builder.writes == []
    assert decoding.preserve_frame_confidence is False
    assert decoding.preserve_token_confidence is False
    assert decoding.preserve_word_confidence is False
    (cfg,) = seen
    text = json.dumps(omegaconf.OmegaConf.to_container(cfg))
    assert hashlib.sha256(text.encode()).hexdigest() == OFF_DIGEST
    facts = observe(pipeline)
    assert (facts.decoder_step_confidence, facts.decoder_word_confidence) == (False, False)
    # The recorder sees a write: an "on" build through the same stand-in makes one.
    on = _NeMoBuilder(tokenizer, nets, record=True)
    build_pipeline(_spec("paper-best"), _importer(on.build_pipeline, [], omegaconf))
    assert on.writes == [("preserve_word_confidence", False)]
