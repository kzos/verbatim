# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Drive LiveKit's own NVIDIA speech-to-text plugin against a Verbatim server, unchanged.

Verbatim speaks the Riva ``StreamingRecognize`` subset that ``livekit-plugins-nvidia`` was
written for, so the plugin is pointed at the server and nothing else is written: this file
is the recipe, not a plugin. Three settings decide it. ``server`` is the Verbatim gRPC
listener; ``use_ssl=False`` because the open build serves plaintext and the plugin then
needs no API key; ``model`` is the name Verbatim reports for what it loaded, which
``GetRivaSpeechRecognitionConfig`` returns and ``verbatim serve`` prints in its banner,
because the plugin's default names NVIDIA's hosted model and a name the server does not
serve is refused as NOT_FOUND rather than substituted.

The plugin still sends its ``function-id`` metadata header and, when an API key is set,
``authorization``; Verbatim accepts and ignores both. It sends 16 kHz mono PCM16 and asks
for interim results and word times; Verbatim answers with partials, then a final whose
words carry millisecond offsets the plugin divides by a thousand.

Run it against a live server with a 16 kHz mono WAV:

    uv run --with livekit-plugins-nvidia --with livekit-agents --with soundfile \\
        examples/livekit/verbatim_livekit_stt.py --server 127.0.0.1:50051 \\
        --model nvidia/stt_en_fastconformer_hybrid_large_streaming_multi audio.wav

Without a GPU, ``verbatim serve <name> --pipeline fake --bucket 8`` serves scripted
transcripts over the same wire, and the same command against it is the protocol proof:
``tests/examples/test_livekit_live_call.py`` runs exactly that in process.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

FRAME_MS = 20
SAMPLE_RATE_HZ = 16000
_FINAL_WAIT_S = 15.0


@dataclass
class CallResult:
    """What one call through the plugin produced."""

    interim: list[str] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    word_times_ms: list[tuple[str, int, int]] = field(default_factory=list)
    error: str | None = None

    @property
    def transcript(self) -> str:
        return " ".join(self.finals)


def frames(pcm: bytes, *, frame_ms: int = FRAME_MS) -> Iterable[bytes]:
    """PCM16 mono bytes as wire frames of ``frame_ms``; the last one may be short."""
    step = frame_ms * SAMPLE_RATE_HZ // 1000 * 2
    for start in range(0, len(pcm), step):
        yield pcm[start : start + step]


async def run_call(
    pcm: bytes,
    *,
    server: str,
    model: str,
    language_code: str = "en-US",
    real_time: bool = True,
    frame_ms: int = FRAME_MS,
) -> CallResult:
    """One recognition call through ``livekit.plugins.nvidia.STT`` with no plugin code.

    Frames are pushed at their real-time cadence when ``real_time`` is set, as a room
    would deliver them; the stream is then ended and read until the plugin closes it.
    """
    from livekit import rtc
    from livekit.agents import stt as agents_stt
    from livekit.plugins import nvidia

    result = CallResult()
    engine = nvidia.STT(server=server, use_ssl=False, model=model, language_code=language_code)
    stream = engine.stream()

    async def events() -> AsyncIterator[agents_stt.SpeechEvent]:
        async for event in stream:
            yield event

    async def reader() -> None:
        try:
            async for event in events():
                if event.type == agents_stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    result.interim.append(event.alternatives[0].text)
                elif event.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT:
                    alt = event.alternatives[0]
                    result.finals.append(alt.text)
                    for word in getattr(alt, "words", None) or ():
                        start = getattr(word, "start_time", None)
                        end = getattr(word, "end_time", None)
                        if start is not None and end is not None:
                            result.word_times_ms.append(
                                (word.word, round(start * 1000), round(end * 1000))
                            )
        except Exception as exc:  # the plugin's own failure is the datum
            result.error = f"{type(exc).__name__}: {exc}"

    reading = asyncio.create_task(reader())
    try:
        samples_per_frame = frame_ms * SAMPLE_RATE_HZ // 1000
        t0 = time.monotonic()
        for index, frame in enumerate(frames(pcm, frame_ms=frame_ms)):
            audio = rtc.AudioFrame(
                data=frame,
                sample_rate=SAMPLE_RATE_HZ,
                num_channels=1,
                samples_per_channel=len(frame) // 2,
            )
            stream.push_frame(audio)
            if real_time:
                due = t0 + (index + 1) * frame_ms / 1000.0
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        del samples_per_frame
        stream.end_input()
        try:
            await asyncio.wait_for(reading, timeout=_FINAL_WAIT_S)
        except TimeoutError:
            if result.error is None:
                result.error = "TimeoutError: the plugin did not close the stream after end_input"
    finally:
        if not reading.done():
            reading.cancel()
        await stream.aclose()
    return result


def _read_wav(path: str) -> bytes:
    import soundfile as sf

    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE_HZ or info.channels != 1:
        raise SystemExit(
            f"{path}: {info.samplerate} Hz, {info.channels} channel(s); "
            f"this recipe sends {SAMPLE_RATE_HZ} Hz mono as the plugin does"
        )
    data, _ = sf.read(path, dtype="int16", always_2d=False)
    return data.tobytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wav", help="a 16 kHz mono WAV")
    parser.add_argument("--server", default="127.0.0.1:50051", help="Verbatim's gRPC host:port")
    parser.add_argument(
        "--model", required=True, help="the model name Verbatim reports, or an empty string"
    )
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--as-fast-as-possible", action="store_true")
    args = parser.parse_args(argv)
    pcm = _read_wav(args.wav)
    result = asyncio.run(
        run_call(
            pcm,
            server=args.server,
            model=args.model,
            language_code=args.language,
            real_time=not args.as_fast_as_possible,
        )
    )
    for text in result.interim:
        print(f"partial: {text}")
    for text in result.finals:
        print(f"final:   {text}")
    for word, start_ms, end_ms in result.word_times_ms:
        print(f"  {word:<16} {start_ms:6d} {end_ms:6d} ms")
    if result.error is not None:
        print(f"error: {result.error}", file=sys.stderr)
        return 2
    return 0 if result.finals else 1


if __name__ == "__main__":
    raise SystemExit(main())
