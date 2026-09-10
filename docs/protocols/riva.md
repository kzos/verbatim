# The Riva `StreamingRecognize` subset — supported fields

> **Placeholder.** This page will be **generated** from
> `src/verbatim/protocols/riva/conformance.py`, and CI will fail if the two diverge. Nothing here is
> implemented yet: neither the servicer nor the field table it would be generated from exists. Until then, read this as the specification the generator must
> reproduce, not as a description of running code.

Verbatim implements a *subset* of Riva ASR. A self-hoster deciding whether to point an existing client
at it needs that subset written down before they install anything, so the subset is data
(`conformance.py`), the page is generated from the data, and a proto field with no row in the table is
a build failure.

## Provenance

Protos vendored from [`nvidia-riva/common`](https://github.com/nvidia-riva/common) at
`268890b7286031a6d4950e34f7ce13ed0d4ce621`, MIT. See [`../third_party.md`](../third_party.md) for why
that SHA and not the `stable` tag.

## Service `nvidia.riva.asr.RivaSpeechRecognition`

| RPC | Verbatim |
|---|---|
| `StreamingRecognize(stream StreamingRecognizeRequest) returns (stream StreamingRecognizeResponse)` | **implemented** — the whole product |
| `GetRivaSpeechRecognitionConfig(RivaSpeechRecognitionConfigRequest) returns (RivaSpeechRecognitionConfigResponse)` | **implemented** — `livekit-plugins-nvidia`'s `log_asr_models()` calls it at setup and reads `model_config[].parameters["type"] == "online"` and `["language_code"]`; omitting it breaks the plugin |
| `Recognize(RecognizeRequest) returns (RecognizeResponse)` | `UNIMPLEMENTED`, with a message pointing at NeMo's offline path. Verbatim is a streaming server. |

## `RecognitionConfig` — the three-column table

Status is one of **honoured**, **accepted-and-ignored** (never fatal, always logged) or **rejected**
(an explicit status code naming the field). Unknown *fields* are ignored per proto3 semantics; unknown
*values* of a known field are rejected with `INVALID_ARGUMENT`, because a silent wrong answer is worse
than a refusal for the operators this serves.

| Field | Status | Behaviour |
|---|---|---|
| `encoding` | honoured / rejected | `LINEAR_PCM` is the default path (a buffer view, no decoder). `MULAW`/`ALAW` via the G.711 LUT. `FLAC` via `soundfile`. `OGGOPUS` only with the `[opus]` extra. `ENCODING_UNSPECIFIED` and anything else → `INVALID_ARGUMENT`. |
| `sample_rate_hertz` | honoured | any rate; resampled to 16 kHz in a decode worker thread |
| `audio_channel_count` | honoured / rejected | `> 1` is downmixed, unless `enable_separate_recognition_per_channel` → `UNIMPLEMENTED` |
| `language_code` | honoured | for prompt-conditioned checkpoints; an unknown code → `INVALID_ARGUMENT` listing the valid keys. Ignored for monolingual checkpoints. |
| `model` | honoured / rejected | must be `""` or the served model name/alias; otherwise `NOT_FOUND` |
| `max_alternatives` | accepted-and-ignored | 1 (greedy); `> 1` returns the top-1 only in year 1 |
| `profanity_filter` | accepted-and-ignored | no-op, documented |
| `enable_automatic_punctuation` | honoured | the checkpoints emit punctuation; `false` strips it server-side on the emitter, off the tick thread |
| `verbatim_transcripts` | honoured | `true` → ITN off; `false` → ITN on if the engine was started with it, else a warning |
| `enable_word_time_offsets` | honoured | `true` → word granularity, `WordInfo.start_time`/`end_time` in **int32 milliseconds** (LiveKit divides by 1000) |
| `speech_contexts[].phrases` / `.boost` | honoured | per-stream biasing (Pipecat's `add_word_boosting_to_config`) |
| `endpointing_config.stop_history_eou` | honoured | end-of-utterance silence in ms |
| `endpointing_config.stop_history` | honoured | used as the EOU silence when `stop_history_eou` is unset (Pipecat always sets `stop_history=320`) |
| `endpointing_config.start_history` / `start_threshold` / `stop_threshold` / `stop_threshold_eou` | accepted-and-ignored | there is no VAD stage in the pipeline |
| `custom_configuration` | honoured / accepted-and-ignored | `verbatim.chunk_ms`, `verbatim.decoder`; unknown keys ignored with a log line |
| `diarization_config.enable_speaker_diarization` | rejected | `true` → `UNIMPLEMENTED`; diarization is a year-2 session type. `speaker_tag` stays 0. |

## `StreamingRecognizeRequest`

| Field | Status | Behaviour |
|---|---|---|
| `streaming_config` (oneof, `= 1`) | honoured | must be the first message on the stream |
| `audio_content` (oneof, `= 2`) | honoured | raw bytes in the declared encoding |
| `runtime_config["force_eou"]` (`= 3`) | accepted-and-ignored | NeMo's pipeline exposes no force-EOU hook (zero occurrences of `force_eou` in `NVIDIA-NeMo/Speech` as of 2026-09-07; EOU is decided inside the endpointer), and Verbatim does not re-implement finalisation. Accepted with a warning until the upstream ask lands. Neither target plugin sends it. |
| `id` (`RequestId`, `= 100`) | honoured | echoed on every response |
| `StreamingRecognitionConfig.interim_results` | honoured | `false` suppresses partials |

## Responses

Every tick in which a session was scheduled produces one `StreamingRecognizeResponse` with one
`StreamingRecognitionResult`: `is_final=false`, `alternatives=[{transcript: <partial>}]`,
`stability=0.5`, `audio_processed=<seconds consumed>`, `channel_tag=1`. On end-of-utterance, a result
with `is_final=true`, the final transcript, `words` if requested, and `confidence` as the mean of the
word confidences — followed by an empty partial.

## Transport tolerances that decide "zero plugin code"

- `livekit-plugins-nvidia/auth.py` appends a `function-id` metadata header **unconditionally**, whether
  or not an API key is present; Pipecat appends it when set. **The server must accept and ignore
  unknown metadata.**
- `authorization: Bearer …` is likewise accepted and ignored in the default open build.
- Neither plugin needs `NVIDIA_API_KEY` when `use_ssl=False` — LiveKit warns only when SSL is on and
  the key is missing. That is the mechanical basis of the day-60 drop-in gate.
