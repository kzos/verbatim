# Backlog

Known, non-blocking follow-ups found while reviewing the code that introduced them. Every item here was
judged real and also judged not worth holding up a merge over: worth writing down, not worth a dedicated
fix round at the time. Publishing them is deliberate. A project whose one asset is precision about
evidence does not get to be vague about its own defects.

Entries are grouped by the part of the codebase they touch. Each bullet names a file and symbol, says
what is wrong or missing, and suggests a fix. Nothing here carries a severity, an estimate, or a
promised date — that is for triage once an entry becomes an issue.

## Harness and bench

- ~~`bench/src/verbatim_bench/client.py::_run_session_inner` — the loop that matches each sent chunk to
  its following partial event does a linear scan of `partial_events` per send, making the match
  O(chunks × partials) instead of linear. Both lists are already sorted by time, so a single merge walk
  with one cursor into `partial_events` would do the same match in O(chunks + partials).~~
- `bench/src/verbatim_bench/client.py` (`_FINAL_WAIT_S`) — the timeout for the first server message and
  for the final after `end` is a fixed module constant (15 s) with no way to override it per run.
  Thread it through `LoadSpec`/`run_session` as a parameter, so a slower or more loaded server under
  test does not need a code change to avoid false timeouts.
- `bench/src/verbatim_bench/client.py::SessionResult.started_at_s` — defaults to `0.0` and is only
  assigned once the session handshake succeeds, so a session that errors out before the first server
  message reports `started_at_s == 0.0`, indistinguishable from one that genuinely started at the
  clock's zero point. Default it to `None` instead, and only assign a real value once the session frame
  arrives.
- ~~`bench/src/verbatim_bench/results.py::RunResult.to_json_dict` — every neighbouring field in the emitted `config` block is read from `self.spec_dict`, but `x_ms` is the bare literal `150`; it matches the project's own documented SLO threshold (`config.py`'s `p95 partial latency <= chunk + 150 ms`) by coincidence, not by being wired to any config. Give `LoadSpec` an `x_ms` field and read it here like its neighbours, so a run at a different threshold reports the threshold it actually used.~~
- `tests/harness/test_pace.py::test_plan_start_offsets_uniform_spreads_over_the_ramp` — the test
  recomputes the per-session phase offset by replaying `random.Random(seed)` in the same call order as
  `pace.py::plan_start_offsets`, so it would still pass if the implementation's random draws changed in
  a way that broke pacing, as long as the test's replica changed identically. Assert against a fixed,
  recorded set of offsets for a known seed instead of re-deriving them from the same code path.
- `tests/harness/test_pace.py::test_session_error_is_recorded_not_raised` — builds a `RunResult` and
  asserts on the dict returned by `to_json_dict()` directly, never writing it to disk and reloading it,
  so it never exercises the JSON round trip (for example, that a missing latency survives as JSON
  `null` and reads back as `None`). Write the document with the real results writer and reload it before
  asserting, the way `tests/harness/test_verify.py::test_writer_output_round_trips_through_validate_and_verify`
  now does for the happy path.
- ~~`bench/src/verbatim_bench/nullserver.py` (the tail partial) — the null floor now emits a partial for a
  short final tail so its cadence matches the real server, but no test asserts that partial's own
  `audio_s`: the cross-check reads the client-computed value instead, so mutating the server's stamp
  leaves the suite green. Assert the tail partial's `audio_s` directly from the frame the floor sends.~~
- `bench/src/verbatim_bench/client.py::ChunkMode.parse` — accepts spellings beyond the documented `160`,
  `"160"` and `"160ms"` forms, such as `"160 ms"` with a space before the suffix, because it strips and
  re-joins the string instead of matching a strict grammar. Replace the manual strip/suffix logic with a
  single anchored regex so only the documented spellings pass.

## WebSocket protocol

- ~~`src/verbatim/protocols/ws/server.py::WsServer._handle` — the back-pressure drain loop and the
  `ring_seconds` knob it depends on are unreachable, measured at zero line hits while a one-megabyte
  message was driven through the server, because a synchronous recogniser consumes every chunk before
  the next receive.~~ **Resolved.** The transports run on the engine, so a ring sits between the socket
  and the pipeline and fills. The reader holds the remainder, waits a tick and offers it again, reading
  nothing meanwhile, so the peer's own flow control slows the client. The test drives two seconds of
  audio in one message against a half-second ring and asserts every byte reached the engine, that the
  first feed was short, and that at least one tick was waited per short feed.
- **The WebSocket demo protocol has no document, while the Riva subset has one.** The frames, the
  `error` codes a client can receive (`INVALID_ARGUMENT`, `RESOURCE_EXHAUSTED` with a retry hint, and
  `DEADLINE_EXCEEDED` when the idle deadline reclaims a slot) and the close codes (1000 after a refusal,
  1009 for a message over `max_message_bytes`) are described only in the module docstring and the tests.
  A demo protocol that is still moving is a bad thing to freeze into a document, so this is recorded
  rather than written: give it a page under `docs/protocols/` once the day-21 row has been taken over
  this surface, since that row is what makes the frames load-bearing.

- `src/verbatim/protocols/ws/frames.py::PartialFrame.to_json` — builds its JSON with
  `round(float(self.audio_s), 6)` and no guard against `NaN`; Python's `json.dumps` will happily emit
  the bare token `NaN`, which is not valid JSON and will break a strict client-side parser.
  `FinalFrame.to_json` has the same construct. Guard both before a real recognizer can ever produce a
  non-finite `audio_s`.
- `tests/protocol/test_ws_server.py::test_partial_bytes_are_buffered_until_a_chunk_completes` and
  `test_interim_results_false_suppresses_partials` — both prove "no message arrived" by racing a fixed
  50 ms `asyncio.wait_for` against the server, which can false-fail under load and would not notice a
  message that arrives a little later than that. Assert on the server's own internal state (buffered
  byte count, suppressed-partial counter) instead of timing a socket read.

## Riva gRPC subset

- `src/verbatim/protocols/riva/mapping.py::options_from_config` — falls back to the literal `"en-US"`
  whenever a request's `RecognitionConfig.language_code` is empty, ignoring
  `RivaServerConfig.language_code` entirely; the function has no parameter to receive it. Add a
  `default_language_code` parameter (the servicer already has `self._config.language_code` at the call
  site) and fall back to that instead of the literal.
- `tests/protocol/test_riva_conformance_table.py` — only checks `conformance.py::missing_fields()` for
  four named message types via `.get(name, ())`, so a gap in any other message type the function covers
  would pass silently. Add `assert missing_fields() == {}` alongside the named checks so an uncovered
  message type fails the build.
- ~~`src/verbatim/protocols/riva/server.py::RivaSpeechRecognitionServicer._emit` — takes a `recognizer`
  parameter that the method body never reads.~~ **Resolved:** the method and the `Recognizer` type are
  both gone; the servicer forwards what the engine delivers.
- ~~`src/verbatim/protocols/riva/server.py::RivaSpeechRecognitionServicer.StreamingRecognize` — on a
  cancelled RPC the servicer finalizes when its buffer happens to be empty and not when a partial tail
  is buffered, while the WebSocket surface never finalizes an aborted session at all. Pick one rule and
  apply it on both surfaces.~~ **Resolved: a departed client gets no final, on either surface.** A
  WebSocket close without an end message is an abort. A gRPC cancel reaches the library as a half-close
  followed by cancellation of the handler, so the reader ends the session and the cancelled handler
  aborts it; abort after end is legal and wins for whatever the ring still held.
- ~~`src/verbatim/protocols/riva/server.py::RivaServerConfig.max_concurrent_streams` — declared and
  documented but never read; admission is not wired into the Riva surface at all.~~ **Resolved:** the
  field is deleted and admission is the engine's, on both wires. A refusal carries a `retry-after-ms`
  trailer and no response is written.

## Results schema and verify

- ~~`bench/src/verbatim_bench/verify.py::_check_pacing_slip` — the summary's `pacing_slip_ms` percentiles
  can only be checked for internal ordering (`p50 <= p95 <= max`), never recomputed from the underlying
  data, because `sessions[]` in `benchmarks/schema/row.schema.json` carries no per-session slip samples
  and the writer emits none (the function's own docstring says so). Add a per-session slip-samples array
  to the schema and writer if the stronger check is wanted; otherwise leave the ordering-only check as
  the documented, permanent limit.~~
- ~~`bench/src/verbatim_bench/verify.py::_check_invariance` — a document with exactly one reported hash
  and `equal: null` currently raises no finding at all, since the three-way consistency check only fires
  once two or more hashes are present, so a row that never ran a second concurrency arm reports a clean
  invariance verdict. Confirm this silent-pass behaviour is really what the batch-invariance gate is
  meant to accept before relying on an `equal: null` row as passing evidence.~~

## Scheduler

- `src/verbatim/scheduler/tick.py::TickLoop.run_tick` — **this entry was wrong and is kept as a
  correction, and it has now been narrowed twice.** It first claimed the branch handling `frame is
  None` while a session is `DRAINING` was unreachable and should be deleted; it was reached both after
  a failed step and after `open_stream` failed on a frame that was both first and last. Closing failed
  sessions in the failing tick removed the first path, and closing a session whose open failed in its
  own tick removed the second, so **no tick-loop path produces the state today**. The branch stays as a
  safety net, because reaching it and deleting it leaks a slot and a registry entry, and the test that
  covers it now builds the state through the session directly rather than through a loop path that no
  longer exists. An entry that once said "delete this" now says "keep this and never let a path reach
  it", which is the opposite conclusion from the same code.
- ~~`src/verbatim/scheduler/tick.py::TickLoop.run_tick` — when `open_stream` raises on a first frame
  that is **not** also the last, the loop steps a synthetic abort on the next tick for a stream the
  pipeline never opened. Harmless while the abort suppresses emission, but not a contract the real NeMo
  adapter will honour. Decide it in the adapter brief: close the session outright, or open it lazily on
  first successful audio.~~ **Resolved: close outright.** A session whose open fails now closes in the
  same tick and no frame of it ever reaches the adapter. Lazy opening was rejected because the failure
  that reaches `open_stream` is NeMo's `create_state` refusing a per-stream option, and raising that
  inside a batch step would fail every session in the batch rather than the one that asked for it.

- `src/verbatim/config.py::EngineConfig` (`_DEFAULT_BUCKET`, `buckets: tuple[int, ...] | None`) — when
  neither `buckets` nor `calibrated_ceiling` is supplied, the config silently falls back to a
  hard-coded batch size of 8 (commented as a CPU-test convenience, not a measurement), and the public
  `buckets` field's declared type had to widen to allow `None` to make room for that fallback. Either
  make `calibrated_ceiling` (or `buckets`) mandatory, or give the fallback a name and a docstring making
  clear no server should ever run it unmeasured.
- `src/verbatim/engine.py::Engine.open_session` (`_sessions`, `_queues`) — the per-session dictionaries
  are populated on every open and never remove closed sessions, retaining roughly 188 KiB of preallocated
  ring buffer per closed session. Measured: 9.60 MB after fifty fully-drained sessions. Resolve before
  TASK-007 puts a long-lived transport on the engine.
- `src/verbatim/engine.py::EngineSession.results` — the per-session result queue has no maximum size and
  no overflow policy. Resolve before TASK-007 puts a long-lived transport on the engine.

## Cross-cutting

- `src/verbatim/protocols/ws/frames.py` and `src/verbatim/protocols/riva/mapping.py` — the two surfaces
  report the same audio duration to within about `5e-7` seconds, not exactly: the Riva wire field
  `audio_processed` is a protobuf `float` (32-bit) while the WebSocket `audio_s` is a Python double
  rounded to six decimal places before serialising. The underlying clock is identical, so this is a
  serialisation artefact rather than a disagreement, but any cross-surface comparison should assert a
  tolerance no tighter than `1e-6` and no published claim should imply the two are bit-identical.

- `src/verbatim/pipelines/__init__.py`, `CONTRIBUTING.md` and `docs/third_party.md` — all three state
  that a test walks the AST of every module outside `src/verbatim/pipelines/` and fails the build if it
  imports `nemo` or `torch`; no such test exists in the repository, and CI only checks that
  `nemo-toolkit`/`torch` are not pip-installed at all, which would miss a stray import in the wrong
  module in any environment where those packages happen to be absent for an unrelated reason. Write the
  AST-walk test the docs already promise; as a side effect it also stops the project's own
  `grep -rn "import nemo\|import torch" src/` sanity check from permanently flagging the docstring line
  that describes the rule.
