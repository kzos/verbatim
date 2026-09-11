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
- **`bench/src/verbatim_bench/nullserver.py` (the tail partial) — re-opened 2026-09-10; it was struck
  through as resolved and the property it asks for is still not asserted.** The null floor emits a
  partial for a short final tail so its cadence matches the real server, but
  `test_short_tail_emits_a_final_partial_so_the_floor_matches_the_server` asserts that a final partial
  arrives, not that partial's own `audio_s` taken from the frame the floor sent. Mutating the server's
  stamp can still leave the suite green. Assert it directly. **Re-opening this is the point:** a
  strikethrough is a claim, and an entry struck through without a test that fails without the fix is the
  same defect as a guard that cannot fail, one level up.
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
  `DEADLINE_EXCEEDED` when the idle deadline reclaims a slot, and `UNAVAILABLE` when the server stops
  under a live session) and the close codes (1000 after a refusal, **1001 when the server goes away**,
  1009 for a message over `max_message_bytes`) are described only in the module docstring and the tests.
  A demo protocol that is still moving is a bad thing to freeze into a document, so this is recorded
  rather than written: give it a page under `docs/protocols/` once the day-21 row has been taken over
  this surface, since that row is what makes the frames load-bearing. Note that it is already frozen in
  practice by two implementations, because the harness client under `bench/` parses these frames and the
  null server emits them, so **the tests are the spec until the row lands, and whoever changes a frame
  changes the harness client in the same commit**.

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

- ~~**The frozen run-schema constant is read by nothing, and the code writes a different value.**
  Found 2026-09-10. `benchmarks/METHODOLOGY.md` and `bench/src/verbatim_bench/constants.py` both declare
  `SCHEMA_VERSION_FOR_RUN = "vb-results/2"`, and the freeze test confirms they agree. But
  `SCHEMA_VERSION_FOR_RUN` is referenced nowhere outside those two places: `results.py` stamps the
  literal `"vb-results/1"` when it builds a run document and the literal `"vb-results/2"` in a separate
  method, so which version a produced row claims depends on which method ran, not on the frozen
  constant. **The freeze guard cannot see this**, because it compares the document against
  `constants.py` and never checks that the code path uses the constant. That is the same defect as the
  fabricated-sentence hole one level down: the guard verifies the wrong pair. Fix by making `results.py`
  read the constant, then add the assertion that a produced row's `schema` field equals
  `constants.SCHEMA_VERSION_FOR_RUN`.~~ **Resolved 2026-09-11, before the first row exists.**
  `to_json_dict_v2` reads the constant and
  `test_a_produced_run_stamps_the_frozen_schema_version` asserts a produced document carries it. The
  guard was mutation-tested three ways: it stays quiet when a literal happens to agree with the
  constant, it **fails** when the constant moves while the code keeps a literal, and it passes once the
  code follows the constant. One correction to the entry as first written: `to_json_dict`'s
  `vb-results/1` is a deliberate intermediate that `to_json_dict_v2` builds on, not a wrong value, so
  the defect was never "the code writes v1" — it was that **the freeze bound two documents to each
  other and nothing to the code.**

- `tests/harness/test_methodology_freeze.py` — **the freeze guard does not catch a fabricated result
  sentence.** It enforces agreement between the frozen constants block and `constants.py`, and it
  reddens when a value on either side moves, which was verified on 2026-09-10 by changing one and
  watching `test_document_constants_equal_the_code_constants` go red. But appending a sentence claiming
  a measured throughput to `benchmarks/METHODOLOGY.md` still passes all twenty-four assertions. That is
  the exact failure mode found on 2026-09-09 and `docs/decisions/0001` wrongly recorded it as fixed. A
  guard for it has to be different in kind: the document may state definitions and thresholds, but a
  number describing this project's own measured behaviour belongs only in a row under `rows/`, so the
  check is that no unfrozen numeric claim about Verbatim's performance appears in the document at all.

- `README.md` — **the demo page at `/` does not exist.** The README listed it twice as a thing the
  WebSocket listener serves. Probed against a live server on 2026-09-11: `GET /` and `GET /index.html`
  are both 404, because `HealthReporter.ROUTES` is the four health paths and nothing else. The README is
  corrected; either the page gets written or the claim stays deleted.

- `src/verbatim/cli.py` — the ready banner's `websocket` line names only `/v1/stream`, so an operator
  cannot discover `/healthz`, `/readyz`, `/metrics` or `/admission` from the process that serves them.
  Small, and worth doing while the endpoints are new.

- `src/verbatim/scheduler/` — **a session's latency is a phase offset drawn at connect and never
  repaid.** The client's chunk cadence and the tick are both 160 ms and stay in lockstep, so a chunk
  completing at phase p waits one period minus p for its tick, every time, for the session's whole life.
  Measured on 2026-09-11 by stepping sixteen sessions' start delays across one period against a live
  A6000: a sawtooth of slope minus one and amplitude 160 ms, range 168.8 to 321.6 ms, within-session
  spread 2 to 5 ms over 85 chunks, and the wrap point moves when the sweep is repeated, which anchors it
  to the server's tick rather than to anything about the sessions. `probes/tick_phase_sweep.py`.
  **Against a 310 ms budget and a p95 criterion this fails at any concurrency, including one stream**,
  for clients arriving at arbitrary times: a uniform phase draw puts the p95 near 320 ms. Windowed load
  runs look better only because a replaced session inherits the phase of one that ended on a tick
  boundary. Fix: align a session's chunk boundary to the tick at admission, paying under one period once
  at connect instead of a fraction of a period on every chunk forever. Running the tick faster than the
  chunk is the same fix at the price of a step per tick.

- `bench/src/verbatim_bench/client.py` — **the pacing slip is scored against a deadline the sender does
  not follow.** The slip is measured against `t0 + i * frame_period + jitter`, jitter uniform in plus or
  minus `FRAME_JITTER_MS` = 10, while the sleep before each send targets the unjittered
  `t0 + i * frame_period`. So the reported slip is `max(0, ε − J)` with ε the true overshoot near 1 ms,
  which gives p50 about 1.1 ms, p99 about 10.8 and max about 12.8 regardless of anything the server
  does. Those are exactly the four gated arms measured on 2026-09-11, to the tenth of a millisecond, and
  they are identical across precisions because the jitter is seeded per session. The generator's true
  slip against the schedule it follows is about 1 ms at p50, inside the 5.0 ms tolerance.
  **This record previously said the generator misses its own tolerance and that no run is valid. That
  was wrong and is withdrawn.** The frozen document lists `FRAME_JITTER_MS` under the workload
  constants, so the jitter belongs on the wire: the sender should sleep to the jittered deadline, which
  makes both the workload and the metric honest. Until it does, a `pacing_slip` criterion would
  invalidate every run for a harness reason.

- `bench/src/verbatim_bench/client.py` — **the load client stops measuring at the first final, and a
  stream has several.** The server emits a `final` frame for every hypothesis with `is_final`, which is
  one per endpointed utterance, so a stream with an internal silence of the endpointing length produces
  several. The reader returns on the first one. Three consequences, all live:
  `final_text` holds one segment rather than the utterance, so a WER criterion computed from it would be
  wrong in a way no ordering check can see; `finals_received` cannot exceed 1, so a
  `sessions_without_final` count built on it is asking a question the client cannot answer; and every
  partial after the first final is never read, so those chunks get no matching watermark. That last is
  the likely source of the measured coverage loss: 4,798 chunks matched of 5,047 sent at bfloat16 on
  2026-09-11, 95.1 percent. **This is a harness defect, not a server integrity failure**, and the two
  must not be conflated in the rung's integrity criterion.
  The same defect bit a probe of mine the same day: reading one final made leading silence look as
  though it changed the transcript, when it had only moved where the server segmented.

- `bench/src/verbatim_bench/cli.py::_make_rung` — **the ladder never runs the window it reports.** The
  rung executor builds its `LoadSpec` without `window_s`, which in `run_load` takes the branch that runs
  each slot exactly once, and it overrides `ramp_s` to zero. `warm_up_s` reaches no measurement anywhere:
  every use outside `constants.py` stores, stamps, compares or serialises it. Measured on 2026-09-11
  against the live A6000 bfloat16 server, a six-stream rung is 15.1 s of wall clock and 106 latency
  samples whose p95 is the sixth-worst; with the window applied the same rung is 188.6 s and 4,798
  samples. Four repeats of one identical rung give p95 values of 289.8, 320.3, 302.6 and 301.8 ms against
  the 310 ms threshold, so its verdict is not repeatable. The `canonical_window` flag is
  `args.warm_up_s == WARM_UP_S and args.window_s == WINDOW_S`, true whenever nothing was overridden, so
  it certifies the arguments and not the run. Fix: pass the window through, decide what the warm-up does
  operationally or delete it, derive `canonical_window` from the executed load, and guard it with a test
  that reddens against today's `_make_rung`. Until then no ladder output is a capacity.

- `bench/src/verbatim_bench/cli.py::_make_rung` — **the rung evaluates one of the four criteria the
  methodology defines and reports the other three as passing.** `benchmarks/METHODOLOGY.md` section 56
  requires latency, WER within 0.1 absolute of the batch-1 reference, integrity (no stream refused,
  dropped, back-pressured into audio loss, or ended without a final, with the load generator under half
  its pinned CPU budget) and zero GPU throttle events. The code evaluates latency, counts refusals, and
  writes `wer_vs_batch1=None`, `sessions_dropped=0` and `sessions_without_final=0` as literals.
  `Criterion.WER`, `INTEGRITY_DROPPED`, `INTEGRITY_NO_FINAL` and `THERMAL` are defined and unreachable.
  Every rung also carries `valid=True, invalid_reason=None` unconditionally, so all eight
  `InvalidReason` values and the eight validity constants that name them are dead.
  Three of the missing checks need no new collection: `SessionResult` already carries
  `finals_received`, `pacing_slip_ms` and `final_text`/`reference_text`.
  **The first fix is not any of these checks. It is that a rung must not report `passed=True` for a
  criterion it did not evaluate**, which turns four hardcoded passes into an honest invalid. Every
  ladder output this project has produced would come back invalid under that rule, which is correct.

- `bench/src/verbatim_bench/cli.py::_make_rung` — the ladder reports the **secondary** latency metric
  as though it were the primary. `LATENCY_PRIMARY = "word_emission"`, with `chunk_watermark` as the
  secondary and the fallback permitted only when the primary is unavailable. The rung computes p95 from
  `partial_ms`, which is the watermark metric, and never passes `words=True`, so the primary is
  unavailable by construction rather than by circumstance and nothing in the output records the
  substitution. Related: `SESSION_PROFILE = "m180"` describes 180-second sessions and the corpus is
  LibriSpeech utterances of a few seconds each.

- `bench/src/verbatim_bench/` — the warm-up is a convergence protocol, not a duration, and it does not
  exist. `WARM_UP_READING_S = 30`, `WARM_UP_CONVERGENCE = 0.10` and `WARM_UP_CAP_S = 120`: two
  consecutive 30-second readings within 10 percent open the window, and failure to converge by the cap
  fails the rung as `unstable`. `Criterion.UNSTABLE` is defined and unreachable. The window also has to
  *open* at a wall-clock moment rather than only *end* at one, and `run_load` has no notion of that.


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

- ~~`src/verbatim/config.py::EngineConfig` (`_DEFAULT_BUCKET`) — when neither `buckets` nor
  `calibrated_ceiling` is supplied, the config silently falls back to a hard-coded batch size of 8, a
  CPU-test convenience and not a measurement.~~ **Resolved:** `_DEFAULT_BUCKET` is deleted and a config
  with neither is refused, so a number nobody measured can no longer reach a published row through a
  default. `serve` requires either `--ceiling N` from a measured row or an explicitly named uncalibrated
  `--bucket N`. Two neighbouring rules landed with it: a `pad_pool` between 1 and `max(buckets) - 1` is
  refused as the invariance break it is, since the steady batch cannot then fill its own shape, and the
  edge pad rows are a named term of `num_slots` rather than an accident of `edge_batch + drain_margin`.
- ~~`src/verbatim/engine.py::Engine.open_session` (`_sessions`, `_queues`) — the per-session dictionaries
  are populated on every open and never remove closed sessions, retaining roughly 188 KiB of preallocated
  ring buffer per closed session. Measured: 9.60 MB after fifty fully-drained sessions.~~ **Resolved:**
  `_wake` drops both entries on the terminal row or on an error and `stop()` clears them, while the
  handle keeps its own queue and ring so `results()` still drains what was queued. Covered by
  `tests/protocol/test_engine_session.py::test_finished_sessions_are_dropped_from_the_engine`, which
  asserts both dictionaries are empty after five sessions finish. It stayed open here after the fix
  landed, which is the failure mode this file exists to prevent, so: **an entry is not resolved until
  someone has checked the code and named the test.**
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
