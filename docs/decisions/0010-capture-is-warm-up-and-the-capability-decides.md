# DR-0010 — CUDA-graph capture is a warm-up step, and a capability decides whether it happens

**Date:** 2026-09-13
**Status:** accepted

## What the scope asks for and what was actually here

Scope item 1 says the scheduler assembles fixed bucket sizes "whose CUDA graphs are captured and
retained". Before this change nothing in Verbatim captured anything. `graph_budget.py` counted the
keys a configuration would need and `buckets.py` kept the steady batch at a constant shape, which is
the precondition for a graph but not a graph. `serve` passed `asr.use_cuda_graphs` to NeMo's builder
and then left the whole matter to NeMo: whichever shapes happened to arrive first got captured, on
whichever tick they arrived, until `max_graphs` filled and every later shape ran eager for the life
of the process. Nothing observed which of those two things had happened.

## The three decisions

**1. Capture is warm-up at the shape, taken before the boundary grid starts.**
`CaptureController.warmup` steps an all-pad batch at each configured bucket, at construction of the
`TickLoop`, and the clock that fixes the boundary grid is read after it. There is no NeMo API being
called here to "capture a graph": NeMo captures a shape when it has seen enough of it, so the only
honest way to choose which shapes get the budget is to show it those shapes first, deliberately,
outside a tick. Each shape is stepped `GRAPH_WARMUP_STEPS = 3` times, matching upstream's own
`CudaGraphsStreamingEncoderStep(warmup_steps=3)` as read in `docs/ISSUES.md`: one step per shape
would have left the capture landing on live tick three. The rejected alternative was to let the
first live ticks do the capturing — which is what the code did before, and it puts a capture inside
a tick that has a deadline, on a shape chosen by whoever happened to connect.

**2. The graph path is selected from a capability the adapter reports, and an absent one is
refused.** `PipelineAdapter.graph_capability()` returns one of exactly three states: requested and
available, not requested (with the reason — `--eager`, or an adapter with no graph path at all), or
requested and absent. The third raises `GraphPathUnavailable` naming the reason, at construction,
before any step. This is [DR-0002](0002-the-graph-path-is-not-in-a-released-wheel.md) enforced one
layer lower: `serve` already refuses, and now so does the scheduler, so an adapter reached any other
way cannot quietly run the steady batch eager. The rejected alternative was a boolean flag on
`EngineConfig` threaded from the CLI: it would have made the eager-by-choice case and the
runtime-has-none case indistinguishable at the point where they matter, and those two must never
collapse into one.

**3. `transcribe_step` carries the intent, as a required keyword.** `graph=True` is only ever passed
for a steady batch whose shape was captured; the edge batch is `graph=False` unconditionally. The
keyword has no default: a default would let an adapter run the steady batch eager with nobody told,
which is the failure mode this whole path exists to prevent. The flag is Verbatim's intent and a
verification hook, not a per-call switch inside NeMo — there is no such switch that we can see from
here, and inventing one would be a fiction, since the graphed step is on no wheel we have.

## The edge shape is named, reserved and never captured

Finals are peeled into an eager side-batch and that shape is never warmed. Upstream agrees for its
own reasons: per the reading in `docs/ISSUES.md` (issue 3), NeMo's `_can_use_graphs()` never
captures a `keep_all_outputs=True` call at all. So the peel is not Verbatim declining a graph it
could have had — what the peel buys is the *steady* key, because a final left in the steady batch
shrinks the non-final sub-batch by a row and that is a different key every tick.

`graph_budget.graph_keys_required` is left exactly as it was, reserving one key alongside the
buckets. Reserving it per chunk mode rather than once was considered and rejected here: the chunk
mode really is part of NeMo's key, so the per-mode count would be the right arithmetic *if* the edge
shape could be captured, but by the reading above it cannot be, and no server this repository runs
today has more than one chunk mode. Tightening a budget on a shape upstream refuses to capture would
have cost bucket slots for nothing, and it is a separate decision from this one.

What `CapturePlan` adds is the key itself, as a named value: `edge_key` exists so that "the edge
shape is not in the captured set" is an assertion about a specific shape rather than about an
absence.

First steps are still not peeled, and that stays deliberate: NeMo keys on the per-batch pre-encode
drop, which a first frame does not change, so a first frame is safe inside the steady batch. A test
asserts a tick carrying a first frame replays the same key and captures nothing new.

## What is not tested here, and cannot be

The graphed encoder step is absent from this machine — `verbatim doctor` on this box reports
`graphed step ABSENT: serve refuses the graph path`, and this venv has neither NeMo nor torch. **No
test in this change runs a real CUDA graph, and none claims to.** What the tests drive is the
decision and the requests, through the same seam the NeMo adapter uses, with the capability faked in
all three states. The default arm of the new suite is the *graphed* one, and it asserts that capture
and replay were requested; a suite whose only arm was eager would pass on this box for the wrong
reason, which is the trap this project keeps finding.

What would retire this record: a run of these paths against a NeMo that has PR #15863, on the B300,
confirming that the shapes warm-up captures are the shapes the process retains — which is a
statement about NeMo's own key that this repository currently takes from upstream's source and not
from a measurement of its own.
