# Security policy

## Scope

Verbatim is a network service that accepts audio from untrusted clients over WebSocket and gRPC and runs
it through a speech recognition pipeline on a shared GPU. The parts of that worth reporting are:

- anything that lets one session read, alter or influence another session's audio, transcript or timing;
- anything that lets a client exhaust the process, the GPU or the admission controller beyond its
  declared ceiling, or bypass admission control altogether;
- memory-safety or deserialisation faults reachable from a client frame, a protocol field, or a
  `RecognitionConfig` supplied over the Riva subset;
- anything that causes audio or transcripts to be written, logged or retained where the design says they
  are not.

The last one matters more than it looks. This server is intended for settings where the audio is a
patient, a party to a case, or an account holder.

## Reporting

Open a **private security advisory** through this repository's Security tab. That routes to the
maintainer without disclosing the report. Please do not open a public issue for anything in the list
above.

Include what you need to reproduce it: the protocol, the frame or config that triggers it, the checkpoint
if it matters, and whether it needs more than one concurrent session. A reproducer that needs two sessions
is more interesting here than one that needs a thousand.

## What to expect

This is a personal project with one maintainer and no paid support. There is no service-level agreement
and no bounty. What there is: an acknowledgement within a week, an assessment of whether it reproduces,
and a public advisory when a fix lands or when I decide not to fix it. If a report goes unanswered for a
month, treat that as an invitation to disclose it publicly.

## Not in scope

Findings against the exploratory probe scripts under `probes/`, which are research code that runs
locally, and against the benchmark harness, which drives a server rather than serving anything.
Denial-of-service by simply sending more audio than the declared ceiling is a documented property of the
admission controller, not a vulnerability; a way to exceed the ceiling *without* being admitted is.
