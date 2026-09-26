#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# step1_gate_runbook.sh -- Verbatim's batch-invariance gate (the server test) on
# nvidia/nemotron-speech-streaming-en-0.6b, on card 3: fixed-shape padding against its
# ragged control arm (DR-0014), with the max level churned and at constant occupancy.
#
# WRITTEN, NOT RUN ON A GPU. Its checks have run only on the CPU, over `--pipeline fake`,
# with nvidia-smi and the HF cache stubbed (scripts/test_step1_gate_runbook.py). Every
# `verbatim serve` and `verbatim-bench invariance` flag carries the file:line where it was
# read at HEAD 854d4b9. src/verbatim/serve.py and src/verbatim/protocols/health.py had
# uncommitted work in progress when this was written, so those two are cited by name. The
# run refuses a dirty src/ or bench/src/ and records the HEAD it ran. Run it inside tmux or
# screen: if the terminal drops, the run stops.
#
# EAGER ONLY. Every arm passes --eager. Without it `verbatim serve` on this stack takes the
# graph path (src/verbatim/cli.py:345-353; the installed NeMo carries both halves of PR
# #15863), so nothing here says anything about the graph path. summary.json says so too.
#
# THE ENVIRONMENT A PRODUCTION RUN NEEDS. These five, each set, none with a default; the
# runbook refuses to start (exit 2) while any of them is unset or empty, in every mode but
# --help. The step-1 run on card 3 is exactly:
#   RUNBOOK_VENV=<the Python environment the server runs in> \
#   RUNBOOK_CORPUS_ROOT=<the local root holding librispeech-test-other-256/> \
#   RUNBOOK_MEASUREMENT_DIR=<the directory its run directory is written under> \
#   RUNBOOK_BUCKET=128 RUNBOOK_MAX_LEVEL=128 \
#   scripts/step1_gate_runbook.sh
# with no other RUNBOOK_* variable set: the test-mode overrides are refused in production,
# and the operator limits (RUNBOOK_WS_PORT, RUNBOOK_READY_TIMEOUT_S, RUNBOOK_GATE_TIMEOUT_S,
# RUNBOOK_CARD_IDLE_WAIT_S, RUNBOOK_SMOKE_SOLO_S) keep their defaults, which change nothing
# that is measured. What each of the five is: Locations and Parameters, below.
#
# Usage (the five variables above are required):
#   scripts/step1_gate_runbook.sh [ARM...]           the arms named, in order; all four if none
#   scripts/step1_gate_runbook.sh --dry-run [ARM...] each arm's env and argv as one JSON line;
#                                                    checks nothing, starts nothing
#   scripts/step1_gate_runbook.sh --derive-spec fixed|ragged
#                                    what the server's own code builds from that arm's flags:
#                                    the NeMo spec (matmul precision included), the banner and
#                                    the tick budget. CPU only, the model build stubbed
#   scripts/step1_gate_runbook.sh --check-banner fixed|ragged LOG
#                                    the banner check this runbook runs on a live server log,
#                                    run on a saved one
#   scripts/step1_gate_runbook.sh --facts-expect
#                                    what every arm's /readyz and /admission must report
#                                    (contract C5 for "observed", C7 for "code"), as the
#                                    JSON the facts check compares against; starts nothing
#   scripts/step1_gate_runbook.sh --help
#                                    this header; the only use that needs no environment
#
# Arms:
#   fixed-churn   --padding fixed,  levels 1 / 32a / 32b / MAX_LEVEL, max churned, 20 s period
#   ragged-churn  --padding ragged, the same levels and churn: fixed-churn's control
#   fixed-const   --padding fixed,  levels 1 / 32a / 32b / MAX_LEVEL, constant occupancy
#   ragged-const  --padding ragged, the same: fixed-const's control
#
# Parameters (environment, required, no defaults):
#   RUNBOOK_BUCKET     --bucket, the fixed batch shape. Fixed padding steps all BUCKET rows on
#                      every tick, level 1 included, so the bucket sets the tick cost at every
#                      level. At least 32 (the gate's 32a/32b levels hold 32 in flight, and
#                      the server refuses a session once the live count fills the largest
#                      bucket, admission.py:200-204).
#   RUNBOOK_MAX_LEVEL  --max, the gate's max level; at most RUNBOOK_BUCKET and at most the
#                      corpus's 256 utterances.
#   The step-1 run uses RUNBOOK_BUCKET=128 RUNBOOK_MAX_LEVEL=128. The bucket comes from the
#   step profile of this checkpoint on card 3 (profile-nemotron-160-eager.json in the
#   measurement directory: A6000, NeMo cf724ac, bfloat16, 160 ms chunks, eager, 40 steps
#   with 8 warm-up steps discarded, real speech): median step 30.5 ms at batch 32, 65.6 ms at
#   128 and 77.0 ms at 160, against the server's tick budget of 112 ms at 160 ms. Those are
#   medians only. Whether this card keeps the budget at the p95 the admission controller
#   counts is what the smoke below decides before every gate; the profile does not decide
#   it. The max level is the bucket itself, the most streams the fixed shape holds, so the
#   top level fills the batch it pads to. That leaves no headroom: if the server still
#   counts a finished session live when the gate opens the next one, the new one is refused
#   (admission.py:200-204) and the monitor stops the gate, exit 2. Whether that happens has
#   not been observed. The published B300 churn record ran bucket 128 and max 42 with the
#   older checkpoint. The one capacity search on this card
#   (rows/exploratory/ladder-a6000-fixed-dr0016-2026-09-15.json: the older checkpoint,
#   fixed, bfloat16, eager, 160 ms, gpu_index 3; bucket not recorded) ended at S = 17 on its
#   latency criterion.
#
# Locations (environment, required, no defaults; nothing machine-specific is written here):
#   RUNBOOK_VENV             the Python environment with `verbatim` and `verbatim-bench`
#                            installed (its bin/python, bin/verbatim, bin/verbatim-bench). The
#                            code is this repository's whatever that environment has installed:
#                            every process gets PYTHONPATH=<repo>/src:<repo>/bench/src,
#                            preflight refuses to run unless both packages import from there,
#                            and each arm refuses a server whose /readyz does not say it
#                            imported verbatim from <repo>/src/verbatim (contract C7, below).
#   RUNBOOK_CORPUS_ROOT      the local directory that stands where the published runs had
#                            /opt/verbatim/corpus/. The manifest is
#                            <root>/librispeech-test-other-256/librispeech-test-other-256.jsonl.
#   RUNBOOK_MEASUREMENT_DIR  where run directories are written. Test mode may not write under it.
# The repository is the one this file is in (git rev-parse --show-toplevel); the runbook
# refuses to run a copy of itself from anywhere else.
#
# Before each gate: a smoke on the same server (about SMOKE_SOLO_S of audio as one session,
# then max(32, MAX_LEVEL) sessions at once) with /admission read throughout. The run stops,
# exit 2, before the long gate if p95_tick_ms reaches the server's tick budget (chunk x
# utilisation target, config.py:225-228, derived from the server's own code), if any
# session was refused, or if degradation_level reached 1. During the gate a monitor reads
# /admission, and asks nvidia-smi what holds card 3, sleeping MONITOR_EVERY_S between polls
# (the spacing asked for: each poll also waits up to ADMISSION_READ_TIMEOUT_S and
# CARD_QUERY_TIMEOUT_S; monitor-summary.json records the spacing the polls actually had and
# the longest each read took, measured by the monitor around each read). It stops the gate
# as soon as refused_total rises or degradation_level reaches
# 1: a refused session leaves its level incomplete, so the verdict is already lost. What
# its polls saw, and what they failed to see, is judged after the gate (monitor_summary),
# against the gate's own start and end times, and each of these takes the arm's verdict
# away: a process other than the server at even one poll, or on the reading after the
# gate; a poll that did not show the server; a poll nvidia-smi failed, since what held the
# card then was not observed; no poll at all; no /admission reading at all; any stretch of
# the gate longer than 3 x MONITOR_EVERY_S plus both timeouts (125 s in production) with no
# poll in it, counting the stretch from the gate's start to the first poll and from the
# last poll to the gate's end, since nothing watched the card or the tick budget then; a
# monitor that did not exit 0 by itself once the gate ended (killed, crashed, or still
# running that long after it); a monitor whose first line does not say it parsed the sleep
# and the two timeouts the runbook asked for. A process that came and went between
# two polls is not seen. An /admission read that fails now and then is counted, not acted
# on: refused_total is cumulative and is read again after the gate. That reading after the
# gate is the runbook's own, and each of these takes the verdict away as well: the server
# no longer running, no /admission answer, refused_total above its value when the gate
# started, degradation_level 1 or more, nvidia-smi unable to list card 3, or not showing
# the server on it. The smoke also warms the server, identically for every arm.
#
# What the server is (contract C5): /readyz's "observed" object is read off the built model
# and decoder, not off the flags. Before each gate the run requires
# observed.att_context_size == [70, ATT_RIGHT], observed.decoder_graphs false,
# observed.decoder_step_confidence false and word_confidence "off", and refuses null (not
# observed) for any of them. Only the CPU fake, which has no encoder and no decoder, may
# report null, and only in test mode. The banner check runs as well.
#
# Which code the server runs (contract C7): /readyz's "code" object names the directory the
# server process imported its verbatim package from (and verbatim_bench's, or null when it
# cannot import it), resolved, as that process sees it. Before each gate the run requires
# code.verbatim_path == <repo>/src/verbatim and code.bench_path null or
# <repo>/bench/src/verbatim_bench (both resolved), in test mode too: the fake runs the same
# server code. A server that reports no "code" is refused. This is what sees a server that
# kept this repository's PYTHONPATH (checked from /proc) but imports another checkout's
# package anyway, through a .pth file, an editable install's finder or a sitecustomize.
#
# Code identity: preflight records the git HEAD, refuses uncommitted changes under src/ or
# bench/src/, and hashes this runbook. The same is checked again after each gate and before
# each server start: other sessions work in this repository, and arms start hours apart. A
# change seen after a gate takes that arm's verdict away; a change seen before a server
# starts stops the run there (exit 2), so no arm runs code other than what preflight named.
#
# Exit codes, in the order summary() decides them:
#   1  a fixed arm diverged within itself: batch dependence under fixed padding. Decided
#      first, whatever the other arms did, and also when a later arm stopped the run: a
#      divergence in a record that passed every record check needs no control. An
#      operational problem beside it (a missing reading, another process on the card) is
#      named in the reading and does not take the 1 away. A problem that says the record is
#      not this configuration's does, and makes it 2: a record or finals check failing, the
#      level-1 WER over its sanity bound, the code or this runbook changing during the arm,
#      NeMo's look-ahead warning
#   2  the run stopped before it finished (a refusal or failure in some arm; the arms that
#      finished are summarised anyway), or the arms in one run directory (continued with
#      RUNBOOK_RUN_ID) ran at different settings or did not record them: bucket, max level,
#      runbook sha256, git HEAD, package versions
#   4  a server restart changed the bytes: fixed-churn and fixed-const differ at a level
#      they share, or ragged-churn and ragged-const differ at level 1 (one stream in flight)
#   2  no verdict somewhere (refused, errored, vacuous, record or finals missing or
#      malformed, level-1 WER over its bound), or an operational problem in an arm (card
#      shared, a reading missing, the monitor stopped the gate, ...)
#   3  nothing failed, but the pass is not controlled: an arm did not run, or a fixed arm's
#      comparison against level 1 (32a, 32b or max) has no divergence at the same comparison
#      in the ragged arm of the same occupancy. The reading names each such comparison.
#      Per DR-0014 such a pass is not evidence that the padding holds the property
#   0  both fixed arms invariant; each of their six comparisons against level 1 controlled
#      by its ragged twin diverging there; the fixed arms byte-identical across the restart
#      at all four levels and the ragged arms at level 1. Needs all four arms
# Anything unexpected is 2, never 1: an ERR trap maps a command failing where nothing
# expected it to, and the helper maps an exception it raises.
#
# Why the verdict is read from the record and not from `verbatim-bench invariance`'s exit
# code: main() returns 1 for an argparse usage error (bench/src/verbatim_bench/cli.py:1009)
# and 1 for "divergent" (invariance.py:93), so exit 1 alone cannot tell them apart.
#
# Wall clock, from the records (all B300, the older checkpoint; nothing measured on an
# A6000 or with this checkpoint). Level 1 runs 256 clips one at a time at real-time pace,
# and the corpus holds 1684.29 s of audio (corpus YAML total_audio_s), so it cannot finish
# sooner than that on any GPU:
#   fixed,  constant, max 38  1933.7 s   invariance-control-arm-b300-2026-09-14.json raw.fixed
#   ragged, constant, max 38  1936.4 s   same record, raw.ragged
#   fixed,  churned 42/20 s   1965.8 s   invariance-decgraph-b300-2026-09-16.json (with
#                                        --decoder-graphs, which this runbook does not pass)
#   level max alone, churned: 87.5 s fixed, 92.5 s ragged (invariance-churn-b300-2026-09-14.json)
#   at the chosen bucket and max on this card, server start and the smoke: <placeholder>
# END OF HELP

set -Eeuo pipefail

DIE_MSG=""
die() { DIE_MSG="$*"; printf '[runbook] FATAL: %s\n' "$*" >&2; exit 2; }
log() { printf '[runbook] %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
now() { date +%s.%N; }
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# A command failing where nothing above expected it to is an operational failure: exit 2,
# never the 1 that means a fixed arm diverged (set -e alone would pass its status on).
on_error() { die "unexpected failure (exit $1) at line $2: $3"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

# ---------------------------------------------------------------------------------------
# What is being run
# ---------------------------------------------------------------------------------------
readonly SELF="$(realpath "${BASH_SOURCE[0]}")"
if [[ $# == 1 && ( "$1" == -h || "$1" == --help ) ]]; then
    sed -n '2,/^# END OF HELP$/p' "$SELF" | sed '$d' || true  # a reader that stops early is fine
    exit 0
fi
REPO=$(git -C "$(dirname "$SELF")" rev-parse --show-toplevel 2>/dev/null || true)
[[ -n "$REPO" ]] \
    || die "$SELF is not inside a git repository; the runbook runs from the repository whose code it tests"
[[ "$REPO/scripts/$(basename "$SELF")" == "$SELF" ]] \
    || die "$SELF is not scripts/$(basename "$SELF") of the repository it is in ($REPO)"
readonly REPO
# The code under test is this repository's, for every process the run starts: the helper,
# the spec derivation, the server and the gate. Preflight checks where it imports from.
export PYTHONPATH="$REPO/src:$REPO/bench/src"
# Where the server must say it imported its code from (contract C7), resolved as the server
# resolves os.path.dirname(verbatim.__file__). verbatim_bench's may also be null there.
readonly SERVER_VERBATIM_PATH="$(realpath -m "$REPO/src/verbatim")"
readonly SERVER_BENCH_PATH="$(realpath -m "$REPO/bench/src/verbatim_bench")"

# The environment a run needs (the header's first block): each set, none with a default.
# Defaults once stood here (.venv at the repository root, /opt/verbatim/corpus,
# probe-output/step1 in this checkout); where the step-1 run is made, the first two do not
# exist and the third is not where its measurements go. Nothing is assumed now.
readonly REQUIRED_ENV=(RUNBOOK_VENV RUNBOOK_CORPUS_ROOT RUNBOOK_MEASUREMENT_DIR RUNBOOK_BUCKET RUNBOOK_MAX_LEVEL)
[[ -z "${RUNBOOK_MAX:-}" ]] || die "RUNBOOK_MAX is now RUNBOOK_MAX_LEVEL; unset RUNBOOK_MAX"
unset_env=()
for var in "${REQUIRED_ENV[@]}"; do
    [[ -n "${!var:-}" ]] || unset_env+=("$var")
done
(( ${#unset_env[@]} == 0 )) \
    || die "set ${unset_env[*]}: a run needs all of ${REQUIRED_ENV[*]} and none has a default (the step-1 run: the header's first block, RUNBOOK_BUCKET=128 RUNBOOK_MAX_LEVEL=128)"

readonly VENV="$RUNBOOK_VENV"
readonly PY="$VENV/bin/python"
readonly VERBATIM="$VENV/bin/verbatim"        # verbatim.cli:console_main, pyproject.toml:55
readonly BENCH="$VENV/bin/verbatim-bench"     # verbatim_bench.cli:main, bench/pyproject.toml:36

readonly MODEL=nvidia/nemotron-speech-streaming-en-0.6b
readonly MODEL_REVISION=ebe59e5a817142986528bbbee5dba8db7b38ed50
readonly MODEL_FILENAME=nemotron-speech-streaming-en-0.6b.nemo

readonly GPU_INDEX=3
# The card the step-1 stock probe ran on with CUDA_VISIBLE_DEVICES=3, as nvidia-smi -i 3
# reported it: probe-output/step1/nemotron-bf16-1120.json, "driver_and_uuid".
readonly EXPECTED_GPU_UUID=GPU-b43f9262-f250-444a-bfbf-461dd3500f1e

readonly CHUNK_MS=160
readonly CHUNK="${CHUNK_MS}ms"
readonly DTYPE=bfloat16
readonly ATT_LEFT=70            # also KNOWN_LEFT_CONTEXT for this checkpoint, nemo_runtime.py:59
readonly ATT_RIGHT=$(( CHUNK_MS / 80 - 1 ))   # nemo_runtime.py:76-80
readonly HOST=127.0.0.1         # the published records' endpoint was ws://127.0.0.1:8765/v1/stream
readonly GRPC_PORT=0            # ephemeral: the gate uses the WebSocket wire only
readonly IDLE_TIMEOUT_S=30.0    # the flag's default, cli.py:173, passed so the run pins it
readonly RING_SECONDS=3.0       # the flag's default, cli.py:180
readonly EOU_MS=800             # the flag's default, cli.py:187
readonly LANG_CODE=en-US        # the default of both --language-code (cli.py:249) and --lang (bench cli.py:245)
readonly SEED=20260914          # the gate's default, invariance.py:89
readonly UTTERANCES=256         # invariance-churn-b300-2026-09-14.json shared.streams_per_level
readonly MIN_BUCKET=32          # the gate's 32a/32b levels, invariance.py default_levels

# A sanity bound on level 1's corpus WER against the manifest's reference text, under the
# gate's own pinned normaliser (bench/src/verbatim_bench/wer.py normalise). It is not a
# measurement and not a quality gate: it is there so a server that emits wrong but
# non-empty text cannot pass as invariant, which the digests alone cannot see.
readonly LEVEL1_WER_MAX=0.5

# NeMo's own warning when the attention context asked for is not one the checkpoint lists
# (nemo/collections/asr/modules/conformer_encoder.py:1000-1004, logged at level 30, the
# level the server builds NeMo with, nemo_runtime.py:166). It must not appear.
readonly NEMO_LOOKAHEAD_WARNING="is not among the list of the supported look-aheads"

CHURN_S=20                      # invariance-churn-b300-2026-09-14.json shared.churn_period_s

# The published runs read /opt/verbatim/corpus/librispeech-test-other-256/
# librispeech-test-other-256.jsonl (corpus.path in every invariance record), corpus_id
# sha256:6a142a96... A local copy holds the same 256 entries in the same order with the same
# per-clip PCM sha256 and differs ONLY in the audio path prefix: with RUNBOOK_CORPUS_ROOT/
# rewritten to /opt/verbatim/corpus/ it hashes to exactly the published id. Preflight checks
# that on every production run and records the local id it observed (on the measurement box
# that is sha256:d1bae7c7..., the corpus_id the A6000 ladder records carry). The record
# check then requires the gate's record to carry that observed id.
CORPUS_ROOT="${RUNBOOK_CORPUS_ROOT%/}"
MANIFEST="$CORPUS_ROOT/librispeech-test-other-256/librispeech-test-other-256.jsonl"
readonly LOCAL_PREFIX="$CORPUS_ROOT/"
readonly PUBLISHED_CORPUS_ID=sha256:6a142a960379d48ed31d5ec8c3bc07dbdfcd34ef2b6db041a888b07bdc0003be
readonly PUBLISHED_PREFIX=/opt/verbatim/corpus/

readonly MEASUREMENT_OUT_DIR="$(realpath -m "$RUNBOOK_MEASUREMENT_DIR")"
OUT_DIR=$MEASUREMENT_OUT_DIR
PIPELINE=cache_aware_rnnt
readonly PRODUCTION_PIPELINE=cache_aware_rnnt
CODE_GIT_DIR=$REPO

# Operator-chosen limits, not measurements. None of them changes what is measured.
WS_PORT="${RUNBOOK_WS_PORT:-8765}"
READY_TIMEOUT_S="${RUNBOOK_READY_TIMEOUT_S:-1800}"
GATE_TIMEOUT_S="${RUNBOOK_GATE_TIMEOUT_S:-14400}"
CARD_IDLE_WAIT_S="${RUNBOOK_CARD_IDLE_WAIT_S:-60}"
# Longer than the 200-tick p95 window (admission.py WINDOW_TICKS; 32 s at 160 ms), so the
# p95 read at the end of the solo phase no longer holds the first ticks of a fresh server.
SMOKE_SOLO_S="${RUNBOOK_SMOKE_SOLO_S:-60}"
STOP_TIMEOUT_S=60
MONITOR_EVERY_S=30
# How long one monitor poll may wait on nvidia-smi and on /admission. With the spacing
# above they set the longest stretch of the gate the polls may leave unwatched:
# 3 x MONITOR_EVERY_S + both timeouts (monitor_summary; 125 s here).
CARD_QUERY_TIMEOUT_S=30
ADMISSION_READ_TIMEOUT_S=5

readonly ALL_ARMS=(fixed-churn ragged-churn fixed-const ragged-const)

# ---------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------
BUCKET="$RUNBOOK_BUCKET"
MAX_LEVEL="$RUNBOOK_MAX_LEVEL"
is_int "$BUCKET" && (( BUCKET >= 1 )) || die "RUNBOOK_BUCKET must be a positive integer, got '$BUCKET'"
is_int "$MAX_LEVEL" && (( MAX_LEVEL >= 1 )) || die "RUNBOOK_MAX_LEVEL must be a positive integer, got '$MAX_LEVEL'"
for var in CARD_IDLE_WAIT_S GATE_TIMEOUT_S READY_TIMEOUT_S WS_PORT; do
    is_int "${!var}" || die "RUNBOOK_$var must be a whole number, got '${!var}'"
done
[[ "$SMOKE_SOLO_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "RUNBOOK_SMOKE_SOLO_S must be a number of seconds"
[[ -x "$PY" ]] || die "no Python at $PY: set RUNBOOK_VENV to the environment the server runs in"

# ---------------------------------------------------------------------------------------
# Test mode: the same script over `--pipeline fake` on the CPU with a small corpus. Every
# check still runs, against whatever nvidia-smi is first on PATH (which must be a test
# stub), the HF cache HF_HUB_CACHE names, and the git repository RUNBOOK_TEST_GIT_DIR names.
# What test mode relaxes, and why each is covered anyway:
#   * the served pipeline is the fake, so the NeMo lines of the banner are not there; the
#     same check runs on the banner the server's own code renders (--check-banner). The
#     fake has no encoder and no decoder, so /readyz may report null for what it observes;
#     that is accepted from the fake only; the check, and the production expectation it
#     runs with (--facts-expect), are tested on a NeMo-shaped /readyz;
#   * the corpus is not the published one, so it is not compared with the published id; the
#     comparison is tested against the real manifest and in production mode;
#   * the parameter bounds (bucket >= 32, max <= bucket) are not enforced, so a test can
#     make the server refuse sessions; the bounds are tested in production mode.
# It never writes into the measurement directory, and its overrides are refused outside it.
# ---------------------------------------------------------------------------------------
TEST_FAKE="${RUNBOOK_TEST_FAKE:-0}"
SMOKE_BURST=$(( MAX_LEVEL > MIN_BUCKET ? MAX_LEVEL : MIN_BUCKET ))

if [[ "$TEST_FAKE" == 1 ]]; then
    OUT_DIR="${RUNBOOK_OUT_DIR:?test mode needs RUNBOOK_OUT_DIR}"
    case "$(realpath -m "$OUT_DIR")/" in
        "$MEASUREMENT_OUT_DIR"/*) die "test mode may not write under $MEASUREMENT_OUT_DIR" ;;
    esac
    MANIFEST="${RUNBOOK_MANIFEST:?test mode needs RUNBOOK_MANIFEST}"
    CHURN_S="${RUNBOOK_CHURN_S:-$CHURN_S}"
    CODE_GIT_DIR="${RUNBOOK_TEST_GIT_DIR:?test mode needs RUNBOOK_TEST_GIT_DIR, a git repository standing in for this one}"
    [[ -n "${HF_HUB_CACHE:-}" ]] || die "test mode needs HF_HUB_CACHE pointing at a stub cache"
    SMOKE_BURST="${RUNBOOK_TEST_SMOKE_BURST:-$SMOKE_BURST}"
    PIPELINE=fake
    MONITOR_EVERY_S=0.2
    # A stub nvidia-smi and a local /admission answer in milliseconds; short timeouts keep
    # the unwatched-stretch bound (3.6 s) well inside a rehearsal gate.
    CARD_QUERY_TIMEOUT_S=2
    ADMISSION_READ_TIMEOUT_S=1
elif [[ "$TEST_FAKE" != 0 ]]; then
    die "RUNBOOK_TEST_FAKE must be 0 or 1, got '$TEST_FAKE'"
else
    for var in RUNBOOK_OUT_DIR RUNBOOK_MANIFEST RUNBOOK_CHURN_S RUNBOOK_TEST_GIT_DIR RUNBOOK_TEST_SMOKE_BURST; do
        if [[ -n "${!var:-}" ]]; then
            die "$var is a test-mode override and would change the measurement; unset it (or set RUNBOOK_TEST_FAKE=1 for a CPU rehearsal)"
        fi
    done
    (( BUCKET >= MIN_BUCKET )) \
        || die "RUNBOOK_BUCKET=$BUCKET is below $MIN_BUCKET: the gate's 32a/32b levels hold $MIN_BUCKET streams in flight and the server would refuse the rest"
    (( MAX_LEVEL <= BUCKET )) \
        || die "RUNBOOK_MAX_LEVEL=$MAX_LEVEL exceeds RUNBOOK_BUCKET=$BUCKET: the server refuses a session once the live count fills the bucket"
    (( MAX_LEVEL <= UTTERANCES )) \
        || die "RUNBOOK_MAX_LEVEL=$MAX_LEVEL exceeds the corpus's $UTTERANCES utterances"
fi
readonly OUT_DIR MANIFEST MAX_LEVEL BUCKET CHURN_S PIPELINE WS_PORT READY_TIMEOUT_S GATE_TIMEOUT_S
readonly STOP_TIMEOUT_S CARD_IDLE_WAIT_S MONITOR_EVERY_S TEST_FAKE CODE_GIT_DIR SMOKE_BURST SMOKE_SOLO_S
readonly CARD_QUERY_TIMEOUT_S ADMISSION_READ_TIMEOUT_S

# ---------------------------------------------------------------------------------------
# The embedded helper: JSON, hashing, HTTP and the decisions. The standard library only,
# except the functions that say otherwise, which import this repository's own packages.
# None of it imports torch or NeMo. An exception in any command exits 2.
# ---------------------------------------------------------------------------------------
read -r -d '' HELPER <<'PYEOF' || true
import contextlib, hashlib, json, os, re, signal, subprocess, sys, time, urllib.error, urllib.request
from collections import Counter
from pathlib import Path

SLOTS = ["1", "32a", "32b", "max"]
PAIRS = [("32a", "1"), ("32b", "1"), ("max", "1"), ("32b", "32a")]
AGAINST_ONE = ["32a_vs_1", "32b_vs_1", "max_vs_1"]
ORDER = ["fixed-churn", "ragged-churn", "fixed-const", "ragged-const"]
TWIN = {"fixed-churn": "ragged-churn", "fixed-const": "ragged-const"}
FINALS_RECORD = "vb-invariance-finals/1"
#: /readyz "observed" (contract C5; src/verbatim/protocols/health.py OBSERVED_KEYS).
OBSERVED = ("att_context_size", "decoder_step_confidence", "decoder_graphs")
SCOPE = ("EAGER ONLY: every arm ran the encoder step eager (--eager). Without --eager, "
         "`verbatim serve` on this stack takes the graph path, which this run did not measure")
#: Environment names whose values are not written into a record.
SECRET = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_?KEY|PRIVATE|AUTH|COOKIE|(^|_)KEY($|_)", re.I)


def read_json_url(url, timeout=5.0):
    """The JSON body at `url`, or {"error": ...} when nothing usable answered."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def http(url, out):
    """Print the HTTP status (0 when nothing answered); write the body to `out`."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
    except Exception:
        print(0)
        return 0
    Path(out).write_bytes(body)
    print(status)
    return 0


def get(path, *keys):
    """Print the value at `keys` in the JSON file (strings bare, the rest as JSON); an
    empty line when the file or the key is missing."""
    try:
        doc = json.loads(Path(path).read_text())
    except Exception:
        print("")
        return 0
    for key in keys:
        doc = doc.get(key) if isinstance(doc, dict) else None
    print("" if doc is None else doc if isinstance(doc, str) else json.dumps(doc))
    return 0


def lines(path, *keys):
    """Each item of the JSON list at `keys` in the file, one per line (a line break inside
    an item becomes a space). Raises, so exit 2, when there is no list there: a summary
    that does not say what it found is not a summary that found nothing."""
    doc = json.loads(Path(path).read_text())
    for key in keys:
        doc = doc[key]
    if not isinstance(doc, list):
        raise TypeError(f"{'.'.join(keys)} in {path} is {type(doc).__name__}, not a list")
    for item in doc:
        print(" ".join(str(item).splitlines()))
    return 0


def code_origin(root):
    """Where verbatim and verbatim_bench import from in this process, and the installed
    versions. Exit 1 unless both import from `root`'s src/ and bench/src/."""
    import importlib.metadata as md
    import importlib.util

    root = root.rstrip("/")
    doc = {"repo": root, "pythonpath": os.environ.get("PYTHONPATH")}
    for mod, sub in (("verbatim", "/src/verbatim/"), ("verbatim_bench", "/bench/src/verbatim_bench/")):
        spec = importlib.util.find_spec(mod)
        doc[mod] = spec.origin if spec else None
        doc[mod + "_from_repo"] = bool(spec and spec.origin and spec.origin.startswith(root + sub))
    for dist in ("nemo_toolkit", "torch", "verbatim", "verbatim-bench", "huggingface_hub"):
        try:
            doc["version_" + dist] = md.version(dist)
        except md.PackageNotFoundError:
            doc["version_" + dist] = None
    print(json.dumps(doc, indent=2))
    return 0 if doc["verbatim_from_repo"] and doc["verbatim_bench_from_repo"] else 1


def corpus(manifest, local_prefix, published_prefix):
    raw = Path(manifest).read_bytes()
    text = raw.decode("utf-8")
    entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    print(json.dumps({
        "manifest": manifest,
        "local_id": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "published_form_id": "sha256:" + hashlib.sha256(
            text.replace(local_prefix, published_prefix).encode("utf-8")).hexdigest(),
        "utterances": len(entries),
        "total_duration_s": round(sum(float(e["duration"]) for e in entries), 4),
        "local_prefix_occurrences": text.count(local_prefix),
        "every_audio_path_under_local_prefix": all(
            str(e["audio_filepath"]).startswith(local_prefix) for e in entries),
    }))
    return 0


def corpus_check(corpus_json, published_id, utterances):
    """The manifest is the published corpus with only its path prefix moved. Exit 1 if not."""
    c = json.loads(Path(corpus_json).read_text())
    bad = []
    if c["published_form_id"] != published_id:
        bad.append(f"with the prefix rewritten it hashes to {c['published_form_id']}, not the published {published_id}")
    if c["utterances"] != int(utterances) or c["local_prefix_occurrences"] != c["utterances"]:
        bad.append(f"{c['utterances']} utterances, prefix seen {c['local_prefix_occurrences']} times")
    if not c["every_audio_path_under_local_prefix"]:
        bad.append("an audio path is outside the local prefix")
    for line in bad:
        print(line)
    return 1 if bad else 0


def facts(readyz, admission, expect):
    """What the server says it is, against what this arm asked for. Exit 1 on any mismatch.
    The settings keys echo the flags (the flags reached the server); "observed" is read off
    the built model and decoder (contract C5), and null there means not observed, which is
    accepted only from the fake, which has no encoder and no decoder. "code" is where the
    server process imported its packages from (contract C7): verbatim's must be this
    repository's src/verbatim, from every server, the fake's included; verbatim_bench's
    this repository's bench/src/verbatim_bench or null (the server need not import it)."""
    r = json.loads(Path(readyz).read_text())
    a = json.loads(Path(admission).read_text())
    exp = json.loads(expect)
    bad = []
    if r.get("ready") is not True:
        bad.append(f"/readyz ready={r.get('ready')!r} reason={r.get('reason')!r}")
    for key in ("model", "pipeline", "chunk_ms", "precision", "execution", "biasing", "word_confidence"):
        if r.get(key) != exp[key]:
            bad.append(f"/readyz {key}={r.get(key)!r}, expected {exp[key]!r}")
    observed = r.get("observed")
    null_ok = exp["pipeline"] == "fake" and r.get("pipeline") == "fake"
    if not isinstance(observed, dict):
        bad.append(f"/readyz observed={observed!r}: the server did not report what it built")
    else:
        for key in OBSERVED:
            got, want = observed.get(key), exp["observed"][key]
            if got is None and null_ok:
                continue
            if got != want:
                why = " (null: not observed)" if got is None else ""
                bad.append(f"/readyz observed.{key}={got!r}, expected {want!r}{why}")
    code, want_code = r.get("code"), exp["code"]
    if not isinstance(code, dict):
        bad.append(f"/readyz code={code!r}: the server did not report where it imported its code from")
    else:
        if code.get("verbatim_path") != want_code["verbatim_path"]:
            bad.append(f"/readyz code.verbatim_path={code.get('verbatim_path')!r}, expected "
                       f"{want_code['verbatim_path']!r}: the server does not run this repository's code")
        if code.get("bench_path") not in (None, want_code["bench_path"]):
            bad.append(f"/readyz code.bench_path={code.get('bench_path')!r}, expected "
                       f"{want_code['bench_path']!r} or null")
    if a.get("bucket") != exp["bucket"]:
        bad.append(f"/admission bucket={a.get('bucket')!r}, expected {exp['bucket']!r}")
    for line in bad:
        print(line)
    return 1 if bad else 0


def environ(pid, out):
    """The server process's own environment, read from /proc/PID/environ: everything it
    inherited, not only the overrides this runbook passed. Values of names that look like
    credentials are replaced. Prints the PYTHONPATH it saw."""
    source = f"/proc/{int(pid)}/environ"
    env = {}
    for item in Path(source).read_bytes().split(b"\0"):
        if item:
            key, _, value = item.decode("utf-8", "replace").partition("=")
            env[key] = "<redacted>" if SECRET.search(key) else value
    Path(out).write_text(json.dumps({"source": source, "environ": dict(sorted(env.items()))}, indent=2) + "\n")
    print(env.get("PYTHONPATH", ""))
    return 0


def derive_spec(*serve_argv):
    """What `verbatim serve` builds from this argv, by the server's own code, on the CPU:
    the NeMo pipeline spec it hands the model build (att_context, matmul precision, graph
    switches, dtype), the banner it prints, and the tick budget its admission control
    counts overruns against. It runs the server's entry point, verbatim.cli.main, through
    its injectable seams: the runtime probe reports the graph step present, as the
    installed NeMo does, so only --eager keeps the step eager; the model build is the
    repo's NeMo-shaped fake, so nothing is loaded; the server is a no-op, so no port is
    bound. This is the code's answer for this argv, not a reading from a running process."""
    import io

    from verbatim import cli
    from verbatim.pipelines.nemo_fake import FakeCacheAwarePipeline, boundary_for
    from verbatim.pipelines.nemo_runtime import RuntimeReport
    from verbatim.serve import engine_config

    specs, served = [], {}

    def build(spec):
        specs.append(spec)
        return boundary_for(FakeCacheAwarePipeline(spec.chunk.ms, num_slots=max(64, spec.num_slots)))

    async def no_server(settings, adapter, **kwargs):
        served.update(settings=settings, execution=kwargs.get("execution"))

    report = RuntimeReport(nemo_version="<stub>", torch_version="<stub>", cuda_available=True,
                           device_name=None, inference_package=True, graph_step=True)
    out, err = io.StringIO(), io.StringIO()
    hooks = cli.Hooks(inspect_runtime=lambda: report, build_boundary=build,
                      run_server=no_server, stdout=out, stderr=err)
    code = cli.main(list(serve_argv), hooks=hooks)
    if code != 0 or "settings" not in served:
        print(json.dumps({"error": f"verbatim serve exited {code}: {err.getvalue().strip()}"}))
        return 1
    spec = specs[0] if specs else None
    print(json.dumps({
        "derived_from": list(serve_argv),
        "how": derive_spec.__doc__.split(":")[0].strip(),
        "execution": served["execution"],
        "budget_ms": engine_config(served["settings"]).budget_ms,
        "spec": None if spec is None else {
            "model": spec.model,
            "att_context": list(spec.att_context),
            "matmul_precision": spec.matmul_precision,
            "compute_dtype": spec.compute_dtype,
            "use_cuda_graphs": spec.use_cuda_graphs,
            "use_cuda_graph_decoder": spec.use_cuda_graph_decoder,
            "enable_per_stream_biasing": spec.enable_per_stream_biasing,
            "word_confidence": getattr(spec, "word_confidence", None),
            "batch_size": spec.batch_size,
            "num_slots": spec.num_slots,
            "decoding": spec.decoding,
            "stop_history_eou_ms": spec.stop_history_eou_ms,
            "device_id": spec.device_id,
            "log_level": spec.log_level,
        },
        "banner": [line for line in out.getvalue().splitlines()
                   if line.startswith("[verbatim] ") and line != "[verbatim] stopped"],
    }, indent=2))
    return 0


def spec_check(path, expect):
    """The derived spec is the arm asked for. Exit 1 on any mismatch."""
    doc = json.loads(Path(path).read_text())
    exp = json.loads(expect)
    spec = doc.get("spec") or {}
    bad = []
    if not spec:
        bad.append("the server's code built no NeMo spec for this argv")
    for key in ("model", "att_context", "compute_dtype", "use_cuda_graphs",
                "use_cuda_graph_decoder", "enable_per_stream_biasing", "batch_size"):
        if spec and spec.get(key) != exp[key]:
            bad.append(f"spec {key}={spec.get(key)!r}, expected {exp[key]!r}")
    if spec and spec.get("word_confidence") not in (None, "off"):
        bad.append(f"word_confidence={spec.get('word_confidence')!r}: a separate configuration")
    if doc.get("execution") != "eager":
        bad.append(f"execution {doc.get('execution')!r}, expected 'eager'")
    if spec and not isinstance(spec.get("matmul_precision"), str):
        bad.append(f"matmul_precision {spec.get('matmul_precision')!r} is not stated")
    budget = doc.get("budget_ms")
    if not isinstance(budget, (int, float)) or budget <= 0:
        bad.append(f"no tick budget ({budget!r})")
    for line in bad:
        print(line)
    return 1 if bad else 0


def smoke(ws, admission_url, manifest, chunk, lang, solo_s, burst, out):
    """About `solo_s` of audio as one session, then `burst` sessions at once, through the
    gate's own client, with /admission read every half second throughout. Writes what it
    saw to `out`; smoke_check decides. Imports verbatim_bench."""
    import asyncio

    from verbatim_bench.client import ChunkMode
    from verbatim_bench.invariance import Level, manifest_corpus, run_level

    solo_s, burst = float(solo_s), int(burst)
    clips = manifest_corpus(Path(manifest))
    solo, audio = [], 0.0
    for clip in clips:
        if audio >= solo_s:
            break
        solo.append(clip)
        audio += clip.duration_s
    burst_clips = clips[:burst]
    doc = {"solo_target_s": solo_s, "burst_concurrency": burst, "phases": {},
           "admission": {}, "samples": []}

    def phase_doc(run, clips_run):
        return {
            "sessions": len(run.finals) + len(run.errors),
            "errors": len(run.errors),
            "first_error": next(iter(run.errors.values()), None),
            "audio_s": round(sum(c.duration_s for c in clips_run), 3),
            "wall_clock_s": round(run.wall_clock_s, 3),
            "pacing_slip_p99_ms": run.pacing_slip_p99_ms,
        }

    async def main():
        mode = ChunkMode.parse(chunk)
        phase = {"now": "solo"}
        stop = asyncio.Event()

        async def poll():
            while not stop.is_set():
                reading = await asyncio.to_thread(read_json_url, admission_url)
                doc["samples"].append({"t": time.time(), "phase": phase["now"], "admission": reading})
                with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                    await asyncio.wait_for(stop.wait(), 0.5)

        doc["admission"]["before"] = await asyncio.to_thread(read_json_url, admission_url)
        poller = asyncio.create_task(poll())
        try:
            run = await run_level(ws, solo, Level("1", 1), chunk=mode, lang=lang)
            doc["phases"]["solo"] = phase_doc(run, solo)
            doc["admission"]["end_of_solo"] = await asyncio.to_thread(read_json_url, admission_url)
            phase["now"] = "burst"
            run = await run_level(ws, burst_clips, Level("max", burst), chunk=mode, lang=lang)
            doc["phases"]["burst"] = phase_doc(run, burst_clips)
            doc["admission"]["end_of_burst"] = await asyncio.to_thread(read_json_url, admission_url)
        finally:
            stop.set()
            await poller

    try:
        asyncio.run(main())
    finally:
        Path(out).write_text(json.dumps(doc, indent=2) + "\n")
    return 0


def smoke_check(path, budget_ms):
    """Whether the long gate can keep its verdict on this server. Prints a JSON summary;
    exit 1 with the reasons when it cannot: a smoke session errored, p95_tick_ms reached
    the budget, a session was refused, or admissions were held (degradation_level >= 1).
    The p95 is read where the window no longer holds a fresh server's first ticks: at the
    end of the solo phase, during the burst, and at its end."""
    doc = json.loads(Path(path).read_text())
    budget = float(budget_ms)
    bad = []
    phases = doc.get("phases") or {}
    for name in ("solo", "burst"):
        ph = phases.get(name)
        if not ph or not ph.get("sessions"):
            bad.append(f"the {name} phase ran no session")
        elif ph.get("errors"):
            bad.append(f"{ph['errors']} of {ph['sessions']} {name} sessions errored (first: {ph.get('first_error')})")
    burst = phases.get("burst") or {}
    if burst.get("sessions") and burst["sessions"] < int(doc.get("burst_concurrency") or 0):
        bad.append(f"the burst ran {burst['sessions']} sessions, not {doc['burst_concurrency']}: the corpus is too small")
    adm = doc.get("admission") or {}
    ok = lambda r: isinstance(r, dict) and "error" not in r
    samples = [s for s in doc.get("samples") or [] if ok(s.get("admission"))]
    p95_readings = [r for r in [adm.get("end_of_solo"), adm.get("end_of_burst")] if ok(r)]
    p95_readings += [s["admission"] for s in samples if s.get("phase") == "burst"]
    p95_values = [r["p95_tick_ms"] for r in p95_readings if isinstance(r.get("p95_tick_ms"), (int, float))]
    every = [r for r in [adm.get("before"), adm.get("end_of_solo"), adm.get("end_of_burst")] if ok(r)]
    every += [s["admission"] for s in samples]
    out = {
        "budget_ms": budget,
        "max_p95_tick_ms": max(p95_values) if p95_values else None,
        "max_consecutive_overruns": max((r.get("consecutive_overruns") or 0 for r in every), default=None),
        "max_degradation_level": max((r.get("degradation_level") or 0 for r in every), default=None),
        "refused_total_before": (adm.get("before") or {}).get("refused_total"),
        "refused_total_after": (adm.get("end_of_burst") or {}).get("refused_total"),
        "admission_readings": len(every),
        "problems": bad,
    }
    if not p95_values:
        bad.append("no p95_tick_ms reading after the solo phase: the smoke cannot say the card keeps time")
    elif out["max_p95_tick_ms"] >= budget:
        bad.append(f"p95_tick_ms reached {out['max_p95_tick_ms']} ms against a budget of {budget} ms: "
                   "this card does not step this bucket inside the tick budget")
    before, after = out["refused_total_before"], out["refused_total_after"]
    if not isinstance(before, int) or not isinstance(after, int):
        bad.append(f"refused_total was not read before and after the smoke ({before!r}, {after!r})")
    elif after > before:
        bad.append(f"refused_total rose from {before} to {after} during the smoke: sessions were refused")
    if out["max_degradation_level"]:
        bad.append(f"degradation_level reached {out['max_degradation_level']}: the server held admissions")
    print(json.dumps(out, indent=2))
    return 1 if bad else 0


def card_reading(gpu_index, server_pid, timeout):
    """One nvidia-smi reading of the compute processes on the card: whether the server is
    among them and every line that is not the server, or {"error": ...}."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=float(timeout))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if done.returncode != 0:
        return {"error": f"nvidia-smi exited {done.returncode}: {done.stderr.strip()[:200]}"}
    rows = [ln.strip() for ln in done.stdout.splitlines() if ln.split(",")[0].strip().isdigit()]
    mine = [ln for ln in rows if ln.split(",")[0].strip() == str(server_pid)]
    return {"server_seen": bool(mine), "foreign": [ln for ln in rows if ln not in mine]}


def monitor(url, out, gate_pid, every_s, baseline_refused, abort_file, server_pid, gpu_index,
            card_timeout_s, read_timeout_s):
    """Read /admission, and what holds the card, sleeping `every_s` between polls while the
    gate runs; append each reading, stamped with the time it was written, to `out`. Exits
    0 by itself once the gate is gone. As soon as refused_total exceeds the baseline or
    degradation_level reaches 1, write the reason to `abort_file` and SIGTERM the gate: a
    refused session leaves its level incomplete, so the verdict is lost and the rest of the
    gate is wasted card time. Another process on the card is recorded, not acted on here.
    The first line it writes is the arguments it parsed: the sleep and the two timeouts it
    then passes on, which is a statement of intent. What they did is measured: each poll
    line carries how long its /admission read and its nvidia-smi query took ("took_s"),
    so a timeout that fired is seen at the length it had, and the poll timestamps give
    the spacing the sleep made."""
    pid, every, base = int(gate_pid), float(every_s), int(baseline_refused)
    card_s, read_s = float(card_timeout_s), float(read_timeout_s)
    with open(out, "a") as fh:
        fh.write(json.dumps({"t": time.time(), "monitor": {
            "poll_sleep_s": every, "card_query_timeout_s": card_s, "admission_read_timeout_s": read_s,
            "gate_pid": pid, "server_pid": str(server_pid), "gpu_index": str(gpu_index),
            "baseline_refused": base, "pid": os.getpid()}}) + "\n")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 0
        began = time.monotonic()
        reading = read_json_url(url, timeout=read_s)
        read_took = time.monotonic() - began
        began = time.monotonic()
        card = card_reading(gpu_index, server_pid, card_s)
        card_took = time.monotonic() - began
        with open(out, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "admission": reading, "card": card,
                                 "took_s": {"admission": round(read_took, 3),
                                            "card": round(card_took, 3)}}) + "\n")
        reason = None
        if "error" not in reading:
            refused, level = reading.get("refused_total"), reading.get("degradation_level")
            if isinstance(refused, int) and refused > base:
                reason = f"refused_total rose from {base} to {refused}: a level is incomplete, so there is no verdict"
            elif isinstance(level, int) and level >= 1:
                reason = f"degradation_level {level}: the server is holding admissions, so sessions will be refused"
        if reason:
            Path(abort_file).write_text(reason + "\n")
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
            return 0
        time.sleep(every)


def gap_bound(every_s, card_timeout_s, read_timeout_s):
    """The longest stretch of a gate the monitor may leave without a poll: three times the
    sleep between polls plus both of a poll's timeouts. A working monitor stays inside it
    with room to spare (one sleep plus, at worst, both timeouts); a dead, hung or starved
    one does not. Printed for the runbook, which waits this long for the monitor to stop."""
    bound = _gap_bound(every_s, card_timeout_s, read_timeout_s)
    print(f"{bound:g}")
    return 0


def _gap_bound(every_s, card_timeout_s, read_timeout_s):
    return 3 * float(every_s) + float(card_timeout_s) + float(read_timeout_s)


def monitor_summary(path, gpu_index, every_s, card_timeout_s, read_timeout_s,
                    t_gate_start, t_gate_end, monitor_exit, monitor_end=""):
    """What the monitor saw during the gate, for the arm record: whether this card kept the
    tick budget through the whole run, and what else held the card at its polls. A line
    that does not parse (cut short mid-write) is counted, not fatal. "problems" lists what
    takes the arm's verdict away, from what was seen and from what was not: a poll that
    failed or never happened observed nothing, and nothing observed is never read as
    nothing there. How much of the gate the polls covered is read from their own
    timestamps against the gate's start and end, not from the spacing that was asked for:
    every stretch without a poll longer than gap_bound, the first (gate start to first
    poll) and the last (last poll to gate end) included, is a problem, and so is a monitor
    that did not exit 0 by itself (`monitor_exit` is its status as the runbook's `wait`
    returned it, `monitor_end` how it ended when the runbook had to stop it). `every_s`,
    `card_timeout_s` and `read_timeout_s` are what the runbook asked the monitor for, and
    set the bound; the arguments the monitor parsed are read from the first line it
    wrote, and a monitor that wrote none, or parsed anything else, is a problem too. How
    long its reads took is what it measured at each poll ("took_s"), reported as the
    longest of each."""
    start, end = float(t_gate_start), float(t_gate_end)
    bound = _gap_bound(every_s, card_timeout_s, read_timeout_s)
    asked = [float(every_s), float(card_timeout_s), float(read_timeout_s)]
    readings, errors, unparsable, polls, poll_errors, foreign_polls, absent_polls = [], 0, 0, 0, 0, 0, 0
    foreign_seen, first_poll_error, first_read_error, stamps, ran = [], None, None, [], None
    took = {"admission": [], "card": []}
    with contextlib.suppress(FileNotFoundError):
        for line in Path(path).read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                unparsable += 1
                continue
            if isinstance(entry, dict) and "monitor" in entry:
                # The arguments the monitor parsed: not a poll, and its time is not one.
                if ran is None and isinstance(entry["monitor"], dict):
                    ran = entry["monitor"]
                continue
            stamp = entry.get("t") if isinstance(entry, dict) else None
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
                stamps.append(float(stamp))
            measured = entry.get("took_s") if isinstance(entry, dict) else None
            for key, values in took.items():
                value = measured.get(key) if isinstance(measured, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            reading = entry.get("admission") if isinstance(entry, dict) else None
            if isinstance(reading, dict) and "error" not in reading:
                readings.append(reading)
            else:
                errors += 1
                if first_read_error is None:
                    first_read_error = reading.get("error") if isinstance(reading, dict) else repr(reading)
            card = entry.get("card") if isinstance(entry, dict) else None
            if isinstance(card, dict):
                polls += 1
                if "error" in card:
                    poll_errors += 1
                    if first_poll_error is None:
                        first_poll_error = card["error"]
                    continue
                if card.get("foreign"):
                    foreign_polls += 1
                    foreign_seen += [f for f in card["foreign"] if f not in foreign_seen]
                if not card.get("server_seen"):
                    absent_polls += 1
    top = lambda key: max((r.get(key) for r in readings if isinstance(r.get(key), (int, float))), default=None)
    # The stretches of the gate between observations: gate start, each poll (clamped into
    # the gate: a poll that began before the gate ended can be written just after it), end.
    stamps.sort()
    points = [start] + [min(max(t, start), end) for t in stamps] + [end]
    stretches = [(a - start, b - start, b - a) for a, b in zip(points, points[1:], strict=False)]
    unwatched = [s for s in stretches if s[2] > bound]
    longest = max(stretches, key=lambda s: s[2]) if stretches else None
    spacings = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    median_spacing = sorted(spacings)[len(spacings) // 2] if spacings else None
    cadence = (f"polls a median {median_spacing:.2f} s apart, longest stretch without one "
               f"{longest[2]:.1f} s" if median_spacing is not None and longest else
               f"{len(stamps)} timestamped poll(s)")
    card = f"card {gpu_index}"
    problems = []
    if polls == 0:
        problems.append(f"the monitor made no nvidia-smi poll of {card} during the gate: what held the card "
                        "while the gate ran was not observed")
    elif poll_errors:
        problems.append(f"nvidia-smi failed at {poll_errors} of {polls} polls during the gate (first: "
                        f"{first_poll_error}): what held {card} then was not observed")
    if unwatched:
        worst = max(unwatched, key=lambda s: s[2])
        problems.append(f"the monitor observed nothing for {worst[2]:.1f} s of the gate, from {worst[0]:.1f} s "
                        f"to {worst[1]:.1f} s after it started ({len(unwatched)} stretch(es) over the "
                        f"{bound:g} s bound: 3 x the {asked[0]:g} s sleep between polls plus the "
                        f"{asked[1]:g} s nvidia-smi and {asked[2]:g} s /admission timeouts the runbook "
                        f"set): what held {card}, and the tick budget, were not watched then")
    parsed = None if ran is None else [ran.get("poll_sleep_s"), ran.get("card_query_timeout_s"),
                                       ran.get("admission_read_timeout_s")]
    if parsed is None:
        problems.append("the monitor wrote no line saying what it parsed: its sleep between polls and its "
                        "nvidia-smi and /admission timeouts were not recorded")
    elif parsed != asked:
        problems.append(f"the monitor parsed a {parsed[0]!r} s sleep between polls, a {parsed[1]!r} s "
                        f"nvidia-smi timeout and a {parsed[2]!r} s /admission timeout, not the "
                        f"{asked[0]:g} s, {asked[1]:g} s and {asked[2]:g} s the runbook asked for")
    exit_status = int(monitor_exit) if str(monitor_exit).strip().isdigit() else None
    if exit_status != 0:
        how = f" ({monitor_end})" if monitor_end else ""
        status = "not recorded" if exit_status is None else exit_status
        problems.append(f"the monitor's exit status is {status}{how}, not 0: it did not run to the gate's end "
                        "and stop by itself, so what it did not write was not observed")
    if foreign_polls:
        problems.append(f"nvidia-smi showed a process other than the server on {card} at {foreign_polls} "
                        f"of {polls} polls during the gate ({cadence}): {foreign_seen[:10]}")
    if absent_polls:
        problems.append(f"nvidia-smi did not show the server on {card} at {absent_polls} of {polls} polls "
                        "during the gate")
    if not readings:
        problems.append(f"the monitor got no /admission reading during the gate ({errors} failed, first: "
                        f"{first_read_error}): the tick budget and refusals were not watched while it ran")
    print(json.dumps({
        "readings": len(readings),
        "read_errors": errors,
        "first_read_error": first_read_error,
        "unparsable_lines": unparsable,
        "max_p95_tick_ms": top("p95_tick_ms"),
        "max_consecutive_overruns": top("consecutive_overruns"),
        "max_degradation_level": top("degradation_level"),
        "max_live": top("live"),
        "max_refused_total": top("refused_total"),
        "card_polls": polls,
        "card_poll_errors": poll_errors,
        "first_card_poll_error": first_poll_error,
        "card_polls_with_foreign": foreign_polls,
        "card_polls_without_server": absent_polls,
        "card_foreign_seen": foreign_seen[:10],
        "gate_start": start,
        "gate_end": end,
        # The arguments the monitor parsed, from the line it wrote first (null when it wrote
        # none), and what the runbook asked it for: both statements of intent.
        "monitor_parsed": ran,
        "poll_sleep_parsed_s": None if parsed is None else parsed[0],
        "card_query_timeout_parsed_s": None if parsed is None else parsed[1],
        "admission_read_timeout_parsed_s": None if parsed is None else parsed[2],
        "poll_sleep_requested_s": asked[0],
        "card_query_timeout_requested_s": asked[1],
        "admission_read_timeout_requested_s": asked[2],
        # What was measured: how long the longest read of each kind took, and the spacing
        # of the polls' own timestamps.
        "longest_admission_read_s": max(took["admission"], default=None),
        "longest_card_query_s": max(took["card"], default=None),
        "timestamped_polls": len(stamps),
        "poll_spacing_median_s": None if median_spacing is None else round(median_spacing, 3),
        "first_poll_after_s": round(stamps[0] - start, 3) if stamps else None,
        "last_poll_before_end_s": round(end - stamps[-1], 3) if stamps else None,
        "longest_unwatched_s": None if longest is None else round(longest[2], 3),
        "unwatched_bound_s": round(bound, 3),
        "unwatched_stretches_over_bound": [[round(a, 3), round(b, 3)] for a, b, _ in unwatched],
        "monitor_exit": exit_status,
        "monitor_end": monitor_end or None,
        "problems": problems,
    }))
    return 0


def record(path, expect):
    """Validate a vb-invariance/1 record (and the finals file beside it) against the arm
    that produced it and summarise it. Prints the summary as JSON; exit 1 when either is
    missing or does not describe the arm asked for (the verdict is reported, not judged)."""
    exp = json.loads(expect)
    p = Path(path)
    out = {"record_path": str(p), "record_present": p.is_file(), "problems": [],
           "gate_exit_checked": exp.get("gate_exit")}
    if not p.is_file():
        out["problems"].append("no record was written")
        print(json.dumps(out))
        return 1
    raw = p.read_bytes()
    out["record_sha256"] = hashlib.sha256(raw).hexdigest()
    rec = json.loads(raw)
    problems = out["problems"]
    if rec.get("record") != "vb-invariance/1":
        problems.append(f"record kind {rec.get('record')!r}")
    levels = rec.get("levels") or []
    if [lv.get("slot") for lv in levels] != SLOTS:
        problems.append(f"slots {[lv.get('slot') for lv in levels]}")
    if [lv.get("concurrency") for lv in levels] != [1, 32, 32, exp["max"]]:
        problems.append(f"concurrency {[lv.get('concurrency') for lv in levels]}, expected 1/32/32/{exp['max']}")
    for lv in levels[:3]:
        if lv.get("churn_period_s") is not None:
            problems.append(f"level {lv.get('slot')} is churned; only max may be")
    top = levels[-1] if levels else {}
    if top.get("churn_period_s") != exp["churn_period_s"]:
        problems.append(f"max level churn_period_s={top.get('churn_period_s')!r}, expected {exp['churn_period_s']!r}")
    hist = top.get("admission_occupancy_histogram")
    if exp["churn_period_s"] is None:
        if hist is not None:
            problems.append("a constant max level carries an admission histogram")
    elif not hist:
        problems.append("the churned level recorded no admission occupancy: the wave is not shown")
    else:
        seen = sorted(int(k) for k in hist)
        out["max_level_admission_occupancy_range"] = [seen[0], seen[-1]]
        if seen[-1] <= 1:
            problems.append("the churned level never admitted more than one stream")
    if rec.get("chunk_ms") != exp["chunk_ms"]:
        problems.append(f"chunk_ms {rec.get('chunk_ms')!r}")
    corpus_block = rec.get("corpus") or {}
    out["corpus_id"] = corpus_block.get("id")
    if corpus_block.get("id") != exp["corpus_id"]:
        problems.append(f"corpus id {corpus_block.get('id')!r}, expected {exp['corpus_id']!r}")
    if corpus_block.get("utterances") != exp["utterances"]:
        problems.append(f"corpus utterances {corpus_block.get('utterances')!r}, expected {exp['utterances']!r}")
    if rec.get("biasing") is not None or rec.get("controls") is not None:
        problems.append("the record carries biasing: this runbook runs the bare arm only")
    # Word timings are a channel of their own: on this checkpoint they moved for more
    # streams than the text did (the step-1 probe). A pass over text alone is not this gate.
    if rec.get("timings_present") is not True:
        problems.append(f"timings_present={rec.get('timings_present')!r}: only the text was compared")
    verdict = rec.get("verdict")
    divs = rec.get("divergences") or []
    digests = {lv.get("slot"): lv.get("digest") for lv in levels}
    out.update({
        "verdict": verdict,
        "exit_code_in_record": rec.get("exit_code"),
        "invariance": rec.get("invariance"),
        "digests": digests,
        "distinct_digests": len({d for d in digests.values() if d}),
        "timings_present": rec.get("timings_present"),
        "wall_clock_s": rec.get("wall_clock_s"),
        # Per comparison, the way the gate's own FINAL line counts them
        # (invariance.py:665-683), never the union alone.
        "streams_differing": {
            f"{a}_vs_{b}": len({d["stream_id"] for d in divs if d["level"] == a and d["against"] == b})
            for a, b in PAIRS
        },
        "streams_differing_from_1_any_level": len({d["stream_id"] for d in divs if d["against"] == "1"}),
        "divergence_kinds": dict(Counter(d.get("kind") for d in divs)),
        "levels": [
            {k: lv.get(k) for k in ("slot", "concurrency", "churn_period_s", "streams", "errors",
                                    "first_error", "wall_clock_s", "pacing_slip_p99_ms")}
            for lv in levels
        ],
    })
    if verdict == "invariant" and out["distinct_digests"] != 1:
        problems.append("verdict invariant with more than one digest")
    # The gate gives a verdict only over four levels that each digested (invariance.py
    # assess): a level without a digest is not "the same as the others", and the restart
    # comparison would have nothing to compare there.
    undigested = [slot for slot in SLOTS if not digests.get(slot)]
    if verdict in ("invariant", "divergent") and undigested:
        problems.append(f"verdict {verdict} with no digest at level(s) {undigested}")
    if verdict == "divergent" and not divs:
        problems.append("verdict divergent with no divergence listed")
    if exp.get("gate_exit") is not None and rec.get("exit_code") != exp["gate_exit"]:
        problems.append(f"the gate exited {exp['gate_exit']} and its record says {rec.get('exit_code')!r}")
    finals_doc = None
    if exp.get("finals_path") is not None:
        finals_doc = _finals(Path(exp["finals_path"]), levels, out, problems)
    if exp.get("manifest") is not None:
        _level1_wer(finals_doc, Path(exp["manifest"]), float(exp["level1_wer_max"]), out, problems)
    print(json.dumps(out))
    return 1 if problems else 0


def _finals(path, levels, out, problems):
    """The finals file re-digests, level by level, to the record's digests through the
    gate's own canonicaliser (imports verbatim_bench.canonical): it is this run's.
    Returns the parsed file, or None when there is none."""
    out["finals_path"] = str(path)
    if not path.is_file():
        problems.append("no finals file was written")
        return None
    raw = path.read_bytes()
    out["finals_sha256"] = hashlib.sha256(raw).hexdigest()
    from verbatim_bench.canonical import FinalRecord, finals_digest
    doc = json.loads(raw)
    if doc.get("record") != FINALS_RECORD:
        problems.append(f"finals kind {doc.get('record')!r}")
    by_slot = {lv.get("slot"): lv for lv in doc.get("levels") or []}
    match = {}
    for lv in levels:
        slot = lv.get("slot")
        mine = by_slot.get(slot)
        if mine is None:
            match[slot] = False
        elif lv.get("digest") is None:
            match[slot] = None
        else:
            records = [FinalRecord(e["stream_id"], e["text"], tuple((w, s, t) for w, s, t in e["words"]))
                       for e in mine.get("finals") or []]
            match[slot] = finals_digest(records) == lv.get("digest")
    out["finals_digest_match"] = match
    if any(v is False for v in match.values()):
        problems.append(f"the finals file does not re-digest to the record's digests: {match}")
    return doc


def _level1_wer(finals_doc, manifest, bound, out, problems):
    """Level 1's corpus WER against the manifest's reference text, with the gate's own
    pinned normaliser (imports verbatim_bench.wer and verbatim_bench.corpus). A stream with
    no level-1 final counts its whole reference as deletions. A sanity bound, not a quality
    claim: identical wrong text at every level is invariant, and only this sees it."""
    from verbatim_bench.corpus import load_manifest
    from verbatim_bench.wer import corpus_wer

    level = next((lv for lv in (finals_doc or {}).get("levels") or [] if lv.get("slot") == "1"), None)
    if level is None:
        problems.append("level 1 could not be scored against the reference: no level-1 finals")
        return
    hyp = {e["stream_id"]: e["text"] for e in level.get("finals") or []}
    refs = load_manifest(manifest)
    count = corpus_wer((u.text, hyp.get(u.stream_id, "")) for u in refs)
    out["level1_wer"] = {
        "wer": None if count.wer is None else round(count.wer, 6),
        "errors": count.errors,
        "reference_words": count.reference_words,
        "streams_in_manifest": len(refs),
        "streams_with_a_final": sum(1 for u in refs if u.stream_id in hyp),
        "bound": bound,
        "normaliser": "verbatim_bench.wer.normalise",
    }
    if count.wer is None:
        problems.append("level 1 could not be scored: the manifest has no reference words")
    elif count.wer > bound:
        problems.append(f"level-1 WER {count.wer:.4f} is over the sanity bound {bound}: "
                        "the server's text is not a transcript of this corpus")


def argv(out, *args):
    Path(out).write_text(json.dumps(list(args)) + "\n")
    return 0


def _obj(pairs):
    """key=value pairs; '@path' loads JSON from a file (null when absent or not JSON),
    '#literal' is a JSON literal, anything else is a string."""
    doc = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if value.startswith("@"):
            try:
                doc[key] = json.loads(Path(value[1:]).read_text())
            except Exception:
                doc[key] = None
        elif value.startswith("#"):
            doc[key] = json.loads(value[1:])
        else:
            doc[key] = value
    return doc


def obj(*pairs):
    print(json.dumps(_obj(pairs)))
    return 0


def arm(out, *pairs):
    Path(out).write_text(json.dumps(_obj(pairs), indent=2) + "\n")
    return 0


def dry(arm_name, record_path, finals_path, *rest):
    cut1, cut2 = rest.index("::SERVE::"), rest.index("::GATE::")
    print(json.dumps({
        "arm": arm_name,
        "record": record_path,
        "finals": finals_path,
        "env": list(rest[:cut1]),
        "serve": list(rest[cut1 + 1:cut2]),
        "gate": list(rest[cut2 + 1:]),
    }))
    return 0


def summary(run_dir, stopped=""):
    """Every arm in the run directory, the controls, the restart comparisons, the exit code.
    `stopped` is why the run stopped before its last arm, when it did."""
    run = Path(run_dir)
    arms = {}
    for p in sorted(run.glob("*/arm.json")):
        doc = json.loads(p.read_text())
        arms[doc["arm"]] = doc
    verdicts, rows, precision, divergent_with_ops, divergent_unusable = {}, [], {}, {}, {}
    for name in ORDER:
        doc = arms.get(name)
        if doc is None:
            continue
        rs = doc.get("record_summary") or {}
        record_problems = list(rs.get("problems") or []) + ([] if rs else ["no record summary was written"])
        invalidating = list(doc.get("invalidating_problems") or [])
        operational = list(doc.get("operational_problems") or [])
        verdict = rs.get("verdict")
        complete = bool(rs.get("record_present")) and not record_problems and not invalidating
        if complete and verdict == "divergent" and (name in TWIN or not operational):
            # A fixed arm's divergence in a complete record stands; problems that do not
            # touch the record are named beside it. A ragged arm controls nothing unless
            # its run was clean.
            verdicts[name] = "divergent"
            if operational:
                divergent_with_ops[name] = operational
        elif complete and verdict == "invariant" and not operational:
            verdicts[name] = "invariant"
        else:
            verdicts[name] = None
            if verdict == "divergent":
                divergent_unusable[name] = record_problems + invalidating + operational
        precision[name] = (((doc.get("serve_spec") or {}).get("spec")) or {}).get("matmul_precision")
        diff = rs.get("streams_differing") or {}
        rows.append({
            "arm": name,
            "padding": doc["padding"],
            "occupancy": doc["occupancy"],
            "verdict": verdicts[name] or f"NO VERDICT ({verdict})",
            "distinct_digests": rs.get("distinct_digests"),
            "differing_vs_1_32a_32b_max": [diff.get(c) for c in AGAINST_ONE],
            "differing_32b_vs_32a": diff.get("32b_vs_32a"),
            "errors_per_level": [lv.get("errors") for lv in rs.get("levels") or []],
            "level1_wer": rs.get("level1_wer"),
            "gate_wall_clock_s": rs.get("wall_clock_s"),
            "monitor": doc.get("monitor_summary"),
            "problems": record_problems + invalidating + operational,
        })

    def digest(name, slot):
        if verdicts.get(name) is None:
            return None
        return ((arms[name].get("record_summary") or {}).get("digests") or {}).get(slot)

    # Two server processes, same padding, same corpus, same seeds: the levels the two arms
    # share must give byte-identical digests. All four for the fixed pair (the served
    # configuration); level 1 only for the ragged pair, where one stream is in flight and
    # the batch cannot differ. The ragged 32a/32b comparison is information only: those
    # batches follow arrival timing, which differs between two runs.
    cross = {}
    for a, b, slots in (("fixed-churn", "fixed-const", SLOTS), ("ragged-churn", "ragged-const", SLOTS[:3])):
        cross[f"{a}_vs_{b}"] = {
            s: (digest(a, s) == digest(b, s)) if digest(a, s) and digest(b, s) else None
            for s in slots
        }
    fixed_cross, ragged_cross = cross["fixed-churn_vs_fixed-const"], cross["ragged-churn_vs_ragged-const"]
    restart_failures = [f"fixed-churn and fixed-const differ at level {s}" for s, same in fixed_cross.items() if same is False]
    if ragged_cross["1"] is False:
        restart_failures.append("ragged-churn and ragged-const differ at level 1, one stream in flight")
    restart_missing = [f"fixed level {s}" for s, same in fixed_cross.items() if same is None]
    if ragged_cross["1"] is None:
        restart_missing.append("ragged level 1")

    # Each fixed arm's comparisons against level 1 are controlled only by the ragged arm of
    # the SAME occupancy diverging at the SAME comparison: a ragged divergence elsewhere,
    # or at 32 alone, says nothing about the level it did not touch.
    controls, uncontrolled = {}, []
    for fixed, twin in TWIN.items():
        if verdicts.get(fixed) != "invariant":
            continue
        tdiff = ((arms[twin].get("record_summary") or {}).get("streams_differing") or {}) if verdicts.get(twin) else {}
        per = {c: tdiff.get(c) for c in AGAINST_ONE}
        controls[fixed] = {"twin": twin, "twin_verdict": verdicts.get(twin), "twin_streams_differing": per}
        for c, n in per.items():
            if isinstance(n, int) and n > 0:
                continue
            state = ("did not run" if twin not in arms else
                     "has no verdict" if verdicts.get(twin) is None else
                     f"shows {n} streams differing there" if isinstance(n, int) else "carries no count there")
            uncontrolled.append(f"{fixed} {c.replace('_vs_', ' vs ')} (twin {twin} {state})")

    # A run directory can be continued (RUNBOOK_RUN_ID), so its arms may come from
    # different invocations. Arms at different buckets or max levels, or run by a different
    # runbook, git HEAD or package set, are different experiments: no control, and no
    # restart comparison, holds between them. Settings an arm did not record are unknown,
    # never assumed equal.
    def setting(doc):
        gate = doc.get("gate_argv") or []
        code = doc.get("code") or {}
        return {"bucket": (((doc.get("serve_spec") or {}).get("spec")) or {}).get("batch_size"),
                "max_level": gate[gate.index("--max") + 1] if "--max" in gate[:-1] else None,
                "runbook_sha256": (doc.get("runbook") or {}).get("sha256"),
                "git_head": doc.get("git_head"),
                "versions": {k: code[k] for k in sorted(code) if k.startswith("version_")}}
    settings = {n: setting(arms[n]) for n in ORDER if n in arms}
    unrecorded = [n for n, v in settings.items()
                  if None in (v["bucket"], v["max_level"], v["runbook_sha256"], v["git_head"]) or not v["versions"]]
    mixed = len({json.dumps(v, sort_keys=True) for v in settings.values()}) > 1

    ran = [n for n in ORDER if n in arms]
    missing = [n for n in ORDER if n not in arms]
    no_verdict = [n for n in ran if verdicts[n] is None]
    fixed_div = [n for n in TWIN if verdicts.get(n) == "divergent"]
    fixed_inv = [n for n in TWIN if verdicts.get(n) == "invariant"]
    unusable = [f"{n}'s record says divergent, but: {'; '.join(p)}" for n, p in divergent_unusable.items()]
    also = ([f"no verdict for {no_verdict}"] if no_verdict else []) + unusable + restart_failures
    if fixed_div:
        beside = [f"{n} also had operational problems, none of which touches its record: {'; '.join(p)}"
                  for n, p in divergent_with_ops.items()]
        code, reading = 1, (f"fixed-shape padding DIVERGED within {fixed_div}"
                            + "".join(f"; {b}" for b in beside)
                            + (f"; also: {'; '.join(also)}" if also else "")
                            + (f"; the run stopped before it finished: {stopped}" if stopped else ""))
    elif stopped:
        code, reading = 2, (f"the run stopped before it finished: {stopped}. Of the arms in this run "
                            f"directory: ran {ran or 'none'}, not run {missing or 'none'}"
                            + (f"; {'; '.join(also)}" if also else ""))
    elif unrecorded:
        code, reading = 2, (f"arms {unrecorded} did not record their settings, so none can be read "
                            f"against another: {settings}")
    elif mixed:
        code, reading = 2, (f"the arms in this run directory ran at different settings, so none "
                            f"controls or reproduces another: {settings}")
    elif restart_failures:
        code, reading = 4, ("a server restart changed the bytes: " + "; ".join(restart_failures)
                            + (f"; also: no verdict for {no_verdict}" if no_verdict else ""))
    elif no_verdict:
        code, reading = 2, f"no verdict for {no_verdict}" + "".join(f"; {u}" for u in unusable)
    elif not fixed_inv:
        code, reading = 3, "no fixed arm in this run directory: nothing to read"
    elif uncontrolled:
        code, reading = 3, (
            "fixed invariant, but these comparisons are UNCONTROLLED: " + "; ".join(uncontrolled)
            + ". DR-0014: a pass the control arm could not fail is not evidence that the padding "
            "holds the property; either something else holds it or this corpus and span do not discriminate")
    elif missing or restart_missing:
        code, reading = 3, (
            f"fixed invariant and controlled, but not run: {missing or 'none'}; restart comparison "
            f"not made at: {restart_missing or 'none'}. Exit 0 needs all four arms")
    else:
        counts = "; ".join(f"{twin} {'/'.join(str(controls[f]['twin_streams_differing'][c]) for c in AGAINST_ONE)}"
                           for f, twin in TWIN.items())
        code, reading = 0, (
            "fixed invariant at both occupancies; every comparison against level 1 controlled by its "
            f"ragged twin (streams differing at 32a/32b/max vs 1: {counts}); fixed digests identical "
            "across the restart at all four levels, ragged at level 1")
    doc = {"run_dir": str(run), "scope": SCOPE, "settings_by_arm": settings,
           "stopped": stopped or None,
           "matmul_precision_by_arm": precision,
           "matmul_precision_source": "the NeMo spec the server's own code builds from the served argv (derive_spec), not read from the running process",
           "arms": rows, "controls": controls, "cross_restart_digest_agreement": cross,
           "exit_code": code, "reading": reading}
    (run / "summary.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))
    print(f"EXIT {code}: {reading}. {SCOPE}")
    return code


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    try:
        status = globals()[command](*args)
    except Exception:
        import traceback

        traceback.print_exc()
        print(f"[runbook helper] {command} raised; exit 2 (no verdict), never 1", file=sys.stderr)
        status = 2
    sys.exit(status)
PYEOF
readonly HELPER

helper() { "$PY" -c "$HELPER" "$@"; }

# ---------------------------------------------------------------------------------------
# The two command lines. Each flag's source is cited beside it (src/ and bench/src/ at
# HEAD 854d4b9).
# ---------------------------------------------------------------------------------------
build_serve() {  # $1 = fixed | ragged, $2 = pipeline (default: this mode's); sets SERVE_ENV and SERVE_ARGV
    local padding=$1 pipeline=${2:-$PIPELINE}
    if [[ "$TEST_FAKE" == 1 ]]; then
        SERVE_ENV=(CUDA_VISIBLE_DEVICES= HF_HUB_CACHE="$HF_HUB_DIR" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
                   PYTHONPATH="$PYTHONPATH")
    else
        SERVE_ENV=(
            CUDA_VISIBLE_DEVICES="$GPU_INDEX"   # card 3; the process then sees it as cuda:0
            CUDA_DEVICE_ORDER=PCI_BUS_ID        # so "3" is the card nvidia-smi calls 3
            HF_HUB_CACHE="$HF_HUB_DIR"          # the cache preflight checked, as huggingface_hub resolves it
            HF_HUB_OFFLINE=1                    # resolve refs/main in the cache (checked = the pinned revision), never the network
            PYTHONUNBUFFERED=1
            PYTHONPATH="$PYTHONPATH"            # this repository's src/ and bench/src/, first on the path
        )
    fi
    SERVE_ARGV=(
        "$VERBATIM" serve                       # subcommand: src/verbatim/cli.py:135
        "$MODEL"                                # positional checkpoint: src/verbatim/cli.py:144
        --chunk "$CHUNK"                        # src/verbatim/cli.py:146 (parsed by _chunk_arg, cli.py:93)
        --bucket "$BUCKET"                      # src/verbatim/cli.py:160 (xor --ceiling, cli.py:152-154)
        --padding "$padding"                    # src/verbatim/cli.py:205, choices fixed|ragged
        --eager                                 # src/verbatim/cli.py:199; ragged + graph path is refused anyway (DR-0014)
        --compute-dtype "$DTYPE"                # src/verbatim/cli.py:251
        --att-context-left "$ATT_LEFT"          # src/verbatim/cli.py:244; right context follows the chunk
        --pipeline "$pipeline"                  # src/verbatim/cli.py:192
        --host "$HOST"                          # src/verbatim/cli.py:165
        --ws-port "$WS_PORT"                    # src/verbatim/cli.py:169
        --grpc-port "$GRPC_PORT"                # src/verbatim/cli.py:167 (0: ephemeral)
        --device-id 0                           # src/verbatim/cli.py:256
        --idle-timeout "$IDLE_TIMEOUT_S"        # src/verbatim/cli.py:171
        --ring-seconds "$RING_SECONDS"          # src/verbatim/cli.py:178
        --stop-history-eou-ms "$EOU_MS"         # src/verbatim/cli.py:185
        --language-code "$LANG_CODE"            # src/verbatim/cli.py:249
        --traceback                             # src/verbatim/cli.py:257
    )
    # Deliberately absent, so all three stay off: --biasing (cli.py:214), --decoder-graphs
    # (cli.py:223) and --word-confidence (cli.py:233, default off). Each is a separate arm
    # whose digests do not compare with this one.
}

build_gate() {  # $1 = churn | const, $2 = record path, $3 = finals path; sets GATE_ARGV
    local occupancy=$1 record=$2 finals=$3
    GATE_ARGV=(
        "$BENCH" invariance                     # subcommand: bench/src/verbatim_bench/cli.py:199
        --endpoint "ws://$HOST:$WS_PORT/v1/stream"  # bench/src/verbatim_bench/cli.py:206
        --manifest "$MANIFEST"                  # bench/src/verbatim_bench/cli.py:208
        --chunk "$CHUNK"                        # bench/src/verbatim_bench/cli.py:220
        --seed "$SEED"                          # bench/src/verbatim_bench/cli.py:219
        --lang "$LANG_CODE"                     # bench/src/verbatim_bench/cli.py:245
        --max "$MAX_LEVEL"                      # bench/src/verbatim_bench/cli.py:236
        --out "$record"                         # bench/src/verbatim_bench/cli.py:272
        --finals-out "$finals"                  # bench/src/verbatim_bench/cli.py:275, the transcripts behind the digests
    )
    if [[ "$occupancy" == churn ]]; then
        GATE_ARGV+=(--churn-period-s "$CHURN_S")  # bench/src/verbatim_bench/cli.py:222
    fi
    # Deliberately absent: --phrases (bench cli.py:247) and --control-clips (bench
    # cli.py:262), the biasing arm; --synthetic (bench cli.py:210), which replaces the
    # corpus with noise.
}

arm_settings() {  # $1 = arm; sets ARM_PADDING and ARM_OCCUPANCY
    case "$1" in
        fixed-churn)  ARM_PADDING=fixed;  ARM_OCCUPANCY=churn ;;
        ragged-churn) ARM_PADDING=ragged; ARM_OCCUPANCY=churn ;;
        fixed-const)  ARM_PADDING=fixed;  ARM_OCCUPANCY=const ;;
        ragged-const) ARM_PADDING=ragged; ARM_OCCUPANCY=const ;;
        *) die "unknown arm '$1'; the arms are ${ALL_ARMS[*]}" ;;
    esac
}

resolve_hf_cache() {  # sets HF_HUB_DIR and MODEL_CACHE the way huggingface_hub resolves them
    HF_HUB_DIR=$(HF_HUB_OFFLINE=1 "$PY" -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)') \
        || die "huggingface_hub could not say where its cache is"
    MODEL_CACHE="$HF_HUB_DIR/models--${MODEL//\//--}"
}

# ---------------------------------------------------------------------------------------
# What the server's own code builds from these flags, and what its banner says
# ---------------------------------------------------------------------------------------
derive_spec() {  # $1 padding: the JSON on stdout. Always the production pipeline's argv.
    build_serve "$1" "$PRODUCTION_PIPELINE"
    CUDA_VISIBLE_DEVICES= "$PY" -c "$HELPER" derive_spec "${SERVE_ARGV[@]:1}"
}

check_banner() {  # $1 padding, $2 log file, $3 nemo | fake. Prints each failure; returns 1 on any.
    # The banner echoes the flags (src/verbatim/cli.py:356-358 and 394-430, _banner at
    # 539-555), so a match says THE FLAGS REACHED THE SERVER, not what the loaded model
    # does; what the built model does is /readyz "observed" (require_facts). Only NeMo's
    # own look-ahead warning here comes from the model build. Each absence is paired with
    # presences on the same log, so an empty or truncated log cannot pass.
    local padding=$1 log_file=$2 kind=$3
    local -a bad=()
    has() { grep -qF -- "$1" "$log_file"; }
    [[ -s "$log_file" ]] || bad+=("the log $log_file is empty or missing")
    has "admission    bucket $BUCKET streams, UNCALIBRATED" \
        || bad+=("the flags did not reach the server: no 'admission    bucket $BUCKET' line (cli.py:544)")
    has "NeMo slots: bucket $BUCKET," || bad+=("the flags did not reach the server: no 'NeMo slots: bucket $BUCKET' line (cli.py:550)")
    if has "$NEMO_LOOKAHEAD_WARNING"; then
        bad+=("NeMo warned that the attention context is not one this checkpoint supports: '$NEMO_LOOKAHEAD_WARNING' (conformer_encoder.py:1000-1004)")
    fi
    if [[ "$kind" == fake ]]; then
        has "pipeline     FAKE" || bad+=("test mode is not serving the fake")
    else
        has "checkpoint   $MODEL" || bad+=("the flags did not reach the server: the checkpoint line (cli.py:546) does not name $MODEL")
        has "chunk mode   $CHUNK_MS ms (att_context_size [$ATT_LEFT, $ATT_RIGHT])" \
            || bad+=("the flags did not reach the server: no 'att_context_size [$ATT_LEFT, $ATT_RIGHT]' at $CHUNK_MS ms (cli.py:406)")
        has "graphs       EAGER encoder step, by --eager" \
            || bad+=("the flags did not reach the server: the encoder step is not eager (cli.py:394-398)")
        has "precision    $DTYPE" || bad+=("the flags did not reach the server: precision is not $DTYPE (cli.py:526)")
        has "decoder      CUDA graphs ON" && bad+=("decoder CUDA graphs are ON (cli.py:421); that is a separate arm")
        has "biasing      ON" && bad+=("biasing is ON (cli.py:453); that is a separate arm")
        if [[ "$padding" == ragged ]]; then
            has "padding      RAGGED" || bad+=("the ragged arm's server does not say RAGGED (cli.py:410)")
        else
            has "padding      RAGGED" && bad+=("the fixed arm's server says RAGGED (cli.py:410)")
        fi
    fi
    (( ${#bad[@]} == 0 )) && return 0
    printf '%s\n' "${bad[@]}"
    return 1
}

# ---------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------
tcp_accepts() {  # $1 host, $2 port: 0 only when a TCP connect to it completes
    timeout 3 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "$1" "$2" 2>/dev/null
}

port_listeners() {  # every listening TCP socket on the port, any address, with owner
    ss -H -ltnp "sport = :$1" 2>/dev/null || true
}

card_apps() {  # compute processes on card 3, one "pid, name, MiB" line each
    nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits
}

card_pids() {  # only the lines that name a process; nvidia-smi prints nothing (or a notice) when idle
    awk -F', ' '$1 ~ /^[0-9]+$/'
}

foreign_apps() {  # $1 server pid; stdin: card_pids lines. The lines that are not the server.
    awk -F', ' -v pid="$1" '$1 != pid'
}

require_card_idle() {  # waits up to CARD_IDLE_WAIT_S for a previous arm's context to go, then refuses
    local apps waited=0
    while :; do
        apps=$(card_apps) || die "nvidia-smi could not list card $GPU_INDEX's compute processes"
        apps=$(card_pids <<<"$apps")
        [[ -z "$apps" ]] && return 0
        if (( waited >= CARD_IDLE_WAIT_S )); then
            die "card $GPU_INDEX is in use; refusing to start (the current measurement?):
$apps"
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
}

require_port_free() {
    local listeners
    listeners=$(port_listeners "$WS_PORT")
    if [[ -n "$listeners" ]] || tcp_accepts "$HOST" "$WS_PORT"; then
        die "something already accepts on :$WS_PORT, so a successful connect would prove nothing about the server this runbook starts; free it or set RUNBOOK_WS_PORT:
$listeners"
    fi
}

# What preflight recorded; code_changes compares against these.
PREFLIGHT_HEAD=""
RUNBOOK_SHA=""

code_changes() {  # one line per way the code under test, or this runbook, differs from preflight's record
    local head dirty sha
    head=$(git -C "$CODE_GIT_DIR" rev-parse HEAD 2>/dev/null || echo "<unreadable>")
    [[ "$head" == "$PREFLIGHT_HEAD" ]] || echo "HEAD is $head, not the $PREFLIGHT_HEAD preflight recorded"
    dirty=$(git -C "$CODE_GIT_DIR" --no-optional-locks status --porcelain --untracked-files=all -- src bench/src 2>&1) \
        || dirty="git status failed: $dirty"
    [[ -z "$dirty" ]] || echo "uncommitted changes under src/ or bench/src/: $(tr '\n' ' ' <<<"$dirty")"
    sha=$(sha256sum < "$SELF" | cut -d' ' -f1)
    [[ "$sha" == "$RUNBOOK_SHA" ]] || echo "this runbook now hashes to $sha, not the $RUNBOOK_SHA it started as"
}

SERVER_PID=""
MONITOR_PID=""
GATE_PID=""

server_alive() { [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; }

fail_server() {  # $1 log, $2 message
    printf '[runbook] last 60 lines of %s:\n' "$1" >&2
    tail -n 60 "$1" >&2 || true
    die "$2"
}

wait_for_accept() {  # $1 server log. A connect that completes, not a log line.
    local log_file=$1 deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
    while :; do
        if ! server_alive; then
            local rc=0
            wait "$SERVER_PID" || rc=$?
            SERVER_PID=""
            fail_server "$log_file" "the server exited with code $rc before accepting on $HOST:$WS_PORT"
        fi
        if tcp_accepts "$HOST" "$WS_PORT"; then
            return 0
        fi
        if (( $(date +%s) >= deadline )); then
            fail_server "$log_file" "no TCP accept on $HOST:$WS_PORT within ${READY_TIMEOUT_S} s"
        fi
        sleep 2
    done
}

require_listener_is_ours() {
    local listeners
    listeners=$(port_listeners "$WS_PORT")
    if [[ "$listeners" != *"pid=$SERVER_PID,"* ]]; then
        die "the socket accepting on :$WS_PORT does not belong to the server just started (pid $SERVER_PID):
$listeners"
    fi
}

require_server_environment() {  # $1 arm dir: what the server process actually inherited, recorded and checked
    local dir=$1 seen
    seen=$(helper environ "$SERVER_PID" "$dir/server-environ.json") \
        || die "could not read the server's environment from /proc/$SERVER_PID/environ"
    [[ "$seen" == "$PYTHONPATH" ]] \
        || die "the server runs with PYTHONPATH='$seen', not '$PYTHONPATH': it may not be running this repository's code"
}

wait_for_ready() {  # $1 arm dir: /readyz 200, i.e. engine started and one tick completed (health.py HealthReporter._readiness)
    local dir=$1 status deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
    while :; do
        server_alive || fail_server "$dir/server.log" "the server exited while waiting for /readyz"
        status=$(helper http "http://$HOST:$WS_PORT/readyz" "$dir/readyz-before.json")
        [[ "$status" == 200 ]] && return 0
        if (( $(date +%s) >= deadline )); then
            fail_server "$dir/server.log" "/readyz never answered 200 within ${READY_TIMEOUT_S} s (last status $status)"
        fi
        sleep 2
    done
}

facts_expect() {  # what every arm's /readyz and /admission must report, as JSON on stdout
    # The settings keys echo the flags; "observed" is read off the built model and decoder
    # (contract C5, src/verbatim/pipelines/observed.py), so it is what the arm IS. The
    # padding is not on /readyz; the banner check reads it. The fake serves no NeMo
    # precision or execution, and reports null under "observed" (facts() accepts that from
    # the fake only), so test mode changes those two settings keys and nothing else. "code"
    # (contract C7) is the same in both modes: the fake is served by the same server code.
    local precision=$DTYPE execution=eager code
    if [[ "$TEST_FAKE" == 1 ]]; then
        precision=none execution=fake
    fi
    code=$("$PY" -c 'import json, sys; print(json.dumps({"verbatim_path": sys.argv[1], "bench_path": sys.argv[2]}))' \
        "$SERVER_VERBATIM_PATH" "$SERVER_BENCH_PATH")
    helper obj "model=$MODEL" "pipeline=$PIPELINE" "chunk_ms=#$CHUNK_MS" "precision=$precision" \
        "execution=$execution" "biasing=#false" "word_confidence=off" "bucket=#$BUCKET" \
        "observed=#{\"att_context_size\": [$ATT_LEFT, $ATT_RIGHT], \"decoder_step_confidence\": false, \"decoder_graphs\": false}" \
        "code=#$code"
}

require_facts() {  # $1 padding, $2 arm dir
    local padding=$1 dir=$2 expect status bad kind=nemo
    status=$(helper http "http://$HOST:$WS_PORT/admission" "$dir/admission-before.json")
    [[ "$status" == 200 ]] || die "/admission answered $status"
    if [[ "$TEST_FAKE" == 1 ]]; then
        kind=fake
    fi
    expect=$(facts_expect)
    printf '%s\n' "$expect" > "$dir/facts-expected.json"
    helper facts "$dir/readyz-before.json" "$dir/admission-before.json" "$expect" > "$dir/facts-check.txt" \
        || die "the server is not what this arm asked for: $(cat "$dir/facts-check.txt")"
    bad=$(check_banner "$padding" "$dir/server.log" "$kind") || die "banner: $bad"
}

require_on_card() {  # $1 arm dir: the server's own pid holds a context on card 3, and nothing else does
    local dir=$1 apps foreign
    apps=$(card_apps) || die "nvidia-smi could not list card $GPU_INDEX's compute processes"
    printf '%s\n' "$apps" > "$dir/card-apps-before.txt"
    apps=$(card_pids <<<"$apps")
    if ! awk -F', ' -v pid="$SERVER_PID" '$1 == pid {found = 1} END {exit !found}' <<<"$apps"; then
        die "nvidia-smi does not show the server (pid $SERVER_PID) on card $GPU_INDEX, so where it runs is unobserved:
$apps"
    fi
    foreign=$(foreign_apps "$SERVER_PID" <<<"$apps")
    if [[ -n "$foreign" ]]; then
        die "a process other than the server (pid $SERVER_PID) holds card $GPU_INDEX; refusing to measure on a shared card:
$foreign"
    fi
}

run_smoke() {  # $1 arm dir: dies (exit 2) when the long gate could not keep its verdict
    local dir=$1 rc=0
    log "smoke: about ${SMOKE_SOLO_S} s of audio as one session, then $SMOKE_BURST at once; /admission read throughout"
    "$PY" -c "$HELPER" smoke "ws://$HOST:$WS_PORT/v1/stream" "http://$HOST:$WS_PORT/admission" \
        "$MANIFEST" "$CHUNK" "$LANG_CODE" "$SMOKE_SOLO_S" "$SMOKE_BURST" "$dir/smoke.json" \
        > "$dir/smoke.log" 2>&1 || rc=$?
    (( rc == 0 )) || fail_server "$dir/smoke.log" "the smoke did not complete (exit $rc)"
    helper smoke_check "$dir/smoke.json" "$BUDGET_MS" > "$dir/smoke-check.json" \
        || die "smoke: stopping before the long gate, whose verdict this server would lose:
$(helper get "$dir/smoke-check.json" problems)"
    log "smoke: p95_tick_ms $(helper get "$dir/smoke-check.json" max_p95_tick_ms) ms (budget $BUDGET_MS), nothing refused"
}

start_monitor() {  # $1 arm dir, $2 gate pid, $3 refused_total when the gate started
    "$PY" -c "$HELPER" monitor "http://$HOST:$WS_PORT/admission" "$1/admission-monitor.jsonl" \
        "$2" "$MONITOR_EVERY_S" "$3" "$1/monitor-abort.txt" "$SERVER_PID" "$GPU_INDEX" \
        "$CARD_QUERY_TIMEOUT_S" "$ADMISSION_READ_TIMEOUT_S" &
    MONITOR_PID=$!
}

MONITOR_EXIT=""
MONITOR_END=""
finish_monitor() {  # after the gate: the monitor sees the gate gone and exits 0 by itself; records how it ended
    local bound ticks=0 limit rc=0
    MONITOR_EXIT="" MONITOR_END=""
    [[ -n "$MONITOR_PID" ]] || { MONITOR_END="no monitor was started"; return 0; }
    bound=$(helper gap_bound "$MONITOR_EVERY_S" "$CARD_QUERY_TIMEOUT_S" "$ADMISSION_READ_TIMEOUT_S")
    limit=$(awk -v b="$bound" 'BEGIN { printf "%d\n", int(b * 10) + 1 }')
    while kill -0 "$MONITOR_PID" 2>/dev/null && (( ticks < limit )); do
        sleep 0.1
        ticks=$(( ticks + 1 ))
    done
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        MONITOR_END="still running $bound s after the gate ended; stopped by the runbook"
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    wait "$MONITOR_PID" 2>/dev/null || rc=$?
    MONITOR_EXIT=$rc
    MONITOR_PID=""
}

stop_monitor() {  # cleanup only, when the run is stopping anyway: nothing is read from it
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

stop_gate() {
    if [[ -n "$GATE_PID" ]]; then
        kill -TERM "$GATE_PID" 2>/dev/null || true
        wait "$GATE_PID" 2>/dev/null || true
        GATE_PID=""
    fi
}

SERVER_EXIT=""
stop_server() {  # SIGTERM by pid (the server shuts down on it, cli.py:569-571), SIGKILL after STOP_TIMEOUT_S
    [[ -n "$SERVER_PID" ]] || return 0
    local waited=0 rc=0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        while kill -0 "$SERVER_PID" 2>/dev/null; do
            if (( waited == STOP_TIMEOUT_S )); then
                log "server $SERVER_PID ignored SIGTERM for ${STOP_TIMEOUT_S} s; SIGKILL"
                kill -KILL "$SERVER_PID" 2>/dev/null || true
            fi
            if (( waited > STOP_TIMEOUT_S + 30 )); then
                die "server $SERVER_PID survived SIGKILL; card $GPU_INDEX may still be held"
            fi
            sleep 1
            waited=$(( waited + 1 ))
        done
    fi
    wait "$SERVER_PID" 2>/dev/null || rc=$?
    SERVER_EXIT=$rc
    SERVER_PID=""
}

# Whatever stops the run, the server is stopped; and when this invocation had finished at
# least one arm, those arms are summarised, so a fixed-arm divergence found before a later
# arm failed still exits 1.
RUN_DIR=""
CURRENT_ARM=""
CURRENT_ARM_DIR=""
COMPLETED_ARMS=0
SUMMARY_DONE=0
cleanup() {
    local rc=$? summary_rc=0 reason
    trap - ERR
    set +e
    stop_gate
    stop_monitor
    if [[ -n "$SERVER_PID" ]]; then
        log "cleanup: stopping server $SERVER_PID"
        stop_server
    fi
    if (( rc != 0 && ! SUMMARY_DONE && COMPLETED_ARMS > 0 )); then
        reason="${CURRENT_ARM:+arm $CURRENT_ARM: }${DIE_MSG:-exit $rc}"
        [[ -n "$CURRENT_ARM_DIR" && -d "$CURRENT_ARM_DIR" ]] && printf '%s\n' "$reason" > "$CURRENT_ARM_DIR/fatal.txt"
        log "the run stopped (exit $rc) after $COMPLETED_ARMS finished arm(s); summarising the run directory"
        helper summary "$RUN_DIR" "$reason"
        summary_rc=$?
        (( summary_rc == 1 )) && rc=1
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------------------
# Once per invocation: what the runbook, the code, the corpus, the model and the card are
# ---------------------------------------------------------------------------------------
preflight() {  # $1 preflight dir
    local dir=$1 corpus_json gpu_line rev resolved blob actual model_file dirty in_head self_blob head_blob
    for tool in "$PY" "$VERBATIM" "$BENCH"; do
        [[ -x "$tool" ]] || die "not executable: $tool"
    done
    for tool in ss timeout sha256sum realpath git nvidia-smi; do
        command -v "$tool" >/dev/null || die "missing tool: $tool"
    done
    if [[ "$TEST_FAKE" == 1 ]] && ! grep -q "runbook-test-stub" "$(command -v nvidia-smi)"; then
        die "test mode needs a stub nvidia-smi first on PATH (one carrying 'runbook-test-stub'), not $(command -v nvidia-smi)"
    fi

    # This runbook, exactly as it ran: its hash and a copy, and whether HEAD holds it.
    sha256sum "$SELF" > "$dir/runbook.sha256"
    RUNBOOK_SHA=$(cut -d' ' -f1 < "$dir/runbook.sha256")
    cp -- "$SELF" "$dir/runbook.sh"
    self_blob=$(git -C "$REPO" hash-object -- "$SELF" 2>/dev/null || true)
    head_blob=$(git -C "$REPO" rev-parse -q --verify "HEAD:scripts/$(basename "$SELF")" 2>/dev/null || true)
    in_head=false
    if [[ -n "$self_blob" && "$self_blob" == "$head_blob" ]]; then
        in_head=true
    fi
    helper obj "path=$SELF" "sha256=$RUNBOOK_SHA" "committed_at_head=#$in_head" > "$dir/runbook.json"

    # The code under test is this repository's: both packages import from its src/ and
    # bench/src/ under the PYTHONPATH every process here gets ...
    helper code_origin "$REPO" > "$dir/code.json" \
        || die "verbatim / verbatim_bench do not import from $REPO (see $dir/code.json)"

    # ... the corpus is the published one ...
    corpus_json=$(helper corpus "$MANIFEST" "$LOCAL_PREFIX" "$PUBLISHED_PREFIX")
    printf '%s\n' "$corpus_json" > "$dir/corpus.json"
    if [[ "$TEST_FAKE" != 1 ]]; then
        helper corpus_check "$dir/corpus.json" "$PUBLISHED_CORPUS_ID" "$UTTERANCES" > "$dir/corpus-check.txt" \
            || die "the corpus is not the published one: $(cat "$dir/corpus-check.txt")"
    fi

    # ... and the code is committed, so the HEAD recorded here names the code that ran.
    PREFLIGHT_HEAD=$(git -C "$CODE_GIT_DIR" rev-parse HEAD 2>/dev/null) || die "$CODE_GIT_DIR is not a git repository"
    printf '%s\n' "$PREFLIGHT_HEAD" > "$dir/git-head.txt"
    git -C "$CODE_GIT_DIR" --no-optional-locks status --porcelain > "$dir/git-status.txt"
    dirty=$(git -C "$CODE_GIT_DIR" --no-optional-locks status --porcelain --untracked-files=all -- src bench/src)
    [[ -z "$dirty" ]] || die "uncommitted changes under src/ or bench/src/ in $CODE_GIT_DIR; commit them so the recorded HEAD is the code that ran:
$dirty"

    # The checkpoint NeMo will load: it asks huggingface_hub's try_to_load_from_cache
    # (nemo/core/classes/common.py:1154-1158), so this asks the same function, in the same
    # cache, and requires it to name the pinned revision's file, whose bytes hash to the
    # blob name they are stored under.
    rev=$(cat "$MODEL_CACHE/refs/main" 2>/dev/null || true)
    [[ "$rev" == "$MODEL_REVISION" ]] \
        || die "the HF cache's refs/main for $MODEL is '$rev', not the pinned $MODEL_REVISION ($MODEL_CACHE)"
    model_file="$MODEL_CACHE/snapshots/$MODEL_REVISION/$MODEL_FILENAME"
    resolved=$(HF_HUB_CACHE="$HF_HUB_DIR" HF_HUB_OFFLINE=1 "$PY" -c '
import sys
from huggingface_hub import try_to_load_from_cache
path = try_to_load_from_cache(repo_id=sys.argv[1], filename=sys.argv[2])
print(path if isinstance(path, str) else "")' "$MODEL" "$MODEL_FILENAME") || die "huggingface_hub could not resolve $MODEL"
    [[ "$resolved" == "$model_file" ]] \
        || die "huggingface_hub resolves $MODEL to '$resolved', not $model_file: NeMo would load something else"
    [[ -f "$model_file" ]] || die "no checkpoint at $model_file"
    # The blob name is the FIRST link's target: huggingface_hub names blobs/<sha256> from the
    # LFS sha256 it downloaded against. With Xet storage that blob is itself a link into a
    # content-addressed store (blobs/ce/<xet hash>), so following every link would compare
    # against the Xet hash instead.
    link=$(readlink "$model_file") || die "$model_file is not a link into the cache's blobs"
    blob=$(basename "$link")
    [[ "$blob" =~ ^[0-9a-f]{64}$ ]] || die "the snapshot links to '$link', not to a blob named for a sha256"
    [[ "$(realpath -m "$(dirname "$model_file")/$link")" == "$(realpath -m "$MODEL_CACHE/blobs/$blob")" ]] \
        || die "the snapshot links to '$link', outside $MODEL_CACHE/blobs"
    log "hashing the checkpoint ($(du -Lh "$model_file" | cut -f1))"
    actual=$(sha256sum < "$model_file" | cut -d' ' -f1)
    [[ "$actual" == "$blob" ]] \
        || die "the cached checkpoint hashes to $actual, not to its blob name $blob"
    helper obj "model=$MODEL" "revision=$rev" "hf_hub_cache=$HF_HUB_DIR" "nemo_file=$model_file" \
        "nemo_sha256=$actual" > "$dir/model.json"

    # The card: the same physical card the step-1 probe used.
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version --format=csv,noheader \
        > "$dir/gpus.txt" || die "nvidia-smi failed"
    gpu_line=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader) \
        || die "nvidia-smi -i $GPU_INDEX failed"
    [[ "$gpu_line" == "$EXPECTED_GPU_UUID" ]] \
        || die "card $GPU_INDEX is $gpu_line, not $EXPECTED_GPU_UUID (the card the step-1 probe ran on)"
}

# ---------------------------------------------------------------------------------------
# One arm: derive, start, wait for a real accept, check, smoke, gate, stop, read the record
# ---------------------------------------------------------------------------------------
BUDGET_MS=""
run_arm() {  # $1 arm, $2 run dir
    local arm=$1 run_dir=$2 dir record finals expect gate_rc t_start t_accept t_ready t_gate0 t_gate1 t_stop
    local operational=() invalidating=() rec_rc=0 churn_json refused0 refused1 level1 apps foreign spec_expect
    local changed monitor_problems ops_json inv_json
    local -a found=()
    arm_settings "$arm"
    dir="$run_dir/$arm"
    record="$run_dir/invariance-$arm.json"
    finals="$run_dir/finals-$arm.json"
    [[ -e "$dir" || -e "$record" || -e "$finals" ]] && die "refusing to overwrite $dir, $record or $finals"
    mkdir -p "$dir"
    CURRENT_ARM=$arm
    CURRENT_ARM_DIR=$dir

    # What the server's code builds from this arm's flags, before anything is started.
    derive_spec "$ARM_PADDING" > "$dir/serve-spec.json" \
        || die "the server's own code refused this arm's flags: $(cat "$dir/serve-spec.json")"
    local production_argv=("${SERVE_ARGV[@]}")
    spec_expect=$(printf '{"model":"%s","att_context":[%d,%d],"compute_dtype":"%s","use_cuda_graphs":false,"use_cuda_graph_decoder":false,"enable_per_stream_biasing":false,"batch_size":%d}' \
        "$MODEL" "$ATT_LEFT" "$ATT_RIGHT" "$DTYPE" "$BUCKET")
    helper spec_check "$dir/serve-spec.json" "$spec_expect" > "$dir/spec-check.txt" \
        || die "the server's code would not build this arm: $(cat "$dir/spec-check.txt")"
    BUDGET_MS=$(helper get "$dir/serve-spec.json" budget_ms)

    build_serve "$ARM_PADDING"
    build_gate "$ARM_OCCUPANCY" "$record" "$finals"
    if [[ "$TEST_FAKE" != 1 && "${production_argv[*]}" != "${SERVE_ARGV[*]}" ]]; then
        die "the served argv is not the argv the spec was derived from"
    fi
    helper argv "$dir/serve-env.json" "${SERVE_ENV[@]}"
    helper argv "$dir/serve-argv.json" "${SERVE_ARGV[@]}"
    helper argv "$dir/gate-argv.json" "${GATE_ARGV[@]}"

    require_card_idle
    require_port_free
    changed=$(code_changes)
    [[ -z "$changed" ]] || die "the code under test changed since preflight, before arm $arm's server started: $changed"

    log "arm $arm: starting the server (padding $ARM_PADDING, bucket $BUCKET, $CHUNK, $DTYPE, eager, matmul precision $(helper get "$dir/serve-spec.json" spec matmul_precision))"
    t_start=$(now)
    env "${SERVE_ENV[@]}" "${SERVE_ARGV[@]}" > "$dir/server.log" 2>&1 &
    SERVER_PID=$!
    printf '%s\n' "$SERVER_PID" > "$dir/server.pid"

    wait_for_accept "$dir/server.log"
    t_accept=$(now)
    require_listener_is_ours
    require_server_environment "$dir"
    log "arm $arm: pid $SERVER_PID accepts on $HOST:$WS_PORT; waiting for /readyz"
    wait_for_ready "$dir"
    t_ready=$(now)
    require_facts "$ARM_PADDING" "$dir"
    require_on_card "$dir"
    run_smoke "$dir"

    [[ "$(helper http "http://$HOST:$WS_PORT/admission" "$dir/admission-gate-start.json")" == 200 ]] \
        || die "/admission did not answer before the gate"
    refused0=$(helper get "$dir/admission-gate-start.json" refused_total)
    is_int "$refused0" || die "/admission gave no refused_total before the gate"

    log "arm $arm: running the gate (levels 1/32a/32b/$MAX_LEVEL$([[ $ARM_OCCUPANCY == churn ]] && printf ', max churned %s s' "$CHURN_S")); output in $dir/gate.log"
    t_gate0=$(now)
    gate_rc=0
    timeout --foreground --signal=TERM --kill-after=60 "$GATE_TIMEOUT_S" "${GATE_ARGV[@]}" > "$dir/gate.log" 2>&1 &
    GATE_PID=$!
    start_monitor "$dir" "$GATE_PID" "$refused0"
    wait "$GATE_PID" || gate_rc=$?
    GATE_PID=""
    t_gate1=$(now)
    finish_monitor
    log "arm $arm: the gate exited $gate_rc; the monitor exited $MONITOR_EXIT${MONITOR_END:+ ($MONITOR_END)}"

    # The code under test, and this runbook, must still be what preflight recorded.
    changed=$(code_changes)
    if [[ -n "$changed" ]]; then
        invalidating+=("the code under test changed during the arm: $changed")
    fi
    if [[ -s "$dir/monitor-abort.txt" ]]; then
        operational+=("the monitor stopped the gate: $(head -n 1 "$dir/monitor-abort.txt")")
    fi
    # What the monitor's polls saw, and failed to see, between the gate's start and end
    # (monitor_summary's "problems"): each one is an operational problem of this arm.
    helper monitor_summary "$dir/admission-monitor.jsonl" "$GPU_INDEX" "$MONITOR_EVERY_S" \
        "$CARD_QUERY_TIMEOUT_S" "$ADMISSION_READ_TIMEOUT_S" "$t_gate0" "$t_gate1" \
        "$MONITOR_EXIT" "$MONITOR_END" > "$dir/monitor-summary.json"
    monitor_problems=$(helper lines "$dir/monitor-summary.json" problems)
    if [[ -n "$monitor_problems" ]]; then
        mapfile -t found <<<"$monitor_problems"
        operational+=("${found[@]}")
    fi

    if server_alive; then
        helper http "http://$HOST:$WS_PORT/readyz" "$dir/readyz-after.json" > /dev/null
        helper http "http://$HOST:$WS_PORT/admission" "$dir/admission-after.json" > /dev/null
        refused1=$(helper get "$dir/admission-after.json" refused_total)
        level1=$(helper get "$dir/admission-after.json" degradation_level)
        if ! is_int "$refused1"; then
            operational+=("no /admission reading after the gate")
        elif (( refused1 > refused0 )); then
            operational+=("refused_total rose during the gate ($refused0 -> $refused1): sessions were refused, so a level is incomplete")
        fi
        if is_int "$level1" && (( level1 >= 1 )); then
            operational+=("degradation_level $level1 after the gate: the server was holding admissions")
        fi
        if apps=$(card_apps); then
            printf '%s\n' "$apps" > "$dir/card-apps-after.txt"
            apps=$(card_pids <<<"$apps")
            foreign=$(foreign_apps "$SERVER_PID" <<<"$apps")
            [[ -z "$foreign" ]] || operational+=("nvidia-smi showed a process other than the server on card $GPU_INDEX after the gate: $(tr '\n' ';' <<<"$foreign")")
            awk -F', ' -v pid="$SERVER_PID" '$1 == pid {found = 1} END {exit !found}' <<<"$apps" \
                || operational+=("nvidia-smi no longer shows the server on card $GPU_INDEX after the gate")
        else
            operational+=("nvidia-smi could not list card $GPU_INDEX's compute processes after the gate")
        fi
    else
        operational+=("the server was not running when the runbook read it after the gate")
    fi
    [[ "$gate_rc" == 124 ]] && operational+=("the gate hit the ${GATE_TIMEOUT_S} s runbook limit")
    stop_server
    t_stop=$(now)
    grep -qF "[verbatim] stopped" "$dir/server.log" || log "note: the server log has no clean '[verbatim] stopped' line (exit $SERVER_EXIT)"
    if grep -qF -- "$NEMO_LOOKAHEAD_WARNING" "$dir/server.log"; then
        invalidating+=("NeMo warned during the run that the attention context is not a supported look-ahead")
    fi

    if [[ "$ARM_OCCUPANCY" == churn ]]; then churn_json="$CHURN_S"; else churn_json=null; fi
    expect=$(helper obj "max=#$MAX_LEVEL" "churn_period_s=#$churn_json" "chunk_ms=#$CHUNK_MS" \
        "utterances=#$(helper get "$PREFLIGHT_DIR/corpus.json" utterances)" \
        "corpus_id=$(helper get "$PREFLIGHT_DIR/corpus.json" local_id)" \
        "gate_exit=#$( [[ "$gate_rc" =~ ^[012]$ ]] && echo "$gate_rc" || echo null)" \
        "finals_path=$finals" "manifest=$MANIFEST" "level1_wer_max=#$LEVEL1_WER_MAX")
    helper record "$record" "$expect" > "$dir/record-summary.json" || rec_rc=$?

    ops_json=$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' ${operational[@]+"${operational[@]}"})
    inv_json=$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' ${invalidating[@]+"${invalidating[@]}"})
    helper arm "$dir/arm.json" \
        arm="$arm" padding="$ARM_PADDING" occupancy="$ARM_OCCUPANCY" \
        record="$record" finals="$finals" preflight="$PREFLIGHT_DIR" \
        "runbook=@$PREFLIGHT_DIR/runbook.json" "git_head=$PREFLIGHT_HEAD" "code=@$PREFLIGHT_DIR/code.json" \
        "server_pid=#$(cat "$dir/server.pid")" \
        "gate_exit=#$gate_rc" "server_exit=#${SERVER_EXIT:-null}" "record_check_exit=#$rec_rc" \
        "monitor_exit=#${MONITOR_EXIT:-null}" \
        "t_start=#$t_start" "t_accept=#$t_accept" "t_ready=#$t_ready" \
        "t_gate_start=#$t_gate0" "t_gate_end=#$t_gate1" "t_stopped=#$t_stop" \
        "serve_env=@$dir/serve-env.json" "server_environ=@$dir/server-environ.json" \
        "serve_argv=@$dir/serve-argv.json" \
        "gate_argv=@$dir/gate-argv.json" "serve_spec=@$dir/serve-spec.json" \
        "readyz_before=@$dir/readyz-before.json" "admission_before=@$dir/admission-before.json" \
        "smoke=@$dir/smoke-check.json" "admission_gate_start=@$dir/admission-gate-start.json" \
        "monitor_summary=@$dir/monitor-summary.json" \
        "readyz_after=@$dir/readyz-after.json" "admission_after=@$dir/admission-after.json" \
        "record_summary=@$dir/record-summary.json" "operational_problems=#$ops_json" \
        "invalidating_problems=#$inv_json"
    COMPLETED_ARMS=$(( COMPLETED_ARMS + 1 ))
    log "arm $arm: record $( [[ -f $record ]] && echo "$record" || echo 'NOT WRITTEN'); summary $dir/arm.json"
    tail -n 3 "$dir/gate.log" | sed 's/^/[gate] /'
    # A change during this arm took its verdict away above. The next arm's own check before
    # its server starts stops the run if the change is still there.
    CURRENT_ARM=""
    CURRENT_ARM_DIR=""
}

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
DRY_RUN=0
ARMS=()
case "${1:-}" in
    -h|--help)
        sed -n '2,/^# END OF HELP$/p' "$0" | sed '$d'
        exit 0
        ;;
    --derive-spec)
        [[ $# == 2 && ( "$2" == fixed || "$2" == ragged ) ]] || die "usage: --derive-spec fixed|ragged"
        resolve_hf_cache
        derive_spec "$2"
        exit $?
        ;;
    --check-banner)
        [[ $# == 3 && ( "$2" == fixed || "$2" == ragged ) ]] || die "usage: --check-banner fixed|ragged LOG"
        if check_banner "$2" "$3" nemo; then echo "banner ok"; exit 0; fi
        exit 2
        ;;
    --facts-expect)
        [[ $# == 1 ]] || die "usage: --facts-expect"
        facts_expect
        exit $?
        ;;
esac
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) arm_settings "$arg"; ARMS+=("$arg") ;;
    esac
done
(( ${#ARMS[@]} )) || ARMS=("${ALL_ARMS[@]}")

resolve_hf_cache
readonly HF_HUB_DIR MODEL_CACHE
RUN_ID="${RUNBOOK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUT_DIR/gate-nemotron-card$GPU_INDEX-$RUN_ID"

if (( DRY_RUN )); then
    for arm in "${ARMS[@]}"; do
        arm_settings "$arm"
        build_serve "$ARM_PADDING"
        build_gate "$ARM_OCCUPANCY" "$RUN_DIR/invariance-$arm.json" "$RUN_DIR/finals-$arm.json"
        helper dry "$arm" "$RUN_DIR/invariance-$arm.json" "$RUN_DIR/finals-$arm.json" \
            "${SERVE_ENV[@]}" ::SERVE:: "${SERVE_ARGV[@]}" ::GATE:: "${GATE_ARGV[@]}"
    done
    exit 0
fi

mkdir -p "$RUN_DIR"
log "run directory $RUN_DIR; arms: ${ARMS[*]}; bucket $BUCKET, max level $MAX_LEVEL; EAGER ONLY"
[[ "$TEST_FAKE" == 1 ]] && log "TEST MODE: --pipeline fake on the CPU; nothing here is a measurement"
# Every invocation checks the runbook, code, corpus, checkpoint and card again, into its own
# directory, including one that continues an earlier run directory through RUNBOOK_RUN_ID.
PREFLIGHT_DIR="$RUN_DIR/preflight-$(date -u +%Y%m%dT%H%M%S.%NZ)"
mkdir -p "$PREFLIGHT_DIR"
preflight "$PREFLIGHT_DIR"
readonly PREFLIGHT_DIR PREFLIGHT_HEAD RUNBOOK_SHA
for arm in "${ARMS[@]}"; do
    run_arm "$arm" "$RUN_DIR"
done

status=0
SUMMARY_DONE=1
helper summary "$RUN_DIR" || status=$?
exit "$status"
