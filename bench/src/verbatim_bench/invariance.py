# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The day-45 gate: digest transcripts and timestamps at concurrency 1 / 32a / 32b / max.

Finals are canonicalised -- sorted ``(utterance_id, text, [(word, start_ms,
end_ms)...])`` with timestamps as integer milliseconds exactly as emitted, no
normalisation -- and hashed with SHA-256. The row's ``invariance`` field is true
only if all four hashes are equal.
"""
