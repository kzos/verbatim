# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Decode and resample, in worker threads. Never on the event loop or the tick thread.

The default path has no decoder at all: both target plugins hard-code
``LINEAR_PCM`` mono at 16 kHz, so the overwhelmingly common request costs one
buffer view. This package never touches the GPU.
"""
