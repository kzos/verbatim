# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Stateful chunked resampling for the rare non-16 kHz stream.

``soxr.ResampleStream`` with a ``scipy.signal.resample_poly`` fallback. Never on
the GPU: that would add a per-session variable-shape operation to the tick path,
which is precisely what costs CUDA graphs.
"""
