# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Verbatim -- a multi-tenant tick-scheduled streaming ASR server.

Verbatim is an Apache-2.0 asynchronous multi-client server for NVIDIA's
cache-aware FastConformer streaming checkpoints, built *on* NeMo's Apache-2.0
``nemo.collections.asr.inference`` pipeline rather than re-implementing it.

Nothing in this package is implemented yet: this is the repository skeleton, and
every module below is a named, importable placeholder whose docstring states what
belongs in it.

Scope discipline (year 1): NeMo cache-aware streaming pipelines only. No Whisper,
no custom kernels, no quantised encoders, no dashboards.
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

__all__ = ["__version__"]
