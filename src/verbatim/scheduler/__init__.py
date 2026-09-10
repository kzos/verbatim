# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""The tick scheduler -- the project's one original idea.

Keep a churning multi-tenant batch on NeMo's CUDA-graph path: sessions join and
leave every chunk period, but the batch handed to NeMo is always one of a few
fixed bucket sizes whose graphs stay captured, and the ragged first/last steps are
peeled into a small eager side-batch.

This package is maintainers-only by design (07_community.md §0): no contribution
surface requires reading it, and nothing under ``pipelines/``, ``bench/`` or the
protocol conformance cases may import from it.
"""
