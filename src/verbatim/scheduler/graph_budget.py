# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``buckets x chunk_modes <= max_graphs``, asserted at config time, not discovered at runtime.

NeMo's ``CudaGraphsStreamingEncoderStep`` defaults to ``max_graphs = 8`` and keys a
capture on ``(shape, dtype, device, keep_all_outputs, drop_extra_pre_encoded,
att_context_size, last_channel_cache_size, valid_out_len)``. Once the budget is
full every further shape runs eager for the life of the process.

This module is pure arithmetic over configured bucket lists, so it is testable on
CPU. This module must not depend on the NeMo toolkit or on PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["ConfigError", "assert_graph_budget", "graph_keys_required"]


class ConfigError(Exception):
    """A configuration that the design rules reject at construction time."""


def graph_keys_required(buckets_per_mode: Mapping[int, Sequence[int]]) -> int:
    """One key per (mode, bucket), plus one for the edge shape."""
    return sum(len(buckets) for buckets in buckets_per_mode.values()) + 1


def assert_graph_budget(buckets_per_mode: Mapping[int, Sequence[int]], max_graphs: int = 8) -> None:
    """Raise ConfigError when the budget is exceeded, naming the count and the limit.

    Asserted at configuration time, deliberately: exceeding it at runtime is silent --
    the excess shapes simply never get captured -- and a silent loss of the graph path
    is the failure this design exists to prevent.
    """
    required = graph_keys_required(buckets_per_mode)
    if required > max_graphs:
        raise ConfigError(
            f"graph budget exceeded: {required} capture keys required "
            f"(one per (mode, bucket) plus one for the edge shape) "
            f"but max_graphs={max_graphs}"
        )
