# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Generated protobuf and gRPC stubs for the Riva ASR subset. Do not edit by hand.

Regenerate with ``scripts/gen_protos.sh``; CI reruns that script and fails on any
diff, so the committed stubs cannot drift from the committed protos. The protos
themselves are vendored verbatim from ``nvidia-riva/common`` at the SHA pinned in
``proto/riva/UPSTREAM`` and are MIT-licensed -- see ``docs/third_party.md``.

The generated tree under this package mirrors the proto package path
(``riva/proto/``), because protoc emits absolute imports keyed on it. Importing
this package puts that tree on ``sys.path`` so those imports resolve, and
re-exports the four modules so callers never have to::

    from verbatim.protocols.riva._gen import riva_asr_pb2, riva_asr_pb2_grpc

``nvidia-riva-client`` must not be imported into the same process: it registers
the same ``riva/proto/*.proto`` file paths and the same ``nvidia.riva.asr``
symbols in the descriptor pool from a different generation, which is not a
supported configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_GEN_ROOT = str(Path(__file__).resolve().parent)
if _GEN_ROOT not in sys.path:
    sys.path.insert(0, _GEN_ROOT)

from riva.proto import (  # noqa: E402
    riva_asr_pb2,
    riva_asr_pb2_grpc,
    riva_audio_pb2,
    riva_common_pb2,
)

__all__ = [
    "riva_asr_pb2",
    "riva_asr_pb2_grpc",
    "riva_audio_pb2",
    "riva_common_pb2",
]
