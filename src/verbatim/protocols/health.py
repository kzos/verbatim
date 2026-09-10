# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``/healthz``, ``/readyz``, ``/metrics``, ``/admission`` on the same :8080 listener.

Served through the ``websockets`` library's ``process_request`` hook -- no web
framework. ``/readyz`` stays red until the model is loaded, every bucket key is
captured, the graph mode is FULL_GRAPH and the invariance self-test has passed.
"""
