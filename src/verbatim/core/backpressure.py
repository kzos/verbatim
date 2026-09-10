# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Policy for a session whose ring overruns the tick budget.

No audio is ever dropped server-side. When a ring fills, the transport stops
reading that stream: gRPC's HTTP/2 flow control back-pressures the client, and
the WebSocket demo stops ``recv()``-ing on that socket.
"""
