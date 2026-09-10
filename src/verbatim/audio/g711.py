# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""``MULAW`` / ``ALAW`` via a 256-entry int16 lookup table. No dependency.

``audioop`` was removed from the standard library in 3.13 (PEP 594) and its PyPI
replacement requires >= 3.13, so depending on it would split the dependency graph
across the supported Python range in order to obtain a table lookup. Telephony
operators are a named constituency; G.711 must work on every supported Python.
"""
