# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Pipeline adapters -- THE ONLY PACKAGE ALLOWED TO ``import nemo``.

``tests/test_import_boundary.py`` walks every other module's AST and fails the
build otherwise. That boundary is the mechanism that keeps "vendor nothing"
honest and makes a NeMo API break a one-directory fix.
"""
