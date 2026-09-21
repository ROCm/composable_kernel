#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Regression tests for OPERATOR_TILE_CONSTRAINTS in arch_filter.py.

Guards against a malformed constraints table (e.g. an operator entry left
without a body/closing brace), which raises a SyntaxError on import and
breaks all GEMM codegen -- and therefore every downstream build/test that
imports unified_gemm_codegen.

Can be run as:
    python3 tests/test_arch_filter_constraints.py
    ctest -R test_arch_filter_constraints
"""

import sys
import unittest
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

# Importing at all fails if arch_filter.py has a syntax error.
from arch_filter import OPERATOR_TILE_CONSTRAINTS, OperatorType  # noqa: E402

REQUIRED_KEYS = {
    "min_tile_m",
    "min_tile_n",
    "min_tile_k",
    "tile_m_alignment",
    "tile_n_alignment",
    "tile_k_alignment",
}


class TestOperatorTileConstraints(unittest.TestCase):
    """Validate the OPERATOR_TILE_CONSTRAINTS table is well-formed."""

    def test_every_operator_has_an_entry(self):
        """Every OperatorType must have a constraints entry."""
        for op in OperatorType:
            self.assertIn(
                op,
                OPERATOR_TILE_CONSTRAINTS,
                f"{op} is missing from OPERATOR_TILE_CONSTRAINTS",
            )

    def test_every_entry_is_a_well_formed_dict(self):
        """Each entry must be a dict with the required integer keys."""
        for op, constraints in OPERATOR_TILE_CONSTRAINTS.items():
            self.assertIsInstance(
                constraints, dict, f"{op} constraints should be a dict"
            )
            self.assertEqual(
                REQUIRED_KEYS,
                set(constraints.keys()),
                f"{op} constraints keys mismatch",
            )
            for key, value in constraints.items():
                self.assertIsInstance(
                    value, int, f"{op}[{key}] should be an int, got {type(value)}"
                )
                self.assertGreater(value, 0, f"{op}[{key}] should be positive")

    def test_gemm_grouped_entry_present(self):
        """Regression: GEMM_GROUPED must have its own distinct constraints dict."""
        self.assertIn(OperatorType.GEMM_GROUPED, OPERATOR_TILE_CONSTRAINTS)
        grouped = OPERATOR_TILE_CONSTRAINTS[OperatorType.GEMM_GROUPED]
        self.assertEqual(REQUIRED_KEYS, set(grouped.keys()))
        # GEMM_GROUPED and GEMM_STREAMK are separate entries, not a merged one.
        self.assertIn(OperatorType.GEMM_STREAMK, OPERATOR_TILE_CONSTRAINTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
