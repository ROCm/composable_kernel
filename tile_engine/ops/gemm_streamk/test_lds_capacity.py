# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for validate_lds_capacity (gemm_streamk)."""

import os
import sys
import unittest

# Make the validation utils importable by inserting the directory into sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gemm_streamk_validation_utils import validate_lds_capacity  # noqa: E402


class TestValidateLdsCapacity(unittest.TestCase):
    def test_128kb_config_passes_gfx950_fails_gfx942(self):
        """gfx950 has 160KB LDS, gfx942 has 64KB."""
        valid_950, _ = validate_lds_capacity(256, 256, 128, "fp16", "fp16", "mem", "gfx950")
        valid_942, _ = validate_lds_capacity(256, 256, 128, "fp16", "fp16", "mem", "gfx942")
        self.assertTrue(valid_950)
        self.assertFalse(valid_942)

    def test_double_buffer_halves_capacity(self):
        """compv4 halves LDS budget. 48KB fits gfx950 (80KB) but not gfx942 (32KB)."""
        valid_950, _ = validate_lds_capacity(256, 128, 64, "fp16", "fp16", "compv4", "gfx950")
        valid_942, _ = validate_lds_capacity(256, 128, 64, "fp16", "fp16", "compv4", "gfx942")
        self.assertTrue(valid_950)
        self.assertFalse(valid_942)
