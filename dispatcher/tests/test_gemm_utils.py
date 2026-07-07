#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for python/gemm_utils.py.

Locks in the bit-level helpers that the TE -> Dispatcher GEMM bridge relies on:
  * bf16 <-> uint16 encoding (round-to-nearest-even), since numpy has no native
    bf16 and the runner carries bf16 as a uint16 bit pattern.
  * dtype / layout parsing from the compiled kernel name, which drives how the
    runner lays out host buffers.

No GPU is touched -- all functions under test are pure host-side logic.
Run: python3 -m pytest tests/test_gemm_utils.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

import numpy as np  # noqa: E402

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    _fp32_to_bf16_u16,
    _bf16_u16_to_fp32,
    _dtype_from_kernel_name,
    _layout_from_kernel_name,
)


class TestBf16Encoding(unittest.TestCase):
    """bf16 = top 16 bits of fp32 with round-to-nearest-even."""

    def test_exactly_representable_roundtrip(self):
        # Values whose low 16 fp32 mantissa bits are zero are exact in bf16.
        exact = np.array([0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 4.0, 256.0],
                         dtype=np.float32)
        out = _bf16_u16_to_fp32(_fp32_to_bf16_u16(exact))
        np.testing.assert_array_equal(out, exact)

    def test_roundtrip_within_bf16_tolerance(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(10000) * 100.0).astype(np.float32)
        out = _bf16_u16_to_fp32(_fp32_to_bf16_u16(x))
        # bf16 has 8 bits of significand -> relative error <= 2^-8.
        rel = np.abs(out - x) / (np.abs(x) + 1e-30)
        self.assertLessEqual(float(rel.max()), 2.0 ** -8)

    def test_round_to_nearest_even_ties(self):
        # Tie halfway between bf16 1.0 (0x3F80, even) and 0x3F81 (odd):
        # fp32 0x3F808000 must round DOWN to the even neighbor 0x3F80.
        tie_down = np.array([0x3F808000], dtype=np.uint32).view(np.float32)
        self.assertEqual(int(_fp32_to_bf16_u16(tie_down)[0]), 0x3F80)
        # Tie halfway between 0x3F81 (odd) and 0x3F82 (even):
        # fp32 0x3F818000 must round UP to the even neighbor 0x3F82.
        tie_up = np.array([0x3F818000], dtype=np.uint32).view(np.float32)
        self.assertEqual(int(_fp32_to_bf16_u16(tie_up)[0]), 0x3F82)

    def test_special_values(self):
        inf = np.array([np.inf, -np.inf], dtype=np.float32)
        out = _bf16_u16_to_fp32(_fp32_to_bf16_u16(inf))
        self.assertTrue(np.isinf(out[0]) and out[0] > 0)
        self.assertTrue(np.isinf(out[1]) and out[1] < 0)

        nan = np.array([np.nan], dtype=np.float32)
        out_nan = _bf16_u16_to_fp32(_fp32_to_bf16_u16(nan))
        self.assertTrue(np.isnan(out_nan[0]))

    def test_dtype_and_size(self):
        u16 = _fp32_to_bf16_u16(np.zeros(4, dtype=np.float32))
        self.assertEqual(u16.dtype, np.uint16)
        self.assertEqual(u16.itemsize, 2)  # must match sizeof(bf16_t) on device


class TestKernelNameParsing(unittest.TestCase):
    """The runner reads dtype + layout straight from the compiled .so name."""

    _NAME = ("gemm_bf16_rcr_compv3_cshuffle_intrawave_"
             "False_False_False_False_64x64x64_4x1x1_16x16x16")

    def test_dtype_from_name(self):
        self.assertEqual(_dtype_from_kernel_name(self._NAME), "bf16")
        self.assertEqual(
            _dtype_from_kernel_name("gemm_fp16_rrr_compv4_cshuffle_intrawave"),
            "fp16",
        )

    def test_dtype_fallback(self):
        # Malformed / single-token name falls back to fp16.
        self.assertEqual(_dtype_from_kernel_name("gemm"), "fp16")

    def test_layout_from_name(self):
        self.assertEqual(_layout_from_kernel_name(self._NAME), "rcr")
        for lay in ("rrr", "ccr", "crr", "rcc"):
            name = f"gemm_fp16_{lay}_compv3_cshuffle_intrawave"
            self.assertEqual(_layout_from_kernel_name(name), lay)

    def test_layout_fallback(self):
        # A token that is not a 3-char r/c string falls back to rcr.
        self.assertEqual(
            _layout_from_kernel_name("gemm_fp16_xyz_compv3"), "rcr"
        )
        self.assertEqual(_layout_from_kernel_name("gemm"), "rcr")


class TestConfigNameContract(unittest.TestCase):
    """GemmKernelConfig.name is the single source of truth tying config ->
    codegen -> runtime; parsing it back must recover dtype and layout."""

    def test_name_roundtrips_through_parsers(self):
        for dtype in ("fp16", "bf16"):
            for la, lb, lc in (("row", "col", "row"),
                               ("row", "row", "row"),
                               ("col", "col", "row"),
                               ("col", "row", "row")):
                cfg = GemmKernelConfig(
                    dtype_a=dtype, dtype_b=dtype, dtype_c=dtype,
                    layout_a=la, layout_b=lb, layout_c=lc,
                )
                name = cfg.name
                self.assertEqual(_dtype_from_kernel_name(name), dtype)
                self.assertEqual(_layout_from_kernel_name(name), cfg.layout)


if __name__ == "__main__":
    unittest.main()
