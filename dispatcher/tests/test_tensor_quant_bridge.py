#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the tensor_quant GEMM TileEngine -> Dispatcher bridge.

Locks the byte-exact fp8/bf8 x rcr scope, the arch-derived warp_tile_k trap, and
the config problem defaults for the Old-TE gemm_quant_tensor.cpp instance builder.
No GPU, no hipcc, no Old-TE builder import required.

The config-name prefix / tiles-in-name contract, the byte-exact codegen<->utils
kernel-name contract, and the codegen-JSON projection roundtrip are exercised for
every quant bridge (including this one) by the shared parametrized tests in
test_quant_bridge_shared.py, driven by _quant_bridge_descriptors.py.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_tensor_quant_utils import (  # noqa: E402
    TensorQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    fp8_warp_tile_k_for_arch,
)


class TestScope(unittest.TestCase):
    """gemm_quant_tensor.cpp registers exactly fp8/tensor and bf8/tensor, rcr only."""

    def test_default_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")

    def test_layout_is_rcr(self):
        self.assertEqual(default_fp8_config().layout, "rcr")
        self.assertEqual(default_bf8_config().layout, "rcr")


class TestArchWarpTileK(unittest.TestCase):
    """WarpTileK must be arch-derived (get_k_warp_tile<fp8/bf8, 16>()).

    Hardcoding warp_tile_k=128 on gfx942 compiles but silently outputs
    all-zeros (confirmed on GPU, MI300X): there is no valid 16x16x128 fp8/bf8
    warp-gemm on gfx942. The correct value there is 32; gfx950 uses 128.
    """

    def test_helper_gfx942_is_32(self):
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx942"), 32)

    def test_helper_gfx950_is_128(self):
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx950"), 128)

    def test_fp8_default_gfx942_warp_tile_k_32(self):
        self.assertEqual(default_fp8_config("gfx942").warp_tile_k, 32)

    def test_fp8_default_gfx950_warp_tile_k_128(self):
        self.assertEqual(default_fp8_config("gfx950").warp_tile_k, 128)

    def test_bf8_default_gfx942_warp_tile_k_32(self):
        self.assertEqual(default_bf8_config("gfx942").warp_tile_k, 32)

    def test_bf8_default_gfx950_warp_tile_k_128(self):
        self.assertEqual(default_bf8_config("gfx950").warp_tile_k, 128)

    def test_name_reflects_arch_warp_tile_k(self):
        self.assertIn("16x16x32", default_fp8_config("gfx942").name)
        self.assertIn("16x16x128", default_fp8_config("gfx950").name)


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = TensorQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)


if __name__ == "__main__":
    unittest.main()
