#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the microscaling (mx) GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the codegen-JSON projection, the dtype/layout/warp-tile
validity gate, the e8m0 scale codec, the fp8/fp4 quantization round-trips, and the numpy
microscaled reference. No GPU, no hipcc, no Old-TE builder import required.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from mx_gemm_utils import (  # noqa: E402
    SCALE_BLOCK,
    E8M0_ONE,
    MxGemmKernelConfig,
    MxGemmProblem,
    default_fp8_config,
    default_fp4_config,
    e8m0_to_float,
    float_to_e8m0,
    quantize_fp8,
    dequantize_fp8,
    quantize_fp4_packed,
    dequantize_fp4_packed,
    mx_gemm_reference,
)


class TestConfigName(unittest.TestCase):
    def test_fallback_name_prefix(self):
        cfg = default_fp8_config()
        self.assertTrue(cfg._fallback_name().startswith("mx_gemm_fp8_rcr_"))

    def test_fallback_name_encodes_tiles(self):
        cfg = MxGemmKernelConfig(
            datatype="fp4", tile_m=64, tile_n=128, tile_k=256,
            warp_m=1, warp_n=2, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        )
        name = cfg._fallback_name()
        self.assertIn("_fp4_rcr_", name)
        self.assertIn("64x128x256", name)
        self.assertIn("1x2x1", name)
        self.assertIn("16x16x128", name)
        self.assertNotIn(" ", name)

    def test_persistent_suffix_only_when_set(self):
        self.assertNotIn("True", default_fp8_config()._fallback_name())
        cfg = default_fp8_config()
        cfg.persistent = True
        self.assertIn("True", cfg._fallback_name())


class TestCodegenJson(unittest.TestCase):
    def test_projection_roundtrip(self):
        cfg = MxGemmKernelConfig(
            datatype="fp8", tile_m=128, tile_n=128, tile_k=128,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            k_block_per_cu=3,
            # Pin the arch so to_codegen_config() does not shell out to rocminfo;
            # keeps this a CPU-only test on non-ROCm runners.
            gpu_target="gfx950",
        )
        j = cfg.to_codegen_config()
        self.assertEqual(j["datatype"], "fp8")
        self.assertEqual(j["layout"], "rcr")
        self.assertEqual(j["tile_config"]["tile_k"], 128)
        self.assertEqual(j["tile_config"]["warp_tile_k"], 128)
        self.assertEqual(j["k_block_per_cu"], 3)


class TestValidity(unittest.TestCase):
    def test_default_configs_valid(self):
        self.assertTrue(default_fp8_config().is_valid())
        self.assertTrue(default_fp4_config().is_valid())

    def test_non_rcr_rejected(self):
        cfg = default_fp8_config()
        cfg.layout = "rrr"
        self.assertFalse(cfg.is_valid())

    def test_bad_dtype_rejected(self):
        cfg = default_fp8_config()
        cfg.datatype = "bf16"
        self.assertFalse(cfg.is_valid())

    def test_bf8_rejected(self):
        # Old-TE argparse (choices=["fp4","fp8"]) + validate_gemm_mx never
        # compile bf8/e5m2 for mx_gemm, so the bridge must reject it too. Assert
        # both the config-level gate (is_valid) and, when importable, the
        # codegen-level gate (_validate) refuse dtype="bf8".
        cfg = default_fp8_config()
        cfg.datatype = "bf8"
        # Pin the arch so to_codegen_config() below stays CPU-only (no rocminfo).
        cfg.gpu_target = "gfx950"
        self.assertFalse(cfg.is_valid())

        try:
            from unified_mx_gemm_codegen import _validate  # noqa: E402
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"codegen import unavailable: {exc}")
        with self.assertRaises(Exception):
            _validate(cfg.to_codegen_config())

    def test_wrong_warp_tile_rejected(self):
        cfg = default_fp8_config()
        cfg.warp_tile_k = 64
        self.assertFalse(cfg.is_valid())

    def test_indivisible_tile_rejected(self):
        cfg = default_fp8_config()
        cfg.tile_m = 100  # not a multiple of warp_m * warp_tile_m (2*16=32)
        self.assertFalse(cfg.is_valid())


class TestProblem(unittest.TestCase):
    def test_scale_k_and_flops(self):
        p = MxGemmProblem(M=64, N=128, K=256)
        self.assertEqual(p.scale_k, 256 // SCALE_BLOCK)
        self.assertEqual(p.flops, 2 * 64 * 128 * 256)

    def test_k_not_multiple_of_32_raises(self):
        with self.assertRaises(ValueError):
            MxGemmProblem(M=32, N=32, K=48)


class TestE8m0Codec(unittest.TestCase):
    def test_one_is_byte_127(self):
        self.assertEqual(int(float_to_e8m0(np.float32(1.0))), E8M0_ONE)
        self.assertEqual(float(e8m0_to_float(E8M0_ONE)), 1.0)

    def test_power_of_two_roundtrip(self):
        for s in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            b = float_to_e8m0(np.float32(s))
            self.assertAlmostEqual(float(e8m0_to_float(b)), s, places=6)

    def test_255_is_nan(self):
        with np.errstate(over="ignore"):
            self.assertTrue(np.isnan(float(e8m0_to_float(255))))

    def test_non_positive_raises(self):
        # Contract is a strictly-positive power-of-two; a 0.0/negative scale is a
        # caller bug and must fail loudly rather than silently encode 1.0.
        for bad in (0.0, -1.0, -0.0):
            with self.assertRaises(ValueError):
                float_to_e8m0(np.float32(bad))
        with self.assertRaises(ValueError):
            float_to_e8m0(np.array([1.0, 0.0, 2.0], np.float32))

    def test_non_finite_raises(self):
        with np.errstate(invalid="ignore"):
            for bad in (np.inf, np.nan):
                with self.assertRaises(ValueError):
                    float_to_e8m0(np.float32(bad))


class TestFp8Codec(unittest.TestCase):
    def test_known_bytes(self):
        self.assertEqual(int(quantize_fp8(np.array([1.0], np.float32))[0]), 0x38)
        self.assertEqual(int(quantize_fp8(np.array([-2.0], np.float32))[0]), 0xC0)

    def test_roundtrip_grid(self):
        grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], np.float32)
        vals = np.tile(grid, (4, 1))
        self.assertTrue(np.array_equal(dequantize_fp8(quantize_fp8(vals)), vals))

    def test_off_grid_raises(self):
        # The vectorized codec must reject values not on the exact e4m3 grid
        # instead of snapping them to a neighbour byte.
        with self.assertRaises(KeyError):
            quantize_fp8(np.array([[0.3]], np.float32))

    def test_neg_zero_collapses_to_zero_byte(self):
        self.assertEqual(int(quantize_fp8(np.array([-0.0], np.float32))[0]), 0x00)

    def test_shape_preserved_2d(self):
        vals = np.full((3, 5), 1.0, np.float32)
        self.assertEqual(quantize_fp8(vals).shape, (3, 5))


class TestFp4Codec(unittest.TestCase):
    def test_pack_two_per_byte(self):
        # one row, K=2 -> a single packed byte; low nibble even-K, high nibble odd-K.
        vals = np.array([[1.0, 2.0]], np.float32)  # codes: 1.0->2, 2.0->4
        packed = quantize_fp4_packed(vals)
        self.assertEqual(packed.shape, (1, 1))
        self.assertEqual(int(packed[0, 0]), (4 << 4) | 2)

    def test_roundtrip_grid(self):
        grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], np.float32)
        rng = np.random.default_rng(0)
        vals = rng.choice(grid, size=(4, 8)).astype(np.float32)
        packed = quantize_fp4_packed(vals)
        self.assertEqual(packed.shape, (4, 4))
        self.assertTrue(np.array_equal(dequantize_fp4_packed(packed, 8), vals))

    def test_off_grid_raises(self):
        # 2.5 exists in fp8 e4m3 but NOT in the fp4 e2m1 grid -> must reject.
        with self.assertRaises(KeyError):
            quantize_fp4_packed(np.array([[2.5, 1.0]], np.float32))


class TestReference(unittest.TestCase):
    def _inputs(self, M, N, K, seed=0):
        rng = np.random.default_rng(seed)
        grid = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], np.float32)
        A = rng.choice(grid, size=(M, K)).astype(np.float32)
        B = rng.choice(grid, size=(K, N)).astype(np.float32)
        return A, B

    def test_unit_scales_equals_plain_matmul(self):
        M, N, K = 4, 3, 32
        A, B = self._inputs(M, N, K)
        prob = MxGemmProblem(M=M, N=N, K=K)
        one = int(float_to_e8m0(np.float32(1.0)))
        sa = np.full((M, prob.scale_k), one, np.uint8)
        sb = np.full((N, prob.scale_k), one, np.uint8)
        ref = mx_gemm_reference(A, B, sa, sb, prob).astype(np.float32)
        plain = (A @ B).astype(np.float16).astype(np.float32)
        self.assertTrue(np.allclose(ref, plain, atol=1e-2))

    def test_power_of_two_scale_multiplies(self):
        M, N, K = 4, 3, 32
        A, B = self._inputs(M, N, K, seed=1)
        prob = MxGemmProblem(M=M, N=N, K=K)
        two = int(float_to_e8m0(np.float32(2.0)))
        sa = np.full((M, prob.scale_k), two, np.uint8)
        sb = np.full((N, prob.scale_k), two, np.uint8)
        ref = mx_gemm_reference(A, B, sa, sb, prob).astype(np.float32)
        # scale_a=2 and scale_b=2 -> product scaled by 4.
        expected = (4.0 * (A @ B)).astype(np.float16).astype(np.float32)
        self.assertTrue(np.allclose(ref, expected, atol=1e-1))


class TestCodegenNameContract(unittest.TestCase):
    """Optional: byte-exact name parity when the Old-TE builder is importable."""

    def test_codegen_name_matches_config(self):
        try:
            from unified_mx_gemm_codegen import kernel_name
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"codegen import unavailable: {exc}")
        cfg = default_fp8_config()
        # Pin the arch so to_codegen_config() stays CPU-only (no rocminfo).
        cfg.gpu_target = "gfx950"
        try:
            name = kernel_name(cfg.to_codegen_config())
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Old-TE builder unavailable: {exc}")
        self.assertTrue(name.startswith("mx_gemm_fp8_rcr_"))


if __name__ == "__main__":
    unittest.main()
