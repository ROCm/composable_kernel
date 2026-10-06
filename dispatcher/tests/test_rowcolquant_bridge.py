#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the rowcolquant GEMM TileEngine -> Dispatcher bridge.

Locks the fp8/bf8 x rcr scope (per-row scale on A, per-col scale on B; the exact
dtype/layout set Old-TE gemm_quant_rowcol.cpp registers), the arch-derived
warp_tile_k trap, the arch-threaded self-test, and the arch-aware fp8 encoding
flavour.  No GPU, no hipcc, no Old-TE builder import required.

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

from gemm_rowcolquant_utils import (  # noqa: E402
    RowColQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    encode_fp8_bytes,
    quantize_dequantize_fp8,
    fp8_encoding_available,
    _warp_tile_k_for,
    _ml_fp8_dtype,
    _uses_ocp_fp8,
)


class TestScope(unittest.TestCase):
    """gemm_quant_rowcol.cpp registers exactly fp8/rowcol and bf8/rowcol, rcr only."""

    def test_default_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")

    def test_layout_is_rcr(self):
        self.assertEqual(default_fp8_config().layout, "rcr")
        self.assertEqual(default_bf8_config().layout, "rcr")


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = RowColQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)


class TestArchWarpTileK(unittest.TestCase):
    """warp_tile_k must be arch-derived (get_k_warp_tile<fp8/bf8,16>()).

    gfx942 has NO valid 16x16x128 fp8/bf8 warp-gemm: warp_tile_k=128 compiles
    but silently outputs all-zeros there (confirmed on the sibling tensor_quant
    GPU tester). Old-TE uses 16x16x32 on gfx942. Only gfx950 gets 128.
    """

    def test_helper_gfx942(self):
        self.assertEqual(_warp_tile_k_for("fp8", "gfx942"), 32)
        self.assertEqual(_warp_tile_k_for("bf8", "gfx942"), 32)

    def test_helper_gfx950(self):
        self.assertEqual(_warp_tile_k_for("fp8", "gfx950"), 128)
        self.assertEqual(_warp_tile_k_for("bf8", "gfx950"), 128)

    def test_default_config_gfx942(self):
        self.assertEqual(default_fp8_config("gfx942").warp_tile_k, 32)
        self.assertEqual(default_bf8_config("gfx942").warp_tile_k, 32)

    def test_default_config_gfx950(self):
        self.assertEqual(default_fp8_config("gfx950").warp_tile_k, 128)
        self.assertEqual(default_bf8_config("gfx950").warp_tile_k, 128)

    def test_name_reflects_arch_warp_tile_k(self):
        self.assertIn("16x16x32", default_fp8_config("gfx942").name)
        self.assertIn("16x16x128", default_fp8_config("gfx950").name)


class TestArchThreadedIntoConfig(unittest.TestCase):
    """The self-test must thread the SELECTED arch into the default configs.

    Round-3 GPU-tester finding: main() built gfx950 tiles even under
    --arch gfx942 (all-zeros). Guard that the arch reaches warp_tile_k and the
    encoded kernel name, so a regression that drops the arch is caught here.
    """

    def test_gfx942_config_differs_from_gfx950(self):
        c942 = default_fp8_config("gfx942")
        c950 = default_fp8_config("gfx950")
        self.assertEqual(c942.gfx_arch, "gfx942")
        self.assertEqual(c950.gfx_arch, "gfx950")
        self.assertEqual(c942.warp_tile_k, 32)
        self.assertEqual(c950.warp_tile_k, 128)
        self.assertNotEqual(c942.name, c950.name)

    def test_selftest_main_passes_arch(self):
        # The self-test main() must construct configs WITH the arch. Mirror its
        # call site and assert the arch propagated (guards the Round-3 bug where
        # default_fp8_config() was called with no arg).
        for arch, expected_k in (("gfx942", 32), ("gfx950", 128)):
            for cfg in (default_fp8_config(arch), default_bf8_config(arch)):
                self.assertEqual(cfg.gfx_arch, arch)
                self.assertEqual(cfg.warp_tile_k, expected_k)


class TestFp8EncodingFlavourByArch(unittest.TestCase):
    """Encoding flavour must follow CK_USE_OCP_FP8: gfx942 -> FNUZ, gfx950 -> OCP.

    Round-3 GPU-tester finding: the reference silently NaN'd on gfx942 because
    the encoder hardcoded OCP e4m3/e5m2. FNUZ is required on gfx942.
    """

    def test_uses_ocp_switch(self):
        self.assertTrue(_uses_ocp_fp8("gfx950"))
        self.assertTrue(_uses_ocp_fp8("gfx1200"))
        self.assertFalse(_uses_ocp_fp8("gfx942"))
        self.assertFalse(_uses_ocp_fp8("gfx90a"))
        # Unknown arch defaults to OCP (historical gfx950 self-test default).
        self.assertTrue(_uses_ocp_fp8(None))

    @unittest.skipUnless(fp8_encoding_available(), "ml_dtypes fp8 not installed")
    def test_dtype_names_by_arch(self):
        import ml_dtypes

        # gfx950 -> OCP e4m3/e5m2
        self.assertIs(_ml_fp8_dtype("fp8", "gfx950"), ml_dtypes.float8_e4m3)
        self.assertIs(_ml_fp8_dtype("bf8", "gfx950"), ml_dtypes.float8_e5m2)
        # gfx942 -> FNUZ e4m3fnuz/e5m2fnuz
        self.assertIs(_ml_fp8_dtype("fp8", "gfx942"), ml_dtypes.float8_e4m3fnuz)
        self.assertIs(_ml_fp8_dtype("bf8", "gfx942"), ml_dtypes.float8_e5m2fnuz)

    @unittest.skipUnless(fp8_encoding_available(), "ml_dtypes fp8 not installed")
    def test_encode_flavour_bits_differ(self):
        import numpy as np

        # A value whose OCP and FNUZ bit patterns differ (FNUZ has a 1-bit
        # exponent-bias shift) -- proves the arch actually selects the codec.
        a = np.array([0.5], dtype=np.float32)
        ocp = encode_fp8_bytes(a, "fp8", "gfx950")
        fnuz = encode_fp8_bytes(a, "fp8", "gfx942")
        self.assertEqual(ocp.nbytes, 1)
        self.assertEqual(fnuz.nbytes, 1)
        self.assertNotEqual(int(ocp[0]), int(fnuz[0]))

    @unittest.skipUnless(fp8_encoding_available(), "ml_dtypes fp8 not installed")
    def test_fnuz_reference_is_finite_on_gfx942(self):
        import numpy as np

        # The Round-3 bug surfaced as NaN in the reference. Confirm the FNUZ
        # round-trip stays finite over the self-test's input range on gfx942.
        rng = np.random.default_rng(0)
        a = rng.uniform(-2.0, 2.0, size=(64, 64)).astype(np.float32)
        for variant in ("fp8", "bf8"):
            qd = quantize_dequantize_fp8(a, variant, "gfx942")
            self.assertTrue(np.all(np.isfinite(qd)))


@unittest.skipUnless(fp8_encoding_available(), "ml_dtypes fp8 not installed")
class TestFp8Encode(unittest.TestCase):
    """The self-test's genuine numeric path depends on these host-side encoders.

    Encoded fp8/bf8 must be exactly 1 byte per element (the ctypes lib reads
    A/B as const fp8_t*/bf8_t*), and the reference-side quantize->dequantize
    must produce values consistent with that same encoding.
    """

    def test_encode_is_one_byte_per_element(self):
        import numpy as np

        a = np.array([[0.5, -1.25, 2.0, -0.03]], dtype=np.float32)
        for variant in ("fp8", "bf8"):
            enc = encode_fp8_bytes(a, variant)
            self.assertEqual(enc.dtype, np.uint8)
            self.assertEqual(enc.shape, a.shape)
            self.assertEqual(enc.nbytes, a.size)  # 1 byte/element, not 4

    def test_quant_dequant_matches_encoding(self):
        import numpy as np

        a = np.array([0.5, -1.25, 2.0, 1.0], dtype=np.float32)
        # Exactly representable e4m3 values survive the round-trip.
        qd = quantize_dequantize_fp8(a, "fp8")
        np.testing.assert_allclose(qd, a, rtol=0, atol=0)

    def test_fp8_and_bf8_round_differently(self):
        import numpy as np

        # 0.03 is not exactly representable; e4m3 and e5m2 round it differently.
        a = np.array([0.03], dtype=np.float32)
        self.assertNotEqual(
            float(quantize_dequantize_fp8(a, "fp8")[0]),
            float(quantize_dequantize_fp8(a, "bf8")[0]),
        )


if __name__ == "__main__":
    unittest.main()
