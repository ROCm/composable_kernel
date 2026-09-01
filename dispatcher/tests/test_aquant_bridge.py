#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the aquant (A-only quantized) GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the byte-exact codegen<->utils kernel-name contract, the
codegen-JSON projection, and the dtype/pipeline scope (fp8/bf8/fp8i4/bf8i4, decode via the
mem/interwave pipeline and preshufflequant via the compv3/intrawave pipeline) that Old-TE
gemm_aquant_quantgrouped{,_preshufflequant}.cpp register. No GPU, no hipcc required.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_aquant_utils import (  # noqa: E402
    AQuantGemmProblem,
    _aq_stride,
    _validate_arch,
    _warp_tile_k_for,
    _LAYOUTS_DECODE,
    _LAYOUTS_PRESHUFFLEQUANT,
    _LAYOUTS_AQ_COLMAJOR,
    _SUPPORTED_ARCHS,
    _VARIANT_META,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshufflequant_config,
    default_bf8_preshufflequant_config,
    default_fp8i4_preshufflequant_config,
    default_bf8i4_preshufflequant_config,
)

_DECODE = [
    ("fp8", default_fp8_config),
    ("bf8", default_bf8_config),
    ("fp8i4", default_fp8i4_config),
    ("bf8i4", default_bf8i4_config),
]


class TestConfigName(unittest.TestCase):
    # The name prefix / tiles-in-name and the byte-exact codegen<->utils
    # kernel-name contract are exercised by the shared parametrized tests in
    # test_quant_bridge_shared.py. Only the aquant-specific pipeline_key mapping
    # (mem for decode, compv3 for preshufflequant) is asserted here.
    def test_pipeline_key_reflects_preshuffle(self):
        self.assertEqual(default_fp8_config().pipeline_key, "mem")
        self.assertEqual(default_fp8_preshufflequant_config().pipeline_key, "compv3")


class TestScope(unittest.TestCase):
    def test_decode_variants(self):
        self.assertEqual([v for v, _ in _DECODE],
                         [ctor().variant_key for _, ctor in _DECODE])

    def test_decode_uses_mem_pipeline(self):
        for _, ctor in _DECODE:
            self.assertFalse(ctor().preshuffle_aquant)

    def test_preshufflequant_flag(self):
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_aquant)


class TestProblem(unittest.TestCase):
    def test_problem_constructs(self):
        p = AQuantGemmProblem(M=128, N=256, K=512)
        self.assertEqual((p.M, p.N, p.K), (128, 256, 512))


class TestAQStride(unittest.TestCase):
    """AQ is always RowMajor (matches Old-TE, which hardcodes AQLayout=RowMajor for
    every layout), so every layout -- including ccr -- uses the row-major stride QK_A."""

    M, QK_A = 96, 4

    def test_all_layouts_row_major(self):
        # Old-TE hardcodes AQLayout=RowMajor for every layout; emitting ColumnMajor for
        # ccr produced a KERNEL_MISMATCH under the strict objdump same-kernel gate
        # (the ccr mem AQ-layout bug). ccr must NOT be column-major.
        self.assertEqual(_LAYOUTS_AQ_COLMAJOR, frozenset())
        for layout in ("rcr", "rrr", "crr", "ccr"):
            self.assertNotIn(layout, _LAYOUTS_AQ_COLMAJOR)
            self.assertEqual(_aq_stride(layout, self.M, self.QK_A), self.QK_A)

    def test_ccr_stride_matches_row_major(self):
        # Post AQLayout fix: ccr uses the same row-major stride (QK_A) as rcr.
        self.assertEqual(
            _aq_stride("ccr", self.M, self.QK_A),
            _aq_stride("rcr", self.M, self.QK_A),
        )


class TestLayoutScope(unittest.TestCase):
    """Lock the layout scope and the 28-kernel (4 variant x 7 layout) count."""

    def test_decode_layouts(self):
        self.assertEqual(_LAYOUTS_DECODE, ("rcr", "rrr", "crr", "ccr"))

    def test_preshufflequant_excludes_ccr(self):
        self.assertEqual(_LAYOUTS_PRESHUFFLEQUANT, ("rcr", "rrr", "crr"))
        self.assertNotIn("ccr", _LAYOUTS_PRESHUFFLEQUANT)

    def test_full_kernel_count_is_28(self):
        # 4 variants x (4 decode layouts + 3 preshufflequant layouts) = 28.
        variants = len(_VARIANT_META)
        total = variants * (len(_LAYOUTS_DECODE) + len(_LAYOUTS_PRESHUFFLEQUANT))
        self.assertEqual(variants, 4)
        self.assertEqual(total, 28)


class TestArchWarpTileK(unittest.TestCase):
    """warp_tile_k must be arch-derived (get_k_warp_tile<8bit_float,16,IsFlatMM>()).

    Every AQuant variant (fp8/bf8/fp8i4/bf8i4) instantiates the GEMM config with an
    8-bit float PrecType (fp8_t/bf8_t; pk_int4 A does not drive the K warp tile), so
    warp_tile_k depends only on arch and pipeline. gfx942 has NO valid 16x16x128
    fp8/bf8 warp-gemm: warp_tile_k=128 compiles but SILENTLY OUTPUTS ALL-ZEROS there
    (GPU-confirmed on gfx942 MI300X). Old-TE uses 16x16x32 (decode) / 16x16x64
    (preshufflequant) on gfx942. Only gfx950 gets 128.
    """

    _DECODE_CTORS = [
        default_fp8_config,
        default_bf8_config,
        default_fp8i4_config,
        default_bf8i4_config,
    ]
    _PRESHUF_CTORS = [
        default_fp8_preshufflequant_config,
        default_bf8_preshufflequant_config,
        default_fp8i4_preshufflequant_config,
        default_bf8i4_preshufflequant_config,
    ]

    def test_helper_decode(self):
        self.assertEqual(_warp_tile_k_for("gfx942", preshuffle_aquant=False), 32)
        self.assertEqual(_warp_tile_k_for("gfx950", preshuffle_aquant=False), 128)

    def test_helper_preshufflequant(self):
        self.assertEqual(_warp_tile_k_for("gfx942", preshuffle_aquant=True), 64)
        self.assertEqual(_warp_tile_k_for("gfx950", preshuffle_aquant=True), 128)

    def test_helper_arch_suffix_tolerant(self):
        # Real rocm_agent_enumerator output carries feature suffixes.
        self.assertEqual(_warp_tile_k_for("gfx942:sramecc+:xnack-"), 32)
        self.assertEqual(_warp_tile_k_for("gfx950:sramecc+:xnack-"), 128)

    def test_decode_configs_gfx942(self):
        # BLOCKING: all four decode variants must be 32 on gfx942, NOT 128.
        for ctor in self._DECODE_CTORS:
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 32, ctor.__name__)

    def test_decode_configs_gfx950(self):
        for ctor in self._DECODE_CTORS:
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_preshufflequant_configs_gfx942(self):
        for ctor in self._PRESHUF_CTORS:
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 64, ctor.__name__)

    def test_preshufflequant_configs_gfx950(self):
        for ctor in self._PRESHUF_CTORS:
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_name_reflects_arch_warp_tile_k(self):
        # The arch-derived value must flow into the byte-exact .name.
        self.assertIn("16x16x32", default_fp8_config(gfx_arch="gfx942").name)
        self.assertIn("16x16x128", default_fp8_config(gfx_arch="gfx950").name)
        self.assertIn("16x16x64", default_bf8_preshufflequant_config(gfx_arch="gfx942").name)

    def test_explicit_override_respected(self):
        # An explicit warp_tile_k still wins (for sweeps / experimentation).
        self.assertEqual(default_fp8_config(gfx_arch="gfx942", warp_tile_k=128).warp_tile_k, 128)


class TestSupportedArchs(unittest.TestCase):
    def test_gfx90a_supported_and_validated(self):
        self.assertIn("gfx90a", _SUPPORTED_ARCHS)
        self.assertEqual(_validate_arch("gfx90a"), "gfx90a")
        self.assertEqual(_validate_arch("gfx90a:sramecc+:xnack-"), "gfx90a:sramecc+:xnack-")

    def test_unsupported_arch_raises(self):
        with self.assertRaises(ValueError):
            _validate_arch("gfx1030")


if __name__ == "__main__":
    unittest.main()
