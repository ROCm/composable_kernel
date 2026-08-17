#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for python/gemm_utils.py.

Locks in the bit-level helpers that the TE -> Dispatcher GEMM bridge relies on:
  * bf16 <-> uint16 encoding (round-to-nearest-even), since numpy has no native
    bf16 and the runner carries bf16 as a uint16 bit pattern.
  * fp8 (E4M3) / bf8 (E5M2) FNUZ <-> uint8 encoding, used for the gfx942 8-bit
    float surface. The decode must be exact to the device format; the encode
    only needs to land on the nearest representable byte.
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
    _fp32_to_fp8_u8,
    _fp8_u8_to_fp32,
    _fp32_to_bf8_u8,
    _bf8_u8_to_fp32,
    _fnuz_decode_table,
    _output_dtype,
    _dtype_from_kernel_name,
    _layout_from_kernel_name,
    _cshuffle_store_ok,
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


class TestFp8Bf8Encoding(unittest.TestCase):
    """fp8 E4M3 / bf8 E5M2 in the FNUZ format used by gfx942.

    The decode is the load-bearing half (it must equal the device value for a
    byte); the encode must land on the nearest representable byte and saturate.
    """

    def test_format_ranges(self):
        # FNUZ maxima: E4M3 -> 2^7 * 1.875 = 240; E5M2 -> 2^15 * 1.75 = 57344.
        t43 = _fnuz_decode_table(4, 3)
        t52 = _fnuz_decode_table(5, 2)
        self.assertEqual(float(np.nanmax(t43)), 240.0)
        self.assertEqual(float(np.nanmin(t43)), -240.0)
        self.assertEqual(float(np.nanmax(t52)), 57344.0)
        self.assertEqual(float(np.nanmin(t52)), -57344.0)

    def test_zero_and_nan_slots(self):
        # 0x00 is +0; the negative-zero slot 0x80 is the lone NaN (FNUZ).
        for tab in (_fnuz_decode_table(4, 3), _fnuz_decode_table(5, 2)):
            self.assertEqual(float(tab[0x00]), 0.0)
            self.assertTrue(np.isnan(tab[0x80]))

    def test_exactly_representable_roundtrip(self):
        exact = np.array([0.0, 0.5, 1.0, -1.0, 2.0, -2.0, 1.5, -0.25, 4.0, 8.0],
                         dtype=np.float32)
        np.testing.assert_array_equal(
            _fp8_u8_to_fp32(_fp32_to_fp8_u8(exact)), exact)
        np.testing.assert_array_equal(
            _bf8_u8_to_fp32(_fp32_to_bf8_u8(exact)), exact)

    def test_decode_is_consistent_with_encode(self):
        # The parity contract: ref multiplies decode(encode(x)), so the pair must
        # be self-consistent and every encoded byte must decode finite.
        rng = np.random.default_rng(1)
        x = (rng.standard_normal(5000) * 0.1).astype(np.float32)
        for enc, dec in ((_fp32_to_fp8_u8, _fp8_u8_to_fp32),
                         (_fp32_to_bf8_u8, _bf8_u8_to_fp32)):
            d = dec(enc(x))
            self.assertTrue(np.all(np.isfinite(d)))

    def test_saturates_no_inf(self):
        # FNUZ has no infinity: huge magnitudes clamp to the finite max.
        big = np.array([1e30, -1e30], dtype=np.float32)
        self.assertEqual(float(_fp8_u8_to_fp32(_fp32_to_fp8_u8(big))[0]), 240.0)
        self.assertEqual(float(_bf8_u8_to_fp32(_fp32_to_bf8_u8(big))[1]), -57344.0)

    def test_dtype_and_size(self):
        for enc in (_fp32_to_fp8_u8, _fp32_to_bf8_u8):
            u8 = enc(np.zeros(4, dtype=np.float32))
            self.assertEqual(u8.dtype, np.uint8)
            self.assertEqual(u8.itemsize, 1)  # must match sizeof(fp8_t/bf8_t)


class TestOutputDtype(unittest.TestCase):
    """Output (C) element dtype must mirror the codegen's get_output_dtype."""

    def test_mapping(self):
        self.assertEqual(_output_dtype("fp16"), "fp16")
        self.assertEqual(_output_dtype("bf16"), "bf16")
        self.assertEqual(_output_dtype("fp8"), "fp16")
        self.assertEqual(_output_dtype("bf8"), "fp16")
        self.assertEqual(_output_dtype("int8"), "int32")


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
        for dtype in ("fp16", "bf16", "fp8", "bf8", "int8"):
            for la, lb, lc in (("row", "col", "row"),
                               ("row", "row", "row"),
                               ("col", "col", "row"),
                               ("col", "row", "row")):
                cfg = GemmKernelConfig(
                    dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
                    dtype_acc=("int32" if dtype == "int8" else "fp32"),
                    layout_a=la, layout_b=lb, layout_c=lc,
                )
                name = cfg.name
                self.assertEqual(_dtype_from_kernel_name(name), dtype)
                self.assertEqual(_layout_from_kernel_name(name), cfg.layout)


class TestCShuffleStoreGate(unittest.TestCase):
    """Narrowed CShuffle-store correctness gate (issue #9684).

    Only an ODD per-wave repeat (>1) with a 32-wide warp tile in that dimension
    is numerically wrong; every other non-power-of-two repeat is correct. These
    expectations were GPU-verified on gfx942 (26 broken / 90 correct across the
    tile_m=192 cshuffle config space).
    """

    def test_broken_signature_rejected(self):
        # tile_m=192 / wave_m=2 / warp_tile_m=32 -> MRepeat = 192/(2*32) = 3.
        # The 26 verified-wrong configs all match this (odd repeat + 32 warp).
        self.assertFalse(_cshuffle_store_ok(3, 2, 32, 32))
        self.assertFalse(_cshuffle_store_ok(3, 4, 32, 16))  # M side triggers
        self.assertFalse(_cshuffle_store_ok(4, 3, 16, 32))  # N side triggers

    def test_odd_repeat_with_16_warp_tile_allowed(self):
        # MRepeat=3 via wave_m=4 / warp_tile_m=16 is numerically correct.
        self.assertTrue(_cshuffle_store_ok(3, 2, 16, 16))

    def test_even_nonpow2_repeat_allowed(self):
        # Repeats 6 and 12 are non-power-of-two but verified correct, incl. w/32.
        self.assertTrue(_cshuffle_store_ok(6, 4, 32, 16))
        self.assertTrue(_cshuffle_store_ok(12, 2, 16, 32))

    def test_power_of_two_repeats_allowed(self):
        for rep in (1, 2, 4, 8):
            self.assertTrue(_cshuffle_store_ok(rep, rep, 32, 32))
            self.assertTrue(_cshuffle_store_ok(rep, rep, 16, 16))


class TestModuleImportsAndRunnerShape(unittest.TestCase):
    """Guards against a merge truncating gemm_utils (regression: #9308 dropped
    the tail of GpuMultiDGemmRunner.run, leaving an unterminated
    ``MultiDGemmResult(`` that made the whole module fail to import).

    Importing this test file already exercises ``import gemm_utils``; these
    assertions additionally pin the multi_d / multi_abd runner shapes so the
    method can't silently land in the wrong class again.
    """

    def test_module_imports(self):
        import gemm_utils  # noqa: F401  (import must not raise)

    def test_codegen_module_parses(self):
        # unified_gemm_codegen.py was truncated by the same #9308 merge (an
        # unterminated f-string in _multi_d_single_include). Parse it directly so
        # a syntax-level truncation is caught even without importing its deps.
        import ast

        codegen = DISPATCHER_DIR / "codegen" / "unified_gemm_codegen.py"
        ast.parse(codegen.read_text(), filename=str(codegen))

    def test_multi_d_runner_has_run_returning_multi_d_result(self):
        import inspect
        import gemm_utils as g

        self.assertTrue(
            callable(getattr(g.GpuMultiDGemmRunner, "run", None)),
            "GpuMultiDGemmRunner must expose run()",
        )
        src = inspect.getsource(g.GpuMultiDGemmRunner.run)
        self.assertIn("return MultiDGemmResult(", src)
        # The return must be complete (all dataclass fields present).
        for field in ("output=", "time_ms=", "status=", "tflops=", "kernel_name="):
            self.assertIn(field, src, f"multi_d run() missing {field} in result")

    def test_multi_abd_runner_has_no_stray_multi_d_code(self):
        import inspect
        import gemm_utils as g

        src = inspect.getsource(g.GpuMultiABDRunner)
        self.assertIn("_parse_layout4", src)
        self.assertNotIn(
            "MultiDGemmResult",
            src,
            "GpuMultiABDRunner must not contain multi_d result code (merge slip)",
        )


if __name__ == "__main__":
    unittest.main()
