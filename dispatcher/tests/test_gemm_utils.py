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
    _fp8_decode_table,
    _fp8_encode,
    numpy_dtype_for,
    _output_dtype,
    _dtype_from_kernel_name,
    _layout_from_kernel_name,
    _cshuffle_store_ok,
)


def _require_ml_dtypes():
    """Return the ``ml_dtypes`` module, skipping the test when it is absent.

    It is the independent oracle for the fp8/bf8 tables, so a test that needs it
    must skip rather than silently assert nothing.
    """
    try:
        import ml_dtypes  # noqa: WPS433 (lazy: optional dep)
    except ImportError:  # pragma: no cover - env-dependent
        raise unittest.SkipTest("ml_dtypes is required for fp8/bf8 table checks")
    return ml_dtypes


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
        # FNUZ has no infinity: huge magnitudes clamp to the finite max. Pin the
        # format explicitly -- the wrappers otherwise follow the local GPU, and on
        # a gfx950 runner these are the OCP maxima (448 / -57344) instead.
        big = np.array([1e30, -1e30], dtype=np.float32)
        enc8 = _fp32_to_fp8_u8(big, use_ocp=False)
        enc5 = _fp32_to_bf8_u8(big, use_ocp=False)
        self.assertEqual(float(_fp8_u8_to_fp32(enc8, use_ocp=False)[0]), 240.0)
        self.assertEqual(float(_bf8_u8_to_fp32(enc5, use_ocp=False)[1]), -57344.0)

    def test_dtype_and_size(self):
        for enc in (_fp32_to_fp8_u8, _fp32_to_bf8_u8):
            u8 = enc(np.zeros(4, dtype=np.float32))
            self.assertEqual(u8.dtype, np.uint8)
            self.assertEqual(u8.itemsize, 1)  # must match sizeof(fp8_t/bf8_t)


class TestFp8Bf8OcpEncoding(unittest.TestCase):
    """The OCP (gfx950/gfx12) half of the fp8/bf8 codec.

    The decode table is the load-bearing half -- it defines what the numpy
    reference multiplies -- so it is checked byte-for-byte against ml_dtypes
    rather than spot-checked. Getting the format wrong does not crash; it
    silently rescales every reference value by a factor of two.
    """

    def test_decode_matches_ml_dtypes_for_every_byte(self):
        ml_dtypes = _require_ml_dtypes()
        cases = (
            (4, 3, True, ml_dtypes.float8_e4m3fn),
            (4, 3, False, ml_dtypes.float8_e4m3fnuz),
            (5, 2, True, ml_dtypes.float8_e5m2),
            (5, 2, False, ml_dtypes.float8_e5m2fnuz),
        )
        all_bytes = np.arange(256, dtype=np.uint8)
        for exp_bits, mant_bits, use_ocp, ml_type in cases:
            with self.subTest(dtype=ml_type.__name__):
                ours = _fp8_decode_table(exp_bits, mant_bits, use_ocp)
                theirs = all_bytes.view(ml_type).astype(np.float32)
                same = (ours == theirs) | (np.isnan(ours) & np.isnan(theirs))
                self.assertTrue(
                    np.all(same),
                    f"bytes {np.flatnonzero(~same).tolist()} disagree with ml_dtypes",
                )

    def test_ocp_format_ranges(self):
        # OCP bias is one less than FNUZ, so e4m3fn reaches 448 (not 240);
        # e5m2 keeps 57344 but gains a real infinity.
        t43 = _fp8_decode_table(4, 3, True)
        t52 = _fp8_decode_table(5, 2, True)
        self.assertEqual(float(np.nanmax(t43[np.isfinite(t43)])), 448.0)
        self.assertEqual(float(np.nanmin(t43[np.isfinite(t43)])), -448.0)
        self.assertEqual(float(np.nanmax(t52[np.isfinite(t52)])), 57344.0)
        self.assertEqual(np.count_nonzero(np.isinf(t43)), 0)  # e4m3fn has no Inf
        self.assertEqual(np.count_nonzero(np.isinf(t52)), 2)  # e5m2 does

    def test_ocp_zero_slots(self):
        # Unlike FNUZ, 0x80 is a genuine -0.0 rather than the lone NaN.
        for tab in (_fp8_decode_table(4, 3, True), _fp8_decode_table(5, 2, True)):
            self.assertEqual(float(tab[0x00]), 0.0)
            self.assertFalse(np.isnan(tab[0x80]))
            self.assertEqual(float(tab[0x80]), 0.0)

    def test_every_representable_value_roundtrips_exactly(self):
        for exp_bits, mant_bits in ((4, 3), (5, 2)):
            for use_ocp in (False, True):
                with self.subTest(bits=(exp_bits, mant_bits), ocp=use_ocp):
                    table = _fp8_decode_table(exp_bits, mant_bits, use_ocp)
                    vals = table[np.isfinite(table)]
                    enc = _fp8_encode(vals, exp_bits, mant_bits, use_ocp=use_ocp)
                    np.testing.assert_array_equal(table[enc.astype(np.intp)], vals)

    def test_ocp_saturates_to_max_finite_not_inf(self):
        # e5m2 has an Inf byte; a naive nearest-neighbour search would round
        # huge inputs to it. CK's convert saturates, so the codec must too.
        big = np.array([1e30, -1e30], dtype=np.float32)
        table = _fp8_decode_table(5, 2, True)
        got = table[_fp8_encode(big, 5, 2, use_ocp=True).astype(np.intp)]
        self.assertEqual(float(got[0]), 57344.0)
        self.assertEqual(float(got[1]), -57344.0)
        t43 = _fp8_decode_table(4, 3, True)
        got8 = t43[_fp8_encode(big, 4, 3, use_ocp=True).astype(np.intp)]
        self.assertEqual(float(got8[0]), 448.0)

    def test_nan_maps_to_a_nan_byte_in_both_formats(self):
        nan = np.array([np.nan], dtype=np.float32)
        for exp_bits, mant_bits in ((4, 3), (5, 2)):
            for use_ocp in (False, True):
                with self.subTest(bits=(exp_bits, mant_bits), ocp=use_ocp):
                    table = _fp8_decode_table(exp_bits, mant_bits, use_ocp)
                    enc = _fp8_encode(nan, exp_bits, mant_bits, use_ocp=use_ocp)
                    self.assertTrue(np.isnan(table[int(enc[0])]))

    def test_fnuz_aliases_forward_to_the_shared_table(self):
        # _fnuz_decode_table is kept as a thin FNUZ view; callers rely on the
        # lru_cache identity, so it must return the cached instance itself.
        self.assertIs(_fnuz_decode_table(4, 3), _fp8_decode_table(4, 3, False))
        self.assertIsNot(_fnuz_decode_table(4, 3), _fp8_decode_table(4, 3, True))

    def test_numpy_dtype_follows_the_requested_format(self):
        ml_dtypes = _require_ml_dtypes()
        self.assertEqual(numpy_dtype_for("fp8", use_ocp=True),
                         np.dtype(ml_dtypes.float8_e4m3fn))
        self.assertEqual(numpy_dtype_for("fp8", use_ocp=False),
                         np.dtype(ml_dtypes.float8_e4m3fnuz))
        self.assertEqual(numpy_dtype_for("bf8", use_ocp=True),
                         np.dtype(ml_dtypes.float8_e5m2))
        self.assertEqual(numpy_dtype_for("bf8", use_ocp=False),
                         np.dtype(ml_dtypes.float8_e5m2fnuz))


class TestFp8EncodeMatchesMlDtypes(unittest.TestCase):
    """_fp8_encode vs ml_dtypes: identical in range, divergent on overflow.

    _fp8_encode exists because ml_dtypes cannot express CK's saturating convert,
    not because the rounding differs -- so everything except overflow has to
    agree exactly, and the one place it does not has to be pinned as deliberate.
    """

    _CASES = (
        (4, 3, True,  "float8_e4m3fn"),
        (5, 2, True,  "float8_e5m2"),
        (4, 3, False, "float8_e4m3fnuz"),
        (5, 2, False, "float8_e5m2fnuz"),
    )

    @staticmethod
    def _finite_magnitudes(table):
        return np.unique(table[np.isfinite(table)].astype(np.float64))

    def test_midpoints_round_half_to_even(self):
        # The regression this guards: ties were broken toward the smaller
        # magnitude, which disagreed with the hardware convert (and ml_dtypes)
        # on roughly half of all representable midpoints -- 126 of 252 for
        # e4m3. Every one of those is a silently low reference value.
        ml_dtypes = _require_ml_dtypes()
        for exp_bits, mant_bits, use_ocp, ml_name in self._CASES:
            with self.subTest(bits=(exp_bits, mant_bits), ocp=use_ocp):
                table = _fp8_decode_table(exp_bits, mant_bits, use_ocp)
                mags = self._finite_magnitudes(table)
                mids = ((mags[:-1] + mags[1:]) / 2.0).astype(np.float32)
                ours = table[_fp8_encode(mids, exp_bits, mant_bits,
                                         use_ocp=use_ocp).astype(np.intp)]
                theirs = mids.astype(getattr(ml_dtypes, ml_name))
                np.testing.assert_array_equal(ours.astype(np.float64),
                                              theirs.astype(np.float64))

    def test_agrees_across_the_whole_finite_range(self):
        ml_dtypes = _require_ml_dtypes()
        for exp_bits, mant_bits, use_ocp, ml_name in self._CASES:
            with self.subTest(bits=(exp_bits, mant_bits), ocp=use_ocp):
                table = _fp8_decode_table(exp_bits, mant_bits, use_ocp)
                mags = self._finite_magnitudes(table)
                max_finite = float(mags[-1])
                vals = np.unique(np.concatenate([
                    mags,
                    -mags,
                    np.linspace(-max_finite, max_finite, 4001),
                ]))
                vals = vals[np.abs(vals) <= max_finite].astype(np.float32)
                ours = table[_fp8_encode(vals, exp_bits, mant_bits,
                                         use_ocp=use_ocp).astype(np.intp)]
                theirs = vals.astype(getattr(ml_dtypes, ml_name))
                np.testing.assert_array_equal(ours.astype(np.float64),
                                              theirs.astype(np.float64))

    def test_overflow_saturates_where_ml_dtypes_does_not(self):
        # Intended divergence, not a gap: CK's convert clamps, so the host
        # reference must clamp too. Deferring to ml_dtypes here would poison the
        # reference with NaN/Inf on any input the kernel merely saturates.
        ml_dtypes = _require_ml_dtypes()
        big = np.array([1e30, -1e30], dtype=np.float32)
        for exp_bits, mant_bits, use_ocp, ml_name in self._CASES:
            with self.subTest(bits=(exp_bits, mant_bits), ocp=use_ocp):
                table = _fp8_decode_table(exp_bits, mant_bits, use_ocp)
                max_finite = float(self._finite_magnitudes(table)[-1])
                ours = table[_fp8_encode(big, exp_bits, mant_bits,
                                         use_ocp=use_ocp).astype(np.intp)]
                self.assertEqual(float(ours[0]), max_finite)
                self.assertEqual(float(ours[1]), -max_finite)
                theirs = big.astype(getattr(ml_dtypes, ml_name)).astype(np.float64)
                self.assertFalse(np.all(np.isfinite(theirs)))


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


# --- gfx1250 (CDNA5, WMMA) enablement --------------------------------------
# The regular-GEMM bridge historically allow-listed only CDNA (gfx90a/942/950)
# and carried FNUZ-only fp8 codecs. gfx1250 uses WMMA + OCP fp8, so it needs an
# arch entry and an OCP codec path. These CPU-only tests lock that surface in.
from gemm_utils import (  # noqa: E402
    _SUPPORTED_ARCHES,
    _fp32_to_fp8_ocp_u8,
    _fp32_to_bf8_ocp_u8,
    _use_ocp_fp8,
)


class TestGfx1250Fp8Ocp(unittest.TestCase):
    def test_gfx1250_in_supported_arches(self):
        self.assertIn("gfx1250", _SUPPORTED_ARCHES)

    def test_ocp_fp8_shape_and_dtype(self):
        x = (np.random.RandomState(0).randn(8, 16) * 0.1).astype(np.float32)
        u8 = _fp32_to_fp8_ocp_u8(x)
        self.assertEqual(u8.dtype, np.uint8)
        self.assertEqual(u8.shape, x.shape)

    def test_ocp_fp8_roundtrip_exact_values(self):
        # 1.0, 0.5, 2.0, -1.0 are exactly representable in fp8 E4M3 (OCP).
        import ml_dtypes

        x = np.array([[1.0, 0.5, 2.0, -1.0]], dtype=np.float32)
        u8 = _fp32_to_fp8_ocp_u8(x)
        back = u8.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        np.testing.assert_array_equal(back, x)

    def test_ocp_differs_from_fnuz(self):
        # OCP and FNUZ are distinct encodings; the byte patterns must not match
        # for a generic value (guards against silently reusing the FNUZ codec).
        #
        # use_ocp=False is required, not incidental: _fp32_to_fp8_u8 defaults to
        # autodetecting the local GPU, so on an OCP arch (gfx950/gfx12) both
        # sides would encode OCP and the assertion below would fail for a reason
        # unrelated to the codecs being distinct.
        x = np.array([[0.1, 0.3, 1.5]], dtype=np.float32)
        self.assertFalse(
            np.array_equal(_fp32_to_fp8_ocp_u8(x),
                           _fp32_to_fp8_u8(x, use_ocp=False))
        )

    def test_ocp_bf8_shape(self):
        x = (np.random.RandomState(1).randn(4, 4) * 0.1).astype(np.float32)
        self.assertEqual(_fp32_to_bf8_ocp_u8(x).shape, x.shape)

    def test_use_ocp_fp8_is_callable(self):
        # Returns a bool; on a CPU-only box arch detection may fail -> False.
        self.assertIsInstance(_use_ocp_fp8(), bool)


# --- grouped GEMM on gfx1250 -----------------------------------------------
# The grouped bridge (#9000) shares the arch gate + WMMA warp tiles with the
# regular bridge, so it runs on gfx1250 once gfx1250 is enabled. Its CI config
# already uses the gfx1250-valid WMMA warp tile 16x16x32. These CPU-only checks
# lock that in (multi-problem plumbing + gfx1250 arch acceptance).
from gemm_utils import GroupedGemmProblem, _resolve_arch  # noqa: E402


class TestGroupedGfx1250(unittest.TestCase):
    def test_grouped_problem_roundtrip_and_flops(self):
        groups = [(1024, 1024, 1024), (512, 2048, 256)]
        p = GroupedGemmProblem(groups=groups)
        self.assertEqual(p.group_count, 2)
        self.assertEqual(GroupedGemmProblem.from_dict(p.to_dict()).groups, groups)
        self.assertEqual(p.flops, sum(2.0 * m * n * k for (m, n, k) in groups))

    def test_gfx1250_arch_accepted(self):
        self.assertEqual(_resolve_arch("gfx1250"), "gfx1250")

    def test_grouped_gfx1250_ci_config_uses_wmma_warp_tile(self):
        import json as _json
        from pathlib import Path as _Path

        cfg = (
            _Path(__file__).parent.parent.parent
            / "tile_engine/ops/gemm/grouped_gemm/configs/default_ci_config_gfx1250.json"
        )
        tc = _json.load(open(cfg))["tile_config"]
        # gfx1250 fp16/bf16 WMMA is 16x16x32 (not the CDNA MFMA 32x32x16).
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32])


# --- feature-suffixed arch names -------------------------------------------
# _resolve_arch() validated its input against _SUPPORTED_ARCHES *before*
# normalizing it, so a target carrying feature flags was rejected outright:
#
#     gfx950:sramecc+:xnack-  -> ValueError
#
# Bare targets were unaffected -- gfx1250 is in _SUPPORTED_ARCHES and passed both
# before and after (test_bare_supported_arches_are_unchanged below).
#
# The suffixed form is what hipDeviceProp_t::gcnArchName reports, the form rocminfo prints
# for the ISA line, and the form this repository's own CMakeLists.txt uses for
# GPU_TARGETS ("gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"), so a
# caller copying a target from any of those could not drive the bridge with it.
# Normalization now runs first and the base name is validated.
from gemm_utils import _validate_arch, normalize_gfx_arch  # noqa: E402


class TestSuffixedArchNames(unittest.TestCase):
    def test_supported_arch_with_feature_suffix_resolves_to_its_base(self):
        for given, expected in [
            ("gfx90a:xnack+", "gfx90a"),
            ("gfx942:xnack+", "gfx942"),
            ("gfx942:sramecc+:xnack-", "gfx942"),
            ("gfx950:sramecc+:xnack-", "gfx950"),
            ("gfx1250:xnack-", "gfx1250"),
        ]:
            with self.subTest(arch=given):
                self.assertEqual(_resolve_arch(given), expected)

    def test_bare_supported_arches_are_unchanged(self):
        # The whole point of normalizing before validating is that it costs the
        # existing paths nothing.
        for arch in _SUPPORTED_ARCHES:
            with self.subTest(arch=arch):
                self.assertEqual(_resolve_arch(arch), arch)

    def test_an_unsupported_arch_still_raises_with_or_without_a_suffix(self):
        # Normalizing must not turn the gate off: gfx908 is a real target, it is
        # simply not one this bridge supports, and neither form may pass.
        for arch in ("gfx908", "gfx908:xnack+", "gfx1200", "gfx1201:xnack-"):
            with self.subTest(arch=arch):
                with self.assertRaises(ValueError):
                    _resolve_arch(arch)

    def test_a_typo_still_raises(self):
        for arch in ("gfx942x", "gfx1205", "gfx", "gfx950x:xnack-"):
            with self.subTest(arch=arch):
                with self.assertRaises(ValueError):
                    _resolve_arch(arch)

    def test_the_error_shows_both_the_input_and_the_normalized_form(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_arch("gfx908:xnack+")
        message = str(ctx.exception)
        self.assertIn("gfx908:xnack+", message)
        self.assertIn("gfx908", message)


class TestArchNormalizationMatchesCodegen(unittest.TestCase):
    """gemm_utils cannot import codegen_common at module scope -- nothing has put
    the codegen dir on sys.path by then -- so it carries a fallback. Pin the two
    to identical behaviour, the same way this branch pins tile_engine's
    _base_gfx_arch to codegen_common.normalize_gfx_arch."""

    CASES = (
        "gfx942",
        "gfx942:sramecc+:xnack-",
        "gfx1250:xnack-",
        "gfx950:sramecc+",
        "gfx908",
        "",
        "not-an-arch",
    )

    def test_delegates_to_codegen_common_when_importable(self):
        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        from codegen_common import normalize_gfx_arch as canonical

        for arch in self.CASES:
            with self.subTest(arch=arch):
                self.assertEqual(normalize_gfx_arch(arch), canonical(arch))

    def test_the_fallback_agrees_with_the_canonical_helper(self):
        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        from codegen_common import normalize_gfx_arch as canonical

        import builtins

        real_import = builtins.__import__

        def no_codegen_common(name, *args, **kwargs):
            if name == "codegen_common":
                raise ImportError("simulated: codegen dir not on sys.path")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_codegen_common
        try:
            for arch in self.CASES:
                with self.subTest(arch=arch):
                    self.assertEqual(normalize_gfx_arch(arch), canonical(arch))
        finally:
            builtins.__import__ = real_import
