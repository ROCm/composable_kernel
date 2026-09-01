#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the abquant (A+B both quantized) GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the byte-exact codegen<->utils kernel-name contract, the
codegen-JSON projection, and the fp8/bf8/fp4 x rcr scope with the preshuffleB /
preshuffleQuant families that Old-TE gemm_abquant_quantgrouped*.cpp register. No GPU / hipcc.
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_abquant_utils import (  # noqa: E402
    ABQuantDispatcherLib,
    default_fp8_config,
    default_bf8_config,
    default_fp4_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp4_preshuffleb_config,
    default_fp8_preshuffleb_preshufflequant_config,
    _generate_abquant_kernel,
)

# The ctypes lib source (checked for the B-matrix shuffle step, no GPU needed).
_CTYPES_SRC = (_DISP / "bindings" / "ctypes" / "gemm_abquant_ctypes_lib.cpp").read_text()
# The AQ/BQ scale-tensor prep steps live in the header shared with the bquant and
# aquant bridges, so assertions about them grep there rather than in the .cpp.
_SHUFFLE_SRC = (_DISP / "bindings" / "ctypes" / "quant_bridge_shuffle.hpp").read_text()

_ALL = [
    default_fp8_config,
    default_bf8_config,
    default_fp4_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp4_preshuffleb_config,
    default_fp8_preshuffleb_preshufflequant_config,
]

# The config-name prefix (gemm_abquant_<variant> + rcr) / tiles-in-name contract
# and the byte-exact codegen<->utils kernel-name contract for all _ALL ctors are
# exercised by the shared parametrized tests in test_quant_bridge_shared.py
# (driven by _quant_bridge_descriptors.py). The abquant-specific scope, warp_tile_k,
# EightWaves, preshuffle-B, AQ-layout and permute-N regression tests below stay
# explicit and per-op.


class TestScope(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")
        self.assertEqual(default_fp4_config().variant_key, "fp4")

    def test_layout_is_rcr(self):
        for ctor in _ALL:
            self.assertEqual(ctor().layout, "rcr")

    def test_preshuffle_flags(self):
        self.assertFalse(default_fp8_config().preshuffle_b)
        self.assertTrue(default_fp8_preshuffleb_config().preshuffle_b)
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_bquant)
        pq = default_fp8_preshuffleb_preshufflequant_config()
        self.assertTrue(pq.preshuffle_b and pq.preshuffle_bquant)


class TestGfx950WarpTileK(unittest.TestCase):
    """Finding #1/#2: on gfx950 fp8/bf8 use K_Warp_Tile=128 (get_k_warp_tile),
    fp4 stays 32. Locks the *compiled shape*, not just the name string."""

    def test_fp8_bf8_warp_tile_k_is_128_on_gfx950(self):
        for ctor in (
            default_fp8_config,
            default_bf8_config,
            default_fp8_preshufflequant_config,
            default_fp8_preshuffleb_preshufflequant_config,
        ):
            cfg = ctor(gfx_arch="gfx950")
            self.assertEqual(cfg.warp_tile_k, 128, ctor.__name__)

    def test_fp4_warp_tile_k_is_32_on_gfx950(self):
        self.assertEqual(default_fp4_config(gfx_arch="gfx950").warp_tile_k, 32)
        self.assertEqual(default_fp4_preshuffleb_config(gfx_arch="gfx950").warp_tile_k, 32)

    def test_warp_tile_k_on_gfx942(self):
        # gfx942 (no CK_GFX950_SUPPORT): the preshuffleb prefill configs are
        # IsFlatMM, so get_k_warp_tile<PrecType,16,IsFlatMM=true>() == 64 for all
        # their 1-byte variants (fp8/bf8/pk_fp4); every non-preshuffleb config
        # uses IsFlatMM=false == 32. It must never be 128 (the gfx942 fp8/bf8
        # all-zeros warp-gemm trap) and never eight_waves.
        for ctor in _ALL:
            cfg = ctor(gfx_arch="gfx942")
            expected = 64 if cfg.preshuffle_b else 32
            self.assertEqual(cfg.warp_tile_k, expected, ctor.__name__)
            self.assertNotEqual(cfg.warp_tile_k, 128, ctor.__name__)
            self.assertFalse(cfg.eight_waves, ctor.__name__)
class TestGfx950EightWaves(unittest.TestCase):
    """Finding #1: exactly the 6 fp8/bf8 kernels that route through the
    GemmConfig / GemmConfigPrefill aliases become EightWaves on gfx950:
      non-preshuffleb non-pq 1x128x128 (fp8, bf8)
      preshuffleb            {1,128}   (fp8, bf8)
    All other kernels (fp8 1x1x128 non-pq, all preshufflequant, all fp4) do not."""

    def _ew(self, cfg):
        # An eight_waves kernel must carry the flag, the 192x256x128 tile,
        # the 4x2x1 warps, warp_tile_k=128, the eightwaves pipeline and name tag.
        self.assertTrue(cfg.eight_waves, cfg.name)
        self.assertEqual((cfg.tile_m, cfg.tile_n, cfg.tile_k), (192, 256, 128), cfg.name)
        self.assertEqual((cfg.warp_m, cfg.warp_n, cfg.warp_k), (4, 2, 1), cfg.name)
        self.assertEqual(cfg.warp_tile_k, 128, cfg.name)
        self.assertEqual(cfg.pipeline, "eightwaves", cfg.name)
        self.assertIn("eightwaves", cfg.name)
        # eight_waves always uses the CShuffle epilogue (TiledMMAPermuteN=false).
        self.assertNotIn("permute_n", cfg.name, cfg.name)

    def test_the_six_eight_waves_kernels(self):
        ew = [
            default_fp8_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_bf8_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_fp8_preshuffleb_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshuffleb_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_bf8_preshuffleb_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_bf8_preshuffleb_config(bquant_group_n=128, gfx_arch="gfx950"),
        ]
        self.assertEqual(len(ew), 6)
        for cfg in ew:
            self._ew(cfg)
        # preshuffleb eight_waves still carries preshuffle_b / double_smem.
        for cfg in ew[2:]:
            self.assertTrue(cfg.preshuffle_b and cfg.double_smem_buffer, cfg.name)

    def test_non_eight_waves_kernels(self):
        not_ew = [
            default_fp8_config(bquant_group_n=1, gfx_arch="gfx950"),   # hardcoded ABQuantPrefill
            default_fp4_config(gfx_arch="gfx950"),
            default_fp4_preshuffleb_config(gfx_arch="gfx950"),
            default_fp8_preshufflequant_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshufflequant_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=128, gfx_arch="gfx950"),
        ]
        for cfg in not_ew:
            self.assertFalse(cfg.eight_waves, cfg.name)
            self.assertNotIn("eightwaves", cfg.name, cfg.name)


def _header_text(cfg):
    """Codegen the header for a config and return its text (no hipcc)."""
    tmp = Path(tempfile.mkdtemp(prefix="abq_test_"))
    hpp = _generate_abquant_kernel(cfg, tmp)
    assert hpp is not None, f"codegen failed for {cfg.name}"
    return hpp.read_text()


def _static_bool(text, field):
    m = re.search(rf"bool\s+{field}\s*=\s*(\w+)", text)
    assert m, f"{field} not found in generated header"
    return m.group(1) == "true"


class TestPreshuffleBMatrixShuffle(unittest.TestCase):
    """Round-3 BUG #1: PreshuffleB kernels must pre-shuffle the B WEIGHT matrix
    (Old-TE shuffle_b / shuffle_b_permuteN, run_gemm_quant_example.inc:770-789).
    Previously only the AQ/BQ scale tensors were shuffled, so all 6 preshuffleb
    families failed on gfx950 (max_rel ~50-78)."""

    def test_ctypes_lib_has_b_matrix_shuffle_step(self):
        # The ctypes lib must call shuffle_b / shuffle_b_permuteN on B for
        # PreshuffleB kernels, gated by SelectedKernel::PreshuffleB.
        self.assertIn("SelectedKernel::PreshuffleB", _CTYPES_SRC)
        self.assertIn("shuffle_b<typename SelectedKernel::BShuffleConfig>", _CTYPES_SRC)
        self.assertIn(
            "shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>", _CTYPES_SRC
        )
        # permute_n variant is selected exactly when TiledMMAPermuteN && kN==1.
        self.assertIn("SelectedKernel::TiledMMAPermuteN", _CTYPES_SRC)
        self.assertIn("BGroupSizeN == 1", _CTYPES_SRC)

    def test_preshuffleb_headers_expose_bshuffle_config(self):
        preshuffleb_ctors = [
            lambda: default_fp8_preshuffleb_config(bquant_group_n=1),
            lambda: default_fp8_preshuffleb_config(bquant_group_n=128),
            lambda: default_bf8_preshuffleb_config(bquant_group_n=1),
            lambda: default_bf8_preshuffleb_config(bquant_group_n=128),
            lambda: default_fp4_preshuffleb_config(),
            lambda: default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=1),
        ]
        for ctor in preshuffleb_ctors:
            cfg = ctor()
            self.assertTrue(cfg.preshuffle_b, cfg.name)
            text = _header_text(cfg)
            self.assertTrue(_static_bool(text, "PreshuffleB"), cfg.name)
            self.assertIn("struct BShuffleConfig", text, cfg.name)
            # BShuffleConfig must expose the member names shuffle_b expects.
            for member in ("N_Tile", "N_Warp", "N_Warp_Tile", "K_Warp_Tile"):
                self.assertIn(member, text, f"{member} missing in {cfg.name}")

    def test_non_preshuffleb_kernels_still_no_b_shuffle(self):
        # Non-preshuffleB kernels must NOT pre-shuffle B (PreshuffleB=false).
        for ctor in (default_fp8_config, default_bf8_config, default_fp4_config,
                     default_fp8_preshufflequant_config):
            cfg = ctor()
            self.assertFalse(cfg.preshuffle_b, cfg.name)
            self.assertFalse(_static_bool(_header_text(cfg), "PreshuffleB"), cfg.name)


class TestBqPermuteNForPermuteNKernels(unittest.TestCase):
    """Round-4 BUG: the permute_n kernel (PreshuffleB && TiledMMAPermuteN &&
    BGroupSizeN==1, e.g. fp8 preshuffleb+pq n=1) riffles the N columns in its
    B-epilogue, so the BQ scale tensor must ALSO be permuted with bq_permuteN,
    IN ADDITION to shuffle_b_permuteN on B (Old-TE run_gemm_quant_example.inc:
    799-814). Without it col 0 is exact but every other N column is scrambled.
    The ctypes lib previously applied only shuffle_bq (never bq_permuteN) to BQ."""

    def test_ctypes_lib_applies_bq_permuteN(self):
        # bq_permuteN must be invoked on the BQ scale tensor, using the same
        # BShuffleConfig that shuffle_b_permuteN uses. BQ prep is shared with the
        # bquant bridge, so the call lives in quant_bridge_shuffle.hpp.
        self.assertIn("bq_permuteN<typename KernelT::BShuffleConfig>", _SHUFFLE_SRC)
        self.assertIn("prepare_bq_device<SelectedKernel,", _CTYPES_SRC)

    def test_bq_permute_n_gated_on_permute_n_predicate(self):
        # The BQ permute must be gated by exactly PreshuffleB && TiledMMAPermuteN
        # && BGroupSizeN==1 (the permute_n kernel), matching the B-matrix path.
        # The first two conjuncts are the shared helper's gate; the third arrives
        # as its GroupN template argument, which abquant binds to BGroupSizeN.
        self.assertIn("KernelT::PreshuffleB && KernelT::TiledMMAPermuteN", _SHUFFLE_SRC)
        self.assertIn("(GroupN == 1)", _SHUFFLE_SRC)
        self.assertIn("use_permute_n", _SHUFFLE_SRC)
        self.assertIn("BQuantGroupSize::kK, BGroupSizeN", _CTYPES_SRC)

    def test_bq_permute_n_then_shuffle_bq_when_preshufflequant(self):
        # For permute_n + BPreshuffleQuant kernels, bq_permuteN is applied FIRST,
        # then shuffle_bq -- exactly Old-TE inc:805-810. Both calls must appear.
        self.assertIn("bq_permuteN", _SHUFFLE_SRC)
        self.assertIn("shuffle_bq", _SHUFFLE_SRC)

    def test_preshuffleb_config_is_cshuffle_not_permute_n(self):
        # PreshuffleB (and EightWaves) use TransposeC=true, which is incompatible
        # with the PermuteN epilogue, so abquant_effective_epilogue always emits
        # cshuffle for preshuffleb kernels -- never permute_n -- regardless of
        # bquant_group_n (see abquant_effective_epilogue in codegen_common.py).
        cfg = default_fp8_preshuffleb_preshufflequant_config(
            bquant_group_n=1, gfx_arch="gfx950"
        )
        self.assertEqual(cfg.bquant_group_n, 1, cfg.name)
        self.assertTrue(cfg.preshuffle_b and cfg.preshuffle_bquant, cfg.name)
        self.assertIn("cshuffle", cfg.name, cfg.name)
        self.assertNotIn("permute_n", cfg.name, cfg.name)
class TestRunnerNoPostHocPermuteN(unittest.TestCase):
    """Round-5 FIX: now that round-4's bq_permuteN makes the kernel/ctypes
    epilogue write C in correct logical column order for permute_n kernels,
    ABQuantGpuGemmRunner.run() must NOT apply an extra post-hoc permute_n
    de-permute on C. The old wrapper-side r-group riffle undo double-corrected
    the (already-correct) ctypes output and scrambled 4/256 columns for
    permute_n kernels. run() must now pass the ctypes output through unchanged."""

    def test_runner_has_no_post_hoc_c_permute(self):
        runner_src = (_DISP / "python" / "gemm_abquant_utils.py").read_text()
        # The obsolete de-permute built a reindexed copy of C via an r-group
        # riffle; none of that machinery may remain in the runner.
        self.assertNotIn("_Cp[:, _logical] = C", runner_src)
        self.assertNotIn("np.empty_like(C)", runner_src)
        self.assertNotIn("_logical", runner_src)
        self.assertNotIn("(c % _r) * _half", runner_src)

    def test_runner_returns_ctypes_c_unchanged(self):
        # run() must return the same C object it handed to the ctypes lib --
        # no reassignment of C between the ctypes call and the return.
        import inspect
        from gemm_abquant_utils import ABQuantGpuGemmRunner

        src = inspect.getsource(ABQuantGpuGemmRunner.run)
        # Grab the tail from the ctypes self._lib.run(...) call to the return.
        after = src[src.index("self._lib.run("):]
        # No re-binding of C (e.g. "C = _Cp" / "C = np...") after the kernel runs.
        self.assertNotRegex(after, r"\n\s*C\s*=\s")
        # And it still returns C in the result.
        self.assertIn("ABQuantGemmResult(C=C", after)


class TestFp4PreshuffleBReject(unittest.TestCase):
    """Round-4 SHOULD-FIX: fp4 + PreshuffleB is unsupported. Old-TE THROWS
    ("Preshuffling weight matrix is not supported for ... bf16_fp4_gemm",
    run_gemm_quant_example.inc:994-1001); the bridge previously malloc-aborted.
    The ctypes lib now returns error code -3 BEFORE any device alloc, and the
    Python runner raises a clear RuntimeError instead of crashing."""

    def test_ctypes_lib_rejects_fp4_preshuffleb(self):
        # Compile-time guard: fp4 (pk_fp4_t) + PreshuffleB returns -3 early.
        self.assertIn("std::is_same_v<BDataType, ck_tile::pk_fp4_t>", _CTYPES_SRC)
        self.assertIn("return -3;", _CTYPES_SRC)
        self.assertIn("SelectedKernel::PreshuffleB &&", _CTYPES_SRC)

    def test_python_runner_maps_rc_minus3_to_clear_error(self):
        runner_src = (_DISP / "python" / "gemm_abquant_utils.py").read_text()
        self.assertIn("rc == -3", runner_src)
        self.assertIn("not supported for bf16_fp4_gemm", runner_src)


class TestEightWavesColumnMajorAQ(unittest.TestCase):
    """Round-3 BUG #2: the n=128 EightWaves kernels must use AQLayout=ColumnMajor
    (StrideAQ=M), matching Old-TE (run_gemm_quant_example.inc:1013-1021). The n=1
    EightWaves kernels stay RowMajor. Wrong AQ layout builds a slower kernel
    (fp8/bf8 EightWaves n=128 were +9..25% on gfx950)."""

    def test_n128_eightwaves_use_column_major_aq(self):
        for ctor in (default_fp8_config, default_bf8_config):
            cfg = ctor(bquant_group_n=128, gfx_arch="gfx950")
            self.assertTrue(cfg.eight_waves, cfg.name)
            text = _header_text(cfg)
            self.assertTrue(_static_bool(text, "AQIsColumnMajor"), cfg.name)
            self.assertIn(
                "using AQLayout = ck_tile::tensor_layout::gemm::ColumnMajor", text, cfg.name
            )
            # Python side must agree so it supplies StrideAQ=M / col-major AQ.
            self.assertTrue(ABQuantDispatcherLib.kernel_uses_column_major_aq(cfg.name), cfg.name)

        # preshuffleb EightWaves n=128 is also ColumnMajor AQ.
        for ctor in (default_fp8_preshuffleb_config, default_bf8_preshuffleb_config):
            cfg = ctor(bquant_group_n=128, gfx_arch="gfx950")
            self.assertTrue(cfg.eight_waves, cfg.name)
            self.assertTrue(_static_bool(_header_text(cfg), "AQIsColumnMajor"), cfg.name)
            self.assertTrue(ABQuantDispatcherLib.kernel_uses_column_major_aq(cfg.name), cfg.name)

    def test_n1_eightwaves_stay_row_major_aq(self):
        for ctor in (default_fp8_preshuffleb_config, default_bf8_preshuffleb_config):
            cfg = ctor(bquant_group_n=1, gfx_arch="gfx950")
            self.assertTrue(cfg.eight_waves, cfg.name)
            text = _header_text(cfg)
            self.assertFalse(_static_bool(text, "AQIsColumnMajor"), cfg.name)
            self.assertIn(
                "using AQLayout = ck_tile::tensor_layout::gemm::RowMajor", text, cfg.name
            )
            self.assertFalse(ABQuantDispatcherLib.kernel_uses_column_major_aq(cfg.name), cfg.name)

    def test_non_eightwaves_stay_row_major_aq(self):
        # All non-EightWaves kernels (fp8 n=1, fp4, all preshufflequant) use
        # RowMajor AQ regardless of arch.
        non_ew = [
            default_fp8_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp4_config(gfx_arch="gfx950"),
            default_fp4_preshuffleb_config(gfx_arch="gfx950"),
            default_fp8_preshufflequant_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshufflequant_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=128, gfx_arch="gfx950"),
        ]
        for cfg in non_ew:
            self.assertFalse(cfg.eight_waves, cfg.name)
            self.assertFalse(_static_bool(_header_text(cfg), "AQIsColumnMajor"), cfg.name)
            self.assertFalse(ABQuantDispatcherLib.kernel_uses_column_major_aq(cfg.name), cfg.name)

    def test_ctypes_lib_derives_column_major_aq_stride(self):
        # The ctypes stride check must use M for ColumnMajor AQ, QK_A otherwise.
        self.assertIn("SelectedKernel::AQIsColumnMajor ? M : QK_A", _CTYPES_SRC)


if __name__ == "__main__":
    unittest.main()
