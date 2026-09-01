#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the (non-grouped) bquant GEMM TileEngine -> Dispatcher bridge.

Locks the config name format (distinct `gemm_bquant` prefix, NOT the grouped bridge), the
byte-exact codegen<->utils kernel-name contract, the codegen-JSON projection, and the
fp8/bf8/fp8i4/bf8i4 + MX(bf16bf16/bf16bf8/bf16fp4) scope with preshuffleB / preshuffleQuant
families that Old-TE gemm_bquant_quantgrouped*.cpp register. No GPU / hipcc.
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_bquant_utils import (  # noqa: E402
    NAME_PREFIX,
    BQuantGemmProblem,
    _MX_VARIANTS,
    _require_mx_arch,
    _warp_tile_k_for,
    _generate_bquant_kernel,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp8i4_preshuffleb_config,
    default_bf8i4_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
    setup_multiple_bquant_dispatchers,
)

# The ctypes lib source (checked for the B-matrix shuffle / pk_int4 permute steps,
# no GPU needed).
_CTYPES_SRC = (
    _DISP / "bindings" / "ctypes" / "gemm_bquant_ctypes_lib.cpp"
).read_text()

# Shared host-load primitives (load_host_tensor / permute_i4_inplace) live here
# after the common-layer refactor: the size-bounded packed copy and the pk_int4
# permute helper (+ its include) moved out of the per-op .cpp into this header.
_SHUFFLE_SRC = (
    _DISP / "bindings" / "ctypes" / "quant_bridge_shuffle.hpp"
).read_text()

# The Python runner source (checked for the epilogue-dependent C de-permute).
_UTILS_SRC = (_DISP / "python" / "gemm_bquant_utils.py").read_text()


def _header_text(cfg):
    """Codegen the header for a config and return its text (no hipcc)."""
    tmp = Path(tempfile.mkdtemp(prefix="bq_test_"))
    hpp = _generate_bquant_kernel(cfg, tmp)
    assert hpp is not None, f"codegen failed for {cfg.name}"
    return hpp.read_text()


def _static_bool(text, field):
    m = re.search(rf"bool\s+{field}\s*=\s*(\w+)", text)
    assert m, f"{field} not found in generated header"
    return m.group(1) == "true"

_BASE = [default_fp8_config, default_bf8_config, default_fp8i4_config, default_bf8i4_config]
_MX = [default_mx_bf16bf16_config, default_mx_bf16bf8_config, default_mx_bf16fp4_config]
_ALL = _BASE + [
    default_fp8_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
] + _MX


class TestPrefix(unittest.TestCase):
    # The gemm_bquant_<variant> prefix loop and the byte-exact codegen<->utils
    # kernel-name contract for every _ALL ctor are exercised by the shared
    # parametrized tests in test_quant_bridge_shared.py (driven by
    # _quant_bridge_descriptors.py, which uses NAME_PREFIX in the name builder).
    # The NAME_PREFIX-constant value and the distinct "not grouped_" namespace
    # guard stay here.
    def test_name_prefix_constant(self):
        self.assertEqual(NAME_PREFIX, "gemm_bquant")

    def test_not_grouped_prefix(self):
        # Must NOT collide with the grouped_gemm_bquant bridge namespace.
        for ctor in _ALL:
            self.assertFalse(ctor().name.startswith("grouped_"), ctor().name)


class TestScope(unittest.TestCase):
    def test_base_variants(self):
        self.assertEqual([c().variant_key for c in _BASE],
                         ["fp8", "bf8", "fp8i4", "bf8i4"])

    def test_layout_is_rcr(self):
        for ctor in _ALL:
            self.assertEqual(ctor().layout, "rcr")

    def test_mx_pipeline_is_microscale(self):
        for ctor in _MX:
            self.assertEqual(ctor().pipeline, "microscale")

    def test_preshuffle_flags(self):
        self.assertFalse(default_fp8_config().preshuffle_b)
        self.assertTrue(default_fp8_preshuffleb_config().preshuffle_b)
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_bquant)
        pq = default_fp8_preshuffleb_bquant_config()
        self.assertTrue(pq.preshuffle_b and pq.preshuffle_bquant)


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = BQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)
        self.assertEqual(p.quant_group_k, 128)


class TestArchSafety(unittest.TestCase):
    """Round-2 arch-safety hardening (get_arch+throw)."""

    def test_mx_requires_gfx950(self):
        # Every MX variant must reject a non-gfx950 arch with a clear error.
        for v in sorted(_MX_VARIANTS):
            with self.assertRaises(ValueError):
                _require_mx_arch(v, "gfx942")
        # gfx950 is accepted (no raise).
        for v in sorted(_MX_VARIANTS):
            _require_mx_arch(v, "gfx950")

    def test_non_mx_variant_any_arch_ok(self):
        # Non-MX variants are not restricted by the MX guard.
        for v in ("fp8", "bf8", "fp8i4", "bf8i4"):
            _require_mx_arch(v, "gfx942")
            _require_mx_arch(v, "gfx950")

    def test_setup_rejects_mx_on_non_gfx950(self):
        # The build entry point must fail early (before hipcc) for MX on gfx942.
        cfg = default_mx_bf16bf16_config(gfx_arch="gfx942")
        with self.assertRaises(ValueError):
            setup_multiple_bquant_dispatchers([cfg], gfx_arch="gfx942")


class TestArchAwareWarpTileK(unittest.TestCase):
    """Round-4: warp_tile_k must be arch-derived, mirroring get_k_warp_tile.

    The fp8/bf8 (and i4, which instantiate an 8-bit-float PrecType) default
    configs previously hardcoded warp_tile_k=128, which is a gfx950-only value.
    On gfx942 a warp_tile_k=128 fp8/bf8 kernel *compiles* but silently outputs
    ALL-ZEROS (there is no valid 16x16x128 fp8/bf8 warp-gemm on gfx942) -- the
    same trap already GPU-confirmed on the sibling tensor_quant/rowcolquant/
    aquant/abquant bridges.  So warp_tile_k MUST be 32 (decode) / 64 (preshuffle_b)
    on gfx942 and 128 on gfx950, and that value must flow into the byte-exact .name.
    """

    def test_helper_decode(self):
        # IsFlatMM=false (decode / preshufflequant): 128 gfx950, 32 gfx942.
        self.assertEqual(_warp_tile_k_for("gfx942"), 32)
        self.assertEqual(_warp_tile_k_for("gfx950"), 128)
        # Arch strings with feature suffixes must still resolve.
        self.assertEqual(_warp_tile_k_for("gfx942:sramecc+:xnack-"), 32)
        self.assertEqual(_warp_tile_k_for("gfx950:sramecc+:xnack-"), 128)

    def test_helper_preshuffleb_flatmm(self):
        # IsFlatMM=true (preshuffle_b): 128 gfx950, 64 gfx942.
        self.assertEqual(_warp_tile_k_for("gfx942", is_flatmm=True), 64)
        self.assertEqual(_warp_tile_k_for("gfx950", is_flatmm=True), 128)

    def test_decode_configs_arch_aware(self):
        # fp8/bf8 AND fp8i4/bf8i4 decode: 32 on gfx942, 128 on gfx950.
        for ctor in (default_fp8_config, default_bf8_config,
                     default_fp8i4_config, default_bf8i4_config):
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 32, ctor.__name__)
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_preshufflequant_configs_arch_aware(self):
        # preshuffle_bquant (IsFlatMM=false): 32 gfx942, 128 gfx950.
        for ctor in (default_fp8_preshufflequant_config,):
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 32, ctor.__name__)
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_preshuffleb_configs_arch_aware(self):
        # preshuffle_b (IsFlatMM=true): 64 gfx942, 128 gfx950 (fp8 + i4).
        for ctor in (default_fp8_preshuffleb_config,
                     default_fp8i4_preshuffleb_config,
                     default_bf8i4_preshuffleb_config,
                     default_fp8_preshuffleb_bquant_config):
            self.assertEqual(ctor(gfx_arch="gfx942").warp_tile_k, 64, ctor.__name__)
            self.assertEqual(ctor(gfx_arch="gfx950").warp_tile_k, 128, ctor.__name__)

    def test_warp_tile_k_flows_into_name(self):
        # The chosen warp_tile_k must appear byte-exact in the kernel .name.
        n942 = default_fp8_config(gfx_arch="gfx942").name
        n950 = default_fp8_config(gfx_arch="gfx950").name
        self.assertIn("16x16x32", n942)
        self.assertIn("16x16x128", n950)
        self.assertNotEqual(n942, n950)

    def test_mx_gfx950_values(self):
        # MX is gfx950-only; verified against Old-TE get_k_warp_tile<bf16,16>()
        # (=32 for bf16bf16/bf16fp4) and GemmConfigMixedPrecision (=64 for bf16bf8).
        self.assertEqual(default_mx_bf16bf16_config(gfx_arch="gfx950").warp_tile_k, 32)
        self.assertEqual(default_mx_bf16fp4_config(gfx_arch="gfx950").warp_tile_k, 32)
        self.assertEqual(default_mx_bf16bf8_config(gfx_arch="gfx950").warp_tile_k, 64)


class TestSplitKTrap(unittest.TestCase):
    """k_batch > 1 must be rejected, never silently passed through."""

    def test_kbatch_gt1_rejected(self):
        # Exercise the runner's guard without a GPU by stubbing the ctypes lib.
        import gemm_bquant_utils as gbu

        runner = gbu.BQuantGpuGemmRunner.__new__(gbu.BQuantGpuGemmRunner)
        runner._lib = None  # guard must fire before any lib access

        prob = BQuantGemmProblem(M=16, N=64, K=256, k_batch=2)
        with self.assertRaises(ValueError):
            runner.run(A=None, B=None, BQ=None, problem=prob)


class TestPreshuffleBMatrixShuffle(unittest.TestCase):
    """Round-3 BUG #1: PreshuffleB kernels must pre-shuffle the B WEIGHT matrix
    (Old-TE shuffle_b / shuffle_b_permuteN, run_gemm_quant_example.inc:770-789).
    Previously the ctypes lib only plain-copied B, so fp8/bf8 preshuffleb (+pq)
    returned garbage (max_rel ~67-69 on gfx950). The bq_permuteN path for the BQ
    scales (inc:799-815) must be applied too."""

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
        self.assertIn("QuantGroupSize::kN == 1", _CTYPES_SRC)
        # The BQ scales must also be bq_permuteN'd for the permuteN case. BQ prep
        # is shared with the abquant bridge, so that call lives in the header;
        # the .cpp binds GroupN to QuantGroupSize::kN at the call site.
        self.assertIn("bq_permuteN<typename KernelT::BShuffleConfig>", _SHUFFLE_SRC)
        self.assertIn(
            "prepare_bq_device<SelectedKernel, QuantGroupSize::kK, QuantGroupSize::kN>",
            _CTYPES_SRC,
        )

    def test_preshuffleb_headers_expose_bshuffle_config(self):
        preshuffleb_ctors = [
            default_fp8_preshuffleb_config,
            default_bf8_preshuffleb_config,
            default_fp8i4_preshuffleb_config,
            default_bf8i4_preshuffleb_config,
            default_fp8_preshuffleb_bquant_config,
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
            # preshuffleb default (tile_n=128, warp_n=4, warp_tile_n=16):
            # N_Repeat = 128/16/4 = 2 -> TiledMMAPermuteN true.
            self.assertTrue(_static_bool(text, "TiledMMAPermuteN"), cfg.name)

    def test_non_preshuffleb_kernels_have_no_b_shuffle(self):
        # Non-preshuffleB kernels must NOT pre-shuffle B (PreshuffleB=false,
        # TiledMMAPermuteN=false).
        for ctor in (default_fp8_config, default_bf8_config, default_fp8i4_config,
                     default_bf8i4_config, default_fp8_preshufflequant_config,
                     default_mx_bf16bf16_config):
            cfg = ctor()
            self.assertFalse(cfg.preshuffle_b, cfg.name)
            text = _header_text(cfg)
            self.assertFalse(_static_bool(text, "PreshuffleB"), cfg.name)
            self.assertFalse(_static_bool(text, "TiledMMAPermuteN"), cfg.name)


class TestPkInt4Permute(unittest.TestCase):
    """Round-3 BUG #2: pk_int4 B (fp8i4 / bf8i4) must be permuted with
    permute_vectors_i4x4_b UNCONDITIONALLY before the device copy, exactly as
    Old-TE does (run_gemm_quant_example.inc:784-787). Without it fp8i4/bf8i4 were
    broken in all phases (NaN on random, all-zeros on constant)."""

    def test_ctypes_lib_permutes_pk_int4_b(self):
        # The permute primitive + its include live in the shared shuffle header;
        # the per-op .cpp invokes it via permute_i4_inplace, gated on pk_int4 B.
        self.assertIn("permute_vectors_i4x4_b", _SHUFFLE_SRC)
        self.assertIn("ck_tile/host/permute_pk_int4.hpp", _SHUFFLE_SRC)
        self.assertIn("permute_i4_inplace", _CTYPES_SRC)
        # Applied for pk_int4 B specifically.
        self.assertIn("std::is_same_v<BDataType, ck_tile::pk_int4_t>", _CTYPES_SRC)

    def test_i4_variants_use_pk_int4_bdatatype(self):
        for ctor in (default_fp8i4_config, default_bf8i4_config,
                     default_fp8i4_preshuffleb_config, default_bf8i4_preshuffleb_config):
            cfg = ctor()
            text = _header_text(cfg)
            self.assertIn("using BDataType   = ck_tile::pk_int4_t", text, cfg.name)


class TestBCastPolicy(unittest.TestCase):
    """Round-3 BUG #3: mx_bf16bf16 (and every A==B kernel) must compile the same
    pipeline Old-TE uses. Old-TE (run_gemm_quant_example.inc:117-120) sets
    b_cast_policy = (A==B) ? BeforeLDSWrite : AfterLDSRead. The bridge previously
    left the GemmBQuantPipelineProblem BCastPolicy_ arg at its AfterLDSRead default
    for every kernel, so mx_bf16bf16 built a slower pipeline (~43% off on gfx950)."""

    def test_same_dtype_kernels_use_before_lds_write(self):
        # A == B: fp8/fp8, bf8/bf8, mx bf16/bf16.
        for ctor in (default_fp8_config, default_bf8_config, default_mx_bf16bf16_config):
            text = _header_text(ctor())
            self.assertIn("ck_tile::CastPolicy::BeforeLDSWrite", text, ctor().name)
            self.assertNotIn("ck_tile::CastPolicy::AfterLDSRead", text, ctor().name)

    def test_mixed_dtype_kernels_use_after_lds_read(self):
        # A != B: fp8i4 (fp8/pk_int4), bf8i4, mx_bf16bf8 (bf16/bf8),
        # mx_bf16fp4 (bf16/pk_fp4).
        for ctor in (default_fp8i4_config, default_bf8i4_config,
                     default_mx_bf16bf8_config, default_mx_bf16fp4_config):
            text = _header_text(ctor())
            self.assertIn("ck_tile::CastPolicy::AfterLDSRead", text, ctor().name)
            self.assertNotIn("ck_tile::CastPolicy::BeforeLDSWrite", text, ctor().name)


class TestPackedBCopyCount(unittest.TestCase):
    """Round-5 BUG #1: for packed B (pk_int4_t / pk_fp4_t; PackedSize=2) the
    host copy into b_k_n must copy the DESTINATION element count, not K*N.
    HostTensor<T>::get_element_space_size() divides by PackedSize, so the tensor
    holds only K*N/2 elements; copying K*N overran the buffer and corrupted the
    heap BEFORE permute_vectors_i4x4_b ran, crashing all i4 (fp8i4/bf8i4) and
    mx_bf16fp4 configs."""

    def test_packed_b_copy_uses_destination_size(self):
        # The overflowing copy (B_host + K * N into b_k_n) must be gone.
        self.assertNotIn("B_host + K * N", _CTYPES_SRC)
        # The size-bounded copy now lives in load_host_tensor (shared header):
        # it must copy the destination tensor's own size, not rows*cols.
        self.assertIn("std::copy(src, src + t.size(), t.begin())", _SHUFFLE_SRC)

    def test_packed_pk_int4_permute_still_runs(self):
        # The pk_int4 permute must still be present (it only runs once the copy
        # no longer corrupts the heap). Primitive in the shared header; invoked
        # from the per-op .cpp gated on pk_int4 B.
        self.assertIn("permute_vectors_i4x4_b", _SHUFFLE_SRC)
        self.assertIn("permute_i4_inplace", _CTYPES_SRC)
        self.assertIn("std::is_same_v<BDataType, ck_tile::pk_int4_t>", _CTYPES_SRC)


class TestEpilogueDependentCDepermute(unittest.TestCase):
    """Round-6 BUG #1: the permute_n C de-permute is EPILOGUE-DEPENDENT *and*
    N-TILE-AWARE.  The round-5 code used a GLOBAL riffle (width N // r), correct
    only at N == TileN; at N >= 2*TileN it scrambled columns (gfx942/gfx950
    tester: MX max_rel 50-74 at N=256/512).  Round-6:
      * PreshuffleB (WPQuantB) kernels -> IDENTITY (no de-permute); the host-side
        shuffle_b_permuteN / bq_permuteN already put C in logical order (gfx942
        tester: any C riffle scrambled, max_rel 57-58; identity is exact).
      * CompV3 / preshufflequant / MX (microscale) -> per-TileN INVERSE riffle.
    """

    def test_utils_applies_per_tile_riffle(self):
        # The runner must scope the riffle to a TileN-wide block, iterating over N
        # in TileN steps (NOT a single global N // r split).
        self.assertIn("range(0, N, _tile_n)", _UTILS_SRC)
        # Inverse riffle (scatter) is retained for CompV3 / preshufflequant / MX.
        self.assertIn("_dst[:, _logical] = _src", _UTILS_SRC)
        # PreshuffleB must be IDENTITY now: the branch skips the riffle entirely.
        self.assertIn("if not _is_preshuffleb:", _UTILS_SRC)
        # The delimiter-aware token match distinguishes preshuffleb from the
        # "preshufflebq" preshufflequant token.
        self.assertIn(r"(?:^|_)preshuffleb(?:_|$)", _UTILS_SRC)
        # The round-5 global-width formula must be gone.
        self.assertNotIn("_half = N // _r", _UTILS_SRC)

    def test_preshuffleb_token_match_excludes_preshufflequant(self):
        # The token regex must fire for PreshuffleB names and NOT for the
        # preshufflequant ("preshufflebq") CompV3 name.
        tok = re.compile(r'(?:^|_)preshuffleb(?:_|$)')
        self.assertTrue(tok.search(default_fp8_preshuffleb_config().name))
        self.assertTrue(tok.search(default_fp8_preshuffleb_bquant_config().name))
        self.assertIsNone(tok.search(default_fp8_preshufflequant_config().name))

    def _global_depermute(self, C, tile_n, warp_n, wt_n):
        """Reimplementation of the ROUND-5 (buggy) global inverse riffle."""
        import numpy as np
        N = C.shape[1]
        r = tile_n // wt_n // warp_n
        half = N // r
        logical = [(c % r) * half + (c // r) for c in range(N)]
        out = np.empty_like(C)
        out[:, logical] = C
        return out

    def _per_tile_depermute(self, C, tile_n, warp_n, wt_n):
        """Reference of the ROUND-6 per-TileN inverse riffle (mirrors the runner)."""
        import numpy as np
        N = C.shape[1]
        r = tile_n // wt_n // warp_n
        within = tile_n // r
        logical = [(c % r) * within + (c // r) for c in range(tile_n)]
        out = np.empty_like(C)
        for n0 in range(0, N, tile_n):
            w = min(tile_n, N - n0)
            src = C[:, n0:n0 + w]
            if w == tile_n:
                dst = np.empty_like(src)
                dst[:, logical] = src
                out[:, n0:n0 + tile_n] = dst
            else:
                out[:, n0:n0 + w] = src
        return out

    def test_single_ntile_matches_round5(self):
        # At N == TileN the per-tile riffle and the old global riffle coincide
        # (that is why N=128 passed before) -- guards against a regression on the
        # single-N-tile case that was already validated.
        import numpy as np
        tile_n, warp_n, wt_n = 128, 4, 16   # NRepeat = 2
        C = np.arange(2 * tile_n).reshape(2, tile_n)
        g = self._global_depermute(C, tile_n, warp_n, wt_n)
        p = self._per_tile_depermute(C, tile_n, warp_n, wt_n)
        self.assertTrue(np.array_equal(g, p))

    def test_multi_ntile_differs_from_round5(self):
        # At N = 2*TileN and 4*TileN the two MUST differ -- this is exactly the
        # bug the round-5 global riffle had (it mixed columns across N-tiles).
        import numpy as np
        tile_n, warp_n, wt_n = 128, 4, 16
        for N in (256, 512):
            C = np.arange(2 * N).reshape(2, N)
            g = self._global_depermute(C, tile_n, warp_n, wt_n)
            p = self._per_tile_depermute(C, tile_n, warp_n, wt_n)
            self.assertFalse(np.array_equal(g, p), f"N={N}")

    def test_per_tile_is_block_diagonal(self):
        # The per-tile de-permute must never move a column out of its TileN block:
        # each output column's source must lie in the same TileN-wide slice.
        import numpy as np
        tile_n, warp_n, wt_n = 128, 4, 16
        N = 512
        # Encode each column with its tile index; after de-permute the tile index
        # of column j must still be j // tile_n.
        tile_id = (np.arange(N) // tile_n).reshape(1, N)
        out = self._per_tile_depermute(tile_id, tile_n, warp_n, wt_n)
        expected = (np.arange(N) // tile_n).reshape(1, N)
        self.assertTrue(np.array_equal(out, expected))

    def test_runner_depermute_end_to_end_multi_tile(self):
        # Drive the real runner code path (stubbed lib) with a known permuted C at
        # N=256 and confirm it recovers the logical identity per tile.
        import numpy as np
        import gemm_bquant_utils as gbu

        tile_n, warp_n, wt_n = 128, 4, 16
        r = tile_n // wt_n // warp_n
        within = tile_n // r
        N = 256
        M = 3
        # Forward-riffle a logical C the way the epilogue would (per tile), then
        # verify the runner inverts it back to logical.
        logical = [(c % r) * within + (c // r) for c in range(tile_n)]
        C_logical = np.arange(M * N).reshape(M, N).astype(np.float16)
        C_permuted = np.empty_like(C_logical)
        for n0 in range(0, N, tile_n):
            blk = C_logical[:, n0:n0 + tile_n]
            C_permuted[:, n0:n0 + tile_n] = blk[:, logical]

        runner = gbu.BQuantGpuGemmRunner.__new__(gbu.BQuantGpuGemmRunner)

        class _StubLib:
            # CompV3 permute_n kernel name (NOT preshuffleb) at tile 128x*x*.
            _name = ("gemm_bquant_fp8_rcr_compv3_permute_n_intrawave_"
                     f"16x{tile_n}x256_1x{warp_n}x1_16x{wt_n}x128_qg1x1x128")

            def __init__(self, C_ret):
                self._C_ret = C_ret

            def run(self, A, B, BQ, C, **kw):
                C[...] = self._C_ret
                return 0, 1.0

            def get_kernel_name(self):
                return self._name

        runner._lib = _StubLib(C_permuted)
        prob = BQuantGemmProblem(M=M, N=N, K=256)
        res = runner.run(A=np.zeros((M, 256)), B=np.zeros((256, N)),
                         BQ=np.ones((2, N)), problem=prob)
        self.assertTrue(np.array_equal(res.C.astype(np.float32),
                                       C_logical.astype(np.float32)))

    def test_runner_preshuffleb_is_identity(self):
        # PreshuffleB kernels must NOT de-permute: the runner returns C untouched.
        import numpy as np
        import gemm_bquant_utils as gbu

        N, M = 256, 2
        C_dev = np.arange(M * N).reshape(M, N).astype(np.float16)

        runner = gbu.BQuantGpuGemmRunner.__new__(gbu.BQuantGpuGemmRunner)

        class _StubLib:
            _name = ("gemm_bquant_fp8_rcr_preshuffleb_permute_n_intrawave_"
                     "128x128x128_1x4x1_16x16x128_qg1x1x128_preshuffleb")

            def __init__(self, C_ret):
                self._C_ret = C_ret

            def run(self, A, B, BQ, C, **kw):
                C[...] = self._C_ret
                return 0, 1.0

            def get_kernel_name(self):
                return self._name

        runner._lib = _StubLib(C_dev)
        prob = BQuantGemmProblem(M=M, N=N, K=256)
        res = runner.run(A=np.zeros((M, 256)), B=np.zeros((256, N)),
                         BQ=np.ones((2, N)), problem=prob)
        self.assertTrue(np.array_equal(res.C.astype(np.float32),
                                       C_dev.astype(np.float32)))


class TestQDataTypeAwareBQEncoding(unittest.TestCase):
    """Round-6 BUG #2: BQ must be encoded to the kernel's QDataType.
    fp8/bf8 -> float32; fp8i4 -> fp8; bf8i4 -> bf8; mx_* -> e8m0.  The round-5
    runner passed BQ as float32 for every variant, so i4 kernels reinterpreted a
    4-byte float32 as a 1-byte fp8/bf8 -> NaN in all 8 i4 configs."""

    def test_variant_extracted_from_name(self):
        import gemm_bquant_utils as gbu
        cases = {
            default_fp8_config: "fp8",
            default_bf8_config: "bf8",
            default_fp8i4_config: "fp8i4",
            default_bf8i4_config: "bf8i4",
            default_mx_bf16bf16_config: "mx_bf16bf16",
            default_mx_bf16bf8_config: "mx_bf16bf8",
            default_mx_bf16fp4_config: "mx_bf16fp4",
        }
        for ctor, expected in cases.items():
            self.assertEqual(
                gbu._variant_from_kernel_name(ctor().name), expected, ctor.__name__)

    def test_fp8_bf8_bq_stays_float32(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        bq = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
        for v in ("fp8", "bf8"):
            out = gbu._encode_bq_for_variant(bq, v)
            self.assertEqual(out.dtype, np.float32, v)

    def test_i4_bq_encoded_to_single_byte(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        bq = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
        # fp8i4 -> fp8 bytes; bf8i4 -> bf8 bytes; both must be 1 byte per scale.
        for v in ("fp8i4", "bf8i4"):
            out = gbu._encode_bq_for_variant(bq, v)
            self.assertEqual(out.dtype, np.uint8, v)
            self.assertEqual(out.shape, bq.shape, v)
            self.assertEqual(out.itemsize, 1, v)

    def test_mx_bq_encoded_to_e8m0(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        bq = np.array([[1.0, 2.0], [4.0, 0.5]], dtype=np.float32)
        for v in ("mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4"):
            out = gbu._encode_bq_for_variant(bq, v)
            self.assertEqual(out.dtype, np.uint8, v)
            # e8m0: byte == floor(log2(s)) + 127.
            self.assertEqual(int(out[0, 0]), 127, v)   # 2^0
            self.assertEqual(int(out[0, 1]), 128, v)   # 2^1
            self.assertEqual(int(out[1, 0]), 129, v)   # 2^2
            self.assertEqual(int(out[1, 1]), 126, v)   # 2^-1

    def test_prencoded_bytes_pass_through(self):
        # A caller that already handed uint8 bytes (e.g. pre-encoded MX/i4) must
        # not be double-encoded.
        import numpy as np
        import gemm_bquant_utils as gbu
        raw = np.array([[127, 128]], dtype=np.uint8)
        for v in ("mx_bf16bf16", "fp8i4", "bf8i4"):
            out = gbu._encode_bq_for_variant(raw, v)
            self.assertTrue(np.array_equal(out, raw), v)

    def test_unknown_variant_passthrough(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        bq = np.array([1.0, 2.0], dtype=np.float32)
        out = gbu._encode_bq_for_variant(bq, None)
        self.assertTrue(np.array_equal(out, bq))


class TestEpilogueGating(unittest.TestCase):
    """PermuteNEpilogue must be gated on PreshuffleB, mirroring Old-TE.

    run_gemm_quant_example.inc:208-252 selects the epilogue via
        TiledMMAPermuteN = PreshuffleB && (N_Repeat % 2 == 0)   (GemmConfig)
        TiledPermuteN    = (kN > 1) ? false : TiledMMAPermuteN
        GemmEpilogue     = TiledPermuteN ? PermuteN : CShuffle
    TiledMMAPermuteN is false in GemmConfigBase and only overridden by the
    PreshuffleB configs, so every non-PreshuffleB kernel (compv3, preshufflequant,
    MX microscale) must use CShuffleEpilogue. A prior bug omitted the PreshuffleB
    gate, making even-N_Repeat MX kernels (e.g. mx_bf16bf8 128-tile, N_Repeat=2)
    emit a PermuteNEpilogue -- a different, ~16-17% slower kernel than Old-TE.
    """

    def _emits_permute_n(self, cfg):
        txt = _header_text(cfg)
        has_permute = "PermuteNEpilogue<" in txt
        has_cshuffle = "CShuffleEpilogue<" in txt
        # Exactly one epilogue must be emitted.
        self.assertNotEqual(has_permute, has_cshuffle,
                            f"{cfg.name}: ambiguous epilogue in header")
        return has_permute

    def test_mx_configs_use_cshuffle(self):
        # MX microscale kernels are PreshuffleB=false -> CShuffle (never PermuteN).
        for ctor in _MX:
            cfg = ctor()
            self.assertFalse(cfg.preshuffle_b, cfg.name)
            self.assertFalse(self._emits_permute_n(cfg),
                             f"{cfg.name} must emit CShuffleEpilogue, not PermuteN")
            self.assertIn("_cshuffle_", cfg.name)

    def test_preshufflequant_uses_cshuffle(self):
        # preshuffle_bquant (not preshuffle_b) is still CShuffle in Old-TE.
        cfg = default_fp8_preshufflequant_config()
        self.assertFalse(cfg.preshuffle_b, cfg.name)
        self.assertFalse(self._emits_permute_n(cfg),
                         f"{cfg.name} must emit CShuffleEpilogue, not PermuteN")
        self.assertIn("_cshuffle_", cfg.name)

    def test_compv3_decode_uses_cshuffle(self):
        # Plain decode-tile fp8/bf8 (PreshuffleB=false) -> CShuffle.
        for ctor in _BASE:
            cfg = ctor()
            self.assertFalse(self._emits_permute_n(cfg),
                             f"{cfg.name} must emit CShuffleEpilogue, not PermuteN")

    def test_preshuffleb_uses_permute_n(self):
        # PreshuffleB with even N_Repeat -> PermuteNEpilogue (the ONE permute case).
        cfg = default_fp8_preshuffleb_config()
        self.assertTrue(cfg.preshuffle_b, cfg.name)
        self.assertTrue(self._emits_permute_n(cfg),
                        f"{cfg.name} must emit PermuteNEpilogue")
        self.assertIn("_permute_n_", cfg.name)


class TestADataTypeGuard(unittest.TestCase):
    """A must be coerced to the kernel's ADataType byte-width before the copy.

    The ctypes lib reads ``M*K * sizeof(ADataType)`` bytes from the host A
    pointer.  MX kernels have ADataType == bf16 (2 bytes); all other bquant
    variants have a 1-byte ADataType.  A caller that hands a 1-byte (uint8) A to
    an MX kernel used to make the .so over-read M*K bytes past the numpy
    allocation -- harmless slack at small M*K, but a host-pin failure -> SEGFAULT
    at large M*K (M=K=2048).  _coerce_a_for_variant closes that gap so the device
    copy is always in bounds regardless of the caller's A dtype.
    """

    def test_mx_promotes_1byte_a_to_2byte(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        M, K = 16, 32
        a = np.arange(M * K, dtype=np.uint8).reshape(M, K)
        for v in ("mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4"):
            out = np.asarray(gbu._coerce_a_for_variant(a, v))
            self.assertEqual(out.dtype.itemsize, 2,
                             f"{v}: A must be 2 bytes/elem (bf16), got {out.dtype}")
            self.assertEqual(out.shape, (M, K))
            # Values preserved (integer byte values are exactly representable in bf16).
            self.assertTrue(np.allclose(out.astype(np.float32),
                                        a.astype(np.float32), rtol=0, atol=0))

    def test_mx_narrows_float32_a_to_2byte(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        a = (np.arange(64, dtype=np.float32).reshape(8, 8) / 8.0)
        out = np.asarray(gbu._coerce_a_for_variant(a, "mx_bf16bf8"))
        self.assertEqual(out.dtype.itemsize, 2)

    def test_mx_passthrough_when_already_2byte(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        try:
            import ml_dtypes
            a = np.ones((4, 4), dtype=ml_dtypes.bfloat16)
        except Exception:
            a = np.ones((4, 4), dtype=np.float16)
        out = np.asarray(gbu._coerce_a_for_variant(a, "mx_bf16bf8"))
        self.assertEqual(out.dtype.itemsize, 2)

    def test_non_mx_keeps_1byte_a(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        a = np.arange(16, dtype=np.uint8).reshape(4, 4)
        for v in ("fp8", "bf8", "fp8i4", "bf8i4"):
            out = np.asarray(gbu._coerce_a_for_variant(a, v))
            self.assertEqual(out.dtype.itemsize, 1, f"{v}: A must stay 1 byte/elem")

    def test_unknown_variant_passthrough(self):
        import numpy as np
        import gemm_bquant_utils as gbu
        a = np.arange(16, dtype=np.uint8).reshape(4, 4)
        self.assertIs(gbu._coerce_a_for_variant(a, None), a)


class TestFairCodegenFlags(unittest.TestCase):
    """The bridge .so must build with the same -mllvm TE backend flags Old-TE
    uses, or the kernel codegen differs and the A/B comparison is unfair (mx
    4096^3 measured ~+24% vs Old-TE without them, ~+8% with them)."""

    def test_te_flags_present_and_coerce_probe_gated(self):
        import gemm_bquant_utils as gbu
        # The unconditional TE flag set must match Old-TE (mx_gemm_utils) exactly.
        for pair in ("-amdgpu-early-inline-all=true",
                     "-amdgpu-function-calls=false",
                     "--lsr-drop-solution=1",
                     "-enable-post-misched=0"):
            self.assertIn(pair, gbu._BQUANT_CODEGEN_FLAGS)
        self.assertIn("-fno-offload-uniform-block", gbu._BQUANT_CODEGEN_FLAGS)
        self.assertIn("--offload-compress", gbu._BQUANT_CODEGEN_FLAGS)
        # coerce-illegal-types is probe-gated (not unconditional) so the build
        # stays portable to toolchains that reject it (ROCm 7.2).
        self.assertNotIn("-amdgpu-coerce-illegal-types=1", gbu._BQUANT_CODEGEN_FLAGS)
        self.assertEqual(gbu._BQUANT_PROBED_CODEGEN_FLAGS,
                         (("-mllvm", "-amdgpu-coerce-illegal-types=1"),))


if __name__ == "__main__":
    unittest.main()
