#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for grouped_gemm_bquant_utils.py.

Tests kernel name generation, config serialization, and problem dimension helpers.
No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_grouped_gemm_bquant_utils.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_bquant_utils import (
    BQuantKernelConfig,
    BQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp8i4_preshuffleb_config,
    default_bf8i4_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_bf8_preshufflequant_config,
    default_fp8i4_preshufflequant_config,
    default_bf8i4_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
    default_bf8_preshuffleb_bquant_config,
    default_fp8i4_preshuffleb_bquant_config,
    default_bf8i4_preshuffleb_bquant_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
)


# =============================================================================
# BQuantKernelConfig.name — byte-exact match with codegen KERNEL_NAME
# =============================================================================


class TestKernelName:

    def test_fp8_rcr_default_name(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_bquant_fp8_rcr_compv3_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_qg1x1x128"
        )

    def test_bf8_rcr_default_name(self):
        cfg = BQuantKernelConfig(
            variant_key="bf8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_bquant_bf8_rcr_compv3_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_qg1x1x128"
        )

    def test_different_quant_groups_produce_different_names(self):
        def make(gk, gn):
            return BQuantKernelConfig(
                variant_key="fp8", layout="rcr",
                pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
                tile_m=16, tile_n=64, tile_k=256,
                warp_m=1, warp_n=4, warp_k=1,
                warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
                quant_group_m=1, quant_group_n=gn, quant_group_k=gk,
            ).name

        names = [make(64, 1), make(128, 1), make(128, 8), make(128, 128)]
        assert len(names) == len(set(names)), "All quant-group variants must have unique names"

    def test_preshuffle_b_suffix(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_b=True,
        )
        assert cfg.name.endswith("_preshuffleb")

    def test_preshuffle_bquant_suffix(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_b=True, preshuffle_bquant=True,
        )
        assert "_preshuffleb_preshufflebq" in cfg.name

    def test_preshuffle_bquant_only_suffix(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_bquant=True,
        )
        # preshuffle_bquant without preshuffle_b: _preshufflebq present but not _preshuffleb_
        # Use word-boundary check: the suffix "_preshuffleb" must not appear as a standalone token
        name = cfg.name
        assert "_preshufflebq" in name
        # "_preshuffleb" is a prefix of "_preshufflebq" — check it's not a standalone suffix
        assert not name.endswith("_preshuffleb")

    def test_name_no_spaces(self):
        cfg = default_fp8_config()
        assert " " not in cfg.name

    def test_name_only_valid_chars(self):
        import re
        cfg = default_fp8_config()
        assert re.match(r'^[a-z0-9_]+$', cfg.name), f"Invalid chars in name: {cfg.name}"

    def test_default_fp8_config_name(self):
        cfg = default_fp8_config(quant_group_k=128)
        assert "fp8" in cfg.name
        assert "qg1x1x128" in cfg.name

    def test_default_bf8_config_name(self):
        cfg = default_bf8_config(quant_group_k=128)
        assert "bf8" in cfg.name
        assert "qg1x1x128" in cfg.name

    def test_fp8i4_rcr_default_name(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8i4",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_bquant_fp8i4_rcr_compv3_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_qg1x1x128"
        )

    def test_bf8i4_rcr_default_name(self):
        cfg = BQuantKernelConfig(
            variant_key="bf8i4",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_bquant_bf8i4_rcr_compv3_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_qg1x1x128"
        )

    def test_default_fp8i4_config_name(self):
        cfg = default_fp8i4_config(quant_group_k=128)
        assert "fp8i4" in cfg.name
        assert "qg1x1x128" in cfg.name

    def test_default_bf8i4_config_name(self):
        cfg = default_bf8i4_config(quant_group_k=128)
        assert "bf8i4" in cfg.name
        assert "qg1x1x128" in cfg.name

    def test_fp8i4_quant_group64_name(self):
        cfg = default_fp8i4_config(quant_group_k=64)
        assert "qg1x1x64" in cfg.name

    def test_bf8i4_quant_groupn8_name(self):
        cfg = default_bf8i4_config(quant_group_k=128, quant_group_n=8)
        assert "qg1x8x128" in cfg.name


# =============================================================================
# BQuantKernelConfig.to_codegen_config — round-trip shape
# =============================================================================


class TestCodegenConfig:

    def test_codegen_config_contains_correct_variant(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["variant_keys"] == ["fp8"]

    def test_codegen_config_tile_roundtrip(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=32, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=4, quant_group_k=64,
        )
        d = cfg.to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 32
        assert tc["tile_n"] == 128
        assert tc["tile_k"] == 64
        assert tc["warp_m"] == 2
        qg = d["quant_groups"][0]
        assert qg["quant_group_k"] == 64
        assert qg["quant_group_n"] == 4

    def test_codegen_config_single_layout(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["layouts"] == ["rcr"]


# =============================================================================
# BQuantGemmProblem dimension helpers
# =============================================================================


class TestBQuantGemmProblem:

    def test_QK_B_exact_multiple(self):
        p = BQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert p.QK_B == 2   # 256 / 128

    def test_QK_B_ceil(self):
        p = BQuantGemmProblem(M=16, N=64, K=300, quant_group_k=128)
        assert p.QK_B == math.ceil(300 / 128)  # == 3

    def test_QN_B_exact_multiple(self):
        p = BQuantGemmProblem(M=16, N=64, K=256, quant_group_n=32)
        assert p.QN_B == 2   # 64 / 32

    def test_QN_B_default_one_group(self):
        p = BQuantGemmProblem(M=16, N=64, K=256)
        assert p.QN_B == 64  # default quant_group_n=1 → every column is its own group

    def test_QN_B_ceil(self):
        p = BQuantGemmProblem(M=16, N=65, K=256, quant_group_n=32)
        assert p.QN_B == math.ceil(65 / 32)  # == 3

    def test_default_k_batch(self):
        p = BQuantGemmProblem(M=16, N=64, K=256)
        assert p.k_batch == 1


# =============================================================================
# Phase 3 infrastructure — preshuffle fields and new pipeline key
# =============================================================================


class TestPhase3Infrastructure:

    def test_double_smem_buffer_defaults_false(self):
        cfg = default_fp8_config()
        assert cfg.double_smem_buffer is False

    def test_k_block_per_cu_defaults_one(self):
        cfg = default_fp8_config()
        assert cfg.k_block_per_cu == 1

    def test_preshuffleb_pipeline_accepted(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="preshuffleb", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_b=True, double_smem_buffer=True, k_block_per_cu=2,
        )
        assert "preshuffleb" in cfg.name
        assert cfg.double_smem_buffer is True
        assert cfg.k_block_per_cu == 2

    def test_codegen_config_carries_preshuffle_fields(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="preshuffleb", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_b=True, double_smem_buffer=True, k_block_per_cu=2,
        )
        d = cfg.to_codegen_config()
        assert d["preshuffle_b"] is True
        assert d["preshuffle_bquant"] is False
        assert d["double_smem_buffer"] is True
        assert d["k_block_per_cu"] == 2

    def test_codegen_config_default_preshuffle_fields_false(self):
        d = default_fp8_config().to_codegen_config()
        assert d["preshuffle_b"] is False
        assert d["preshuffle_bquant"] is False
        assert d["double_smem_buffer"] is False
        assert d["k_block_per_cu"] == 1


# =============================================================================
# Phase 3a/b/c — preshuffle convenience configs
# =============================================================================


class TestPhase3Configs:

    # ---- 3a: preshuffle_b only ----

    def test_fp8_preshuffleb_name(self):
        cfg = default_fp8_preshuffleb_config()
        assert cfg.name == (
            "grouped_gemm_bquant_fp8_rcr_preshuffleb_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x128_qg1x1x128_preshuffleb"
        )

    def test_bf8_preshuffleb_name(self):
        cfg = default_bf8_preshuffleb_config()
        assert cfg.name == (
            "grouped_gemm_bquant_bf8_rcr_preshuffleb_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x128_qg1x1x128_preshuffleb"
        )

    def test_fp8i4_preshuffleb_name(self):
        cfg = default_fp8i4_preshuffleb_config()
        assert cfg.name == (
            "grouped_gemm_bquant_fp8i4_rcr_preshuffleb_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x32_qg1x1x128_preshuffleb"
        )

    def test_bf8i4_preshuffleb_name(self):
        cfg = default_bf8i4_preshuffleb_config()
        assert cfg.name == (
            "grouped_gemm_bquant_bf8i4_rcr_preshuffleb_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x32_qg1x1x128_preshuffleb"
        )

    def test_preshuffleb_flags(self):
        cfg = default_fp8_preshuffleb_config()
        assert cfg.preshuffle_b is True
        assert cfg.preshuffle_bquant is False
        assert cfg.double_smem_buffer is True
        assert cfg.k_block_per_cu == 2
        assert cfg.pipeline == "preshuffleb"

    def test_preshuffleb_i4_warp_tile_k(self):
        # pk_int4 is not 8-bit float → K_warp_tile=32 on gfx950
        cfg = default_fp8i4_preshuffleb_config()
        assert cfg.warp_tile_k == 32

    def test_preshuffleb_fp8_warp_tile_k(self):
        cfg = default_fp8_preshuffleb_config()
        assert cfg.warp_tile_k == 128

    # ---- 3b: preshuffle_bquant only ----

    def test_fp8_preshufflequant_name(self):
        cfg = default_fp8_preshufflequant_config()
        assert cfg.name == (
            "grouped_gemm_bquant_fp8_rcr_compv3_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x128_qg1x1x128_preshufflebq"
        )

    def test_bf8_preshufflequant_name(self):
        cfg = default_bf8_preshufflequant_config()
        assert cfg.name == (
            "grouped_gemm_bquant_bf8_rcr_compv3_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x128_qg1x1x128_preshufflebq"
        )

    def test_fp8i4_preshufflequant_name(self):
        cfg = default_fp8i4_preshufflequant_config()
        assert cfg.name == (
            "grouped_gemm_bquant_fp8i4_rcr_compv3_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x32_qg1x1x128_preshufflebq"
        )

    def test_preshufflequant_flags(self):
        cfg = default_fp8_preshufflequant_config()
        assert cfg.preshuffle_b is False
        assert cfg.preshuffle_bquant is True
        assert cfg.double_smem_buffer is False
        assert cfg.k_block_per_cu == 1
        assert cfg.pipeline == "compv3"

    def test_preshufflequant_16n_group(self):
        # 1x16x128 is the extra quant group for 3b — verify config accepts it
        cfg = default_fp8_preshufflequant_config(quant_group_n=16)
        assert "qg1x16x128" in cfg.name

    # ---- 3c: preshuffle_b + preshuffle_bquant ----

    def test_fp8_preshuffleb_bquant_name(self):
        cfg = default_fp8_preshuffleb_bquant_config()
        assert cfg.name == (
            "grouped_gemm_bquant_fp8_rcr_preshuffleb_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x128_qg1x1x128_preshuffleb_preshufflebq"
        )

    def test_bf8i4_preshuffleb_bquant_name(self):
        cfg = default_bf8i4_preshuffleb_bquant_config()
        assert cfg.name == (
            "grouped_gemm_bquant_bf8i4_rcr_preshuffleb_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x32_qg1x1x128_preshuffleb_preshufflebq"
        )

    def test_preshuffleb_bquant_flags(self):
        cfg = default_fp8_preshuffleb_bquant_config()
        assert cfg.preshuffle_b is True
        assert cfg.preshuffle_bquant is True
        assert cfg.double_smem_buffer is True
        assert cfg.k_block_per_cu == 2
        assert cfg.pipeline == "preshuffleb"

    def test_all_preshuffle_names_unique(self):
        configs = [
            default_fp8_preshuffleb_config(),
            default_bf8_preshuffleb_config(),
            default_fp8i4_preshuffleb_config(),
            default_bf8i4_preshuffleb_config(),
            default_fp8_preshufflequant_config(),
            default_bf8_preshufflequant_config(),
            default_fp8i4_preshufflequant_config(),
            default_bf8i4_preshufflequant_config(),
            default_fp8_preshuffleb_bquant_config(),
            default_bf8_preshuffleb_bquant_config(),
            default_fp8i4_preshuffleb_bquant_config(),
            default_bf8i4_preshuffleb_bquant_config(),
        ]
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), f"Duplicate: {[n for n in names if names.count(n) > 1]}"


# =============================================================================
# Phase 4 — MX microscale variants
# =============================================================================


class TestPhase4MXConfigs:

    def test_mx_bf16bf16_name(self):
        cfg = default_mx_bf16bf16_config(quant_group_k=32)
        assert cfg.name == (
            "grouped_gemm_bquant_mx_bf16bf16_rcr_microscale_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x32_qg1x1x32"
        )

    def test_mx_bf16bf8_name(self):
        cfg = default_mx_bf16bf8_config(quant_group_k=128)
        assert cfg.name == (
            "grouped_gemm_bquant_mx_bf16bf8_rcr_microscale_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x64_qg1x1x128"
        )

    def test_mx_bf16fp4_name(self):
        cfg = default_mx_bf16fp4_config(quant_group_k=32)
        assert cfg.name == (
            "grouped_gemm_bquant_mx_bf16fp4_rcr_microscale_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x32_qg1x1x32"
        )

    def test_mx_bf16bf16_warp_tile_k(self):
        # GemmConfigQuantPrefill<bf16_t>: get_k_warp_tile<bf16_t,16,false> on gfx950 = 32
        assert default_mx_bf16bf16_config().warp_tile_k == 32

    def test_mx_bf16bf8_warp_tile_k(self):
        # GemmConfigMixedPrecision: hardcoded K_Warp_Tile = 64
        assert default_mx_bf16bf8_config().warp_tile_k == 64

    def test_mx_bf16fp4_warp_tile_k(self):
        # GemmConfigQuantPrefill<bf16_t>: same as bf16bf16
        assert default_mx_bf16fp4_config().warp_tile_k == 32

    def test_mx_all_use_microscale_pipeline(self):
        for cfg in [
            default_mx_bf16bf16_config(),
            default_mx_bf16bf8_config(),
            default_mx_bf16fp4_config(),
        ]:
            assert cfg.pipeline == "microscale", f"{cfg.variant_key} should use microscale"

    def test_mx_no_preshuffle_flags(self):
        for cfg in [
            default_mx_bf16bf16_config(),
            default_mx_bf16bf8_config(),
            default_mx_bf16fp4_config(),
        ]:
            assert cfg.preshuffle_b is False
            assert cfg.preshuffle_bquant is False
            assert cfg.double_smem_buffer is False

    def test_mx_bf16bf16_quant_group_k64(self):
        cfg = default_mx_bf16bf16_config(quant_group_k=64)
        assert "qg1x1x64" in cfg.name

    def test_mx_bf16fp4_quant_group_k128(self):
        cfg = default_mx_bf16fp4_config(quant_group_k=128)
        assert "qg1x1x128" in cfg.name

    def test_mx_all_names_unique(self):
        configs = [
            default_mx_bf16bf16_config(quant_group_k=32),
            default_mx_bf16bf16_config(quant_group_k=64),
            default_mx_bf16bf8_config(quant_group_k=64),
            default_mx_bf16bf8_config(quant_group_k=128),
            default_mx_bf16fp4_config(quant_group_k=32),
            default_mx_bf16fp4_config(quant_group_k=64),
            default_mx_bf16fp4_config(quant_group_k=128),
        ]
        names = [c.name for c in configs]
        assert len(names) == len(set(names))


# =============================================================================
# Name uniqueness across a small sweep
# =============================================================================


class TestNameUniqueness:

    def _make_configs(self):
        configs = []
        for variant in ("fp8", "bf8", "fp8i4", "bf8i4"):
            for gk in (64, 128):
                for gn in (1, 8):
                    configs.append(BQuantKernelConfig(
                        variant_key=variant,
                        layout="rcr",
                        pipeline="compv3",
                        epilogue="cshuffle",
                        scheduler="intrawave",
                        tile_m=16, tile_n=64, tile_k=256,
                        warp_m=1, warp_n=4, warp_k=1,
                        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
                        quant_group_m=1, quant_group_n=gn, quant_group_k=gk,
                    ))
        return configs

    def test_all_names_unique(self):
        configs = self._make_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


# =============================================================================
# Codegen GroupSizeK — generated headers must export the quantization group size
# =============================================================================


class TestCodegenGroupSizeK:

    def _generate_header(self, quant_group_k=128):
        """Generate a kernel header string via the codegen and return it."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
        from unified_grouped_gemm_bquant_codegen import (
            BQuantKernelHeaderGenerator,
            BQuantKernelSpec,
            BQuantTileConfig,
        )
        tile = BQuantTileConfig(
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        )
        spec = BQuantKernelSpec(
            variant_key="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile=tile,
            quant_group_m=1,
            quant_group_n=1,
            quant_group_k=quant_group_k,
        )
        gen = BQuantKernelHeaderGenerator()
        return gen.generate(spec)

    def test_group_size_k_emitted_in_struct(self):
        header = self._generate_header(quant_group_k=128)
        assert "static constexpr ck_tile::index_t GroupSizeK = 128;" in header, (
            "GroupSizeK is missing from the generated SelectedKernel struct. "
            "gemm_bquant_benchmark_single.cpp references SelectedKernel::GroupSizeK "
            "and will fail to compile without it."
        )

    def test_group_size_k_matches_config(self):
        for gk in (64, 128, 256):
            header = self._generate_header(quant_group_k=gk)
            expected = f"static constexpr ck_tile::index_t GroupSizeK = {gk};"
            assert expected in header, (
                f"Expected '{expected}' in generated header for quant_group_k={gk}"
            )

    def test_group_size_k_exported_in_single_kernel_include(self):
        header = self._generate_header(quant_group_k=128)
        # The CK_TILE_SINGLE_KERNEL_INCLUDE block must also export GroupSizeK
        # so force-include users see it at global scope.
        assert "constexpr ck_tile::index_t GroupSizeK = " in header, (
            "GroupSizeK is not exported in the CK_TILE_SINGLE_KERNEL_INCLUDE block"
        )


# =============================================================================
# expand_bquant_sweep — JSON config -> BQuantKernelConfig list
# =============================================================================


class TestExpandBquantSweep:

    def _write_config(self, tmp_path, config_dict):
        import json as _json
        p = tmp_path / "test_config.json"
        p.write_text(_json.dumps(config_dict))
        return str(p)

    def test_basic_expansion(self, tmp_path):
        from grouped_gemm_bquant_utils import expand_bquant_sweep
        cfg = {
            "variant_keys": ["fp8", "bf8"],
            "layouts": ["rcr"],
            "pipeline": "compv3",
            "epilogue": "cshuffle",
            "scheduler": "intrawave",
            "tile_configs": [
                {"tile_m": 128, "tile_n": 128, "tile_k": 128,
                 "warp_m": 2, "warp_n": 2, "warp_k": 1,
                 "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16}
            ],
            "quant_groups": [
                {"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}
            ],
        }
        p = self._write_config(tmp_path, cfg)
        configs = expand_bquant_sweep(p, gfx_arch="gfx950")
        assert len(configs) == 2  # fp8 + bf8
        names = {c.name for c in configs}
        assert any("fp8" in n for n in names)
        assert any("bf8" in n for n in names)

    def test_deduplication(self, tmp_path):
        from grouped_gemm_bquant_utils import expand_bquant_sweep
        # Two identical configs should collapse to one
        tile = {"tile_m": 128, "tile_n": 128, "tile_k": 128,
                "warp_m": 2, "warp_n": 2, "warp_k": 1,
                "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16}
        cfg = {
            "variant_keys": ["fp8"],
            "layouts": ["rcr"],
            "pipeline": "compv3",
            "epilogue": "cshuffle",
            "scheduler": "intrawave",
            "tile_configs": [tile, tile],  # duplicate
            "quant_groups": [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}],
        }
        p = self._write_config(tmp_path, cfg)
        configs = expand_bquant_sweep(p)
        assert len(configs) == 1

    def test_name_matches_kernel_config_name(self, tmp_path):
        from grouped_gemm_bquant_utils import expand_bquant_sweep, BQuantKernelConfig
        cfg = {
            "variant_keys": ["fp8"],
            "layouts": ["rcr"],
            "pipeline": "compv3",
            "epilogue": "cshuffle",
            "scheduler": "intrawave",
            "tile_configs": [
                {"tile_m": 128, "tile_n": 128, "tile_k": 128,
                 "warp_m": 2, "warp_n": 2, "warp_k": 1,
                 "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16}
            ],
            "quant_groups": [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}],
        }
        p = self._write_config(tmp_path, cfg)
        configs = expand_bquant_sweep(p)
        assert len(configs) == 1
        # The name produced by expand_bquant_sweep must equal what a manually
        # constructed BQuantKernelConfig would produce.
        manual = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert configs[0].name == manual.name
