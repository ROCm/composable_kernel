#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for grouped_gemm_abquant_utils.py.

Tests kernel name generation, config serialization, and problem dimension helpers.
No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_grouped_gemm_abquant_utils.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_abquant_utils import (
    ABQuantKernelConfig,
    ABQuantGemmProblem,
    default_fp8_compv3_config,
    default_bf8_compv3_config,
    default_fp8_eightwaves_config,
    default_bf8_eightwaves_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
)


# =============================================================================
# ABQuantKernelConfig.name — byte-exact match with codegen KERNEL_NAME
# =============================================================================


class TestKernelName:

    def test_fp8_compv3_default_name(self):
        # tile_n=128, warp_n=4, warp_tile_n=16 → N_repeat=2 (even) → effective epilogue=permute_n
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            aquant_group_m=1, aquant_group_n=1, aquant_group_k=128,
            bquant_group_m=1, bquant_group_n=1, bquant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_abquant_fp8_rcr_compv3_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x16_aqg1x1x128_bqg1x1x128"
        )

    def test_bf8_compv3_default_name(self):
        # tile_n=128, warp_n=4, warp_tile_n=16 → N_repeat=2 (even) → effective epilogue=permute_n
        cfg = ABQuantKernelConfig(
            variant_key="bf8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            aquant_group_m=1, aquant_group_n=1, aquant_group_k=128,
            bquant_group_m=1, bquant_group_n=1, bquant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_abquant_bf8_rcr_compv3_permute_n_intrawave_"
            "128x128x128_1x4x1_16x16x16_aqg1x1x128_bqg1x1x128"
        )

    def test_prefix_is_grouped_gemm_abquant(self):
        for cfg in [default_fp8_compv3_config(), default_bf8_compv3_config(),
                    default_fp8_eightwaves_config(), default_fp8_preshuffleb_config()]:
            assert cfg.name.startswith("grouped_gemm_abquant_")

    def test_aqg_and_bqg_segments_present(self):
        cfg = default_fp8_compv3_config(quant_group_k=128)
        assert "aqg1x1x128" in cfg.name
        assert "bqg1x1x128" in cfg.name

    def test_different_bquant_group_n_unique_names(self):
        cfg1 = default_fp8_compv3_config(bquant_group_n=1)
        cfg2 = default_fp8_compv3_config(bquant_group_n=128)
        assert cfg1.name != cfg2.name
        assert "bqg1x1x128" in cfg1.name
        assert "bqg1x128x128" in cfg2.name

    def test_transpose_c_suffix(self):
        cfg = default_fp8_eightwaves_config()
        assert "transposec" in cfg.name

    def test_no_transpose_c_no_suffix(self):
        cfg = default_fp8_compv3_config()
        assert "transposec" not in cfg.name

    def test_preshuffleb_suffix(self):
        cfg = default_fp8_preshuffleb_config()
        assert "_preshuffleb" in cfg.name

    def test_preshuffle_aq_suffix(self):
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            aquant_group_m=1, aquant_group_n=1, aquant_group_k=128,
            bquant_group_m=1, bquant_group_n=1, bquant_group_k=128,
            preshuffle_aq=True,
        )
        assert "_preshuffleaq" in cfg.name

    def test_preshuffle_bq_suffix(self):
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            aquant_group_m=1, aquant_group_n=1, aquant_group_k=128,
            bquant_group_m=1, bquant_group_n=1, bquant_group_k=128,
            preshuffle_bq=True,
        )
        assert "_preshufflebq" in cfg.name

    def test_name_no_spaces(self):
        for cfg in [default_fp8_compv3_config(), default_fp8_eightwaves_config(),
                    default_fp8_preshuffleb_config()]:
            assert " " not in cfg.name

    def test_name_only_valid_chars(self):
        cfg = default_fp8_compv3_config()
        assert re.match(r'^[a-z0-9_x]+$', cfg.name), f"Unexpected chars in: {cfg.name}"

    def test_variant_key_in_name(self):
        assert "fp8" in default_fp8_compv3_config().name
        assert "bf8" in default_bf8_compv3_config().name

    def test_pipeline_in_name(self):
        assert "compv3" in default_fp8_compv3_config().name
        assert "eightwaves" in default_fp8_eightwaves_config().name
        assert "preshuffleb" in default_fp8_preshuffleb_config().name

    def test_all_defaults_unique_names(self):
        names = [
            default_fp8_compv3_config().name,
            default_bf8_compv3_config().name,
            default_fp8_eightwaves_config().name,
            default_bf8_eightwaves_config().name,
            default_fp8_preshuffleb_config().name,
            default_bf8_preshuffleb_config().name,
        ]
        assert len(names) == len(set(names)), "All defaults must have unique names"


# =============================================================================
# ABQuantKernelConfig — constraint validation
# =============================================================================


class TestConstraintValidation:

    def test_mismatched_kk_raises(self):
        with pytest.raises(ValueError, match="aquant_group_k"):
            ABQuantKernelConfig(
                variant_key="fp8", layout="rcr",
                pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
                tile_m=128, tile_n=128, tile_k=128,
                warp_m=1, warp_n=4, warp_k=1,
                warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
                aquant_group_m=1, aquant_group_n=1, aquant_group_k=64,   # different!
                bquant_group_m=1, bquant_group_n=1, bquant_group_k=128,
            )

    def test_matching_kk_succeeds(self):
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            aquant_group_m=1, aquant_group_n=1, aquant_group_k=64,
            bquant_group_m=1, bquant_group_n=1, bquant_group_k=64,  # same!
        )
        assert cfg.aquant_group_k == cfg.bquant_group_k


# =============================================================================
# ABQuantGemmProblem dimension properties
# =============================================================================


class TestABQuantGemmProblem:

    def test_qk_a_basic(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256, aquant_group_k=128)
        assert p.QK_A == 2

    def test_qk_b_basic(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256, bquant_group_k=128)
        assert p.QK_B == 2

    def test_qm_a_is_m_when_gm1(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256, aquant_group_m=1)
        assert p.QM_A == 128

    def test_qn_b_basic(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256, bquant_group_n=1)
        assert p.QN_B == 128

    def test_qn_b_grouped(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256, bquant_group_n=128)
        assert p.QN_B == 1

    def test_qk_a_ceil(self):
        p = ABQuantGemmProblem(M=128, N=128, K=257, aquant_group_k=128)
        assert p.QK_A == 3  # ceil(257/128)=3

    def test_qk_b_ceil(self):
        p = ABQuantGemmProblem(M=128, N=128, K=257, bquant_group_k=128)
        assert p.QK_B == 3

    def test_k_batch_default(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256)
        assert p.k_batch == 1

    def test_default_group_sizes(self):
        p = ABQuantGemmProblem(M=128, N=128, K=256)
        assert p.aquant_group_k == 128
        assert p.bquant_group_k == 128
        assert p.aquant_group_m == 1
        assert p.bquant_group_n == 1


# =============================================================================
# to_codegen_config serialization
# =============================================================================


class TestCodegenConfig:

    def test_round_trip_variant_key(self):
        cfg = default_bf8_compv3_config()
        d = cfg.to_codegen_config()
        assert d["variant_keys"] == ["bf8"]

    def test_round_trip_pipeline(self):
        cfg = default_fp8_compv3_config()
        d = cfg.to_codegen_config()
        assert d["pipeline"] == "compv3"

    def test_round_trip_aquant_groups(self):
        cfg = default_fp8_compv3_config(quant_group_k=64, bquant_group_n=4)
        d = cfg.to_codegen_config()
        aqg = d["aquant_groups"][0]
        assert aqg["aquant_group_m"] == 1
        assert aqg["aquant_group_n"] == 1
        assert aqg["aquant_group_k"] == 64

    def test_round_trip_bquant_groups(self):
        cfg = default_fp8_compv3_config(quant_group_k=64, bquant_group_n=4)
        d = cfg.to_codegen_config()
        bqg = d["bquant_groups"][0]
        assert bqg["bquant_group_n"] == 4
        assert bqg["bquant_group_k"] == 64

    def test_round_trip_preshuffle_b(self):
        cfg = default_fp8_preshuffleb_config()
        d = cfg.to_codegen_config()
        assert d["preshuffle_b"] is True
        assert d["preshuffle_aq"] is False
        assert d["preshuffle_bq"] is False

    def test_round_trip_transpose_c(self):
        cfg = default_fp8_eightwaves_config()
        d = cfg.to_codegen_config()
        assert d["transpose_c"] is True

    def test_json_serializable(self):
        cfg = default_fp8_compv3_config()
        json_str = json.dumps(cfg.to_codegen_config())
        parsed = json.loads(json_str)
        assert parsed["variant_keys"] == ["fp8"]


# =============================================================================
# Default config factories
# =============================================================================


class TestDefaultConfigs:

    def test_compv3_not_transpose_c(self):
        assert default_fp8_compv3_config().transpose_c is False
        assert default_bf8_compv3_config().transpose_c is False

    def test_eightwaves_transpose_c(self):
        assert default_fp8_eightwaves_config().transpose_c is True
        assert default_bf8_eightwaves_config().transpose_c is True

    def test_preshuffleb_preshuffle_b_set(self):
        assert default_fp8_preshuffleb_config().preshuffle_b is True
        assert default_bf8_preshuffleb_config().preshuffle_b is True

    def test_compv3_preshuffle_flags_unset(self):
        cfg = default_fp8_compv3_config()
        assert cfg.preshuffle_b is False
        assert cfg.preshuffle_aq is False
        assert cfg.preshuffle_bq is False

    def test_custom_bquant_group_n(self):
        cfg = default_fp8_compv3_config(bquant_group_n=128)
        assert cfg.bquant_group_n == 128
        assert "bqg1x128x128" in cfg.name

    def test_gfx_arch_propagated(self):
        cfg = default_fp8_compv3_config(gfx_arch="gfx942")
        assert cfg.gfx_arch == "gfx942"

    def test_aquant_and_bquant_kk_equal(self):
        for cfg in [
            default_fp8_compv3_config(),
            default_fp8_eightwaves_config(),
            default_fp8_preshuffleb_config(),
        ]:
            assert cfg.aquant_group_k == cfg.bquant_group_k


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:

    def test_qk_a_k_equals_group_k(self):
        p = ABQuantGemmProblem(M=128, N=128, K=128, aquant_group_k=128)
        assert p.QK_A == 1

    def test_name_consistency_across_instances(self):
        assert default_fp8_compv3_config().name == default_fp8_compv3_config().name

    def test_large_dimensions(self):
        p = ABQuantGemmProblem(
            M=4096, N=4096, K=8192,
            aquant_group_m=1, aquant_group_k=128,
            bquant_group_n=128, bquant_group_k=128,
        )
        assert p.QK_A == 64     # 8192/128=64
        assert p.QK_B == 64
        assert p.QN_B == 32     # 4096/128=32
        assert p.QM_A == 4096

    def test_transpose_c_in_name_only_when_true(self):
        with_tc = default_fp8_eightwaves_config()
        without_tc = default_fp8_compv3_config()
        assert "transposec" in with_tc.name
        assert "transposec" not in without_tc.name


# =============================================================================
# gfx1250 (MI400 / WMMA) default configs
# =============================================================================


from grouped_gemm_abquant_utils import (  # noqa: E402
    default_fp8_compv3_config_gfx1250,
    default_bf8_compv3_config_gfx1250,
)


class TestGfx1250Configs:
    # gfx1250 correct config was determined empirically on MI400/gfx1250:
    # ABQuant CompV3 fp8/bf8 verify only with warp_tile_k=128 (FlatMM tile);
    # warp_tile_k=32 (gfx9 MFMA) and warp_tile_k=16 (WMMA) both produce all-zeros
    # on gfx12. Still uses the standard CompV3 pipeline with transpose_c=False
    # (NOT the gfx950-native eightwaves/preshuffleb path).

    def _all(self):
        return [
            default_fp8_compv3_config_gfx1250(),
            default_bf8_compv3_config_gfx1250(),
        ]

    def test_gfx1250_uses_flatmm_warp_tile_k_128(self):
        # GPU-verified: CompV3 fp8/bf8 are correct on gfx1250 only with
        # warp_tile_k=128; warp_tile_k=32 and warp_tile_k=16 both zero out.
        for cfg in self._all():
            assert cfg.warp_tile_m == 16
            assert cfg.warp_tile_n == 16
            assert cfg.warp_tile_k == 128, f"{cfg.name} must use FlatMM warp_tile_k=128"

    def test_gfx1250_uses_compv3_not_eightwaves(self):
        # eightwaves / preshuffleb are gfx950-only; gfx1250 uses standard compv3.
        for cfg in self._all():
            assert cfg.pipeline == "compv3"
            assert cfg.transpose_c is False
            assert cfg.preshuffle_b is False

    def test_gfx1250_arch_propagated(self):
        for cfg in self._all():
            assert cfg.gfx_arch == "gfx1250"

    def test_gfx1250_names_16x16x128(self):
        for cfg in self._all():
            assert "16x16x128" in cfg.name

    def test_gfx1250_unique_names(self):
        names = [cfg.name for cfg in self._all()]
        assert len(names) == len(set(names))
