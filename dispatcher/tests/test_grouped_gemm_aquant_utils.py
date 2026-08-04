#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for grouped_gemm_aquant_utils.py.

Tests kernel name generation, config serialization, and problem dimension helpers.
No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_grouped_gemm_aquant_utils.py -v
"""

import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_aquant_utils import (
    AQuantKernelConfig,
    AQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshuffleaq_config,
    default_bf8_preshuffleaq_config,
)


# =============================================================================
# AQuantKernelConfig.name — byte-exact match with codegen KERNEL_NAME
# =============================================================================


class TestKernelName:

    def test_fp8_rcr_mem_default_name(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8",
            layout="rcr",
            pipeline="mem",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_aquant_fp8_rcr_mem_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_aqg1x1x128"
        )

    def test_bf8_rcr_mem_default_name(self):
        cfg = AQuantKernelConfig(
            variant_key="bf8",
            layout="rcr",
            pipeline="mem",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_aquant_bf8_rcr_mem_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_aqg1x1x128"
        )

    def test_preshuffle_aq_suffix(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_aq=True,
        )
        assert cfg.name.endswith("_preshuffleaq")

    def test_no_preshuffle_no_suffix(self):
        cfg = default_fp8_config()
        assert "preshuffleaq" not in cfg.name

    def test_prefix_is_grouped_gemm_aquant(self):
        for cfg in [default_fp8_config(), default_bf8_config(),
                    default_fp8i4_config(), default_bf8i4_config()]:
            assert cfg.name.startswith("grouped_gemm_aquant_")

    def test_different_variants_unique_names(self):
        names = [
            default_fp8_config().name,
            default_bf8_config().name,
            default_fp8i4_config().name,
            default_bf8i4_config().name,
        ]
        assert len(names) == len(set(names))

    def test_different_quant_groups_unique_names(self):
        def make(gk):
            return AQuantKernelConfig(
                variant_key="fp8", layout="rcr",
                pipeline="mem", epilogue="cshuffle", scheduler="intrawave",
                tile_m=16, tile_n=64, tile_k=256,
                warp_m=1, warp_n=4, warp_k=1,
                warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
                quant_group_m=1, quant_group_n=1, quant_group_k=gk,
            ).name

        names = [make(64), make(128), make(256)]
        assert len(names) == len(set(names))

    def test_aqg_segment_present(self):
        cfg = default_fp8_config(quant_group_k=128, quant_group_m=1)
        assert "aqg1x1x128" in cfg.name

    def test_name_no_spaces(self):
        cfg = default_fp8_config()
        assert " " not in cfg.name

    def test_name_only_valid_chars(self):
        cfg = default_fp8_config()
        assert re.match(r'^[a-z0-9_x]+$', cfg.name), f"Unexpected chars in: {cfg.name}"

    def test_permute_n_epilogue_selected_when_tile_allows(self):
        # tile_n=128, warp_n=4, warp_tile_n=16 → N_repeat=128/(4*16)=2; 2%2==0 → permute_n
        cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="mem", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert "permute_n" in cfg.name

    def test_cshuffle_epilogue_when_quant_group_n_gt1(self):
        # quant_group_n=8 → PermuteN not selected even if N_repeat is even
        cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="mem", epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=128,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=8, quant_group_k=128,
        )
        assert "cshuffle" in cfg.name
        assert "permute_n" not in cfg.name

    def test_preshuffle_aq_compv3_name(self):
        cfg = default_fp8_preshuffleaq_config()
        assert "compv3" in cfg.name
        assert cfg.name.endswith("_preshuffleaq")

    def test_fp8i4_in_name(self):
        cfg = default_fp8i4_config()
        assert "fp8i4" in cfg.name

    def test_bf8i4_in_name(self):
        cfg = default_bf8i4_config()
        assert "bf8i4" in cfg.name


# =============================================================================
# AQuantGemmProblem dimension properties
# =============================================================================


class TestAQuantGemmProblem:

    def test_qk_a_exact_divisible(self):
        p = AQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert p.QK_A == 2

    def test_qk_a_ceil_division(self):
        p = AQuantGemmProblem(M=16, N=64, K=257, quant_group_k=128)
        assert p.QK_A == 3  # ceil(257/128)=3

    def test_qm_a_is_m_when_gm1(self):
        p = AQuantGemmProblem(M=32, N=64, K=256, quant_group_m=1, quant_group_k=128)
        assert p.QM_A == 32  # ceil(32/1)=32

    def test_qm_a_groups_when_gm_gt1(self):
        p = AQuantGemmProblem(M=32, N=64, K=256, quant_group_m=8, quant_group_k=128)
        assert p.QM_A == 4  # ceil(32/8)=4

    def test_qm_a_ceil(self):
        p = AQuantGemmProblem(M=33, N=64, K=256, quant_group_m=8)
        assert p.QM_A == 5  # ceil(33/8)=5

    def test_qk_a_group64(self):
        p = AQuantGemmProblem(M=16, N=64, K=256, quant_group_k=64)
        assert p.QK_A == 4

    def test_default_group_sizes(self):
        p = AQuantGemmProblem(M=16, N=64, K=256)
        assert p.quant_group_m == 1
        assert p.quant_group_n == 1
        assert p.quant_group_k == 128

    def test_k_batch_default(self):
        p = AQuantGemmProblem(M=16, N=64, K=256)
        assert p.k_batch == 1

    def test_k_batch_splitk(self):
        p = AQuantGemmProblem(M=16, N=64, K=256, k_batch=4)
        assert p.k_batch == 4


# =============================================================================
# to_codegen_config serialization
# =============================================================================


class TestCodegenConfig:

    def test_round_trip_variant_key(self):
        cfg = default_bf8_config()
        d = cfg.to_codegen_config()
        assert d["variant_keys"] == ["bf8"]

    def test_round_trip_pipeline(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["pipeline"] == "mem"

    def test_round_trip_preshuffle_aq_false(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["preshuffle_aq"] is False

    def test_round_trip_preshuffle_aq_true(self):
        cfg = default_fp8_preshuffleaq_config()
        d = cfg.to_codegen_config()
        assert d["preshuffle_aq"] is True
        assert d["pipeline"] == "compv3"

    def test_round_trip_tile_config(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 16
        assert tc["tile_n"] == 64
        assert tc["tile_k"] == 256

    def test_round_trip_quant_groups(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr", pipeline="mem",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=2, quant_group_n=1, quant_group_k=64,
        )
        d = cfg.to_codegen_config()
        qg = d["quant_groups"][0]
        assert qg["quant_group_m"] == 2
        assert qg["quant_group_n"] == 1
        assert qg["quant_group_k"] == 64

    def test_round_trip_transpose_c(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_aq=True, transpose_c=True,
        )
        d = cfg.to_codegen_config()
        assert d["transpose_c"] is True

    def test_config_is_serializable_as_json(self):
        import json
        cfg = default_fp8_config()
        json_str = json.dumps(cfg.to_codegen_config())
        parsed = json.loads(json_str)
        assert parsed["variant_keys"] == ["fp8"]


# =============================================================================
# Default config factories
# =============================================================================


class TestDefaultConfigs:

    def test_default_fp8_pipeline_is_mem(self):
        assert default_fp8_config().pipeline == "mem"

    def test_default_bf8_pipeline_is_mem(self):
        assert default_bf8_config().pipeline == "mem"

    def test_default_fp8i4_variant_key(self):
        assert default_fp8i4_config().variant_key == "fp8i4"

    def test_default_bf8i4_variant_key(self):
        assert default_bf8i4_config().variant_key == "bf8i4"

    def test_preshuffle_aq_pipeline_is_compv3(self):
        assert default_fp8_preshuffleaq_config().pipeline == "compv3"
        assert default_bf8_preshuffleaq_config().pipeline == "compv3"

    def test_preshuffle_aq_flag_set(self):
        assert default_fp8_preshuffleaq_config().preshuffle_aq is True
        assert default_bf8_preshuffleaq_config().preshuffle_aq is True

    def test_non_preshuffle_flag_unset(self):
        assert default_fp8_config().preshuffle_aq is False
        assert default_bf8_config().preshuffle_aq is False

    def test_default_layout_is_rcr(self):
        for cfg in [default_fp8_config(), default_bf8_config(),
                    default_fp8i4_config(), default_bf8i4_config()]:
            assert cfg.layout == "rcr"

    def test_custom_quant_group_k(self):
        cfg = default_fp8_config(quant_group_k=64)
        assert cfg.quant_group_k == 64
        assert "aqg1x1x64" in cfg.name

    def test_custom_quant_group_m(self):
        cfg = default_fp8_config(quant_group_m=4)
        assert cfg.quant_group_m == 4
        assert "aqg4x1x128" in cfg.name

    def test_gfx_arch_propagated(self):
        cfg = default_fp8_config(gfx_arch="gfx942")
        assert cfg.gfx_arch == "gfx942"

    def test_all_defaults_have_unique_names(self):
        names = [
            default_fp8_config().name,
            default_bf8_config().name,
            default_fp8i4_config().name,
            default_bf8i4_config().name,
            default_fp8_preshuffleaq_config().name,
            default_bf8_preshuffleaq_config().name,
        ]
        assert len(names) == len(set(names)), "All default configs must produce unique names"


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:

    def test_qk_a_k_equals_group_k(self):
        p = AQuantGemmProblem(M=16, N=64, K=128, quant_group_k=128)
        assert p.QK_A == 1

    def test_qk_a_k_one(self):
        p = AQuantGemmProblem(M=16, N=64, K=1, quant_group_k=128)
        assert p.QK_A == 1

    def test_qm_a_large_m(self):
        p = AQuantGemmProblem(M=4096, N=1024, K=8192, quant_group_m=1, quant_group_k=128)
        assert p.QM_A == 4096
        assert p.QK_A == 64  # 8192/128=64

    def test_name_consistency_across_instances(self):
        cfg1 = default_fp8_config()
        cfg2 = default_fp8_config()
        assert cfg1.name == cfg2.name

    def test_preshuffle_aq_false_does_not_append_suffix(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="mem", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_aq=False,
        )
        assert not cfg.name.endswith("_preshuffleaq")

    def test_mem_and_compv3_pipelines_produce_different_names(self):
        mem_cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="mem", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        compv3_cfg = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_aq=True,
        )
        assert mem_cfg.name != compv3_cfg.name
        assert "mem" in mem_cfg.name
        assert "compv3" in compv3_cfg.name
