#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for grouped_gemm_rowcolquant_utils.py.

Tests kernel name generation, config serialization, and problem dimension helpers.
No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_grouped_gemm_rowcolquant_utils.py -v
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_rowcolquant_utils import (
    RowColQuantKernelConfig,
    RowColQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)


# =============================================================================
# RowColQuantKernelConfig.name — byte-exact match with codegen KERNEL_NAME
# =============================================================================


class TestKernelName:

    def test_fp8_rcr_default_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=False, pad_n=False, pad_k=True,
            persistent=False,
        )
        assert cfg.name == (
            "grouped_gemm_rowcolquant_fp8_rcr_compv3_cshuffle_intrawave_"
            "False_False_True_False_"
            "128x128x64_2x2x1_32x32x16"
        )

    def test_bf8_rcr_default_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="bf8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=False, pad_n=False, pad_k=True,
            persistent=False,
        )
        assert cfg.name == (
            "grouped_gemm_rowcolquant_bf8_rcr_compv3_cshuffle_intrawave_"
            "False_False_True_False_"
            "128x128x64_2x2x1_32x32x16"
        )

    def test_pad_flags_reflected_in_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        assert "True_True_True_False" in cfg.name

    def test_persistent_reflected_in_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=False, pad_n=False, pad_k=False, persistent=True,
        )
        assert cfg.name.endswith("True_128x128x64_2x2x1_32x32x16")

    def test_name_no_spaces(self):
        cfg = default_fp8_config()
        assert " " not in cfg.name

    def test_name_only_valid_chars(self):
        import re
        cfg = default_fp8_config()
        assert re.match(r'^[a-zA-Z0-9_]+$', cfg.name), f"Invalid chars in name: {cfg.name}"

    def test_fp8_bf8_names_differ(self):
        fp8 = default_fp8_config()
        bf8 = default_bf8_config()
        assert fp8.name != bf8.name

    def test_different_tiles_produce_different_names(self):
        def make(tm, tn, tk):
            return RowColQuantKernelConfig(
                dtype="fp8", layout="rcr", pipeline="compv3",
                epilogue="cshuffle", scheduler="intrawave",
                tile_m=tm, tile_n=tn, tile_k=tk,
                warp_m=2, warp_n=2, warp_k=1,
                warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            ).name
        names = [make(64, 64, 32), make(128, 64, 32), make(128, 128, 64)]
        assert len(names) == len(set(names))

    def test_default_fp8_config_name(self):
        cfg = default_fp8_config()
        assert "fp8" in cfg.name
        assert "grouped_gemm_rowcolquant" in cfg.name

    def test_default_bf8_config_name(self):
        cfg = default_bf8_config()
        assert "bf8" in cfg.name
        assert "grouped_gemm_rowcolquant" in cfg.name


# =============================================================================
# RowColQuantKernelConfig.to_codegen_config — round-trip shape
# =============================================================================


class TestCodegenConfig:

    def test_codegen_config_contains_correct_dtype(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["dtypes"] == ["fp8"]

    def test_codegen_config_tile_roundtrip(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=64, tile_n=128, tile_k=32,
            warp_m=2, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        )
        d = cfg.to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 64
        assert tc["tile_n"] == 128
        assert tc["tile_k"] == 32
        assert tc["warp_m"] == 2
        assert tc["warp_n"] == 4

    def test_codegen_config_single_layout(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["layouts"] == ["rcr"]

    def test_codegen_config_pad_flags(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=True, pad_n=False, pad_k=True, persistent=True,
        )
        d = cfg.to_codegen_config()
        assert d["pad_m"] is True
        assert d["pad_n"] is False
        assert d["pad_k"] is True
        assert d["persistent"] is True


# =============================================================================
# RowColQuantGemmProblem dimension helpers
# =============================================================================


class TestRowColQuantGemmProblem:

    def test_QK_A_equals_M(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert p.QK_A == 16

    def test_QK_B_equals_N(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert p.QK_B == 64

    def test_default_k_batch(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert p.k_batch == 1

    def test_k_batch_set(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256, k_batch=4)
        assert p.k_batch == 4


# =============================================================================
# Name uniqueness across a small sweep
# =============================================================================


class TestNameUniqueness:

    def _make_configs(self):
        configs = []
        for dtype in ("fp8", "bf8"):
            for pad_k in (True, False):
                configs.append(RowColQuantKernelConfig(
                    dtype=dtype, layout="rcr", pipeline="compv3",
                    epilogue="cshuffle", scheduler="intrawave",
                    tile_m=128, tile_n=128, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1,
                    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
                    pad_k=pad_k,
                ))
        return configs

    def test_all_names_unique(self):
        configs = self._make_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


# =============================================================================
# Codegen header generation (CPU-only, no GPU)
# =============================================================================


class TestCodegenHeaderGeneration:

    def _generate_header(self, dtype="fp8"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
        from unified_grouped_gemm_rowcolquant_codegen import (
            RowColQuantKernelHeaderGenerator,
            RowColQuantKernelSpec,
            RowColQuantTileConfig,
        )
        tile = RowColQuantTileConfig(
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        )
        spec = RowColQuantKernelSpec(
            dtype=dtype, layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            pad_m=False, pad_n=False, pad_k=True, persistent=False,
            tile=tile,
        )
        gen = RowColQuantKernelHeaderGenerator()
        return gen.generate(spec)

    def test_header_contains_kernel_name(self):
        header = self._generate_header()
        assert "grouped_gemm_rowcolquant_fp8_rcr" in header

    def test_header_contains_rowcolquant_type(self):
        header = self._generate_header()
        assert "ck_tile::QuantType::RowColQuant" in header

    def test_header_contains_selected_kernel(self):
        header = self._generate_header()
        assert "using SelectedKernel" in header

    def test_header_contains_single_kernel_include_guard(self):
        header = self._generate_header()
        assert "CK_TILE_SINGLE_KERNEL_INCLUDE" in header

    def test_bf8_header_uses_bf8_datatype(self):
        header = self._generate_header(dtype="bf8")
        assert "ck_tile::bf8_t" in header

    def test_fp8_header_uses_fp8_datatype(self):
        header = self._generate_header(dtype="fp8")
        assert "ck_tile::fp8_t" in header

    def test_header_contains_aq_bq_layouts(self):
        header = self._generate_header()
        assert "using AQLayout" in header
        assert "using BQLayout" in header

    def test_header_launch_takes_vector_of_host_args(self):
        header = self._generate_header()
        assert "std::vector<ck_tile::QuantGroupedGemmHostArgs>" in header


# =============================================================================
# Default config alignment: utils default vs codegen default
# =============================================================================


class TestDefaultConfigAlignment:
    """Ensure default_fp8_config/default_bf8_config stay in sync with _default_config()."""

    def _codegen_default_names(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
        from unified_grouped_gemm_rowcolquant_codegen import _default_config, _build_specs
        specs = _build_specs(_default_config())
        return {s.name for s in specs}

    def test_default_fp8_config_name_in_codegen_defaults(self):
        cfg = default_fp8_config()
        assert cfg.name in self._codegen_default_names(), (
            f"default_fp8_config().name '{cfg.name}' is not produced by _default_config() "
            f"in the codegen. The two defaults have drifted — update one to match the other."
        )

    def test_default_bf8_config_name_in_codegen_defaults(self):
        cfg = default_bf8_config()
        assert cfg.name in self._codegen_default_names(), (
            f"default_bf8_config().name '{cfg.name}' is not produced by _default_config() "
            f"in the codegen. The two defaults have drifted — update one to match the other."
        )
