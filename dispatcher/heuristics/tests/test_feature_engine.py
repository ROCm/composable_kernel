#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for feature_engine.py.

Covers: feature count consistency, formula correctness (tile efficiency, LDS,
arithmetic intensity), corner-case shapes (M=1, huge M, square, skinny-K),
parameter space validity, config validation, and batch vs single extraction parity.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engine import (
    GemmUniversalFeatureEngine,
)


@pytest.fixture
def fe():
    """Default feature engine with MI355X-like hardware."""
    return GemmUniversalFeatureEngine(
        num_cus=256,
        lds_capacity=65536,
        max_clock_mhz=2400,
        simds_per_cu=4,
        shader_engines=32,
        max_waves_per_cu=32,
        wavefront_size=64,
        l1_cache_kb=32,
        l2_cache_kb=4096,
        l3_cache_kb=262144,
        num_xcd=8,
    )


def _make_problem(m=1024, n=1024, k=1024, dtype="fp8", layout="rcr", split_k=1):
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": dtype,
        "layout": layout,
        "split_k": split_k,
    }


def _make_kernel(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_k=1,
    warp_tile_m=32,
    warp_tile_n=32,
    warp_tile_k=16,
    pipeline="compv3",
    scheduler="intrawave",
    epilogue="cshuffle",
    pad_m=False,
    pad_n=False,
    pad_k=False,
    persistent=False,
):
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_k": warp_k,
        "warp_tile_m": warp_tile_m,
        "warp_tile_n": warp_tile_n,
        "warp_tile_k": warp_tile_k,
        "pipeline": pipeline,
        "scheduler": scheduler,
        "epilogue": epilogue,
        "pad_m": pad_m,
        "pad_n": pad_n,
        "pad_k": pad_k,
        "persistent": persistent,
    }


# ---------------------------------------------------------------------------
# Basic consistency
# ---------------------------------------------------------------------------


class TestFeatureConsistency:
    def test_feature_count_matches_names(self, fe):
        prob = _make_problem()
        kern = _make_kernel()
        vec = fe.extract(prob, kern)
        assert len(vec) == len(fe.get_feature_names())

    def test_feature_count_is_72(self, fe):
        assert len(fe.get_feature_names()) == 72

    def test_no_nan_in_output(self, fe):
        prob = _make_problem()
        kern = _make_kernel()
        vec = fe.extract(prob, kern)
        assert not np.any(np.isnan(vec))

    def test_no_inf_in_output(self, fe):
        prob = _make_problem()
        kern = _make_kernel()
        vec = fe.extract(prob, kern)
        assert not np.any(np.isinf(vec))

    def test_categorical_features_in_names(self, fe):
        names = fe.get_feature_names()
        for cat in fe.get_categorical_features():
            assert cat in names


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


class TestTileEfficiency:
    """Tile efficiency: fraction of the last tile that is useful work."""

    def test_perfectly_divisible(self, fe):
        prob = _make_problem(m=256, n=256, k=128)
        kern = _make_kernel(tile_m=128, tile_n=128, tile_k=64)
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        assert vec[names.index("tile_eff_m")] == 1.0
        assert vec[names.index("tile_eff_n")] == 1.0
        assert vec[names.index("tile_eff_k")] == 1.0
        assert vec[names.index("overall_tile_efficiency")] == 1.0

    def test_not_divisible(self, fe):
        prob = _make_problem(m=100, n=100, k=100)
        kern = _make_kernel(tile_m=128, tile_n=128, tile_k=64)
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        assert vec[names.index("tile_eff_m")] == pytest.approx(100 / 128)
        assert vec[names.index("tile_eff_n")] == pytest.approx(100 / 128)
        assert vec[names.index("tile_eff_k")] == pytest.approx(36 / 64)

    def test_m_equals_1(self, fe):
        """Single-token inference: M=1, tile_m=128 => eff = 1/128."""
        prob = _make_problem(m=1)
        kern = _make_kernel(tile_m=128)
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        assert vec[names.index("tile_eff_m")] == pytest.approx(1.0 / 128.0)


class TestLDSUsage:
    def test_lds_formula(self, fe):
        prob = _make_problem(dtype="fp8")
        kern = _make_kernel(tile_m=128, tile_n=128, tile_k=64)
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        expected = (128 * 64 + 128 * 64) * 1.0  # fp8 = 1 byte
        assert vec[names.index("lds_usage_estimate")] == expected

    def test_lds_ratio_compv4(self, fe):
        """compv4 has 32KB LDS limit, not 64KB."""
        prob = _make_problem(dtype="fp8")
        kern = _make_kernel(tile_m=128, tile_n=128, tile_k=64, pipeline="compv4")
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        lds_est = (128 * 64 + 128 * 64) * 1.0
        assert vec[names.index("lds_usage_ratio")] == pytest.approx(lds_est / 32768)

    def test_lds_fp16_doubles(self, fe):
        prob = _make_problem(dtype="fp16")
        kern = _make_kernel(tile_m=128, tile_n=128, tile_k=64)
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        expected = (128 * 64 + 128 * 64) * 2.0  # fp16 = 2 bytes
        assert vec[names.index("lds_usage_estimate")] == expected


class TestArithmeticIntensity:
    def test_square_shape(self, fe):
        M, N, K = 1024, 1024, 1024
        prob = _make_problem(m=M, n=N, k=K, dtype="fp8")
        kern = _make_kernel()
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        mem = (M * K + K * N + M * N) * 1.0
        expected = (2.0 * M * N * K) / mem
        assert vec[names.index("arithmetic_intensity")] == pytest.approx(expected)

    def test_skinny_k(self, fe):
        """Small K => low arithmetic intensity (memory-bound)."""
        prob = _make_problem(m=8192, n=8192, k=32, dtype="fp8")
        kern = _make_kernel()
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        assert vec[names.index("arithmetic_intensity")] < 100

    def test_deep_k(self, fe):
        """Large K => high arithmetic intensity (compute-bound)."""
        prob = _make_problem(m=256, n=256, k=8192, dtype="fp8")
        kern = _make_kernel()
        vec = fe.extract(prob, kern)
        names = fe.get_feature_names()
        assert vec[names.index("arithmetic_intensity")] > 100


# ---------------------------------------------------------------------------
# Corner-case shapes
# ---------------------------------------------------------------------------


class TestCornerCaseShapes:
    def test_m1_single_token(self, fe):
        vec = fe.extract(_make_problem(m=1, n=4096, k=4096), _make_kernel())
        assert not np.any(np.isnan(vec))

    def test_m1_n1_k1_minimum(self, fe):
        vec = fe.extract(_make_problem(m=1, n=1, k=1), _make_kernel())
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_very_large_m(self, fe):
        vec = fe.extract(_make_problem(m=20480, n=7168, k=7168), _make_kernel())
        assert not np.any(np.isnan(vec))

    def test_non_power_of_2(self, fe):
        vec = fe.extract(_make_problem(m=1536, n=7168, k=2304), _make_kernel())
        assert not np.any(np.isnan(vec))

    def test_prime_dimensions(self, fe):
        vec = fe.extract(_make_problem(m=17, n=31, k=127), _make_kernel())
        assert not np.any(np.isnan(vec))

    def test_tall_matrix(self, fe):
        """M >> N (tall matrix)."""
        prob = _make_problem(m=16384, n=64, k=1024)
        vec = fe.extract(prob, _make_kernel())
        names = fe.get_feature_names()
        assert vec[names.index("aspect_ratio_mn")] > 100

    def test_wide_matrix(self, fe):
        """N >> M (wide matrix)."""
        prob = _make_problem(m=64, n=16384, k=1024)
        vec = fe.extract(prob, _make_kernel())
        names = fe.get_feature_names()
        assert vec[names.index("aspect_ratio_mn")] < 0.01


# ---------------------------------------------------------------------------
# Batch vs single extraction parity
# ---------------------------------------------------------------------------


class TestBatchParity:
    def test_batch_matches_single(self, fe):
        """Vectorized batch should produce identical results to row-by-row."""
        rows = [
            {
                "m": 16,
                "n": 1536,
                "k": 7168,
                "split_k": 1,
                "dtype": "fp8",
                "layout": "rcr",
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 128,
                "warp_m": 1,
                "warp_n": 4,
                "warp_k": 1,
                "warp_tile_m": 16,
                "warp_tile_n": 16,
                "warp_tile_k": 128,
                "pipeline": "compv3",
                "scheduler": "intrawave",
                "epilogue": "cshuffle",
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "persistent": False,
            },
            {
                "m": 20480,
                "n": 7168,
                "k": 256,
                "split_k": 1,
                "dtype": "fp8",
                "layout": "rcr",
                "tile_m": 64,
                "tile_n": 64,
                "tile_k": 128,
                "warp_m": 2,
                "warp_n": 2,
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 16,
                "pipeline": "mem",
                "scheduler": "interwave",
                "epilogue": "default",
                "pad_m": True,
                "pad_n": True,
                "pad_k": True,
                "persistent": True,
            },
        ]
        df = pd.DataFrame(rows)
        batch_result = fe.extract_batch(df)

        for i, row_dict in enumerate(rows):
            single_result = fe.extract(row_dict, row_dict)
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Mismatch at row {i}",
            )


# ---------------------------------------------------------------------------
# Parameter space and validation
# ---------------------------------------------------------------------------


class TestParameterSpace:
    def test_parameter_space_non_empty(self, fe):
        ps = fe.get_parameter_space()
        assert len(ps) > 0
        assert "tile_m" in ps
        assert "pipeline" in ps

    def test_valid_config_passes(self, fe):
        config = {
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 2,
            "warp_k": 1,
            "pipeline": "compv3",
            "scheduler": "intrawave",
            "epilogue": "cshuffle",
            "pad_m": False,
            "pad_n": False,
            "pad_k": False,
            "persistent": False,
        }
        assert fe.validate_config(config) is True

    def test_invalid_tile_rejected(self, fe):
        config = {"tile_m": 999}
        assert fe.validate_config(config) is False

    def test_lds_constraint_rejects_huge_tile(self, fe):
        config = {
            "tile_m": 256,
            "tile_n": 256,
            "tile_k": 256,
            "warp_m": 2,
            "warp_n": 2,
            "warp_k": 1,
            "pipeline": "compv4",
        }
        assert fe.validate_config(config) is False

    def test_project_to_valid_snaps(self, fe):
        config = {"tile_m": 100, "tile_n": 200, "pipeline": "compv3"}
        projected = fe.project_to_valid(config)
        assert projected["tile_m"] == 128
        assert projected["tile_n"] == 192
        assert projected["pipeline"] == "compv3"


# ---------------------------------------------------------------------------
# Hardware features
# ---------------------------------------------------------------------------


class TestHardwareFeatures:
    def test_hardware_values_propagated(self, fe):
        vec = fe.extract(_make_problem(), _make_kernel())
        names = fe.get_feature_names()
        assert vec[names.index("hw_num_cus")] == 256
        assert vec[names.index("hw_max_clock_mhz")] == 2400
        assert vec[names.index("hw_total_simds")] == 256 * 4
        assert vec[names.index("hw_num_xcd")] == 8

    def test_different_hardware(self):
        fe_small = GemmUniversalFeatureEngine(num_cus=120, max_clock_mhz=1800)
        vec = fe_small.extract(_make_problem(), _make_kernel())
        names = fe_small.get_feature_names()
        assert vec[names.index("hw_num_cus")] == 120
        assert vec[names.index("hw_max_clock_mhz")] == 1800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
