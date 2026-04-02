#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for dispatcher_integration.py.

Covers: kernel name parsing to feature dict, feature dict to dispatcher config
(name mapping inversion), MLKernelSpec creation, binary pool loading, and
the ML heuristic function.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dispatcher_integration import (
    kernel_config_to_feature_dict,
    feature_dict_to_dispatcher_config,
    feature_dict_to_ml_spec,
    ml_spec_to_dispatcher_config,
    create_ml_heuristic,
    load_kernel_pool_from_binaries,
    MLKernelSpec,
    LAYOUT_TO_DISPATCHER,
)
from feature_engine import GemmUniversalFeatureEngine


SAMPLE_KERNEL_NAME = (
    "gemm_universal_fp8_rcr_compv3_cshuffle_intrawave"
    "_False_False_False_False_128x128x128_1x4x1_16x16x128"
)


# ---------------------------------------------------------------------------
# kernel_config_to_feature_dict
# ---------------------------------------------------------------------------


class TestKernelConfigToFeatureDict:
    def test_parses_standard_name(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        assert feat["tile_m"] == 128
        assert feat["tile_n"] == 128
        assert feat["tile_k"] == 128
        assert feat["warp_m"] == 1  # warps per block
        assert feat["warp_n"] == 4
        assert feat["warp_k"] == 1
        assert feat["warp_tile_m"] == 16
        assert feat["warp_tile_n"] == 16
        assert feat["warp_tile_k"] == 128
        assert feat["pipeline"] == "compv3"
        assert feat["scheduler"] == "intrawave"
        assert feat["epilogue"] == "cshuffle"
        assert feat["kernel_name"] == SAMPLE_KERNEL_NAME

    def test_empty_name_returns_empty(self):
        assert kernel_config_to_feature_dict("") == {}

    def test_invalid_name_returns_empty(self):
        assert kernel_config_to_feature_dict("not_a_kernel") == {}


# ---------------------------------------------------------------------------
# Name mapping: feature dict <-> dispatcher config
# ---------------------------------------------------------------------------


class TestNameMapping:
    """The critical inversion: feature engine warp_m/n/k (warps per block)
    maps to dispatcher wave_m/n/k, and feature engine warp_tile_m/n/k
    maps to dispatcher warp_m/n/k."""

    def test_warp_to_wave_mapping(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        disp = feature_dict_to_dispatcher_config(feat)
        assert disp["wave_m"] == feat["warp_m"]  # 1
        assert disp["wave_n"] == feat["warp_n"]  # 4
        assert disp["wave_k"] == feat["warp_k"]  # 1

    def test_warp_tile_to_warp_mapping(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        disp = feature_dict_to_dispatcher_config(feat)
        assert disp["warp_m"] == feat["warp_tile_m"]  # 16
        assert disp["warp_n"] == feat["warp_tile_n"]  # 16
        assert disp["warp_k"] == feat["warp_tile_k"]  # 128

    def test_tile_dims_pass_through(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        disp = feature_dict_to_dispatcher_config(feat)
        assert disp["tile_m"] == 128
        assert disp["tile_n"] == 128
        assert disp["tile_k"] == 128

    def test_pipeline_passes_through(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        disp = feature_dict_to_dispatcher_config(feat)
        assert disp["pipeline"] == "compv3"
        assert disp["scheduler"] == "intrawave"
        assert disp["epilogue"] == "cshuffle"

    def test_rcr_layout_mapping(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        disp = feature_dict_to_dispatcher_config(feat, dtype="fp8")
        assert disp["layout_a"] == "row"
        assert disp["layout_b"] == "col"
        assert disp["layout_c"] == "row"

    def test_all_layouts(self):
        for layout, (la, lb, lc) in LAYOUT_TO_DISPATCHER.items():
            feat = {"layout": layout, "tile_m": 128}
            disp = feature_dict_to_dispatcher_config(feat)
            assert disp["layout_a"] == la
            assert disp["layout_b"] == lb
            assert disp["layout_c"] == lc


# ---------------------------------------------------------------------------
# MLKernelSpec
# ---------------------------------------------------------------------------


class TestMLKernelSpec:
    def test_from_feature_dict(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        spec = feature_dict_to_ml_spec(feat, predicted_tflops=123.4)
        assert spec.kernel_name == SAMPLE_KERNEL_NAME
        assert spec.predicted_tflops == 123.4
        assert spec.tile_m == 128
        assert spec.wave_m == 1  # was warp_m in feature space
        assert spec.warp_m == 16  # was warp_tile_m in feature space

    def test_spec_to_dispatcher_config(self):
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        spec = feature_dict_to_ml_spec(feat, 100.0)
        disp = ml_spec_to_dispatcher_config(spec, dtype="fp8", arch="gfx950")
        assert disp["tile_m"] == 128
        assert disp["wave_m"] == 1
        assert disp["warp_m"] == 16
        assert disp["gfx_arch"] == "gfx950"
        assert disp["dtype_a"] == "fp8"

    def test_roundtrip_preserves_values(self):
        """feature_dict -> MLKernelSpec -> dispatcher_config should be consistent."""
        feat = kernel_config_to_feature_dict(SAMPLE_KERNEL_NAME)
        spec = feature_dict_to_ml_spec(feat, 0.0)
        disp_from_spec = ml_spec_to_dispatcher_config(spec)
        disp_from_feat = feature_dict_to_dispatcher_config(feat)
        for key in [
            "tile_m",
            "tile_n",
            "tile_k",
            "wave_m",
            "wave_n",
            "wave_k",
            "warp_m",
            "warp_n",
            "warp_k",
            "pipeline",
            "scheduler",
            "epilogue",
        ]:
            assert disp_from_spec[key] == disp_from_feat[key], f"Mismatch on {key}"


# ---------------------------------------------------------------------------
# Binary pool loading
# ---------------------------------------------------------------------------


class TestLoadKernelPool:
    def test_loads_from_real_bin_dir(self):
        bin_dir = Path("/workspace/ck_tile/bin")
        if not bin_dir.exists():
            pytest.skip("No /workspace/ck_tile/bin")
        pool = load_kernel_pool_from_binaries(bin_dir)
        assert len(pool) > 0
        assert "tile_m" in pool[0]
        assert "kernel_name" in pool[0]

    def test_empty_dir_returns_empty(self, tmp_path):
        pool = load_kernel_pool_from_binaries(tmp_path)
        assert pool == []


# ---------------------------------------------------------------------------
# ML heuristic function
# ---------------------------------------------------------------------------


class TestCreateMLHeuristic:
    @pytest.fixture
    def mock_model_dir(self, tmp_path):
        """Create a minimal model for testing the heuristic flow."""
        fe = GemmUniversalFeatureEngine()
        n_features = len(fe.get_feature_names())
        np.random.seed(42)
        X = np.random.rand(100, n_features)
        y = np.random.rand(100) * 500
        model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
        model.fit(X, y)
        model.booster_.save_model(str(tmp_path / "model_tflops.lgbm"))
        spec = {
            "feature_names": fe.get_feature_names(),
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        return tmp_path

    def _make_pool(self):
        """Create a small synthetic kernel pool."""
        names = [
            "gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128",
            "gemm_universal_fp8_rcr_compv4_default_intrawave_False_False_False_False_128x128x64_2x2x1_32x32x16",
            "gemm_universal_fp8_rcr_mem_cshuffle_interwave_False_False_False_False_64x64x128_1x4x1_16x16x32",
        ]
        return [kernel_config_to_feature_dict(n) for n in names]

    def test_returns_ml_kernel_spec(self, mock_model_dir):
        pool = self._make_pool()
        heuristic = create_ml_heuristic(mock_model_dir, kernel_pool=pool)
        result = heuristic(1024, 1024, 1024)
        assert isinstance(result, MLKernelSpec)
        assert result.tile_m > 0
        assert isinstance(result.predicted_tflops, float)

    def test_returns_valid_kernel_from_pool(self, mock_model_dir):
        pool = self._make_pool()
        pool_names = {p["kernel_name"] for p in pool}
        heuristic = create_ml_heuristic(mock_model_dir, kernel_pool=pool)
        result = heuristic(1024, 1024, 1024)
        assert result.kernel_name in pool_names

    def test_different_shapes_may_select_different_kernels(self, mock_model_dir):
        pool = self._make_pool()
        heuristic = create_ml_heuristic(mock_model_dir, kernel_pool=pool)
        r1 = heuristic(16, 1536, 7168)
        r2 = heuristic(8192, 8192, 256)
        # At minimum both should return valid specs
        assert r1.tile_m > 0
        assert r2.tile_m > 0

    def test_m1_corner_case(self, mock_model_dir):
        pool = self._make_pool()
        heuristic = create_ml_heuristic(mock_model_dir, kernel_pool=pool)
        result = heuristic(1, 4096, 4096)
        assert isinstance(result, MLKernelSpec)
        assert np.isfinite(result.predicted_tflops)

    def test_empty_pool_raises(self, mock_model_dir):
        with pytest.raises(ValueError, match="No kernel configs"):
            create_ml_heuristic(mock_model_dir, kernel_pool=[])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
