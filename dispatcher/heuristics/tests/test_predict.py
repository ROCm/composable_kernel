#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for predict.py.

Covers: Predictor initialization, single prediction, ranking, select_best,
missing model handling, and edge cases (single kernel, empty list).
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engine import GemmUniversalFeatureEngine
from predict import Predictor


@pytest.fixture
def model_dir(tmp_path):
    """Create a minimal trained model for testing."""
    fe = GemmUniversalFeatureEngine()
    n_features = len(fe.get_feature_names())

    np.random.seed(42)
    X = np.random.rand(200, n_features)
    y = np.random.rand(200) * 100

    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(tmp_path / "model_tflops.lgbm"))

    y_lat = np.random.rand(200) * 0.1
    model_lat = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model_lat.fit(X, y_lat)
    model_lat.booster_.save_model(str(tmp_path / "model_latency.lgbm"))

    spec = {
        "feature_names": fe.get_feature_names(),
        "categorical_features": fe.get_categorical_features(),
    }
    with open(tmp_path / "feature_spec.json", "w") as f:
        json.dump(spec, f)

    return tmp_path


@pytest.fixture
def predictor(model_dir):
    return Predictor(model_dir)


def _problem():
    return {
        "m": 1024,
        "n": 1024,
        "k": 1024,
        "dtype": "fp8",
        "layout": "rcr",
        "split_k": 1,
    }


def _kernel(tile_m=128, pipeline="compv3"):
    return {
        "kernel_name": f"test_kernel_{tile_m}_{pipeline}",
        "tile_m": tile_m,
        "tile_n": 128,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": pipeline,
        "scheduler": "intrawave",
        "epilogue": "cshuffle",
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": False,
    }


class TestPredictor:
    def test_predict_tflops_returns_float(self, predictor):
        result = predictor.predict_tflops(_problem(), _kernel())
        assert isinstance(result, float)

    def test_predict_latency_returns_float(self, predictor):
        result = predictor.predict_latency(_problem(), _kernel())
        assert isinstance(result, float)

    def test_predict_all_returns_dict(self, predictor):
        result = predictor.predict_all(_problem(), _kernel())
        assert "tflops" in result
        assert "latency_ms" in result

    def test_rank_kernels_sorted_descending(self, predictor):
        kernels = [_kernel(64, "compv3"), _kernel(128, "compv4"), _kernel(256, "mem")]
        ranked = predictor.rank_kernels(_problem(), kernels)
        assert len(ranked) == 3
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_returns_name(self, predictor):
        kernels = [_kernel(64), _kernel(128)]
        best = predictor.select_best(_problem(), kernels)
        assert isinstance(best, str)
        assert best in [k["kernel_name"] for k in kernels]

    def test_single_kernel(self, predictor):
        kernels = [_kernel(128)]
        ranked = predictor.rank_kernels(_problem(), kernels)
        assert len(ranked) == 1

    def test_missing_bandwidth_model(self, model_dir):
        pred = Predictor(model_dir)
        with pytest.raises(FileNotFoundError):
            pred.predict_bandwidth(_problem(), _kernel())

    def test_empty_kernel_list(self, predictor):
        with pytest.raises(ValueError):
            predictor.select_best(_problem(), [])

    def test_corner_case_m1(self, predictor):
        prob = {
            "m": 1,
            "n": 4096,
            "k": 4096,
            "dtype": "fp8",
            "layout": "rcr",
            "split_k": 1,
        }
        result = predictor.predict_tflops(prob, _kernel())
        assert np.isfinite(result)

    def test_different_shapes_give_different_results(self, predictor):
        k = _kernel()
        r1 = predictor.predict_tflops(
            {
                "m": 16,
                "n": 1536,
                "k": 7168,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            k,
        )
        r2 = predictor.predict_tflops(
            {
                "m": 20480,
                "n": 7168,
                "k": 256,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            k,
        )
        assert r1 != r2


class TestPredictorEdgeCases:
    def test_nonexistent_model_dir(self):
        with pytest.raises(Exception):
            pred = Predictor("/nonexistent/path")
            pred.predict_tflops(_problem(), _kernel())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
