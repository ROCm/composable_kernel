#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for train.py.

Covers: group key computation, TFLOPS efficiency calculation, edge cases
(single group, all-invalid data, tied predictions), and warm-start
incremental training (feature compat, lineage, quality).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engine import GemmUniversalFeatureEngine
from train import (
    compute_group_keys,
    compute_tflops_efficiency,
    check_feature_compatibility,
    load_warm_start_model,
    train_final_model,
    DEFAULT_PARAMS,
)


class TestComputeGroupKeys:
    def test_basic(self):
        df = pd.DataFrame(
            {"m": [16, 16, 32], "n": [1536, 1536, 1536], "k": [7168, 7168, 7168]}
        )
        keys = compute_group_keys(df, "gemm_universal")
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]

    def test_unique_shapes(self):
        df = pd.DataFrame({"m": [1, 2, 3], "n": [4, 5, 6], "k": [7, 8, 9]})
        keys = compute_group_keys(df, "gemm_universal")
        assert len(set(keys)) == 3


class TestComputeTflopsEfficiency:
    def test_perfect_prediction(self):
        """Model predicts highest TFLOPS kernel => efficiency = 1.0."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 150],
                "pred_tflops": [50, 300, 100],  # correctly ranks kernel 1 highest
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] == pytest.approx(1.0)

    def test_worst_prediction(self):
        """Model picks the worst kernel."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 150],
                "pred_tflops": [999, 1, 1],  # incorrectly ranks kernel 0 highest
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert eff["efficiency"].iloc[0] == pytest.approx(100 / 200)

    def test_multiple_shapes(self):
        df = pd.DataFrame(
            {
                "m": [16, 16, 32, 32],
                "n": [1536, 1536, 1536, 1536],
                "k": [7168, 7168, 7168, 7168],
                "measured_tflops": [10, 20, 100, 200],
                "pred_tflops": [5, 25, 150, 190],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 2
        assert eff.iloc[0]["efficiency"] == pytest.approx(1.0)
        assert eff.iloc[1]["efficiency"] == pytest.approx(1.0)

    def test_zero_tflops_shape_skipped(self):
        df = pd.DataFrame(
            {
                "m": [16, 16],
                "n": [16, 16],
                "k": [16, 16],
                "measured_tflops": [0, 0],
                "pred_tflops": [1, 2],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 0

    def test_single_kernel_per_shape(self):
        df = pd.DataFrame(
            {
                "m": [1024],
                "n": [1024],
                "k": [1024],
                "measured_tflops": [150],
                "pred_tflops": [100],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] == pytest.approx(1.0)

    def test_tied_predictions(self):
        """When multiple kernels have the same predicted TFLOPS, pandas idxmax picks the first."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 200],
                "pred_tflops": [50, 50, 50],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] >= 0.5


# ---------------------------------------------------------------------------
# Helpers for warm-start tests
# ---------------------------------------------------------------------------


def _make_dummy_data(n_rows=200, n_shapes=5):
    """Create a small synthetic benchmark DataFrame for testing training."""
    rng = np.random.RandomState(42)
    rows = []
    for _ in range(n_rows):
        m = rng.choice([64, 128, 256, 512, 1024])
        n = rng.choice([64, 128, 256, 512, 1024])
        k = rng.choice([64, 128, 256, 512, 1024])
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "split_k": 1,
                "dtype": "fp8",
                "layout": "rcr",
                "op_type": "gemm_universal",
                "tile_m": rng.choice([64, 128, 256]),
                "tile_n": rng.choice([64, 128, 256]),
                "tile_k": rng.choice([32, 64, 128]),
                "warp_m": rng.choice([1, 2, 4]),
                "warp_n": rng.choice([1, 2, 4]),
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 16,
                "pipeline": rng.choice(["compv3", "compv4", "mem"]),
                "scheduler": rng.choice(["intrawave", "interwave"]),
                "epilogue": "cshuffle",
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "persistent": False,
                "measured_tflops": float(rng.uniform(10, 500)),
                "latency_ms": float(rng.uniform(0.01, 1.0)),
                "bandwidth_gb_s": float(rng.uniform(50, 1500)),
                "is_valid": True,
                "kernel_name": f"test_kernel_{rng.randint(0, 100)}",
            }
        )
    return pd.DataFrame(rows)


def _save_feature_spec(model_dir, fe):
    """Save a feature_spec.json matching the given feature engine."""
    spec = {
        "feature_names": fe.get_feature_names(),
        "categorical_features": fe.get_categorical_features(),
    }
    with open(model_dir / "feature_spec.json", "w") as f:
        json.dump(spec, f)


def _train_and_save_base_model(model_dir, df, fe, target="tflops"):
    """Train a small base model and save it to model_dir."""
    params = dict(DEFAULT_PARAMS)
    params["n_estimators"] = 20
    params["n_jobs"] = 1
    model = train_final_model(df, fe, target, params, "gemm_universal")
    model.booster_.save_model(str(model_dir / f"model_{target}.lgbm"))
    _save_feature_spec(model_dir, fe)
    return model


# ---------------------------------------------------------------------------
# Warm-start tests
# ---------------------------------------------------------------------------


class TestCheckFeatureCompatibility:
    def test_compatible_passes(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        _save_feature_spec(tmp_path, fe)
        check_feature_compatibility(tmp_path, fe)

    def test_missing_spec_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        with pytest.raises(FileNotFoundError, match="feature_spec.json"):
            check_feature_compatibility(tmp_path, fe)

    def test_added_feature_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names()[:-1],
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            check_feature_compatibility(tmp_path, fe)

    def test_removed_feature_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names() + ["extra_feature"],
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            check_feature_compatibility(tmp_path, fe)

    def test_categorical_mismatch_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names(),
            "categorical_features": ["layout", "pipeline"],
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Categorical feature mismatch"):
            check_feature_compatibility(tmp_path, fe)


class TestLoadWarmStartModel:
    def test_loads_existing_model(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data()
        _train_and_save_base_model(tmp_path, df, fe)
        path = load_warm_start_model(tmp_path, "tflops")
        assert path is not None
        assert Path(path).exists()

    def test_returns_none_for_missing_target(self, tmp_path):
        assert load_warm_start_model(tmp_path, "tflops") is None

    def test_returns_none_for_wrong_target(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data()
        _train_and_save_base_model(tmp_path, df, fe, target="tflops")
        assert load_warm_start_model(tmp_path, "bandwidth") is None


class TestWarmStartTraining:
    def test_warm_start_produces_more_trees(self, tmp_path):
        """A warm-started model should have more trees than the base."""
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data(n_rows=300)

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _train_and_save_base_model(base_dir, df, fe)
        base_n_trees = base_model.booster_.num_trees()

        init_model_path = load_warm_start_model(base_dir, "tflops")
        params = dict(DEFAULT_PARAMS)
        params["n_estimators"] = 15
        params["n_jobs"] = 1
        warm_model = train_final_model(
            df, fe, "tflops", params, "gemm_universal", init_model=init_model_path
        )
        warm_n_trees = warm_model.booster_.num_trees()

        assert warm_n_trees > base_n_trees

    def test_warm_start_does_not_degrade(self, tmp_path):
        """Warm-started model on the same data should not be significantly worse."""
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data(n_rows=300)

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _train_and_save_base_model(base_dir, df, fe)

        X = fe.extract_batch(df[df["is_valid"]].reset_index(drop=True))
        y = df[df["is_valid"]]["measured_tflops"].values
        base_rmse = np.sqrt(np.mean((base_model.predict(X) - y) ** 2))

        init_model_path = load_warm_start_model(base_dir, "tflops")
        params = dict(DEFAULT_PARAMS)
        params["n_estimators"] = 15
        params["n_jobs"] = 1
        warm_model = train_final_model(
            df, fe, "tflops", params, "gemm_universal", init_model=init_model_path
        )
        warm_rmse = np.sqrt(np.mean((warm_model.predict(X) - y) ** 2))

        assert warm_rmse <= base_rmse * 1.1

    def test_warm_start_from_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            check_feature_compatibility(
                Path("/nonexistent/model/dir"), GemmUniversalFeatureEngine()
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
