#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for evaluate.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import classify_shape_family, classify_k_regime, evaluate_model


class TestClassifyShapeFamily:
    def test_tiny_m(self):
        assert classify_shape_family(1, 4096, 4096) == "tiny_m"
        assert classify_shape_family(16, 1536, 7168) == "tiny_m"

    def test_small_m(self):
        assert classify_shape_family(32, 1536, 7168) == "small_m"
        assert classify_shape_family(128, 4096, 4096) == "small_m"

    def test_medium_m(self):
        assert classify_shape_family(256, 1024, 1024) == "medium_m"
        assert classify_shape_family(2048, 2048, 2048) == "medium_m"

    def test_large_m(self):
        assert classify_shape_family(4096, 4096, 4096) == "large_m"
        assert classify_shape_family(20480, 7168, 256) == "large_m"


class TestClassifyKRegime:
    def test_shallow(self):
        assert classify_k_regime(256) == "shallow_k"
        assert classify_k_regime(32) == "shallow_k"

    def test_medium(self):
        assert classify_k_regime(1024) == "medium_k"
        assert classify_k_regime(2048) == "medium_k"

    def test_deep(self):
        assert classify_k_regime(4096) == "deep_k"
        assert classify_k_regime(7168) == "deep_k"


class _StubModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)
        self.last_X = None

    def predict(self, X):
        assert len(X) == len(self._preds)
        self.last_X = np.asarray(X)
        return self._preds


class _StubPredictor:
    def __init__(
        self, preds, log_targets=(), feature_indices=None, feature_engine=None
    ):
        self._model = _StubModel(preds)
        self._log_targets = log_targets
        self._feature_indices = feature_indices
        self._feature_engine = feature_engine or _StubFeatureEngine()

    @property
    def feature_engine(self):
        return self._feature_engine

    def _load_model(self, target):
        assert target == "tflops"
        return self._model

    def select_features(self, X):
        if self._feature_indices is None:
            return X
        return X[:, self._feature_indices]


class _StubFeatureEngine:
    def __init__(self, width=1):
        self._width = width

    def extract_batch(self, df):
        return np.tile(np.arange(self._width, dtype=float), (len(df), 1))


def _frame():
    return pd.DataFrame(
        {
            "m": [128, 128, 256, 256],
            "n": [128, 128, 256, 256],
            "k": [128, 128, 256, 256],
            "measured_tflops": [100.0, 50.0, 100.0, 25.0],
            "is_valid": [True, True, True, True],
            "pipeline": ["compv3", "compv3", "compv3", "compv3"],
        }
    )


class TestEvaluateModel:
    def test_runs_end_to_end(self):
        df = _frame()
        preds = [100.0, 50.0, 25.0, 100.0]
        res = evaluate_model(_StubPredictor(preds), df, _StubFeatureEngine())

        gm = res["global_metrics"]
        assert gm["num_shapes"] == 2
        assert gm["num_valid_rows"] == 4
        assert gm["efficiency_mean"] == pytest.approx(0.625)
        assert gm["ndcg_at_1"] == pytest.approx(0.5)
        assert len(res["per_shape_efficiency"]) == 2

    def test_slice_metrics_populated(self):
        res = evaluate_model(
            _StubPredictor([100.0, 50.0, 25.0, 100.0]), _frame(), _StubFeatureEngine()
        )
        for key in ("shape_family_metrics", "k_regime_metrics", "pipeline_metrics"):
            assert res[key], f"{key} is empty"
            for name, metrics in res[key].items():
                assert metrics["count"] > 0, f"{key}[{name}] has no rows"
                assert 0.0 < metrics["mean"] <= 1.0


class TestLogTransform:
    MEASURED = [100.0, 50.0, 100.0, 25.0]

    def test_log_space_predictions_are_inverted(self):
        res = evaluate_model(
            _StubPredictor(np.log1p(self.MEASURED), log_targets=("tflops",)),
            _frame(),
            _StubFeatureEngine(),
        )
        gm = res["global_metrics"]
        assert gm["r2"] == pytest.approx(1.0, abs=1e-9)
        assert gm["efficiency_mean"] == pytest.approx(1.0)

    def test_without_the_inverse_transform_the_fit_collapses(self):
        res = evaluate_model(
            _StubPredictor(np.log1p(self.MEASURED)),  # log_targets=() by default
            _frame(),
            _StubFeatureEngine(),
        )
        assert res["global_metrics"]["r2"] < 0.5


class TestFeatureRemapIsApplied:
    def test_the_model_receives_the_remapped_columns(self):
        pred = _StubPredictor(
            [100.0, 50.0, 25.0, 100.0],
            feature_indices=np.array([3, 1]),
            feature_engine=_StubFeatureEngine(width=5),
        )
        evaluate_model(pred, _frame())
        seen = pred._model.last_X
        assert seen.shape[1] == 2, "remap was not applied"
        np.testing.assert_array_equal(seen[0], [3.0, 1.0])

    def test_without_a_remap_the_columns_pass_through(self):
        pred = _StubPredictor(
            [100.0, 50.0, 25.0, 100.0], feature_engine=_StubFeatureEngine(width=5)
        )
        evaluate_model(pred, _frame())
        assert pred._model.last_X.shape[1] == 5


class TestArgParser:
    def test_the_supported_operation_parses(self):
        from evaluate import _OPERATION, build_arg_parser

        args = build_arg_parser().parse_args(
            ["--model_dir", "m", "--data_dir", "d", "--op", _OPERATION]
        )
        assert args.op == _OPERATION

    def test_it_defaults_to_the_supported_operation(self):
        from evaluate import _OPERATION, build_arg_parser

        parsed = build_arg_parser().parse_args(["--model_dir", "m", "--data_dir", "d"])
        assert parsed.op == _OPERATION

    @pytest.mark.parametrize("op", ["grouped_conv", "fmha", "gemm_universal_vec"])
    def test_an_unsupported_operation_is_rejected(self, op):
        from evaluate import build_arg_parser

        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(
                ["--model_dir", "m", "--data_dir", "d", "--op", op]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
