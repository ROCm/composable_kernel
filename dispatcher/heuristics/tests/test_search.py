#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for search.py.

Covers: random search, DE search, config validity, result ordering,
budget compliance, and edge cases.
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
from search import SurrogateSearch


@pytest.fixture
def model_dir(tmp_path):
    """Create a minimal trained model."""
    fe = GemmUniversalFeatureEngine()
    n_features = len(fe.get_feature_names())
    np.random.seed(42)
    X = np.random.rand(200, n_features)
    y = np.random.rand(200) * 500
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(tmp_path / "model_tflops.lgbm"))
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


class TestRandomSearch:
    def test_returns_results(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=50, top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_results_sorted_descending(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=100, top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_configs_are_valid(self, predictor):
        fe = GemmUniversalFeatureEngine()
        searcher = SurrogateSearch(predictor, feature_engine=fe, strategy="random")
        results = searcher.search(_problem(), budget=50, top_k=5)
        for cfg, _ in results:
            ps = fe.get_parameter_space()
            for k, v in cfg.items():
                if k in ps:
                    assert v in ps[k], f"{k}={v} not in {ps[k]}"

    def test_respects_top_k(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=100, top_k=3)
        assert len(results) <= 3

    def test_different_problems_produce_results(self, predictor):
        """Both problem sizes should produce valid search results."""
        searcher = SurrogateSearch(predictor, strategy="random", seed=42)
        r1 = searcher.search(
            {
                "m": 16,
                "n": 1536,
                "k": 7168,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            budget=50,
            top_k=3,
        )
        searcher2 = SurrogateSearch(predictor, strategy="random", seed=42)
        r2 = searcher2.search(
            {
                "m": 20480,
                "n": 7168,
                "k": 256,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            budget=50,
            top_k=3,
        )
        assert len(r1) > 0
        assert len(r2) > 0
        for _, score in r1 + r2:
            assert np.isfinite(score)

    def test_m1_corner_case(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(
            {
                "m": 1,
                "n": 4096,
                "k": 4096,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            budget=50,
            top_k=5,
        )
        assert len(results) > 0
        for _, score in results:
            assert np.isfinite(score)


class TestDESearch:
    def test_returns_results(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="de")
        results = searcher.search(_problem(), budget=100, top_k=5)
        assert len(results) > 0

    def test_results_sorted_descending(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="de")
        results = searcher.search(_problem(), budget=100, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_de_improves_over_initial(self, predictor):
        """DE should generally find at least as good as random initialization."""
        searcher_r = SurrogateSearch(predictor, strategy="random", seed=42)
        r_results = searcher_r.search(_problem(), budget=100, top_k=1)
        searcher_d = SurrogateSearch(predictor, strategy="de", seed=42)
        d_results = searcher_d.search(_problem(), budget=100, top_k=1)
        if r_results and d_results:
            assert d_results[0][1] >= r_results[0][1] * 0.9

    def test_small_budget(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="de")
        results = searcher.search(_problem(), budget=30, top_k=5)
        assert len(results) > 0


class TestSearchEdgeCases:
    def test_unknown_strategy_raises(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="unknown")
        with pytest.raises(ValueError):
            searcher.search(_problem(), budget=10)

    def test_zero_budget(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=0, top_k=5)
        assert len(results) == 0

    def test_deterministic_with_same_seed(self, predictor):
        s1 = SurrogateSearch(predictor, strategy="random", seed=123)
        s2 = SurrogateSearch(predictor, strategy="random", seed=123)
        r1 = s1.search(_problem(), budget=50, top_k=5)
        r2 = s2.search(_problem(), budget=50, top_k=5)
        assert len(r1) == len(r2)
        for (c1, s1_), (c2, s2_) in zip(r1, r2):
            assert s1_ == pytest.approx(s2_)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
