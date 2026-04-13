#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Predictor for CK Tile kernel performance.

Loads trained LightGBM models and provides:
  - predict_tflops(): predicted TFLOPS for a single (problem, kernel) pair
  - predict_latency(): predicted latency in ms
  - predict_bandwidth(): predicted bandwidth in GB/s
  - predict_all(): all three predictions at once
  - rank_kernels(): rank all candidate kernels by predicted TFLOPS
  - select_best(): return the best kernel ID

Usage:
    predictor = Predictor("models/gemm_universal_fp8_gfx950")
    best_kernel = predictor.select_best(
        problem={"m": 128, "n": 1536, "k": 7168, "dtype": "fp8", "layout": "rcr"},
        kernel_configs=[...],
    )
"""

import gzip
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_engine import GemmUniversalFeatureEngine


class Predictor:
    """Loads trained models and feature spec for kernel performance prediction.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing model artifacts:
        - model_tflops.lgbm (required)
        - model_latency.lgbm (optional)
        - model_bandwidth.lgbm (optional)
        - feature_spec.json (required)

    feature_engine : FeatureEngine, optional
        Override the feature engine. If None, constructs one from feature_spec.json.
    """

    def __init__(self, model_dir: str | Path, feature_engine=None):
        self._model_dir = Path(model_dir)
        self._models: dict[str, lgb.Booster] = {}

        spec_path = self._model_dir / "feature_spec.json"
        if spec_path.exists():
            with open(spec_path) as f:
                self._spec = json.load(f)
        else:
            self._spec = {}

        self._log_targets = set(self._spec.get("log_targets", []))

        if feature_engine is not None:
            self._feature_engine = feature_engine
        else:
            self._feature_engine = GemmUniversalFeatureEngine()

    def _load_model(self, target: str) -> Optional[lgb.Booster]:
        """Lazy-load a model for the given target.

        Automatically decompresses .lgbm.gz files if the .lgbm file doesn't exist.
        The decompressed file is cached to disk for subsequent loads.
        """
        if target in self._models:
            return self._models[target]

        path = self._model_dir / f"model_{target}.lgbm"
        gz_path = self._model_dir / f"model_{target}.lgbm.gz"

        # Auto-decompress if needed
        if not path.exists() and gz_path.exists():
            with gzip.open(gz_path, 'rb') as f_in:
                with open(path, 'wb') as f_out:
                    f_out.write(f_in.read())

        if not path.exists():
            return None

        model = lgb.Booster(model_file=str(path))
        self._models[target] = model
        return model

    def _predict_single(self, target: str, problem: dict, kernel_config: dict) -> float:
        """Predict a single target value, applying inverse log transform if needed."""
        model = self._load_model(target)
        if model is None:
            raise FileNotFoundError(f"No model_{target}.lgbm in {self._model_dir}")
        features = self._feature_engine.extract(problem, kernel_config)
        raw = float(model.predict(features.reshape(1, -1))[0])
        if target in self._log_targets:
            return float(np.expm1(raw))
        # Clamp to non-negative even for non-log models
        return float(max(0.0, raw))

    def predict_tflops(self, problem: dict, kernel_config: dict) -> float:
        """Predict TFLOPS for a single (problem, kernel) pair.

        Returns a real TFLOPS estimate (interpretable, usable as DE surrogate).
        If the model was trained in log-space, the inverse transform is applied
        automatically.
        """
        return self._predict_single("tflops", problem, kernel_config)

    def predict_latency(self, problem: dict, kernel_config: dict) -> float:
        """Predict latency in milliseconds for a single (problem, kernel) pair."""
        return self._predict_single("latency", problem, kernel_config)

    def predict_bandwidth(self, problem: dict, kernel_config: dict) -> float:
        """Predict bandwidth in GB/s for a single (problem, kernel) pair."""
        return self._predict_single("bandwidth", problem, kernel_config)

    def predict_all(self, problem: dict, kernel_config: dict) -> dict[str, float]:
        """Predict all available targets for a single (problem, kernel) pair.

        Returns dict with keys 'tflops', 'latency_ms', 'bandwidth_gb_s' (if models exist).

        Note: Applies inverse log transform for targets in log_targets and clamps
        negatives to 0.0, consistent with _predict_single().
        """
        features = self._feature_engine.extract(problem, kernel_config).reshape(1, -1)
        result = {}
        for target, key in [
            ("tflops", "tflops"),
            ("latency", "latency_ms"),
            ("bandwidth", "bandwidth_gb_s"),
        ]:
            model = self._load_model(target)
            if model is not None:
                raw = float(model.predict(features)[0])
                # Apply inverse log transform if model was trained in log-space
                if target in self._log_targets:
                    result[key] = float(np.expm1(raw))
                else:
                    # Clamp to non-negative even for non-log models
                    result[key] = float(max(0.0, raw))
        return result

    def rank_kernels(
        self, problem: dict, kernel_configs: list[dict]
    ) -> list[tuple[str, float]]:
        """Rank candidate kernels by predicted TFLOPS (descending).

        Parameters
        ----------
        problem : dict
            Problem specification with keys: m, n, k, dtype, layout, split_k.
        kernel_configs : list of dict
            Each dict must have a 'kernel_name' key plus kernel parameters.

        Returns
        -------
        list of (kernel_name, predicted_tflops) tuples, sorted descending.
        """
        if not kernel_configs:
            return []

        model = self._load_model("tflops")
        if model is None:
            raise FileNotFoundError(f"No model_tflops.lgbm in {self._model_dir}")

        rows = []
        for kc in kernel_configs:
            merged = {**problem, **kc}
            rows.append(merged)

        df = pd.DataFrame(rows)
        X = self._feature_engine.extract_batch(df)
        preds = model.predict(X)
        if "tflops" in self._log_targets:
            preds = np.expm1(preds)

        results = []
        for i, kc in enumerate(kernel_configs):
            name = kc.get("kernel_name", f"kernel_{i}")
            results.append((name, float(preds[i])))

        results.sort(key=lambda x: -x[1])
        return results

    def select_best(self, problem: dict, kernel_configs: list[dict]) -> str:
        """Return the kernel_name of the best predicted kernel."""
        ranked = self.rank_kernels(problem, kernel_configs)
        if not ranked:
            raise ValueError("No kernel configs provided")
        return ranked[0][0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict kernel performance")
    parser.add_argument(
        "--model_dir", required=True, help="Directory with trained models"
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--layout", default="rcr")
    parser.add_argument("--dtype", default="fp8")
    args = parser.parse_args()

    predictor = Predictor(args.model_dir)
    problem = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "dtype": args.dtype,
        "layout": args.layout,
        "split_k": 1,
    }

    print(f"Loading models from {args.model_dir}...")
    print(
        f"Problem: M={args.m} N={args.n} K={args.k} dtype={args.dtype} layout={args.layout}"
    )

    data_dir = Path(args.model_dir).parent.parent / "data"
    if data_dir.exists():
        for pq in data_dir.glob("*.parquet"):
            df = pd.read_parquet(pq)
            kernel_names = df["kernel_name"].unique()
            configs = []
            for kn in kernel_names[:10]:
                row = df[df["kernel_name"] == kn].iloc[0]
                configs.append(row.to_dict())
            if configs:
                ranked = predictor.rank_kernels(problem, configs)
                print(f"\nTop 5 kernels (from {len(configs)} candidates):")
                for name, tflops in ranked[:5]:
                    print(f"  {tflops:8.2f} TFLOPS  {name}")
                break
