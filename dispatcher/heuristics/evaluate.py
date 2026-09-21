#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Evaluation and reporting for CK Tile kernel performance models.

Computes:
  - Global metrics: TFLOPS efficiency (mean, p10, p50, min), R2, NDCG@1, Top-K hit rate
  - Per-slice breakdowns: by layout, shape family, K-depth regime, pipeline
  - Cross-target consistency checks
  - Feature importance analysis

Usage:
    python evaluate.py --model_dir models/gemm_universal_fp8_gfx950 --data_dir data/
"""

import argparse
import json

import numpy as np
import pandas as pd

from data_pipeline import build_training_dataset
from feature_engine import GemmUniversalFeatureEngine
from predict import Predictor
from train import compute_tflops_efficiency


def classify_shape_family(m: int, n: int, k: int) -> str:
    """Classify a GEMM shape into a family for sliced evaluation.

    Families:
      - tiny_m: M < 32 (single-token / very small batch inference)
      - small_m: 32 <= M < 256
      - medium_m: 256 <= M < 4096
      - large_m: M >= 4096
      - square: 0.5 <= M/N <= 2.0 and 0.5 <= M/K <= 2.0
      - tall: M/N > 2.0
      - wide: M/N < 0.5
    """
    if m < 32:
        return "tiny_m"
    elif m < 256:
        return "small_m"
    elif m < 4096:
        return "medium_m"
    elif m >= 4096:
        return "large_m"
    return "other"


def classify_k_regime(k: int) -> str:
    """Classify K dimension into depth regime."""
    if k < 512:
        return "shallow_k"
    elif k < 4096:
        return "medium_k"
    else:
        return "deep_k"


def evaluate_model(
    predictor: Predictor,
    df: pd.DataFrame,
    feature_engine: GemmUniversalFeatureEngine,
) -> dict:
    """Run full evaluation on a dataset. Returns a metrics dictionary.

    Parameters
    ----------
    predictor : Predictor
        Trained predictor with at least a TFLOPS model loaded.
    df : pd.DataFrame
        Benchmark data in canonical schema.
    feature_engine : GemmUniversalFeatureEngine
        Feature engine matching the trained model.

    Returns
    -------
    dict with keys: global_metrics, shape_family_metrics, k_regime_metrics,
                    pipeline_metrics, per_shape_efficiency.
    """
    valid = df[df["is_valid"].fillna(False) & (df["measured_tflops"] > 0)].copy()
    valid = valid.reset_index(drop=True)

    X = feature_engine.extract_batch(valid)
    model = predictor._load_model("tflops")
    if model is None:
        raise FileNotFoundError("No TFLOPS model found")

    # Predict and apply inverse log transform if model was trained in log-space
    raw_pred = model.predict(X)
    if "tflops" in predictor._log_targets:
        valid["pred_tflops"] = np.expm1(raw_pred)
    else:
        # Clamp to non-negative even for non-log models
        valid["pred_tflops"] = np.maximum(0.0, raw_pred)

    y_true = valid["measured_tflops"].values
    y_pred = valid["pred_tflops"].values

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    eff_df = compute_tflops_efficiency(valid, "pred_tflops")

    ndcg1_count = 0
    total_shapes = 0
    topk_hits = {3: 0, 5: 0, 10: 0}

    for (m, n, k), group in valid.groupby(["m", "n", "k"]):
        if group["measured_tflops"].max() <= 0:
            continue
        total_shapes += 1
        oracle_idx = group["measured_tflops"].idxmax()
        pred_ranking = group.sort_values("pred_tflops", ascending=False).index.tolist()

        if pred_ranking[0] == oracle_idx:
            ndcg1_count += 1

        oracle_rank = pred_ranking.index(oracle_idx)
        for topk in topk_hits:
            if oracle_rank < topk:
                topk_hits[topk] += 1

    global_metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "num_valid_rows": len(valid),
        "num_shapes": total_shapes,
        "efficiency_mean": float(eff_df["efficiency"].mean()) if len(eff_df) > 0 else 0,
        "efficiency_p10": float(eff_df["efficiency"].quantile(0.1))
        if len(eff_df) > 0
        else 0,
        "efficiency_p50": float(eff_df["efficiency"].quantile(0.5))
        if len(eff_df) > 0
        else 0,
        "efficiency_min": float(eff_df["efficiency"].min()) if len(eff_df) > 0 else 0,
        "ndcg_at_1": ndcg1_count / max(total_shapes, 1),
        "top3_hit_rate": topk_hits[3] / max(total_shapes, 1),
        "top5_hit_rate": topk_hits[5] / max(total_shapes, 1),
        "top10_hit_rate": topk_hits[10] / max(total_shapes, 1),
    }

    def _slice_efficiency(slice_df):
        if len(slice_df) == 0:
            return {"count": 0}
        eff = compute_tflops_efficiency(slice_df, "pred_tflops")
        if len(eff) == 0:
            return {"count": 0}
        return {
            "count": len(eff),
            "mean": float(eff["efficiency"].mean()),
            "p10": float(eff["efficiency"].quantile(0.1)),
            "min": float(eff["efficiency"].min()),
        }

    valid["shape_family"] = valid.apply(
        lambda r: classify_shape_family(r["m"], r["n"], r["k"]), axis=1
    )
    valid["k_regime"] = valid["k"].apply(classify_k_regime)

    shape_family_metrics = {}
    for family, group in valid.groupby("shape_family"):
        shape_family_metrics[family] = _slice_efficiency(group)

    k_regime_metrics = {}
    for regime, group in valid.groupby("k_regime"):
        k_regime_metrics[regime] = _slice_efficiency(group)

    pipeline_metrics = {}
    if "pipeline" in valid.columns:
        for pipeline, group in valid.groupby("pipeline"):
            pipeline_metrics[str(pipeline)] = _slice_efficiency(group)

    return {
        "global_metrics": global_metrics,
        "shape_family_metrics": shape_family_metrics,
        "k_regime_metrics": k_regime_metrics,
        "pipeline_metrics": pipeline_metrics,
        "per_shape_efficiency": eff_df.to_dict(orient="records")
        if len(eff_df) > 0
        else [],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CK Tile performance model")
    parser.add_argument(
        "--model_dir", required=True, help="Directory with trained models"
    )
    parser.add_argument("--data_dir", required=True, help="Directory with parquet data")
    parser.add_argument("--op", default="gemm_universal")
    parser.add_argument("--dtype", default="fp8")
    parser.add_argument("--output", "-o", help="Output JSON path for metrics")
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}...")
    df = build_training_dataset(args.data_dir, op_type=args.op, dtype=args.dtype)
    print(f"  {len(df)} rows, {df.groupby(['m', 'n', 'k']).ngroups} shapes")

    fe = GemmUniversalFeatureEngine()
    predictor = Predictor(args.model_dir, feature_engine=fe)

    print("Evaluating...")
    results = evaluate_model(predictor, df, fe)

    gm = results["global_metrics"]
    print("\nGlobal Metrics:")
    print(f"  R2:             {gm['r2']:.4f}")
    print(f"  RMSE:           {gm['rmse']:.2f}")
    print(f"  Efficiency Mean: {gm['efficiency_mean']:.4f}")
    print(f"  Efficiency P10:  {gm['efficiency_p10']:.4f}")
    print(f"  Efficiency P50:  {gm['efficiency_p50']:.4f}")
    print(f"  Efficiency Min:  {gm['efficiency_min']:.4f}")
    print(f"  NDCG@1:          {gm['ndcg_at_1']:.4f}")
    print(f"  Top-3 Hit Rate:  {gm['top3_hit_rate']:.4f}")
    print(f"  Top-5 Hit Rate:  {gm['top5_hit_rate']:.4f}")
    print(f"  Top-10 Hit Rate: {gm['top10_hit_rate']:.4f}")

    print("\nShape Family Breakdown:")
    for family, metrics in sorted(results["shape_family_metrics"].items()):
        if metrics.get("count", 0) > 0:
            print(
                f"  {family:12s}: mean={metrics['mean']:.4f} p10={metrics['p10']:.4f} min={metrics['min']:.4f} (n={metrics['count']})"
            )

    print("\nK-Depth Regime Breakdown:")
    for regime, metrics in sorted(results["k_regime_metrics"].items()):
        if metrics.get("count", 0) > 0:
            print(
                f"  {regime:12s}: mean={metrics['mean']:.4f} p10={metrics['p10']:.4f} min={metrics['min']:.4f} (n={metrics['count']})"
            )

    print("\nPipeline Breakdown:")
    for pipeline, metrics in sorted(results["pipeline_metrics"].items()):
        if metrics.get("count", 0) > 0:
            print(
                f"  {pipeline:15s}: mean={metrics['mean']:.4f} p10={metrics['p10']:.4f} (n={metrics['count']})"
            )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
