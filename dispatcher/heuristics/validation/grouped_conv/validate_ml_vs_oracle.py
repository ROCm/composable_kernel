#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validate ML heuristic predictions against oracle-best performance.

This script:
1. Loads 300 validation problems
2. Runs ML heuristic to predict best kernel for each
3. Compares predicted kernel TFLOPS vs oracle-best TFLOPS
4. Reports efficiency metrics
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

_THIS_DIR = Path(__file__).parent
# This file lives at: <repo>/projects/composablekernel/dispatcher/heuristics/validation/grouped_conv/
# Walk up three levels (validation -> heuristics -> dispatcher) to find the dispatcher root.
_DISPATCHER_ROOT = _THIS_DIR.parent.parent.parent
_CK_ROOT = _DISPATCHER_ROOT.parent
# Problem definitions still live with the benchmarking harness in tile_engine.
_TILE_ENGINE_GROUPED_CONV = _CK_ROOT / "tile_engine" / "ops" / "grouped_conv"

sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "heuristics"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))
sys.path.insert(0, str(_TILE_ENGINE_GROUPED_CONV / "problems"))

from validation_holdout import VALIDATION_PROBLEMS  # noqa: E402
from predict import Predictor  # noqa: E402
from feature_engine_grouped_conv import GroupedConvFeatureEngine  # noqa: E402
from grouped_conv.grouped_config_rules_default import COMMON_TILES, TILE_TO_WAVE, iter_pipeline_variants  # noqa: E402


# Generate kernel pool (suffix-aware; sourced from grouped_config_rules)
def _generate_kernel_pool(pipelines=None):
    """Generate kernel pool from tile configs × suffix-aware pipeline variants."""
    kernels = []
    variants = list(iter_pipeline_variants(pipelines))
    for tile_m, tile_n, tile_k in COMMON_TILES:
        if (tile_m, tile_n, tile_k) not in TILE_TO_WAVE:
            continue

        wave_m, wave_n, wave_k = TILE_TO_WAVE[(tile_m, tile_n, tile_k)]
        block_size = wave_m * wave_n * wave_k * 64

        for pipeline, wave_mode, has_dsb, has_si in variants:
            kernels.append(
                {
                    "block_size": block_size,
                    "gemm_m_per_block": tile_m,
                    "gemm_n_per_block": tile_n,
                    "pipeline": pipeline,
                    "wave_mode": wave_mode,
                    "has_dsb": has_dsb,
                    "has_si": has_si,
                }
            )

    return kernels


# Kernel pool for forward convolutions: full suffix-aware pool (300 entries).
kernel_pool = _generate_kernel_pool()


def _build_kernel_name(kconf, ndim):
    """Reconstruct the full suffix-aware kernel name from a kconf dict.

    Mirrors the naming produced by the codegen / benchmark harness so
    predicted names match measured names exactly.
    """
    suffix = f"_{kconf['wave_mode']}"
    if kconf.get("has_dsb", 0):
        suffix += "_dsb"
    if kconf.get("has_si", 0):
        suffix += "_si"
    return (
        f"grouped_conv_forward_bf16_{ndim}_"
        f"{kconf['gemm_m_per_block']}x{kconf['gemm_n_per_block']}x64_"
        f"{kconf['pipeline']}{suffix}"
    )


# Parse CLI args
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--oracle-csv",
    type=Path,
    default=_TILE_ENGINE_GROUPED_CONV / "validation_oracle_results.csv",
    help="Oracle benchmark CSV (produced by tile_engine/ops/grouped_conv/grouped_conv_full_benchmark.py)",
)
_parser.add_argument(
    "--model-dir",
    type=Path,
    default=_DISPATCHER_ROOT
    / "heuristics/models/grouped_conv_forward_bf16_gfx950_2d_3d_no_compv5",
    help="Trained LightGBM model directory.",
)
_parser.add_argument(
    "--output",
    type=Path,
    default=_THIS_DIR / "validation_heuristic_vs_oracle.csv",
    help="Where to write the per-problem comparison CSV.",
)
_args = _parser.parse_args()

# Load model
model_dir = _args.model_dir
feature_engine = GroupedConvFeatureEngine()
predictor = Predictor(model_dir, feature_engine=feature_engine)

print("=" * 80)
print("ML Heuristic Validation")
print("=" * 80)
print(f"Model: {model_dir.name}")
print(f"Kernel pool: {len(kernel_pool)} candidates")
print(f"Validation problems: {len(VALIDATION_PROBLEMS)}")
print()

# Load oracle benchmark results
oracle_df = pd.read_csv(_args.oracle_csv)
print(f"Oracle measurements: {len(oracle_df)}")
print()

# Get oracle-best for each problem
oracle_best = {}
for prob_idx in range(len(VALIDATION_PROBLEMS)):
    prob_measurements = oracle_df[oracle_df["problem_idx"] == prob_idx]
    if len(prob_measurements) > 0:
        best_idx = prob_measurements["tflops"].idxmax()
        best_row = prob_measurements.loc[best_idx]
        oracle_best[prob_idx] = {
            "kernel": best_row["kernel"],
            "tflops": best_row["tflops"],
            "latency_ms": best_row["latency_ms"],
        }

print(
    f"Oracle-best available for {len(oracle_best)} / {len(VALIDATION_PROBLEMS)} problems"
)
print()

# Run heuristic predictions
print("Running ML heuristic predictions...")
print()

heuristic_predictions = []
for prob_idx, prob in enumerate(VALIDATION_PROBLEMS):
    # Build problem dictionary
    problem = {
        "N": prob.N,
        "C": prob.C,
        "K": prob.K,
        "G": prob.G,
        "Hi": prob.Hi,
        "Wi": prob.Wi,
        "Y": prob.Y,
        "X": prob.X,
        "stride_h": prob.stride_h,
        "stride_w": prob.stride_w,
        "pad_h": prob.pad_h,
        "pad_w": prob.pad_w,
        "dtype": "bf16",
    }

    # Predict for all kernels
    predictions = []
    for kernel in kernel_pool:
        try:
            pred_tflops = predictor.predict_tflops(problem, kernel)
            predictions.append(
                {
                    "kernel_config": kernel,
                    "predicted_tflops": pred_tflops,
                }
            )
        except Exception:
            # Skip kernels that fail (e.g., dimension mismatches)
            pass

    if predictions:
        # Find best predicted kernel
        best_pred = max(predictions, key=lambda x: x["predicted_tflops"])

        # Generate full suffix-aware kernel name for matching with oracle
        kconf = best_pred["kernel_config"]
        Di = getattr(prob, "Di", 1)
        ndim = "3d" if Di > 1 else "2d"
        kernel_name = _build_kernel_name(kconf, ndim)

        heuristic_predictions.append(
            {
                "problem_idx": prob_idx,
                "predicted_kernel": kernel_name,
                "predicted_tflops": best_pred["predicted_tflops"],
                "num_candidates": len(predictions),
            }
        )

print(f"Heuristic predictions: {len(heuristic_predictions)}")
print()

# Compare heuristic vs oracle-best
print("=" * 80)
print("Comparison: Heuristic vs Oracle-Best")
print("=" * 80)

efficiencies = []
results = []

for pred in heuristic_predictions:
    prob_idx = pred["problem_idx"]

    if prob_idx in oracle_best:
        oracle = oracle_best[prob_idx]

        # Get actual TFLOPS of the predicted kernel from oracle data
        prob_measurements = oracle_df[
            (oracle_df["problem_idx"] == prob_idx)
            & (oracle_df["kernel"] == pred["predicted_kernel"])
        ]

        if len(prob_measurements) > 0:
            actual_tflops = prob_measurements.iloc[0]["tflops"]
            oracle_tflops = oracle["tflops"]

            efficiency = actual_tflops / oracle_tflops if oracle_tflops > 0 else 0
            efficiencies.append(efficiency)

            results.append(
                {
                    "problem_idx": prob_idx,
                    "oracle_kernel": oracle["kernel"],
                    "oracle_tflops": oracle_tflops,
                    "predicted_kernel": pred["predicted_kernel"],
                    "actual_tflops": actual_tflops,
                    "efficiency": efficiency,
                    "match": pred["predicted_kernel"] == oracle["kernel"],
                }
            )
        else:
            # Predicted kernel wasn't benchmarked (may have timed out)
            results.append(
                {
                    "problem_idx": prob_idx,
                    "oracle_kernel": oracle["kernel"],
                    'oracle["tflops"]': oracle["tflops"],
                    "predicted_kernel": pred["predicted_kernel"],
                    "actual_tflops": 0.0,
                    "efficiency": 0.0,
                    "match": False,
                }
            )

# Calculate metrics
if len(efficiencies) > 0:
    efficiencies = np.array(efficiencies)
    matches = sum(1 for r in results if r["match"])

    print(f"Problems compared: {len(results)}")
    print(f"  Predictions with oracle data: {len(efficiencies)}")
    print(f"  Predictions missing oracle data: {len(results) - len(efficiencies)}")
    print(
        f"Kernel match rate: {matches / len(results) * 100:.1f}% ({matches}/{len(results)})"
    )
    print()
    print("TFLOPS Efficiency (predicted_kernel_tflops / oracle_best_tflops):")
    print(f"  Mean:   {efficiencies.mean():.4f} ({efficiencies.mean() * 100:.2f}%)")
    print(
        f"  Median: {np.median(efficiencies):.4f} ({np.median(efficiencies) * 100:.2f}%)"
    )
    print(
        f"  P10:    {np.percentile(efficiencies, 10):.4f} ({np.percentile(efficiencies, 10) * 100:.2f}%)"
    )
    print(
        f"  P90:    {np.percentile(efficiencies, 90):.4f} ({np.percentile(efficiencies, 90) * 100:.2f}%)"
    )
    print(f"  Min:    {efficiencies.min():.4f} ({efficiencies.min() * 100:.2f}%)")
    print(f"  Max:    {efficiencies.max():.4f} ({efficiencies.max() * 100:.2f}%)")
    print()

    # Show worst cases
    print("Worst 10 predictions (lowest efficiency):")
    print()
    results_df = pd.DataFrame(results)
    worst_10 = results_df.nsmallest(10, "efficiency")
    for idx, row in worst_10.iterrows():
        prob = VALIDATION_PROBLEMS[row["problem_idx"]]
        Di = getattr(prob, "Di", 1)
        ndim = "3D" if Di > 1 else "2D"
        print(
            f"Problem {row['problem_idx']}: N={prob.N} C={prob.C} K={prob.K} H={prob.Hi} W={prob.Wi} ({ndim})"
        )
        print(
            f"  Oracle: {row['oracle_kernel']:<50} {row['oracle_tflops']:>8.2f} TFLOPS"
        )
        print(
            f"  Predicted: {row['predicted_kernel']:<47} {row['actual_tflops']:>8.2f} TFLOPS"
        )
        print(f"  Efficiency: {row['efficiency']:.2%}")
        print()

    # Save detailed results
    results_df.to_csv(_args.output, index=False)
    print(f"Detailed results saved to: {_args.output}")
else:
    print("ERROR: No predictions could be compared with oracle data")
