#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ML Heuristic Validation: Test ML predictions against oracle-best from training data

This script validates ML-based kernel selection by:
1. Loading benchmark data (oracle-best results for each shape)
2. Using ML model to predict best kernel for each shape
3. Comparing ML selection with oracle-best to compute efficiency

Usage:
    python validate_ml_heuristic.py --dtype fp16 --model_dir models/gemm_universal_fp16_gfx950
    python validate_ml_heuristic.py --dtype fp8 --layout rcr
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from predict import Predictor


def validate_ml_heuristic(dtype: str, layout: str, model_dir: str, data_dir: str):
    """Validate ML heuristic predictions against oracle-best"""

    print("=" * 100)
    print(f"  ML Heuristic Validation: {dtype.upper()} {layout.upper()}")
    print("=" * 100)
    print()

    # Load training data
    print(f"Loading training data from {data_dir}...")

    # Try dtype-specific parquet first, then fall back to combined
    dtype_specific = (
        Path(data_dir) / f"{dtype}_original" / f"{dtype}_training_data.parquet"
    )
    combined = Path(data_dir) / "all_training_data_fixed.parquet"

    if dtype_specific.exists():
        training_data = pd.read_parquet(dtype_specific)
        print(f"✓ Loaded {len(training_data):,} benchmark runs from {dtype_specific}")
    elif combined.exists():
        training_data = pd.read_parquet(combined)
        training_data = training_data[
            (training_data["dtype"] == dtype) & (training_data["layout"] == layout)
        ]
        print(f"✓ Loaded {len(training_data):,} benchmark runs from {combined}")
    else:
        print(f"❌ Error: No training data found at {dtype_specific} or {combined}")
        return

    if len(training_data) == 0:
        print(f"❌ Error: No data found for dtype={dtype}, layout={layout}")
        return

    # Get unique shapes with oracle-best
    shape_groups = training_data.groupby(["m", "n", "k"])
    print(f"Unique shapes: {len(shape_groups)}")
    print()

    # Load ML predictor
    print(f"Loading ML predictor from {model_dir}...")
    try:
        predictor = Predictor(model_dir)
        print("✓ Loaded ML predictor")
        print(f"  Log targets: {predictor._log_targets}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print()
    print("=" * 100)
    print("  Computing Oracle-Best Efficiency for Each Shape")
    print("=" * 100)
    print()

    results = []

    for shape_idx, ((m, n, k), group) in enumerate(shape_groups):
        # Find oracle-best (max TFLOPS across all kernels tested)
        oracle_best_row = group.loc[group["measured_tflops"].idxmax()]
        oracle_best_tflops = oracle_best_row["measured_tflops"]
        oracle_best_kernel = oracle_best_row["kernel_name"]

        # Get all kernel configs tested for this shape
        kernel_configs = []
        for _, row in group.iterrows():
            kernel_dict = {
                "tile_m": row["tile_m"],
                "tile_n": row["tile_n"],
                "tile_k": row["tile_k"],
                "warp_m": row["warp_m"],
                "warp_n": row["warp_n"],
                "warp_k": row["warp_k"],
                "warp_tile_m": row["warp_tile_m"],
                "warp_tile_n": row["warp_tile_n"],
                "warp_tile_k": row["warp_tile_k"],
                "pipeline": row["pipeline"],
                "scheduler": row["scheduler"],
                "epilogue": row["epilogue"],
                "pad_m": row["pad_m"],
                "pad_n": row["pad_n"],
                "pad_k": row["pad_k"],
                "persistent": row["persistent"],
                "kernel_name": row["kernel_name"],
            }
            kernel_configs.append(kernel_dict)

        # Use ML model to rank kernels
        problem = {
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "layout": layout,
            "split_k": 1,
        }

        try:
            ranked = predictor.rank_kernels(problem, kernel_configs)

            if ranked:
                ml_best_kernel, ml_predicted_tflops = ranked[0]

                # Find actual TFLOPS for the ML-predicted kernel
                ml_kernel_row = group[group["kernel_name"] == ml_best_kernel]
                if len(ml_kernel_row) > 0:
                    ml_actual_tflops = ml_kernel_row["measured_tflops"].values[0]

                    # Calculate efficiency
                    efficiency_pct = 100.0 * (ml_actual_tflops / oracle_best_tflops)

                    # Determine if ML picked oracle-best
                    is_oracle_best = ml_best_kernel == oracle_best_kernel

                    results.append(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "oracle_best_tflops": oracle_best_tflops,
                            "oracle_best_kernel": oracle_best_kernel,
                            "ml_predicted_tflops": ml_predicted_tflops,
                            "ml_selected_kernel": ml_best_kernel,
                            "ml_actual_tflops": ml_actual_tflops,
                            "efficiency_pct": efficiency_pct,
                            "is_oracle_best": is_oracle_best,
                            "num_kernels": len(group),
                        }
                    )

                    if (shape_idx + 1) % 20 == 0:
                        status = "✓" if is_oracle_best else f"{efficiency_pct:.1f}%"
                        print(
                            f"  [{shape_idx + 1:3d}/{len(shape_groups)}] "
                            f"M={m:4d} N={n:5d} K={k:5d}: {status}"
                        )
        except Exception as e:
            print(f"  Error on shape M={m} N={n} K={k}: {e}")
            continue

    print()
    print("=" * 100)
    print("  Results Summary")
    print("=" * 100)
    print()

    if results:
        df_results = pd.DataFrame(results)
        efficiencies = df_results["efficiency_pct"].values
        oracle_matches = df_results["is_oracle_best"].sum()

        print(f"Total shapes tested: {len(results)}")
        print()
        print("Efficiency Statistics (% of Oracle-Best TFLOPS):")
        print(f"  Mean:           {np.mean(efficiencies):.2f}%")
        print(f"  Median:         {np.median(efficiencies):.2f}%")
        print(f"  Min:            {np.min(efficiencies):.2f}%")
        print(f"  Max:            {np.max(efficiencies):.2f}%")
        print(f"  P10:            {np.percentile(efficiencies, 10):.2f}%")
        print(f"  P50:            {np.percentile(efficiencies, 50):.2f}%")
        print(f"  P90:            {np.percentile(efficiencies, 90):.2f}%")
        print()
        print(
            f"Oracle-best matches: {oracle_matches}/{len(results)} ({100 * oracle_matches / len(results):.1f}%)"
        )
        print()

        # Classify by M size
        df_results["m_class"] = pd.cut(
            df_results["m"],
            bins=[0, 8, 128, 1024, float("inf")],
            labels=[
                "Tiny (M<8)",
                "Small (8≤M<128)",
                "Medium (128≤M<1024)",
                "Large (M≥1024)",
            ],
        )

        print("Efficiency by M size:")
        for m_class in [
            "Tiny (M<8)",
            "Small (8≤M<128)",
            "Medium (128≤M<1024)",
            "Large (M≥1024)",
        ]:
            subset = df_results[df_results["m_class"] == m_class]
            if len(subset) > 0:
                print(
                    f"  {m_class:25s}: {subset['efficiency_pct'].mean():6.2f}% "
                    f"(n={len(subset)}, P10={subset['efficiency_pct'].quantile(0.1):.2f}%)"
                )

        print()

        # Save results
        output_file = f"validation_results_{dtype}_{layout}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"✓ Results saved to {output_file}")

        # Show best and worst shapes
        print()
        print("Top 5 shapes (best efficiency):")
        top5 = df_results.nlargest(5, "efficiency_pct")[
            ["m", "n", "k", "efficiency_pct", "oracle_best_tflops", "is_oracle_best"]
        ]
        for idx, row in top5.iterrows():
            match = "✓" if row["is_oracle_best"] else " "
            print(
                f"  {match} M={row['m']:5d} N={row['n']:5d} K={row['k']:5d}: "
                f"{row['efficiency_pct']:.2f}% ({row['oracle_best_tflops']:.2f} TFLOPS)"
            )

        print()
        print("Bottom 5 shapes (worst efficiency):")
        bottom5 = df_results.nsmallest(5, "efficiency_pct")[
            ["m", "n", "k", "efficiency_pct", "oracle_best_tflops", "is_oracle_best"]
        ]
        for idx, row in bottom5.iterrows():
            match = "✓" if row["is_oracle_best"] else " "
            print(
                f"  {match} M={row['m']:5d} N={row['n']:5d} K={row['k']:5d}: "
                f"{row['efficiency_pct']:.2f}% ({row['oracle_best_tflops']:.2f} TFLOPS)"
            )

    else:
        print("No results to display")

    print()
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ML heuristic predictions against oracle-best from training data"
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp8"],
        help="Data type to validate",
    )
    parser.add_argument(
        "--layout",
        default="rcr",
        choices=["rcr", "rrr", "crr", "ccr"],
        help="Matrix layout",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to model directory (auto-detect if not specified)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to training data directory (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Auto-detect model directory if not specified
    if args.model_dir is None:
        heuristics_dir = Path(__file__).parent
        model_candidates = [
            heuristics_dir / "models" / f"gemm_universal_{args.dtype}_gfx950",
            heuristics_dir / "models" / f"gemm_universal_{args.dtype}_gfx942",
        ]
        for candidate in model_candidates:
            if candidate.exists():
                args.model_dir = str(candidate)
                break

        if args.model_dir is None:
            print(f"❌ Error: Could not find model directory for {args.dtype}")
            print(f"   Searched: {[str(c) for c in model_candidates]}")
            print("   Please specify --model_dir explicitly")
            return 1

    # Auto-detect data directory if not specified
    if args.data_dir is None:
        heuristics_dir = Path(__file__).parent
        args.data_dir = str(heuristics_dir / "data")

    validate_ml_heuristic(args.dtype, args.layout, args.model_dir, args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
