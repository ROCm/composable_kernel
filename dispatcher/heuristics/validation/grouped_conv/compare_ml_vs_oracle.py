#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Compare ML heuristic predictions against oracle benchmark results.

MODE 1: CSV Comparison (SUPPORTED)
  Reads:
    - Oracle CSV: benchmark results with all kernel measurements
    - ML CSV: ML predictions with rankings
  Outputs:
    - Efficiency metrics: ML_picked_actual_TFLOPS / Oracle_best_TFLOPS

MODE 2: End-to-End Workflow (NOT YET IMPLEMENTED)
  Planned feature to automatically run benchmarks and ML predictions.
  Currently shows manual workflow instructions instead.

Usage:
  # Mode 1: Compare existing CSVs
  python compare_ml_vs_oracle.py --oracle-csv oracle.csv --ml-csv ml.csv --plot result.png

  # Mode 2: Not yet implemented (shows manual workflow instructions)
  python compare_ml_vs_oracle.py --shapes "N=1,C=64,K=64,Hi=28,Wi=28,Y=3,X=3,stride_h=1,stride_w=1"
  python compare_ml_vs_oracle.py --problem-set forward_validation_300
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_oracle_results(csv_path):
    """Load oracle benchmark results.

    Returns:
        dict: {problem_idx: {kernel_name: tflops}}
    """
    results = defaultdict(dict)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob_idx = int(row["problem_idx"])
            kernel_name = row.get("kernel_name", row.get("kernel", ""))
            tflops_str = row.get("tflops", row.get("tflops", "0"))
            tflops = float(tflops_str) if tflops_str not in ("N/A", "") else 0.0

            results[prob_idx][kernel_name] = tflops

    return results


def load_ml_predictions(csv_path):
    """Load ML predictions.

    Returns:
        dict: {problem_idx: ml_top1_kernel_name}
    """
    ml_top1 = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob_idx = int(row["problem_idx"])
            kernel_name = row["kernel_name"]
            rank = int(row["rank"])

            if rank == 1:
                ml_top1[prob_idx] = kernel_name

    return ml_top1


def compute_efficiency(oracle_best_tflops, ml_picked_actual_tflops):
    """Compute efficiency: ML_picked / Oracle_best."""
    if oracle_best_tflops <= 0:
        return 0.0
    return (ml_picked_actual_tflops / oracle_best_tflops) * 100.0


def parse_shape(shape_str):
    """Parse shape string like 'N=1,C=64,K=64,Hi=28,Wi=28,Y=3,X=3,stride_h=1,stride_w=1'"""
    shape = {}
    for part in shape_str.split(","):
        key, val = part.split("=")
        shape[key.strip()] = int(val.strip())

    # Set defaults
    shape.setdefault("G", 1)
    shape.setdefault("pad_h", 0)
    shape.setdefault("pad_w", 0)
    shape.setdefault("dilation_h", 1)
    shape.setdefault("dilation_w", 1)

    return shape


def run_end_to_end_workflow(args):
    """Run full workflow: benchmark oracle + ML prediction + comparison"""

    print("=" * 100)
    print("  END-TO-END ML vs ORACLE COMPARISON")
    print("=" * 100)
    print()

    # Parse shapes
    if args.shapes:
        print(f"Custom shapes: {len(args.shapes)}")
        problems = [parse_shape(s) for s in args.shapes]
        for i, p in enumerate(problems):
            print(
                f"  {i}: N={p['N']} C={p['C']} K={p['K']} Hi={p['Hi']}x{p['Wi']} Y={p['Y']}x{p['X']}"
            )
    elif args.problem_set:
        print(f"Problem set: {args.problem_set}")
        # Import problem set dynamically
        # Problem sets live with the benchmarking harness in tile_engine.
        _THIS_DIR = Path(__file__).parent
        _TILE_ENGINE_GROUPED_CONV = (
            _THIS_DIR.parent.parent.parent.parent
            / "tile_engine"
            / "ops"
            / "grouped_conv"
        )
        sys.path.insert(0, str(_TILE_ENGINE_GROUPED_CONV / "problems"))
        try:
            problem_module = __import__(args.problem_set)
            problem_attr = (
                args.problem_set.upper()
                .replace("_", "_")
                .replace("FORWARD", "PROBLEMS_FORWARD")
            )
            if not hasattr(problem_module, problem_attr):
                # Try alternate naming
                problem_attr = [
                    attr for attr in dir(problem_module) if "PROBLEM" in attr.upper()
                ][0]
            problems_list = getattr(problem_module, problem_attr)
            problems = []
            for prob in problems_list:
                problems.append(
                    {
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
                        "dilation_h": getattr(prob, "dilation_h", 1),
                        "dilation_w": getattr(prob, "dilation_w", 1),
                    }
                )
            print(f"  Loaded {len(problems)} problems from {args.problem_set}")
        except Exception as e:
            print(f"❌ Error loading problem set: {e}")
            return 1
    else:
        print("❌ Error: Must specify --shapes or --problem-set")
        return 1

    print()

    # Mode 2 is not yet implemented - show helpful message
    print("-" * 100)
    print("⚠️  End-to-end workflow not yet implemented")
    print("-" * 100)
    print()
    print("Please use the manual workflow documented in README.md:")
    print()
    print("  1. Create problem set file in tile_engine/ops/grouped_conv/problems/")
    print(
        "  2. Run: cd tile_engine/ops/grouped_conv && python grouped_conv_full_benchmark.py --problems <your_set> --csv oracle.csv"
    )
    print(
        "  3. Run: cd dispatcher/heuristics && python predict_cli.py --problem-module <your_set> --output ml.csv"
    )
    print(
        "  4. Run: cd dispatcher/heuristics/validation/grouped_conv && python compare_ml_vs_oracle.py --oracle-csv oracle.csv --ml-csv ml.csv --plot result.png"
    )
    print()

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML vs Oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Compare existing CSVs (SUPPORTED)
  python compare_ml_vs_oracle.py --oracle-csv oracle.csv --ml-csv ml.csv --plot result.png

  # Mode 2: End-to-end workflow (NOT YET IMPLEMENTED)
  # Use manual workflow instead - see error message when attempting Mode 2
        """,
    )

    # Mode 1: CSV comparison (existing)
    parser.add_argument("--oracle-csv", help="Oracle benchmark CSV")
    parser.add_argument("--ml-csv", help="ML predictions CSV")

    # Mode 2: End-to-end workflow (new)
    parser.add_argument(
        "--shapes",
        nargs="+",
        help='Custom shapes (e.g., "N=1,C=64,K=64,Hi=28,Wi=28,Y=3,X=3,stride_h=1,stride_w=1")',
    )
    parser.add_argument(
        "--problem-set", help="Problem set module name (e.g., forward_validation_300)"
    )
    parser.add_argument(
        "--variant", default="forward", choices=["forward", "bwd_data", "bwd_weight"]
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--arch", default="gfx950")

    # Common options
    parser.add_argument("--output", default=None, help="Output summary CSV (optional)")
    parser.add_argument(
        "--plot", default=None, help="Generate scatter plot PNG (optional)"
    )

    args = parser.parse_args()

    # Determine mode
    if args.shapes or args.problem_set:
        # Mode 2: End-to-end workflow
        return run_end_to_end_workflow(args)
    elif args.oracle_csv and args.ml_csv:
        # Mode 1: CSV comparison (existing workflow)
        pass
    else:
        parser.error(
            "Must specify either (--oracle-csv and --ml-csv) OR (--shapes or --problem-set)"
        )

    print("=" * 80)
    print("ML vs Oracle Comparison")
    print("=" * 80)
    print(f"Oracle: {args.oracle_csv}")
    print(f"ML:     {args.ml_csv}")
    print()

    # Load results
    oracle = load_oracle_results(args.oracle_csv)
    ml_top1 = load_ml_predictions(args.ml_csv)

    if not oracle:
        print("Error: No oracle results found")
        return 1

    if not ml_top1:
        print("Error: No ML predictions found")
        return 1

    # Analyze each problem
    efficiencies = []
    oracle_tflops_list = []
    ml_tflops_list = []
    top1_matches = 0
    top5_matches = 0
    total_problems = 0

    print(
        f"{'Prob':<6} {'Oracle Best':<30} {'ML Top-1':<30} {'Oracle TFLOPS':<15} {'ML Actual TFLOPS':<18} {'Efficiency':<12}"
    )
    print("-" * 135)

    for prob_idx in sorted(oracle.keys()):
        if prob_idx not in ml_top1:
            continue

        total_problems += 1

        # Get oracle best kernel for this problem
        oracle_kernels = oracle[prob_idx]
        sorted_oracle = sorted(oracle_kernels.items(), key=lambda x: x[1], reverse=True)

        if not sorted_oracle:
            continue

        oracle_best_name, oracle_best_tflops = sorted_oracle[0]

        # Get ML's top-1 prediction
        ml_picked_name = ml_top1[prob_idx]

        # Get actual TFLOPS for ML's pick from oracle results
        ml_picked_actual_tflops = oracle_kernels.get(ml_picked_name, 0.0)

        # Compute efficiency
        efficiency = compute_efficiency(oracle_best_tflops, ml_picked_actual_tflops)
        efficiencies.append(efficiency)
        oracle_tflops_list.append(oracle_best_tflops)
        ml_tflops_list.append(ml_picked_actual_tflops)

        # Check if ML top-1 matches oracle top-1
        if ml_picked_name == oracle_best_name:
            top1_matches += 1

        # Check if ML top-1 is in oracle top-5
        oracle_top5_names = [k[0] for k in sorted_oracle[:5]]
        if ml_picked_name in oracle_top5_names:
            top5_matches += 1

        # Print row (shorten kernel names for readability)
        oracle_short = (
            oracle_best_name.split("_")[-2] + "_" + oracle_best_name.split("_")[-1]
        )
        ml_short = ml_picked_name.split("_")[-2] + "_" + ml_picked_name.split("_")[-1]

        print(
            f"{prob_idx:<6} {oracle_short:<30} {ml_short:<30} "
            f"{oracle_best_tflops:<15.2f} {ml_picked_actual_tflops:<18.2f} {efficiency:<12.1f}%"
        )

    # Compute summary statistics
    if efficiencies:
        mean_eff = sum(efficiencies) / len(efficiencies)
        sorted_eff = sorted(efficiencies)
        p10_eff = (
            sorted_eff[len(sorted_eff) // 10]
            if len(sorted_eff) >= 10
            else sorted_eff[0]
        )
        p50_eff = sorted_eff[len(sorted_eff) // 2]
        min_eff = min(efficiencies)
        max_eff = max(efficiencies)

        print()
        print("=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(f"Total problems:        {total_problems}")
        print(f"Mean Efficiency:       {mean_eff:.2f}%")
        print(f"P10 Efficiency:        {p10_eff:.2f}%")
        print(f"P50 Efficiency:        {p50_eff:.2f}%")
        print(f"Min Efficiency:        {min_eff:.2f}%")
        print(f"Max Efficiency:        {max_eff:.2f}%")
        print()
        print(
            f"Top-1 Accuracy:        {top1_matches}/{total_problems} ({100.0 * top1_matches / total_problems:.1f}%)"
        )
        print(
            f"Top-5 Hit Rate:        {top5_matches}/{total_problems} ({100.0 * top5_matches / total_problems:.1f}%)"
        )

        # Save summary to file if requested
        if args.output:
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["total_problems", total_problems])
                writer.writerow(["mean_efficiency", f"{mean_eff:.2f}"])
                writer.writerow(["p10_efficiency", f"{p10_eff:.2f}"])
                writer.writerow(["p50_efficiency", f"{p50_eff:.2f}"])
                writer.writerow(["min_efficiency", f"{min_eff:.2f}"])
                writer.writerow(["max_efficiency", f"{max_eff:.2f}"])
                writer.writerow(
                    ["top1_accuracy", f"{100.0 * top1_matches / total_problems:.1f}"]
                )
                writer.writerow(
                    ["top5_hit_rate", f"{100.0 * top5_matches / total_problems:.1f}"]
                )
            print(f"\n✓ Saved summary to: {args.output}")

        # Generate scatter plot if requested
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                oracle_tflops_list = np.array(oracle_tflops_list)
                ml_tflops_list = np.array(ml_tflops_list)
                efficiencies_arr = np.array(efficiencies)

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))

                # Color by efficiency
                scatter = ax.scatter(
                    oracle_tflops_list,
                    ml_tflops_list,
                    c=efficiencies_arr,
                    cmap="RdYlGn",
                    vmin=60,
                    vmax=100,
                    alpha=0.7,
                    s=60,
                    edgecolors="black",
                    linewidth=0.5,
                )

                # Add Y=X reference line (perfect prediction)
                max_val = max(oracle_tflops_list.max(), ml_tflops_list.max())
                min_val = 0
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    linewidth=2.5,
                    label="Perfect Prediction (Y=X)",
                    alpha=0.8,
                    zorder=5,
                )

                # Add efficiency lines
                ax.plot(
                    [min_val, max_val],
                    [0.9 * min_val, 0.9 * max_val],
                    "orange",
                    linestyle=":",
                    linewidth=2,
                    label="90% Efficiency",
                    alpha=0.7,
                    zorder=4,
                )
                ax.plot(
                    [min_val, max_val],
                    [0.8 * min_val, 0.8 * max_val],
                    "gold",
                    linestyle=":",
                    linewidth=2,
                    label="80% Efficiency",
                    alpha=0.7,
                    zorder=4,
                )
                ax.plot(
                    [min_val, max_val],
                    [0.7 * min_val, 0.7 * max_val],
                    "yellow",
                    linestyle=":",
                    linewidth=1.5,
                    label="70% Efficiency",
                    alpha=0.6,
                    zorder=4,
                )

                # Labels and title
                ax.set_xlabel(
                    "Oracle TFLOPS (Best Kernel)", fontsize=13, fontweight="bold"
                )
                ax.set_ylabel(
                    "ML Heuristic TFLOPS (Top-1 Prediction)",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.set_title(
                    "ML Heuristic vs Oracle Performance\nGrouped Convolution Forward (bf16, gfx950)",
                    fontsize=15,
                    fontweight="bold",
                    pad=20,
                )

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Efficiency (%)", fontsize=11, fontweight="bold")

                # Add grid
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

                # Add legend
                ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

                # Add statistics text
                text = f"Mean Efficiency: {mean_eff:.2f}%\n"
                text += f"P10 Efficiency: {p10_eff:.2f}%\n"
                text += f"Median Efficiency: {p50_eff:.2f}%\n"
                text += f"Problems: {total_problems}\n"
                text += f"TFLOPS Range: {oracle_tflops_list.min():.2f} - {oracle_tflops_list.max():.2f}"

                ax.text(
                    0.97,
                    0.03,
                    text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="lightblue",
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=1.5,
                    ),
                )

                # Set limits to start from 0
                ax.set_xlim(0, max_val * 1.05)
                ax.set_ylim(0, max_val * 1.05)

                plt.tight_layout()
                plt.savefig(args.plot, dpi=150, bbox_inches="tight")
                print(f"✓ Saved plot to: {args.plot}")

            except ImportError:
                print("Warning: matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
