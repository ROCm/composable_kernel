#!/usr/bin/env python3
"""
Validate ML Heuristic vs Oracle Best on Hardware

For each test problem:
1. Load oracle best kernel from training data (highest measured TFLOPS)
2. Use ML to predict and select best kernel
3. Build and run both kernels on hardware
4. Compare: ML selected TFLOPS vs Oracle TFLOPS

This shows real-world ML heuristic efficiency on hardware.
"""

import sys
import json
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # heuristics

import pandas as pd
import numpy as np

from predict import Predictor
from feature_engine_grouped_conv import GroupedConvFeatureEngine
from grouped_conv_utils import (
    GroupedConvKernelConfig,
    setup_multiple_grouped_conv_dispatchers,
)


@dataclass
class KernelSpec:
    """Grouped convolution kernel specification"""

    name: str
    block_size: int
    gemm_m_per_block: int
    gemm_n_per_block: int
    pipeline: str = "compv3"

    def to_kernel_config(
        self, dtype: str = "bf16", arch: str = "gfx950"
    ) -> GroupedConvKernelConfig:
        """Convert to GroupedConvKernelConfig for building."""
        return GroupedConvKernelConfig(
            variant="forward",
            dtype=dtype,
            ndim_spatial=2,
            layout="NHWGC_KYXGC_NHWGK",
            arch=arch,
            tile_m=self.block_size,
            tile_n=self.gemm_m_per_block,
            tile_k=self.gemm_n_per_block,
            wave_m=2,
            wave_n=2,
            wave_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
            pipeline=self.pipeline,
            scheduler="default",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            pad_k=True,
        )


def build_kernel(
    spec: KernelSpec, dtype: str, arch: str, verbose: bool = False
) -> Path:
    """Build a kernel on-demand using JIT compilation."""
    kernel_config = spec.to_kernel_config(dtype=dtype, arch=arch)

    lib_paths = setup_multiple_grouped_conv_dispatchers(
        [kernel_config], verbose=verbose, max_workers=1
    )

    if not lib_paths or lib_paths[0] is None:
        return None

    return lib_paths[0]


def run_kernel_on_hw(so_path: Path, problem: dict, kernel_name: str) -> dict:
    """Run a kernel on hardware via subprocess."""
    script_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "tile_engine"
        / "ops"
        / "grouped_conv"
        / "run_one_grouped_conv_kernel.py"
    )

    input_data = {
        "so_path": str(so_path),
        "problem": {**problem, "direction": "forward"},
        "kernel_name": kernel_name,
    }

    env = {
        **os.environ,
        "GCONV_PYPATH": str(Path(__file__).parent.parent.parent.parent / "python"),
    }

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout, stderr = proc.communicate(input=json.dumps(input_data).encode())

    try:
        result = json.loads(stdout.decode().strip())
        return result
    except:
        return {"ok": False, "error": "Failed to parse output"}


def create_kernel_spec_from_row(row: pd.Series) -> KernelSpec:
    """Create KernelSpec from training data row."""
    return KernelSpec(
        name=f"k{row['block_size']}_{row['gemm_m_per_block']}x{row['gemm_n_per_block']}_{row['pipeline']}",
        block_size=int(row["block_size"]),
        gemm_m_per_block=int(row["gemm_m_per_block"]),
        gemm_n_per_block=int(row["gemm_n_per_block"]),
        pipeline=str(row["pipeline"]),
    )


def main():
    print("=" * 100)
    print("  ML Heuristic vs Oracle Best - Hardware Validation")
    print("=" * 100)

    # Load training data
    data_path = (
        Path(__file__).parent.parent.parent.parent
        / "heuristics"
        / "data"
        / "grouped_conv_forward_bf16_gfx950"
        / "training_data.parquet"
    )
    df = pd.read_parquet(data_path)

    print(f"\nLoaded {len(df)} training samples")

    # Load ML model
    model_dir = (
        Path(__file__).parent.parent.parent.parent
        / "heuristics"
        / "models"
        / "grouped_conv_forward_bf16_gfx950"
    )
    feature_engine = GroupedConvFeatureEngine()
    predictor = Predictor(model_dir, feature_engine=feature_engine)

    print(f"Loaded ML model from {model_dir}")

    # Select diverse test problems from training data
    # Group by problem shape and find problems with multiple kernels
    shape_cols = [
        "N",
        "C",
        "K",
        "G",
        "Hi",
        "Wi",
        "Y",
        "X",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
    ]

    # Get problems with at least 5 kernels to have good oracle vs ML comparison
    problem_groups = df.groupby(shape_cols)
    problems_with_many_kernels = [
        (shape, group) for shape, group in problem_groups if len(group) >= 5
    ]

    # Sort by diversity and select 5 test problems
    np.random.seed(42)
    selected_indices = np.random.choice(
        len(problems_with_many_kernels), size=min(5, len(problems_with_many_kernels)), replace=False
    )
    test_problems = [problems_with_many_kernels[i] for i in selected_indices]

    print(f"\nSelected {len(test_problems)} test problems with multiple kernels each")
    print()

    # Test each problem
    results = []

    header = (
        f"{'Problem':<40} {'Oracle':<20} {'ML Sel':<20} "
        f"{'Or TFLOPS':>10} {'ML TFLOPS':>10} {'Efficiency':>12}"
    )
    print(header)
    print("-" * len(header))

    for shape, group in test_problems:
        # Build problem dict
        problem = {col: int(shape[i]) for i, col in enumerate(shape_cols)}
        problem["dtype"] = "bf16"

        # Get oracle best from training data
        oracle_row = group.loc[group["tflops"].idxmax()]
        oracle_spec = create_kernel_spec_from_row(oracle_row)
        oracle_train_tflops = oracle_row["tflops"]

        # Get all kernels for this problem
        all_kernels = [create_kernel_spec_from_row(row) for _, row in group.iterrows()]

        # ML prediction
        kernel_dicts = [
            {
                "kernel_name": s.name,
                "block_size": s.block_size,
                "gemm_m_per_block": s.gemm_m_per_block,
                "gemm_n_per_block": s.gemm_n_per_block,
                "pipeline": s.pipeline,
                "dtype": "bf16",
            }
            for s in all_kernels
        ]

        ranked = predictor.rank_kernels(problem, kernel_dicts)
        ml_name, ml_pred_tflops = ranked[0]
        ml_spec = next(s for s in all_kernels if s.name == ml_name)

        # Build both kernels
        oracle_so = build_kernel(oracle_spec, "bf16", "gfx950", verbose=False)
        ml_so = build_kernel(ml_spec, "bf16", "gfx950", verbose=False)

        if not oracle_so or not ml_so:
            print("  SKIP: Failed to build kernels")
            continue

        # Run both on hardware
        oracle_kernel_name = (
            oracle_so.stem[3:] if oracle_so.stem.startswith("lib") else oracle_so.stem
        )
        ml_kernel_name = ml_so.stem[3:] if ml_so.stem.startswith("lib") else ml_so.stem

        oracle_result = run_kernel_on_hw(oracle_so, problem, oracle_kernel_name)
        ml_result = run_kernel_on_hw(ml_so, problem, ml_kernel_name)

        if not oracle_result.get("ok") or not ml_result.get("ok"):
            print("  SKIP: Failed to run kernels")
            continue

        oracle_hw_tflops = oracle_result["tflops"]
        ml_hw_tflops = ml_result["tflops"]
        efficiency = (ml_hw_tflops / oracle_hw_tflops) * 100

        # Format problem description
        Ho = (problem["Hi"] - problem["Y"]) // problem["stride_h"] + 1
        Wo = (problem["Wi"] - problem["X"]) // problem["stride_w"] + 1
        prob_str = (
            f"C{problem['C']:4d}→K{problem['K']:4d} "
            f"{problem['Hi']:3d}x{problem['Wi']:3d}→{Ho:2d}x{Wo:2d} "
            f"f{problem['Y']}x{problem['X']} s{problem['stride_h']}x{problem['stride_w']}"
        )

        print(
            f"{prob_str:<40} {oracle_spec.name:<20} {ml_spec.name:<20} "
            f"{oracle_hw_tflops:>10.2f} {ml_hw_tflops:>10.2f} {efficiency:>11.1f}%"
        )

        results.append(
            {
                "problem": prob_str,
                "oracle_name": oracle_spec.name,
                "ml_name": ml_spec.name,
                "oracle_train_tflops": oracle_train_tflops,
                "oracle_hw_tflops": oracle_hw_tflops,
                "ml_pred_tflops": ml_pred_tflops,
                "ml_hw_tflops": ml_hw_tflops,
                "efficiency": efficiency,
                "same_kernel": oracle_spec.name == ml_spec.name,
            }
        )

    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    if results:
        avg_efficiency = np.mean([r["efficiency"] for r in results])
        same_kernel_count = sum(1 for r in results if r["same_kernel"])

        print(f"\nTests completed: {len(results)}")
        print(f"ML selected same kernel as oracle: {same_kernel_count}/{len(results)} ({(same_kernel_count/len(results))*100:.1f}%)")
        print(f"Average efficiency (ML vs Oracle): {avg_efficiency:.2f}%")

        avg_oracle = np.mean([r["oracle_hw_tflops"] for r in results])
        avg_ml = np.mean([r["ml_hw_tflops"] for r in results])
        print(f"\nAverage Oracle TFLOPS (on HW): {avg_oracle:.2f}")
        print(f"Average ML Selected TFLOPS (on HW): {avg_ml:.2f}")

        # Prediction accuracy (ML predicted vs actual HW for ML selected kernel)
        pred_accuracy = np.mean(
            [(r["ml_hw_tflops"] / r["ml_pred_tflops"]) * 100 for r in results]
        )
        print(f"\nML Prediction Accuracy (pred vs actual): {pred_accuracy:.1f}%")

        if avg_efficiency >= 95:
            print("\n✓ EXCELLENT: ML achieves >95% of oracle performance!")
        elif avg_efficiency >= 90:
            print("\n✓ GOOD: ML achieves >90% of oracle performance")
        else:
            print(f"\n⚠ ML efficiency {avg_efficiency:.1f}% - room for improvement")

    print("=" * 100)


if __name__ == "__main__":
    main()
