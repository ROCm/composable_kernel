#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 09: ML-Based Kernel Selection for Grouped Convolution

Uses a trained LightGBM model to select the optimal kernel for each convolution
problem. The model predicts TFLOPS for every candidate in the kernel pool and
picks the highest-scoring one, which is then invoked via the dispatcher.

This replaces hand-crafted heuristics with a data-driven approach achieving
97%+ of oracle-best TFLOPS efficiency.

Supports forward, bwd_data, and bwd_weight variants.

Complexity: *****

Prerequisites:
    - Trained models in dispatcher/heuristics/models/grouped_conv_*_bf16_gfx950/
    - lightgbm, pandas, numpy, pyarrow installed
    - grouped_conv dispatcher built

Usage:
    python3 09_ml_heuristic.py --variant forward
    python3 09_ml_heuristic.py --variant bwd_data
    python3 09_ml_heuristic.py --variant bwd_weight
    python3 09_ml_heuristic.py --variant forward --dtype bf16 --arch gfx950
"""

import sys
import os
import argparse
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "heuristics"))


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

    def to_kernel_config(self, dtype: str = "bf16", arch: str = "gfx950", variant: str = "forward") -> GroupedConvKernelConfig:
        """Convert to GroupedConvKernelConfig for building."""
        return GroupedConvKernelConfig(
            variant=variant,
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


# Kernel pools for different variants

# Forward pool: compv3, compv4, compv5 (30 kernels)
FORWARD_KERNEL_POOL = [
    # Block size 16
    KernelSpec("k16_64x64_v3", 16, 64, 64, "compv3"),
    KernelSpec("k16_64x64_v4", 16, 64, 64, "compv4"),
    KernelSpec("k16_64x64_v5", 16, 64, 64, "compv5"),
    KernelSpec("k16_64x128_v3", 16, 64, 128, "compv3"),
    KernelSpec("k16_64x128_v4", 16, 64, 128, "compv4"),
    KernelSpec("k16_64x128_v5", 16, 64, 128, "compv5"),
    # Block size 32
    KernelSpec("k32_64x64_v3", 32, 64, 64, "compv3"),
    KernelSpec("k32_64x64_v4", 32, 64, 64, "compv4"),
    KernelSpec("k32_64x64_v5", 32, 64, 64, "compv5"),
    KernelSpec("k32_64x128_v3", 32, 64, 128, "compv3"),
    KernelSpec("k32_64x128_v4", 32, 64, 128, "compv4"),
    KernelSpec("k32_64x128_v5", 32, 64, 128, "compv5"),
    KernelSpec("k32_128x64_v3", 32, 128, 64, "compv3"),
    KernelSpec("k32_128x64_v4", 32, 128, 64, "compv4"),
    KernelSpec("k32_128x64_v5", 32, 128, 64, "compv5"),
    # Block size 64
    KernelSpec("k64_64x64_v3", 64, 64, 64, "compv3"),
    KernelSpec("k64_64x64_v4", 64, 64, 64, "compv4"),
    KernelSpec("k64_64x64_v5", 64, 64, 64, "compv5"),
    KernelSpec("k64_64x128_v3", 64, 64, 128, "compv3"),
    KernelSpec("k64_64x128_v4", 64, 64, 128, "compv4"),
    KernelSpec("k64_64x128_v5", 64, 64, 128, "compv5"),
    KernelSpec("k64_128x64_v3", 64, 128, 64, "compv3"),
    KernelSpec("k64_128x64_v4", 64, 128, 64, "compv4"),
    KernelSpec("k64_128x64_v5", 64, 128, 64, "compv5"),
    # Block size 128
    KernelSpec("k128_64x128_v3", 128, 64, 128, "compv3"),
    KernelSpec("k128_64x128_v4", 128, 64, 128, "compv4"),
    KernelSpec("k128_64x128_v5", 128, 64, 128, "compv5"),
    KernelSpec("k128_128x64_v3", 128, 128, 64, "compv3"),
    KernelSpec("k128_128x64_v4", 128, 128, 64, "compv4"),
    KernelSpec("k128_128x64_v5", 128, 128, 64, "compv5"),
]

# Backward pool: compv3, mem (20 kernels)
BACKWARD_KERNEL_POOL = [
    # Block size 16
    KernelSpec("k16_64x64_v3", 16, 64, 64, "compv3"),
    KernelSpec("k16_64x64_mem", 16, 64, 64, "mem"),
    KernelSpec("k16_64x128_v3", 16, 64, 128, "compv3"),
    KernelSpec("k16_64x128_mem", 16, 64, 128, "mem"),
    # Block size 32
    KernelSpec("k32_64x64_v3", 32, 64, 64, "compv3"),
    KernelSpec("k32_64x64_mem", 32, 64, 64, "mem"),
    KernelSpec("k32_64x128_v3", 32, 64, 128, "compv3"),
    KernelSpec("k32_64x128_mem", 32, 64, 128, "mem"),
    KernelSpec("k32_128x64_v3", 32, 128, 64, "compv3"),
    KernelSpec("k32_128x64_mem", 32, 128, 64, "mem"),
    # Block size 64
    KernelSpec("k64_64x64_v3", 64, 64, 64, "compv3"),
    KernelSpec("k64_64x64_mem", 64, 64, 64, "mem"),
    KernelSpec("k64_64x128_v3", 64, 64, 128, "compv3"),
    KernelSpec("k64_64x128_mem", 64, 64, 128, "mem"),
    KernelSpec("k64_128x64_v3", 64, 128, 64, "compv3"),
    KernelSpec("k64_128x64_mem", 64, 128, 64, "mem"),
    # Block size 128
    KernelSpec("k128_64x128_v3", 128, 64, 128, "compv3"),
    KernelSpec("k128_64x128_mem", 128, 64, 128, "mem"),
    KernelSpec("k128_128x64_v3", 128, 128, 64, "compv3"),
    KernelSpec("k128_128x64_mem", 128, 128, 64, "mem"),
]

# Legacy name for backward compatibility
KERNEL_POOL = FORWARD_KERNEL_POOL


def spec_to_feature_dict(spec: KernelSpec, dtype: str) -> dict:
    """Convert a KernelSpec to the dict format the feature engine expects."""
    return {
        "kernel_name": spec.name,
        "block_size": spec.block_size,
        "gemm_m_per_block": spec.gemm_m_per_block,
        "gemm_n_per_block": spec.gemm_n_per_block,
        "pipeline": spec.pipeline,
        "dtype": dtype,
    }


def build_kernel(spec: KernelSpec, dtype: str, arch: str, variant: str = "forward", verbose: bool = False) -> Path:
    """Build a kernel on-demand using the dispatcher's JIT compilation.

    Uses the same workflow as tile_engine benchmark:
      1. Convert KernelSpec to GroupedConvKernelConfig
      2. Call setup_multiple_grouped_conv_dispatchers to build
      3. Return path to .so file

    Returns:
        Path to compiled .so file, or None if build failed
    """
    kernel_config = spec.to_kernel_config(dtype=dtype, arch=arch, variant=variant)

    if verbose:
        print(f"  Building kernel: {spec.name}")
        print(f"    Config: variant={variant}, tile={kernel_config.tile_str}, pipeline={kernel_config.pipeline}")

    # Build kernel (returns list of paths)
    lib_paths = setup_multiple_grouped_conv_dispatchers(
        [kernel_config], verbose=verbose, max_workers=1
    )

    if not lib_paths or lib_paths[0] is None:
        return None

    return lib_paths[0]


def run_kernel_via_subprocess(so_path: Path, problem: dict, kernel_name: str) -> dict:
    """Run a kernel via the isolated subprocess runner.

    This uses the same pattern as the tile_engine benchmark to avoid GPU context issues.
    """
    script_path = Path(__file__).parent.parent.parent.parent.parent / "tile_engine" / "ops" / "grouped_conv" / "run_one_grouped_conv_kernel.py"

    # Prepare input JSON
    input_data = {
        "so_path": str(so_path),
        "problem": problem,
        "kernel_name": kernel_name
    }

    # Set environment for Python path
    env = {
        "GCONV_PYPATH": str(Path(__file__).parent.parent.parent.parent / "python")
    }

    # Run subprocess
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **env}
    )

    stdout, stderr = proc.communicate(input=json.dumps(input_data).encode())

    # Parse result
    try:
        result = json.loads(stdout.decode().strip())
        return result
    except:
        return {"ok": False, "error": f"Failed to parse output: {stdout.decode()}"}


def ml_select_and_run(
    predictor: Predictor,
    pool: List[KernelSpec],
    N: int,
    C: int,
    K: int,
    G: int,
    Hi: int,
    Wi: int,
    Y: int,
    X: int,
    stride_h: int,
    stride_w: int,
    pad_h: int = 0,
    pad_w: int = 0,
    dtype: str = "bf16",
    arch: str = "gfx950",
    variant: str = "forward",
    run_on_hw: bool = True,
) -> dict:
    """
    Step 1: Call predictor to get best kernel
    Step 2: Invoke dispatcher using tile_engine pattern

    Returns dict with prediction and (optional) hardware results.
    """
    # Step 1: Predict best kernel
    problem = {
        "N": N,
        "C": C,
        "K": K,
        "G": G,
        "Hi": Hi,
        "Wi": Wi,
        "Y": Y,
        "X": X,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "dtype": dtype,
    }

    kernel_dicts = [spec_to_feature_dict(s, dtype) for s in pool]
    ranked = predictor.rank_kernels(problem, kernel_dicts)

    if not ranked:
        return {"success": False, "error": "No valid kernel predictions"}

    best_name, pred_tflops = ranked[0]
    best_spec = next((s for s in pool if s.name == best_name), pool[0])

    result = {
        "success": True,
        "kernel_name": best_spec.name,
        "kernel_spec": best_spec,
        "predicted_tflops": pred_tflops,
    }

    if not run_on_hw:
        return result

    # Step 2: Build and run on hardware via dispatcher
    # Build kernel on-demand using JIT compilation
    so_path = build_kernel(best_spec, dtype, arch, variant=variant, verbose=False)

    if not so_path:
        result["hw_success"] = False
        result["hw_error"] = f"Failed to build kernel: {best_spec.name}"
        return result

    # Prepare problem dict for dispatcher
    problem_with_direction = {**problem, "direction": variant}

    # Get kernel name from .so path (e.g., libgrouped_conv_forward_bf16_2d_16x64x128_compv3.so -> grouped_conv_...)
    kernel_name = so_path.stem[3:] if so_path.stem.startswith("lib") else so_path.stem

    # Run via subprocess
    hw_result = run_kernel_via_subprocess(so_path, problem_with_direction, kernel_name)

    if hw_result.get("ok"):
        result["hw_success"] = True
        result["hw_time_ms"] = hw_result["ms"]
        result["hw_tflops"] = hw_result["tflops"]
    else:
        result["hw_success"] = False
        result["hw_error"] = hw_result.get("error", "Unknown error")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ML-based kernel selection for grouped convolution"
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument(
        "--variant",
        default="forward",
        choices=["forward", "bwd_data", "bwd_weight"],
        help="Convolution variant (default: forward)",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Model directory (default: auto-detect from variant)",
    )
    parser.add_argument(
        "--no_run", action="store_true", help="Only predict, don't run on hardware"
    )
    args = parser.parse_args()

    # Auto-detect model directory from variant if not specified
    if args.model_dir is None:
        model_name = f"grouped_conv_{args.variant}_bf16_{args.arch}"
        args.model_dir = str(
            Path(__file__).parent.parent.parent.parent
            / "heuristics"
            / "models"
            / model_name
        )

    # Select kernel pool based on variant
    if args.variant == "forward":
        kernel_pool = FORWARD_KERNEL_POOL
    else:
        kernel_pool = BACKWARD_KERNEL_POOL

    print("=" * 80)
    print(f"  Example 09: ML-Based Kernel Selection for Grouped Convolution ({args.variant.upper()})")
    print("=" * 80)
    print(f"\n  Variant: {args.variant}")
    print(f"  Model:   {args.model_dir}")
    print(f"  Dtype:   {args.dtype}")
    print(f"  Arch:    {args.arch}")
    print(f"  Pool:    {len(kernel_pool)} kernels")

    # Load ML model with grouped conv feature engine
    feature_engine = GroupedConvFeatureEngine()
    predictor = Predictor(args.model_dir, feature_engine=feature_engine)
    print("  Model loaded successfully")

    # Test problems: diverse convolution shapes from MIOpen
    # (N, C, K, G, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w)
    if args.variant == "forward":
        test_problems = [
            # ResNet-50 layers
            (1, 256, 512, 1, 56, 56, 1, 1, 2, 2, 0, 0),  # stride-2 1x1 conv
            (1, 128, 256, 1, 32, 32, 2, 2, 2, 2, 0, 0),  # stride-2 2x2 conv
            (1, 512, 256, 1, 28, 28, 1, 1, 1, 1, 0, 0),  # 1x1 bottleneck
            # 3x3 convolutions
            (1, 128, 256, 1, 64, 64, 3, 3, 1, 1, 1, 1),  # standard 3x3
            (1, 64, 128, 1, 128, 128, 3, 3, 1, 1, 1, 1),  # larger spatial
            # Small spatial
            (1, 832, 128, 1, 7, 7, 1, 1, 1, 1, 0, 0),  # 7x7 input
            # Large channels
            (1, 1024, 512, 1, 14, 14, 1, 1, 1, 1, 0, 0),  # large C/K
        ]
    elif args.variant == "bwd_data":
        test_problems = [
            # Typical backward data problems (with padding for 3x3)
            (32, 128, 256, 1, 28, 28, 3, 3, 1, 1, 1, 1),  # 3x3 standard
            (16, 256, 512, 1, 14, 14, 3, 3, 1, 1, 1, 1),  # 3x3 larger channels
            (64, 64, 128, 1, 56, 56, 1, 1, 1, 1, 0, 0),  # 1x1 conv
            (32, 512, 256, 1, 7, 7, 3, 3, 1, 1, 1, 1),  # small spatial
        ]
    else:  # bwd_weight
        test_problems = [
            # Typical backward weight problems (with padding for 3x3)
            (64, 256, 512, 1, 14, 14, 3, 3, 1, 1, 1, 1),  # 3x3 standard
            (32, 128, 256, 1, 28, 28, 3, 3, 1, 1, 1, 1),  # 3x3 medium
            (128, 64, 128, 1, 56, 56, 1, 1, 1, 1, 0, 0),  # 1x1 conv
            (64, 512, 1024, 1, 7, 7, 3, 3, 1, 1, 1, 1),  # large channels
        ]

    run_on_hw = not args.no_run

    if run_on_hw:
        header = f"{'Problem':<35} {'Selected':<22} {'Pred TFLOPS':>12} {'HW Time':>10} {'HW TFLOPS':>10} {'Status':<8}"
    else:
        header = f"{'Problem':<35} {'Selected':<22} {'Pred TFLOPS':>12}"

    print(f"\n  {header}")
    print("  " + "-" * len(header))

    results = []

    for N, C, K, G, Hi, Wi, Y, X, sh, sw, ph, pw in test_problems:
        result = ml_select_and_run(
            predictor, kernel_pool, N, C, K, G, Hi, Wi, Y, X, sh, sw, ph, pw,
            dtype=args.dtype, arch=args.arch, variant=args.variant, run_on_hw=run_on_hw
        )

        # Compute output size
        Ho = (Hi + 2*ph - Y) // sh + 1
        Wo = (Wi + 2*pw - X) // sw + 1

        prob_str = f"C{C:4d}→K{K:4d} {Hi:3d}x{Wi:3d}→{Ho:2d}x{Wo:2d} f{Y}x{X}"

        if not result["success"]:
            line = f"  {prob_str:<35} {'ERROR':<22} {'N/A':>12}"
            print(line)
            continue

        line = f"  {prob_str:<35} {result['kernel_name']:<22} {result['predicted_tflops']:>12.2f}"

        if run_on_hw:
            if result.get("hw_success"):
                hw_time = result["hw_time_ms"]
                hw_tflops = result["hw_tflops"]
                status = "PASS"
                line += f" {hw_time:>10.4f} {hw_tflops:>10.2f} {status:<8}"
                results.append((prob_str, result['kernel_name'], True, hw_time, hw_tflops, result['predicted_tflops']))
            else:
                error = result.get("hw_error", "Unknown")
                line += f" {'N/A':>10} {'N/A':>10} {'FAIL':<8}"
                print(line)
                print(f"    Error: {error}")
                results.append((prob_str, result['kernel_name'], False, 0, 0, result['predicted_tflops']))
                continue
        else:
            results.append((prob_str, result['kernel_name'], True, 0, 0, result['predicted_tflops']))

        print(line)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if run_on_hw:
        passed = sum(1 for r in results if r[2])
        print(f"\n  Results: {passed}/{len(results)} tests passed")
        valid = [r for r in results if r[2] and r[4] > 0]
        if valid:
            avg_hw = sum(r[4] for r in valid) / len(valid)
            avg_pred = sum(r[5] for r in valid) / len(valid)
            print(f"  Average HW TFLOPS: {avg_hw:.2f}")
            print(f"  Average Predicted TFLOPS: {avg_pred:.2f}")
            print(f"  Prediction Accuracy: {(avg_hw/avg_pred)*100:.1f}%")
        if passed == len(results):
            print("\n  *** ALL TESTS PASSED ***")
    else:
        print(f"\n  Results: {len(results)} predictions completed")
        avg_pred = sum(r[5] for r in results) / len(results)
        print(f"  Average Predicted TFLOPS: {avg_pred:.2f}")
        print("\n  Note: Hardware execution disabled (--no_run)")

    print("=" * 80)
    return 0 if (not run_on_hw or sum(1 for r in results if r[2]) == len(results)) else 1


if __name__ == "__main__":
    import os
    sys.exit(main())
