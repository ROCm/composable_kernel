#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 09: ML-Based Kernel Selection

Uses a trained LightGBM model to select the optimal kernel for each problem
size. The model predicts TFLOPS for every candidate in the kernel pool and
picks the highest-scoring one, which is then JIT-compiled and run.

This replaces the hand-crafted rules in 08_heuristics.py with a data-driven
approach achieving 97-98% of oracle-best TFLOPS efficiency.

Complexity: *****

Prerequisites:
    - Trained model in dispatcher/heuristics/models/gemm_universal_fp8_gfx950/
    - lightgbm, pandas, numpy, pyarrow installed

Usage:
    python3 09_ml_heuristic.py
    python3 09_ml_heuristic.py --dtype fp16 --arch gfx942
"""

import sys
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "heuristics"))

import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
)

from predict import Predictor


@dataclass
class KernelSpec:
    """Kernel specification -- same structure as 08_heuristics.py"""

    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    pipeline: str = "compv3"
    scheduler: str = "intrawave"
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1
    warp_m: int = 32
    warp_n: int = 32
    warp_k: int = 16


# Kernel pool: representative configs spanning small to large tiles,
# compv3/compv4/mem pipelines, and intrawave/interwave schedulers.
KERNEL_POOL = [
    # Small tiles
    KernelSpec("s_64x64_k32_v3", 64, 64, 32, "compv3", warp_m=16, warp_n=16),
    KernelSpec("s_64x64_k64_v3", 64, 64, 64, "compv3", warp_m=16, warp_n=16),
    KernelSpec("s_64x64_k128_v3", 64, 64, 128, "compv3", warp_m=16, warp_n=16),
    KernelSpec("s_64x64_k32_v4", 64, 64, 32, "compv4", warp_m=16, warp_n=16),
    KernelSpec("s_64x64_k64_mem", 64, 64, 64, "mem", warp_m=16, warp_n=16),
    KernelSpec("s_64x64_k128_mem", 64, 64, 128, "mem", warp_m=16, warp_n=16),
    # Medium tiles
    KernelSpec("m_128x128_k32_v3", 128, 128, 32, "compv3"),
    KernelSpec("m_128x128_k64_v3", 128, 128, 64, "compv3"),
    KernelSpec("m_128x128_k128_v3", 128, 128, 128, "compv3"),
    KernelSpec("m_128x128_k32_v4", 128, 128, 32, "compv4"),
    KernelSpec("m_128x128_k64_v4", 128, 128, 64, "compv4"),
    KernelSpec("m_128x128_k64_mem", 128, 128, 64, "mem"),
    KernelSpec("m_128x128_k128_mem", 128, 128, 128, "mem"),
    # Rectangular medium
    KernelSpec("r_64x128_k32", 64, 128, 32, "compv3", warp_m=16),
    KernelSpec("r_128x64_k32", 128, 64, 32, "compv3", warp_n=16),
    KernelSpec("r_64x128_k64", 64, 128, 64, "compv3", warp_m=16),
    KernelSpec("r_128x64_k64", 128, 64, 64, "compv3", warp_n=16),
    # Large tiles
    KernelSpec("l_256x128_k32", 256, 128, 32, "compv3"),
    KernelSpec("l_128x256_k32", 128, 256, 32, "compv3"),
    KernelSpec("l_256x256_k32", 256, 256, 32, "compv3"),
    KernelSpec("l_256x256_k64", 256, 256, 64, "compv3"),
    # Interwave variants
    KernelSpec("m_128x128_k64_iw", 128, 128, 64, "compv3", "interwave"),
    KernelSpec("m_128x128_k64_mem_iw", 128, 128, 64, "mem", "interwave"),
]


def spec_to_feature_dict(spec: KernelSpec, dtype: str, layout: str) -> dict:
    """Convert a KernelSpec to the dict format the feature engine expects.

    Note: pad_m/n/k default to True to match KernelConfig defaults and actual
    compiled kernels. This ensures the ML model receives the correct padding
    flags that will be used during JIT compilation.
    """
    return {
        "kernel_name": spec.name,
        "tile_m": spec.tile_m,
        "tile_n": spec.tile_n,
        "tile_k": spec.tile_k,
        "warp_m": spec.wave_m,
        "warp_n": spec.wave_n,
        "warp_k": spec.wave_k,
        "warp_tile_m": spec.warp_m,
        "warp_tile_n": spec.warp_n,
        "warp_tile_k": spec.warp_k,
        "pipeline": spec.pipeline,
        "scheduler": spec.scheduler,
        "epilogue": "cshuffle",
        "pad_m": True,  # Match KernelConfig default
        "pad_n": True,  # Match KernelConfig default
        "pad_k": True,  # Match KernelConfig default
        "persistent": False,
        "dtype": dtype,
        "layout": layout,
    }


def spec_to_kernel_config(spec: KernelSpec, dtype: str, arch: str) -> KernelConfig:
    """Convert a KernelSpec to the dispatcher's KernelConfig for JIT compilation."""
    return KernelConfig(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=dtype,
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        wave_m=spec.wave_m,
        wave_n=spec.wave_n,
        wave_k=spec.wave_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        warp_k=spec.warp_k,
        pipeline=spec.pipeline,
        scheduler=spec.scheduler,
        epilogue="cshuffle",
        gfx_arch=arch,
    )


def ml_select_kernel(
    predictor: Predictor,
    pool: List[KernelSpec],
    M: int,
    N: int,
    K: int,
    dtype: str,
    layout: str,
) -> tuple:
    """Score all kernels in the pool and return (best_spec, predicted_tflops)."""
    problem = {"m": M, "n": N, "k": K, "dtype": dtype, "layout": layout, "split_k": 1}
    kernel_dicts = [spec_to_feature_dict(s, dtype, layout) for s in pool]

    ranked = predictor.rank_kernels(problem, kernel_dicts)
    if not ranked:
        return pool[0], 0.0

    best_name, best_tflops = ranked[0]
    best_spec = next((s for s in pool if s.name == best_name), pool[0])
    return best_spec, best_tflops


def main():
    parser = argparse.ArgumentParser(description="ML-based kernel selection for GEMM")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp8"])
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument(
        "--model_dir",
        default=str(
            Path(__file__).parent.parent.parent.parent
            / "heuristics"
            / "models"
            / "gemm_universal_fp8_gfx950"
        ),
    )
    parser.add_argument(
        "--no_run", action="store_true", help="Only predict, don't run GEMMs"
    )
    args = parser.parse_args()

    print("=" * 75)
    print("  Example 09: ML-Based Kernel Selection")
    print("=" * 75)
    print(f"\n  Model:  {args.model_dir}")
    print(f"  Dtype:  {args.dtype}")
    print(f"  Arch:   {args.arch}")
    print(f"  Pool:   {len(KERNEL_POOL)} kernels")

    predictor = Predictor(args.model_dir)
    print("  Model loaded successfully")

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float16

    test_sizes = [
        (128, 128, 64),
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
        (2048, 2048, 1024),
    ]

    header = f"{'Shape':<20} {'Selected Kernel':<25} {'Pred TFLOPS':>12}"
    if not args.no_run:
        header += f" {'Time (ms)':>10} {'TFLOPS':>10} {'Status':<8}"
    print(f"\n  {header}")
    print("  " + "-" * len(header))

    results = []

    for M, N, K in test_sizes:
        t0 = time.time()
        best_spec, pred_tflops = ml_select_kernel(
            predictor, KERNEL_POOL, M, N, K, args.dtype, "rcr"
        )
        _ = (time.time() - t0) * 1000  # ML selection time (unused)

        size_str = f"{M}x{N}x{K}"
        line = f"  {size_str:<20} {best_spec.name:<25} {pred_tflops:>12.2f}"

        if args.no_run:
            print(line)
            results.append((size_str, best_spec.name, True, 0, pred_tflops))
            continue

        config = spec_to_kernel_config(best_spec, args.dtype, args.arch)

        setup = setup_gemm_dispatcher(
            config=config,
            registry_name=f"ml_{best_spec.name}",
            verbose=False,
            auto_rebuild=True,
        )

        if not setup.success:
            line += f" {'N/A':>10} {'N/A':>10} {'BUILD':>8}"
            print(line)
            results.append((size_str, best_spec.name, False, 0, 0))
            cleanup_gemm()
            continue

        dispatcher = setup.dispatcher
        if not dispatcher.is_supported(M, N, K):
            line += f" {'N/A':>10} {'N/A':>10} {'UNSUP':>8}"
            print(line)
            results.append((size_str, best_spec.name, False, 0, 0))
            cleanup_gemm()
            continue

        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
        B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

        result = dispatcher.run(A, B, M, N, K)

        if result.success:
            C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(
                np_dtype
            )
            max_err = np.max(np.abs(result.output - C_ref))
            passed = max_err < 1e-2
            status = "PASS" if passed else "FAIL"
            line += f" {result.time_ms:>10.4f} {result.tflops:>10.2f} {status:<8}"
            results.append(
                (size_str, best_spec.name, passed, result.time_ms, result.tflops)
            )
        else:
            line += f" {'N/A':>10} {'N/A':>10} {'FAIL':<8}"
            results.append((size_str, best_spec.name, False, 0, 0))

        print(line)
        cleanup_gemm()

    # Summary
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    passed = sum(1 for r in results if r[2])
    print(f"\n  Results: {passed}/{len(results)} tests passed")
    valid = [r for r in results if r[2] and r[4] > 0]
    if valid:
        avg = sum(r[4] for r in valid) / len(valid)
        print(f"  Average TFLOPS: {avg:.2f}")
    if passed == len(results):
        print("\n  *** ALL TESTS PASSED ***")
    print("=" * 75)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
