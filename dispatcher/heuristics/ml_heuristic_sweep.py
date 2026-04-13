#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ML Heuristic Sweep: Comprehensive GEMM Performance Evaluation

Sweeps across diverse problem shapes with ML-based kernel selection to measure
TFLOPS performance. Supports multiple dtypes (fp16, bf16, fp8) and validates
ML model predictions by executing kernels on GPU.

Shape Constraints (fp16/bf16 on gfx950):
- M >= 1 (any M is valid)
- N % 8 == 0 AND N >= 64
- K % 2 == 0 AND K >= 32

Usage:
    python ml_heuristic_sweep.py --dtype fp16 --num_shapes 256
    python ml_heuristic_sweep.py --dtypes fp16 bf16 --output sweep_results.csv
    python ml_heuristic_sweep.py --dtype fp16 --dry_run  # Prediction only, no GPU execution
"""

import sys
import argparse
import time
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
)

try:
    from predict import Predictor
    # from feature_engine import GemmUniversalFeatureEngine

    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("WARNING: ML heuristic modules not available. Will use first-fit selection.")


@dataclass
class KernelSpec:
    """Kernel specification for ML heuristic"""

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


# Comprehensive kernel pool covering diverse tile sizes and configurations
KERNEL_POOL = [
    # Small tiles (64x64)
    KernelSpec(
        "s_64x64_k32_v3", 64, 64, 32, "compv3", "intrawave", 2, 2, 1, 16, 16, 16
    ),
    KernelSpec(
        "s_64x64_k64_v3", 64, 64, 64, "compv3", "intrawave", 2, 2, 1, 16, 16, 16
    ),
    KernelSpec(
        "s_64x64_k128_v3", 64, 64, 128, "compv3", "intrawave", 2, 2, 1, 16, 16, 16
    ),
    KernelSpec(
        "s_64x64_k64_v4", 64, 64, 64, "compv4", "intrawave", 2, 2, 1, 16, 16, 16
    ),
    KernelSpec("s_64x64_k64_mem", 64, 64, 64, "mem", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec(
        "s_64x64_k128_mem", 64, 64, 128, "mem", "intrawave", 2, 2, 1, 16, 16, 16
    ),
    # Medium tiles (128x128)
    KernelSpec("m_128x128_k32_v3", 128, 128, 32, "compv3", "intrawave"),
    KernelSpec("m_128x128_k64_v3", 128, 128, 64, "compv3", "intrawave"),
    KernelSpec("m_128x128_k128_v3", 128, 128, 128, "compv3", "intrawave"),
    KernelSpec("m_128x128_k64_v4", 128, 128, 64, "compv4", "intrawave"),
    KernelSpec("m_128x128_k128_v4", 128, 128, 128, "compv4", "intrawave"),
    KernelSpec("m_128x128_k64_mem", 128, 128, 64, "mem", "intrawave"),
    KernelSpec("m_128x128_k128_mem", 128, 128, 128, "mem", "intrawave"),
    # Rectangular medium (M != N)
    KernelSpec(
        "r_64x128_k32_v3", 64, 128, 32, "compv3", "intrawave", 2, 2, 1, 16, 32, 16
    ),
    KernelSpec(
        "r_128x64_k32_v3", 128, 64, 32, "compv3", "intrawave", 2, 2, 1, 32, 16, 16
    ),
    KernelSpec(
        "r_64x128_k64_v3", 64, 128, 64, "compv3", "intrawave", 2, 2, 1, 16, 32, 16
    ),
    KernelSpec(
        "r_128x64_k64_v3", 128, 64, 64, "compv3", "intrawave", 2, 2, 1, 32, 16, 16
    ),
    KernelSpec(
        "r_64x256_k32_v3", 64, 256, 32, "compv3", "intrawave", 2, 2, 1, 16, 32, 16
    ),
    KernelSpec(
        "r_256x64_k32_v3", 256, 64, 32, "compv3", "intrawave", 2, 2, 1, 32, 16, 16
    ),
    # Large tiles (256x256)
    KernelSpec("l_256x128_k32_v3", 256, 128, 32, "compv3", "intrawave"),
    KernelSpec("l_128x256_k32_v3", 128, 256, 32, "compv3", "intrawave"),
    KernelSpec("l_256x256_k32_v3", 256, 256, 32, "compv3", "intrawave"),
    KernelSpec("l_256x256_k64_v3", 256, 256, 64, "compv3", "intrawave"),
    KernelSpec("l_256x256_k64_v4", 256, 256, 64, "compv4", "intrawave"),
    # Interwave variants
    KernelSpec("m_128x128_k64_iw_v3", 128, 128, 64, "compv3", "interwave"),
    KernelSpec("m_128x128_k128_iw_v3", 128, 128, 128, "compv3", "interwave"),
    KernelSpec("l_256x256_k32_iw_v3", 256, 256, 32, "compv3", "interwave"),
]


def generate_problem_shapes(num_shapes: int = 1024) -> List[Tuple[int, int, int]]:
    """
    Generate diverse problem shapes with hardware constraints:
    - M >= 1 (any M is valid, including tiny M for inference)
    - N % 8 == 0 AND N >= 64 (hardware alignment requirement)
    - K % 2 == 0 AND K >= 32 (fp16 requirement)

    Covers:
    - Powers of 2 (square and rectangular)
    - ML workloads (LLM attention, MLP, batch inference)
    - Non-power-of-2 dimensions (aligned to constraints)
    - Edge cases (tiny M, very large matrices, extreme aspect ratios)
    """
    shapes = []

    # 1. Powers of 2 - Square (64 to 8192) with K variations
    for p in range(6, 14):  # 2^6=64 to 2^13=8192
        dim = 2**p
        shapes.append((dim, dim, dim))
        if dim >= 128:
            # K variations (must be even and >= 32)
            shapes.append((dim, dim, dim // 2))
            shapes.append((dim, dim, dim * 2))
            shapes.append((dim, dim, max(32, dim // 4)))

    # 2. Small batch inference (1-256 batch, common hidden dims)
    # N must be multiple of 8 and >= 64
    hidden_dims = [768, 1024, 2048, 3072, 4096, 5120, 8192, 11008, 12288, 16384]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for hidden in hidden_dims:
        for batch in batch_sizes[:8]:
            shapes.append((batch, hidden, hidden))
            if hidden >= 4096:
                # LLM MLP projections (ensure K is even)
                k_mlp = hidden * 3 // 4
                if k_mlp % 2 == 1:
                    k_mlp += 1  # Make even
                if k_mlp >= 32:
                    shapes.append((batch, hidden, k_mlp))
                    shapes.append((batch, k_mlp, hidden))

    # 3. Attention patterns (seq_len x head_dim)
    # seq_len can be any value >= 1, total_dim must be multiple of 8
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    head_dims = [64, 80, 96, 128, 256]
    num_heads = [8, 12, 16, 32, 40, 64]

    for seq in seq_lens:
        for head_dim in head_dims:
            for nh in num_heads[:4]:
                total_dim = nh * head_dim
                # total_dim should be multiple of 8 (naturally satisfied for most cases)
                if total_dim % 8 == 0 and total_dim >= 64:
                    # head_dim must be even for K
                    if head_dim % 2 == 0 and head_dim >= 32:
                        shapes.append((seq, total_dim, head_dim))
                        shapes.append((seq, head_dim, total_dim))

    # 4. Rectangular matrices (extreme aspect ratios)
    # All dims must satisfy constraints
    dims_m = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    dims_n = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # N >= 64, N % 8 == 0
    dims_k = [
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]  # K >= 32, K % 2 == 0

    # Sample to avoid explosion
    for i, m in enumerate(dims_m):
        for j, n in enumerate(dims_n):
            for _l, k in enumerate(dims_k):
                if (i + j + _l) % 3 == 0:  # Stratified sampling
                    shapes.append((m, n, k))

    # 5. Non-power-of-2 dimensions (aligned to constraints)
    # N values: multiples of 8, >= 64
    non_pow2_n = [
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        136,
        144,
        152,
        160,
        176,
        184,
        192,
        200,
        224,
        240,
        272,
        288,
        304,
        320,
        336,
        352,
        368,
        384,
        400,
        416,
        448,
        480,
        544,
        576,
        640,
        672,
        704,
        736,
        768,
        800,
        832,
        896,
        960,
        1088,
        1152,
        1216,
        1280,
        1344,
        1408,
        1472,
        1536,
        1600,
        1664,
        1728,
        1792,
        1856,
        1920,
        2176,
        2304,
        2432,
        2560,
        2688,
        2816,
        2944,
        3072,
        3200,
        3328,
        3456,
        3584,
        3712,
        3840,
        3968,
        4224,
        4352,
        4480,
        4608,
        4736,
        4864,
        4992,
    ]

    # K values: even numbers >= 32
    non_pow2_k = [
        34,
        36,
        38,
        40,
        42,
        44,
        48,
        50,
        52,
        56,
        60,
        66,
        68,
        72,
        76,
        80,
        88,
        96,
        100,
        112,
        120,
        136,
        144,
        160,
        176,
        192,
        224,
        240,
        272,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        544,
        576,
        640,
        672,
        704,
        768,
        800,
        832,
        896,
        960,
        1088,
        1152,
        1280,
        1344,
        1408,
        1536,
        1600,
        1664,
        1792,
        1920,
    ]

    # M values: any value >= 1
    non_pow2_m = [
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        23,
        27,
        31,
        33,
        37,
        41,
        47,
        51,
        57,
        63,
        65,
        71,
        79,
        87,
        95,
        97,
        111,
        119,
        127,
        129,
        143,
        159,
        175,
        191,
        193,
        223,
        239,
        255,
        257,
        287,
        319,
        351,
        383,
        385,
        447,
        479,
        511,
        513,
        575,
        639,
        703,
        767,
        769,
        895,
        959,
        1023,
        1025,
    ]

    # Sample non-power-of-2 shapes
    for i, m in enumerate(non_pow2_m[:30]):
        for j, n in enumerate(non_pow2_n[:20]):
            for _l, k in enumerate(non_pow2_k[:15]):
                if (i + j + _l) % 4 == 0:  # Stratified sampling
                    shapes.append((m, n, k))

    # 6. Very tall K (memory-bound) - ensure N % 8 == 0, K % 2 == 0
    for mn in [64, 128, 256, 512, 1024]:
        for k in [4096, 8192, 16384]:
            shapes.append((mn, mn, k))

    # 7. Very short K (compute-bound) - ensure K >= 32, K % 2 == 0
    for mn in [512, 1024, 2048, 4096]:
        for k in [32, 64, 128]:
            shapes.append((mn, mn, k))

    # 8. Tiny M (edge cases for batch-1 inference)
    for m in [1, 2, 4, 8, 16, 32]:
        for n in [64, 128, 256, 512, 1024, 2048]:  # N >= 64, N % 8 == 0
            for k in [32, 64, 128, 256, 512]:  # K >= 32, K % 2 == 0
                shapes.append((m, n, k))

    # 9. Stress test sizes (aligned to constraints)
    stress_sizes = [
        (10000, 10000, 10000),
        (1000, 10000, 1000),
        (1000, 1000, 10000),
        (5000, 5000, 5000),
        (7168, 7168, 7168),  # Common LLM hidden dim
        (8192, 11008, 8192),  # LLaMA MLP dimensions
    ]
    shapes.extend(stress_sizes)

    # Remove duplicates while preserving order
    seen = set()
    unique_shapes = []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            unique_shapes.append(s)

    # Filter to ensure all shapes meet constraints
    valid_shapes = []
    for m, n, k in unique_shapes:
        if m >= 1 and n >= 64 and n % 8 == 0 and k >= 32 and k % 2 == 0:
            valid_shapes.append((m, n, k))

    # Sample down to target number if we have too many
    if len(valid_shapes) > num_shapes:
        # Stratified sampling to preserve diversity
        step = len(valid_shapes) / num_shapes
        valid_shapes = [valid_shapes[int(i * step)] for i in range(num_shapes)]

    return valid_shapes


def spec_to_feature_dict(spec: KernelSpec, dtype: str, layout: str) -> dict:
    """Convert KernelSpec to feature dict for ML predictor"""
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
        "pad_m": True,  # Enable padding to support arbitrary M dimensions
        "pad_n": True,  # Enable padding to support arbitrary N dimensions
        "pad_k": True,  # Enable padding to support arbitrary K dimensions
        "persistent": False,
        "dtype": dtype,
        "layout": layout,
    }


def spec_to_kernel_config(
    spec: KernelSpec, dtype: str, arch: str, dtype_acc: str = "fp32"
) -> KernelConfig:
    """Convert KernelSpec to KernelConfig for dispatcher"""
    return KernelConfig(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=dtype,
        dtype_acc=dtype_acc,
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
    predictor, pool: List[KernelSpec], M: int, N: int, K: int, dtype: str, layout: str
) -> Tuple[KernelSpec, float]:
    """Use ML model to select best kernel"""
    if not HAS_ML or predictor is None:
        # Fallback: select first kernel
        return pool[0], 0.0

    problem = {"m": M, "n": N, "k": K, "dtype": dtype, "layout": layout, "split_k": 1}
    kernel_dicts = [spec_to_feature_dict(s, dtype, layout) for s in pool]

    ranked = predictor.rank_kernels(problem, kernel_dicts)
    if not ranked:
        return pool[0], 0.0

    best_name, best_tflops = ranked[0]
    best_spec = next((s for s in pool if s.name == best_name), pool[0])
    return best_spec, best_tflops


def run_single_gemm(
    M: int,
    N: int,
    K: int,
    dtype: str,
    arch: str,
    predictor,
    dry_run: bool = False,
    dtype_acc: str = "fp32",
) -> dict:
    """Run a single GEMM with ML heuristic selection"""

    # Select kernel via ML heuristic
    t0 = time.time()
    best_spec, pred_tflops = ml_select_kernel(
        predictor, KERNEL_POOL, M, N, K, dtype, "rcr"
    )
    select_time_ms = (time.time() - t0) * 1000

    result = {
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "selected_kernel": best_spec.name,
        "predicted_tflops": pred_tflops,
        "selection_time_ms": select_time_ms,
        "actual_time_ms": 0,
        "actual_tflops": 0,
        "status": "SKIP" if dry_run else "PENDING",
        "error": None,
    }

    if dry_run:
        return result

    # Build and run kernel
    config = spec_to_kernel_config(best_spec, dtype, arch, dtype_acc)

    try:
        setup = setup_gemm_dispatcher(
            config=config,
            registry_name=f"sweep_{dtype}_{best_spec.name}",
            verbose=False,
            auto_rebuild=True,
        )

        if not setup.success:
            result["status"] = "BUILD_FAIL"
            result["error"] = "Failed to build kernel"
            cleanup_gemm()
            return result

        dispatcher = setup.dispatcher
        if not dispatcher.is_supported(M, N, K):
            result["status"] = "UNSUPPORTED"
            result["error"] = "Problem size not supported by kernel"
            cleanup_gemm()
            return result

        # Create input data
        np_dtype = {"fp16": np.float16, "bf16": np.float16, "fp8": np.float16}[dtype]
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
        B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

        # Run GEMM
        exec_result = dispatcher.run(A, B, M, N, K)

        if exec_result.success:
            result["actual_time_ms"] = exec_result.time_ms
            result["actual_tflops"] = exec_result.tflops
            result["status"] = "SUCCESS"
        else:
            # Decode status code for better error message
            status_messages = {
                0: "Success",
                -1: "GPU/HIP error (check permissions, memory, or kernel validity)",
                -2: "No suitable kernel found for this problem size",
            }
            error_msg = status_messages.get(exec_result.status, f"Unknown error (status={exec_result.status})")
            result["status"] = "RUN_FAIL"
            result["error"] = f"{error_msg} (status_code={exec_result.status})"

            # Print detailed error for debugging
            print(f"  ERROR: {error_msg}")
            print(f"  Status code: {exec_result.status}")
            print(f"  Time returned: {exec_result.time_ms}")
            print(f"  Kernel: {exec_result.kernel_name}")

        cleanup_gemm()

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]
        cleanup_gemm()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ML Heuristic Sweep: Test GEMM across many shapes and dtypes"
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["fp16"],
        choices=["fp16", "bf16", "fp8"],
        help="Data types to test (default: fp16)",
    )
    parser.add_argument(
        "--arch", default="gfx950", help="GPU architecture (default: gfx950)"
    )
    parser.add_argument(
        "--dtype_acc",
        default="fp32",
        choices=["fp16", "fp32"],
        help="Accumulator data type (default: fp32)",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to model directory (auto-detect if not specified)",
    )
    parser.add_argument(
        "--num_shapes",
        type=int,
        default=256,
        help="Number of problem shapes to test (default: 256)",
    )
    parser.add_argument(
        "--output",
        default="ml_heuristic_sweep_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only predict, do not run kernels (fast validation)",
    )

    args = parser.parse_args()

    # Setup ML predictor
    predictor = None
    if HAS_ML:
        if args.model_dir is None:
            # Auto-detect model directory based on first dtype
            first_dtype = args.dtypes[0]
            heuristics_dir = Path(__file__).parent
            model_candidates = [
                heuristics_dir / "models" / f"gemm_universal_{first_dtype}_{args.arch}",
            ]
            for model_dir in model_candidates:
                if model_dir.exists():
                    args.model_dir = str(model_dir)
                    break

        if args.model_dir and Path(args.model_dir).exists():
            try:
                predictor = Predictor(args.model_dir)
                print(f"✓ Loaded ML model from: {args.model_dir}")
            except Exception as e:
                print(f"⚠ Failed to load ML model: {e}")
                print("  Will use first-fit selection instead")
        else:
            print(f"⚠ Model directory not found: {args.model_dir}")
            print("  Will use first-fit selection instead")

    # Generate problem shapes
    print(f"\nGenerating {args.num_shapes} problem shapes...")
    shapes = generate_problem_shapes(args.num_shapes)
    print(
        f"✓ Generated {len(shapes)} valid shapes (M>=1, N%8==0, N>=64, K%2==0, K>=32)"
    )

    # Validate all shapes meet constraints
    invalid = [
        (m, n, k)
        for m, n, k in shapes
        if not (m >= 1 and n >= 64 and n % 8 == 0 and k >= 32 and k % 2 == 0)
    ]
    if invalid:
        print(f"⚠ WARNING: {len(invalid)} shapes violate constraints!")
        print(f"  First few: {invalid[:5]}")

    # Print configuration
    print("\n" + "=" * 80)
    print("  ML Heuristic Sweep Configuration")
    print("=" * 80)
    print(
        f"  Model:          {args.model_dir if args.model_dir else 'first-fit (no ML)'}"
    )
    print(f"  Data types:     {', '.join(args.dtypes)}")
    print(f"  Accumulator:    {args.dtype_acc}")
    print(f"  Architecture:   {args.arch}")
    print(f"  Kernel pool:    {len(KERNEL_POOL)} kernels")
    print(f"  Problem shapes: {len(shapes)}")
    print(f"  Total tests:    {len(shapes) * len(args.dtypes)}")
    print(
        f"  Mode:           {'DRY RUN (prediction only)' if args.dry_run else 'FULL RUN (execute kernels)'}"
    )
    print(f"  Output:         {args.output}")
    print("=" * 80)

    # Open output CSV
    csv_file = open(args.output, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "dtype",
            "M",
            "N",
            "K",
            "selected_kernel",
            "predicted_tflops",
            "selection_time_ms",
            "actual_time_ms",
            "actual_tflops",
            "status",
            "error",
        ],
    )
    csv_writer.writeheader()

    # Run sweep
    total_tests = len(shapes) * len(args.dtypes)
    completed = 0
    start_time = time.time()

    print("\nStarting sweep... (Ctrl+C to stop and save partial results)\n")

    try:
        for dtype in args.dtypes:
            print(f"\n{'=' * 80}")
            print(f"  Testing dtype: {dtype.upper()}")
            print(f"{'=' * 80}\n")

            for i, (M, N, K) in enumerate(shapes):
                result = run_single_gemm(
                    M, N, K, dtype, args.arch, predictor, args.dry_run, args.dtype_acc
                )

                # Write to CSV
                csv_writer.writerow(result)
                csv_file.flush()

                completed += 1

                # Progress update
                if completed % 10 == 0 or result["status"] != "SUCCESS":
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_tests - completed) / rate if rate > 0 else 0

                    status_emoji = {
                        "SUCCESS": "✓",
                        "SKIP": "→",
                        "BUILD_FAIL": "✗",
                        "UNSUPPORTED": "○",
                        "RUN_FAIL": "✗",
                        "ERROR": "✗",
                    }.get(result["status"], "?")

                    print(
                        f"  [{completed:4d}/{total_tests}] {status_emoji} "
                        f"{dtype:4s} {M:5d}x{N:5d}x{K:5d} → "
                        f"{result['selected_kernel']:20s} "
                        f"pred={result['predicted_tflops']:6.1f} "
                        f"actual={result['actual_tflops']:6.1f} TFLOPS  "
                        f"[{rate:.1f} tests/s, ETA {eta / 60:.1f}m]"
                    )

    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted! Saving partial results to {args.output}...")

    finally:
        csv_file.close()

    # Summary
    print("\n" + "=" * 80)
    print("  SWEEP COMPLETE")
    print("=" * 80)

    # Read back results and compute statistics
    results = []
    with open(args.output, "r") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    print(f"\n  Total tests:     {len(results)}")
    print(f"  Output file:     {args.output}")

    if not args.dry_run:
        success = [r for r in results if r["status"] == "SUCCESS"]
        print(
            f"  Successful:      {len(success)} ({100 * len(success) / len(results):.1f}%)"
        )

        if success:
            avg_tflops = np.mean([float(r["actual_tflops"]) for r in success])
            max_tflops = max([float(r["actual_tflops"]) for r in success])
            print(f"  Avg TFLOPS:      {avg_tflops:.2f}")
            print(f"  Max TFLOPS:      {max_tflops:.2f}")

            # Per-dtype breakdown
            for dtype in args.dtypes:
                dtype_results = [r for r in success if r["dtype"] == dtype]
                if dtype_results:
                    avg = np.mean([float(r["actual_tflops"]) for r in dtype_results])
                    print(
                        f"    {dtype:4s}:          {avg:.2f} TFLOPS (n={len(dtype_results)})"
                    )

    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
