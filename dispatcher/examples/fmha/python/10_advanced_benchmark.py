#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 10: Advanced FMHA Benchmarking

Benchmarks FMHA forward across multiple problem sizes with configurable
warmup, repeat, and cache-flush settings.  Reports min/avg/max/median
time and TFLOPS for each problem.

Usage:
    python3 10_advanced_benchmark.py
    python3 10_advanced_benchmark.py --warmup 10 --repeat 50
    python3 10_advanced_benchmark.py --flush-cache
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    setup_fmha_dispatcher,
    detect_gpu_arch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced FMHA benchmarking with full parameter control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 10_advanced_benchmark.py                          # Defaults
  python3 10_advanced_benchmark.py --warmup 10 --repeat 50  # More samples
  python3 10_advanced_benchmark.py --flush-cache             # Flush L2
        """,
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Allocate a scratch buffer between runs to flush GPU cache",
    )
    parser.add_argument(
        "--arch", default=detect_gpu_arch(), help="GPU architecture (auto-detected)"
    )
    parser.add_argument(
        "--lib", default=None, help="Path to prebuilt .so (JIT-builds if omitted)"
    )
    args = parser.parse_args()
    return args


PROBLEM_TABLE = [
    # (batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim, label)
    (1, 8, 8, 64, 64, 128, "tiny"),
    (2, 8, 8, 128, 128, 128, "small"),
    (2, 16, 16, 256, 256, 128, "medium"),
    (4, 16, 16, 512, 512, 128, "large"),
    (2, 32, 32, 1024, 1024, 128, "xlarge"),
    (1, 32, 8, 256, 256, 128, "GQA-4:1"),
]


def flush_gpu_cache():
    """Allocate and touch a large buffer to evict L2 cache lines."""
    scratch = np.random.randint(0, 255, size=32 * 1024 * 1024, dtype=np.uint8)
    _ = scratch.sum()


def run_benchmark(
    runner, prob: FmhaProblem, warmup: int, repeat: int, flush_cache: bool
) -> list:
    """Run warmup + repeat iterations and return list of times in ms."""
    Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float16)
    K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float16)
    V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float16)

    for _ in range(warmup):
        runner.run(Q, K, V, prob)

    times = []
    for _ in range(repeat):
        if flush_cache:
            flush_gpu_cache()
        result = runner.run(Q, K, V, prob)
        if result.success:
            times.append(result.time_ms)
    return times


def main():
    args = parse_args()

    print("=" * 70)
    print("Example 10: Advanced FMHA Benchmarking")
    print("=" * 70)

    print("\nBenchmark Configuration:")
    print(f"  Warmup:      {args.warmup} iterations")
    print(f"  Repeat:      {args.repeat} iterations")
    print(f"  Flush Cache: {args.flush_cache}")
    print(f"  Arch:        {args.arch}")
    print(f"  Problems:    {len(PROBLEM_TABLE)}")

    # Step 1: Load or JIT-build kernel
    print("\n" + "=" * 70)
    print("Step 1: Load / Build Kernel")
    print("=" * 70)

    print("  JIT building kernel...")
    config = FmhaKernelConfig(
        family="fwd",
        data_type="fp16",
        hdim_q=128,
        hdim_v=128,
        pipeline="qr_async",
        # Stage 0 (Q*K^T): seqlen_q x seqlen_k x hdim_q
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
        # Stage 1 (Attn*V): hdim_v x seqlen_k x alignment
        tile_n1=128,
        tile_k1=32,
        tile_k0max=128,
        # Wave config per stage
        wave_m0=4,
        wave_n0=1,
        wave_k0=1,
        wave_m1=4,
        wave_n1=1,
        wave_k1=1,
        # Warp tile per stage
        warp_m0=32,
        warp_n0=32,
        warp_k0=16,
        warp_m1=32,
        warp_n1=32,
        warp_k1=16,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config, verbose=True)
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
        return 1
    runner = setup.runner
    print(f"  JIT built: {setup.library_path} ({setup.build_time_s:.1f} s)")

    print(f"  Kernels: {runner.kernel_count}")

    # Step 2: Benchmark all problems
    print("\n" + "=" * 70)
    print("Step 2: Benchmark Results")
    print("=" * 70)

    header = (
        f"  {'Label':<10} {'Shape':^30} "
        f"{'Min':>8} {'Avg':>8} {'Max':>8} {'Med':>8} {'TFLOPS':>8}"
    )
    print(f"\n{header}")
    print("  " + "-" * 85)

    all_results = []
    np.random.seed(42)

    for batch, hq, hk, sq, sk, hdim, label in PROBLEM_TABLE:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=sq,
            seqlen_k=sk,
            hdim_q=hdim,
            hdim_v=hdim,
        )
        shape_str = f"B{batch}_Hq{hq}_Hk{hk}_S{sq}_D{hdim}"

        times = run_benchmark(runner, prob, args.warmup, args.repeat, args.flush_cache)

        if not times:
            print(
                f"  {label:<10} {shape_str:^30} {'FAIL':>8} {'---':>8} "
                f"{'---':>8} {'---':>8} {'---':>8}"
            )
            continue

        t_min = min(times)
        t_max = max(times)
        t_avg = sum(times) / len(times)
        t_med = float(np.median(times))

        tflops = prob.num_ops / (t_med * 1e-3) / 1e12 if t_med > 0 else 0

        print(
            f"  {label:<10} {shape_str:^30} "
            f"{t_min:>7.3f}ms {t_avg:>7.3f}ms {t_max:>7.3f}ms {t_med:>7.3f}ms "
            f"{tflops:>7.2f}"
        )

        all_results.append((label, shape_str, t_min, t_avg, t_max, t_med, tflops))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if all_results:
        best = max(all_results, key=lambda r: r[6])
        print(f"\n  Best TFLOPS:  {best[6]:.2f}  ({best[0]}: {best[1]})")
        avg_tflops = sum(r[6] for r in all_results) / len(all_results)
        print(f"  Avg TFLOPS:   {avg_tflops:.2f}")
        print(f"  Problems run: {len(all_results)}/{len(PROBLEM_TABLE)}")
    else:
        print("\n  No successful benchmarks")

    print(
        f"\n  Settings: warmup={args.warmup}, repeat={args.repeat}, "
        f"flush_cache={args.flush_cache}"
    )

    print("\n" + "=" * 70)
    print("BENCHMARK PARAMETERS REFERENCE")
    print("=" * 70)
    print("""
  --warmup N        Warmup iterations (results discarded)
                    Higher = more stable results, longer run
                    Default: 5

  --repeat N        Timed iterations
                    Higher = more accurate statistics
                    Default: 20

  --flush-cache     Flush GPU L2 cache between iterations
                    Use for memory-bandwidth measurements
                    Default: off

  --arch ARCH       GPU architecture (e.g. gfx950)
                    Auto-detected from rocminfo
""")
    print("=" * 70)

    runner.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
