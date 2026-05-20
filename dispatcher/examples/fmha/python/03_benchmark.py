#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 03: FMHA Benchmark

Performance benchmarking with warmup and repeated iterations across
multiple (batch, sequence length) configurations.

Usage:
    python3 03_benchmark.py
    python3 03_benchmark.py --help
    python3 03_benchmark.py --warmup 5 --repeat 20
    python3 03_benchmark.py --arch gfx942
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelSpec,
    FmhaProblem,
    detect_gpu_arch,
    setup_fmha_dispatcher,
    spec_to_config,
)


def main():
    parser = argparse.ArgumentParser(
        description="FMHA Benchmark Example - performance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 03_benchmark.py                     # Default benchmark suite
  python3 03_benchmark.py --warmup 5          # More warmup iterations
  python3 03_benchmark.py --repeat 20         # More benchmark iterations
        """,
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Benchmark iterations (default: 10)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 03: FMHA Benchmark")
    print("=" * 70)

    # Step 1: Setup dispatcher with compute-optimized config
    print("\nStep 1: Setup Dispatcher")

    # FmhaKernelSpec fields:
    #   name      -- human-readable kernel identifier
    #   hdim      -- head dimension (hdim_q = hdim_v)
    #   pipeline  -- "qr_async" (async prefetch) or "qr" (synchronous)
    #   tile_m0   -- Stage 0 tile along seqlen_q  (Q*K^T M dimension)
    #   tile_n0   -- Stage 0 tile along seqlen_k  (Q*K^T N dimension)
    #   tile_k0   -- Stage 0 tile along hdim_q    (Q*K^T K dimension)
    spec = FmhaKernelSpec(name="benchmark", hdim=128, pipeline="qr_async")
    config = spec_to_config(spec, dtype="fp16", arch=args.arch)

    setup = setup_fmha_dispatcher(config, verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    runner = setup.runner
    print(f"  Library: {setup.library_path}")
    print(f"  Build:   {setup.build_time_s:.1f} s")

    # Step 2: Benchmark
    print("\nStep 2: Benchmark")

    bench_configs = [
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (2, 128),
        (2, 256),
        (2, 512),
        (2, 1024),
        (4, 128),
        (4, 256),
        (4, 512),
        (8, 128),
        (8, 256),
    ]

    print(f"  Warmup: {args.warmup}, Repeat: {args.repeat}\n")

    print(
        f"  {'Batch':>5} {'SeqLen':>7} | {'Min(ms)':>10} {'Avg(ms)':>10} {'Max(ms)':>10} | {'TFLOPS':>10}"
    )
    print("  " + "-" * 62)

    all_tflops = []

    for batch, seqlen in bench_configs:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=8,
            nhead_k=8,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            hdim_q=128,
            hdim_v=128,
        )

        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)

        for _ in range(args.warmup):
            runner.run(Q, K, V, prob)

        times = []
        for _ in range(args.repeat):
            result = runner.run(Q, K, V, prob)
            if result.success:
                times.append(result.time_ms)

        if times:
            min_time = min(times)
            avg_time = sum(times) / len(times)
            max_time = max(times)
            tflops = prob.num_ops / (avg_time * 1e-3) / 1e12
            all_tflops.append(tflops)
            print(
                f"  {batch:>5} {seqlen:>7} | {min_time:>10.4f} {avg_time:>10.4f} {max_time:>10.4f} | {tflops:>10.2f}"
            )
        else:
            print(
                f"  {batch:>5} {seqlen:>7} | {'---':>10} {'---':>10} {'---':>10} | {'FAIL':>10}"
            )

    runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if all_tflops:
        print(f"  Average: {sum(all_tflops) / len(all_tflops):.2f} TFLOPS")
        print(f"  Peak:    {max(all_tflops):.2f} TFLOPS")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
