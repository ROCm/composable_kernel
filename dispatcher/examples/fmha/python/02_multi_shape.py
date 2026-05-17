#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 02: Multi-Shape FMHA

Runs FMHA forward with a single kernel across multiple problem shapes
(varying batch, sequence length, and head count).

Usage:
    python3 02_multi_shape.py
    python3 02_multi_shape.py --help
    python3 02_multi_shape.py --dtype bf16
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
        description="Multi-Shape FMHA Example - runs multiple shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 02_multi_shape.py                    # Default FP16
  python3 02_multi_shape.py --dtype bf16       # BF16 FMHA
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 02: Multi-Shape FMHA")
    print("=" * 70)

    # Step 1: Setup dispatcher
    print("\nStep 1: Setup Dispatcher")

    # FmhaKernelSpec fields:
    #   name      -- human-readable kernel identifier
    #   hdim      -- head dimension (hdim_q = hdim_v)
    #   pipeline  -- "qr_async" (async prefetch) or "qr" (synchronous)
    #   tile_m0   -- Stage 0 tile along seqlen_q  (Q*K^T M dimension)
    #   tile_n0   -- Stage 0 tile along seqlen_k  (Q*K^T N dimension)
    #   tile_k0   -- Stage 0 tile along hdim_q    (Q*K^T K dimension)
    spec = FmhaKernelSpec(name="multi_shape", hdim=128, pipeline="qr_async")
    config = spec_to_config(spec, dtype=args.dtype, arch=args.arch)

    setup = setup_fmha_dispatcher(config, verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    runner = setup.runner
    print(f"  Library: {setup.library_path}")
    print(f"  Build:   {setup.build_time_s:.1f} s")

    # Step 2: Run batch of different shapes
    print("\nStep 2: Run Shapes")

    shapes = [
        # (batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim)
        (1, 4, 4, 64, 64, 128),
        (2, 8, 8, 128, 128, 128),
        (4, 8, 8, 128, 128, 128),
        (1, 16, 16, 256, 256, 128),
        (2, 8, 8, 256, 256, 128),
        (1, 8, 8, 512, 512, 128),
        (2, 4, 4, 512, 512, 128),
        (1, 8, 8, 1024, 1024, 128),
    ]

    print(f"\n  {'#':<3} {'Shape':<36} {'Time(ms)':>10} {'TFLOPS':>10} {'Status':>8}")
    print("  " + "-" * 70)

    total_ops = 0
    total_time = 0.0

    for idx, (b, hq, hk, sq, sk, d) in enumerate(shapes, 1):
        prob = FmhaProblem(
            batch=b,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=sq,
            seqlen_k=sk,
            hdim_q=d,
            hdim_v=d,
        )
        shape_str = f"B{b}_Hq{hq}_Hk{hk}_S{sq}x{sk}_D{d}"

        np.random.seed(42 + idx)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)

        result = runner.run(Q, K, V, prob)

        if result.success:
            total_ops += prob.num_ops
            total_time += result.time_ms
            print(
                f"  {idx:<3} {shape_str:<36} {result.time_ms:>10.4f} {result.tflops:>10.2f} {'OK':>8}"
            )
        else:
            print(f"  {idx:<3} {shape_str:<36} {'N/A':>10} {'N/A':>10} {'Error':>8}")

    print("  " + "-" * 70)

    if total_time > 0:
        avg_tflops = (total_ops / 1e12) / (total_time / 1000)
        print(f"\n  Total: {total_time:.2f} ms, Average: {avg_tflops:.2f} TFLOPS")

    runner.cleanup()

    print("\n" + "=" * 70)
    print("Multi-Shape FMHA complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
