#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 04: FMHA Validation

Validates GPU FMHA against CPU reference across multiple test cases
including standard shapes, GQA ratios, and edge cases.

Usage:
    python3 04_validation.py
    python3 04_validation.py --help
    python3 04_validation.py --dtype bf16
    python3 04_validation.py --rtol 1e-2
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelSpec,
    FmhaProblem,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
    spec_to_config,
)


def main():
    parser = argparse.ArgumentParser(
        description="FMHA Validation Example - validates GPU results against CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 04_validation.py                    # Default FP16 validation
  python3 04_validation.py --dtype bf16       # BF16 validation
  python3 04_validation.py --rtol 1e-2        # Relaxed tolerance
        """,
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-2, help="Relative tolerance (default: 1e-2)"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance (default: 1e-2)"
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 04: FMHA Validation")
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
    spec = FmhaKernelSpec(name="validation", hdim=128, pipeline="qr_async")
    config = spec_to_config(spec, dtype=args.dtype, arch=args.arch)

    setup = setup_fmha_dispatcher(config, verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    runner = setup.runner
    print(f"  Library: {setup.library_path}")
    print(f"  Build:   {setup.build_time_s:.1f} s")

    # Step 2: Run validation tests
    print("\nStep 2: Validation Tests")

    validator = FmhaValidator(rtol=args.rtol, atol=args.atol)

    # (name, batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim)
    test_cases = [
        ("Small", 1, 4, 4, 64, 64, 128),
        ("Medium", 2, 8, 8, 128, 128, 128),
        ("Large", 1, 8, 8, 256, 256, 128),
        ("Long-seq", 1, 4, 4, 512, 512, 128),
        ("Non-square", 2, 4, 4, 64, 256, 128),
        ("GQA-2:1", 2, 8, 4, 128, 128, 128),
        ("GQA-4:1", 1, 16, 4, 128, 128, 128),
        ("GQA-8:1", 1, 16, 2, 64, 64, 128),
        ("Single-query", 1, 4, 4, 1, 128, 128),
        ("Batched", 4, 8, 8, 128, 128, 128),
    ]

    passed = 0
    failed = 0

    print(f"\n  {'#':<3} {'Test':<14} {'Shape':<30} {'MaxErr':>10} {'Status':>8}")
    print("  " + "-" * 70)

    for idx, (name, b, hq, hk, sq, sk, d) in enumerate(test_cases, 1):
        prob = FmhaProblem(
            batch=b,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=sq,
            seqlen_k=sk,
            hdim_q=d,
            hdim_v=d,
        )
        shape_str = f"B{b}_Hq{hq}_Hk{hk}_S{sq}x{sk}"

        np.random.seed(42 + idx)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)

        result = runner.run(Q, K, V, prob)
        if not result.success:
            print(
                f"  {idx:<3} {name:<14} {shape_str:<30} {'GPU Err':>10} {'FAILED':>8}"
            )
            failed += 1
            continue

        O_ref = cpu_attention_fwd(
            Q.astype(np.float32),
            K.astype(np.float32),
            V.astype(np.float32),
            prob.scale,
        )
        is_valid, max_abs, _ = validator.check(result.output, O_ref)

        if is_valid:
            print(
                f"  {idx:<3} {name:<14} {shape_str:<30} {max_abs:>10.2e} {'PASSED':>8}"
            )
            passed += 1
        else:
            print(
                f"  {idx:<3} {name:<14} {shape_str:<30} {max_abs:>10.2e} {'FAILED':>8}"
            )
            failed += 1

    runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"  Results:  {passed}/{total} passed")
    print(f"  Settings: dtype={args.dtype}, rtol={args.rtol}, atol={args.atol}")
    print(f"  Status:   {'PASS' if failed == 0 else 'FAIL'}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
