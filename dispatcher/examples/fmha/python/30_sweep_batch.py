#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 30: Sweep Batch Size

Demonstrates how FMHA performance scales with batch size.
FMHA compute scales linearly with batch, so time should increase
linearly while TFLOPS remains roughly constant once the GPU is saturated.

Fixed: seqlen=128, nhead=8, hdim=128
Sweep: batch in [1, 2, 4, 8, 16, 32]

Usage:
    python3 30_sweep_batch.py
    python3 30_sweep_batch.py --arch gfx942
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)

SEQLEN = 128
NHEAD = 8
HDIM = 128
BATCHES = [1, 2, 4, 8, 16, 32]


def main():
    parser = argparse.ArgumentParser(description="Sweep Batch Size FMHA")
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 30: Sweep Batch Size")
    print("=" * 70)

    print(f"\n  Fixed: seqlen={SEQLEN}, nhead={NHEAD}, hdim={HDIM}")
    print(f"  Sweep: batch in {BATCHES}")
    print(f"  Arch:  {args.arch}")

    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    # Step 1: JIT-compile FMHA kernel
    print("\nStep 1: JIT-Compile FMHA Kernel")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=HDIM,
        hdim_v=HDIM,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
        return 1
    runner = setup.runner
    print(f"  JIT build: {setup.build_time_s:.1f}s")

    # Step 2: Sweep
    print("\nStep 2: Batch Size Sweep")

    hdr = f"  {'Batch':>8} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'MaxErr':>10} | {'Status':<6}"
    print(f"\n{hdr}")
    print("  " + "-" * 60)

    np.random.seed(42)
    results = []

    for batch in BATCHES:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=NHEAD,
            nhead_k=NHEAD,
            seqlen_q=SEQLEN,
            seqlen_k=SEQLEN,
            hdim_q=HDIM,
            hdim_v=HDIM,
        )

        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)

        O_ref = cpu_attention_fwd(
            Q.astype(np.float32),
            K.astype(np.float32),
            V.astype(np.float32),
            prob.scale,
        )

        res = runner.run(Q, K, V, prob)
        if not res.success:
            print(
                f"  {batch:>8} | {'---':>10} | {'---':>10} | {'---':>10} | {'FAIL':<6}"
            )
            results.append((batch, False, 0.0, 0.0, 0.0))
            continue

        max_err = float(np.abs(res.output.astype(np.float32) - O_ref).max())
        ok, _, _ = validator.check(res.output, O_ref)
        tag = "PASS" if ok else "FAIL"

        print(
            f"  {batch:>8} | {res.time_ms:>10.4f} | {res.tflops:>10.2f} | {max_err:>10.2e} | {tag:<6}"
        )
        results.append((batch, ok, res.time_ms, res.tflops, max_err))

    # Step 3: Linearity analysis
    print("\nStep 3: Linear Scaling Analysis")
    valid = [(b, t, tf) for b, ok, t, tf, _ in results if ok and t > 0]
    if len(valid) >= 2:
        b0, t0, tf0 = valid[0]
        b_last, t_last, tf_last = valid[-1]
        batch_ratio = b_last / b0
        time_ratio = t_last / t0
        linearity = time_ratio / batch_ratio

        print(
            f"  Batch {b0} -> {b_last}: {batch_ratio:.0f}x batch, {time_ratio:.1f}x time"
        )
        print(f"  Linearity factor: {linearity:.2f} (1.0 = perfect linear scaling)")
        print(f"  TFLOPS range: {tf0:.2f} - {tf_last:.2f}")

    # Summary
    passed = sum(1 for _, ok, *_ in results if ok)
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{len(results)} passed")
    print(f"  Fixed:   S={SEQLEN} H={NHEAD} D={HDIM}")
    print(f"  Sweep:   batch={BATCHES}")
    print(f"  Status:  {'PASS' if passed == len(results) else 'FAIL'}")
    print("=" * 70)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
