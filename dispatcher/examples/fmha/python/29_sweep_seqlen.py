#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 29: Sweep Sequence Length

Demonstrates how FMHA performance scales with sequence length.
FMHA has O(n^2) compute in seqlen (Q*K^T), so TFLOPS should increase
with longer sequences as the GPU becomes better utilized.

Fixed: batch=2, nhead=8, hdim=128
Sweep: seqlen in [32, 64, 128, 256, 512, 1024, 2048]

Usage:
    python3 29_sweep_seqlen.py
    python3 29_sweep_seqlen.py --arch gfx942
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

BATCH = 2
NHEAD = 8
HDIM = 128
SEQLENS = [32, 64, 128, 256, 512, 1024, 2048]


def main():
    parser = argparse.ArgumentParser(description="Sweep Sequence Length FMHA")
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 29: Sweep Sequence Length")
    print("=" * 70)

    print(f"\n  Fixed: batch={BATCH}, nhead={NHEAD}, hdim={HDIM}")
    print(f"  Sweep: seqlen in {SEQLENS}")
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
    print("\nStep 2: Sequence Length Sweep")

    hdr = f"  {'SeqLen':>8} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'MaxErr':>10} | {'Status':<6}"
    print(f"\n{hdr}")
    print("  " + "-" * 60)

    np.random.seed(42)
    results = []

    for seqlen in SEQLENS:
        prob = FmhaProblem(
            batch=BATCH,
            nhead_q=NHEAD,
            nhead_k=NHEAD,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
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
                f"  {seqlen:>8} | {'---':>10} | {'---':>10} | {'---':>10} | {'FAIL':<6}"
            )
            results.append((seqlen, False, 0.0, 0.0, 0.0))
            continue

        max_err = float(np.abs(res.output.astype(np.float32) - O_ref).max())
        ok, _, _ = validator.check(res.output, O_ref)
        tag = "PASS" if ok else "FAIL"

        print(
            f"  {seqlen:>8} | {res.time_ms:>10.4f} | {res.tflops:>10.2f} | {max_err:>10.2e} | {tag:<6}"
        )
        results.append((seqlen, ok, res.time_ms, res.tflops, max_err))

    # Step 3: Scaling analysis
    print("\nStep 3: Scaling Analysis")
    valid = [(s, t, tf) for s, ok, t, tf, _ in results if ok and tf > 0]
    if len(valid) >= 2:
        s0, _, tf0 = valid[0]
        s_last, _, tf_last = valid[-1]
        print(f"  Shortest (seqlen={s0}):  {tf0:.2f} TFLOPS")
        print(f"  Longest  (seqlen={s_last}): {tf_last:.2f} TFLOPS")
        print(f"  Speedup: {tf_last / tf0:.1f}x TFLOPS improvement")
        print("  Note: Longer sequences expose more parallelism to the GPU")

    # Summary
    passed = sum(1 for _, ok, *_ in results if ok)
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{len(results)} passed")
    print(f"  Fixed:   B={BATCH} H={NHEAD} D={HDIM}")
    print(f"  Sweep:   seqlen={SEQLENS}")
    print(f"  Status:  {'PASS' if passed == len(results) else 'FAIL'}")
    print("=" * 70)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
