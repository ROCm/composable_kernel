#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 31: Sweep Number of Heads (MHA + GQA)

Demonstrates FMHA performance across different head counts, including
Grouped Query Attention (GQA) where nhead_q > nhead_k.

Part 1 - MHA sweep: nhead_q == nhead_k
Part 2 - GQA variants: nhead_q != nhead_k (multiple Q heads share K/V)

Fixed: batch=2, seqlen=128, hdim=128

Usage:
    python3 31_sweep_nhead.py
    python3 31_sweep_nhead.py --arch gfx942
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
SEQLEN = 128
HDIM = 128

MHA_NHEADS = [1, 2, 4, 8, 16, 32]
GQA_CONFIGS = [
    (8, 1, "GQA 8:1"),
    (16, 4, "GQA 4:1"),
    (32, 8, "GQA 4:1"),
]


def run_sweep(runner, validator, configs, label):
    """Run a sweep over (nhead_q, nhead_k) configurations."""
    hdr = f"  {'nhead_q':>8} | {'nhead_k':>8} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'MaxErr':>10} | {'Status':<6}"
    print(f"\n{hdr}")
    print("  " + "-" * 70)

    np.random.seed(42)
    results = []

    for nhead_q, nhead_k in configs:
        prob = FmhaProblem(
            batch=BATCH,
            nhead_q=nhead_q,
            nhead_k=nhead_k,
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
                f"  {nhead_q:>8} | {nhead_k:>8} | {'---':>10} | {'---':>10} | {'---':>10} | {'FAIL':<6}"
            )
            results.append((nhead_q, nhead_k, False, 0.0, 0.0, 0.0))
            continue

        max_err = float(np.abs(res.output.astype(np.float32) - O_ref).max())
        ok, _, _ = validator.check(res.output, O_ref)
        tag = "PASS" if ok else "FAIL"

        print(
            f"  {nhead_q:>8} | {nhead_k:>8} | {res.time_ms:>10.4f} | {res.tflops:>10.2f} | {max_err:>10.2e} | {tag:<6}"
        )
        results.append((nhead_q, nhead_k, ok, res.time_ms, res.tflops, max_err))

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep Number of Heads FMHA")
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 31: Sweep Number of Heads (MHA + GQA)")
    print("=" * 70)

    print(f"\n  Fixed: batch={BATCH}, seqlen={SEQLEN}, hdim={HDIM}")
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

    # Step 2: MHA sweep (nhead_q == nhead_k)
    print("\nStep 2: MHA Sweep (nhead_q == nhead_k)")
    mha_configs = [(n, n) for n in MHA_NHEADS]
    mha_results = run_sweep(runner, validator, mha_configs, "MHA")

    # Step 3: GQA sweep (nhead_q > nhead_k)
    print("\nStep 3: GQA Sweep (nhead_q > nhead_k)")
    print("  GQA: multiple Q heads share fewer K/V heads")
    gqa_configs = [(nq, nk) for nq, nk, _ in GQA_CONFIGS]
    gqa_results = run_sweep(runner, validator, gqa_configs, "GQA")

    # Step 4: Comparison
    print("\nStep 4: MHA vs GQA Comparison")
    all_results = mha_results + gqa_results
    valid_mha = [(nq, nk, tf) for nq, nk, ok, _, tf, _ in mha_results if ok and tf > 0]
    valid_gqa = [(nq, nk, tf) for nq, nk, ok, _, tf, _ in gqa_results if ok and tf > 0]

    if valid_mha:
        best_mha = max(valid_mha, key=lambda x: x[2])
        print(f"  Best MHA: nhead={best_mha[0]}, {best_mha[2]:.2f} TFLOPS")
    if valid_gqa:
        best_gqa = max(valid_gqa, key=lambda x: x[2])
        print(
            f"  Best GQA: nhead_q={best_gqa[0]}, nhead_k={best_gqa[1]}, {best_gqa[2]:.2f} TFLOPS"
        )
        print(f"  GQA saves K/V memory: {best_gqa[0]}:{best_gqa[1]} ratio")

    # Summary
    passed = sum(1 for *_, ok, _, _, _ in all_results if ok)
    total = len(all_results)
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} passed")
    print(f"  Fixed:   B={BATCH} S={SEQLEN} D={HDIM}")
    print(f"  MHA:     nhead={MHA_NHEADS}")
    print(f"  GQA:     {[(nq, nk) for nq, nk, _ in GQA_CONFIGS]}")
    print(f"  Status:  {'PASS' if passed == total else 'FAIL'}")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
