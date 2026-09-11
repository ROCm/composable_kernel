#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 36: Backward Pass Benchmark

Benchmarks the FMHA backward pass across problem sizes. The backward
pass is approximately 4x the forward FLOPS because it computes dQ, dK,
and dV through two matrix multiplications each (plus the dot_do_o stage).

Backward FLOPS estimate:
  FWD:  2 * B * H * Sq * Sk * (Dq + Dv)
  BWD:  ~4 * FWD_FLOPS
      = 2 * B * H * Sq * Sk * Dq  (dP = dO @ V^T, part of dS computation)
      + 2 * B * H * Sq * Sk * Dq  (dQ = dS @ K)
      + 2 * B * H * Sq * Sk * Dq  (dK = dS^T @ Q)
      + 2 * B * H * Sq * Sk * Dv  (dV = P^T @ dO)

When GPU JIT is unavailable, benchmarks CPU reference instead.

Usage:
    python3 36_bwd_benchmark_fmha.py
    python3 36_bwd_benchmark_fmha.py --repeat 5
    python3 36_bwd_benchmark_fmha.py --arch gfx942
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    setup_fmha_dispatcher,
    detect_gpu_arch,
    cpu_attention_fwd_with_intermediates,
    cpu_attention_bwd,
)


cpu_fwd_with_intermediates = cpu_attention_fwd_with_intermediates


def bwd_flops(prob: FmhaProblem) -> int:
    """Estimate backward FLOPS (~4x forward)."""
    B, Hq, Sq, Sk = prob.batch, prob.nhead_q, prob.seqlen_q, prob.seqlen_k
    Dq, Dv = prob.hdim_q, prob.hdim_v
    fwd = 2 * B * Hq * Sq * Sk * (Dq + Dv)
    return 4 * fwd


def main():
    parser = argparse.ArgumentParser(description="Backward Pass Benchmark")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--repeat", type=int, default=3, help="Benchmark iterations")
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 36: Backward Pass Benchmark")
    print("=" * 70)

    print(f"\n  Arch:    {args.arch}")
    print(f"  nhead:   {args.nhead}")
    print(f"  hdim:    {args.hdim}")
    print(f"  Repeat:  {args.repeat}")

    # --- JIT compile a basic fp16 h128 fwd kernel ---
    print("\n--- JIT Compilation ---")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=args.hdim,
        hdim_v=args.hdim,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if setup.success:
        print(f"  Fwd kernel compiled: {setup.build_time_s:.1f}s")
        print("  Backward GPU kernel: Not available (bwd JIT tile structure issue)")
        print("  Benchmarking CPU backward reference instead")
    else:
        print(f"  JIT build: {setup.error}")
        print("  Benchmarking CPU backward reference")

    # --- Benchmark configs ---
    bench_configs = [
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
        (2, 64),
        (2, 128),
        (2, 256),
        (2, 512),
        (4, 64),
        (4, 128),
        (4, 256),
        (8, 64),
        (8, 128),
    ]

    # --- FLOPS estimate table ---
    print("\n--- FLOPS Estimates (BWD ~4x FWD) ---")
    print(
        f"\n  {'Batch':>5} {'SeqLen':>7} | {'FWD FLOPS':>14} {'BWD FLOPS':>14} {'Ratio':>6}"
    )
    print("  " + "-" * 52)

    for batch, seqlen in [(1, 128), (1, 1024), (4, 256), (8, 128)]:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=args.nhead,
            nhead_k=args.nhead,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            hdim_q=args.hdim,
            hdim_v=args.hdim,
        )
        fwd_ops = prob.num_ops
        bwd_ops = bwd_flops(prob)
        print(
            f"  {batch:>5} {seqlen:>7} | {fwd_ops:>14,} {bwd_ops:>14,} {bwd_ops / fwd_ops:>5.1f}x"
        )

    # --- CPU backward benchmark ---
    print("\n--- CPU Backward Benchmark ---")
    print(
        f"\n  {'Batch':>5} {'SeqLen':>7} | {'Time(ms)':>10} {'TFLOPS':>10}"
        f" | {'dQ range':>22} {'Finite':>6}"
    )
    print("  " + "-" * 76)

    all_tflops = []

    for batch, seqlen in bench_configs:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=args.nhead,
            nhead_k=args.nhead,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            hdim_q=args.hdim,
            hdim_v=args.hdim,
        )
        ops = bwd_flops(prob)

        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
        dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

        out, P = cpu_fwd_with_intermediates(Q, K, V, prob.scale)

        times = []
        dQ = dK = dV = None
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            dQ, dK, dV = cpu_attention_bwd(Q, K, V, out, dO, P, prob.scale)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        avg_ms = sum(times) / len(times)
        tflops = ops / (avg_ms * 1e-3) / 1e12 if avg_ms > 0 else 0.0
        all_tflops.append(tflops)

        is_finite = bool(np.all(np.isfinite(dQ)))
        dq_range = f"[{dQ.min():.4e}, {dQ.max():.4e}]"

        print(
            f"  {batch:>5} {seqlen:>7} | {avg_ms:>10.4f} {tflops:>10.4f}"
            f" | {dq_range:>22} {'OK' if is_finite else 'NaN!':>6}"
        )

    # --- Scaling analysis ---
    print("\n--- Scaling Analysis ---")
    print("  Backward time should scale as O(B * H * Sq * Sk * D).")
    print("  Doubling seqlen -> ~4x time (quadratic in sequence length).\n")

    ref_configs = [(1, 128), (1, 256), (1, 512)]
    ref_times = {}
    for batch, seqlen in ref_configs:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=args.nhead,
            nhead_k=args.nhead,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            hdim_q=args.hdim,
            hdim_v=args.hdim,
        )
        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
        dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)
        out, P = cpu_fwd_with_intermediates(Q, K, V, prob.scale)

        t0 = time.perf_counter()
        cpu_attention_bwd(Q, K, V, out, dO, P, prob.scale)
        ref_times[seqlen] = (time.perf_counter() - t0) * 1000.0

    if 128 in ref_times and ref_times[128] > 0:
        base = ref_times[128]
        print(f"  {'SeqLen':>7} {'Time(ms)':>10} {'vs S=128':>10}")
        print("  " + "-" * 30)
        for sl in sorted(ref_times):
            ratio = ref_times[sl] / base
            print(f"  {sl:>7} {ref_times[sl]:>10.4f} {ratio:>9.1f}x")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  Configs tested: {len(bench_configs)}")
    print("  BWD FLOPS:      ~4x forward FLOPS")
    if all_tflops:
        print(f"  CPU avg:        {sum(all_tflops) / len(all_tflops):.4f} TFLOPS")
        print(f"  CPU peak:       {max(all_tflops):.4f} TFLOPS")
    print("  GPU:            Requires bwd-family JIT kernel")
    print("  Status:         DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
