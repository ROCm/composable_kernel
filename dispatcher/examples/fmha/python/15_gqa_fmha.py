#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 15: Grouped-Query Attention (GQA / MQA)

Demonstrates GQA with various nhead_q:nhead_k ratios:
- 1:1  (MHA)  -- Standard multi-head attention
- 2:1         -- Each KV head serves 2 query heads
- 4:1         -- Each KV head serves 4 query heads
- 8:1         -- Each KV head serves 8 query heads
- 16:1 (MQA)  -- Single KV head serves all query heads

GQA reduces KV cache memory and bandwidth while maintaining quality.
CPU reference uses np.repeat to expand K,V heads to match Q heads.

Usage:
    python3 15_gqa_fmha.py
    python3 15_gqa_fmha.py --nhead-q 32
    python3 15_gqa_fmha.py --seqlen 256
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    FmhaKernelConfig,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def main():
    parser = argparse.ArgumentParser(description="GQA / MQA Attention")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead-q", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 15: Grouped-Query Attention (GQA / MQA)")
    print("=" * 70)

    hq = args.nhead_q

    gqa_ratios = []
    for ratio in [1, 2, 4, 8, 16]:
        if hq % ratio == 0:
            gqa_ratios.append(ratio)

    print(f"\n  nhead_q:  {hq}")
    print(f"  Ratios:   {', '.join(f'{r}:1' for r in gqa_ratios)}")
    print(f"  Problem:  B={args.batch} S={args.seqlen} D={args.hdim}")

    # --- Try GPU runner ---
    runner = None
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=args.hdim,
        hdim_v=args.hdim,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if setup.success:
        runner = setup.runner
        print(f"  GPU:      Loaded (JIT build: {setup.build_time_s:.1f}s)")
    else:
        print(f"  GPU:      Not available ({setup.error})")

    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    print(
        f"\n  {'#':<3} {'Ratio':<8} {'nhead_q':>8} {'nhead_k':>8} {'KV_save':>8} "
        f"{'Time(ms)':>10} {'TFLOPS':>10} {'MaxErr':>10} {'Status':>8}"
    )
    print("  " + "-" * 82)

    results = []
    for i, ratio in enumerate(gqa_ratios, 1):
        hk = hq // ratio
        kv_saving = (1.0 - hk / hq) * 100

        prob = FmhaProblem(
            batch=args.batch,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=args.seqlen,
            seqlen_k=args.seqlen,
            hdim_q=args.hdim,
            hdim_v=args.hdim,
        )

        np.random.seed(42 + i)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)

        O_ref = cpu_attention_fwd(
            Q.astype(np.float32),
            K.astype(np.float32),
            V.astype(np.float32),
            prob.scale,
        )

        # GPU attempt
        time_str = "---"
        tflops_str = "---"
        gpu_out = None
        if runner is not None:
            res = runner.run(Q, K, V, prob)
            if res.success:
                gpu_out = res.output
                time_str = f"{res.time_ms:.4f}"
                tflops_str = f"{res.tflops:.2f}"

        if gpu_out is not None:
            ok, max_abs, _ = validator.check(gpu_out, O_ref)
            tag = "PASS" if ok else "FAIL"
            err_str = f"{max_abs:.2e}"
        else:
            ok = True
            tag = "DEMO"
            err_str = "---"
            max_abs = 0.0

        label = f"{ratio}:1"
        if ratio == 1:
            label += " MHA"
        elif hk == 1:
            label += " MQA"

        print(
            f"  {i:<3} {label:<8} {hq:>8} {hk:>8} {kv_saving:>7.0f}% "
            f"{time_str:>10} {tflops_str:>10} {err_str:>10} {tag:>8}"
        )
        results.append((ratio, hk, ok, max_abs))

    # --- Memory analysis ---
    print("\n--- KV Cache Memory Analysis ---")
    base_kv_size = args.batch * hq * args.seqlen * args.hdim * 2 * 2  # K+V, fp16

    print(f"\n  {'Ratio':<8} {'nhead_k':>8} {'KV Size':>12} {'Savings':>10}")
    print("  " + "-" * 42)

    for ratio in gqa_ratios:
        hk = hq // ratio
        kv_size = args.batch * hk * args.seqlen * args.hdim * 2 * 2
        saving = (1.0 - kv_size / base_kv_size) * 100
        size_str = (
            f"{kv_size / 1024:.1f} KB"
            if kv_size < 1024 * 1024
            else f"{kv_size / (1024 * 1024):.2f} MB"
        )
        print(f"  {ratio}:1{'':<4} {hq // ratio:>8} {size_str:>12} {saving:>9.0f}%")

    # --- GQA correctness: verify np.repeat equivalence ---
    print("\n--- GQA Equivalence Check ---")
    prob_gqa = FmhaProblem(
        batch=1,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=64,
        seqlen_k=64,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )
    np.random.seed(99)
    Q_g = (np.random.randn(*prob_gqa.q_shape()) * 0.1).astype(np.float32)
    K_g = (np.random.randn(*prob_gqa.k_shape()) * 0.1).astype(np.float32)
    V_g = (np.random.randn(*prob_gqa.v_shape()) * 0.1).astype(np.float32)

    O_gqa = cpu_attention_fwd(Q_g, K_g, V_g, prob_gqa.scale)

    K_exp = np.repeat(K_g, 4, axis=1)
    V_exp = np.repeat(V_g, 4, axis=1)
    prob_mha = FmhaProblem(
        batch=1,
        nhead_q=8,
        nhead_k=8,
        seqlen_q=64,
        seqlen_k=64,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )
    O_mha = cpu_attention_fwd(Q_g, K_exp, V_exp, prob_mha.scale)

    equiv_err = float(np.abs(O_gqa - O_mha).max())
    print(f"  GQA(4:1) vs MHA(expanded): max_err = {equiv_err:.2e}")
    print("  cpu_attention_fwd handles GQA internally via np.repeat")

    # --- Summary ---
    all_ok = all(ok for _, _, ok, _ in results)
    print("\n" + "=" * 70)
    print(f"  GQA ratios tested: {len(gqa_ratios)}")
    print("  MHA (1:1):  All heads have unique KV (baseline)")
    print("  GQA (N:1):  N query heads share one KV head")
    print("  MQA (H:1):  All query heads share single KV head (max saving)")
    print("  GPU:        Prebuilt kernel supports GQA via nhead_q != nhead_k")
    print(f"  Status:     {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
