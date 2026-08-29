#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 34: Backward Pass with GQA (Grouped-Query Attention)

Demonstrates the FMHA backward pass when nhead_q != nhead_k.
GQA groups multiple query heads per KV head. The backward pass
must account for this by:
  - Expanding K/V heads via np.repeat for dQ computation
  - Summing dK/dV over query head groups back to KV head count

Tested GQA ratios: 1:1 (MHA), 2:1, 4:1, 8:1

CPU backward reference:
  K_exp = repeat(K, ratio)          # [B, Hq, Sk, D]
  V_exp = repeat(V, ratio)          # [B, Hq, Sk, Dv]
  dQ = scale * (P * (dO@V_exp^T - D)) @ K_exp
  dK_exp = scale * (P * (dO@V_exp^T - D))^T @ Q
  dV_exp = P^T @ dO
  dK = sum_over_groups(dK_exp)      # [B, Hk, Sk, D]
  dV = sum_over_groups(dV_exp)      # [B, Hk, Sk, Dv]

Usage:
    python3 34_bwd_gqa_fmha.py
    python3 34_bwd_gqa_fmha.py --nhead-q 32
    python3 34_bwd_gqa_fmha.py --arch gfx942
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


def cpu_fwd_with_intermediates(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> tuple:
    """Forward pass returning out, P, LSE (handles GQA via repeat)."""
    nhead_q, nhead_k = Q.shape[1], K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum
    out = np.matmul(P, V)
    lse = (np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)).astype(np.float32)
    return out, P, lse


def cpu_bwd_gqa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
    nhead_q: int,
    nhead_k: int,
) -> tuple:
    """CPU backward with GQA head grouping.

    P is already computed on expanded heads [B, Hq, Sq, Sk].
    K, V are original (unexpanded) [B, Hk, Sk, D].

    Returns: (dQ, dK, dV) where dK/dV have shape [B, Hk, Sk, ...]
    """
    ratio = nhead_q // nhead_k
    K_exp = np.repeat(K, ratio, axis=1)
    V_exp = np.repeat(V, ratio, axis=1)

    D = (dO * out).sum(axis=-1, keepdims=True)
    dP = np.matmul(dO, V_exp.transpose(0, 1, 3, 2))
    dS = P * (dP - D)

    dQ = np.matmul(dS, K_exp) * scale

    dK_exp = np.matmul(dS.transpose(0, 1, 3, 2), Q) * scale
    dV_exp = np.matmul(P.transpose(0, 1, 3, 2), dO)

    B = Q.shape[0]
    Sk, Dq = K.shape[2], K.shape[3]
    Dv = V.shape[3]

    dK = dK_exp.reshape(B, nhead_k, ratio, Sk, Dq).sum(axis=2)
    dV = dV_exp.reshape(B, nhead_k, ratio, Sk, Dv).sum(axis=2)

    return dQ, dK, dV


def main():
    parser = argparse.ArgumentParser(description="Backward Pass with GQA")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead-q", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 34: Backward Pass with GQA")
    print("=" * 70)

    hq = args.nhead_q

    gqa_ratios = []
    for ratio in [1, 2, 4, 8]:
        if hq % ratio == 0 and hq // ratio >= 1:
            gqa_ratios.append(ratio)

    print(f"\n  nhead_q: {hq}")
    print(f"  Ratios:  {', '.join(f'{r}:1' for r in gqa_ratios)}")
    print(f"  Problem: B={args.batch} S={args.seqlen} D={args.hdim}")

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
        print("  Note: Backward GQA requires bwd-family kernel (separate JIT)")
    else:
        print(f"  JIT build: {setup.error}")
        print("  Continuing with CPU reference only")

    # --- Sweep GQA ratios ---
    print("\n--- Backward Gradients per GQA Ratio ---")
    print(
        f"\n  {'#':<3} {'Ratio':<8} {'Hq':>4} {'Hk':>4} "
        f"| {'|dQ| mean':>10} {'|dK| mean':>10} {'|dV| mean':>10} "
        f"| {'dK shape':>18} {'dV shape':>18}"
    )
    print("  " + "-" * 104)

    all_results = {}

    for i, ratio in enumerate(gqa_ratios, 1):
        hk = hq // ratio
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
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
        dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

        out, P, lse = cpu_fwd_with_intermediates(Q, K, V, prob.scale)
        dQ, dK, dV = cpu_bwd_gqa(Q, K, V, out, dO, P, prob.scale, hq, hk)

        dq_mean = float(np.abs(dQ).mean())
        dk_mean = float(np.abs(dK).mean())
        dv_mean = float(np.abs(dV).mean())

        label = f"{ratio}:1"
        if ratio == 1:
            label += " MHA"
        elif hk == 1:
            label += " MQA"

        print(
            f"  {i:<3} {label:<8} {hq:>4} {hk:>4} "
            f"| {dq_mean:>10.4e} {dk_mean:>10.4e} {dv_mean:>10.4e} "
            f"| {str(dK.shape):>18} {str(dV.shape):>18}"
        )
        all_results[ratio] = (dQ, dK, dV, Q, K, V, out, dO, P, prob)

    # --- Verify GQA backward via expanded MHA ---
    print("\n--- GQA Backward Equivalence Check ---")
    print("  Verifying: GQA bwd == MHA bwd with expanded K/V, then summed")

    for ratio in gqa_ratios:
        if ratio == 1:
            continue

        dQ_gqa, dK_gqa, dV_gqa, Q, K, V, out, dO, P, prob = all_results[ratio]
        hk = hq // ratio

        K_exp = np.repeat(K, ratio, axis=1)
        V_exp = np.repeat(V, ratio, axis=1)

        O_mha, P_mha, _ = cpu_fwd_with_intermediates(Q, K_exp, V_exp, prob.scale)
        dQ_mha, dK_mha, dV_mha = cpu_bwd_gqa(
            Q,
            K_exp,
            V_exp,
            O_mha,
            dO,
            P_mha,
            prob.scale,
            hq,
            hq,
        )

        B = Q.shape[0]
        Sk = K.shape[2]
        dK_mha_grouped = dK_mha.reshape(B, hk, ratio, Sk, K.shape[3]).sum(axis=2)
        dV_mha_grouped = dV_mha.reshape(B, hk, ratio, Sk, V.shape[3]).sum(axis=2)

        dq_err = float(np.abs(dQ_gqa - dQ_mha).max())
        dk_err = float(np.abs(dK_gqa - dK_mha_grouped).max())
        dv_err = float(np.abs(dV_gqa - dV_mha_grouped).max())

        tag = "PASS" if max(dq_err, dk_err, dv_err) < 1e-5 else "FAIL"
        print(
            f"  Ratio {ratio}:1 -- dQ err={dq_err:.2e}  dK err={dk_err:.2e}  "
            f"dV err={dv_err:.2e}  {tag}"
        )

    # --- Gradient accumulation analysis ---
    print("\n--- Head-Group Gradient Accumulation ---")
    print("  When ratio > 1, dK/dV are summed over query heads in each group.")
    print("  Higher ratio -> more terms summed -> larger gradient magnitudes.\n")

    print(f"  {'Ratio':<8} {'||dK||_2':>12} {'||dV||_2':>12} {'dK/dV ratio':>12}")
    print("  " + "-" * 48)

    for ratio in gqa_ratios:
        dQ, dK, dV, *_ = all_results[ratio]
        l2_dk = float(np.sqrt((dK**2).sum()))
        l2_dv = float(np.sqrt((dV**2).sum()))
        dk_dv_ratio = l2_dk / (l2_dv + 1e-12)
        print(f"  {ratio}:1{'':<4} {l2_dk:>12.4e} {l2_dv:>12.4e} {dk_dv_ratio:>12.2f}")

    # --- Backward GPU API pattern ---
    print("\n--- Backward GPU API Pattern ---")
    print("  GPU backward with GQA dispatches with nhead_q != nhead_k.")
    print("  The dq_dk_dv kernel handles head grouping internally:")
    print("    - dQ: computed per query head (no grouping needed)")
    print("    - dK, dV: accumulated across head groups via atomicAdd")
    print("              or multi-buffer reduction (deterministic mode)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  GQA ratios tested:  {len(gqa_ratios)}")
    print("  Backward math:      expand K/V -> compute grads -> sum dK/dV")
    print("  Equivalence:        GQA bwd == MHA(expanded) bwd + group sum")
    print("  GPU:                Requires bwd-family JIT kernel")
    print("  Status:             DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
