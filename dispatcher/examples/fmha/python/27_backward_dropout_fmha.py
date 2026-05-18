#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 27: Backward Pass with Dropout FMHA

Demonstrates the FMHA backward pass with dropout. The backward pass
computes dQ, dK, dV given dO (gradient of the output). When dropout is
applied during forward, the same dropout mask must be replayed during
backward for correctness.

Key concepts:
  - Deterministic mode (no atomics): reproducible gradients, may be slower
  - Non-deterministic mode: uses atomicAdd for dQ, faster but non-reproducible
  - store_randval: optionally store the dropout random values for debugging

The prebuilt library only has a forward kernel. This example validates
the backward CPU reference and shows the API pattern.

Usage:
    python3 27_backward_dropout_fmha.py
    python3 27_backward_dropout_fmha.py --dropout 0.2
    python3 27_backward_dropout_fmha.py --seqlen 128 --deterministic
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
)


def cpu_attention_fwd_dropout(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    dropout_p: float,
    seed: int = 42,
) -> tuple:
    """CPU reference: forward with dropout, returning intermediates for backward.

    Returns:
        O: [B, H, Sq, Dv] output
        P_drop: [B, H, Sq, Sk] attention weights after dropout
        lse: [B, H, Sq] log-sum-exp for numerical stability
        drop_mask: [B, H, Sq, Sk] binary dropout mask
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)

    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum

    lse = np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)

    rng = np.random.RandomState(seed)
    drop_mask = (rng.rand(*P.shape) >= dropout_p).astype(np.float32)
    drop_scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0
    P_drop = P * drop_mask * drop_scale

    out = np.matmul(P_drop, V)
    return out, P_drop, lse, drop_mask


def cpu_attention_bwd_dropout(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    lse: np.ndarray,
    scale: float,
    dropout_p: float,
    drop_mask: np.ndarray,
    deterministic: bool = False,
) -> tuple:
    """CPU reference: backward with dropout.

    Args:
        Q:    [B, H, Sq, Dq]  float32
        K:    [B, H, Sk, Dq]  float32 (already GQA-expanded if needed)
        V:    [B, H, Sk, Dv]  float32
        out:  [B, H, Sq, Dv]  float32  (forward output)
        dO:   [B, H, Sq, Dv]  float32  (output gradient)
        lse:  [B, H, Sq]       float32  (log-sum-exp from forward)
        scale: softmax scale
        dropout_p: dropout probability
        drop_mask: [B, H, Sq, Sk] binary mask from forward
        deterministic: if True, avoid any non-deterministic accumulation

    Returns:
        dQ: [B, H, Sq, Dq]
        dK: [B, H, Sk, Dq]
        dV: [B, H, Sk, Dv]
    """
    drop_scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0

    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max) / np.exp(S - S_max).sum(axis=-1, keepdims=True)

    P_drop = P * drop_mask * drop_scale

    dV = np.matmul(P_drop.transpose(0, 1, 3, 2), dO)

    dP_drop = np.matmul(dO, V.transpose(0, 1, 3, 2))

    dP = dP_drop * drop_mask * drop_scale

    D = (dO * out).sum(axis=-1, keepdims=True)
    dS = P * (dP - D) * scale

    dQ = np.matmul(dS, K)
    dK = np.matmul(dS.transpose(0, 1, 3, 2), Q)

    return dQ, dK, dV


def main():
    parser = argparse.ArgumentParser(
        description="Backward Pass with Dropout FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic mode"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 27: Backward Pass with Dropout FMHA")
    print("=" * 70)

    prob = FmhaProblem(
        batch=args.batch,
        nhead_q=args.nhead,
        nhead_k=args.nhead,
        seqlen_q=args.seqlen,
        seqlen_k=args.seqlen,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )

    # Step 1: Forward with dropout
    print("\nStep 1: Forward Pass with Dropout")

    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float32)

    O_nodrop = cpu_attention_fwd(Q, K, V, prob.scale)
    O_drop, P_drop, lse, drop_mask = cpu_attention_fwd_dropout(
        Q,
        K,
        V,
        prob.scale,
        args.dropout,
        seed=42,
    )

    print(f"  Shape:      {prob.q_shape()}")
    print(f"  Dropout:    p={args.dropout}")
    print(
        f"  Drop mask:  {drop_mask.sum():.0f}/{drop_mask.size} kept "
        f"({100 * drop_mask.mean():.1f}%, expected {100 * (1 - args.dropout):.1f}%)"
    )
    print(f"  O (no drop): range=[{O_nodrop.min():.4f}, {O_nodrop.max():.4f}]")
    print(f"  O (dropout): range=[{O_drop.min():.4f}, {O_drop.max():.4f}]")
    print(f"  LSE shape:  {lse.shape}")

    # Step 2: Backward pass
    print("\nStep 2: Backward Pass")

    np.random.seed(123)
    dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

    dQ, dK, dV = cpu_attention_bwd_dropout(
        Q,
        K,
        V,
        O_drop,
        dO,
        lse,
        prob.scale,
        args.dropout,
        drop_mask,
        deterministic=args.deterministic,
    )

    print(f"  dQ shape: {dQ.shape}  range=[{dQ.min():.6f}, {dQ.max():.6f}]")
    print(f"  dK shape: {dK.shape}  range=[{dK.min():.6f}, {dK.max():.6f}]")
    print(f"  dV shape: {dV.shape}  range=[{dV.min():.6f}, {dV.max():.6f}]")
    print(f"  Deterministic: {args.deterministic}")

    # Step 3: Verify gradient correctness via finite differences
    print("\nStep 3: Gradient Verification (Finite Differences)")

    eps = 1e-3
    num_checks = 5
    rng = np.random.RandomState(99)

    print(f"\n  Checking {num_checks} random elements per tensor:")
    print(
        f"  {'Tensor':>8} {'Index':>24} {'Analytic':>14} {'Numerical':>14} {'RelErr':>12}"
    )
    print("  " + "-" * 76)

    for tensor_name, param, grad in [("dQ", Q, dQ), ("dK", K, dK), ("dV", V, dV)]:
        for _ in range(num_checks):
            idx = tuple(rng.randint(0, s) for s in param.shape)

            param_plus = param.copy()
            param_plus[idx] += eps
            param_minus = param.copy()
            param_minus[idx] -= eps

            if tensor_name == "dQ":
                O_p, _, _, _ = cpu_attention_fwd_dropout(
                    param_plus, K, V, prob.scale, args.dropout, seed=42
                )
                O_m, _, _, _ = cpu_attention_fwd_dropout(
                    param_minus, K, V, prob.scale, args.dropout, seed=42
                )
            elif tensor_name == "dK":
                O_p, _, _, _ = cpu_attention_fwd_dropout(
                    Q, param_plus, V, prob.scale, args.dropout, seed=42
                )
                O_m, _, _, _ = cpu_attention_fwd_dropout(
                    Q, param_minus, V, prob.scale, args.dropout, seed=42
                )
            else:
                O_p, _, _, _ = cpu_attention_fwd_dropout(
                    Q, K, param_plus, prob.scale, args.dropout, seed=42
                )
                O_m, _, _, _ = cpu_attention_fwd_dropout(
                    Q, K, param_minus, prob.scale, args.dropout, seed=42
                )

            numerical = (O_p * dO).sum() - (O_m * dO).sum()
            numerical /= 2 * eps
            analytic = grad[idx]

            rel_err = abs(analytic - numerical) / (abs(numerical) + 1e-8)
            idx_str = str(idx)
            print(
                f"  {tensor_name:>8} {idx_str:>24} {analytic:>14.6f} {numerical:>14.6f} {rel_err:>12.2e}"
            )

    # Step 4: Deterministic vs non-deterministic comparison
    print("\nStep 4: Deterministic vs Non-Deterministic")

    dQ_det, dK_det, dV_det = cpu_attention_bwd_dropout(
        Q,
        K,
        V,
        O_drop,
        dO,
        lse,
        prob.scale,
        args.dropout,
        drop_mask,
        deterministic=True,
    )
    dQ_ndet, dK_ndet, dV_ndet = cpu_attention_bwd_dropout(
        Q,
        K,
        V,
        O_drop,
        dO,
        lse,
        prob.scale,
        args.dropout,
        drop_mask,
        deterministic=False,
    )

    validator = FmhaValidator(rtol=1e-5, atol=1e-5)

    for name, g_det, g_ndet in [
        ("dQ", dQ_det, dQ_ndet),
        ("dK", dK_det, dK_ndet),
        ("dV", dV_det, dV_ndet),
    ]:
        ok, max_abs, _ = validator.check(g_det, g_ndet)
        print(
            f"  {name}: det vs non-det max_err={max_abs:.2e}  {'MATCH' if ok else 'DIFFER'}"
        )

    print("\n  NOTE: In CPU reference both modes are identical.")
    print("  On GPU, non-deterministic mode uses atomicAdd for dQ accumulation,")
    print("  which can cause tiny floating-point differences across runs.")

    # Step 5: Dropout probability sweep
    print("\nStep 5: Dropout Probability Sweep")

    probs = [0.0, 0.1, 0.2, 0.3, 0.5]
    print(
        f"\n  {'p':>6} {'|dQ| mean':>12} {'|dK| mean':>12} {'|dV| mean':>12} {'Kept%':>8}"
    )
    print("  " + "-" * 54)

    for p in probs:
        O_p, _, _, dm = cpu_attention_fwd_dropout(Q, K, V, prob.scale, p, seed=42)
        dQ_p, dK_p, dV_p = cpu_attention_bwd_dropout(
            Q,
            K,
            V,
            O_p,
            dO,
            lse,
            prob.scale,
            p,
            dm,
        )
        kept = 100 * dm.mean()
        print(
            f"  {p:>6.2f} {np.abs(dQ_p).mean():>12.6f} {np.abs(dK_p).mean():>12.6f} "
            f"{np.abs(dV_p).mean():>12.6f} {kept:>7.1f}%"
        )

    # Step 6: GPU API pattern
    print("\nStep 6: GPU Backward Kernel Configuration")
    print("  NOTE: The prebuilt library only has a forward kernel.")
    print("  FMHA backward requires 3 kernel stages:")
    print()
    print("    Stage 1: bwd_dot_do_o  -- compute D = rowsum(dO * O)")
    print("    Stage 2: bwd_dq_dk_dv  -- compute dQ, dK, dV")
    print("    Stage 3: bwd_convert_dq -- convert accumulated dQ")
    print()
    print("  With dropout, the signature requires:")
    print("    .dropout(true)")
    print("    .store_randval(false)   // or true to save random values")
    print(f"    .deterministic({'true' if args.deterministic else 'false'})")

    # Summary
    print("\n" + "=" * 70)
    print("  Backward with dropout: replays same mask from forward pass")
    print("  Deterministic mode: reproducible but potentially slower on GPU")
    print("  3-stage backward: dot_do_o -> dq_dk_dv -> convert_dq")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
