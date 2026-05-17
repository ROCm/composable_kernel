#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 28: Backward Bias Gradient (dbias) FMHA

Demonstrates computing the gradient of the elementwise attention bias
during the backward pass. When forward attention uses:
    S = Q @ K^T * scale + bias
the backward pass must compute:
    dbias = sum over batch of (dP)
where dP is the gradient of the attention probabilities.

This is useful for learnable relative position biases (e.g., ALiBi
training, T5-style relative position embeddings).

The prebuilt library only has a forward kernel. This example validates
the dbias CPU reference and shows the API pattern.

Usage:
    python3 28_backward_dbias_fmha.py
    python3 28_backward_dbias_fmha.py --seqlen 128
    python3 28_backward_dbias_fmha.py --bias-type alibi
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    cpu_attention_fwd,
    detect_gpu_arch,
)


def make_elementwise_bias(nhead: int, seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Create a simple elementwise attention bias [nhead, seqlen_q, seqlen_k]."""
    bias = np.zeros((nhead, seqlen_q, seqlen_k), dtype=np.float32)
    for h in range(nhead):
        for i in range(seqlen_q):
            for j in range(seqlen_k):
                bias[h, i, j] = -0.1 * abs(i - j) * (h + 1) / nhead
    return bias


def make_alibi_bias(nhead: int, seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Create ALiBi-style attention bias [nhead, seqlen_q, seqlen_k].

    ALiBi adds a linear penalty proportional to distance:
        bias[h, i, j] = -slope_h * |i - j|
    where slope_h decreases geometrically across heads.
    """
    slopes = np.array([2 ** (-(8 * (h + 1) / nhead)) for h in range(nhead)])
    bias = np.zeros((nhead, seqlen_q, seqlen_k), dtype=np.float32)
    for h in range(nhead):
        for i in range(seqlen_q):
            for j in range(seqlen_k):
                bias[h, i, j] = -slopes[h] * abs(i - j)
    return bias


def cpu_attention_fwd_bias(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    bias: np.ndarray,
) -> tuple:
    """CPU forward with elementwise bias, returning intermediates.

    Args:
        Q:    [B, H, Sq, Dq]
        K:    [B, H, Sk, Dq]
        V:    [B, H, Sk, Dv]
        bias: [H, Sq, Sk] broadcast over batch

    Returns:
        O:   [B, H, Sq, Dv]
        P:   [B, H, Sq, Sk] attention probabilities
        lse: [B, H, Sq] log-sum-exp
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)

    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S = S + bias[np.newaxis, :, :, :]

    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum

    lse = np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)
    out = np.matmul(P, V)
    return out, P, lse


def cpu_attention_bwd_dbias(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
    bias: np.ndarray,
) -> tuple:
    """CPU backward computing dQ, dK, dV, and dbias.

    Args:
        Q, K, V: forward inputs [B, H, Sq/Sk, D]
        out:     forward output [B, H, Sq, Dv]
        dO:      output gradient [B, H, Sq, Dv]
        P:       attention probabilities [B, H, Sq, Sk]
        scale:   softmax scale
        bias:    [H, Sq, Sk] attention bias

    Returns:
        dQ:    [B, H, Sq, Dq]
        dK:    [B, H, Sk, Dq]
        dV:    [B, H, Sk, Dv]
        dbias: [H, Sq, Sk]  summed over batch dimension
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)

    dV = np.matmul(P.transpose(0, 1, 3, 2), dO)

    dP = np.matmul(dO, V.transpose(0, 1, 3, 2))

    D = (dO * out).sum(axis=-1, keepdims=True)
    dS = P * (dP - D) * scale

    dQ = np.matmul(dS, K)
    dK = np.matmul(dS.transpose(0, 1, 3, 2), Q)

    dbias = dS.sum(axis=0) / scale

    return dQ, dK, dV, dbias


def main():
    parser = argparse.ArgumentParser(
        description="Backward Bias Gradient (dbias) FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--bias-type", choices=["elementwise", "alibi"], default="elementwise"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 28: Backward Bias Gradient (dbias) FMHA")
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

    # Step 1: Create bias
    print(f"\nStep 1: Create {args.bias_type.title()} Bias")

    if args.bias_type == "alibi":
        bias = make_alibi_bias(args.nhead, args.seqlen, args.seqlen)
    else:
        bias = make_elementwise_bias(args.nhead, args.seqlen, args.seqlen)

    print(f"  Bias shape: {bias.shape}")
    print(f"  Bias range: [{bias.min():.4f}, {bias.max():.4f}]")
    print(f"  Bias type:  {args.bias_type}")

    for h in range(min(4, args.nhead)):
        print(
            f"  Head {h}: range=[{bias[h].min():.4f}, {bias[h].max():.4f}]  "
            f"mean={bias[h].mean():.4f}"
        )

    # Step 2: Forward pass with bias
    print("\nStep 2: Forward Pass with Bias")

    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float32)

    O_nobias = cpu_attention_fwd(Q, K, V, prob.scale)
    O_bias, P, lse = cpu_attention_fwd_bias(Q, K, V, prob.scale, bias)

    diff = np.abs(O_nobias - O_bias)
    print(f"  O (no bias): range=[{O_nobias.min():.4f}, {O_nobias.max():.4f}]")
    print(f"  O (biased):  range=[{O_bias.min():.4f}, {O_bias.max():.4f}]")
    print(f"  Bias effect: max_diff={diff.max():.6e}  mean_diff={diff.mean():.6e}")

    # Step 3: Backward pass with dbias
    print("\nStep 3: Backward Pass (dQ, dK, dV, dbias)")

    np.random.seed(123)
    dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

    dQ, dK, dV, dbias = cpu_attention_bwd_dbias(
        Q,
        K,
        V,
        O_bias,
        dO,
        P,
        prob.scale,
        bias,
    )

    print(f"  dQ shape:    {dQ.shape}    range=[{dQ.min():.6f}, {dQ.max():.6f}]")
    print(f"  dK shape:    {dK.shape}    range=[{dK.min():.6f}, {dK.max():.6f}]")
    print(f"  dV shape:    {dV.shape}    range=[{dV.min():.6f}, {dV.max():.6f}]")
    print(f"  dbias shape: {dbias.shape}  range=[{dbias.min():.6f}, {dbias.max():.6f}]")

    # Step 4: Verify dbias via finite differences
    print("\nStep 4: dbias Gradient Verification (Finite Differences)")

    eps = 1e-3
    num_checks = 8
    rng = np.random.RandomState(99)

    print(
        f"\n  {'Index':>20} {'Analytic':>14} {'Numerical':>14} {'RelErr':>12} {'Status':>8}"
    )
    print("  " + "-" * 72)

    all_grad_ok = True
    for _ in range(num_checks):
        h = rng.randint(0, args.nhead)
        i = rng.randint(0, args.seqlen)
        j = rng.randint(0, args.seqlen)

        bias_plus = bias.copy()
        bias_plus[h, i, j] += eps
        bias_minus = bias.copy()
        bias_minus[h, i, j] -= eps

        O_p, _, _ = cpu_attention_fwd_bias(Q, K, V, prob.scale, bias_plus)
        O_m, _, _ = cpu_attention_fwd_bias(Q, K, V, prob.scale, bias_minus)

        numerical = ((O_p * dO).sum() - (O_m * dO).sum()) / (2 * eps)
        analytic = dbias[h, i, j]

        rel_err = abs(analytic - numerical) / (abs(numerical) + 1e-8)
        ok = rel_err < 1e-2
        all_grad_ok = all_grad_ok and ok
        idx_str = f"({h},{i},{j})"
        print(
            f"  {idx_str:>20} {analytic:>14.6f} {numerical:>14.6f} {rel_err:>12.2e} {'OK' if ok else 'FAIL':>8}"
        )

    # Step 5: dbias structure analysis
    print("\nStep 5: dbias Structure Analysis")

    print("\n  Per-head dbias statistics:")
    print(f"  {'Head':>6} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-" * 56)

    for h in range(min(8, args.nhead)):
        db_h = dbias[h]
        print(
            f"  {h:>6} {db_h.mean():>12.6f} {db_h.std():>12.6f} "
            f"{db_h.min():>12.6f} {db_h.max():>12.6f}"
        )

    # Step 6: Batch size effect on dbias
    print("\nStep 6: Batch Size Effect on dbias")
    print("  dbias = sum of per-sample dS / scale over batch dimension")
    print("  Larger batch -> dbias aggregates more gradient signal")

    batch_sizes = [1, 2, 4, 8]
    print(
        f"\n  {'Batch':>6} {'|dbias| mean':>14} {'|dbias| max':>14} {'dbias std':>14}"
    )
    print("  " + "-" * 52)

    for b in batch_sizes:
        Q_b = (np.random.randn(b, args.nhead, args.seqlen, args.hdim) * 0.3).astype(
            np.float32
        )
        K_b = (np.random.randn(b, args.nhead, args.seqlen, args.hdim) * 0.3).astype(
            np.float32
        )
        V_b = (np.random.randn(b, args.nhead, args.seqlen, args.hdim) * 0.3).astype(
            np.float32
        )
        dO_b = (np.random.randn(b, args.nhead, args.seqlen, args.hdim) * 0.1).astype(
            np.float32
        )

        O_b, P_b, lse_b = cpu_attention_fwd_bias(Q_b, K_b, V_b, prob.scale, bias)
        _, _, _, dbias_b = cpu_attention_bwd_dbias(
            Q_b,
            K_b,
            V_b,
            O_b,
            dO_b,
            P_b,
            prob.scale,
            bias,
        )
        print(
            f"  {b:>6} {np.abs(dbias_b).mean():>14.6f} {np.abs(dbias_b).max():>14.6f} "
            f"{dbias_b.std():>14.6f}"
        )

    # Step 7: GPU API pattern
    print("\nStep 7: GPU Kernel Configuration")
    print("  NOTE: The prebuilt library only has a forward kernel without bias.")
    print("  For backward with dbias, compile kernels with:")
    print()
    print("    Forward:  FmhaSignature().bias('bias')  // elementwise bias")
    print("    Backward: FmhaSignature()")
    print("                  .family('bwd_dq_dk_dv')")
    print("                  .bias('bias')")
    print("                  .dbias(true)  // enable dbias computation")
    print()
    print("  In codegen JSON:")
    print("    'bias': 'bias',     // forward: elementwise bias")
    print("    'dbias': true,      // backward: compute bias gradient")

    # Summary
    print("\n" + "=" * 70)
    print("  dbias = sum_batch(P * (dP - D)) (gradient of elementwise bias)")
    print(f"  Shape: [{args.nhead}, {args.seqlen}, {args.seqlen}] (same as bias)")
    print(f"  Gradient check: {'PASS' if all_grad_ok else 'FAIL'}")
    print("  Use case: learnable relative position biases (ALiBi, T5, etc.)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
