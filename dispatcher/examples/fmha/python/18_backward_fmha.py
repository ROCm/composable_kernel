#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 18: Backward Pass (dQ, dK, dV)

Demonstrates:
1. Forward pass to obtain O and LSE
2. Backward pass computing gradients dQ, dK, dV from dO
3. Three-stage backward plan:
   - Stage 1 (dot_do_o):      Compute D = rowsum(dO * O)
   - Stage 2 (dq_dk_dv):      Compute dQ, dK, dV using D and LSE
   - Stage 3 (convert_dq):    Optional dtype conversion for dQ
4. CPU reference with analytical gradients
5. Gradient checking via finite differences

Usage:
    python3 18_backward_fmha.py
    python3 18_backward_fmha.py --seqlen 128
    python3 18_backward_fmha.py --check-grad --eps 1e-3
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


def cpu_attention_fwd_with_lse(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> tuple:
    """Forward pass returning O, P (attention weights), and LSE.

    Returns: (O, P, lse)
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
    out = np.matmul(P, V)
    lse = (np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)).astype(np.float32)
    return out, P, lse


def cpu_attention_bwd(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
) -> tuple:
    """CPU reference backward pass.

    Computes analytical gradients dQ, dK, dV.

    Stage 1: D_i = sum_j(dO_ij * O_ij)  (per query position)
    Stage 2: dS = P * (dO @ V^T - D)
             dQ = dS @ K * scale
             dK = dS^T @ Q * scale
             dV = P^T @ dO

    Returns: (dQ, dK, dV, D)
    """
    # Stage 1: dot_do_o
    D = (dO * out).sum(axis=-1, keepdims=True)

    # Stage 2: dq_dk_dv
    dP = np.matmul(dO, V.transpose(0, 1, 3, 2))
    dS = P * (dP - D)

    dQ = np.matmul(dS, K) * scale
    dK = np.matmul(dS.transpose(0, 1, 3, 2), Q) * scale
    dV = np.matmul(P.transpose(0, 1, 3, 2), dO)

    return dQ, dK, dV, D.squeeze(-1)


def finite_difference_check(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    dO: np.ndarray,
    scale: float,
    eps: float = 1e-3,
    param_name: str = "Q",
    max_checks: int = 5,
) -> float:
    """Verify gradients via finite differences on a few elements."""
    param_map = {"Q": Q, "K": K, "V": V}
    param = param_map[param_name]

    O_ref, P_ref, _ = cpu_attention_fwd_with_lse(Q, K, V, scale)
    _, _, _, _ = cpu_attention_bwd(Q, K, V, O_ref, dO, P_ref, scale)

    grad_map = {"Q": 0, "K": 1, "V": 2}
    grad_idx = grad_map[param_name]
    grads = cpu_attention_bwd(Q, K, V, O_ref, dO, P_ref, scale)
    analytical_grad = grads[grad_idx]

    max_err = 0.0
    flat_indices = np.random.choice(
        param.size, min(max_checks, param.size), replace=False
    )

    for flat_idx in flat_indices:
        idx = np.unravel_index(flat_idx, param.shape)
        orig = param[idx]

        param[idx] = orig + eps
        O_plus = cpu_attention_fwd(Q, K, V, scale)
        loss_plus = (O_plus * dO).sum()

        param[idx] = orig - eps
        O_minus = cpu_attention_fwd(Q, K, V, scale)
        loss_minus = (O_minus * dO).sum()

        param[idx] = orig

        fd_grad = (loss_plus - loss_minus) / (2 * eps)
        an_grad = analytical_grad[idx]
        err = abs(fd_grad - an_grad) / (abs(fd_grad) + 1e-8)
        max_err = max(max_err, err)

    return max_err


def main():
    parser = argparse.ArgumentParser(description="Backward Pass (dQ, dK, dV)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--check-grad", action="store_true", help="Run finite-difference check"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-3, help="Finite-difference epsilon"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 18: Backward Pass (dQ, dK, dV)")
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

    print(f"\n  Problem: B={prob.batch} H={prob.nhead_q} S={args.seqlen} D={args.hdim}")

    # --- Generate data ---
    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

    # --- Forward pass ---
    print("\n--- Stage 0: Forward Pass ---")
    out, P, lse = cpu_attention_fwd_with_lse(Q, K, V, prob.scale)
    print(f"  O shape:   {out.shape}")
    print(f"  O range:   [{out.min():.4f}, {out.max():.4f}]")
    print(f"  LSE shape: {lse.shape}")
    print(f"  LSE range: [{lse.min():.4f}, {lse.max():.4f}]")
    print(f"  P sparsity (< 1e-6): {(P < 1e-6).sum() / P.size * 100:.1f}%")

    # --- Backward pass (3 stages) ---
    print("\n--- Stage 1: dot_do_o (D = rowsum(dO * O)) ---")
    D_full = (dO * out).sum(axis=-1)
    print(f"  D shape: {D_full.shape}")
    print(f"  D range: [{D_full.min():.6f}, {D_full.max():.6f}]")

    print("\n--- Stage 2: dq_dk_dv ---")
    dQ, dK, dV, D = cpu_attention_bwd(Q, K, V, out, dO, P, prob.scale)
    print(f"  dQ shape: {dQ.shape}, range: [{dQ.min():.4e}, {dQ.max():.4e}]")
    print(f"  dK shape: {dK.shape}, range: [{dK.min():.4e}, {dK.max():.4e}]")
    print(f"  dV shape: {dV.shape}, range: [{dV.min():.4e}, {dV.max():.4e}]")

    print("\n--- Stage 3: convert_dq (optional fp32 -> fp16) ---")
    dQ_fp16 = dQ.astype(np.float16)
    convert_err = float(np.abs(dQ - dQ_fp16.astype(np.float32)).max())
    print(f"  dQ fp32 -> fp16 max error: {convert_err:.2e}")

    # --- Gradient norms ---
    print("\n--- Gradient Statistics ---")
    print(
        f"\n  {'Param':<6} {'L2 Norm':>12} {'Max Abs':>12} {'Mean Abs':>12} {'Shape'}"
    )
    print("  " + "-" * 60)
    for name, grad in [("dQ", dQ), ("dK", dK), ("dV", dV)]:
        l2 = float(np.sqrt((grad**2).sum()))
        ma = float(np.abs(grad).max())
        mean_a = float(np.abs(grad).mean())
        print(f"  {name:<6} {l2:>12.4e} {ma:>12.4e} {mean_a:>12.4e} {grad.shape}")

    # --- Finite difference check ---
    if args.check_grad:
        print(f"\n--- Finite Difference Gradient Check (eps={args.eps}) ---")
        for pname in ["Q", "K", "V"]:
            Q_c, K_c, V_c = Q.copy(), K.copy(), V.copy()
            err = finite_difference_check(
                Q_c,
                K_c,
                V_c,
                dO,
                prob.scale,
                eps=args.eps,
                param_name=pname,
                max_checks=5,
            )
            tag = "PASS" if err < 1e-2 else "FAIL"
            print(f"  d{pname}: max_rel_err = {err:.4e}  {tag}")

    # --- GPU forward attempt ---
    print("\n--- GPU Execution ---")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=args.hdim,
        hdim_v=args.hdim,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
    else:
        runner = setup.runner
        print(f"  JIT build: {setup.build_time_s:.1f}s")
        Q_fp16 = Q.astype(np.float16)
        K_fp16 = K.astype(np.float16)
        V_fp16 = V.astype(np.float16)
        res = runner.run(Q_fp16, K_fp16, V_fp16, prob)
        if res.success:
            print(f"  Forward GPU: {res.time_ms:.4f} ms, {res.tflops:.2f} TFLOPS")
            validator = FmhaValidator(rtol=1e-2, atol=1e-2)
            ok, ma, _ = validator.check(res.output, out)
            print(f"  Forward validation: max_err={ma:.2e}, {'PASS' if ok else 'FAIL'}")
        else:
            print("  Forward GPU: Kernel returned failure")
        print("  Backward GPU: Not available (requires bwd family kernel)")

    # --- Backward plan structure ---
    print("\n--- Backward Plan Structure ---")
    print("  Stage 1: dot_do_o")
    print(f"    Input:  dO [{prob.o_shape()}], O [{prob.o_shape()}]")
    print(f"    Output: D  [{prob.batch}, {prob.nhead_q}, {prob.seqlen_q}]")
    print("  Stage 2: dq_dk_dv")
    print("    Input:  Q, K, V, dO, LSE, D")
    print("    Output: dQ, dK, dV (in accumulator precision)")
    print("  Stage 3: convert_dq")
    print("    Input:  dQ (fp32)")
    print("    Output: dQ (fp16)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Forward:   O = softmax(Q @ K^T / sqrt(d)) @ V")
    print("  Backward:  3-stage plan (dot_do_o -> dq_dk_dv -> convert_dq)")
    print(f"  Gradients: dQ [{dQ.shape}], dK [{dK.shape}], dV [{dV.shape}]")
    print("  GPU:       Prebuilt supports forward only")
    print("  Status:    DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
