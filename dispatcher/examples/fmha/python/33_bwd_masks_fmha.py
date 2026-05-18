#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 33: Backward Pass with Causal Masks

Demonstrates the FMHA backward pass with causal mask variants:
1. no_mask     -- Full attention (baseline)
2. top_left    -- Causal mask aligned to top-left corner
3. bottom_right -- Causal mask aligned to bottom-right corner

For each mask type:
- Forward: out = softmax(mask(Q @ K^T * scale)) @ V
- Backward: dQ, dK, dV via analytical gradients through the masked softmax

CPU backward reference:
  dP = dO @ V^T
  D  = rowsum(dO * out)     (per-query-position scalar)
  dS = P * (dP - D)
  dQ = scale * dS @ K
  dK = scale * dS^T @ Q
  dV = P^T @ dO

Usage:
    python3 33_bwd_masks_fmha.py
    python3 33_bwd_masks_fmha.py --seqlen-q 128 --seqlen-k 192
    python3 33_bwd_masks_fmha.py --arch gfx942
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


def make_causal_mask_top_left(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Causal mask aligned to top-left: position i attends to positions <= i."""
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    return (col <= row).astype(np.float32)


def make_causal_mask_bottom_right(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Causal mask aligned to bottom-right: accounts for kv longer than q."""
    offset = seqlen_k - seqlen_q
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    return (col <= row + offset).astype(np.float32)


def cpu_masked_fwd_with_intermediates(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    mask: np.ndarray,
) -> tuple:
    """Forward pass with mask, returning out, P, and LSE for backward.

    Args:
        Q: [B, H, Sq, D]   K: [B, H, Sk, D]   V: [B, H, Sk, Dv]
        mask: [Sq, Sk] broadcast over batch and head

    Returns: (out, P, lse)
    """
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    mask_broad = mask[np.newaxis, np.newaxis, :, :]
    S = np.where(mask_broad > 0, S, -1e9)
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum
    out = np.matmul(P, V)
    lse = (np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)).astype(np.float32)
    return out, P, lse


def cpu_masked_bwd(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
) -> tuple:
    """CPU backward through masked softmax attention.

    P already incorporates the mask (zeroed-out positions have P=0).

    Returns: (dQ, dK, dV, D)
    """
    D = (dO * out).sum(axis=-1, keepdims=True)
    dP = np.matmul(dO, V.transpose(0, 1, 3, 2))
    dS = P * (dP - D)
    dQ = np.matmul(dS, K) * scale
    dK = np.matmul(dS.transpose(0, 1, 3, 2), Q) * scale
    dV = np.matmul(P.transpose(0, 1, 3, 2), dO)
    return dQ, dK, dV, D.squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Backward Pass with Causal Masks")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen-q", type=int, default=64)
    parser.add_argument("--seqlen-k", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 33: Backward Pass with Causal Masks")
    print("=" * 70)

    sq, sk = args.seqlen_q, args.seqlen_k
    prob = FmhaProblem(
        batch=args.batch,
        nhead_q=args.nhead,
        nhead_k=args.nhead,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )

    print(f"\n  Problem: B={prob.batch} H={prob.nhead_q} Sq={sq} Sk={sk} D={args.hdim}")
    print(f"  Scale:   {prob.scale:.6f}")
    print(f"  Arch:    {args.arch}")

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
        print(f"  Library: {setup.library_path}")
        print("  Note: Backward requires family='bwd' kernel (separate JIT)")
    else:
        print(f"  JIT build: {setup.error}")
        print("  Continuing with CPU reference only")

    # --- Generate data ---
    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

    # --- Build masks ---
    masks = {
        "no_mask": np.ones((sq, sk), dtype=np.float32),
        "top_left": make_causal_mask_top_left(sq, sk),
        "bottom_right": make_causal_mask_bottom_right(sq, sk),
    }

    # --- Per-mask forward + backward ---
    print(
        f"\n  {'Mask':<16} {'Density':>8} | {'|dQ|':>10} {'|dK|':>10} {'|dV|':>10}"
        f" | {'dQ vs base':>10} {'dK vs base':>10} {'dV vs base':>10}"
    )
    print("  " + "-" * 98)

    base_grads = None
    all_grads = {}

    for name, mask in masks.items():
        density = mask.sum() / mask.size * 100

        out, P, lse = cpu_masked_fwd_with_intermediates(Q, K, V, prob.scale, mask)
        dQ, dK, dV, D = cpu_masked_bwd(Q, K, V, out, dO, P, prob.scale)

        dq_norm = float(np.abs(dQ).mean())
        dk_norm = float(np.abs(dK).mean())
        dv_norm = float(np.abs(dV).mean())

        if base_grads is None:
            base_grads = (dQ, dK, dV)
            diff_str = f"{'---':>10} {'---':>10} {'---':>10}"
        else:
            dq_diff = float(np.abs(dQ - base_grads[0]).max())
            dk_diff = float(np.abs(dK - base_grads[1]).max())
            dv_diff = float(np.abs(dV - base_grads[2]).max())
            diff_str = f"{dq_diff:>10.2e} {dk_diff:>10.2e} {dv_diff:>10.2e}"

        print(
            f"  {name:<16} {density:>7.1f}% | {dq_norm:>10.4e} {dk_norm:>10.4e} {dv_norm:>10.4e}"
            f" | {diff_str}"
        )
        all_grads[name] = (dQ, dK, dV, D)

    # --- Detailed backward breakdown for each mask ---
    print("\n--- Backward Stage Details ---")

    for name, mask in masks.items():
        dQ, dK, dV, D = all_grads[name]
        out, P, lse = cpu_masked_fwd_with_intermediates(Q, K, V, prob.scale, mask)

        print(f"\n  [{name}]")
        print("    Stage 1 (dot_do_o): D = rowsum(dO * out)")
        print(f"      D shape: {D.shape}, range: [{D.min():.6f}, {D.max():.6f}]")
        print("    Stage 2 (dq_dk_dv):")
        print(f"      dQ range: [{dQ.min():.4e}, {dQ.max():.4e}]")
        print(f"      dK range: [{dK.min():.4e}, {dK.max():.4e}]")
        print(f"      dV range: [{dV.min():.4e}, {dV.max():.4e}]")

        p_sparsity = (P < 1e-9).sum() / P.size * 100
        print(f"    P sparsity (< 1e-9): {p_sparsity:.1f}%")

    # --- Gradient norm comparison across masks ---
    print("\n--- Gradient L2 Norms ---")
    print(f"\n  {'Mask':<16} {'||dQ||_2':>12} {'||dK||_2':>12} {'||dV||_2':>12}")
    print("  " + "-" * 54)

    for name in masks:
        dQ, dK, dV, _ = all_grads[name]
        l2_dq = float(np.sqrt((dQ**2).sum()))
        l2_dk = float(np.sqrt((dK**2).sum()))
        l2_dv = float(np.sqrt((dV**2).sum()))
        print(f"  {name:<16} {l2_dq:>12.4e} {l2_dk:>12.4e} {l2_dv:>12.4e}")

    # --- Mask pattern visualization ---
    print("\n--- Mask Patterns (first 8x8 corner) ---")
    view = min(8, sq, sk)
    for name, mask in masks.items():
        corner = mask[:view, :view]
        print(f"\n  {name}:")
        for r in range(view):
            row_str = " ".join("█" if corner[r, c] > 0 else "·" for c in range(view))
            print(f"    {row_str}")

    # --- Backward API pattern ---
    print("\n--- Backward GPU API Pattern ---")
    print("  The GPU backward for masked attention would use:")
    print("    FmhaKernelConfig(family='bwd', mask='top_left', ...)")
    print("  3-stage backward plan:")
    print("    Stage 1: bwd_dot_do_o  -- D = rowsum(dO * out)")
    print("    Stage 2: bwd_dq_dk_dv  -- compute dQ, dK, dV with mask")
    print("    Stage 3: bwd_convert_dq -- optional dtype conversion")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Mask variants:  no_mask, top_left, bottom_right")
    print("  Backward math:  dP = dO @ V^T, dS = P*(dP - D)")
    print("                  dQ = scale*dS@K, dK = scale*dS^T@Q, dV = P^T@dO")
    print("  Causal effect:  Masked positions get P=0, zeroing their gradient flow")
    print("  GPU:            Requires bwd-family JIT kernel with mask support")
    print("  Status:         DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
