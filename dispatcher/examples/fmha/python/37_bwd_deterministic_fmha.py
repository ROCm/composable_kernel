#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 37: Backward Pass Deterministic Mode

Demonstrates deterministic vs non-deterministic backward computation.

Non-deterministic mode (default):
  - dQ is accumulated via atomicAdd across seqlen_k tiles
  - Faster but produces slightly different results each run
  - Acceptable for training where stochastic noise is tolerable

Deterministic mode:
  - Uses multi-buffer reduction instead of atomics
  - Each tile writes to a separate buffer, then a final reduction sums them
  - Bit-exact reproducible gradients across runs
  - Slower due to extra memory and reduction pass

CPU reference simulates both modes. On CPU, both modes are numerically
identical (no atomics), but this example demonstrates the API pattern
and compares GPU-style multi-buffer reduction semantics.

Usage:
    python3 37_bwd_deterministic_fmha.py
    python3 37_bwd_deterministic_fmha.py --seqlen 128
    python3 37_bwd_deterministic_fmha.py --num-tiles 4
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
    """Forward returning out, P, LSE."""
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum
    out = np.matmul(P, V)
    lse = (np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)).astype(np.float32)
    return out, P, lse


def cpu_bwd_nondeterministic(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
) -> tuple:
    """Standard backward (single accumulation). Returns (dQ, dK, dV)."""
    D = (dO * out).sum(axis=-1, keepdims=True)
    dP = np.matmul(dO, V.transpose(0, 1, 3, 2))
    dS = P * (dP - D)
    dQ = np.matmul(dS, K) * scale
    dK = np.matmul(dS.transpose(0, 1, 3, 2), Q) * scale
    dV = np.matmul(P.transpose(0, 1, 3, 2), dO)
    return dQ, dK, dV


def cpu_bwd_deterministic(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
    num_tiles_k: int = 4,
) -> tuple:
    """Deterministic backward with explicit multi-buffer reduction for dQ.

    Simulates the GPU pattern where seqlen_k is split into tiles,
    each tile writes dQ to a separate buffer, then buffers are summed.

    Returns: (dQ, dK, dV, dQ_buffers)
    """
    B, Hq, Sq, Dq = Q.shape
    Sk = K.shape[2]

    D = (dO * out).sum(axis=-1, keepdims=True)

    tile_sk = max(1, Sk // num_tiles_k)
    actual_tiles = (Sk + tile_sk - 1) // tile_sk

    dQ_buffers = np.zeros((actual_tiles, B, Hq, Sq, Dq), dtype=np.float32)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    for t in range(actual_tiles):
        sk_start = t * tile_sk
        sk_end = min(sk_start + tile_sk, Sk)

        K_tile = K[:, :, sk_start:sk_end, :]
        V_tile = V[:, :, sk_start:sk_end, :]
        P_tile = P[:, :, :, sk_start:sk_end]

        dP_tile = np.matmul(dO, V_tile.transpose(0, 1, 3, 2))
        dS_tile = P_tile * (dP_tile - D)

        dQ_buffers[t] = np.matmul(dS_tile, K_tile) * scale
        dK[:, :, sk_start:sk_end, :] = (
            np.matmul(dS_tile.transpose(0, 1, 3, 2), Q) * scale
        )
        dV[:, :, sk_start:sk_end, :] = np.matmul(P_tile.transpose(0, 1, 3, 2), dO)

    dQ = dQ_buffers.sum(axis=0)
    return dQ, dK, dV, dQ_buffers


def main():
    parser = argparse.ArgumentParser(description="Backward Deterministic Mode")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=4,
        help="Number of seqlen_k tiles for deterministic mode",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 37: Backward Pass Deterministic Mode")
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

    print(
        f"\n  Problem:   B={prob.batch} H={prob.nhead_q} S={args.seqlen} D={args.hdim}"
    )
    print(f"  Tiles:     {args.num_tiles} (seqlen_k split)")
    print(f"  Tile size: {max(1, args.seqlen // args.num_tiles)}")

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
        print("  Backward deterministic kernel: separate JIT with deterministic=True")
    else:
        print(f"  JIT build: {setup.error}")
        print("  Continuing with CPU reference only")

    # --- Generate data ---
    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

    out, P, lse = cpu_fwd_with_intermediates(Q, K, V, prob.scale)

    # --- Non-deterministic backward ---
    print("\n--- Non-Deterministic Backward ---")
    dQ_nd, dK_nd, dV_nd = cpu_bwd_nondeterministic(Q, K, V, out, dO, P, prob.scale)

    print(f"  dQ range: [{dQ_nd.min():.4e}, {dQ_nd.max():.4e}]")
    print(f"  dK range: [{dK_nd.min():.4e}, {dK_nd.max():.4e}]")
    print(f"  dV range: [{dV_nd.min():.4e}, {dV_nd.max():.4e}]")

    # --- Deterministic backward ---
    print(f"\n--- Deterministic Backward ({args.num_tiles} tiles) ---")
    dQ_det, dK_det, dV_det, dQ_bufs = cpu_bwd_deterministic(
        Q,
        K,
        V,
        out,
        dO,
        P,
        prob.scale,
        num_tiles_k=args.num_tiles,
    )

    print(f"  dQ range: [{dQ_det.min():.4e}, {dQ_det.max():.4e}]")
    print(f"  dK range: [{dK_det.min():.4e}, {dK_det.max():.4e}]")
    print(f"  dV range: [{dV_det.min():.4e}, {dV_det.max():.4e}]")
    print(f"  dQ buffers: {dQ_bufs.shape[0]} x {dQ_bufs.shape[1:]}")

    # --- Per-buffer analysis ---
    print("\n--- Per-Tile dQ Buffer Analysis ---")
    print(f"\n  {'Tile':>6} {'|buf| mean':>12} {'|buf| max':>12} {'% of total':>12}")
    print("  " + "-" * 46)

    total_l1 = float(np.abs(dQ_det).sum())
    for t in range(dQ_bufs.shape[0]):
        buf = dQ_bufs[t]
        buf_mean = float(np.abs(buf).mean())
        buf_max = float(np.abs(buf).max())
        buf_pct = float(np.abs(buf).sum()) / (total_l1 + 1e-15) * 100
        print(f"  {t:>6} {buf_mean:>12.4e} {buf_max:>12.4e} {buf_pct:>11.1f}%")

    # --- Compare deterministic vs non-deterministic ---
    print("\n--- Deterministic vs Non-Deterministic Comparison ---")
    print(f"\n  {'Grad':<6} {'Max abs diff':>14} {'Mean abs diff':>14} {'Match':>8}")
    print("  " + "-" * 46)

    for name, g_det, g_nd in [
        ("dQ", dQ_det, dQ_nd),
        ("dK", dK_det, dK_nd),
        ("dV", dV_det, dV_nd),
    ]:
        abs_diff = np.abs(g_det - g_nd)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        match = max_abs < 1e-6
        print(
            f"  {name:<6} {max_abs:>14.2e} {mean_abs:>14.2e} {'YES' if match else 'NO':>8}"
        )

    print("\n  NOTE: On CPU, both modes produce identical results.")
    print("  On GPU, non-deterministic mode uses atomicAdd for dQ,")
    print("  causing order-dependent floating-point rounding differences.")

    # --- Reproducibility test ---
    print("\n--- Reproducibility Test (Deterministic Mode) ---")
    num_runs = 5
    dQ_runs = []
    for run in range(num_runs):
        dQ_r, _, _, _ = cpu_bwd_deterministic(
            Q,
            K,
            V,
            out,
            dO,
            P,
            prob.scale,
            num_tiles_k=args.num_tiles,
        )
        dQ_runs.append(dQ_r)

    max_variation = 0.0
    for i in range(1, num_runs):
        diff = float(np.abs(dQ_runs[i] - dQ_runs[0]).max())
        max_variation = max(max_variation, diff)

    print(f"  Runs: {num_runs}")
    print(f"  Max dQ variation across runs: {max_variation:.2e}")
    print(f"  Bit-exact reproducible: {'YES' if max_variation == 0.0 else 'NO'}")

    # --- Memory overhead analysis ---
    print("\n--- Deterministic Mode Memory Overhead ---")
    dq_size = Q.nbytes
    buf_size = dQ_bufs.nbytes
    overhead = buf_size / dq_size

    print(f"  dQ single buffer:    {dq_size:>10,} bytes")
    print(f"  dQ multi-buffer:     {buf_size:>10,} bytes ({args.num_tiles} tiles)")
    print(f"  Memory overhead:     {overhead:.1f}x")
    print(f"  Extra memory:        {buf_size - dq_size:>10,} bytes")

    # --- GPU API pattern ---
    print("\n--- GPU Deterministic API Pattern ---")
    print("  Non-deterministic (default):")
    print("    FmhaKernelConfig(family='bwd', deterministic=False)")
    print("    dQ accumulated via atomicAdd (fast, non-reproducible)")
    print()
    print("  Deterministic:")
    print("    FmhaKernelConfig(family='bwd', deterministic=True)")
    print("    dQ via multi-buffer + final reduction (reproducible)")
    print("    Requires extra workspace: num_tiles_k * sizeof(dQ)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  Tiles:          {args.num_tiles}")
    print(f"  Memory overhead: {overhead:.1f}x for deterministic dQ")
    print("  Reproducible:   Deterministic mode guarantees bit-exact results")
    print("  Performance:    Deterministic ~10-20% slower on GPU (extra reduction)")
    print("  GPU:            Requires bwd-family JIT kernel")
    print("  Status:         DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
