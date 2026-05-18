#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 38: Backward Pass Head Dimension Sweep

Sweeps hdim for the backward pass: 32, 64, 128, 256.

Each hdim requires a dedicated compiled kernel because the tile
dimensions (tile_k0max, tile_n1) must match the head dimension.
This example shows which hdims the backward kernels can support
and computes CPU reference gradients for each.

Backward kernel tile requirements per hdim:
  hdim=32:  tile_k0max=32,  tile_n1=32   (small, fast compile)
  hdim=64:  tile_k0max=64,  tile_n1=64
  hdim=128: tile_k0max=128, tile_n1=128  (standard LLM config)
  hdim=256: tile_k0max=256, tile_n1=256  (large, slow compile)

Fixed: batch=2, nhead=8, seqlen=64

Usage:
    python3 38_bwd_sweep_hdim_fmha.py
    python3 38_bwd_sweep_hdim_fmha.py --arch gfx942
    python3 38_bwd_sweep_hdim_fmha.py --seqlen 128
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
    cpu_attention_bwd,
)

HDIMS = [32, 64, 128, 256]
BATCH = 2
NHEAD = 8


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


def bwd_flops(prob: FmhaProblem) -> int:
    """Backward FLOPS (~4x forward)."""
    return 4 * prob.num_ops


def main():
    parser = argparse.ArgumentParser(description="Backward Head Dimension Sweep")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--seqlen", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 38: Backward Pass Head Dimension Sweep")
    print("=" * 70)

    print(f"\n  Fixed:  batch={BATCH}, nhead={NHEAD}, seqlen={args.seqlen}")
    print(f"  Sweep:  hdim in {HDIMS}")
    print(f"  Arch:   {args.arch}")

    # --- JIT compile a basic fp16 h128 fwd kernel ---
    print("\n--- JIT Compilation (hdim=128 fwd kernel) ---")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=128,
        hdim_v=128,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if setup.success:
        print(f"  Fwd kernel compiled: {setup.build_time_s:.1f}s")
        print("  Backward kernels for each hdim need separate JIT compilation")
    else:
        print(f"  JIT build: {setup.error}")
        print("  Continuing with CPU reference only")

    # --- Kernel tile requirements per hdim ---
    print("\n--- Backward Kernel Tile Requirements ---")
    print(
        f"\n  {'hdim':>6} | {'tile_k0max':>10} {'tile_n1':>8} {'tile_k0':>8}"
        f" | {'scale':>8} | {'Status'}"
    )
    print("  " + "-" * 62)

    for hdim in HDIMS:
        tile_k0 = min(32, hdim)
        bwd_status = "needs bwd JIT"
        if hdim == 128 and setup.success:
            bwd_status = "fwd only (JIT)"
        scale = 1.0 / (hdim**0.5)
        print(
            f"  {hdim:>6} | {hdim:>10} {hdim:>8} {tile_k0:>8}"
            f" | {scale:>8.4f} | {bwd_status}"
        )

    # --- CPU backward for each hdim ---
    print("\n--- CPU Backward Reference per Head Dimension ---")
    print(
        f"\n  {'hdim':>6} | {'FWD ops':>12} {'BWD ops':>12}"
        f" | {'|dQ| mean':>10} {'|dK| mean':>10} {'|dV| mean':>10}"
        f" | {'Time(ms)':>10} {'Finite':>6}"
    )
    print("  " + "-" * 96)

    all_results = {}

    for hdim in HDIMS:
        prob = FmhaProblem(
            batch=BATCH,
            nhead_q=NHEAD,
            nhead_k=NHEAD,
            seqlen_q=args.seqlen,
            seqlen_k=args.seqlen,
            hdim_q=hdim,
            hdim_v=hdim,
        )

        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
        dO = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

        out, P, lse = cpu_fwd_with_intermediates(Q, K, V, prob.scale)

        t0 = time.perf_counter()
        dQ, dK, dV = cpu_attention_bwd(Q, K, V, out, dO, P, prob.scale)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        is_finite = bool(
            np.all(np.isfinite(dQ))
            and np.all(np.isfinite(dK))
            and np.all(np.isfinite(dV))
        )
        fwd_ops = prob.num_ops
        bwd_ops = bwd_flops(prob)

        print(
            f"  {hdim:>6} | {fwd_ops:>12,} {bwd_ops:>12,}"
            f" | {np.abs(dQ).mean():>10.4e} {np.abs(dK).mean():>10.4e}"
            f" {np.abs(dV).mean():>10.4e}"
            f" | {elapsed_ms:>10.4f} {'OK' if is_finite else 'NaN!':>6}"
        )
        all_results[hdim] = (dQ, dK, dV, out, P, Q, K, V, dO, prob)

    # --- Gradient norms vs hdim ---
    print("\n--- Gradient L2 Norms vs Head Dimension ---")
    print(
        f"\n  {'hdim':>6} | {'||dQ||_2':>12} {'||dK||_2':>12} {'||dV||_2':>12} | {'ratio dQ/dK':>12}"
    )
    print("  " + "-" * 62)

    for hdim in HDIMS:
        dQ, dK, dV, *_ = all_results[hdim]
        l2_dq = float(np.sqrt((dQ**2).sum()))
        l2_dk = float(np.sqrt((dK**2).sum()))
        l2_dv = float(np.sqrt((dV**2).sum()))
        ratio = l2_dq / (l2_dk + 1e-12)
        print(
            f"  {hdim:>6} | {l2_dq:>12.4e} {l2_dk:>12.4e} {l2_dv:>12.4e} | {ratio:>12.2f}"
        )

    # --- Scale effect analysis ---
    print("\n--- Scale Effect on Gradients ---")
    print("  scale = 1/sqrt(hdim) -> larger hdim = smaller scale")
    print("  This affects gradient magnitude through the dS = P * (dP - D) term.\n")

    print(f"  {'hdim':>6} {'scale':>10} {'dQ max':>12} {'dK max':>12} {'dV max':>12}")
    print("  " + "-" * 52)

    for hdim in HDIMS:
        dQ, dK, dV, *_ = all_results[hdim]
        scale = 1.0 / (hdim**0.5)
        print(
            f"  {hdim:>6} {scale:>10.4f} {np.abs(dQ).max():>12.4e}"
            f" {np.abs(dK).max():>12.4e} {np.abs(dV).max():>12.4e}"
        )

    # --- FP16 quantization impact per hdim ---
    print("\n--- FP16 Backward Quantization Impact ---")
    print(
        f"\n  {'hdim':>6} | {'dQ fp16 err':>12} {'dK fp16 err':>12} {'dV fp16 err':>12}"
    )
    print("  " + "-" * 50)

    for hdim in HDIMS:
        dQ, dK, dV, *_ = all_results[hdim]
        dq_err = float(np.abs(dQ - dQ.astype(np.float16).astype(np.float32)).max())
        dk_err = float(np.abs(dK - dK.astype(np.float16).astype(np.float32)).max())
        dv_err = float(np.abs(dV - dV.astype(np.float16).astype(np.float32)).max())
        print(f"  {hdim:>6} | {dq_err:>12.2e} {dk_err:>12.2e} {dv_err:>12.2e}")

    # --- Backward GPU API pattern per hdim ---
    print("\n--- Backward GPU Kernel Config per hdim ---")
    for hdim in HDIMS:
        print(f"\n  hdim={hdim}:")
        print("    FmhaKernelConfig(")
        print("      family='bwd', data_type='fp16',")
        print(f"      hdim_q={hdim}, hdim_v={hdim},")
        print(f"      tile_k0max={hdim}, tile_n1={hdim},")
        print(f"      tile_k0={min(32, hdim)}, tile_k1={min(32, hdim)},")
        print("    )")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  Head dims swept:  {HDIMS}")
    print(f"  Fixed:            B={BATCH} H={NHEAD} S={args.seqlen}")
    print("  Scale effect:     1/sqrt(hdim) -> smaller gradients for larger hdim")
    print("  Tile coupling:    tile_k0max and tile_n1 must equal hdim")
    print("  GPU:              Each hdim needs a dedicated bwd-family kernel")
    print("  Status:           DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
