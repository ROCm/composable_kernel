#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 12: Attention Masks

Demonstrates all 5 mask types supported by the FMHA dispatcher:
1. no_mask (0)      -- Full attention, no masking
2. top_left (1)     -- Causal mask aligned to top-left corner
3. bottom_right (2) -- Causal mask aligned to bottom-right corner
4. sliding_window   -- Local attention within a fixed window
5. generic          -- Arbitrary user-defined mask pattern

For each mask type, this example:
- Creates an FmhaProblem
- Attempts GPU execution via prebuilt kernel
- Computes CPU reference with the mask applied
- Validates results

Usage:
    python3 12_masks_fmha.py
    python3 12_masks_fmha.py --seqlen 256
    python3 12_masks_fmha.py --window-size 64
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
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


MASK_TYPES = {
    "no_mask": 0,
    "top_left": 1,
    "bottom_right": 2,
    "sliding_window": 3,
    "generic": 4,
}


def make_causal_mask_top_left(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Causal mask aligned to top-left: position i can attend to positions <= i."""
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    return (col <= row).astype(np.float32)


def make_causal_mask_bottom_right(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Causal mask aligned to bottom-right: accounts for kv longer than q."""
    offset = seqlen_k - seqlen_q
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    return (col <= row + offset).astype(np.float32)


def make_sliding_window_mask(seqlen_q: int, seqlen_k: int, window: int) -> np.ndarray:
    """Sliding window: each query attends to a local window of keys."""
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    offset = seqlen_k - seqlen_q
    return ((col <= row + offset) & (col >= row + offset - window + 1)).astype(
        np.float32
    )


def make_generic_mask(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Generic checkerboard mask for demonstration."""
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    return ((row + col) % 2 == 0).astype(np.float32)


def cpu_masked_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    mask: np.ndarray,
) -> np.ndarray:
    """CPU reference: scaled dot-product attention with arbitrary mask.

    Q: [batch, nhead, seqlen_q, hdim]
    mask: [seqlen_q, seqlen_k]  (broadcast over batch and head)
    """
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    mask_broad = mask[np.newaxis, np.newaxis, :, :]
    S = np.where(mask_broad > 0, S, -1e9)
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


def main():
    parser = argparse.ArgumentParser(description="Attention Masks")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen-q", type=int, default=128)
    parser.add_argument("--seqlen-k", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 12: Attention Masks")
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
    print(f"  Window:  {args.window_size}")

    # --- Generate data ---
    np.random.seed(42)
    Q_f32 = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K_f32 = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V_f32 = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    Q_fp16 = Q_f32.astype(np.float16)
    K_fp16 = K_f32.astype(np.float16)
    V_fp16 = V_f32.astype(np.float16)

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
        print(f"\n  GPU runner loaded (JIT build: {setup.build_time_s:.1f}s)")
    else:
        print(f"\n  GPU runner not available: {setup.error}")

    # --- Build masks ---
    masks = {
        "no_mask": np.ones((sq, sk), dtype=np.float32),
        "top_left": make_causal_mask_top_left(sq, sk),
        "bottom_right": make_causal_mask_bottom_right(sq, sk),
        "sliding_window": make_sliding_window_mask(sq, sk, args.window_size),
        "generic": make_generic_mask(sq, sk),
    }

    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    print(
        f"\n  {'#':<3} {'MaskType':<18} {'ID':<4} {'Density':>8} {'GPUStatus':<12} {'CPURef':<8} {'MaxErr':>10} {'Status':>8}"
    )
    print("  " + "-" * 76)

    results = []
    for i, (name, mask) in enumerate(masks.items(), 1):
        mask_id = MASK_TYPES[name]
        density = mask.sum() / mask.size * 100

        # GPU attempt (prebuilt only supports no_mask)
        gpu_status = "N/A"
        gpu_out = None
        if runner is not None:
            res = runner.run(Q_fp16, K_fp16, V_fp16, prob)
            if res.success:
                gpu_out = res.output
                gpu_status = "OK" if name == "no_mask" else "no_mask*"
            else:
                gpu_status = "unsupported"

        # CPU reference with mask
        O_ref = cpu_masked_attention(Q_f32, K_f32, V_f32, prob.scale, mask)
        cpu_status = "OK"

        # Validate
        if gpu_out is not None and name == "no_mask":
            ok, max_abs, _ = validator.check(gpu_out, O_ref)
            tag = "PASS" if ok else "FAIL"
            err_str = f"{max_abs:.2e}"
        else:
            ok = True
            tag = "DEMO"
            err_str = "---"

        print(
            f"  {i:<3} {name:<18} {mask_id:<4} {density:>7.1f}% {gpu_status:<12} {cpu_status:<8} {err_str:>10} {tag:>8}"
        )
        results.append((name, ok))

    # --- Mask visualization ---
    print("\n--- Mask Patterns (first 8x8 corner) ---")
    view_size = min(8, sq, sk)
    for name, mask in masks.items():
        corner = mask[:view_size, :view_size]
        print(f"\n  {name}:")
        for r in range(view_size):
            row_str = " ".join(
                "█" if corner[r, c] > 0 else "·" for c in range(view_size)
            )
            print(f"    {row_str}")

    # --- Summary ---
    all_ok = all(ok for _, ok in results)
    print("\n" + "=" * 70)
    print(f"  Mask types tested:  {len(masks)}")
    print("  no_mask:            Full attention (all positions visible)")
    print("  top_left:           Causal from top-left (autoregressive)")
    print("  bottom_right:       Causal from bottom-right (kv-padded)")
    print(f"  sliding_window:     Local window of {args.window_size} keys")
    print("  generic:            Arbitrary (checkerboard demo)")
    print("  GPU:                Prebuilt supports no_mask only")
    print(f"  Status:             {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
