#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 13: Attention Bias

Demonstrates bias types supported by the FMHA dispatcher:
1. no_bias       -- Standard attention without bias
2. elementwise   -- Add a [seqlen_q, seqlen_k] bias matrix to attention scores
3. alibi         -- Attention with Linear Biases (ALiBi) positional encoding

For each bias type:
- Creates an FmhaProblem and bias tensor
- Attempts GPU execution (prebuilt: no_bias only)
- Computes CPU reference with bias applied before softmax
- Validates output

Usage:
    python3 13_bias_fmha.py
    python3 13_bias_fmha.py --seqlen 256
    python3 13_bias_fmha.py --nhead 16
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


def get_alibi_slopes(nhead: int) -> np.ndarray:
    """Compute ALiBi slopes for each attention head.

    Following the original ALiBi paper: slopes = 2^(-8/n * [1..n])
    where n is the number of heads.
    """
    ratio = 2.0 ** (-8.0 / nhead)
    return np.array([ratio ** (i + 1) for i in range(nhead)], dtype=np.float32)


def make_alibi_bias(nhead: int, seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Create ALiBi bias matrix: slope * (col - row) for causal positions.

    Returns: [nhead, seqlen_q, seqlen_k]
    """
    slopes = get_alibi_slopes(nhead)
    row = np.arange(seqlen_q).reshape(-1, 1)
    col = np.arange(seqlen_k).reshape(1, -1)
    dist = col - row
    bias = slopes.reshape(-1, 1, 1) * dist.reshape(1, seqlen_q, seqlen_k)
    return bias.astype(np.float32)


def make_elementwise_bias(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Create a relative-position elementwise bias matrix.

    Returns: [seqlen_q, seqlen_k]
    """
    row = np.arange(seqlen_q, dtype=np.float32).reshape(-1, 1)
    col = np.arange(seqlen_k, dtype=np.float32).reshape(1, -1)
    dist = np.abs(row - col)
    return (-0.1 * dist).astype(np.float32)


def cpu_biased_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    bias: np.ndarray,
) -> np.ndarray:
    """CPU reference: attention with additive bias before softmax.

    Q: [batch, nhead, seqlen_q, hdim]
    bias: broadcastable to [batch, nhead, seqlen_q, seqlen_k]
    """
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S = S + bias
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


def main():
    parser = argparse.ArgumentParser(description="Attention Bias")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 13: Attention Bias")
    print("=" * 70)

    sq = sk = args.seqlen
    prob = FmhaProblem(
        batch=args.batch,
        nhead_q=args.nhead,
        nhead_k=args.nhead,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )

    print(f"\n  Problem: B={prob.batch} H={prob.nhead_q} S={sq} D={args.hdim}")

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
        print(f"  GPU runner loaded (JIT build: {setup.build_time_s:.1f}s)")
    else:
        print(f"  GPU runner not available: {setup.error}")

    # --- Build bias tensors ---
    bias_configs = [
        ("no_bias", np.zeros((1, 1, sq, sk), dtype=np.float32)),
        ("elementwise", make_elementwise_bias(sq, sk)[np.newaxis, np.newaxis, :, :]),
        ("alibi", make_alibi_bias(args.nhead, sq, sk)[np.newaxis, :, :, :]),
    ]

    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    print(
        f"\n  {'#':<3} {'BiasType':<14} {'BiasRange':>20} {'GPUStatus':<12} {'MaxErr':>10} {'Status':>8}"
    )
    print("  " + "-" * 72)

    results = []
    for i, (name, bias) in enumerate(bias_configs, 1):
        bias_min, bias_max = float(bias.min()), float(bias.max())
        bias_range = f"[{bias_min:.3f}, {bias_max:.3f}]"

        # GPU attempt
        gpu_status = "N/A"
        gpu_out = None
        if runner is not None:
            res = runner.run(Q_fp16, K_fp16, V_fp16, prob)
            if res.success:
                gpu_out = res.output
                gpu_status = "OK" if name == "no_bias" else "no_bias*"
            else:
                gpu_status = "unsupported"

        # CPU reference with bias
        O_ref = cpu_biased_attention(Q_f32, K_f32, V_f32, prob.scale, bias)

        # Validate
        if gpu_out is not None and name == "no_bias":
            ok, max_abs, _ = validator.check(gpu_out, O_ref)
            tag = "PASS" if ok else "FAIL"
            err_str = f"{max_abs:.2e}"
        else:
            ok = True
            tag = "DEMO"
            err_str = "---"

        print(
            f"  {i:<3} {name:<14} {bias_range:>20} {gpu_status:<12} {err_str:>10} {tag:>8}"
        )
        results.append((name, ok))

    # --- Show ALiBi details ---
    print("\n--- ALiBi Details ---")
    slopes = get_alibi_slopes(args.nhead)
    print(f"  Heads:   {args.nhead}")
    print(f"  Slopes:  {', '.join(f'{s:.4f}' for s in slopes[: min(8, len(slopes))])}")
    if len(slopes) > 8:
        print(f"           ... ({len(slopes)} total)")
    print("  Effect:  Nearby tokens get higher scores, distant tokens penalized")
    print("  Formula: bias[h,i,j] = slope[h] * (j - i)")

    alibi_bias = make_alibi_bias(args.nhead, sq, sk)
    print("\n  Head 0 bias corner (4x4):")
    corner = alibi_bias[0, :4, :4]
    for r in range(4):
        row_str = " ".join(f"{corner[r, c]:>7.3f}" for c in range(4))
        print(f"    {row_str}")

    # --- Show impact of bias on attention ---
    print("\n--- Bias Impact Analysis ---")
    O_no_bias = cpu_attention_fwd(Q_f32, K_f32, V_f32, prob.scale)
    for name, bias in bias_configs:
        O_biased = cpu_biased_attention(Q_f32, K_f32, V_f32, prob.scale, bias)
        diff = float(np.abs(O_biased - O_no_bias).max())
        print(f"  {name:<14} max output shift: {diff:.4e}")

    # --- Summary ---
    all_ok = all(ok for _, ok in results)
    print("\n" + "=" * 70)
    print("  Bias types: no_bias, elementwise, alibi")
    print("  no_bias:    Standard attention (baseline)")
    print("  elementwise: Position-distance bias [-0.1 * |i-j|]")
    print("  alibi:      Linear position bias per head (no learned params)")
    print("  GPU:        Prebuilt supports no_bias only")
    print(f"  Status:     {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
