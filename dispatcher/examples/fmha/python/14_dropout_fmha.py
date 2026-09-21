#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 14: Attention Dropout with LSE

Demonstrates:
1. Dropout applied to attention probabilities
2. Log-sum-exp (LSE) storage for numerical stability
3. Statistical validation (dropout is stochastic)
4. Reproducibility with seed control

Dropout zeros out attention weights with probability p_drop, then scales
remaining weights by 1/(1-p_drop) to preserve expected value.
LSE stores log(sum(exp(scores))) per query position for backward pass.

Usage:
    python3 14_dropout_fmha.py
    python3 14_dropout_fmha.py --p-drop 0.3
    python3 14_dropout_fmha.py --seqlen 256 --seed 123
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


def cpu_attention_with_dropout(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    p_drop: float,
    seed: int,
) -> tuple:
    """CPU reference: attention with dropout and LSE output.

    Returns:
        (O, P_dropped, lse)
        O:         [batch, nhead, seqlen_q, hdim_v]
        P_dropped: [batch, nhead, seqlen_q, seqlen_k]  attention weights after dropout
        lse:       [batch, nhead, seqlen_q]  log-sum-exp of scores
    """
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum

    lse = (np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)).astype(np.float32)

    rng = np.random.RandomState(seed)
    drop_mask = (rng.rand(*P.shape) >= p_drop).astype(np.float32)
    scale_factor = 1.0 / (1.0 - p_drop) if p_drop < 1.0 else 0.0
    P_dropped = P * drop_mask * scale_factor

    out = np.matmul(P_dropped, V)
    return out, P_dropped, lse


def main():
    parser = argparse.ArgumentParser(description="Attention Dropout with LSE")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--p-drop", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 14: Attention Dropout with LSE")
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
    print(f"  p_drop:    {args.p_drop}")
    print(f"  Seed:      {args.seed}")
    print(f"  LSE shape: [{prob.batch}, {prob.nhead_q}, {prob.seqlen_q}]")

    # --- Generate data ---
    np.random.seed(args.seed)
    Q_f32 = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K_f32 = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V_f32 = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    Q_fp16 = Q_f32.astype(np.float16)
    K_fp16 = K_f32.astype(np.float16)
    V_fp16 = V_f32.astype(np.float16)

    # --- GPU execution attempt ---
    print("\n--- GPU Execution ---")
    gpu_output = None
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
        res = runner.run(Q_fp16, K_fp16, V_fp16, prob)
        if res.success:
            gpu_output = res.output
            print(f"  GPU (no dropout): {res.time_ms:.4f} ms, {res.tflops:.2f} TFLOPS")
            print("  Note: JIT kernel runs without dropout; shown for baseline")
        else:
            print("  GPU: Kernel returned failure")

    # --- CPU reference: no dropout (baseline) ---
    print("\n--- CPU Reference ---")
    O_no_drop = cpu_attention_fwd(Q_f32, K_f32, V_f32, prob.scale)

    # --- CPU reference: with dropout ---
    drop_rates = [0.0, 0.1, args.p_drop, 0.5]

    print(
        f"\n  {'p_drop':>8} {'OutMean':>10} {'OutStd':>10} {'MaxDiff':>10} {'DropFrac':>10}"
    )
    print("  " + "-" * 52)

    for p in drop_rates:
        O_drop, P_dropped, lse = cpu_attention_with_dropout(
            Q_f32,
            K_f32,
            V_f32,
            prob.scale,
            p,
            args.seed,
        )

        total_weights = P_dropped.size
        zeros = (P_dropped == 0).sum()
        actual_drop_frac = zeros / total_weights

        diff = float(np.abs(O_drop - O_no_drop).max())
        print(
            f"  {p:>8.2f} {O_drop.mean():>10.4f} {O_drop.std():>10.4f} "
            f"{diff:>10.2e} {actual_drop_frac:>10.2%}"
        )

    # --- LSE analysis ---
    print("\n--- LSE (Log-Sum-Exp) Analysis ---")
    _, _, lse = cpu_attention_with_dropout(
        Q_f32,
        K_f32,
        V_f32,
        prob.scale,
        args.p_drop,
        args.seed,
    )
    print(f"  LSE shape:  {lse.shape}")
    print(f"  LSE range:  [{lse.min():.4f}, {lse.max():.4f}]")
    print(f"  LSE mean:   {lse.mean():.4f}")
    print("  LSE is independent of dropout (computed from raw scores)")

    lse_nodrop = cpu_attention_with_dropout(
        Q_f32,
        K_f32,
        V_f32,
        prob.scale,
        0.0,
        args.seed,
    )[2]
    lse_diff = float(np.abs(lse - lse_nodrop).max())
    print(f"  LSE diff (drop vs no-drop): {lse_diff:.2e} (should be 0)")

    # --- Statistical validation ---
    print("\n--- Statistical Validation ---")
    n_trials = 5
    outputs = []
    for trial in range(n_trials):
        O_t, _, _ = cpu_attention_with_dropout(
            Q_f32,
            K_f32,
            V_f32,
            prob.scale,
            args.p_drop,
            args.seed + trial,
        )
        outputs.append(O_t)

    O_mean = np.mean(outputs, axis=0)
    O_std = np.std(outputs, axis=0)

    mean_diff = float(np.abs(O_mean - O_no_drop).max())
    max_std = float(O_std.max())

    print(f"  Trials:             {n_trials}")
    print(f"  Mean vs no-drop:    {mean_diff:.4e} (should be small)")
    print(f"  Max output stddev:  {max_std:.4e}")
    print("  E[dropout(P)] = P  (unbiased estimator)")

    if gpu_output is not None:
        validator = FmhaValidator(rtol=1e-2, atol=1e-2)
        ok, max_abs, _ = validator.check(gpu_output, O_no_drop)
        print(
            f"\n  GPU vs CPU (no-drop): max_err={max_abs:.2e}, {'PASS' if ok else 'FAIL'}"
        )

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  Dropout:   p_drop={args.p_drop}, seed={args.seed}")
    print(
        f"  LSE:       Stored for backward pass (shape [{prob.batch},{prob.nhead_q},{prob.seqlen_q}])"
    )
    print("  Key:       Dropout is stochastic; validate statistically, not exactly")
    print("  GPU:       Prebuilt kernel does not support dropout")
    print("  Status:    DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
