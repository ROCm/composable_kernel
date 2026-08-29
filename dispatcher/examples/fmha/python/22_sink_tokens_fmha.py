#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 22: Sink Token Attention FMHA

Demonstrates sink token attention where the first N "sink" tokens are
always attended to regardless of the causal mask. This technique is used
in StreamingLLM and similar approaches to keep a few initial tokens as
attention anchors during long-context generation.

Mask format: t:left,right,sink -- a causal mask (top-left or bottom-right)
where the first 'sink' positions are always unmasked.

The prebuilt library does not include a sink token kernel, so this
example validates the CPU reference and shows the API pattern.

Usage:
    python3 22_sink_tokens_fmha.py
    python3 22_sink_tokens_fmha.py --sink-tokens 8
    python3 22_sink_tokens_fmha.py --seqlen 256 --window 64
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def make_causal_mask(seqlen_q: int, seqlen_k: int) -> np.ndarray:
    """Standard causal (top-left) mask: attend only to positions <= current."""
    mask = np.zeros((seqlen_q, seqlen_k), dtype=np.float32)
    for i in range(seqlen_q):
        for j in range(seqlen_k):
            if j <= i:
                mask[i, j] = 1.0
    return mask


def make_causal_sink_mask(
    seqlen_q: int,
    seqlen_k: int,
    num_sink: int,
) -> np.ndarray:
    """Causal mask with sink tokens: always attend to first num_sink positions.

    For each query position i:
      - Always attend to positions [0, num_sink)  (sink tokens)
      - Also attend to positions [j] where j <= i  (standard causal)
    """
    mask = np.zeros((seqlen_q, seqlen_k), dtype=np.float32)
    for i in range(seqlen_q):
        for j in range(seqlen_k):
            if j < num_sink or j <= i:
                mask[i, j] = 1.0
    return mask


def make_sliding_window_sink_mask(
    seqlen_q: int,
    seqlen_k: int,
    window: int,
    num_sink: int,
) -> np.ndarray:
    """Sliding window mask with sink tokens.

    For each query position i:
      - Always attend to positions [0, num_sink)  (sink tokens)
      - Attend to positions in [i - window + 1, i]  (sliding window)
    """
    mask = np.zeros((seqlen_q, seqlen_k), dtype=np.float32)
    for i in range(seqlen_q):
        for j in range(seqlen_k):
            if j < num_sink or (i - window + 1 <= j <= i):
                mask[i, j] = 1.0
    return mask


def cpu_attention_fwd_masked(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    mask: np.ndarray,
) -> np.ndarray:
    """CPU reference: attention with explicit mask.

    Args:
        Q: [batch, nhead_q, seqlen_q, hdim_q]  float32
        K: [batch, nhead_k, seqlen_k, hdim_q]  float32
        V: [batch, nhead_k, seqlen_k, hdim_v]  float32
        scale: softmax scale
        mask: [seqlen_q, seqlen_k] binary mask (1=attend, 0=ignore)

    Returns:
        O: [batch, nhead_q, seqlen_q, hdim_v]  float32
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)

    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    neg_inf = np.finfo(np.float32).min
    S = np.where(mask[np.newaxis, np.newaxis, :, :] > 0, S, neg_inf)

    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


def print_mask(mask: np.ndarray, name: str, max_display: int = 16):
    """Print a small portion of a mask for visualization."""
    rows, cols = mask.shape
    rows_show = min(rows, max_display)
    cols_show = min(cols, max_display)
    print(f"\n  {name} ({rows}x{cols}, showing {rows_show}x{cols_show}):")
    for i in range(rows_show):
        row_str = "".join("1" if mask[i, j] > 0 else "." for j in range(cols_show))
        print(f"    q{i:02d}: {row_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Sink Token Attention FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--sink-tokens", type=int, default=4, help="Number of sink tokens"
    )
    parser.add_argument("--window", type=int, default=32, help="Sliding window size")
    args = parser.parse_args()

    print("=" * 70)
    print("Example 22: Sink Token Attention FMHA")
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

    # Step 1: Visualize mask patterns
    print("\nStep 1: Mask Patterns")

    causal = make_causal_mask(sq, sk)
    causal_sink = make_causal_sink_mask(sq, sk, args.sink_tokens)
    window_sink = make_sliding_window_sink_mask(sq, sk, args.window, args.sink_tokens)

    vis_size = min(16, sq)
    print_mask(causal[:vis_size, :vis_size], "Causal (standard)", vis_size)
    print_mask(
        causal_sink[:vis_size, :vis_size],
        f"Causal + {args.sink_tokens} sink tokens",
        vis_size,
    )
    print_mask(
        window_sink[:vis_size, :vis_size],
        f"Window({args.window}) + {args.sink_tokens} sink tokens",
        vis_size,
    )

    # Step 2: CPU reference for each mask type
    print("\n\nStep 2: CPU Reference Comparison")

    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float32)

    O_no_mask = cpu_attention_fwd(Q, K, V, prob.scale)
    O_causal = cpu_attention_fwd_masked(Q, K, V, prob.scale, causal)
    O_causal_sink = cpu_attention_fwd_masked(Q, K, V, prob.scale, causal_sink)
    O_window_sink = cpu_attention_fwd_masked(Q, K, V, prob.scale, window_sink)

    masks_and_outputs = [
        ("No mask", O_no_mask),
        ("Causal", O_causal),
        (f"Causal+sink({args.sink_tokens})", O_causal_sink),
        (f"Window({args.window})+sink({args.sink_tokens})", O_window_sink),
    ]

    print(f"\n  {'Mask Type':<30} {'Output Range':>20} {'vs NoMask MaxDiff':>18}")
    print("  " + "-" * 70)
    for name, out in masks_and_outputs:
        d = np.abs(out - O_no_mask).max()
        out_range = f"[{out.min():.4f}, {out.max():.4f}]"
        print(f"  {name:<30} {out_range:>20} {d:>18.6e}")

    # Step 3: Verify sink tokens effect
    print("\nStep 3: Sink Token Effect Analysis")

    diff_causal_vs_sink = np.abs(O_causal - O_causal_sink)
    print("  Causal vs Causal+Sink:")
    print(f"    Max diff:  {diff_causal_vs_sink.max():.6e}")
    print(f"    Mean diff: {diff_causal_vs_sink.mean():.6e}")

    n_attend_causal = causal.sum()
    n_attend_sink = causal_sink.sum()
    n_attend_window = window_sink.sum()
    print("\n  Attention density:")
    print(
        f"    Causal:             {n_attend_causal:>8.0f} / {sq * sk} ({100 * n_attend_causal / (sq * sk):.1f}%)"
    )
    print(
        f"    Causal+sink:        {n_attend_sink:>8.0f} / {sq * sk} ({100 * n_attend_sink / (sq * sk):.1f}%)"
    )
    print(
        f"    Window+sink:        {n_attend_window:>8.0f} / {sq * sk} ({100 * n_attend_window / (sq * sk):.1f}%)"
    )

    # Step 4: Sweep sink token count
    print("\nStep 4: Sink Token Sweep")

    sink_counts = [0, 1, 2, 4, 8, 16]
    validator = FmhaValidator(rtol=1e-4, atol=1e-4)

    print(
        f"\n  {'Sinks':>6} {'Density':>10} {'vs Causal MaxDiff':>20} {'vs NoMask MaxDiff':>20}"
    )
    print("  " + "-" * 60)

    for ns in sink_counts:
        if ns > sk:
            continue
        m = make_causal_sink_mask(sq, sk, ns)
        O_s = cpu_attention_fwd_masked(Q, K, V, prob.scale, m)
        d_causal = np.abs(O_s - O_causal).max()
        d_nomask = np.abs(O_s - O_no_mask).max()
        density = 100 * m.sum() / (sq * sk)
        print(f"  {ns:>6} {density:>9.1f}% {d_causal:>20.6e} {d_nomask:>20.6e}")

    # Step 5: GPU API pattern
    print("\nStep 5: GPU Kernel Pattern")
    print("  NOTE: The prebuilt library does not include a sink token kernel.")
    print("  To compile a sink-enabled kernel, use:")
    print()
    print("    FmhaSignature()")
    print("        .mask('top_left')  // causal mask required with sink")
    print("        .sink(true)        // enable sink tokens")
    print()
    print("  At runtime, pass sink count via the mask spec: 't:left,right,sink'")
    print(
        f"  Example: 't:0,0,{args.sink_tokens}' for causal + {args.sink_tokens} sink tokens"
    )

    # Step 6: GPU baseline (no mask, no sink)
    print("\nStep 6: GPU Baseline (standard kernel, no mask)")

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

        Q_f16 = Q.astype(np.float16)
        K_f16 = K.astype(np.float16)
        V_f16 = V.astype(np.float16)

        result = runner.run(Q_f16, K_f16, V_f16, prob)
        if result.success:
            ok, max_abs, _ = validator.check(result.output, O_no_mask)
            print(
                f"  GPU (no mask): time={result.time_ms:.4f}ms  TFLOPS={result.tflops:.2f}  "
                f"max_err={max_abs:.2e}  {'PASS' if ok else 'FAIL'}"
            )
        else:
            print(f"  GPU error: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("  Sink token attention: first N tokens always attended regardless of mask")
    print("  Use case: StreamingLLM, long-context generation with attention anchors")
    print("  Sink tokens preserve global context that causal masking would discard")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
