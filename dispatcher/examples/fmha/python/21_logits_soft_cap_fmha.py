#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 21: Logits Soft Cap FMHA

Demonstrates the logits soft cap feature, which prevents attention logits
from growing unboundedly by applying: tanh(scores / soft_cap) * soft_cap
before the softmax. This technique is used in models like Gemma-2 to
stabilize training at large scale.

The prebuilt library does not include a logits_soft_cap kernel, so this
example validates the CPU reference implementation and shows the API
pattern for when a compiled kernel with logits=True is available.

Usage:
    python3 21_logits_soft_cap_fmha.py
    python3 21_logits_soft_cap_fmha.py --soft-cap 30.0
    python3 21_logits_soft_cap_fmha.py --seqlen 256
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


def cpu_attention_fwd_logits_soft_cap(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    soft_cap: float,
) -> np.ndarray:
    """CPU reference: attention with logits soft cap.

    Before softmax, scores are clamped via:
        scores = tanh(scores / soft_cap) * soft_cap

    Args:
        Q: [batch, nhead_q, seqlen_q, hdim_q]  float32
        K: [batch, nhead_k, seqlen_k, hdim_q]  float32
        V: [batch, nhead_k, seqlen_k, hdim_v]  float32
        scale: softmax scaling factor (1/sqrt(hdim_q))
        soft_cap: logits soft cap value (e.g. 50.0)

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
    S = np.tanh(S / soft_cap) * soft_cap

    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


def show_soft_cap_effect(scale: float, soft_cap: float):
    """Visualize the clamping effect of logits soft cap on score magnitudes."""
    raw_scores = np.array(
        [-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100], dtype=np.float32
    )
    scaled = raw_scores * scale
    capped = np.tanh(scaled / soft_cap) * soft_cap

    print(f"\n  Soft cap effect (scale={scale:.4f}, soft_cap={soft_cap:.1f}):")
    print(
        f"  {'Raw Score':>12} {'After Scale':>14} {'After Cap':>12} {'Reduction':>12}"
    )
    print("  " + "-" * 54)
    for r, s, c in zip(raw_scores, scaled, capped):
        reduction = abs(s) - abs(c) if abs(s) > 0 else 0
        print(f"  {r:>12.1f} {s:>14.4f} {c:>12.4f} {reduction:>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Logits Soft Cap FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 21_logits_soft_cap_fmha.py                  # Default soft_cap=50
  python3 21_logits_soft_cap_fmha.py --soft-cap 30.0  # Tighter cap
  python3 21_logits_soft_cap_fmha.py --seqlen 256
        """,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--soft-cap", type=float, default=50.0, help="Logits soft cap value"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 21: Logits Soft Cap FMHA")
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

    # Step 1: Demonstrate the soft cap transformation
    print("\nStep 1: Soft Cap Transformation")
    show_soft_cap_effect(prob.scale, args.soft_cap)

    # Step 2: CPU reference comparison -- with vs without soft cap
    print("\nStep 2: CPU Reference (with vs without soft cap)")

    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float32)

    O_no_cap = cpu_attention_fwd(Q, K, V, prob.scale)
    O_capped = cpu_attention_fwd_logits_soft_cap(Q, K, V, prob.scale, args.soft_cap)

    diff = np.abs(O_no_cap - O_capped)
    print(f"\n  Shape:     {prob.q_shape()}")
    print(f"  Soft cap:  {args.soft_cap}")
    print(f"  Output range (no cap):  [{O_no_cap.min():.4f}, {O_no_cap.max():.4f}]")
    print(f"  Output range (capped):  [{O_capped.min():.4f}, {O_capped.max():.4f}]")
    print(f"  Max diff (cap effect):  {diff.max():.6e}")
    print(f"  Mean diff (cap effect): {diff.mean():.6e}")

    # Step 3: Validate across different soft_cap values
    print("\nStep 3: Soft Cap Sweep")

    soft_cap_values = [10.0, 20.0, 30.0, 50.0, 100.0, 500.0]
    validator = FmhaValidator(rtol=1e-4, atol=1e-4)

    print(
        f"\n  {'SoftCap':>10} {'OutRange':>20} {'vs NoCap MaxDiff':>18} {'vs NoCap MeanDiff':>18}"
    )
    print("  " + "-" * 70)

    for sc in soft_cap_values:
        O_sc = cpu_attention_fwd_logits_soft_cap(Q, K, V, prob.scale, sc)
        d = np.abs(O_no_cap - O_sc)
        out_range = f"[{O_sc.min():.4f}, {O_sc.max():.4f}]"
        print(f"  {sc:>10.1f} {out_range:>20} {d.max():>18.6e} {d.mean():>18.6e}")

    # Step 4: Self-consistency -- large soft_cap should approach no-cap result
    print("\nStep 4: Self-Consistency Check")

    O_large_cap = cpu_attention_fwd_logits_soft_cap(Q, K, V, prob.scale, 1e6)
    ok, max_abs, _ = validator.check(O_large_cap, O_no_cap)
    print(
        f"  soft_cap=1e6 vs no_cap: max_err={max_abs:.2e} -> {'PASS' if ok else 'FAIL'}"
    )

    # Step 5: GPU API pattern (requires logits=True kernel)
    print("\nStep 5: GPU Kernel Pattern")
    print("  NOTE: The prebuilt library does not include a logits_soft_cap kernel.")
    print("  To run on GPU, compile a kernel with logits=True in the signature:")
    print()
    print("    config = FmhaKernelConfig(")
    print("        family='fwd', data_type='fp16', hdim_q=128, hdim_v=128,")
    print("        pipeline='qr_async',")
    print("    )")
    print('    # In codegen JSON, set: "logits": true')
    print()
    print("  The dispatcher will pass logits_soft_cap to the kernel arguments.")

    # Step 6: GPU run with standard kernel (no soft cap) for baseline
    print("\nStep 6: GPU Baseline (standard kernel, no soft cap)")

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
            ok_gpu, max_abs_gpu, _ = validator.check(result.output, O_no_cap)
            print(
                f"  GPU (no cap):  time={result.time_ms:.4f}ms  TFLOPS={result.tflops:.2f}  "
                f"max_err={max_abs_gpu:.2e}  {'PASS' if ok_gpu else 'FAIL'}"
            )
        else:
            print(f"  GPU error: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("  Logits soft cap: tanh(scores / cap) * cap before softmax")
    print(f"  Large cap -> standard attention (verified: max_err={max_abs:.2e})")
    print("  Small cap -> output variance reduced, stabilizes training")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
