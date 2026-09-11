#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 05: NumPy Integration

Shows how to create a GPU-accelerated attention wrapper that works
seamlessly with NumPy arrays, hiding all HIP memory management.

Usage:
    python3 05_numpy_integration.py
    python3 05_numpy_integration.py --help
    python3 05_numpy_integration.py --seqlen 256
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def fmha_matmul(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float = None,
    runner=None,
) -> np.ndarray:
    """GPU-accelerated scaled dot-product attention via FMHA dispatcher.

    Args:
        Q: [batch, nhead_q, seqlen_q, hdim_q] float16/float32
        K: [batch, nhead_k, seqlen_k, hdim_q] float16/float32
        V: [batch, nhead_k, seqlen_k, hdim_v] float16/float32
        scale: softmax scale (default: 1/sqrt(hdim_q))
        runner: reuse an existing runner from setup_fmha_dispatcher

    Returns:
        O: [batch, nhead_q, seqlen_q, hdim_v] float16
    """
    batch, nhead_q, seqlen_q, hdim_q = Q.shape
    _, nhead_k, seqlen_k, hdim_v = V.shape

    prob = FmhaProblem(
        batch=batch,
        nhead_q=nhead_q,
        nhead_k=nhead_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        hdim_q=hdim_q,
        hdim_v=hdim_v,
    )

    result = runner.run(
        Q.astype(np.float16), K.astype(np.float16), V.astype(np.float16), prob
    )
    if not result.success:
        raise RuntimeError(f"GPU FMHA failed: {result.error}")
    return result.output


def main():
    parser = argparse.ArgumentParser(
        description="NumPy Integration Example - GPU-accelerated attention wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 05_numpy_integration.py                # Default
  python3 05_numpy_integration.py --seqlen 256   # Longer sequences
        """,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 05: NumPy Integration")
    print("=" * 70)

    # Step 1: JIT-compile FMHA kernel
    print("\nStep 1: JIT-Compile FMHA Dispatcher")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=args.hdim,
        hdim_v=args.hdim,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
        return 1
    runner = setup.runner
    print(f"  JIT build: {setup.build_time_s:.1f}s")
    print(f"  Arch:      {args.arch}")

    np_dtype = np.float16

    # Step 2: Demo -- simple attention call
    print("\n" + "=" * 70)
    print("Step 2: Simple Attention Call")
    print("=" * 70)

    np.random.seed(42)
    Q = (np.random.randn(args.batch, args.nhead, args.seqlen, args.hdim) * 0.5).astype(
        np_dtype
    )
    K = (np.random.randn(args.batch, args.nhead, args.seqlen, args.hdim) * 0.5).astype(
        np_dtype
    )
    V = (np.random.randn(args.batch, args.nhead, args.seqlen, args.hdim) * 0.5).astype(
        np_dtype
    )

    out = fmha_matmul(Q, K, V, runner=runner)
    print(f"  Q: {Q.shape} -> O: {out.shape}")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"  Output sum:   {out.sum():.4f}")

    # Step 3: Validate against CPU reference
    print("\n" + "=" * 70)
    print("Step 3: Validate Against CPU Reference")
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
    O_ref = cpu_attention_fwd(
        Q.astype(np.float32),
        K.astype(np.float32),
        V.astype(np.float32),
        prob.scale,
    )

    diff = np.abs(out.astype(np.float32) - O_ref)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(O_ref) + 1e-6)).max())
    match = np.allclose(out.astype(np.float32), O_ref, atol=args.atol, rtol=args.rtol)

    print(f"  Max abs error: {max_abs:.6e}")
    print(f"  Max rel error: {max_rel:.6e}")
    print(f"  Match: {match}")

    # Step 4: Demo -- multi-head attention with GQA
    print("\n" + "=" * 70)
    print("Step 4: GQA Attention (nhead_q=8, nhead_k=2)")
    print("=" * 70)

    nhead_q, nhead_k = 8, 2
    Q_gqa = (np.random.randn(args.batch, nhead_q, args.seqlen, args.hdim) * 0.5).astype(
        np_dtype
    )
    K_gqa = (np.random.randn(args.batch, nhead_k, args.seqlen, args.hdim) * 0.5).astype(
        np_dtype
    )
    V_gqa = (np.random.randn(args.batch, nhead_k, args.seqlen, args.hdim) * 0.5).astype(
        np_dtype
    )

    O_gqa = fmha_matmul(Q_gqa, K_gqa, V_gqa, runner=runner)

    prob_gqa = FmhaProblem(
        batch=args.batch,
        nhead_q=nhead_q,
        nhead_k=nhead_k,
        seqlen_q=args.seqlen,
        seqlen_k=args.seqlen,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )
    O_gqa_ref = cpu_attention_fwd(
        Q_gqa.astype(np.float32),
        K_gqa.astype(np.float32),
        V_gqa.astype(np.float32),
        prob_gqa.scale,
    )
    gqa_match = np.allclose(
        O_gqa.astype(np.float32), O_gqa_ref, atol=args.atol, rtol=args.rtol
    )

    print(f"  Q: {Q_gqa.shape}, K: {K_gqa.shape}, V: {V_gqa.shape}")
    print(f"  O: {O_gqa.shape}")
    print(f"  Match: {gqa_match}")

    # Summary
    print("\n" + "=" * 70)
    print("NumPy Integration Pattern:")
    print("=" * 70)
    print("  1. setup = setup_fmha_dispatcher(config)")
    print("  2. O = fmha_matmul(Q, K, V, runner=setup.runner)")
    print("=" * 70)

    return 0 if match and gqa_match else 1


if __name__ == "__main__":
    sys.exit(main())
