#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 24: Column-Major V Layout FMHA

Demonstrates column-major (vlayout="c") vs row-major (vlayout="r") for
the V tensor. In row-major, V is [batch, nhead, seqlen_k, hdim_v]; in
column-major, V is [batch, nhead, hdim_v, seqlen_k].

Column-major V can improve performance when hdim_v access patterns
benefit from the transposed layout (e.g., certain tile sizes or memory
coalescing characteristics on specific GPU architectures).

The prebuilt library uses row-major V. This example shows both layouts
with CPU reference and validates correctness.

Usage:
    python3 24_vlayout_col_fmha.py
    python3 24_vlayout_col_fmha.py --seqlen 512
    python3 24_vlayout_col_fmha.py --batch 4
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


def cpu_attention_fwd_vlayout_col(
    Q: np.ndarray,
    K: np.ndarray,
    V_col: np.ndarray,
    scale: float,
) -> np.ndarray:
    """CPU reference: attention with column-major V.

    Args:
        Q:     [batch, nhead_q, seqlen_q, hdim_q]  float32 (row-major)
        K:     [batch, nhead_k, seqlen_k, hdim_q]  float32 (row-major)
        V_col: [batch, nhead_k, hdim_v, seqlen_k]  float32 (column-major)
        scale: softmax scale

    Returns:
        O: [batch, nhead_q, seqlen_q, hdim_v]  float32
    """
    V_row = V_col.transpose(0, 1, 3, 2)
    return cpu_attention_fwd(Q, K, V_row, scale)


def analyze_strides(name: str, arr: np.ndarray, dim_names: list):
    """Print stride information for a tensor."""
    strides_bytes = arr.strides
    itemsize = arr.itemsize
    strides_elems = tuple(s // itemsize for s in strides_bytes)
    print(f"  {name}:")
    print(f"    Shape:   {arr.shape}")
    print(f"    Strides: {strides_elems} (elements)")
    for i, (dname, s) in enumerate(zip(dim_names, strides_elems)):
        contiguous = "(contiguous)" if i == len(dim_names) - 1 and s == 1 else ""
        print(f"      {dname}: stride={s} {contiguous}")


def main():
    parser = argparse.ArgumentParser(
        description="Column-Major V Layout FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 24: Column-Major V Layout FMHA")
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

    # Step 1: Layout comparison
    print("\nStep 1: V Tensor Layouts")

    np.random.seed(42)
    V_row = np.ascontiguousarray(
        (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float32)
    )
    V_col = np.ascontiguousarray(V_row.transpose(0, 1, 3, 2))

    analyze_strides(
        "V row-major [B, H, SeqK, Hdim]",
        V_row,
        ["batch", "nhead", "seqlen_k", "hdim_v"],
    )
    analyze_strides(
        "V col-major [B, H, Hdim, SeqK]",
        V_col,
        ["batch", "nhead", "hdim_v", "seqlen_k"],
    )

    print("\n  Row-major: last dim is hdim_v -> sequential hdim access per token")
    print("  Col-major: last dim is seqlen_k -> sequential token access per hdim")

    # Step 2: CPU reference for both layouts
    print("\nStep 2: CPU Reference (both layouts)")

    Q = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float32)

    O_from_row = cpu_attention_fwd(Q, K, V_row, prob.scale)
    O_from_col = cpu_attention_fwd_vlayout_col(Q, K, V_col, prob.scale)

    validator = FmhaValidator(rtol=1e-5, atol=1e-5)
    ok, max_abs, max_rel = validator.check(O_from_row, O_from_col)

    print(
        f"  O from row-major V: shape={O_from_row.shape}  "
        f"range=[{O_from_row.min():.4f}, {O_from_row.max():.4f}]"
    )
    print(
        f"  O from col-major V: shape={O_from_col.shape}  "
        f"range=[{O_from_col.min():.4f}, {O_from_col.max():.4f}]"
    )
    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Match: {'PASS' if ok else 'FAIL'}")

    # Step 3: Memory access pattern analysis
    print("\nStep 3: Memory Access Pattern Analysis")

    tile_sizes = [(128, 128), (64, 128), (128, 64)]
    print("\n  For P @ V matmul (P: [sq, sk] x V: [sk, hdim_v]):")
    print(f"  {'Tile(M,N)':>12} {'V Row Accesses':>18} {'V Col Accesses':>18}")
    print("  " + "-" * 52)

    for tm, tn in tile_sizes:
        row_access = f"sk_stride={args.hdim}"
        col_access = "sk_stride=1"
        print(f"  {f'{tm}x{tn}':>12} {row_access:>18} {col_access:>18}")

    print("\n  Row-major V: coalesced reads when accessing hdim_v (inner loop)")
    print("  Col-major V: coalesced reads when accessing seqlen_k (inner loop)")
    print("  Optimal layout depends on tile shape and GPU memory subsystem")

    # Step 4: Shape sweep with both layouts
    print("\nStep 4: Correctness Sweep")

    shapes = [
        (1, 4, 64, 64, 64),
        (2, 8, 128, 128, 128),
        (1, 8, 256, 256, 128),
        (2, 4, 128, 128, 64),
        (1, 16, 64, 64, 128),
    ]

    print(f"\n  {'Shape':<32} {'MaxErr':>12} {'Status':>8}")
    print("  " + "-" * 55)

    all_ok = True
    for b, h, sq, sk, d in shapes:
        Q_t = (np.random.randn(b, h, sq, d) * 0.3).astype(np.float32)
        K_t = (np.random.randn(b, h, sk, d) * 0.3).astype(np.float32)
        V_r = (np.random.randn(b, h, sk, d) * 0.3).astype(np.float32)
        V_c = np.ascontiguousarray(V_r.transpose(0, 1, 3, 2))

        scale = 1.0 / (d**0.5)
        O_r = cpu_attention_fwd(Q_t, K_t, V_r, scale)
        O_c = cpu_attention_fwd_vlayout_col(Q_t, K_t, V_c, scale)

        ok_t, max_abs_t, _ = validator.check(O_r, O_c)
        all_ok = all_ok and ok_t
        shape_str = f"B{b}_H{h}_S{sq}x{sk}_D{d}"
        print(f"  {shape_str:<32} {max_abs_t:>12.2e} {'PASS' if ok_t else 'FAIL':>8}")

    # Step 5: GPU API pattern
    print("\nStep 5: GPU Kernel Configuration")
    print("  NOTE: The prebuilt library uses row-major V (vlayout='r').")
    print("  For column-major V, compile a kernel with vlayout='c':")
    print()
    print("    FmhaSignature()")
    print("        .vlayout('c')   // column-major V: [B, H, Hdim, SeqK]")
    print()
    print("    FmhaKernelConfig(vlayout='c', ...)")

    # Step 6: GPU baseline (row-major)
    print("\nStep 6: GPU Baseline (row-major V)")

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
        V_f16 = V_row.astype(np.float16)

        result = runner.run(Q_f16, K_f16, V_f16, prob)
        if result.success:
            ok_gpu, max_abs_gpu, _ = validator.check(result.output, O_from_row)
            print(
                f"  GPU (row-major V): time={result.time_ms:.4f}ms  TFLOPS={result.tflops:.2f}  "
                f"max_err={max_abs_gpu:.2e}  {'PASS' if ok_gpu else 'FAIL'}"
            )
        else:
            print(f"  GPU error: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("  vlayout='r': V is [B, H, SeqK, Hdim] (default, row-major)")
    print("  vlayout='c': V is [B, H, Hdim, SeqK] (column-major)")
    print(
        f"  Both layouts produce identical results (verified: {'PASS' if all_ok else 'FAIL'})"
    )
    print("  Choice depends on upstream memory layout and GPU tile access patterns")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
