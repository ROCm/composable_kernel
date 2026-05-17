#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 25: Input/Output Permutation FMHA

Demonstrates different memory layouts for Q/K/V/O tensors via
input permutation (iperm) and output permutation (operm):

  iperm=0 (bshd): [batch, seqlen, nhead, hdim]  -- used by some frameworks
  iperm=1 (bhsd): [batch, nhead, seqlen, hdim]  -- standard/default

  operm=0 (bshd): O is [batch, seqlen, nhead, hdim]
  operm=1 (bhsd): O is [batch, nhead, seqlen, hdim]

The prebuilt library uses bhsd layout (iperm=1, operm=1). This example
shows how to convert between layouts and validates correctness.

Usage:
    python3 25_permutation_fmha.py
    python3 25_permutation_fmha.py --seqlen 256
    python3 25_permutation_fmha.py --batch 4
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


def bhsd_to_bshd(x: np.ndarray) -> np.ndarray:
    """Convert [batch, nhead, seqlen, hdim] -> [batch, seqlen, nhead, hdim]."""
    return x.transpose(0, 2, 1, 3)


def bshd_to_bhsd(x: np.ndarray) -> np.ndarray:
    """Convert [batch, seqlen, nhead, hdim] -> [batch, nhead, seqlen, hdim]."""
    return x.transpose(0, 2, 1, 3)


def cpu_attention_fwd_bshd(
    Q_bshd: np.ndarray,
    K_bshd: np.ndarray,
    V_bshd: np.ndarray,
    scale: float,
    operm: int = 0,
) -> np.ndarray:
    """CPU reference with bshd input, configurable output layout.

    Args:
        Q_bshd: [batch, seqlen_q, nhead_q, hdim_q]  float32
        K_bshd: [batch, seqlen_k, nhead_k, hdim_q]  float32
        V_bshd: [batch, seqlen_k, nhead_k, hdim_v]  float32
        scale: softmax scale
        operm: 0 -> output bshd, 1 -> output bhsd

    Returns:
        O: float32 in requested layout
    """
    Q_bhsd = bshd_to_bhsd(Q_bshd)
    K_bhsd = bshd_to_bhsd(K_bshd)
    V_bhsd = bshd_to_bhsd(V_bshd)

    O_bhsd = cpu_attention_fwd(Q_bhsd, K_bhsd, V_bhsd, scale)

    if operm == 0:
        return bhsd_to_bshd(O_bhsd)
    return O_bhsd


def describe_layout(arr: np.ndarray, layout_name: str, dim_names: list):
    """Print layout details including strides."""
    itemsize = arr.itemsize
    strides_elems = tuple(s // itemsize for s in arr.strides)
    is_contiguous = arr.flags["C_CONTIGUOUS"]
    print(f"  {layout_name}:")
    print(f"    Shape:      {arr.shape}")
    print(f"    Strides:    {strides_elems} (elements)")
    print(f"    Contiguous: {is_contiguous}")
    for dname, s in zip(dim_names, strides_elems):
        print(f"      {dname:>8}: stride={s}")


def main():
    parser = argparse.ArgumentParser(
        description="Input/Output Permutation FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 25: Input/Output Permutation FMHA")
    print("=" * 70)

    B, H, S, D = args.batch, args.nhead, args.seqlen, args.hdim
    prob = FmhaProblem(
        batch=B,
        nhead_q=H,
        nhead_k=H,
        seqlen_q=S,
        seqlen_k=S,
        hdim_q=D,
        hdim_v=D,
    )

    # Step 1: Layout definitions
    print("\nStep 1: Layout Definitions")

    np.random.seed(42)
    Q_bhsd = np.ascontiguousarray(
        (np.random.randn(B, H, S, D) * 0.3).astype(np.float32)
    )
    Q_bshd = np.ascontiguousarray(bhsd_to_bshd(Q_bhsd))

    describe_layout(Q_bhsd, "bhsd (iperm=1)", ["batch", "nhead", "seqlen", "hdim"])
    describe_layout(Q_bshd, "bshd (iperm=0)", ["batch", "seqlen", "nhead", "hdim"])

    print("\n  Key difference:")
    print("    bhsd: heads are contiguous -> good for per-head parallelism")
    print("    bshd: tokens are contiguous -> good for sequence parallelism")

    # Step 2: All permutation combinations
    print("\nStep 2: All Permutation Combinations (CPU Reference)")

    K_bhsd = (np.random.randn(B, H, S, D) * 0.3).astype(np.float32)
    V_bhsd = (np.random.randn(B, H, S, D) * 0.3).astype(np.float32)
    K_bshd = np.ascontiguousarray(bhsd_to_bshd(K_bhsd))
    V_bshd = np.ascontiguousarray(bhsd_to_bshd(V_bhsd))

    O_ref_bhsd = cpu_attention_fwd(Q_bhsd, K_bhsd, V_bhsd, prob.scale)
    O_ref_bshd = bhsd_to_bshd(O_ref_bhsd)

    validator = FmhaValidator(rtol=1e-5, atol=1e-5)

    combos = [
        ("iperm=1 operm=1", "bhsd->bhsd", Q_bhsd, K_bhsd, V_bhsd, 1, O_ref_bhsd),
        ("iperm=1 operm=0", "bhsd->bshd", Q_bhsd, K_bhsd, V_bhsd, 0, O_ref_bshd),
        ("iperm=0 operm=1", "bshd->bhsd", Q_bshd, K_bshd, V_bshd, 1, O_ref_bhsd),
        ("iperm=0 operm=0", "bshd->bshd", Q_bshd, K_bshd, V_bshd, 0, O_ref_bshd),
    ]

    print(
        f"\n  {'Config':<18} {'Transform':<14} {'OutShape':>24} {'MaxErr':>12} {'Status':>8}"
    )
    print("  " + "-" * 80)

    all_ok = True
    for name, transform, Q_in, K_in, V_in, operm, O_expected in combos:
        if Q_in.shape[1] == H:
            O_out = cpu_attention_fwd(Q_in, K_in, V_in, prob.scale)
            if operm == 0:
                O_out = bhsd_to_bshd(O_out)
        else:
            O_out = cpu_attention_fwd_bshd(Q_in, K_in, V_in, prob.scale, operm)

        ok, max_abs, _ = validator.check(O_out, O_expected)
        all_ok = all_ok and ok
        print(
            f"  {name:<18} {transform:<14} {str(O_out.shape):>24} {max_abs:>12.2e} {'PASS' if ok else 'FAIL':>8}"
        )

    # Step 3: Stride comparison table
    print("\nStep 3: Stride Comparison")

    print(f"\n  For B={B}, H={H}, S={S}, D={D}:")
    print(f"  {'Layout':>8} {'Dim Order':>16} {'Strides':>28} {'hdim contiguous':>18}")
    print("  " + "-" * 74)

    bhsd_strides = (H * S * D, S * D, D, 1)
    bshd_strides = (S * H * D, H * D, D, 1)

    print(f"  {'bhsd':>8} {'B,H,S,D':>16} {str(bhsd_strides):>28} {'Yes':>18}")
    print(f"  {'bshd':>8} {'B,S,H,D':>16} {str(bshd_strides):>28} {'Yes':>18}")

    print("\n  Stride analysis:")
    print(f"    bhsd: advancing 1 token = skip {D} elements (hdim)")
    print(f"    bshd: advancing 1 token = skip {H * D} elements (nhead * hdim)")
    print(f"    bhsd: advancing 1 head  = skip {S * D} elements (seqlen * hdim)")
    print(f"    bshd: advancing 1 head  = skip {D} elements (hdim)")

    # Step 4: Conversion cost
    print("\nStep 4: Layout Conversion Cost")

    tensor_bytes = B * H * S * D * 4
    print(f"  Tensor size: {tensor_bytes / 1024:.1f} KB (float32)")
    print("  bhsd <-> bshd conversion: transpose(0,2,1,3) + contiguous copy")
    print(
        "  If upstream provides bshd and kernel wants bhsd, conversion costs ~2x memory bandwidth"
    )
    print("  Using iperm parameter avoids this copy by adjusting kernel strides")

    # Step 5: GPU run (bhsd, default layout)
    print("\nStep 5: GPU Run (bhsd layout, iperm=1)")

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

        Q_f16 = Q_bhsd.astype(np.float16)
        K_f16 = K_bhsd.astype(np.float16)
        V_f16 = V_bhsd.astype(np.float16)

        result = runner.run(Q_f16, K_f16, V_f16, prob)
        if result.success:
            ok_gpu, max_abs_gpu, _ = validator.check(result.output, O_ref_bhsd)
            print(
                f"  GPU (bhsd): time={result.time_ms:.4f}ms  TFLOPS={result.tflops:.2f}  "
                f"max_err={max_abs_gpu:.2e}  {'PASS' if ok_gpu else 'FAIL'}"
            )
        else:
            print(f"  GPU error: {result.error}")

    # Step 6: Kernel configuration for bshd
    print("\nStep 6: GPU Kernel Configuration for bshd")
    print("  The prebuilt library uses bhsd (iperm=1, operm=1).")
    print("  For bshd input/output, the kernel adjusts internal strides:")
    print()
    print("    iperm=0: kernel reads Q,K,V as [B, S, H, D] with stride_head=D")
    print("    iperm=1: kernel reads Q,K,V as [B, H, S, D] with stride_seq=D")
    print("    operm=0: kernel writes O as [B, S, H, D]")
    print("    operm=1: kernel writes O as [B, H, S, D]")

    # Summary
    print("\n" + "=" * 70)
    print("  iperm=0 (bshd): [B, S, H, D] -- sequence-first layout")
    print("  iperm=1 (bhsd): [B, H, S, D] -- head-first layout (default)")
    print(f"  All 4 combinations validated: {'PASS' if all_ok else 'FAIL'}")
    print("  Use iperm/operm to match upstream/downstream layout without copies")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
