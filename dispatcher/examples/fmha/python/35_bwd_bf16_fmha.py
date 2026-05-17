#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 35: Backward Pass with BF16 Data Type

Demonstrates the FMHA backward pass with bfloat16 precision.

BF16 differences from FP16:
  - 8-bit exponent (same as fp32) vs fp16's 5-bit
  - 7-bit mantissa vs fp16's 10-bit
  - Larger dynamic range but lower precision

Tolerance guidance for backward:
  - fp16 bwd: rtol=1.6e-2 typically sufficient
  - bf16 bwd: rtol=3.2e-2 for hdim > 128 (less mantissa precision)
  - bf16 bwd: rtol=2.0e-2 for hdim <= 128

CPU backward reference is computed in float32, then compared against
bf16-quantized inputs to measure the precision impact.

Usage:
    python3 35_bwd_bf16_fmha.py
    python3 35_bwd_bf16_fmha.py --hdim 256
    python3 35_bwd_bf16_fmha.py --arch gfx942
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
    cpu_attention_bwd,
)


def to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 -> bfloat16 (stored as uint16 with bf16 bit pattern)."""
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (uint16) -> float32."""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def cpu_fwd_with_intermediates(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> tuple:
    """Forward pass returning out, P, LSE."""
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = S_exp.sum(axis=-1, keepdims=True)
    P = S_exp / S_sum
    out = np.matmul(P, V)
    lse = (np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)).astype(np.float32)
    return out, P, lse


def get_bwd_tolerance(dtype: str, hdim: int) -> tuple:
    """Recommended tolerances for backward pass validation."""
    if dtype == "bf16":
        if hdim > 128:
            return 3.2e-2, 3.2e-2
        return 2.0e-2, 2.0e-2
    return 1.6e-2, 1.6e-2


def main():
    parser = argparse.ArgumentParser(description="Backward Pass with BF16")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 35: Backward Pass with BF16")
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

    print(f"\n  Problem: B={prob.batch} H={prob.nhead_q} S={args.seqlen} D={args.hdim}")
    print(f"  Scale:   {prob.scale:.6f}")
    print(f"  Arch:    {args.arch}")

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
        print(
            "  Note: Native bf16 bwd kernel requires separate JIT with data_type='bf16'"
        )
    else:
        print(f"  JIT build: {setup.error}")
        print("  Continuing with CPU reference only")

    # --- Generate data in both dtypes ---
    np.random.seed(42)
    Q_f32 = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K_f32 = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V_f32 = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    dO_f32 = (np.random.randn(*prob.o_shape()) * 0.1).astype(np.float32)

    Q_fp16 = Q_f32.astype(np.float16).astype(np.float32)
    K_fp16 = K_f32.astype(np.float16).astype(np.float32)
    V_fp16 = V_f32.astype(np.float16).astype(np.float32)
    dO_fp16 = dO_f32.astype(np.float16).astype(np.float32)

    Q_bf16 = bf16_to_f32(to_bf16(Q_f32))
    K_bf16 = bf16_to_f32(to_bf16(K_f32))
    V_bf16 = bf16_to_f32(to_bf16(V_f32))
    dO_bf16 = bf16_to_f32(to_bf16(dO_f32))

    # --- Quantization error comparison ---
    print("\n--- Quantization Error ---")
    print(
        f"\n  {'Tensor':<6} {'FP16 quant err':>16} {'BF16 quant err':>16} {'BF16/FP16':>10}"
    )
    print("  " + "-" * 52)

    for name, orig, fp16, bf16 in [
        ("Q", Q_f32, Q_fp16, Q_bf16),
        ("K", K_f32, K_fp16, K_bf16),
        ("V", V_f32, V_fp16, V_bf16),
        ("dO", dO_f32, dO_fp16, dO_bf16),
    ]:
        fp16_err = float(np.abs(orig - fp16).max())
        bf16_err = float(np.abs(orig - bf16).max())
        ratio = bf16_err / (fp16_err + 1e-15)
        print(f"  {name:<6} {fp16_err:>16.2e} {bf16_err:>16.2e} {ratio:>10.1f}x")

    # --- Backward with both dtypes ---
    print("\n--- Backward Gradients: FP16 vs BF16 Inputs ---")

    dtype_configs = [
        ("fp16", Q_fp16, K_fp16, V_fp16, dO_fp16),
        ("bf16", Q_bf16, K_bf16, V_bf16, dO_bf16),
    ]

    grad_results = {}
    for dtype_name, Q_d, K_d, V_d, dO_d in dtype_configs:
        out, P, lse = cpu_fwd_with_intermediates(Q_d, K_d, V_d, prob.scale)
        dQ, dK, dV = cpu_attention_bwd(Q_d, K_d, V_d, out, dO_d, P, prob.scale)
        grad_results[dtype_name] = (dQ, dK, dV)

    print(f"\n  {'Dtype':<6} {'|dQ| mean':>12} {'|dK| mean':>12} {'|dV| mean':>12}")
    print("  " + "-" * 48)
    for dtype_name in ["fp16", "bf16"]:
        dQ, dK, dV = grad_results[dtype_name]
        print(
            f"  {dtype_name:<6} {np.abs(dQ).mean():>12.4e} "
            f"{np.abs(dK).mean():>12.4e} {np.abs(dV).mean():>12.4e}"
        )

    # --- Cross-dtype gradient difference ---
    print("\n--- FP16 vs BF16 Backward Difference ---")
    dQ_fp, dK_fp, dV_fp = grad_results["fp16"]
    dQ_bf, dK_bf, dV_bf = grad_results["bf16"]

    print(
        f"\n  {'Grad':<6} {'Max abs diff':>14} {'Mean abs diff':>14} {'Max rel diff':>14}"
    )
    print("  " + "-" * 52)
    for name, g_fp, g_bf in [
        ("dQ", dQ_fp, dQ_bf),
        ("dK", dK_fp, dK_bf),
        ("dV", dV_fp, dV_bf),
    ]:
        abs_diff = np.abs(g_fp - g_bf)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        max_rel = float((abs_diff / (np.abs(g_fp) + 1e-8)).max())
        print(f"  {name:<6} {max_abs:>14.4e} {mean_abs:>14.4e} {max_rel:>14.4e}")

    # --- Tolerance analysis for different hdims ---
    print("\n--- Recommended Backward Tolerances ---")
    print(f"\n  {'Dtype':<6} {'hdim':>6} {'rtol':>10} {'atol':>10} {'Note'}")
    print("  " + "-" * 54)
    for dtype in ["fp16", "bf16"]:
        for hdim in [64, 128, 256]:
            rtol, atol = get_bwd_tolerance(dtype, hdim)
            note = ""
            if dtype == "bf16" and hdim > 128:
                note = "<-- relaxed for large hdim"
            print(f"  {dtype:<6} {hdim:>6} {rtol:>10.1e} {atol:>10.1e} {note}")

    # --- Validate backward with appropriate tolerances ---
    print("\n--- Validation Against F32 Reference ---")
    out_f32, P_f32, _ = cpu_fwd_with_intermediates(Q_f32, K_f32, V_f32, prob.scale)
    dQ_ref, dK_ref, dV_ref = cpu_attention_bwd(
        Q_f32,
        K_f32,
        V_f32,
        out_f32,
        dO_f32,
        P_f32,
        prob.scale,
    )

    for dtype_name in ["fp16", "bf16"]:
        rtol, atol = get_bwd_tolerance(dtype_name, args.hdim)
        dQ, dK, dV = grad_results[dtype_name]

        print(f"\n  [{dtype_name}] rtol={rtol:.1e}, atol={atol:.1e}")
        for gname, g, g_ref in [
            ("dQ", dQ, dQ_ref),
            ("dK", dK, dK_ref),
            ("dV", dV, dV_ref),
        ]:
            max_err = float(np.abs(g - g_ref).max())
            ok = bool(np.allclose(g, g_ref, rtol=rtol, atol=atol))
            print(f"    {gname}: max_err={max_err:.4e}  {'PASS' if ok else 'FAIL'}")

    # --- BF16 backward GPU API pattern ---
    print("\n--- BF16 Backward GPU API Pattern ---")
    print("  Native bf16 backward kernel:")
    print("    FmhaKernelConfig(family='bwd', data_type='bf16', ...)")
    print("  Internal accumulation stays in fp32 for numerical stability.")
    print("  Stage 3 (convert_dq) converts fp32 accumulator -> bf16 output.")
    print("  BF16 advantage: wider dynamic range prevents overflow in")
    print("  intermediate products (S = Q @ K^T) for large sequences.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Data types:    fp16 (10-bit mantissa) vs bf16 (7-bit mantissa)")
    print("  Tolerances:    bf16 bwd needs ~2x relaxed rtol vs fp16")
    rtol_used, _ = get_bwd_tolerance("bf16", args.hdim)
    print(f"  Current hdim:  {args.hdim} -> bf16 rtol={rtol_used:.1e}")
    print("  GPU:           Requires bwd-family JIT kernel with data_type='bf16'")
    print("  Status:        DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
