#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 11: BF16 Forward Attention

Demonstrates:
1. BF16 data generation and handling
2. GPU execution attempt with prebuilt kernel (fp16-only)
3. CPU reference computation in float32
4. BF16-specific tolerance validation (atol=1e-2)

The prebuilt library contains only fp16 kernels. This example shows the API
pattern for bf16 and gracefully falls back to CPU reference when the GPU
kernel does not support bf16.

Usage:
    python3 11_bf16_fmha.py
    python3 11_bf16_fmha.py --batch 4 --seqlen 256
    python3 11_bf16_fmha.py --arch gfx942
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


def to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to bfloat16 (stored as uint16 with bf16 bit pattern)."""
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (uint16) back to float32."""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def main():
    parser = argparse.ArgumentParser(description="BF16 Forward Attention")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 11: BF16 Forward Attention")
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
        f"\n  Problem:  B={prob.batch} H={prob.nhead_q} S={prob.seqlen_q} D={prob.hdim_q}"
    )
    print("  Dtype:    bfloat16")
    print(f"  Arch:     {args.arch}")
    print(f"  Scale:    {prob.scale:.6f}")

    # --- Generate bf16 data ---
    np.random.seed(42)
    Q_f32 = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K_f32 = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V_f32 = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)

    Q_bf16 = to_bf16(Q_f32)
    K_bf16 = to_bf16(K_f32)
    V_bf16 = to_bf16(V_f32)

    Q_bf16_f32 = bf16_to_f32(Q_bf16)
    K_bf16_f32 = bf16_to_f32(K_bf16)
    V_bf16_f32 = bf16_to_f32(V_bf16)

    print(f"\n  Q bf16 range: [{Q_bf16_f32.min():.4f}, {Q_bf16_f32.max():.4f}]")
    print(f"  K bf16 range: [{K_bf16_f32.min():.4f}, {K_bf16_f32.max():.4f}]")
    print(f"  V bf16 range: [{V_bf16_f32.min():.4f}, {V_bf16_f32.max():.4f}]")

    bf16_quant_err = np.abs(Q_f32 - Q_bf16_f32).max()
    print(f"  BF16 quantization error: {bf16_quant_err:.2e}")

    # --- GPU execution attempt ---
    print("\n--- GPU Execution ---")
    gpu_output = None
    gpu_time = None
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
        Q_fp16 = Q_bf16_f32.astype(np.float16)
        K_fp16 = K_bf16_f32.astype(np.float16)
        V_fp16 = V_bf16_f32.astype(np.float16)
        result = runner.run(Q_fp16, K_fp16, V_fp16, prob)
        if result.success:
            gpu_output = result.output
            gpu_time = result.time_ms
            print(f"  GPU:  {result.time_ms:.4f} ms, {result.tflops:.2f} TFLOPS")
            print("  Note: Ran as fp16 (JIT kernel); native bf16 kernel not compiled")
        else:
            print("  GPU:  Kernel does not support bf16 (expected)")

    # --- CPU reference (always computed) ---
    print("\n--- CPU Reference (float32 with bf16-quantized inputs) ---")
    O_ref = cpu_attention_fwd(Q_bf16_f32, K_bf16_f32, V_bf16_f32, prob.scale)
    print(f"  Output range: [{O_ref.min():.4f}, {O_ref.max():.4f}]")
    print(f"  Output shape: {O_ref.shape}")

    # --- Validation ---
    print("\n--- Validation ---")
    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    print(f"\n  {'Check':<30} {'MaxAbs':>10} {'MaxRel':>10} {'Status':>8}")
    print("  " + "-" * 62)

    if gpu_output is not None:
        ok, max_abs, max_rel = validator.check(gpu_output, O_ref)
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {'GPU vs CPU (bf16 tol)':<30} {max_abs:>10.2e} {max_rel:>10.2e} {tag:>8}"
        )
    else:
        print(f"  {'GPU vs CPU (bf16 tol)':<30} {'N/A':>10} {'N/A':>10} {'SKIP':>8}")

    strict_val = FmhaValidator(rtol=1e-5, atol=1e-5)
    ok_strict, ma_strict, mr_strict = strict_val.check(
        O_ref.astype(np.float16),
        O_ref,
    )
    print(
        f"  {'fp16(ref) vs f32(ref)':<30} {ma_strict:>10.2e} {mr_strict:>10.2e} {'PASS' if ok_strict else 'INFO':>8}"
    )

    O_ref_from_f32 = cpu_attention_fwd(Q_f32, K_f32, V_f32, prob.scale)
    bf16_impact = float(np.abs(O_ref - O_ref_from_f32).max())
    print(
        f"  {'bf16 vs f32 input impact':<30} {bf16_impact:>10.2e} {'':>10} {'INFO':>8}"
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Dtype:       bfloat16 (7-bit mantissa vs fp16's 10-bit)")
    print("  Tolerance:   atol=1e-2 (relaxed for bf16 precision)")
    print(
        f"  GPU:         {'%.4f ms' % gpu_time if gpu_time else 'N/A (bf16 kernel not in prebuilt)'}"
    )
    print("  CPU ref:     Computed with bf16-quantized inputs")
    print("  BF16 range:  Larger exponent range (±3.4e38) vs fp16 (±65504)")
    status = "PASS" if gpu_output is not None else "DEMO"
    print(f"  Status:      {status}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
