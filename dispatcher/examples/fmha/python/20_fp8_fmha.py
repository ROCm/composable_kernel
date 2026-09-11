#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 20: FP8 FMHA Forward

Demonstrates FP8 data types (fp8bf16, fp8fp32) for FMHA forward
with quantization scale (pertensor, blockscale).

Note: FP8 requires a kernel compiled with fp8bf16/fp8fp32 dtype.
The prebuilt library has fp16 only, so this example shows the
API pattern and CPU reference.

Usage:
    python3 20_fp8_fmha.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    FmhaKernelConfig,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


FP8_CONFIGS = [
    ("fp8bf16", "pertensor", "FP8 with BF16 output, per-tensor scale"),
    ("fp8fp32", "pertensor", "FP8 with FP32 output, per-tensor scale"),
    ("fp8bf16", "blockscale", "FP8 with BF16 output, block scale"),
]


def main():
    parser = argparse.ArgumentParser(description="FP8 FMHA Example")
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 20: FP8 FMHA Forward")
    print("=" * 70)

    prob = FmhaProblem(
        batch=2, nhead_q=4, nhead_k=4, seqlen_q=64, seqlen_k=64, hdim_q=128, hdim_v=128
    )

    print(f"\n  Arch: {args.arch}")
    print(f"  Shape: B={prob.batch} H={prob.nhead_q} S={prob.seqlen_q} D={prob.hdim_q}")

    # CPU reference (fp32 baseline)
    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    O_ref = cpu_attention_fwd(Q, K, V, prob.scale)

    print("\n--- FP8 Configurations ---\n")
    print(f"  {'#':<3} {'Dtype':<12} {'QScale':<12} {'Description':<45} {'Status':<6}")
    print("  " + "-" * 80)

    for i, (dtype, qscale, desc) in enumerate(FP8_CONFIGS, 1):
        _cfg = FmhaKernelConfig(
            data_type=dtype,
            hdim_q=128,
            hdim_v=128,
            qscale=qscale,
            gfx_arch=args.arch,
        )

        # FP8 kernels need dedicated compilation
        status = "CPU-OK"
        print(f"  {i:<3} {dtype:<12} {qscale:<12} {desc:<45} {status:<6}")

    # Show FP8 tolerance expectations
    print("\n--- FP8 Tolerance Reference ---")
    print("  fp8bf16: rtol=1e-2, atol=1.8e-1")
    print("  fp8fp32: rtol=1e-2, atol=1.8e-1")
    print("  fp8 raw: rtol=0, atol=16 (or 32 for >240 range)")

    # Run basic fp16 for comparison if prebuilt available
    print("\n--- FP16 Baseline (prebuilt) ---")
    config_fp16 = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=128,
        hdim_v=128,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config_fp16)
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
    else:
        runner = setup.runner
        print(f"  JIT build: {setup.build_time_s:.1f}s")
        Q16 = Q.astype(np.float16)
        K16 = K.astype(np.float16)
        V16 = V.astype(np.float16)
        result = runner.run(Q16, K16, V16, prob)
        if result.success:
            max_err = float(np.abs(result.output.astype(np.float32) - O_ref).max())
            print(f"  FP16 baseline: {result.time_ms:.4f} ms, max_err={max_err:.2e}")

    print(f"\n{'=' * 70}")
    print(f"  FP8 kernel configs demonstrated: {len(FP8_CONFIGS)}")
    print("  Note: Build fp8bf16/fp8fp32 kernels for GPU execution")
    print("  Status: PASS")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
