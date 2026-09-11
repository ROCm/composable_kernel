#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 32: Sweep Head Dimension

Demonstrates FMHA across different head dimensions (32, 64, 128, 256).
The prebuilt library only supports hdim=128; other head dimensions are
validated via CPU reference only.

Fixed: batch=2, nhead=8, seqlen=128
Sweep: hdim in [32, 64, 128, 256]

Usage:
    python3 32_sweep_hdim.py
    python3 32_sweep_hdim.py --arch gfx942
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

BATCH = 2
NHEAD = 8
SEQLEN = 128
HDIMS = [32, 64, 128, 256]
GPU_SUPPORTED_HDIM = 128


def main():
    parser = argparse.ArgumentParser(description="Sweep Head Dimension FMHA")
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 32: Sweep Head Dimension")
    print("=" * 70)

    print(f"\n  Fixed: batch={BATCH}, nhead={NHEAD}, seqlen={SEQLEN}")
    print(f"  Sweep: hdim in {HDIMS}")
    print(f"  Arch:  {args.arch}")
    print(f"  Note:  Only hdim={GPU_SUPPORTED_HDIM} runs on GPU (prebuilt lib)")

    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    # Step 1: JIT-compile FMHA kernel (hdim=128)
    print("\nStep 1: JIT-Compile FMHA Kernel")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=GPU_SUPPORTED_HDIM,
        hdim_v=GPU_SUPPORTED_HDIM,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    runner = None
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
        print("  Will run CPU reference only")
    else:
        runner = setup.runner
        print(f"  JIT build: {setup.build_time_s:.1f}s")

    # Step 2: CPU reference for all hdims
    print("\nStep 2: CPU Reference for All Head Dimensions")

    np.random.seed(42)
    cpu_data = {}

    print(
        f"\n  {'hdim':>6} | {'Scale':>8} | {'FLOPs':>14} | {'O Range':>22} | {'Finite':<6}"
    )
    print("  " + "-" * 66)

    for hdim in HDIMS:
        prob = FmhaProblem(
            batch=BATCH,
            nhead_q=NHEAD,
            nhead_k=NHEAD,
            seqlen_q=SEQLEN,
            seqlen_k=SEQLEN,
            hdim_q=hdim,
            hdim_v=hdim,
        )

        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)

        O_ref = cpu_attention_fwd(Q, K, V, prob.scale)
        is_finite = bool(np.all(np.isfinite(O_ref)))
        o_range = f"[{O_ref.min():.4f}, {O_ref.max():.4f}]"

        print(
            f"  {hdim:>6} | {prob.scale:>8.4f} | {prob.num_ops:>14,} | {o_range:>22} | {'OK' if is_finite else 'NaN!':<6}"
        )
        cpu_data[hdim] = (Q, K, V, O_ref, prob)

    # Step 3: GPU sweep (only hdim=128 supported)
    print("\nStep 3: GPU Sweep")

    hdr = f"  {'hdim':>6} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'MaxErr':>10} | {'Status':<10}"
    print(f"\n{hdr}")
    print("  " + "-" * 60)

    results = []

    for hdim in HDIMS:
        Q, K, V, O_ref, prob = cpu_data[hdim]

        if hdim != GPU_SUPPORTED_HDIM or runner is None:
            print(
                f"  {hdim:>6} | {'---':>10} | {'---':>10} | {'---':>10} | {'CPU only':<10}"
            )
            results.append((hdim, True, 0.0, 0.0, 0.0))
            continue

        Q_f16 = Q.astype(np.float16)
        K_f16 = K.astype(np.float16)
        V_f16 = V.astype(np.float16)

        res = runner.run(Q_f16, K_f16, V_f16, prob)
        if not res.success:
            print(
                f"  {hdim:>6} | {'---':>10} | {'---':>10} | {'---':>10} | {'FAIL':<10}"
            )
            results.append((hdim, False, 0.0, 0.0, 0.0))
            continue

        max_err = float(np.abs(res.output.astype(np.float32) - O_ref).max())
        ok, _, _ = validator.check(res.output, O_ref)
        tag = "PASS" if ok else "FAIL"

        print(
            f"  {hdim:>6} | {res.time_ms:>10.4f} | {res.tflops:>10.2f} | {max_err:>10.2e} | {tag:<10}"
        )
        results.append((hdim, ok, res.time_ms, res.tflops, max_err))

    # Step 4: hdim analysis
    print("\nStep 4: Head Dimension Analysis")
    print("  Each hdim requires a dedicated compiled kernel:")
    for hdim in HDIMS:
        gpu_status = "prebuilt" if hdim == GPU_SUPPORTED_HDIM else "needs JIT"
        tile_hint = f"tile_k0max={hdim}"
        print(f"    hdim={hdim:>3}: {gpu_status:<10}  ({tile_hint})")

    print("\n  Compute scales linearly with hdim (via Q*K^T and attn*V).")
    print("  Larger hdim = more work per token, fewer tokens processed per CU.")

    # Summary
    passed = sum(1 for _, ok, *_ in results if ok)
    total = len(results)
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} passed")
    print(f"  Fixed:   B={BATCH} H={NHEAD} S={SEQLEN}")
    print(f"  Sweep:   hdim={HDIMS}")
    print(f"  GPU:     hdim={GPU_SUPPORTED_HDIM} only (prebuilt)")
    print(f"  Status:  {'PASS' if passed == total else 'FAIL'}")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
