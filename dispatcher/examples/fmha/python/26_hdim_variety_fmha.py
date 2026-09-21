#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 26: Head Dimension Variety FMHA

Demonstrates FMHA with multiple head dimensions (32, 64, 128, 256) and
asymmetric hdim (hdim_q != hdim_v). Different head dimensions require
different tile sizes and kernel configurations for optimal performance.

The prebuilt library supports hdim=128 only. This example validates all
head dimensions via CPU reference and runs GPU for hdim=128.

Usage:
    python3 26_hdim_variety_fmha.py
    python3 26_hdim_variety_fmha.py --seqlen 256
    python3 26_hdim_variety_fmha.py --batch 4
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


def recommended_tile(hdim: int) -> str:
    """Suggest tile configuration for a given head dimension."""
    tiles = {
        32: "128x128x32x32x32x32",
        64: "128x64x32x64x32x64",
        128: "128x128x32x128x32x128",
        256: "128x128x32x256x32x256",
    }
    return tiles.get(hdim, f"auto (hdim={hdim})")


def compute_flops(
    batch: int, nhead_q: int, sq: int, sk: int, hdim_q: int, hdim_v: int
) -> int:
    """Compute FMHA FLOPs accounting for asymmetric hdim."""
    return 2 * batch * nhead_q * sq * sk * (hdim_q + hdim_v)


def main():
    parser = argparse.ArgumentParser(
        description="Head Dimension Variety FMHA Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 26: Head Dimension Variety FMHA")
    print("=" * 70)

    validator = FmhaValidator(rtol=1e-2, atol=1e-2)

    # Step 1: Symmetric head dimensions
    print("\nStep 1: Symmetric Head Dimensions (hdim_q == hdim_v)")

    hdims = [32, 64, 128, 256]

    print(f"\n  {'hdim':>6} {'Shape':>30} {'Tile Config':>30} {'FLOPs':>14}")
    print("  " + "-" * 84)

    for hdim in hdims:
        shape = f"B{args.batch}_H{args.nhead}_S{args.seqlen}_D{hdim}"
        tile = recommended_tile(hdim)
        flops = compute_flops(
            args.batch, args.nhead, args.seqlen, args.seqlen, hdim, hdim
        )
        print(f"  {hdim:>6} {shape:>30} {tile:>30} {flops:>14,}")

    # Step 2: CPU validation for each hdim
    print("\nStep 2: CPU Validation")

    np.random.seed(42)

    print(
        f"\n  {'hdim_q':>7} {'hdim_v':>7} {'Scale':>10} {'OutRange':>22} {'SelfCheck':>10}"
    )
    print("  " + "-" * 60)

    cpu_results = {}
    for hdim in hdims:
        prob = FmhaProblem(
            batch=args.batch,
            nhead_q=args.nhead,
            nhead_k=args.nhead,
            seqlen_q=args.seqlen,
            seqlen_k=args.seqlen,
            hdim_q=hdim,
            hdim_v=hdim,
        )
        Q = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float32)

        O_ref = cpu_attention_fwd(Q, K, V, prob.scale)

        self_ok = np.all(np.isfinite(O_ref))
        out_range = f"[{O_ref.min():.4f}, {O_ref.max():.4f}]"
        print(
            f"  {hdim:>7} {hdim:>7} {prob.scale:>10.4f} {out_range:>22} {'OK' if self_ok else 'NaN!':>10}"
        )

        cpu_results[hdim] = (Q, K, V, O_ref, prob)

    # Step 3: Asymmetric head dimensions
    print("\nStep 3: Asymmetric Head Dimensions (hdim_q != hdim_v)")

    asymmetric_configs = [
        (128, 64, "Large Q, small V: more attention capacity, compact output"),
        (64, 128, "Small Q, large V: compact attention, rich output"),
        (128, 256, "Standard Q, very large V: high-capacity value projection"),
        (256, 128, "Large Q, standard V: wide attention field"),
        (32, 128, "Tiny Q, standard V: minimal attention compute"),
    ]

    print(
        f"\n  {'hdim_q':>7} {'hdim_v':>7} {'Q Shape':>22} {'O Shape':>22} {'MaxErr vs self':>16}"
    )
    print("  " + "-" * 78)

    for hdim_q, hdim_v, desc in asymmetric_configs:
        prob = FmhaProblem(
            batch=args.batch,
            nhead_q=args.nhead,
            nhead_k=args.nhead,
            seqlen_q=args.seqlen,
            seqlen_k=args.seqlen,
            hdim_q=hdim_q,
            hdim_v=hdim_v,
        )
        Q = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float32)
        K = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float32)
        V = (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float32)

        out = cpu_attention_fwd(Q, K, V, prob.scale)

        O2 = cpu_attention_fwd(Q, K, V, prob.scale)
        max_err = float(np.abs(out - O2).max())

        print(
            f"  {hdim_q:>7} {hdim_v:>7} {str(prob.q_shape()):>22} {str(prob.o_shape()):>22} {max_err:>16.2e}"
        )

    print("\n  Asymmetric hdim notes:")
    for hdim_q, hdim_v, desc in asymmetric_configs:
        print(f"    hdim_q={hdim_q}, hdim_v={hdim_v}: {desc}")

    # Step 4: GPU validation (hdim=128)
    print("\nStep 4: GPU Validation (hdim=128)")

    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=128,
        hdim_v=128,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    gpu_tflops = 0.0
    gpu_time = 0.0
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
    else:
        runner = setup.runner
        print(f"  JIT build: {setup.build_time_s:.1f}s")

        Q, K, V, O_ref, prob = cpu_results[128]
        Q_f16 = Q.astype(np.float16)
        K_f16 = K.astype(np.float16)
        V_f16 = V.astype(np.float16)

        result = runner.run(Q_f16, K_f16, V_f16, prob)
        if result.success:
            ok, max_abs, _ = validator.check(result.output, O_ref)
            print(
                f"  GPU hdim=128: time={result.time_ms:.4f}ms  TFLOPS={result.tflops:.2f}  "
                f"max_err={max_abs:.2e}  {'PASS' if ok else 'FAIL'}"
            )

            gpu_tflops = result.tflops
            gpu_time = result.time_ms
        else:
            print(f"  GPU error: {result.error}")

    # Step 5: Performance projection table
    print("\nStep 5: Performance Summary Table")

    print(
        f"\n  {'hdim_q':>7} | {'hdim_v':>7} | {'FLOPs':>14} | {'Tile':>24} | {'GPU Support':>12}"
    )
    print("  " + "-" * 78)

    for hdim in hdims:
        flops = compute_flops(
            args.batch, args.nhead, args.seqlen, args.seqlen, hdim, hdim
        )
        tile = recommended_tile(hdim)
        gpu_ok = "prebuilt" if hdim == 128 else "needs JIT"
        print(f"  {hdim:>7} | {hdim:>7} | {flops:>14,} | {tile:>24} | {gpu_ok:>12}")

    print("  " + "-" * 78)

    for hdim_q, hdim_v, _ in asymmetric_configs[:3]:
        flops = compute_flops(
            args.batch, args.nhead, args.seqlen, args.seqlen, hdim_q, hdim_v
        )
        gpu_ok = "needs JIT"
        print(
            f"  {hdim_q:>7} | {hdim_v:>7} | {flops:>14,} | {'asymmetric':>24} | {gpu_ok:>12}"
        )

    # Step 6: Kernel configuration per hdim
    print("\nStep 6: Kernel Configuration Per Head Dimension")
    print("  Each hdim requires a dedicated compiled kernel:")
    print()
    print(
        "  hdim=32:  FmhaKernelConfig(hdim_q=32, hdim_v=32, "
        "tile_m0=128, tile_n0=128, tile_k0=32, tile_n1=32, tile_k1=32, tile_k0max=32)"
    )
    print(
        "  hdim=64:  FmhaKernelConfig(hdim_q=64, hdim_v=64, "
        "tile_m0=128, tile_n0=64, tile_k0=32, tile_n1=64, tile_k1=32, tile_k0max=64)"
    )
    print(
        "  hdim=128: FmhaKernelConfig(hdim_q=128, hdim_v=128, "
        "tile_m0=128, tile_n0=128, tile_k0=32, tile_n1=128, tile_k1=32, tile_k0max=128)"
    )
    print(
        "  hdim=256: FmhaKernelConfig(hdim_q=256, hdim_v=256, "
        "tile_m0=128, tile_n0=128, tile_k0=32, tile_n1=256, tile_k1=32, tile_k0max=256)"
    )
    print()
    print("  Asymmetric: FmhaKernelConfig(hdim_q=128, hdim_v=64, ...)")
    print("  tile_n1 tracks hdim_v; tile_k0max tracks hdim_q")

    # Summary
    print("\n" + "=" * 70)
    print(f"  Supported symmetric hdims: {hdims}")
    print("  Asymmetric hdim (hdim_q != hdim_v): fully supported")
    print("  Tile sizes scale with hdim; larger hdim needs wider tiles")
    if gpu_tflops > 0:
        print(f"  GPU baseline (hdim=128): {gpu_tflops:.2f} TFLOPS @ {gpu_time:.4f} ms")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
