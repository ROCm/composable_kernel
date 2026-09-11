#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 09: Multiple Registries

Creates separate FmhaRegistry instances for different optimization
targets (latency vs throughput), builds both, runs the same problem
through each, and compares results.

Usage:
    python3 09_multi_registry.py
    python3 09_multi_registry.py --help
    python3 09_multi_registry.py --arch gfx950
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    FmhaRegistry,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
)


def make_latency_config(arch: str) -> FmhaKernelConfig:
    """Latency-optimized: smaller tiles, lower launch overhead."""
    return FmhaKernelConfig(
        family="fwd",
        data_type="fp16",
        hdim_q=128,
        hdim_v=128,
        pipeline="qr",
        # Stage 0 (Q*K^T): seqlen_q x seqlen_k x hdim_q
        tile_m0=64,
        tile_n0=128,
        tile_k0=32,
        # Stage 1 (Attn*V): hdim_v x seqlen_k x alignment
        tile_n1=128,
        tile_k1=32,
        tile_k0max=128,
        # Wave config per stage
        wave_m0=4,
        wave_n0=1,
        wave_k0=1,
        wave_m1=4,
        wave_n1=1,
        wave_k1=1,
        # Warp tile per stage
        warp_m0=16,
        warp_n0=16,
        warp_k0=32,
        warp_m1=16,
        warp_n1=16,
        warp_k1=16,
        pad_s=False,
        pad_sk=False,
        pad_d=True,
        pad_dv=True,
        gfx_arch=arch,
    )


def make_throughput_config(arch: str) -> FmhaKernelConfig:
    """Throughput-optimized: larger tiles, async pipeline."""
    return FmhaKernelConfig(
        family="fwd",
        data_type="fp16",
        hdim_q=128,
        hdim_v=128,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
        tile_n1=128,
        tile_k1=32,
        tile_k0max=128,
        wave_m0=4,
        wave_n0=1,
        wave_k0=1,
        wave_m1=4,
        wave_n1=1,
        wave_k1=1,
        warp_m0=32,
        warp_n0=32,
        warp_k0=16,
        warp_m1=32,
        warp_n1=32,
        warp_k1=16,
        gfx_arch=arch,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multiple FMHA Registries - latency vs throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 09_multi_registry.py
  python3 09_multi_registry.py --arch gfx950
        """,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 09: Multiple Registries")
    print("=" * 70)

    # Step 1: Define optimization-specific configs
    print("\nStep 1: Define Optimization Targets")

    latency_cfg = make_latency_config(args.arch)
    throughput_cfg = make_throughput_config(args.arch)

    print(f"  Latency config:    {latency_cfg.name}")
    print(f"    pipeline={latency_cfg.pipeline}, tile={latency_cfg.tile[:2]}")
    print(f"  Throughput config: {throughput_cfg.name}")
    print(f"    pipeline={throughput_cfg.pipeline}, tile={throughput_cfg.tile[:2]}")

    # Step 2: Create separate registries
    print("\n" + "=" * 70)
    print("Step 2: Create and Build Registries")
    print("=" * 70)

    latency_reg = FmhaRegistry("latency")
    latency_reg.register_kernel(latency_cfg)

    throughput_reg = FmhaRegistry("throughput")
    throughput_reg.register_kernel(throughput_cfg)

    print(f"\n  Building 'latency' registry ({len(latency_reg)} kernel) ...")
    t0 = time.perf_counter()
    latency_results = latency_reg.build(verbose=False)
    lat_build_time = time.perf_counter() - t0

    print(f"  Building 'throughput' registry ({len(throughput_reg)} kernel) ...")
    t0 = time.perf_counter()
    throughput_results = throughput_reg.build(verbose=False)
    thr_build_time = time.perf_counter() - t0

    lat_ok = latency_results and latency_results[0].success
    thr_ok = throughput_results and throughput_results[0].success

    print(f"\n  Latency:    {'OK' if lat_ok else 'FAIL'} ({lat_build_time:.1f} s)")
    print(f"  Throughput: {'OK' if thr_ok else 'FAIL'} ({thr_build_time:.1f} s)")

    if not lat_ok and not thr_ok:
        print("  No kernels built -- aborting")
        return 1

    # Step 3: Run same problem through both
    print("\n" + "=" * 70)
    print("Step 3: Run Same Problem Through Both Registries")
    print("=" * 70)

    test_configs = [
        (2, 4, 4, 64, 64, 128, "small"),
        (2, 8, 8, 128, 128, 128, "medium"),
        (2, 8, 8, 256, 256, 128, "large"),
    ]

    validator = FmhaValidator(rtol=args.rtol, atol=args.atol)

    print(f"\n  {'Problem':<12} {'Latency':>18} {'Throughput':>18} {'Match':<6}")
    print("  " + "-" * 60)

    all_match = True

    for batch, hq, hk, sq, sk, hdim, desc in test_configs:
        prob = FmhaProblem(
            batch=batch,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=sq,
            seqlen_k=sk,
            hdim_q=hdim,
            hdim_v=hdim,
        )

        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float16)

        O_ref = cpu_attention_fwd(
            Q.astype(np.float32),
            K.astype(np.float32),
            V.astype(np.float32),
            prob.scale,
        )

        lat_cell = "N/A"
        thr_cell = "N/A"
        results_match = True

        if lat_ok:
            res_lat = latency_results[0].runner.run(Q, K, V, prob)
            if res_lat.success:
                lat_cell = f"{res_lat.tflops:.2f} TFLOPS"
                ok, _, _ = validator.check(res_lat.output, O_ref)
                if not ok:
                    results_match = False

        if thr_ok:
            res_thr = throughput_results[0].runner.run(Q, K, V, prob)
            if res_thr.success:
                thr_cell = f"{res_thr.tflops:.2f} TFLOPS"
                ok, _, _ = validator.check(res_thr.output, O_ref)
                if not ok:
                    results_match = False

        if not results_match:
            all_match = False

        tag = "YES" if results_match else "NO"
        print(f"  {desc:<12} {lat_cell:>18} {thr_cell:>18} {tag:<6}")

    # Step 4: Detailed comparison on a single problem
    print("\n" + "=" * 70)
    print("Step 4: Detailed Comparison (B=2 H=8 S=128 D=128)")
    print("=" * 70)

    prob = FmhaProblem(
        batch=2,
        nhead_q=8,
        nhead_k=8,
        seqlen_q=128,
        seqlen_k=128,
        hdim_q=128,
        hdim_v=128,
    )
    np.random.seed(123)
    Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float16)
    K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float16)
    V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float16)
    O_ref = cpu_attention_fwd(
        Q.astype(np.float32),
        K.astype(np.float32),
        V.astype(np.float32),
        prob.scale,
    )

    for name, results, ok in [
        ("Latency", latency_results, lat_ok),
        ("Throughput", throughput_results, thr_ok),
    ]:
        if not ok:
            print(f"\n  {name}: not available")
            continue
        res = results[0].runner.run(Q, K, V, prob)
        if not res.success:
            print(f"\n  {name}: execution failed")
            continue
        valid, max_abs, max_rel = validator.check(res.output, O_ref)
        print(f"\n  {name}:")
        print(f"    Time:     {res.time_ms:.4f} ms")
        print(f"    TFLOPS:   {res.tflops:.2f}")
        print(f"    Max Abs:  {max_abs:.2e}")
        print(f"    Max Rel:  {max_rel:.2e}")
        print(f"    Valid:    {valid}")

    # Cleanup
    for results in [latency_results, throughput_results]:
        for r in results:
            if r.runner:
                r.runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("Multi-Registry Pattern:")
    print("=" * 70)
    print("  1. Create FmhaRegistry per optimization target")
    print("  2. Register target-specific FmhaKernelConfig in each")
    print("  3. Build both registries")
    print("  4. Route problems to the best registry")
    print("  5. Compare results for correctness")
    print("=" * 70)

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
