#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 08: Kernel Selection Heuristics

Demonstrates how to build multiple FMHA kernels with different tile
sizes and select the best kernel for a given problem.  Shows that
smaller tiles tend to be better for short sequences while larger tiles
are better for long sequences.

Usage:
    python3 08_heuristics.py
    python3 08_heuristics.py --help
    python3 08_heuristics.py --arch gfx950
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    FmhaRegistry,
    detect_gpu_arch,
)


@dataclass
class TileProfile:
    """A kernel profile tagged with a human-readable label."""

    label: str
    config: FmhaKernelConfig
    category: str  # "small", "medium", "large"


def build_tile_profiles(arch: str) -> List[TileProfile]:
    """Create kernel configs with varying tile sizes."""
    return [
        TileProfile(
            label="small_64x64",
            category="small",
            config=FmhaKernelConfig(
                family="fwd",
                data_type="fp16",
                hdim_q=128,
                hdim_v=128,
                pipeline="qr_async",
                # Stage 0 (Q*K^T): seqlen_q x seqlen_k x hdim_q
                tile_m0=64,
                tile_n0=64,
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
                warp_k0=16,
                warp_m1=16,
                warp_n1=16,
                warp_k1=16,
                gfx_arch=arch,
            ),
        ),
        TileProfile(
            label="medium_128x128",
            category="medium",
            config=FmhaKernelConfig(
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
            ),
        ),
        TileProfile(
            label="large_128x256",
            category="large",
            config=FmhaKernelConfig(
                family="fwd",
                data_type="fp16",
                hdim_q=128,
                hdim_v=128,
                pipeline="qr_async",
                tile_m0=128,
                tile_n0=256,
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
            ),
        ),
        TileProfile(
            label="medium_qr_128x128",
            category="medium",
            config=FmhaKernelConfig(
                family="fwd",
                data_type="fp16",
                hdim_q=128,
                hdim_v=128,
                pipeline="qr",
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
                pad_s=False,
                pad_sk=False,
                pad_d=True,
                pad_dv=True,
                gfx_arch=arch,
            ),
        ),
    ]


def select_kernel_heuristic(seqlen: int, profiles: List[TileProfile]) -> TileProfile:
    """Simple heuristic: pick tile size category based on sequence length."""
    if seqlen <= 64:
        target = "small"
    elif seqlen <= 256:
        target = "medium"
    else:
        target = "large"

    candidates = [p for p in profiles if p.category == target]
    if not candidates:
        candidates = profiles
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="FMHA Heuristics - kernel selection by problem size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 08_heuristics.py
  python3 08_heuristics.py --arch gfx950
        """,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 08: Kernel Selection Heuristics")
    print("=" * 70)

    # Step 1: Build kernel pool
    print("\nStep 1: Build Kernel Pool")
    profiles = build_tile_profiles(args.arch)

    reg = FmhaRegistry("heuristic_pool")
    for p in profiles:
        reg.register_kernel(p.config)

    print(f"  Profiles: {len(profiles)}")
    for i, p in enumerate(profiles, 1):
        tile_str = f"{p.config.tile[0]}x{p.config.tile[1]}"
        print(
            f"    [{i}] {p.label:<25} tile={tile_str:<10} pipeline={p.config.pipeline}"
        )

    print("\n  Building kernels ...")
    build_results = reg.build(verbose=False)
    built = sum(1 for r in build_results if r.success)
    print(f"  Built: {built}/{len(profiles)}")

    for i, r in enumerate(build_results):
        tag = "OK" if r.success else f"FAIL: {r.error[:40]}"
        print(f"    [{i + 1}] {profiles[i].label}: {tag}")

    if built == 0:
        print("  No kernels built -- aborting")
        return 1

    # Step 2: Run each kernel on multiple sequence lengths
    print("\n" + "=" * 70)
    print("Step 2: Benchmark Across Sequence Lengths")
    print("=" * 70)

    test_seqlens = [32, 64, 128, 256, 512]

    header = f"  {'SeqLen':>7}"
    for p in profiles:
        header += f" | {p.label:>18}"
    header += " | {'Best':>18}"
    print(f"\n  {'SeqLen':>7}", end="")
    for p in profiles:
        print(f" | {p.label:>18}", end="")
    print(f" | {'Best':>18}")
    print("  " + "-" * (10 + 21 * len(profiles) + 22))

    for seqlen in test_seqlens:
        prob = FmhaProblem(
            batch=2,
            nhead_q=8,
            nhead_k=8,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            hdim_q=128,
            hdim_v=128,
        )

        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float16)

        row = f"  {seqlen:>7}"
        best_tflops = 0.0
        best_label = "---"

        for j, (p, r) in enumerate(zip(profiles, build_results)):
            if not r.success or r.runner is None:
                row += f" | {'N/A':>18}"
                continue

            res = r.runner.run(Q, K, V, prob)
            if res.success:
                cell = f"{res.tflops:.2f} TFLOPS"
                row += f" | {cell:>18}"
                if res.tflops > best_tflops:
                    best_tflops = res.tflops
                    best_label = p.label
            else:
                row += f" | {'ERR':>18}"

        row += f" | {best_label:>18}"
        print(row)

    # Step 3: Demonstrate heuristic selection
    print("\n" + "=" * 70)
    print("Step 3: Heuristic Selection Demo")
    print("=" * 70)

    print(f"\n  {'SeqLen':>7} {'Selected':>25} {'TFLOPS':>10} {'Status':<6}")
    print("  " + "-" * 55)

    for seqlen in test_seqlens:
        selected = select_kernel_heuristic(seqlen, profiles)
        idx = profiles.index(selected)
        r = build_results[idx]

        if not r.success or r.runner is None:
            print(f"  {seqlen:>7} {selected.label:>25} {'---':>10} {'SKIP':<6}")
            continue

        prob = FmhaProblem(
            batch=2,
            nhead_q=8,
            nhead_k=8,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            hdim_q=128,
            hdim_v=128,
        )
        np.random.seed(42)
        Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float16)

        res = r.runner.run(Q, K, V, prob)
        if res.success:
            print(f"  {seqlen:>7} {selected.label:>25} {res.tflops:>10.2f} {'OK':<6}")
        else:
            print(f"  {seqlen:>7} {selected.label:>25} {'---':>10} {'FAIL':<6}")

    # Cleanup
    for r in build_results:
        if r.runner:
            r.runner.cleanup()

    print("\n" + "=" * 70)
    print("Heuristic Insight:")
    print("  - Small tiles: low overhead for short sequences")
    print("  - Large tiles: high throughput for long sequences")
    print("  - Pipeline choice also matters (qr vs qr_async)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
