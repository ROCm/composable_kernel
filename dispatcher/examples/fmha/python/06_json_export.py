#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 06: JSON Export

Builds an FMHA kernel via setup_fmha_dispatcher, then exports the
registry configuration to JSON for inspection or reuse.

Usage:
    python3 06_json_export.py
    python3 06_json_export.py --help
    python3 06_json_export.py --output fmha_kernels.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from fmha_utils import (
    FmhaKernelConfig,
    setup_fmha_dispatcher,
    detect_gpu_arch,
)


def main():
    parser = argparse.ArgumentParser(
        description="JSON Export Example - export FMHA registry to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 06_json_export.py                           # Default output
  python3 06_json_export.py --output fmha_kernels.json  # Custom file
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="fmha_kernels.json",
        help="Output JSON file (default: fmha_kernels.json)",
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 06: JSON Export")
    print("=" * 70)

    # Step 1: Define FMHA kernel configurations
    print("\nStep 1: Define Kernel Configurations")

    configs = [
        FmhaKernelConfig(
            family="fwd",
            data_type="fp16",
            hdim_q=128,
            hdim_v=128,
            pipeline="qr_async",
            # Stage 0 (Q*K^T): seqlen_q x seqlen_k x hdim_q
            tile_m0=128,
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
            warp_m0=32,
            warp_n0=32,
            warp_k0=16,
            warp_m1=32,
            warp_n1=32,
            warp_k1=16,
            gfx_arch=args.arch,
        ),
        FmhaKernelConfig(
            family="fwd",
            data_type="fp16",
            hdim_q=128,
            hdim_v=128,
            pipeline="qr",
            tile_m0=64,
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
            gfx_arch=args.arch,
        ),
        FmhaKernelConfig(
            family="fwd",
            data_type="fp16",
            hdim_q=64,
            hdim_v=64,
            pipeline="qr_async",
            tile_m0=128,
            tile_n0=64,
            tile_k0=32,
            tile_n1=64,
            tile_k1=32,
            tile_k0max=64,
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
            gfx_arch=args.arch,
        ),
    ]

    for i, cfg in enumerate(configs, 1):
        print(f"  [{i}] {cfg.name}: pipeline={cfg.pipeline}, hdim={cfg.hdim_q}")

    # Step 2: Build via setup_fmha_dispatcher
    print("\n" + "=" * 70)
    print("Step 2: Build Kernel (JIT)")
    print("=" * 70)

    setup = setup_fmha_dispatcher(configs[0], verbose=True)
    if setup.success:
        print(f"  Built: {setup.library_path}")
        print(f"  Time:  {setup.build_time_s:.1f} s")
    else:
        print(f"  Build skipped/failed: {setup.error}")
        print("  (Proceeding with config export only)")

    # Step 3: Export to JSON
    print("\n" + "=" * 70)
    print("Step 3: Export to JSON")
    print("=" * 70)

    export_data = {
        "registry": "fmha_export",
        "arch": args.arch,
        "kernel_count": len(configs),
        "kernels": [],
    }

    for cfg in configs:
        kernel_info = {
            "name": cfg.name,
            "family": cfg.family,
            "data_type": cfg.data_type,
            "hdim_q": cfg.hdim_q,
            "hdim_v": cfg.hdim_v,
            "pipeline": cfg.pipeline,
            "tile": list(cfg.tile),
            "wave": list(cfg.wave),
            "warp": list(cfg.warp),
            "padding": list(cfg.padding),
            "mode": cfg.mode,
            "target": cfg.gfx_arch,
            "codegen_json": json.loads(cfg.to_codegen_json()),
        }
        export_data["kernels"].append(kernel_info)

    json_str = json.dumps(export_data, indent=2)

    with open(args.output, "w") as f:
        f.write(json_str)
    print(f"  Saved to: {args.output}")

    file_size = Path(args.output).stat().st_size
    print(f"  File size: {file_size:,} bytes")
    print(f"  Kernel count: {len(configs)}")

    # Step 4: Preview
    print("\n" + "=" * 70)
    print("Step 4: JSON Preview")
    print("=" * 70)
    preview = json_str[:500]
    if len(json_str) > 500:
        preview += "\n  ..."
    print(preview)

    print("\n" + "=" * 70)
    print("JSON Export complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
