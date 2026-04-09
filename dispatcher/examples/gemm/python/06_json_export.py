#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 06: JSON Export

Exports registry configuration to JSON.


Usage:
    python3 06_json_export.py
    python3 06_json_export.py --help
    python3 06_json_export.py --output my_kernels.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    detect_gpu_arch,
)


def main():
    parser = argparse.ArgumentParser(
        description="JSON Export Example - exports registry to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 06_json_export.py                   # Default output to kernels.json
  python3 06_json_export.py --output my.json  # Custom output file
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="kernels.json",
        help="Output JSON file (default: kernels.json)",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    reset_for_example()

    print("=" * 60)
    print("Example 06: JSON Export")
    print("=" * 60)

    # =========================================================================
    # Step 1: Setup dispatcher
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")

    config = KernelConfig(
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c=args.dtype,
        tile_m=128,
        tile_n=128,
        tile_k=32,
        gfx_arch=args.arch,
    )

    setup = setup_gemm_dispatcher(config, registry_name="export_demo", verbose=True)
    if not setup.success:
        print(f"  ERROR: {setup.error}")
        return 1

    # =========================================================================
    # Step 2: Define additional configs for export
    # =========================================================================
    print("\nStep 2: Define Additional Configs")

    configs = [
        config,
        KernelConfig(
            dtype_a=args.dtype,
            dtype_b=args.dtype,
            dtype_c=args.dtype,
            tile_m=256,
            tile_n=256,
            tile_k=64,
            gfx_arch=args.arch,
        ),
        KernelConfig(
            dtype_a=args.dtype,
            dtype_b=args.dtype,
            dtype_c=args.dtype,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            gfx_arch=args.arch,
        ),
    ]

    for cfg in configs:
        print(f"  - {cfg.tile_str}")

    # =========================================================================
    # Step 3: Export to JSON
    # =========================================================================
    print("\nStep 3: Export to JSON")

    export_data = {
        "registry": setup.registry.name,
        "kernel_count": len(configs),
        "kernels": [],
    }

    for cfg in configs:
        kernel_info = {
            "tile": cfg.tile_str,
            "dtypes": {"A": cfg.dtype_a, "B": cfg.dtype_b, "C": cfg.dtype_c},
            "layout": cfg.layout,
            "pipeline": cfg.pipeline,
            "target": cfg.gfx_arch,
        }
        export_data["kernels"].append(kernel_info)

    # Include C++ library info
    if setup.lib:
        cpp_json = setup.lib.export_registry_json()
        try:
            export_data["cpp_registry"] = json.loads(cpp_json)
        except json.JSONDecodeError:
            pass

    json_str = json.dumps(export_data, indent=2)

    with open(args.output, "w") as f:
        f.write(json_str)
    print(f"  Saved to: {args.output}")

    # Preview
    print("\nStep 4: Preview")
    print("-" * 60)
    print(json_str[:500] + ("..." if len(json_str) > 500 else ""))
    print("-" * 60)

    # Cleanup
    cleanup_gemm()

    print("\n" + "=" * 60)
    print("JSON Export complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
