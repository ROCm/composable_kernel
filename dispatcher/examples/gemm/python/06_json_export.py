#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 06: JSON Export

Exports registry configuration to JSON using explicit API.

Complexity: ★★☆☆☆

Usage:
    python3 06_json_export.py [output.json]
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from ctypes_utils import (
    KernelConfig,
    CodegenRunner,
    DispatcherLib,
    Registry,
)


def main():
    print("=" * 60)
    print("Example 06: JSON Export")
    print("=" * 60)

    output_file = sys.argv[1] if len(sys.argv) > 1 else "kernels.json"

    # =========================================================================
    # Step 1: Define multiple kernel configs
    # =========================================================================
    print("\nStep 1: Define Kernel Configurations")

    configs = [
        KernelConfig(tile_m=256, tile_n=256, tile_k=64, pipeline="compv4"),
        KernelConfig(tile_m=128, tile_n=128, tile_k=32, pipeline="compv4"),
        KernelConfig(tile_m=64, tile_n=64, tile_k=32, pipeline="compv3"),
    ]

    for cfg in configs:
        print(f"  - {cfg.tile_str} ({cfg.pipeline})")

    # =========================================================================
    # Step 2: Create registry and register configs
    # =========================================================================
    print("\nStep 2: Create Registry")

    registry = Registry(name="export_demo")
    for cfg in configs:
        registry.register_kernel(cfg)

    print(f"  {registry}")

    # =========================================================================
    # Step 3: Generate kernels and load library
    # =========================================================================
    print("\nStep 3: Setup")

    codegen = CodegenRunner()
    codegen.generate("standard")

    lib = DispatcherLib.auto()
    if lib:
        registry.bind_library(lib)
        print(f"  Library kernel: {lib.get_kernel_name()}")

    # =========================================================================
    # Step 4: Export to JSON
    # =========================================================================
    print("\nStep 4: Export to JSON")

    # Build export data from our configs
    export_data = {
        "registry": registry.name,
        "kernel_count": len(configs),
        "kernels": [],
    }

    for cfg in configs:
        kernel_info = {
            "tile": cfg.tile_str,
            "dtypes": {
                "A": cfg.dtype_a,
                "B": cfg.dtype_b,
                "C": cfg.dtype_c,
                "Acc": cfg.dtype_acc,
            },
            "layout": cfg.layout,
            "pipeline": cfg.pipeline,
            "scheduler": cfg.scheduler,
            "block_size": cfg.block_size,
            "padding": {
                "M": cfg.pad_m,
                "N": cfg.pad_n,
                "K": cfg.pad_k,
            },
            "target": cfg.gfx_arch,
        }
        export_data["kernels"].append(kernel_info)

    # Also include C++ library export if available
    if lib:
        cpp_json = lib.export_registry_json()
        try:
            cpp_data = json.loads(cpp_json)
            export_data["cpp_registry"] = cpp_data
        except json.JSONDecodeError:
            pass

    json_str = json.dumps(export_data, indent=2)

    # Save
    with open(output_file, "w") as f:
        f.write(json_str)
    print(f"  Saved to: {output_file}")

    # =========================================================================
    # Step 5: Preview
    # =========================================================================
    print("\nStep 5: Preview")
    print("-" * 60)
    print(json_str[:800])
    if len(json_str) > 800:
        print("...")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("JSON Export complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
