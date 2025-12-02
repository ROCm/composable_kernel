#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 06: JSON Export

Exports registry configuration to JSON.

Complexity: ★★☆☆☆

Usage:
    python3 06_json_export.py [output.json]
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
)


def main():
    reset_for_example()

    print("=" * 60)
    print("Example 06: JSON Export")
    print("=" * 60)

    output_file = sys.argv[1] if len(sys.argv) > 1 else "kernels.json"

    # =========================================================================
    # Step 1: Setup dispatcher
    # =========================================================================
    print("\nStep 1: Setup Dispatcher")

    config = KernelConfig(dtype_a="fp16", tile_m=128, tile_n=128, tile_k=32)

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
        KernelConfig(dtype_a="fp16", tile_m=256, tile_n=256, tile_k=64),
        KernelConfig(dtype_a="fp16", tile_m=64, tile_n=64, tile_k=32),
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

    with open(output_file, "w") as f:
        f.write(json_str)
    print(f"  Saved to: {output_file}")

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
