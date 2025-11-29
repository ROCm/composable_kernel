#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 08: Convolution Registry JSON Export

Demonstrates exporting the convolution kernel registry to JSON format,
similar to GEMM 06_json_export.py.

Usage:
    python3 08_json_export.py
"""

import json
from datetime import datetime
from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ArchInfo,
    ConvKernelConfig,
    ConvRegistry,
)


def export_kernel_config_to_dict(config: ConvKernelConfig) -> dict:
    """Export a single kernel config to dictionary"""
    sig = config.signature
    algo = config.algorithm
    arch = config.arch

    return {
        "name": config.name(),
        "signature": {
            "dtype_in": sig.dtype_in,
            "dtype_wei": sig.dtype_wei,
            "dtype_out": sig.dtype_out,
            "dtype_acc": sig.dtype_acc,
            "layout": sig.layout,
            "direction": sig.direction,
            "num_dims": sig.num_dims,
            "groups": sig.groups,
            "specialization": sig.specialization,
        },
        "algorithm": {
            "tile": {
                "n": algo.tile_n,
                "k": algo.tile_k,
                "c": algo.tile_c,
            },
            "tile_output": {
                "ho": algo.tile_ho,
                "wo": algo.tile_wo,
            },
            "wave": {
                "m": algo.wave_m,
                "n": algo.wave_n,
                "k": algo.wave_k,
            },
            "warp": {
                "m": algo.warp_m,
                "n": algo.warp_n,
                "k": algo.warp_k,
            },
            "pipeline": algo.pipeline,
            "scheduler": algo.scheduler,
            "epilogue": algo.epilogue,
            "padding": algo.padding,
            "block_size": algo.block_size,
        },
        "arch": {
            "name": arch.name,
            "supports_mfma_fp16": arch.supports_mfma_fp16(),
            "supports_wmma": arch.supports_wmma(),
        },
    }


def export_registry_to_json(registry: ConvRegistry) -> dict:
    """Export entire registry to JSON-serializable dictionary"""
    kernels = []

    for config in registry.get_kernels():
        kernels.append(export_kernel_config_to_dict(config))

    # Categorize by direction
    by_direction = {}
    for k in kernels:
        direction = k["signature"]["direction"]
        if direction not in by_direction:
            by_direction[direction] = 0
        by_direction[direction] += 1

    # Categorize by dtype
    by_dtype = {}
    for k in kernels:
        dtype = k["signature"]["dtype_in"]
        if dtype not in by_dtype:
            by_dtype[dtype] = 0
        by_dtype[dtype] += 1

    # Categorize by arch
    by_arch = {}
    for k in kernels:
        arch = k["arch"]["name"]
        if arch not in by_arch:
            by_arch[arch] = 0
        by_arch[arch] += 1

    return {
        "metadata": {
            "registry_name": registry.name,
            "timestamp": datetime.now().isoformat(),
            "total_kernels": len(kernels),
            "export_version": "1.0",
        },
        "statistics": {
            "by_direction": by_direction,
            "by_dtype": by_dtype,
            "by_arch": by_arch,
        },
        "kernels": kernels,
    }


def main():
    print("=" * 70)
    print("Example 08: Convolution Registry JSON Export")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Create registry with various kernels
    # -------------------------------------------------------------------------
    print("CREATING REGISTRY")
    print("=" * 40)

    registry = ConvRegistry(name="conv_production")

    # Forward kernels - multiple tile sizes
    for tile_k, tile_c in [(64, 64), (128, 128), (256, 256)]:
        sig = ConvSignature()
        sig.dtype("fp16")
        sig.layout = "nhwc"
        sig.direction = "forward"
        sig.num_dims = 2

        algo = ConvAlgorithm()
        algo.tile(1, tile_k, tile_c)
        algo.wave(2, 2, 1)
        algo.pipeline = "compv4"
        algo.scheduler = "intrawave"

        registry.register_kernel(
            ConvKernelConfig(
                signature=sig, algorithm=algo, arch=ArchInfo(name="gfx942")
            )
        )

    # Backward data kernels
    sig = ConvSignature()
    sig.dtype("fp16")
    sig.direction = "bwd_data"

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)

    registry.register_kernel(
        ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name="gfx942"))
    )

    # Backward weight kernels
    sig = ConvSignature()
    sig.dtype("fp16")
    sig.direction = "bwd_weight"

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)

    registry.register_kernel(
        ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name="gfx942"))
    )

    # BF16 forward kernel
    sig = ConvSignature()
    sig.dtype("bf16")
    sig.direction = "forward"

    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)

    registry.register_kernel(
        ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name="gfx942"))
    )

    print(f"Registry: {registry}")
    print(f"Total kernels: {registry.kernel_count}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Export to JSON
    # -------------------------------------------------------------------------
    print("JSON EXPORT")
    print("=" * 40)
    print()

    export_data = export_registry_to_json(registry)
    json_str = json.dumps(export_data, indent=2)

    print(json_str)
    print()

    # -------------------------------------------------------------------------
    # Step 3: Show statistics
    # -------------------------------------------------------------------------
    print("EXPORT STATISTICS")
    print("=" * 40)
    print()

    stats = export_data["statistics"]

    print("By Direction:")
    for direction, count in stats["by_direction"].items():
        print(f"  {direction}: {count}")
    print()

    print("By Data Type:")
    for dtype, count in stats["by_dtype"].items():
        print(f"  {dtype}: {count}")
    print()

    print("By Architecture:")
    for arch, count in stats["by_arch"].items():
        print(f"  {arch}: {count}")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Demonstrate kernel lookup
    # -------------------------------------------------------------------------
    print("KERNEL LOOKUP FROM JSON")
    print("=" * 40)
    print()

    # Parse JSON back
    parsed = json.loads(json_str)

    # Find all forward fp16 kernels
    forward_fp16 = [
        k
        for k in parsed["kernels"]
        if k["signature"]["direction"] == "forward"
        and k["signature"]["dtype_in"] == "fp16"
    ]

    print(f"Found {len(forward_fp16)} forward fp16 kernels:")
    for k in forward_fp16:
        tile = k["algorithm"]["tile"]
        print(f"  - {k['name']}: tile={tile['k']}x{tile['c']}")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Save to file example
    # -------------------------------------------------------------------------
    print("SAVE TO FILE")
    print("=" * 40)
    print()

    # Show how to save
    print("To save the registry to a file:")
    print()
    print("  with open('conv_registry.json', 'w') as f:")
    print("      json.dump(export_data, f, indent=2)")
    print()
    print("To load the registry from a file:")
    print()
    print("  with open('conv_registry.json', 'r') as f:")
    print("      data = json.load(f)")
    print()

    print("=" * 70)
    print("JSON export completed!")


if __name__ == "__main__":
    main()
