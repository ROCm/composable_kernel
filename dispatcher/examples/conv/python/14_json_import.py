#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example 14: JSON-based Conv Kernel Configuration Import

Demonstrates loading convolution kernel configurations from JSON files.
Supports all conv-specific parameters including:
  - Tile dimensions (tile_m/n/k, warp_m/n/k, warp_tile_m/n/k)
  - Pipeline/scheduler/epilogue traits
  - Vector sizes for memory access
  - Occupancy parameters (block_per_cu, num_wave_groups)
  - Padding and double buffering options
  - Group merging for grouped convolution

Complexity: ★★★☆☆

Usage:
    python3 14_json_import.py
    python3 14_json_import.py --config my_conv_kernels.json
    python3 14_json_import.py --export-cpp
"""

import sys
import argparse
import json
from pathlib import Path

# Add codegen to path for kernel_config_loader
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir.parent.parent.parent / "codegen"))
sys.path.insert(0, str(script_dir.parent.parent.parent / "python"))

from kernel_config_loader import (  # noqa: E402
    load_conv_kernel_configs,
    generate_cpp_conv_kernel_set_declaration,
)

# Sample JSON configuration (embedded for demonstration)
SAMPLE_CONV_CONFIG = {
    "_comment": "Sample conv kernel configuration",
    "kernel_set_name": "conv_inference",
    "datatype": {
        "input": "fp16",
        "weight": "fp16",
        "output": "fp16",
        "acc": "fp32",
    },
    "variant": "forward",
    "ndim": 2,
    "layout": "nhwgc",
    "tile_config": {
        "tile_m": {"values": [16, 128]},
        "tile_n": {"values": [64, 128]},
        "tile_k": {"values": [64]},
        "warp_m": {"values": [1, 2]},
        "warp_n": {"values": [2, 4]},
        "warp_k": {"values": [1]},
        "warp_tile_m": {"values": [16, 32]},
        "warp_tile_n": {"values": [16, 32]},
        "warp_tile_k": {"values": [16, 32]},
    },
    "trait_config": {
        "pipeline": {"values": ["compv3"]},
        "scheduler": {"values": ["intrawave"]},
        "epilogue": {"values": ["cshuffle"]},
        "pad_m": {"values": [True]},
        "pad_n": {"values": [True]},
        "pad_k": {"values": [True]},
        "double_smem_buffer": {"values": [False]},
        "num_groups_to_merge": {"values": [1]},
    },
    "vector_config": {
        "vector_size_a": {"values": [4]},
        "vector_size_b": {"values": [8]},
        "vector_size_c": {"values": [8]},
    },
    "occupancy_config": {
        "block_per_cu": {"values": [1]},
        "num_wave_groups": {"values": [1]},
    },
    "gpu_targets": ["gfx942"],
}


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="JSON Conv Kernel Configuration Import Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 14_json_import.py                  # Use embedded sample config
  python3 14_json_import.py --config my.json # Load from file
  python3 14_json_import.py --export-cpp     # Generate C++ declarations
  python3 14_json_import.py --list-all       # List all generated configs
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (uses embedded sample if not provided)",
    )
    parser.add_argument(
        "--export-cpp",
        action="store_true",
        help="Export kernel set as C++ DECL_CONV_KERNEL_SET",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all generated kernel configurations",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        help="Target GPU architecture (default: gfx942)",
    )
    args = parser.parse_args()

    print_section("Example 14: JSON Conv Kernel Configuration Import")

    # =========================================================================
    # Step 1: Load configuration from JSON
    # =========================================================================
    print("Step 1: Load Conv Kernel Configuration from JSON")
    print("-" * 50)

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"  ERROR: Config file not found: {config_path}")
            return 1
        print(f"  Loading from: {config_path}")
        config_set = load_conv_kernel_configs(config_path)
    else:
        # Use embedded sample config
        print("  Using embedded sample configuration")
        temp_path = Path("/tmp/sample_conv_config.json")
        with open(temp_path, "w") as f:
            json.dump(SAMPLE_CONV_CONFIG, f, indent=2)
        config_set = load_conv_kernel_configs(temp_path)

    print(f"\n  Kernel Set Name: {config_set.name}")
    print(f"  Variant: {config_set.variant}")
    print(f"  Spatial Dims: {config_set.ndim}D")
    print(f"  Layout: {config_set.layout}")
    print(
        f"  Data Types: input={config_set.dtype_input}, weight={config_set.dtype_weight}, output={config_set.dtype_output}"
    )
    print(f"  GPU Targets: {config_set.gpu_targets}")
    print(f"  Total Configurations: {config_set.config_count()}")

    # =========================================================================
    # Step 2: Display configuration details
    # =========================================================================
    print("\nStep 2: Configuration Details")
    print("-" * 50)

    print("\n  Tile Configurations:")
    print(f"    tile_m: {config_set.tile_m_values}")
    print(f"    tile_n: {config_set.tile_n_values}")
    print(f"    tile_k: {config_set.tile_k_values}")
    print(
        f"    warp (wave): {config_set.warp_m_values}x{config_set.warp_n_values}x{config_set.warp_k_values}"
    )
    print(
        f"    warp_tile: {config_set.warp_tile_m_values}x{config_set.warp_tile_n_values}x{config_set.warp_tile_k_values}"
    )

    print("\n  Trait Configurations:")
    print(f"    pipeline: {config_set.pipeline_values}")
    print(f"    scheduler: {config_set.scheduler_values}")
    print(f"    epilogue: {config_set.epilogue_values}")
    print(
        f"    padding: m={config_set.pad_m_values}, n={config_set.pad_n_values}, k={config_set.pad_k_values}"
    )
    print(f"    double_smem_buffer: {config_set.double_smem_buffer_values}")
    print(f"    num_groups_to_merge: {config_set.num_groups_to_merge_values}")

    print("\n  Vector Configurations:")
    print(f"    vector_size_a: {config_set.vector_size_a_values}")
    print(f"    vector_size_b: {config_set.vector_size_b_values}")
    print(f"    vector_size_c: {config_set.vector_size_c_values}")

    print("\n  Occupancy Configurations:")
    print(f"    block_per_cu: {config_set.block_per_cu_values}")
    print(f"    num_wave_groups: {config_set.num_wave_groups_values}")

    # =========================================================================
    # Step 3: Generate and display kernel names
    # =========================================================================
    print("\nStep 3: Generated Kernel Names")
    print("-" * 50)

    configs = list(config_set.generate_configs())

    if args.list_all:
        for i, config in enumerate(configs):
            print(f"  {i + 1}. {config.kernel_name()}")
    else:
        for i, config in enumerate(configs[:5]):
            print(f"  {i + 1}. {config.kernel_name()}")
        if len(configs) > 5:
            print(f"  ... and {len(configs) - 5} more configurations")
            print("  (use --list-all to see all)")

    # =========================================================================
    # Step 4: Export to C++ (optional)
    # =========================================================================
    if args.export_cpp:
        print("\nStep 4: C++ Export")
        print("-" * 50)
        print("\n  // Generated DECL_CONV_KERNEL_SET from JSON config:")
        print("  // " + "=" * 56)
        cpp_code = generate_cpp_conv_kernel_set_declaration(config_set)
        for line in cpp_code.split("\n"):
            print(f"  {line}")

    # =========================================================================
    # Step 5: Show config dict for first kernel
    # =========================================================================
    print("\nStep 5: Sample Config Dictionary (for codegen)")
    print("-" * 50)

    if configs:
        first_config = configs[0]
        config_dict = first_config.to_dict()
        print("\n  First configuration as dict:")
        for key, value in config_dict.items():
            print(f"    {key}: {value}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")
    print("  JSON configuration for convolution kernels supports:")
    print()
    print("  Tile Parameters:")
    print("    tile_m/n/k       - Block tile dimensions")
    print("    warp_m/n/k       - Warps per block (wave configuration)")
    print("    warp_tile_m/n/k  - Elements per warp")
    print()
    print("  Trait Parameters:")
    print("    pipeline         - mem, compv3, compv4, compv5")
    print("    scheduler        - intrawave, interwave")
    print("    epilogue         - cshuffle, default")
    print("    pad_m/n/k        - Enable padding for arbitrary sizes")
    print("    double_smem_buffer - Double buffering for pipelining")
    print("    num_groups_to_merge - Group merging for grouped conv")
    print()
    print("  Vector/Occupancy:")
    print("    vector_size_a/b/c - Memory access vector sizes")
    print("    block_per_cu      - Blocks per compute unit")
    print("    num_wave_groups   - Wave groups for scheduling")
    print()
    print("  Usage:")
    print("    config_set = load_conv_kernel_configs('my_kernels.json')")
    print("    for config in config_set.generate_configs():")
    print("        # Use config for codegen or dispatcher setup")

    return 0


if __name__ == "__main__":
    sys.exit(main())
