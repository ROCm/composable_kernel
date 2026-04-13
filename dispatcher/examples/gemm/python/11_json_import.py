#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 11: JSON-based Kernel Configuration Import

Demonstrates loading kernel configurations from JSON files, similar to tile_engine.
This enables easy customization of kernel sets without modifying code.

Key Features:
  - Load tile configs from JSON (compatible with tile_engine format)
  - Generate kernel sets from configuration
  - Use arch_filter validation on loaded configs
  - Export to C++ DECL_KERNEL_SET format


Usage:
    python3 11_json_import.py
    python3 11_json_import.py --config my_kernels.json
    python3 11_json_import.py --export-cpp
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
    load_kernel_configs,
    KernelConfig,
    generate_cpp_kernel_set_declaration,
)

from ctypes_utils import (  # noqa: E402
    KernelConfig as DispatcherKernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    reset_for_example,
    validate_kernel_config,
    detect_gpu_arch,
)

# Sample JSON configuration (embedded for demonstration)
SAMPLE_JSON_CONFIG = {
    "_comment": "Sample kernel configuration for GEMM",
    "kernel_set_name": "inference_kernels",
    "datatype": {"a": "fp16", "b": "fp16", "c": "fp16", "acc": "fp32"},
    "layout": "rcr",
    "tile_config": {
        "tile_m": {"values": [128, 256]},
        "tile_n": {"values": [128, 256]},
        "tile_k": {"values": [32]},
        "warp_m": {"values": [2]},
        "warp_n": {"values": [2]},
        "warp_k": {"values": [1]},
        "warp_tile_m": {"values": [32]},
        "warp_tile_n": {"values": [32]},
        "warp_tile_k": {"values": [16]},
    },
    "trait_config": {
        "pipeline": {"values": ["compv4"]},
        "scheduler": {"values": ["intrawave"]},
        "epilogue": {"values": ["cshuffle"]},
        "pad_m": {"values": [False]},
        "pad_n": {"values": [False]},
        "pad_k": {"values": [False]},
    },
    "gpu_targets": ["gfx942"],
}


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def convert_to_dispatcher_config(
    config: KernelConfig, arch: str = "gfx942"
) -> DispatcherKernelConfig:
    """Convert kernel_config_loader.KernelConfig to dispatcher KernelConfig"""
    return DispatcherKernelConfig(
        dtype_a=config.dtype_a,
        dtype_b=config.dtype_b,
        dtype_c=config.dtype_c,
        dtype_acc=config.dtype_acc,
        tile_m=config.tile.tile_m,
        tile_n=config.tile.tile_n,
        tile_k=config.tile.tile_k,
        wave_m=config.tile.warp_m,
        wave_n=config.tile.warp_n,
        wave_k=config.tile.warp_k,
        warp_m=config.tile.warp_tile_m,
        warp_n=config.tile.warp_tile_n,
        warp_k=config.tile.warp_tile_k,
        pipeline=config.trait.pipeline,
        scheduler=config.trait.scheduler,
        epilogue=config.trait.epilogue,
        pad_m=config.trait.pad_m,
        pad_n=config.trait.pad_n,
        pad_k=config.trait.pad_k,
        gfx_arch=arch,
        variant=config.variant,
    )


def main():
    parser = argparse.ArgumentParser(
        description="JSON Kernel Configuration Import Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 11_json_import.py                  # Use embedded sample config
  python3 11_json_import.py --config my.json # Load from file
  python3 11_json_import.py --export-cpp     # Generate C++ declarations
  python3 11_json_import.py --validate       # Validate configs against arch
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
        help="Export kernel set as C++ DECL_KERNEL_SET",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all configurations against arch filter",
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="Target GPU architecture (auto-detected from rocminfo, override with --arch gfxNNN)",
    )
    args = parser.parse_args()

    reset_for_example()

    print_section("Example 11: JSON Kernel Configuration Import")

    # =========================================================================
    # Step 1: Load configuration from JSON
    # =========================================================================
    print("Step 1: Load Kernel Configuration from JSON")
    print("-" * 50)

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"  ERROR: Config file not found: {config_path}")
            return 1
        print(f"  Loading from: {config_path}")
        config_set = load_kernel_configs(config_path)
    else:
        # Use embedded sample config
        print("  Using embedded sample configuration")
        # Write to temp file and load
        temp_path = Path("/tmp/sample_gemm_config.json")
        with open(temp_path, "w") as f:
            json.dump(SAMPLE_JSON_CONFIG, f, indent=2)
        config_set = load_kernel_configs(temp_path)

    print(f"\n  Kernel Set Name: {config_set.name}")
    print(
        f"  Data Types: A={config_set.dtype_a}, B={config_set.dtype_b}, C={config_set.dtype_c}"
    )
    print(f"  Layout: {config_set.layout}")
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

    # =========================================================================
    # Step 3: Generate and display kernel names
    # =========================================================================
    print("\nStep 3: Generated Kernel Names")
    print("-" * 50)

    configs = list(config_set.generate_configs())
    for i, config in enumerate(configs[:5]):
        print(f"  {i + 1}. {config.kernel_name()}")
    if len(configs) > 5:
        print(f"  ... and {len(configs) - 5} more configurations")

    # =========================================================================
    # Step 4: Validate against arch filter (optional)
    # =========================================================================
    if args.validate:
        print("\nStep 4: Architecture Validation")
        print("-" * 50)

        valid_count = 0
        invalid_count = 0

        for config in configs:
            disp_config = convert_to_dispatcher_config(config, args.arch)
            result = validate_kernel_config(disp_config)

            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if invalid_count <= 3:  # Show first 3 invalid
                    print(f"\n  FAIL Invalid: {config.kernel_name()}")
                    for error in result.errors:
                        print(f"    Error: {error}")

        print("\n  Validation Summary:")
        print(f"    OK Valid: {valid_count}")
        print(f"    FAIL Invalid: {invalid_count}")
        print(f"    Total: {len(configs)}")

    # =========================================================================
    # Step 5: Export to C++ (optional)
    # =========================================================================
    if args.export_cpp:
        print("\nStep 5: C++ Export")
        print("-" * 50)
        print("\n  // Generated DECL_KERNEL_SET from JSON config:")
        print("  // " + "=" * 56)
        cpp_code = generate_cpp_kernel_set_declaration(config_set)
        for line in cpp_code.split("\n"):
            print(f"  {line}")

    # =========================================================================
    # Step 6: Use first config with dispatcher (demo)
    # =========================================================================
    print("\nStep 6: Dispatcher Integration Demo")
    print("-" * 50)

    if configs:
        first_config = configs[0]
        disp_config = convert_to_dispatcher_config(first_config, args.arch)

        print(
            f"\n  Using first config: {first_config.tile.tile_m}x{first_config.tile.tile_n}x{first_config.tile.tile_k}"
        )

        setup = setup_gemm_dispatcher(
            disp_config, registry_name="json_import", verbose=False
        )
        if setup.success:
            print("  OK Dispatcher setup successful")
            print(
                f"    Kernel header: {setup.kernel_header.name if setup.kernel_header else 'N/A'}"
            )
        else:
            print(f"  WARNING Dispatcher setup: {setup.error}")
            print("    (This is expected if kernels aren't generated)")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")
    print("  JSON configuration allows easy kernel set customization:")
    print("    - Define tile sizes and ranges")
    print("    - Specify trait combinations (pipeline, scheduler, etc.)")
    print("    - Target multiple GPU architectures")
    print("    - Export to C++ DECL_KERNEL_SET for static compilation")
    print()
    print("  JSON Format (tile_engine compatible):")
    print('    {"tile_config": {"tile_m": {"values": [128, 256]}, ...},')
    print('     "trait_config": {"pipeline": {"values": ["compv4"]}, ...}}')
    print()
    print("  Usage:")
    print("    config_set = load_kernel_configs('my_kernels.json')")
    print("    for config in config_set.generate_configs():")
    print("        # Use config for codegen or dispatcher setup")

    cleanup_gemm()
    return 0


if __name__ == "__main__":
    sys.exit(main())
