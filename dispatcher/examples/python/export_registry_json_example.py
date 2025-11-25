#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example: Export Dispatcher Registry to JSON

Demonstrates how to export all registered kernels to JSON format,
similar to the tile engine benchmarking JSON export.

This provides comprehensive kernel metadata including:
- Kernel identifiers and names
- Tile shapes (M, N, K dimensions)
- Wave configurations
- Pipeline and scheduler types
- Data types and layouts
- Statistics by kernel type

Usage:
    python3 export_registry_json_example.py [--output kernels.json]
"""

import sys
import json
import argparse
import ctypes
from pathlib import Path
from datetime import datetime


def find_dispatcher_lib():
    """Find the dispatcher dynamic library"""
    script_dir = Path(__file__).parent

    # Possible locations
    search_paths = [
        script_dir.parent.parent / "build" / "examples" / "libdispatcher_gemm.so",
        script_dir.parent.parent / "build" / "lib" / "libdispatcher_gemm.so",
        script_dir / "libdispatcher_gemm.so",
        Path(
            "/workspace/workspace/composable_kernel/dispatcher/build/examples/libdispatcher_gemm.so"
        ),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_dispatcher_lib():
    """Load the dispatcher library"""
    lib_path = find_dispatcher_lib()
    if lib_path is None:
        raise RuntimeError(
            "Could not find libdispatcher_gemm.so\n"
            "Please build the dispatcher first:\n"
            "  cd dispatcher/build && cmake --build ."
        )

    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.dispatcher_init.argtypes = []
    lib.dispatcher_init.restype = ctypes.c_int

    lib.dispatcher_get_kernel_count.argtypes = []
    lib.dispatcher_get_kernel_count.restype = ctypes.c_int

    # Export registry to JSON - returns pointer to static buffer
    lib.dispatcher_export_registry_json.argtypes = []
    lib.dispatcher_export_registry_json.restype = ctypes.c_char_p

    return lib


def export_registry_json(lib):
    """Export registry to JSON string"""
    json_ptr = lib.dispatcher_export_registry_json()
    if json_ptr:
        return json_ptr.decode("utf-8")
    return None


def create_mock_registry_json():
    """Create a mock registry JSON for demonstration when library not available"""
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_kernels": 0,
            "export_version": "1.0",
            "dispatcher_version": "1.0.0",
            "note": "Mock data - library not loaded",
        },
        "statistics": {
            "by_datatype": {},
            "by_pipeline": {},
            "by_scheduler": {},
            "by_layout": {},
        },
        "kernels": [],
    }


def demo_export_to_string(lib):
    """Demo: Export to JSON string"""
    print("\n" + "=" * 60)
    print("Demo 1: Export to JSON String")
    print("=" * 60)

    json_str = export_registry_json(lib)

    if json_str:
        print(f"✓ Generated JSON string ({len(json_str)} bytes)")

        # Parse and show preview
        data = json.loads(json_str)
        print("\nMetadata:")
        for key, value in data.get("metadata", {}).items():
            print(f"  {key}: {value}")
    else:
        print("✗ Failed to export registry")
        data = create_mock_registry_json()
        print("\nUsing mock data for demonstration")

    return data


def demo_export_to_file(lib, filename):
    """Demo: Export to JSON file"""
    print("\n" + "=" * 60)
    print("Demo 2: Export to JSON File")
    print("=" * 60)

    json_str = export_registry_json(lib)

    if json_str:
        data = json.loads(json_str)
    else:
        data = create_mock_registry_json()

    # Write to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    # Verify file was created
    file_path = Path(filename)
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"✓ File created: {filename} ({size_kb:.1f} KB)")

        print("\nFile structure:")
        print(f"  - metadata: {len(data.get('metadata', {}))} fields")
        if "statistics" in data:
            print(f"  - statistics: {len(data['statistics'])} categories")
        print(f"  - kernels: {len(data.get('kernels', []))} kernels")
    else:
        print(f"✗ Failed to create file: {filename}")


def demo_print_summary(lib):
    """Demo: Print human-readable summary"""
    print("\n" + "=" * 60)
    print("Demo 3: Print Registry Summary")
    print("=" * 60)

    json_str = export_registry_json(lib)

    if json_str:
        data = json.loads(json_str)
    else:
        data = create_mock_registry_json()

    total = data.get("metadata", {}).get("total_kernels", 0)
    print(f"\nTotal kernels: {total}")

    if "statistics" in data and total > 0:
        stats = data["statistics"]

        if "by_datatype" in stats:
            print("\nBy Data Type:")
            for dtype, count in sorted(stats["by_datatype"].items()):
                print(f"  {dtype:20s}: {count:3d}")

        if "by_pipeline" in stats:
            print("\nBy Pipeline:")
            for pipeline, count in sorted(stats["by_pipeline"].items()):
                print(f"  {pipeline:20s}: {count:3d}")

        if "by_scheduler" in stats:
            print("\nBy Scheduler:")
            for scheduler, count in sorted(stats["by_scheduler"].items()):
                print(f"  {scheduler:20s}: {count:3d}")


def demo_list_identifiers(lib):
    """Demo: List all kernel identifiers"""
    print("\n" + "=" * 60)
    print("Demo 4: List Kernel Identifiers")
    print("=" * 60)

    json_str = export_registry_json(lib)

    if json_str:
        data = json.loads(json_str)
    else:
        data = create_mock_registry_json()

    kernels = data.get("kernels", [])
    print(f"\nFound {len(kernels)} kernel identifiers:")

    # Show first 10
    for i, kernel in enumerate(kernels[:10]):
        identifier = kernel.get("identifier", "unknown")
        print(f"  {i + 1:2d}. {identifier}")

    if len(kernels) > 10:
        print(f"  ... and {len(kernels) - 10} more")


def demo_analyze_json(lib):
    """Demo: Analyze JSON data"""
    print("\n" + "=" * 60)
    print("Demo 5: Analyze JSON Data")
    print("=" * 60)

    json_str = export_registry_json(lib)

    if json_str:
        data = json.loads(json_str)
    else:
        data = create_mock_registry_json()

    kernels = data.get("kernels", [])
    if len(kernels) == 0:
        print("\nNo kernels to analyze")
        return

    print("\nAnalyzing kernel configurations...")

    # Find tile size distribution
    tile_sizes = {}
    for kernel in kernels:
        algo = kernel.get("algorithm", {})
        tile = algo.get("tile_shape", {})
        tile_str = f"{tile.get('m', 0)}x{tile.get('n', 0)}x{tile.get('k', 0)}"
        tile_sizes[tile_str] = tile_sizes.get(tile_str, 0) + 1

    print("\nTile size distribution:")
    for tile_size, count in sorted(
        tile_sizes.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {tile_size:20s}: {count:3d} kernels")

    # Find block size distribution
    block_sizes = {}
    for kernel in kernels:
        algo = kernel.get("algorithm", {})
        block_size = algo.get("block_size", 0)
        block_sizes[block_size] = block_sizes.get(block_size, 0) + 1

    print("\nBlock size distribution:")
    for block_size, count in sorted(block_sizes.items()):
        print(f"  {block_size:4d}: {count:3d} kernels")


def main():
    parser = argparse.ArgumentParser(
        description="Export dispatcher registry to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o", help="Output JSON filename")
    parser.add_argument("--demo-all", action="store_true", help="Run all demos")

    args = parser.parse_args()

    print("=" * 60)
    print("Dispatcher Registry JSON Export Example")
    print("=" * 60)

    # Try to load library
    try:
        lib = load_dispatcher_lib()
        lib.dispatcher_init()
        num_kernels = lib.dispatcher_get_kernel_count()
        print("\n✓ Loaded dispatcher library")
        print(f"  Registered kernels: {num_kernels}")
    except Exception as e:
        print(f"\n⚠ Could not load dispatcher library: {e}")
        print("  Running with mock data for demonstration")
        lib = None
        num_kernels = 0

    if num_kernels == 0 and lib is not None:
        print("\n[INFO] No kernels registered yet.")
        print("\nTo register kernels:")
        print("  1. Generate kernels:")
        print("     cd codegen && python3 unified_gemm_codegen.py")
        print("  2. Build and link kernels")
        print("  3. Run this example again")

    # Run demos
    if args.demo_all or not args.output:
        demo_export_to_string(lib)
        demo_print_summary(lib)
        demo_list_identifiers(lib)
        demo_analyze_json(lib)

    # Export to file if requested
    if args.output:
        demo_export_to_file(lib, args.output)
    else:
        print("\n" + "=" * 60)
        print("[TIP] Use --output to save JSON to file:")
        print(f"  python3 {sys.argv[0]} --output kernels.json")
        print("=" * 60)

    print("\n✓ Example complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
