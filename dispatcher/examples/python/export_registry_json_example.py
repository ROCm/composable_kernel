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
    python3 export_registry_json_example.py [--output kernels.json] [--no-stats]
"""

import sys
import json
import argparse
from pathlib import Path

# Add dispatcher Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from _dispatcher_native import Registry
    from json_export import (
        export_registry_json,
        print_registry_summary,
        get_registry_statistics,
        list_kernel_identifiers,
        filter_kernels_by_property
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nTo run this example:")
    print("  1. Build dispatcher with Python support:")
    print("     cmake -DBUILD_DISPATCHER_PYTHON=ON")
    print("  2. Ensure PYTHONPATH includes dispatcher/python")
    print("  3. Generate and register some kernels first")
    sys.exit(1)


def demo_export_to_string():
    """Demo: Export to JSON string"""
    print("\n" + "="*60)
    print("Demo 1: Export to JSON String")
    print("="*60)
    
    registry = Registry.instance()
    
    # Get JSON string
    json_str = export_registry_json()
    
    print(f"✓ Generated JSON string ({len(json_str)} bytes)")
    
    # Parse and show preview
    data = json.loads(json_str)
    print(f"\nMetadata:")
    print(f"  Timestamp: {data['metadata']['timestamp']}")
    print(f"  Total Kernels: {data['metadata']['total_kernels']}")
    print(f"  Export Version: {data['metadata']['export_version']}")
    
    if 'statistics' in data:
        print(f"\nStatistics available:")
        print(f"  - By data type: {len(data['statistics']['by_datatype'])} types")
        print(f"  - By pipeline: {len(data['statistics']['by_pipeline'])} pipelines")
        print(f"  - By scheduler: {len(data['statistics']['by_scheduler'])} schedulers")


def demo_export_to_file(filename):
    """Demo: Export to JSON file"""
    print("\n" + "="*60)
    print("Demo 2: Export to JSON File")
    print("="*60)
    
    # Export with statistics
    export_registry_json(filename=filename, include_statistics=True)
    
    # Verify file was created
    file_path = Path(filename)
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"✓ File created: {filename} ({size_kb:.1f} KB)")
        
        # Read and show structure
        with open(filename) as f:
            data = json.load(f)
        
        print(f"\nFile structure:")
        print(f"  - metadata: {len(data['metadata'])} fields")
        if 'statistics' in data:
            print(f"  - statistics: {len(data['statistics'])} categories")
        print(f"  - kernels: {len(data['kernels'])} kernels")
    else:
        print(f"✗ Failed to create file: {filename}")


def demo_print_summary():
    """Demo: Print human-readable summary"""
    print("\n" + "="*60)
    print("Demo 3: Print Registry Summary")
    print("="*60)
    
    print_registry_summary()


def demo_get_statistics():
    """Demo: Get statistics as dictionary"""
    print("\n" + "="*60)
    print("Demo 4: Get Statistics Dictionary")
    print("="*60)
    
    stats = get_registry_statistics()
    
    print(f"\nTotal kernels: {stats['metadata']['total_kernels']}")
    
    if 'statistics' in stats:
        print("\nData type distribution:")
        for dtype, count in sorted(stats['statistics']['by_datatype'].items()):
            print(f"  {dtype:30s}: {count:3d} kernels")
        
        print("\nPipeline distribution:")
        for pipeline, count in sorted(stats['statistics']['by_pipeline'].items()):
            print(f"  {pipeline:30s}: {count:3d} kernels")


def demo_list_identifiers():
    """Demo: List all kernel identifiers"""
    print("\n" + "="*60)
    print("Demo 5: List Kernel Identifiers")
    print("="*60)
    
    identifiers = list_kernel_identifiers()
    
    print(f"\nFound {len(identifiers)} kernel identifiers:")
    
    # Show first 10
    for i, identifier in enumerate(identifiers[:10]):
        print(f"  {i+1:2d}. {identifier}")
    
    if len(identifiers) > 10:
        print(f"  ... and {len(identifiers) - 10} more")


def demo_filter_kernels():
    """Demo: Filter kernels by properties"""
    print("\n" + "="*60)
    print("Demo 6: Filter Kernels by Properties")
    print("="*60)
    
    # Get all kernels first to see what's available
    registry = Registry.instance()
    if registry.size() == 0:
        print("\nNo kernels registered - skipping filter demo")
        return
    
    # Filter by persistent
    persistent_kernels = filter_kernels_by_property(persistent=True)
    print(f"\nPersistent kernels: {len(persistent_kernels)}")
    for kernel in persistent_kernels[:3]:
        print(f"  - {kernel['identifier']}")
    
    # Filter by pipeline
    mem_kernels = filter_kernels_by_property(pipeline="mem")
    print(f"\nMem pipeline kernels: {len(mem_kernels)}")
    for kernel in mem_kernels[:3]:
        print(f"  - {kernel['identifier']}")
    
    # Multiple filters
    try:
        compv4_intra = filter_kernels_by_property(
            pipeline="compv4",
            scheduler="intrawave"
        )
        print(f"\nCompV4 + Intrawave kernels: {len(compv4_intra)}")
        for kernel in compv4_intra[:3]:
            print(f"  - {kernel['identifier']}")
    except:
        pass


def demo_analyze_json():
    """Demo: Analyze JSON data"""
    print("\n" + "="*60)
    print("Demo 7: Analyze JSON Data")
    print("="*60)
    
    # Get full data
    json_str = export_registry_json()
    data = json.loads(json_str)
    
    if len(data['kernels']) == 0:
        print("\nNo kernels to analyze")
        return
    
    print("\nAnalyzing kernel configurations...")
    
    # Find tile size distribution
    tile_sizes = {}
    for kernel in data['kernels']:
        tile = kernel['algorithm']['tile_shape']
        tile_str = f"{tile['m']}x{tile['n']}x{tile['k']}"
        tile_sizes[tile_str] = tile_sizes.get(tile_str, 0) + 1
    
    print("\nTile size distribution:")
    for tile_size, count in sorted(tile_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tile_size:20s}: {count:3d} kernels")
    
    # Find block size distribution
    block_sizes = {}
    for kernel in data['kernels']:
        block_size = kernel['algorithm']['block_size']
        block_sizes[block_size] = block_sizes.get(block_size, 0) + 1
    
    print("\nBlock size distribution:")
    for block_size, count in sorted(block_sizes.items()):
        print(f"  {block_size:4d}: {count:3d} kernels")
    
    # Find feature usage
    print("\nFeature usage:")
    features = {
        'persistent': 0,
        'double_buffer': 0,
        'preshuffle': 0,
        'transpose_c': 0,
    }
    
    for kernel in data['kernels']:
        algo = kernel['algorithm']
        for feature in features:
            if algo[feature]:
                features[feature] += 1
    
    total = len(data['kernels'])
    for feature, count in features.items():
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {feature:20s}: {count:3d} kernels ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Export dispatcher registry to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON filename"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Exclude statistics from export"
    )
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run all demos"
    )
    
    args = parser.parse_args()
    
    # Check if registry has kernels
    registry = Registry.instance()
    num_kernels = registry.size()
    
    print("="*60)
    print("Dispatcher Registry JSON Export Example")
    print("="*60)
    print(f"\nRegistered kernels: {num_kernels}")
    
    if num_kernels == 0:
        print("\n[INFO] No kernels registered yet.")
        print("\nTo register kernels:")
        print("  1. Generate kernels:")
        print("     cd codegen && python3 unified_gemm_codegen.py")
        print("  2. Build and link kernels")
        print("  3. Run this example again")
        print("\nShowing empty registry JSON structure:")
        
        # Show structure with empty registry
        json_str = export_registry_json()
        print(json.dumps(json.loads(json_str), indent=2))
        return 0
    
    # Run demos
    if args.demo_all or not args.output:
        demo_export_to_string()
        demo_print_summary()
        demo_get_statistics()
        demo_list_identifiers()
        demo_filter_kernels()
        demo_analyze_json()
    
    # Export to file if requested
    if args.output:
        demo_export_to_file(args.output)
    else:
        print("\n" + "="*60)
        print("[TIP] Use --output to save JSON to file:")
        print(f"  python3 {sys.argv[0]} --output kernels.json")
        print("="*60)
    
    print("\n✓ Example complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

