#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
JSON Export Utilities for Dispatcher Registry

Provides high-level Python functions to export kernel registry metadata to JSON,
similar to the tile engine benchmarking JSON export functionality.

Example:
    >>> from ck_tile.dispatcher import Registry
    >>> from ck_tile.dispatcher.json_export import export_registry_json
    >>> 
    >>> registry = Registry.instance()
    >>> export_registry_json(registry, "kernels.json")
    >>> # Creates kernels.json with all registered kernel metadata
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

try:
    from _dispatcher_native import Registry
except ImportError:
    Registry = None


def export_registry_json(
    registry: Optional["Registry"] = None,
    filename: Optional[Union[str, Path]] = None,
    include_statistics: bool = True,
    pretty_print: bool = True
) -> Optional[str]:
    """
    Export dispatcher registry kernels to JSON.
    
    This provides functionality similar to the tile engine benchmarking JSON export,
    allowing you to inspect all registered kernels with their full metadata.
    
    Args:
        registry: Registry instance to export. If None, uses global Registry.instance()
        filename: Output filename. If None, returns JSON string instead of writing file
        include_statistics: Whether to include kernel statistics breakdown
        pretty_print: Whether to format JSON with indentation (Python-side only)
    
    Returns:
        JSON string if filename is None, otherwise None
        
    Example:
        >>> # Export to file
        >>> export_registry_json(filename="my_kernels.json")
        
        >>> # Get JSON string
        >>> json_str = export_registry_json()
        >>> print(json_str)
        
        >>> # Parse and analyze
        >>> import json
        >>> data = json.loads(export_registry_json())
        >>> print(f"Total kernels: {data['metadata']['total_kernels']}")
        >>> print(f"By pipeline: {data['statistics']['by_pipeline']}")
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    # Get registry instance
    if registry is None:
        registry = Registry.instance()
    
    # If filename provided, use C++ direct file export (more efficient)
    if filename is not None:
        filename_str = str(filename)
        success = registry.export_json_to_file(filename_str, include_statistics)
        if not success:
            raise IOError(f"Failed to write JSON to {filename_str}")
        print(f"✓ Exported {registry.size()} kernels to {filename_str}")
        return None
    
    # Otherwise, get JSON string from C++
    json_str = registry.export_json(include_statistics)
    
    # Optionally re-parse and pretty-print using Python
    if pretty_print:
        try:
            data = json.loads(json_str)
            json_str = json.dumps(data, indent=2)
        except json.JSONDecodeError:
            pass  # Keep original if parsing fails
    
    return json_str


def print_registry_summary(registry: Optional["Registry"] = None) -> None:
    """
    Print a human-readable summary of the registry.
    
    Args:
        registry: Registry instance. If None, uses global Registry.instance()
        
    Example:
        >>> from ck_tile.dispatcher.json_export import print_registry_summary
        >>> print_registry_summary()
        ========================================
        Dispatcher Registry Summary
        ========================================
        Total Kernels: 6
        
        By Data Type:
          fp16_fp16_fp16: 6
        
        By Pipeline:
          mem: 2
          compv3: 2
          compv4: 2
        ...
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    # Get registry instance
    if registry is None:
        registry = Registry.instance()
    
    # Get JSON data
    json_str = registry.export_json(include_statistics=True)
    data = json.loads(json_str)
    
    print("=" * 60)
    print("Dispatcher Registry Summary")
    print("=" * 60)
    print(f"Timestamp: {data['metadata']['timestamp']}")
    print(f"Total Kernels: {data['metadata']['total_kernels']}")
    
    if 'statistics' in data:
        stats = data['statistics']
        
        print("\nBy Data Type:")
        for dtype, count in sorted(stats['by_datatype'].items()):
            print(f"  {dtype}: {count}")
        
        print("\nBy Pipeline:")
        for pipeline, count in sorted(stats['by_pipeline'].items()):
            print(f"  {pipeline}: {count}")
        
        print("\nBy Scheduler:")
        for scheduler, count in sorted(stats['by_scheduler'].items()):
            print(f"  {scheduler}: {count}")
        
        print("\nBy Layout:")
        for layout, count in sorted(stats['by_layout'].items()):
            print(f"  {layout}: {count}")
        
        print("\nBy GFX Architecture:")
        for arch, count in sorted(stats['by_gfx_arch'].items()):
            print(f"  {arch}: {count}")
    
    print("=" * 60)


def get_registry_statistics(registry: Optional["Registry"] = None) -> Dict:
    """
    Get registry statistics as a Python dictionary.
    
    Args:
        registry: Registry instance. If None, uses global Registry.instance()
    
    Returns:
        Dictionary with metadata and statistics
        
    Example:
        >>> stats = get_registry_statistics()
        >>> print(f"Total: {stats['metadata']['total_kernels']}")
        >>> print(f"FP16 kernels: {stats['statistics']['by_datatype']['fp16_fp16_fp16']}")
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    # Get registry instance
    if registry is None:
        registry = Registry.instance()
    
    # Get and parse JSON
    json_str = registry.export_json(include_statistics=True)
    return json.loads(json_str)


def list_kernel_identifiers(registry: Optional["Registry"] = None) -> List[str]:
    """
    Get list of all kernel identifiers in the registry.
    
    Args:
        registry: Registry instance. If None, uses global Registry.instance()
    
    Returns:
        List of kernel identifier strings
        
    Example:
        >>> identifiers = list_kernel_identifiers()
        >>> for id in identifiers:
        ...     print(id)
        256x256x32_4x4x1_32x32x16_nopers
        128x128x32_2x2x1_32x32x16_nopers
        ...
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    # Get registry instance
    if registry is None:
        registry = Registry.instance()
    
    # Get JSON and extract identifiers
    json_str = registry.export_json(include_statistics=False)
    data = json.loads(json_str)
    
    return [kernel['identifier'] for kernel in data['kernels']]


def filter_kernels_by_property(
    registry: Optional["Registry"] = None,
    **filters
) -> List[Dict]:
    """
    Filter kernels by property values.
    
    Args:
        registry: Registry instance. If None, uses global Registry.instance()
        **filters: Property filters, e.g., pipeline="mem", persistent=True
    
    Returns:
        List of kernel dictionaries matching the filters
        
    Example:
        >>> # Find all persistent kernels
        >>> kernels = filter_kernels_by_property(persistent=True)
        >>> 
        >>> # Find all mem pipeline kernels
        >>> kernels = filter_kernels_by_property(pipeline="mem")
        >>> 
        >>> # Multiple filters
        >>> kernels = filter_kernels_by_property(pipeline="compv4", scheduler="intrawave")
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    # Get registry instance
    if registry is None:
        registry = Registry.instance()
    
    # Get all kernels
    json_str = registry.export_json(include_statistics=False)
    data = json.loads(json_str)
    
    # Filter kernels
    result = []
    for kernel in data['kernels']:
        match = True
        for key, value in filters.items():
            # Check in algorithm section
            if key in kernel.get('algorithm', {}):
                if kernel['algorithm'][key] != value:
                    match = False
                    break
            # Check in signature section
            elif key in kernel.get('signature', {}):
                if kernel['signature'][key] != value:
                    match = False
                    break
            # Check top-level
            elif key in kernel:
                if kernel[key] != value:
                    match = False
                    break
            else:
                match = False
                break
        
        if match:
            result.append(kernel)
    
    return result


def enable_auto_export(
    filename: str,
    include_statistics: bool = True,
    export_on_every_registration: bool = True,
    registry: Optional["Registry"] = None
) -> None:
    """
    Enable automatic JSON export on kernel registration.
    
    When enabled, the registry will automatically export to JSON either:
    - After every kernel registration (if export_on_every_registration=True, default)
    - On program exit / registry destruction (if export_on_every_registration=False)
    
    Args:
        filename: Output filename for auto-export
        include_statistics: Whether to include statistics in auto-export
        export_on_every_registration: If True, exports after every registration (default).
                                      If False, only exports on destruction.
        registry: Registry instance. If None, uses global Registry.instance()
        
    Example:
        >>> from ck_tile.dispatcher import Registry
        >>> from ck_tile.dispatcher.json_export import enable_auto_export
        >>> 
        >>> # Enable auto-export after every registration (default)
        >>> enable_auto_export("auto_kernels.json")
        >>> 
        >>> # Enable auto-export only on program exit (more efficient)
        >>> enable_auto_export("kernels.json", export_on_every_registration=False)
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    if registry is None:
        registry = Registry.instance()
    
    registry.enable_auto_export(filename, include_statistics, export_on_every_registration)
    
    mode = "every registration" if export_on_every_registration else "program exit"
    print(f"✓ Auto-export enabled: {filename} (triggers on {mode})")


def disable_auto_export(registry: Optional["Registry"] = None) -> None:
    """
    Disable automatic JSON export.
    
    Args:
        registry: Registry instance. If None, uses global Registry.instance()
        
    Example:
        >>> from ck_tile.dispatcher.json_export import disable_auto_export
        >>> disable_auto_export()
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    if registry is None:
        registry = Registry.instance()
    
    registry.disable_auto_export()
    print("✓ Auto-export disabled")


def is_auto_export_enabled(registry: Optional["Registry"] = None) -> bool:
    """
    Check if auto-export is enabled.
    
    Args:
        registry: Registry instance. If None, uses global Registry.instance()
    
    Returns:
        True if auto-export is enabled, False otherwise
        
    Example:
        >>> from ck_tile.dispatcher.json_export import is_auto_export_enabled
        >>> if is_auto_export_enabled():
        ...     print("Auto-export is active")
    """
    if Registry is None:
        raise ImportError(
            "Dispatcher native module not available. "
            "Build with: cmake -DBUILD_DISPATCHER_PYTHON=ON"
        )
    
    if registry is None:
        registry = Registry.instance()
    
    return registry.is_auto_export_enabled()


if __name__ == "__main__":
    # Example usage when run as a script
    print("Dispatcher Registry JSON Export")
    print("=" * 60)
    
    try:
        # Print summary
        print_registry_summary()
        
        # Export to file
        output_file = "dispatcher_kernels.json"
        export_registry_json(filename=output_file)
        print(f"\n✓ Full export saved to {output_file}")
        
        # Show auto-export status
        if is_auto_export_enabled():
            print("\n✓ Auto-export is enabled")
        else:
            print("\n✓ Auto-export is disabled")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo use this module, build the dispatcher with Python support:")
        print("  cmake -DBUILD_DISPATCHER_PYTHON=ON")

