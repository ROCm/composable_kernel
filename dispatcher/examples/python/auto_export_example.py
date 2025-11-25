#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Example: Automatic JSON Export on Registration

Demonstrates how to enable automatic JSON export so the registry
automatically exports kernel metadata whenever kernels are registered.

Two modes:
1. Export on program exit (default) - Exports once when program ends
2. Export on every registration - Exports after each kernel registration

Usage:
    python3 auto_export_example.py [mode]
    
    mode: "exit" (default) or "every"
"""

import sys
import argparse
from pathlib import Path

# Add dispatcher Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from _dispatcher_native import Registry
    from json_export import (
        enable_auto_export,
        disable_auto_export,
        is_auto_export_enabled
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nTo run this example:")
    print("  1. Build dispatcher with Python support:")
    print("     cmake -DBUILD_DISPATCHER_PYTHON=ON")
    print("  2. Ensure PYTHONPATH includes dispatcher/python")
    sys.exit(1)


def demo_exit_mode():
    """Demo: Auto-export on program exit"""
    print("\n" + "="*60)
    print("Demo: Auto-Export on Program Exit")
    print("="*60)
    
    output_file = "auto_exit_kernels.json"
    
    print(f"\nEnabling auto-export to: {output_file}")
    print("Mode: Export on program exit")
    
    # Enable auto-export (default mode: export on exit)
    enable_auto_export(output_file, include_statistics=True)
    
    # Check status
    if is_auto_export_enabled():
        print("✓ Auto-export is enabled")
    
    # Get registry info
    registry = Registry.instance()
    print(f"\nCurrent kernel count: {registry.size()}")
    
    if registry.size() == 0:
        print("\n[INFO] No kernels registered in this example.")
        print("In a real application, kernels would be registered via:")
        print("  registry.register_kernel(kernel_instance, Priority.Normal)")
        print("\nWhen program exits:")
        print(f"  - {output_file} will be created automatically")
        print("  - Contains all registered kernels at exit time")
        print("  - Efficient for production use")
    else:
        print(f"\n✓ Registry has {registry.size()} kernels")
        print(f"\nWhen program exits:")
        print(f"  - {output_file} will be created with all kernels")
    
    print("\n✓ Demo complete - watch for file on exit")


def demo_every_mode():
    """Demo: Auto-export after every registration"""
    print("\n" + "="*60)
    print("Demo: Auto-Export on Every Registration")
    print("="*60)
    
    output_file = "auto_every_kernels.json"
    
    print(f"\nEnabling auto-export to: {output_file}")
    print("Mode: Export after every registration")
    
    # Enable auto-export with export_on_every_registration=True
    enable_auto_export(
        output_file,
        include_statistics=True,
        export_on_every_registration=True
    )
    
    # Check status
    if is_auto_export_enabled():
        print("✓ Auto-export is enabled (every mode)")
    
    # Get registry info
    registry = Registry.instance()
    print(f"\nCurrent kernel count: {registry.size()}")
    
    if registry.size() == 0:
        print("\n[INFO] No kernels registered in this example.")
        print("In a real application, with 'every' mode:")
        print("  - File is updated after EACH kernel registration")
        print("  - Useful for debugging and development")
        print("  - Can see kernels as they are registered")
        print("  - Higher I/O overhead")
    else:
        print(f"\n✓ Registry has {registry.size()} kernels")
        print(f"\nWith 'every' mode:")
        print(f"  - {output_file} was updated after each registration")
        print(f"  - File should exist with latest state")
    
    print("\n✓ Demo complete")


def demo_disable():
    """Demo: Disable auto-export"""
    print("\n" + "="*60)
    print("Demo: Disable Auto-Export")
    print("="*60)
    
    # Check initial state
    if is_auto_export_enabled():
        print("\nAuto-export is currently enabled")
    else:
        print("\nAuto-export is currently disabled")
    
    # Disable
    print("\nDisabling auto-export...")
    disable_auto_export()
    
    # Verify
    if not is_auto_export_enabled():
        print("✓ Auto-export is now disabled")
    
    print("\n✓ Demo complete")


def demo_toggle():
    """Demo: Toggle auto-export on/off"""
    print("\n" + "="*60)
    print("Demo: Toggle Auto-Export")
    print("="*60)
    
    output_file = "auto_toggle_kernels.json"
    
    print("\n1. Initial state")
    print(f"   Auto-export enabled: {is_auto_export_enabled()}")
    
    print("\n2. Enable auto-export")
    enable_auto_export(output_file)
    print(f"   Auto-export enabled: {is_auto_export_enabled()}")
    
    print("\n3. Disable auto-export")
    disable_auto_export()
    print(f"   Auto-export enabled: {is_auto_export_enabled()}")
    
    print("\n4. Enable again (with 'every' mode)")
    enable_auto_export(output_file, export_on_every_registration=True)
    print(f"   Auto-export enabled: {is_auto_export_enabled()}")
    
    print("\n✓ Demo complete")


def demo_use_cases():
    """Show common use cases"""
    print("\n" + "="*60)
    print("Common Use Cases")
    print("="*60)
    
    print("\nUse Case 1: Production Application")
    print("-" * 40)
    print("Enable auto-export on program exit to capture final kernel state:")
    print()
    print("    from ck_tile.dispatcher.json_export import enable_auto_export")
    print("    enable_auto_export('production_kernels.json')")
    print()
    print("Benefits:")
    print("  ✓ Low overhead - exports once on exit")
    print("  ✓ Captures complete final state")
    print("  ✓ Good for documentation and auditing")
    
    print("\nUse Case 2: Development and Debugging")
    print("-" * 40)
    print("Enable auto-export on every registration to track kernel additions:")
    print()
    print("    enable_auto_export('debug_kernels.json',")
    print("                        export_on_every_registration=True)")
    print()
    print("Benefits:")
    print("  ✓ See kernels as they are registered")
    print("  ✓ Debug registration issues")
    print("  ✓ Track order of kernel additions")
    
    print("\nUse Case 3: Conditional Export")
    print("-" * 40)
    print("Enable auto-export only in certain conditions:")
    print()
    print("    import os")
    print("    if os.getenv('CK_AUTO_EXPORT'):")
    print("        enable_auto_export('kernels.json')")
    print()
    print("Benefits:")
    print("  ✓ Controlled via environment variable")
    print("  ✓ No code changes needed")
    print("  ✓ Easy to enable/disable")
    
    print("\nUse Case 4: Time-Stamped Exports")
    print("-" * 40)
    print("Export with timestamp in filename:")
    print()
    print("    from datetime import datetime")
    print("    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')")
    print("    enable_auto_export(f'kernels_{timestamp}.json')")
    print()
    print("Benefits:")
    print("  ✓ Track changes over time")
    print("  ✓ No file overwriting")
    print("  ✓ Historical record of kernel states")
    
    print("\n✓ Use cases demonstrated")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-export example for dispatcher registry",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["exit", "every", "disable", "toggle", "usecases", "all"],
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dispatcher Registry Auto-Export Example")
    print("="*60)
    
    if args.mode == "all":
        # Run all demos
        demo_exit_mode()
        demo_every_mode()
        demo_disable()
        demo_toggle()
        demo_use_cases()
    elif args.mode == "exit":
        demo_exit_mode()
    elif args.mode == "every":
        demo_every_mode()
    elif args.mode == "disable":
        demo_disable()
    elif args.mode == "toggle":
        demo_toggle()
    elif args.mode == "usecases":
        demo_use_cases()
    
    print("\n" + "="*60)
    print("✓ Example complete!")
    print("="*60)
    
    # Note: If auto-export is enabled, it will trigger when program exits
    return 0


if __name__ == "__main__":
    sys.exit(main())

