#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Generate per-example kernel registration headers from DECL_KERNEL_SET declarations.

This script:
1. Parses DECL_KERNEL_SET from a C++ source file
2. Generates ONLY the kernel code declared (not all kernels)
3. Creates a registration header specific to that example

Usage:
    python3 generate_example_kernels.py <source_file.cpp> --output-dir <dir>

Example:
    python3 generate_example_kernels.py examples/gemm/cpp/01_basic_gemm.cpp \
        --output-dir build/generated_kernels/generated

Benefits:
- Minimal builds: Only kernels declared in DECL_KERNEL_SET are generated
- Fast iteration: Changing one example doesn't rebuild all kernels
- Single source of truth: C++ code declares what it needs
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

# Import from compile_gemm_examples.py (after sys.path modification)
from compile_gemm_examples import (  # noqa: E402
    extract_kernel_declarations,
    extract_conv_kernel_declarations,
    find_kernel_header,
    find_conv_kernel_header,
    build_exact_kernel_filename,
)


def get_example_name(source_file: Path) -> str:
    """Extract example name from source file path."""
    # 01_basic_gemm.cpp -> 01_basic_gemm
    return source_file.stem


def generate_specific_gemm_kernel(
    decl: dict, kernel_dir: Path, gpu_target: str = "gfx942"
) -> Path:
    """Generate a specific GEMM kernel based on declaration.

    Returns the path to the generated kernel header, or None if generation failed.
    """
    dtype = decl.get("dtype_a", decl.get("dtype", "fp16"))
    layout = decl.get("layout", "rcr")
    pipeline = decl.get("pipeline", "compv3")
    scheduler = decl.get("scheduler", "intrawave")
    epilogue = decl.get("epilogue", "cshuffle")

    tile_m = decl.get("tile_m", 128)
    tile_n = decl.get("tile_n", 128)
    tile_k = decl.get("tile_k", 64)

    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    warp_m = decl.get("warp_m", 32)
    warp_n = decl.get("warp_n", 32)
    warp_k = decl.get("warp_k", 16)

    # Multi-D specific
    elementwise_op = decl.get("elementwise_op", "PassThrough")
    num_d_tensors = decl.get("num_d_tensors", 0)
    is_multi_d = elementwise_op != "PassThrough" and num_d_tensors > 0

    # Build exact filename to check if already exists
    expected_filename = build_exact_kernel_filename(decl)
    expected_path = kernel_dir / expected_filename

    if expected_path.exists():
        print(f"  Kernel exists: {expected_filename}")
        return expected_path

    print(f"  Generating: {expected_filename}")

    # Generate ONLY this specific kernel using --tile-config-json
    codegen_script = DISPATCHER_DIR / "codegen" / "unified_gemm_codegen.py"

    import json

    tile_config = {
        "tile_m": [tile_m],
        "tile_n": [tile_n],
        "tile_k": [tile_k],
        "warp_m": [wave_m],  # Note: config uses warp_m for wave distribution
        "warp_n": [wave_n],
        "warp_k": [wave_k],
        "warp_tile_m": [warp_m],
        "warp_tile_n": [warp_n],
        "warp_tile_k": [warp_k],
        "pipeline": [pipeline],
        "scheduler": [scheduler],
        "epilogue": [epilogue],
        "block_size": [256],
    }

    # Choose variant based on kernel type
    variant = "multi_d" if is_multi_d else "standard"

    # For multi_d, add elementwise config
    if is_multi_d:
        tile_config["elementwise_ops"] = [elementwise_op]
        tile_config["num_d_tensors"] = [num_d_tensors]

    cmd = [
        sys.executable,
        str(codegen_script),
        "--datatype",
        dtype,
        "--layout",
        layout + "r",  # Add 4th char for D layout
        "--variants",
        variant,
        "--output-dir",
        str(kernel_dir),
        "--gpu-target",
        gpu_target,
        "--tile-config-json",
        json.dumps(tile_config),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(DISPATCHER_DIR / "codegen")
    )

    if result.returncode != 0:
        print(f"  Warning: Codegen returned {result.returncode}")
        if result.stderr:
            for line in result.stderr.split("\n")[:3]:
                if line.strip():
                    print(f"    {line.strip()}")

    # Check if exact kernel was generated
    if expected_path.exists():
        return expected_path

    # Try to find closest matching kernel
    return None


def generate_registration_header(
    example_name: str,
    gemm_declarations: list,
    conv_declarations: list,
    output_dir: Path,
    kernel_dir: Path,
    gpu_target: str = "gfx942",
    generate_kernels: bool = True,
) -> Path:
    """Generate a registration header for the example.

    The generated header provides:
    - Includes for all matching dispatcher wrapper headers
    - A register_<example_name>_kernels(registry, arch) function

    If generate_kernels=True, will generate any missing kernels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / f"{example_name}_kernels.hpp"

    # Generate or find matching kernel headers for each declaration
    gemm_kernel_headers = []
    for decl in gemm_declarations:
        # First try to generate the specific kernel if needed
        if generate_kernels:
            header = generate_specific_gemm_kernel(decl, kernel_dir, gpu_target)
            if header:
                gemm_kernel_headers.append(header)
                continue

        # Fall back to finding existing kernel
        header = find_kernel_header(decl, gpu_target)
        if header:
            gemm_kernel_headers.append(header)

    conv_kernel_headers = []
    for decl in conv_declarations:
        # For conv, just try to find existing (TODO: add specific generation)
        header = find_conv_kernel_header(decl, gpu_target)
        if header:
            conv_kernel_headers.append(header)

    # Map kernel headers to their dispatcher wrapper headers
    wrapper_includes = []
    factory_calls = []

    for header in gemm_kernel_headers:
        # gemm_fp16_rcr_compv3_... -> dispatcher_wrapper_gemm_fp16_rcr_compv3_...
        wrapper_name = f"dispatcher_wrapper_{header.stem}.hpp"
        wrapper_path = kernel_dir / "dispatcher_wrappers" / wrapper_name

        if wrapper_path.exists():
            wrapper_includes.append(f'#include "dispatcher_wrappers/{wrapper_name}"')

            # Extract factory function name from wrapper header
            factory_name = extract_factory_name(wrapper_path)
            if factory_name:
                factory_calls.append(
                    f"    registry.register_kernel(generated::{factory_name}(gfx_arch), priority);"
                )

    for header in conv_kernel_headers:
        wrapper_name = f"dispatcher_wrapper_{header.stem}.hpp"
        wrapper_path = kernel_dir / "dispatcher_wrappers" / wrapper_name

        if wrapper_path.exists():
            wrapper_includes.append(f'#include "dispatcher_wrappers/{wrapper_name}"')

            factory_name = extract_factory_name(wrapper_path)
            if factory_name:
                factory_calls.append(
                    f"    registry.register_kernel(generated::{factory_name}(gfx_arch), priority);"
                )

    # Generate the header content
    content = f"""// SPDX-License-Identifier: MIT
// Auto-generated registration header for {example_name}
// Generated from DECL_KERNEL_SET declarations
#pragma once

#include "ck_tile/dispatcher.hpp"
{chr(10).join(wrapper_includes)}

namespace generated {{

using ::ck_tile::dispatcher::Registry;
using Priority = ::ck_tile::dispatcher::Registry::Priority;

/**
 * Register kernels declared in {example_name}.cpp
 * 
 * This function creates kernel instances matching the DECL_KERNEL_SET
 * declarations and registers them to the provided registry.
 */
inline void register_{example_name}_kernels(
    Registry& registry,
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
{chr(10).join(factory_calls) if factory_calls else "    // No matching kernels found"}
}}

}} // namespace generated
"""

    header_path.write_text(content)
    return header_path


def extract_factory_name(wrapper_path: Path) -> str:
    """Extract the factory function name from a dispatcher wrapper header."""
    content = wrapper_path.read_text()

    # Look for: inline KernelInstancePtr make_gemm_...(
    match = re.search(r"inline\s+KernelInstancePtr\s+(make_\w+)\s*\(", content)
    if match:
        return match.group(1)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-example kernel registration headers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script parses DECL_KERNEL_SET from C++ source and:
1. Generates ONLY the kernels declared (minimal build)
2. Creates a registration header for those kernels

Example:
    # Generate only kernels for 01_basic_gemm.cpp
    python3 generate_example_kernels.py examples/gemm/cpp/01_basic_gemm.cpp \\
        --output-dir build/generated_kernels/generated \\
        --generate-kernels
""",
    )
    parser.add_argument("source", help="Source file (.cpp) containing DECL_KERNEL_SET")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated registration header",
    )
    parser.add_argument(
        "--kernel-dir",
        help="Directory containing generated kernels (default: build/generated_kernels)",
    )
    parser.add_argument(
        "--gpu-target", default="gfx942", help="GPU target architecture"
    )
    parser.add_argument(
        "--generate-kernels",
        action="store_true",
        help="Generate missing kernels (minimal build - only declared kernels)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    source_file = Path(args.source)
    if not source_file.exists():
        print(f"Error: Source file not found: {source_file}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    kernel_dir = (
        Path(args.kernel_dir)
        if args.kernel_dir
        else DISPATCHER_DIR / "build" / "generated_kernels"
    )
    kernel_dir.mkdir(parents=True, exist_ok=True)

    example_name = get_example_name(source_file)

    print(f"Processing: {example_name}")
    if args.verbose:
        print(f"  Source: {source_file}")
        print(f"  Output: {output_dir / f'{example_name}_kernels.hpp'}")
        print(f"  Kernel dir: {kernel_dir}")
        print(f"  Generate kernels: {args.generate_kernels}")

    # Extract declarations
    gemm_declarations = extract_kernel_declarations(source_file)
    conv_declarations = extract_conv_kernel_declarations(source_file)

    if not gemm_declarations and not conv_declarations:
        print(f"  Warning: No DECL_KERNEL_SET found in {source_file}")
        # Still generate an empty header
        gemm_declarations = []
        conv_declarations = []
    else:
        print(
            f"  Found {len(gemm_declarations)} GEMM + {len(conv_declarations)} Conv declarations"
        )

    # Generate the registration header (and optionally generate kernels)
    header_path = generate_registration_header(
        example_name,
        gemm_declarations,
        conv_declarations,
        output_dir,
        kernel_dir,
        args.gpu_target,
        generate_kernels=args.generate_kernels,
    )

    print(f"Generated: {header_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
