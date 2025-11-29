#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Self-contained build script for C++ convolution examples.

Parses DECL_CONV_KERNEL_SET declarations from source files,
generates the needed kernels, and compiles the example.

Usage:
    python3 compile_conv_examples.py examples/conv/cpp/02_conv_forward.cpp
    python3 compile_conv_examples.py examples/conv/cpp/03_conv_validation.cpp --no-compile
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import shutil

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CK_ROOT = DISPATCHER_DIR.parent

sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "examples" / "gemm" / "python"))


# Colors
class Colors:
    if sys.platform != "win32" and sys.stdout.isatty():
        GREEN = "\033[0;32m"
        YELLOW = "\033[1;33m"
        RED = "\033[0;31m"
        CYAN = "\033[0;36m"
        NC = "\033[0m"
    else:
        GREEN = YELLOW = RED = CYAN = NC = ""


def print_phase(msg: str):
    print(f"{Colors.YELLOW}{msg}{Colors.NC}")


def print_success(msg: str):
    print(f"{Colors.GREEN}{msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}{msg}{Colors.NC}", file=sys.stderr)


def print_info(msg: str):
    print(f"{Colors.CYAN}{msg}{Colors.NC}")


def find_hipcc() -> str:
    """Find hipcc compiler."""
    candidates = [
        os.environ.get("HIPCC"),
        "/opt/rocm/bin/hipcc",
        shutil.which("hipcc"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def extract_conv_declarations(source_file: Path) -> list:
    """Extract DECL_CONV_KERNEL_SET declarations from C++ source."""
    content = source_file.read_text()
    declarations = []

    # Pattern: DECL_CONV_KERNEL_SET(name, .add(...).add(...))
    set_pattern = r"DECL_CONV_KERNEL_SET\s*\(\s*(\w+)\s*,([^;]+)\)"

    for match in re.finditer(set_pattern, content, re.DOTALL):
        set_name = match.group(1)
        set_body = match.group(2)

        # Pattern 1: Simple add("dtype", "layout", "conv_type", tile_k, tile_c)
        simple_add = (
            r'\.add\s*\(\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*(\d+)\s*,\s*(\d+)'
        )
        for add_match in re.finditer(simple_add, set_body):
            declarations.append(
                {
                    "set": set_name,
                    "dtype": add_match.group(1),
                    "layout": add_match.group(2),
                    "conv_type": add_match.group(3),
                    "tile_k": int(add_match.group(4)),
                    "tile_c": int(add_match.group(5)),
                    "num_dims": 2,
                    "pipeline": "compv4",
                    "scheduler": "intrawave",
                    "wave_m": 2,
                    "wave_n": 2,
                    "wave_k": 1,
                    "warp_m": 32,
                    "warp_n": 32,
                    "warp_k": 16,
                    "arch": "gfx942",
                }
            )

        # Pattern 2: Full ConvSig()/ConvAlgo() specification
        full_add = (
            r'\.add\s*\(\s*ConvSig\(\)([^,]*),\s*ConvAlgo\(\)([^,]*),\s*"(\w+)"\s*\)'
        )
        for add_match in re.finditer(full_add, set_body, re.DOTALL):
            sig_str = add_match.group(1)
            algo_str = add_match.group(2)
            arch = add_match.group(3)

            # Parse signature
            dtype = "fp16"
            dtype_match = re.search(r'\.dtype\s*\(\s*"(\w+)"', sig_str)
            if dtype_match:
                dtype = dtype_match.group(1)

            layout = "nhwgc"
            layout_match = re.search(r'\.layout\s*\(\s*"(\w+)"', sig_str)
            if layout_match:
                layout = layout_match.group(1)

            conv_type = "forward"
            conv_type_match = re.search(r'\.conv_type\s*\(\s*"(\w+)"', sig_str)
            if conv_type_match:
                conv_type = conv_type_match.group(1)

            num_dims = 2
            dims_match = re.search(r"\.dims\s*\(\s*(\d+)", sig_str)
            if dims_match:
                num_dims = int(dims_match.group(1))

            # Parse algorithm
            tile_k, tile_c = 128, 128
            tile_match = re.search(
                r"\.tile\s*\(\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)", algo_str
            )
            if tile_match:
                tile_k = int(tile_match.group(1))
                tile_c = int(tile_match.group(2))

            wave_m, wave_n, wave_k = 2, 2, 1
            wave_match = re.search(
                r"\.wave\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?", algo_str
            )
            if wave_match:
                wave_m = int(wave_match.group(1))
                wave_n = int(wave_match.group(2))
                wave_k = int(wave_match.group(3) or 1)

            warp_m, warp_n, warp_k = 32, 32, 16
            warp_match = re.search(
                r"\.warp\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?", algo_str
            )
            if warp_match:
                warp_m = int(warp_match.group(1))
                warp_n = int(warp_match.group(2))
                warp_k = int(warp_match.group(3) or 16)

            pipeline = "compv4"
            pipeline_match = re.search(r'\.pipeline\s*\(\s*"(\w+)"', algo_str)
            if pipeline_match:
                pipeline = pipeline_match.group(1)

            scheduler = "intrawave"
            scheduler_match = re.search(r'\.scheduler\s*\(\s*"(\w+)"', algo_str)
            if scheduler_match:
                scheduler = scheduler_match.group(1)

            declarations.append(
                {
                    "set": set_name,
                    "dtype": dtype,
                    "layout": layout,
                    "conv_type": conv_type,
                    "tile_k": tile_k,
                    "tile_c": tile_c,
                    "num_dims": num_dims,
                    "pipeline": pipeline,
                    "scheduler": scheduler,
                    "wave_m": wave_m,
                    "wave_n": wave_n,
                    "wave_k": wave_k,
                    "warp_m": warp_m,
                    "warp_n": warp_n,
                    "warp_k": warp_k,
                    "arch": arch,
                }
            )

    return declarations


def generate_conv_kernels(declarations: list, output_dir: Path) -> list:
    """Generate convolution kernels using unified_conv_codegen."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from unified_conv_codegen import (
            UnifiedConvCodegen,
            ConvKernelConfig,
            ConvVariant,
        )
    except ImportError as e:
        print_error(f"Failed to import conv codegen: {e}")
        return []

    codegen = UnifiedConvCodegen(output_dir)
    generated = []

    for decl in declarations:
        # Map conv_type to variant
        variant = ConvVariant.FORWARD
        if decl["conv_type"] == "bwd_data":
            variant = ConvVariant.BWD_DATA
        elif decl["conv_type"] == "bwd_weight":
            variant = ConvVariant.BWD_WEIGHT

        config = ConvKernelConfig(
            variant=variant,
            pipeline=decl["pipeline"],
            scheduler=decl["scheduler"],
            tile_m=decl["tile_k"],
            tile_n=decl["tile_c"],
            tile_k=64,
            wave_m=decl["wave_m"],
            wave_n=decl["wave_n"],
            warp_m=decl["warp_m"],
            warp_n=decl["warp_n"],
            warp_k=decl["warp_k"],
            ndim=decl["num_dims"],
        )

        try:
            filepath = codegen.generate_kernel(config, decl["dtype"])
            generated.append(filepath)
            print_info(f"    Generated: {filepath.name}")
        except Exception as e:
            print_error(f"    Failed: {e}")

    return generated


def compile_example(
    source_file: Path,
    output_bin: Path,
    kernel_headers: list,
    hipcc: str,
    gpu_target: str,
) -> bool:
    """Compile the C++ example with generated kernels."""
    build_dir = DISPATCHER_DIR / "build"
    kernel_dir = build_dir / "generated_kernels"

    includes = [
        f"-I{CK_ROOT / 'include'}",
        f"-I{DISPATCHER_DIR / 'include'}",
        f"-I{kernel_dir}",
    ]

    # Build include flags for generated kernels
    kernel_includes = []
    for header in kernel_headers:
        kernel_includes.extend(["-include", str(header)])

    # Add define to indicate kernels are available
    defines = ["-DCONV_KERNEL_AVAILABLE=1"]

    cmd = [
        hipcc,
        "-std=c++20",
        "-O2",
        f"--offload-arch={gpu_target}",
        *includes,
        *defines,
        *kernel_includes,
        "-o",
        str(output_bin),
        str(source_file),
    ]

    print(f"  Compiling: {source_file.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if result.stderr:
            # Show first few error lines
            lines = result.stderr.split("\n")
            errors = [line for line in lines if "error:" in line.lower()][:5]
            for err_line in errors:
                print_error(f"    {err_line}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build C++ convolution example with self-contained kernel generation"
    )
    parser.add_argument("source", help="Source file (.cpp)")
    parser.add_argument("--output", "-o", help="Output binary name")
    parser.add_argument("--gpu-target", default="gfx942", help="GPU target")
    parser.add_argument(
        "--no-compile", action="store_true", help="Only generate kernels, don't compile"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve source file
    source_file = Path(args.source)
    if not source_file.is_absolute():
        candidates = [
            DISPATCHER_DIR / args.source,
            Path.cwd() / args.source,
        ]
        for c in candidates:
            if c.exists():
                source_file = c
                break

    if not source_file.exists():
        print_error(f"Source file not found: {source_file}")
        return 1

    build_dir = DISPATCHER_DIR / "build"
    kernel_dir = build_dir / "generated_kernels"
    output_name = args.output or source_file.stem
    output_bin = build_dir / output_name

    print_success("=== Conv Example Builder (Self-Contained) ===\n")

    # Phase 1: Extract declarations
    print_phase("Phase 1: Scanning for DECL_CONV_KERNEL_SET...")
    declarations = extract_conv_declarations(source_file)

    if not declarations:
        print_error("  No DECL_CONV_KERNEL_SET declarations found!")
        return 1

    print(f"  Found {len(declarations)} kernel declaration(s):")
    for decl in declarations:
        name = f"{decl['dtype']}_{decl['conv_type']}_{decl['num_dims']}d_{decl['tile_k']}x{decl['tile_c']}"
        print(f"    [{decl['set']}] {name}")
    print()

    # Phase 2: Generate kernels
    print_phase("Phase 2: Generating kernels...")
    generated = generate_conv_kernels(declarations, kernel_dir)

    if not generated:
        print_error("  No kernels generated!")
        return 1

    print(f"  Generated {len(generated)} kernel file(s)")
    print()

    # Phase 3: Compile (optional)
    if args.no_compile:
        print_info("Skipping compilation (--no-compile)")
        print()
        print_success("=== Kernel Generation Complete ===")
        print(f"Kernels in: {kernel_dir}")
        return 0

    print_phase("Phase 3: Compiling example...")
    hipcc = find_hipcc()

    if not hipcc:
        print_error("  hipcc not found. Install ROCm or set HIPCC env var.")
        print("  To compile manually:")
        print(
            f"    hipcc -std=c++20 -O2 -I{CK_ROOT / 'include'} -I{DISPATCHER_DIR / 'include'} \\"
        )
        print(f"          -I{kernel_dir} \\")
        for h in generated[:1]:  # Show first header as example
            print(f"          -include {h} \\")
        print("          -DCONV_KERNEL_AVAILABLE=1 \\")
        print(f"          --offload-arch={args.gpu_target} \\")
        print(f"          {source_file} -o {output_bin}")
        return 1

    build_dir.mkdir(parents=True, exist_ok=True)

    if not compile_example(source_file, output_bin, generated, hipcc, args.gpu_target):
        print_error("  Compilation failed!")
        return 1

    print_success(f"  Output: {output_bin}")
    print()

    print_success("=== Build Complete ===")
    print()
    print("Run with:")
    print(f"  {output_bin}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
