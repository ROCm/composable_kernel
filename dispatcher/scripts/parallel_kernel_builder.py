#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Build kernels in parallel - one translation unit per kernel.

This script is called at make time (not cmake time) to avoid slow cmake configuration.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_hipcc():
    """Find hipcc compiler."""
    candidates = [
        os.environ.get("HIPCC"),
        "/opt/rocm/bin/hipcc",
        shutil.which("hipcc") if shutil else None,
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return "hipcc"  # Assume in PATH


def compile_kernel(args):
    """Compile a single kernel."""
    kernel_hpp, output_dir, include_dirs, hipcc = args
    kernel_name = kernel_hpp.stem

    # Create wrapper .cpp
    wrapper_cpp = output_dir / f"{kernel_name}.cpp"
    wrapper_cpp.write_text(f'''// Auto-generated wrapper
#include "{kernel_hpp.name}"
namespace {{ volatile bool _k = true; }}
''')

    # Compile to object
    obj_file = output_dir / f"{kernel_name}.o"

    cmd = [
        hipcc,
        "-c",
        "-fPIC",
        "-std=c++17",
        "-O3",
        "--offload-arch=gfx942",
        "-mllvm",
        "-enable-noalias-to-md-conversion=0",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "--offload-compress",
    ]

    for inc_dir in include_dirs:
        cmd.extend(["-I", str(inc_dir)])
    cmd.extend(["-I", str(kernel_hpp.parent)])

    cmd.extend(["-o", str(obj_file), str(wrapper_cpp)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return (kernel_name, False, result.stderr)
    return (kernel_name, True, str(obj_file))


def main():
    parser = argparse.ArgumentParser(description="Build kernels in parallel")
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-dirs", type=str, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Find kernel headers
    kernel_headers = list(args.kernel_dir.glob("gemm_*.hpp")) + list(
        args.kernel_dir.glob("conv_*.hpp")
    )

    if not kernel_headers:
        print("No kernels found to build")
        return 0

    print(f"Building {len(kernel_headers)} kernels with {args.jobs} parallel jobs...")

    include_dirs = [Path(p.strip()) for p in args.include_dirs.split(",")]
    hipcc = find_hipcc()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare work items
    work = [(h, args.output_dir, include_dirs, hipcc) for h in kernel_headers]

    # Compile in parallel
    obj_files = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(compile_kernel, w): w[0].name for w in work}

        for i, future in enumerate(as_completed(futures), 1):
            name, success, result = future.result()
            if success:
                obj_files.append(result)
                print(f"[{i}/{len(kernel_headers)}] Built: {name}")
            else:
                failed.append((name, result))
                print(f"[{i}/{len(kernel_headers)}] FAILED: {name}")

    if failed:
        print(f"\n{len(failed)} kernels failed to compile:")
        for name, err in failed[:5]:
            print(f"  {name}: {err[:100]}")
        return 1

    # Link into shared library
    print(f"\nLinking {len(obj_files)} objects into libdispatcher_kernels.so...")
    lib_path = args.output_dir / "libdispatcher_kernels.so"

    link_cmd = [hipcc, "-shared", "-fPIC", "-o", str(lib_path)] + obj_files
    result = subprocess.run(link_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Linking failed: {result.stderr}")
        return 1

    print(f"OK Built: {lib_path}")
    return 0


if __name__ == "__main__":
    import shutil

    sys.exit(main())
