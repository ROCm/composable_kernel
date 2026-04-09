#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Generate one .cpp wrapper file per kernel header for maximum parallel compilation.

Each kernel becomes its own translation unit, enabling:
  - Maximum parallelism with make -j$(nproc)
  - Per-kernel build progress (e.g., [5/128] Building kernel: gemm_fp16_128x128)
  - Incremental rebuilds (only changed kernels recompile)
  - Fine-grained build time analysis

Usage:
    python3 generate_kernel_wrappers.py --kernel-dir build/generated_kernels --output-dir build/kernel_wrappers

Output structure:
    build/kernel_wrappers/
    |---- gemm_fp16_rcr_128x128x32.cpp
    |---- gemm_fp16_rcr_256x256x64.cpp
    |---- conv_fwd_fp16_2d_128x128.cpp
    +---- ...

Each .cpp simply includes its corresponding .hpp and forces symbol emission.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import concurrent.futures


WRAPPER_TEMPLATE_GEMM = """// SPDX-License-Identifier: MIT
// Auto-generated wrapper for: {kernel_name}
// This file enables per-kernel parallel compilation

#include "{kernel_hpp}"

// Force symbol emission for kernel registration
namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

// Marker to prevent dead code elimination
volatile bool _{kernel_id}_registered = true;

}}  // namespace generated
}}  // namespace dispatcher
}}  // namespace ck_tile
"""

WRAPPER_TEMPLATE_CONV = """// SPDX-License-Identifier: MIT
// Auto-generated wrapper for: {kernel_name}
// This file enables per-kernel parallel compilation

#include "{kernel_hpp}"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

volatile bool _{kernel_id}_registered = true;

}}  // namespace generated
}}  // namespace dispatcher
}}  // namespace ck_tile
"""


def generate_wrapper(
    kernel_hpp: Path, output_dir: Path, index: int, total: int
) -> Tuple[Path, bool]:
    """Generate a .cpp wrapper for a single kernel header."""
    kernel_name = kernel_hpp.stem
    kernel_id = kernel_name.replace("-", "_").replace(".", "_")

    # Select template based on kernel type
    if kernel_name.startswith("gemm"):
        template = WRAPPER_TEMPLATE_GEMM
    else:
        template = WRAPPER_TEMPLATE_CONV

    content = template.format(
        kernel_name=kernel_name,
        kernel_hpp=kernel_hpp.name,
        kernel_id=kernel_id,
    )

    output_cpp = output_dir / f"{kernel_name}.cpp"

    # Only write if content changed (for incremental builds)
    if output_cpp.exists():
        existing = output_cpp.read_text()
        if existing == content:
            return output_cpp, False  # No change

    output_cpp.write_text(content)
    return output_cpp, True  # Written


def generate_cmake_list(
    wrappers: List[Path], output_dir: Path, kernel_dir: Path
) -> Path:
    """Generate CMakeLists.txt that compiles each wrapper as a separate object."""

    num_kernels = len(wrappers)

    cmake_content = f'''# SPDX-License-Identifier: MIT
# Auto-generated CMakeLists.txt for per-kernel parallel compilation
# Generated {num_kernels} kernel translation units

cmake_minimum_required(VERSION 3.16)

# =============================================================================
# Per-Kernel Object Targets ({num_kernels} kernels)
# =============================================================================
# Each kernel is compiled as a separate OBJECT library for maximum parallelism.
# Build with: make -j$(nproc) all_kernels
#
# Progress output:
#   [  1/{num_kernels}] Building kernel: gemm_fp16_rcr_128x128x32
#   [  2/{num_kernels}] Building kernel: gemm_fp16_rcr_256x256x64
#   ...

set(KERNEL_INCLUDE_DIR "{kernel_dir}")
set(ALL_KERNEL_OBJECTS "")

'''

    for idx, wrapper in enumerate(wrappers, 1):
        kernel_name = wrapper.stem
        obj_target = f"kobj_{kernel_name}"

        cmake_content += f"""
# [{idx}/{num_kernels}] {kernel_name}
add_library({obj_target} OBJECT {wrapper.name})
target_include_directories({obj_target} PRIVATE ${{KERNEL_INCLUDE_DIR}} ${{CK_INCLUDE_DIR}})
target_compile_options({obj_target} PRIVATE
    -mllvm -enable-noalias-to-md-conversion=0
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
)
set_target_properties({obj_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(hip_FOUND)
    target_link_libraries({obj_target} PRIVATE hip::device hip::host)
endif()
list(APPEND ALL_KERNEL_OBJECTS $<TARGET_OBJECTS:{obj_target}>)
"""

    cmake_content += f"""

# =============================================================================
# Combined Kernel Library
# =============================================================================
# Links all {num_kernels} kernel objects into a single shared library

add_library(all_kernels SHARED ${{ALL_KERNEL_OBJECTS}})
if(hip_FOUND)
    target_link_libraries(all_kernels PRIVATE hip::device hip::host)
endif()
set_target_properties(all_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "dispatcher_kernels"
)

message(STATUS "Configured {num_kernels} kernel objects for parallel compilation")
message(STATUS "Build with: make -j$(nproc) all_kernels")
"""

    cmake_file = output_dir / "CMakeLists.txt"
    cmake_file.write_text(cmake_content)
    return cmake_file


def generate_ninja_build(
    wrappers: List[Path], output_dir: Path, kernel_dir: Path
) -> Path:
    """Generate build.ninja for even faster parallel compilation."""

    num_kernels = len(wrappers)

    ninja_content = f"""# SPDX-License-Identifier: MIT
# Auto-generated build.ninja for per-kernel parallel compilation
# {num_kernels} kernel translation units

# Variables
cxx = hipcc
cxxflags = -fPIC -std=c++17 -O3 -mllvm -enable-noalias-to-md-conversion=0 -Wno-undefined-func-template -Wno-float-equal --offload-compress
includes = -I{kernel_dir} -I/opt/rocm/include

# Rules
rule compile
  command = $cxx $cxxflags $includes -c $in -o $out
  description = [{num_kernels}] Building kernel: $kernel_name

rule link
  command = $cxx -shared $in -o $out -L/opt/rocm/lib -lamdhip64
  description = Linking: $out

# Kernel objects
"""

    obj_files = []
    for idx, wrapper in enumerate(wrappers, 1):
        kernel_name = wrapper.stem
        obj_file = f"{kernel_name}.o"
        obj_files.append(obj_file)

        ninja_content += f"""
build {obj_file}: compile {wrapper.name}
  kernel_name = {kernel_name}
"""

    ninja_content += f"""

# Shared library
build libdispatcher_kernels.so: link {" ".join(obj_files)}

# Default target
default libdispatcher_kernels.so
"""

    ninja_file = output_dir / "build.ninja"
    ninja_file.write_text(ninja_content)
    return ninja_file


def generate_makefile(wrappers: List[Path], output_dir: Path, kernel_dir: Path) -> Path:
    """Generate Makefile for per-kernel parallel compilation."""

    num_kernels = len(wrappers)
    kernel_names = [w.stem for w in wrappers]
    obj_files = [f"{name}.o" for name in kernel_names]

    makefile_content = f"""# SPDX-License-Identifier: MIT
# Auto-generated Makefile for per-kernel parallel compilation
# {num_kernels} kernel translation units
#
# Usage:
#   make -j$(nproc)          # Build all kernels in parallel
#   make -j$(nproc) VERBOSE=1  # With per-kernel progress
#   make clean               # Remove all objects

CXX = hipcc
CXXFLAGS = -fPIC -std=c++17 -O3 -mllvm -enable-noalias-to-md-conversion=0 \\
           -Wno-undefined-func-template -Wno-float-equal --offload-compress
INCLUDES = -I{kernel_dir} -I/opt/rocm/include
LDFLAGS = -shared -L/opt/rocm/lib -lamdhip64

TARGET = libdispatcher_kernels.so
OBJECTS = {" ".join(obj_files)}

# Progress counter (only works with make -j1, use ninja for parallel progress)
TOTAL_KERNELS = {num_kernels}
CURRENT = 0

.PHONY: all clean

all: $(TARGET)
\t@echo "Built $(TARGET) with {num_kernels} kernels"

$(TARGET): $(OBJECTS)
\t@echo "[LINK] Linking {num_kernels} kernel objects -> $@"
\t$(CXX) $(LDFLAGS) $^ -o $@

"""

    for idx, (wrapper, obj) in enumerate(zip(wrappers, obj_files), 1):
        kernel_name = wrapper.stem
        makefile_content += f"""
{obj}: {wrapper.name}
\t@echo "[{idx}/{num_kernels}] Building kernel: {kernel_name}"
\t$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
"""

    makefile_content += f"""

clean:
\trm -f $(OBJECTS) $(TARGET)
\t@echo "Cleaned {num_kernels} kernel objects"
"""

    makefile = output_dir / "Makefile"
    makefile.write_text(makefile_content)
    return makefile


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-kernel wrapper .cpp files for parallel compilation"
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Directory containing generated kernel .hpp files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for wrapper .cpp files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hpp",
        help="Glob pattern for kernel headers (default: *.hpp)",
    )
    parser.add_argument(
        "--generate-cmake",
        action="store_true",
        help="Generate CMakeLists.txt for the wrappers",
    )
    parser.add_argument(
        "--generate-ninja",
        action="store_true",
        help="Generate build.ninja for ninja builds",
    )
    parser.add_argument(
        "--generate-makefile",
        action="store_true",
        help="Generate Makefile for make builds",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Generate wrappers in parallel (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Find kernel headers
    kernel_dir = args.kernel_dir.resolve()
    if not kernel_dir.exists():
        print(f"Error: Kernel directory not found: {kernel_dir}", file=sys.stderr)
        return 1

    kernel_headers = sorted(kernel_dir.glob(args.pattern))
    if not kernel_headers:
        print(
            f"Error: No kernel headers found matching {args.pattern} in {kernel_dir}",
            file=sys.stderr,
        )
        return 1

    num_kernels = len(kernel_headers)
    print(f"Found {num_kernels} kernel headers in {kernel_dir}")

    # Create output directory
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate wrappers
    print(f"Generating {num_kernels} wrapper .cpp files...")

    wrappers = []
    written = 0

    if args.parallel and num_kernels > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    generate_wrapper, hpp, output_dir, idx, num_kernels
                ): hpp
                for idx, hpp in enumerate(kernel_headers, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                wrapper_path, was_written = future.result()
                wrappers.append(wrapper_path)
                if was_written:
                    written += 1
                    if args.verbose:
                        print(f"  Generated: {wrapper_path.name}")
    else:
        for idx, hpp in enumerate(kernel_headers, 1):
            wrapper_path, was_written = generate_wrapper(
                hpp, output_dir, idx, num_kernels
            )
            wrappers.append(wrapper_path)
            if was_written:
                written += 1
                if args.verbose:
                    print(f"  [{idx}/{num_kernels}] Generated: {wrapper_path.name}")

    wrappers.sort(key=lambda p: p.name)

    print(
        f"  Total: {num_kernels} wrappers ({written} written, {num_kernels - written} unchanged)"
    )

    # Generate build files
    if args.generate_cmake:
        cmake_file = generate_cmake_list(wrappers, output_dir, kernel_dir)
        print(f"  Generated: {cmake_file}")

    if args.generate_ninja:
        ninja_file = generate_ninja_build(wrappers, output_dir, kernel_dir)
        print(f"  Generated: {ninja_file}")

    if args.generate_makefile:
        makefile = generate_makefile(wrappers, output_dir, kernel_dir)
        print(f"  Generated: {makefile}")

    print(f"\nOutput directory: {output_dir}")
    print(f"Kernels ready for parallel compilation: {num_kernels}")
    print("\nTo build:")
    print(f"  cd {output_dir}")
    if args.generate_makefile:
        print("  make -j$(nproc)  # Parallel build with progress")
    if args.generate_ninja:
        print("  ninja            # Fast parallel build")
    if args.generate_cmake:
        print("  cmake -B build && cmake --build build -j$(nproc)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
