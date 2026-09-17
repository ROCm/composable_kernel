#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Architecture Specs Generator

Generates both Python and C++ code from a single JSON source of truth.
This ensures consistency between Python codegen and C++ runtime filtering.

Usage:
    python generate_arch_specs.py [--json arch_specs.json] [--output-dir .]

    # Regenerate after editing arch_specs.json:
    python generate_arch_specs.py

Output:
    - arch_specs_generated.py  (Python module with arch data)
    - arch_specs_generated.hpp (C++ header with arch data)
"""

import json
import argparse
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

SCRIPT_DIR = Path(__file__).parent


def load_arch_specs(json_path: Path) -> Dict[str, Any]:
    """Load architecture specifications from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def resolve_pipeline_lds_budgets(specs: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Resolve pipeline_lds_budget against each architecture's lds_capacity_kb.

    Returns {arch: {pipeline: bytes}}, fully populated for every architecture
    and pipeline so that no consumer needs a fallback path.

    The budget is a property of the silicon as much as of the pipeline: a tile
    that does not fit in one architecture's LDS may fit comfortably in
    another's. Resolution happens here, once, so the emitted Python and C++
    receive literal byte counts and cannot drift in how they read the schema.
    """
    budget_spec = specs["pipeline_lds_budget"]
    architectures = specs["architectures"]

    resolved: Dict[str, Dict[str, int]] = {}

    for arch_name, arch_spec in architectures.items():
        if "lds_capacity_kb" not in arch_spec:
            raise ValueError(
                f"Architecture '{arch_name}' is missing required field "
                f"'lds_capacity_kb'. It is mandatory per ADDING_NEW_GPU.md and "
                f"is now consumed by codegen. Source the value from "
                f"get_lds_size() in include/ck_tile/core/arch/arch.hpp."
            )

        capacity_bytes = int(arch_spec["lds_capacity_kb"]) * 1024
        per_pipeline: Dict[str, int] = {}

        for pipeline, entry in budget_spec.items():
            if pipeline.startswith("_"):
                continue

            basis = entry["basis"]
            value = entry["value"]

            if basis == "fraction":
                budget = int(capacity_bytes * value)
            elif basis == "absolute_kb":
                budget = int(value) * 1024
            else:
                raise ValueError(
                    f"Unknown basis '{basis}' for pipeline '{pipeline}'. "
                    f"Expected 'fraction' or 'absolute_kb'."
                )

            # A budget may never promise more LDS than the silicon has.
            per_pipeline[pipeline] = min(budget, capacity_bytes)

        if "default" not in per_pipeline:
            raise ValueError("pipeline_lds_budget must define a 'default' entry.")

        resolved[arch_name] = per_pipeline

    return resolved


def generate_python_module(specs: Dict[str, Any], output_path: Path):
    """Generate Python module from arch specs."""

    timestamp = datetime.now().isoformat()

    # Extract data
    archs = specs["architectures"]
    element_sizes = specs["element_sizes"]
    lds_budgets = resolve_pipeline_lds_budgets(specs)
    unsupported = specs["unsupported_trait_combos"]["combinations"]

    # Build warp configs dict
    warp_configs_str = "{\n"
    for arch, data in archs.items():
        warp_configs_str += f'    "{arch}": {data["warp_configs"]},\n'
    warp_configs_str += "}"

    # Build warp tile combos dict
    warp_tile_str = "{\n"
    for arch, data in archs.items():
        warp_tile_str += f'    "{arch}": {{\n'
        for dtype, combos in data["warp_tile_combos"].items():
            warp_tile_str += f'        "{dtype}": {combos},\n'
        warp_tile_str += "    },\n"
    warp_tile_str += "}"

    # Build arch family map
    arch_family_str = "{\n"
    for arch, data in archs.items():
        arch_family_str += f'    "{arch}": "{data["family"]}",\n'
    arch_family_str += "}"

    # Build unsupported combos set
    unsupported_str = "{\n"
    for combo in unsupported:
        unsupported_str += f'    ("{combo[0]}", "{combo[1]}", "{combo[2]}"),\n'
    unsupported_str += "}"

    # Pipeline LDS budgets, arch -> pipeline -> bytes
    lds_budgets_str = "{\n"
    for arch, per_pipeline in lds_budgets.items():
        capacity_kb = archs[arch]["lds_capacity_kb"]
        lds_budgets_str += f'    # {arch}: {capacity_kb} KB of LDS\n'
        lds_budgets_str += f'    "{arch}": {{\n'
        for pipeline, budget in per_pipeline.items():
            lds_budgets_str += f'        "{pipeline}": {budget},\n'
        lds_budgets_str += "    },\n"
    lds_budgets_str += "}"

    lds_total_str = "{\n"
    for arch in lds_budgets:
        lds_total_str += f'    "{arch}": {archs[arch]["lds_capacity_kb"] * 1024},\n'
    lds_total_str += "}"

    # Build dtype combinations dict
    dtype_combos = specs.get("dtype_combinations", {})
    dtype_combos_str = "{\n"
    for key, info in dtype_combos.items():
        if not key.startswith("_"):
            dtype_combos_str += f'    "{key}": {{"acc": "{info["acc"]}", "notes": "{info["notes"]}"}},\n'
    dtype_combos_str += "}"

    # Build preshuffle warp tile combos dict (operator-specific)
    preshuffle_combos = specs.get("preshuffle_warp_tile_combos", {})
    preshuffle_warp_tile_str = "{\n"
    for arch, dtype_combos_dict in preshuffle_combos.items():
        if not arch.startswith("_"):
            preshuffle_warp_tile_str += f'    "{arch}": {{\n'
            for dtype, combos in dtype_combos_dict.items():
                preshuffle_warp_tile_str += f'        "{dtype}": {combos},\n'
            preshuffle_warp_tile_str += "    },\n"
    preshuffle_warp_tile_str += "}"

    # Build preshuffle pipelines list
    preshuffle_pipelines = specs.get("preshuffle_pipelines", {}).get(
        "supported", ["preshufflev2"]
    )
    preshuffle_pipelines_str = str(preshuffle_pipelines)

    content = f'''# SPDX-License-Identifier: MIT

"""
AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY!

Generated from: arch_specs.json
Generated at: {timestamp}

To update this file:
1. Edit arch_specs.json
2. Run: python generate_arch_specs.py

This module provides architecture-specific configurations for kernel filtering.
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# Architecture Data (Generated from arch_specs.json)
# =============================================================================

# GPU architecture to family mapping
ARCH_FAMILY_MAP: Dict[str, str] = {arch_family_str}

# Element size in bytes for each data type
ELEMENT_SIZE_MAP: Dict[str, float] = {element_sizes}

# Supported warp configurations per architecture [warp_m, warp_n, warp_k]
WARP_SUPPORTED_COMBINATIONS: Dict[str, List[List[int]]] = {warp_configs_str}

# Supported warp tile combinations: arch -> dtype_key -> [[warp_tile_m, n, k], ...]
WARP_TILE_SUPPORTED_COMBINATIONS: Dict[str, Dict[str, List[List[int]]]] = {warp_tile_str}

# Preshuffle-specific warp tile combinations (subset of standard GEMM)
PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS: Dict[str, Dict[str, List[List[int]]]] = {preshuffle_warp_tile_str}

# Preshuffle-supported pipelines
PRESHUFFLE_PIPELINES: List[str] = {preshuffle_pipelines_str}

# LDS staging budget in bytes: arch -> pipeline -> bytes.
# Resolved from each architecture's lds_capacity_kb in arch_specs.json.
LDS_CAPACITY_LIMITS_BY_ARCH: Dict[str, Dict[str, int]] = {lds_budgets_str}

# Total physical LDS per architecture, in bytes. Mirrors get_lds_size() in
# include/ck_tile/core/arch/arch.hpp.
LDS_TOTAL_CAPACITY_BY_ARCH: Dict[str, int] = {lds_total_str}

# Smallest budget shipped, handed to architectures we do not recognise.
_SMALLEST_LDS_BUDGET: Dict[str, int] = LDS_CAPACITY_LIMITS_BY_ARCH[
    min(LDS_CAPACITY_LIMITS_BY_ARCH,
        key=lambda a: LDS_CAPACITY_LIMITS_BY_ARCH[a]["default"])
]
_SMALLEST_LDS_CAPACITY: int = min(LDS_TOTAL_CAPACITY_BY_ARCH.values())

# Unsupported trait combinations: (pipeline, epilogue, scheduler)
TRAIT_UNSUPPORTED_COMBINATIONS: Set[Tuple[str, str, str]] = {unsupported_str}

# Valid dtype combinations: (A_dtype, B_dtype) -> acc_dtype and notes
DTYPE_COMBINATIONS: Dict[str, Dict[str, str]] = {dtype_combos_str}

# =============================================================================
# Helper Functions
# =============================================================================

def get_supported_archs() -> List[str]:
    """Get list of all supported GPU architectures."""
    return list(ARCH_FAMILY_MAP.keys())


def get_arch_family(gpu_arch: str) -> str:
    """Get the GPU family for an architecture."""
    return ARCH_FAMILY_MAP.get(gpu_arch.lower(), "unknown")


def get_element_size(dtype: str) -> float:
    """Get element size in bytes for a data type."""
    return ELEMENT_SIZE_MAP.get(dtype.lower(), 2.0)


def get_warp_configs(gpu_arch: str) -> List[List[int]]:
    """Get supported warp configurations for an architecture."""
    return WARP_SUPPORTED_COMBINATIONS.get(gpu_arch.lower(), [])


def get_warp_tile_combos(gpu_arch: str, dtype_key: str) -> List[List[int]]:
    """Get supported warp tile combinations for arch and data types."""
    gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(gpu_arch.lower(), {{}})
    return gpu_combos.get(dtype_key.lower(), [])


def get_lds_limit(gpu_arch: str, pipeline: str, double_smem_buffer: bool = False) -> int:
    """Get the LDS staging budget in bytes for an architecture and pipeline.

    double_smem_buffer covers the pipelines that stage two LDS buffers by
    configuration rather than by construction (mem, compv3, compv5, compv6).
    Those allocate 2 * (A + B), so the A + B budget is halved. Pipelines that
    always double already carry that in their per-pipeline budget, hence the
    min(): the budget is never halved twice.
    """
    arch = gpu_arch.lower()
    per_pipeline = LDS_CAPACITY_LIMITS_BY_ARCH.get(arch)
    if per_pipeline is None:
        # Unrecognised target: hand back the smallest budget we ship, never the
        # largest. Too small only costs us kernels; too large produces kernels
        # that cannot launch.
        per_pipeline = _SMALLEST_LDS_BUDGET

    budget = per_pipeline.get(pipeline.lower(), per_pipeline["default"])

    if double_smem_buffer:
        capacity = LDS_TOTAL_CAPACITY_BY_ARCH.get(arch, _SMALLEST_LDS_CAPACITY)
        budget = min(budget, capacity // 2)

    return budget


def is_trait_combo_unsupported(pipeline: str, epilogue: str, scheduler: str) -> bool:
    """Check if a trait combination is unsupported."""
    return (pipeline.lower(), epilogue.lower(), scheduler.lower()) in TRAIT_UNSUPPORTED_COMBINATIONS


def get_dtype_info(dtype_a: str, dtype_b: str) -> Dict[str, str]:
    """Get accumulator type and notes for a dtype combination."""
    key = f"{{dtype_a.lower()}}_{{dtype_b.lower()}}"
    return DTYPE_COMBINATIONS.get(key, {{"acc": "fp32", "notes": "unknown"}})


def is_dtype_combo_valid(dtype_a: str, dtype_b: str) -> bool:
    """Check if a dtype combination is valid."""
    key = f"{{dtype_a.lower()}}_{{dtype_b.lower()}}"
    return key in DTYPE_COMBINATIONS


def get_valid_dtype_combos() -> List[str]:
    """Get list of all valid dtype combinations."""
    return list(DTYPE_COMBINATIONS.keys())
'''

    output_path.write_text(content)
    print(f"Generated: {output_path}")


def generate_cpp_header(specs: Dict[str, Any], output_path: Path):
    """Generate C++ header from arch specs."""

    timestamp = datetime.now().isoformat()

    # Extract data
    archs = specs["architectures"]
    element_sizes = specs["element_sizes"]
    lds_budgets = resolve_pipeline_lds_budgets(specs)
    specs["unsupported_trait_combos"]["combinations"]

    # Build arch enum and string functions
    arch_enums = []
    arch_to_string_cases = []
    string_to_arch_cases = []

    for arch, data in archs.items():
        enum_name = arch.upper().replace("GFX", "GFX_")
        arch_enums.append(f"    {enum_name},")
        arch_to_string_cases.append(
            f'        case GpuArch::{enum_name}: return "{arch}";'
        )
        string_to_arch_cases.append(
            f'    if (arch_str == "{arch}") return GpuArch::{enum_name};'
        )

    # Build warp configs switch
    warp_config_cases = []
    for arch, data in archs.items():
        enum_name = arch.upper().replace("GFX", "GFX_")
        configs = ", ".join(
            [f"{{{c[0]}, {c[1]}, {c[2]}}}" for c in data["warp_configs"]]
        )
        warp_config_cases.append(
            f"        case GpuArch::{enum_name}: return {{{configs}}};"
        )

    # Build element size switch
    # Include all data types defined in kernel_key.hpp DataType enum
    elem_size_cases = []
    dtype_enum_map = {
        "fp16": "FP16",
        "bf16": "BF16",
        "fp32": "FP32",
        "fp64": "FP64",
        "fp8": "FP8",
        "bf8": "BF8",
        "int8": "INT8",
        "int4": "INT4",
        "int32": "INT32",
    }
    for dtype, size in element_sizes.items():
        if dtype in dtype_enum_map:
            elem_size_cases.append(
                f"        case DataType::{dtype_enum_map[dtype]}: return {float(size)}f;"
            )

    # Build LDS budgets as a nested switch: architecture, then pipeline.
    pipeline_enum_map = {
        "mem": "Mem",
        "compv1": "CompV1",
        "compv2": "CompV2",
        "compv3": "CompV3",
        "compv4": "CompV4",
        "compv5": "CompV5",
        "compv6": "CompV6",
        "preshufflev1": "PreShuffleV1",
        "preshufflev2": "PreShuffleV2",
        "wavelet": "Wavelet",
        # comp_async has no Pipeline enumerator, so it is intentionally absent:
        # its budget exists only on the Python side.
    }

    def _lds_pipeline_switch(per_pipeline: Dict[str, int], indent: str) -> list:
        lines = [f"{indent}switch(pipeline)", f"{indent}{{"]
        for pipeline, budget in per_pipeline.items():
            if pipeline in pipeline_enum_map:
                lines.append(
                    f"{indent}case Pipeline::{pipeline_enum_map[pipeline]}: "
                    f"return {budget};"
                )
        lines.append(f"{indent}default: return {per_pipeline['default']};")
        lines.append(f"{indent}}}")
        return lines

    lds_budget_cases = []
    for arch, per_pipeline in lds_budgets.items():
        enum_name = arch.upper().replace("GFX", "GFX_")
        capacity_kb = archs[arch]["lds_capacity_kb"]
        lds_budget_cases.append(
            f"    case GpuArch::{enum_name}: // {capacity_kb} KB of LDS"
        )
        lds_budget_cases.extend(_lds_pipeline_switch(per_pipeline, "        "))

    # Unrecognised targets get the smallest budget we ship.
    smallest_arch = min(lds_budgets, key=lambda a: lds_budgets[a]["default"])
    lds_budget_cases.append("    case GpuArch::UNKNOWN:")
    lds_budget_cases.append("    default:")
    lds_budget_cases.extend(_lds_pipeline_switch(lds_budgets[smallest_arch], "        "))

    # Total physical LDS per architecture, used to halve the budget for
    # pipelines that stage two buffers by configuration.
    lds_total_cases = []
    for arch in lds_budgets:
        enum_name = arch.upper().replace("GFX", "GFX_")
        lds_total_cases.append(
            f"    case GpuArch::{enum_name}: return {archs[arch]['lds_capacity_kb'] * 1024};"
        )
    smallest_capacity = min(archs[a]["lds_capacity_kb"] for a in lds_budgets) * 1024
    lds_total_cases.append("    case GpuArch::UNKNOWN:")
    lds_total_cases.append(f"    default: return {smallest_capacity};")

    content = f"""// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY!
 *
 * Generated from: arch_specs.json
 * Generated at: {timestamp}
 * 
 * To update this file:
 * 1. Edit arch_specs.json
 * 2. Run: python generate_arch_specs.py
 */

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include <array>
#include <string>
#include <vector>
#include <cstdint>

namespace ck_tile {{
namespace dispatcher {{
namespace arch_specs {{

// =============================================================================
// GPU Architecture Enum (Generated)
// =============================================================================

enum class GpuArch : std::uint8_t {{
{chr(10).join(arch_enums)}
    UNKNOWN
}};

// =============================================================================
// String Conversion Functions (Generated)
// =============================================================================

inline std::string arch_to_string(GpuArch arch) {{
    switch (arch) {{
{chr(10).join(arch_to_string_cases)}
        default: return "unknown";
    }}
}}

inline GpuArch string_to_arch(const std::string& arch_str) {{
{chr(10).join(string_to_arch_cases)}
    return GpuArch::UNKNOWN;
}}

// =============================================================================
// Element Size (Generated)
// =============================================================================

inline float element_size(DataType dtype) {{
    switch (dtype) {{
{chr(10).join(elem_size_cases)}
        default: return 2.0f;
    }}
}}

// =============================================================================
// Warp Configurations (Generated)
// =============================================================================

using WarpConfig = std::array<int, 3>;

inline std::vector<WarpConfig> get_supported_warp_configs(GpuArch arch) {{
    switch (arch) {{
{chr(10).join(warp_config_cases)}
        default: return {{}};
    }}
}}

// =============================================================================
// LDS Capacity Limits (Generated)
// =============================================================================

// LDS staging budget in bytes for the A+B tiles, per architecture and
// pipeline. The budget depends on the target: a tile that overflows one
// architecture's LDS may fit comfortably in another's.
inline std::size_t get_lds_capacity(GpuArch arch, Pipeline pipeline) {{
    switch(arch)
    {{
{chr(10).join(lds_budget_cases)}
    }}
}}

// Total physical LDS per architecture, in bytes. Mirrors get_lds_size() in
// include/ck_tile/core/arch/arch.hpp.
inline std::size_t get_lds_total_capacity(GpuArch arch) {{
    switch(arch)
    {{
{chr(10).join(lds_total_cases)}
    }}
}}

// =============================================================================
// Unsupported Trait Combinations (Generated)
// =============================================================================

inline bool is_trait_unsupported(Pipeline pipeline, [[maybe_unused]] Epilogue epilogue, Scheduler scheduler) {{
    // Generated from unsupported_trait_combos in arch_specs.json
    if (scheduler == Scheduler::Interwave) {{
        if (pipeline == Pipeline::CompV3 || pipeline == Pipeline::CompV4) {{
            return true;
        }}
    }}
    return false;
}}

}} // namespace arch_specs
}} // namespace dispatcher
}} // namespace ck_tile
"""

    output_path.write_text(content)
    _clang_format_in_place(output_path)
    print(f"Generated: {output_path}")


def _find_clang_format_config() -> Path:
    """Locate the repository's .clang-format by walking up from this script."""
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidate = parent / ".clang-format"
        if candidate.is_file():
            return candidate
    return None


def _clang_format_in_place(path: Path):
    """Format generated C++ so it satisfies the repository's clang-format hook.

    The style file is resolved from this script's location, not from the output
    path. Plain '-style=file' searches upward from the file being formatted, so
    writing the header outside the repository (via --cpp-output-dir) would
    silently fall back to the built-in style and emit a differently formatted,
    non-conforming file. Pinning the config keeps the output byte-identical
    wherever it is written.

    Best effort: if clang-format is unavailable the file is still valid C++, it
    just needs formatting before it can be committed.
    """
    config = _find_clang_format_config()
    style = f"file:{config}" if config else "file"

    for tool in ("clang-format-18", "clang-format"):
        exe = shutil.which(tool)
        if exe is None:
            continue
        try:
            subprocess.run(
                [exe, "-i", f"-style={style}", str(path)],
                check=True,
                capture_output=True,
            )
            return
        except subprocess.CalledProcessError as exc:
            print(f"Warning: {tool} failed on {path}: {exc.stderr.decode().strip()}")
            return
    print(f"Warning: clang-format not found; {path.name} may need formatting.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python and C++ code from arch_specs.json"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=SCRIPT_DIR / "arch_specs.json",
        help="Path to arch_specs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--cpp-output-dir",
        type=Path,
        default=None,
        help="Output directory for C++ header (defaults to dispatcher/include/...)",
    )

    args = parser.parse_args()

    # Load specs
    print(f"Loading: {args.json}")
    specs = load_arch_specs(args.json)

    # Generate Python module
    py_output = args.output_dir / "arch_specs_generated.py"
    generate_python_module(specs, py_output)

    # Generate C++ header
    if args.cpp_output_dir:
        cpp_output = args.cpp_output_dir / "arch_specs_generated.hpp"
    else:
        cpp_output = (
            SCRIPT_DIR.parent
            / "include"
            / "ck_tile"
            / "dispatcher"
            / "arch_specs_generated.hpp"
        )

    cpp_output.parent.mkdir(parents=True, exist_ok=True)
    generate_cpp_header(specs, cpp_output)

    print("\nDone! To apply changes:")
    print("  1. Python code will automatically use arch_specs_generated.py")
    print("  2. C++ code includes arch_specs_generated.hpp")


if __name__ == "__main__":
    main()
