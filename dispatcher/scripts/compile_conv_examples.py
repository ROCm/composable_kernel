#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Self-contained build script for C++ convolution examples.

Parses DECL_CONV_KERNEL_SET declarations from source files,
generates the needed kernels, and compiles the example.

Includes validation and auto-correction via wildcard expansion.

Usage:
    python3 compile_conv_examples.py examples/conv/cpp/02_conv_forward.cpp
    python3 compile_conv_examples.py examples/conv/cpp/03_conv_validation.cpp --no-compile
"""

import argparse
import json
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


# =============================================================================
# VALIDATION AND AUTO-CORRECTION
# =============================================================================


def get_arch_filter_data() -> dict:
    """Load architecture filter data from arch_specs.json."""
    arch_specs_path = DISPATCHER_DIR / "codegen" / "arch_specs.json"

    if arch_specs_path.exists():
        with open(arch_specs_path) as f:
            specs = json.load(f)

        # Build lookup tables
        supported_archs = list(specs.get("architectures", {}).keys())

        # Build warp combos per arch
        warp_combos = {}
        for arch, arch_data in specs.get("architectures", {}).items():
            warp_combos[arch] = arch_data.get("warp_combos", [[2, 2, 1]])

        # Build warp tile combos per arch and dtype
        warp_tile_combos = {}
        for arch, arch_data in specs.get("architectures", {}).items():
            warp_tile_combos[arch] = {}
            for dtype_key, tiles in arch_data.get("warp_tile_combos", {}).items():
                warp_tile_combos[arch][dtype_key] = tiles

        # Unsupported trait combinations
        trait_unsupported = set()
        for combo in specs.get("trait_combinations", {}).get("unsupported", []):
            trait_unsupported.add(tuple(combo))

        return {
            "supported_archs": supported_archs,
            "warp_combos": warp_combos,
            "warp_tile_combos": warp_tile_combos,
            "trait_unsupported": trait_unsupported,
        }

    # Fallback defaults
    return {
        "supported_archs": [
            "gfx90a",
            "gfx942",
            "gfx950",
            "gfx1100",
            "gfx1200",
            "gfx1201",
        ],
        "warp_combos": {
            "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
        },
        "warp_tile_combos": {},
        "trait_unsupported": {("compv4", "cshuffle", "interwave")},
    }


def is_conv_wildcard_declaration(decl: dict) -> bool:
    """Check if a declaration uses wildcards (-1 or '*')."""
    wildcard_fields = ["wave_m", "wave_n", "warp_m", "warp_n", "pipeline", "scheduler"]
    for field in wildcard_fields:
        val = decl.get(field)
        if val == -1 or val == "*":
            return True
    return False


def validate_conv_kernel_config(decl: dict, arch: str = "gfx942") -> tuple:
    """Validate a conv kernel configuration against known supported combinations.

    Returns: (is_valid, error_message)
    """
    # Skip validation for wildcards - expansion will filter invalid combos
    if is_conv_wildcard_declaration(decl):
        return (True, None)

    arch_data = get_arch_filter_data()

    pipeline = decl.get("pipeline", "compv4")
    scheduler = decl.get("scheduler", "intrawave")
    dtype = decl.get("dtype", "fp16")

    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    warp_m = decl.get("warp_m", 32)
    warp_n = decl.get("warp_n", 32)
    warp_k = decl.get("warp_k", 16)

    errors = []

    # Check trait combination (pipeline, epilogue, scheduler)
    combo = (pipeline, "cshuffle", scheduler)
    if combo in arch_data["trait_unsupported"]:
        errors.append(
            f"Unsupported trait combination: pipeline={pipeline}, scheduler={scheduler}\n"
            f"    Valid schedulers for {pipeline}: intrawave"
        )

    # Check wave configuration for this arch
    warp_combos = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
    wave_cfg = [wave_m, wave_n, wave_k]
    if wave_cfg not in warp_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_combos)
        errors.append(
            f"Unsupported wave configuration [{wave_m},{wave_n},{wave_k}] for {arch}\n"
            f"    Valid wave configs: {valid_str}"
        )

    # Check warp tile configuration for this arch and dtype
    dtype_key = f"{dtype}_{dtype}_{dtype}"
    warp_tile_combos = (
        arch_data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16], [16, 16, 32]])
    )
    warp_cfg = [warp_m, warp_n, warp_k]
    if warp_cfg not in warp_tile_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_tile_combos[:5])
        errors.append(
            f"Unsupported warp tile [{warp_m},{warp_n},{warp_k}] for {arch}/{dtype}\n"
            f"    Valid warp tiles: {valid_str}"
        )

    # Check arch is supported
    if arch not in arch_data["supported_archs"]:
        errors.append(
            f"Unsupported architecture: {arch}\n"
            f"    Supported: {', '.join(arch_data['supported_archs'])}"
        )

    if errors:
        return (False, "\n".join(errors))

    return (True, None)


def expand_conv_declaration_with_arch_filter(decl: dict, arch: str = "gfx942") -> list:
    """Expand a conv declaration with wildcards into valid configurations.

    Wildcards:
      - wave_m/wave_n = -1: Try all valid wave configs for this arch
      - warp_m/warp_n = -1: Try all valid warp tiles for this arch/dtype
      - pipeline/scheduler = "*": Try all valid combinations

    Returns a list of fully-specified declarations.
    """
    arch_data = get_arch_filter_data()
    dtype = decl.get("dtype", "fp16")

    # Get valid combinations for this arch
    valid_wave_combos = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
    dtype_key = f"{dtype}_{dtype}_{dtype}"
    valid_warp_tiles = (
        arch_data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
    )

    # Valid pipelines and schedulers
    valid_pipelines = ["compv3", "compv4"]
    valid_schedulers = ["intrawave"]  # interwave often unsupported

    # Determine which fields need expansion
    expand_wave = decl.get("wave_m", 2) == -1 or decl.get("wave_n", 2) == -1
    expand_warp = decl.get("warp_m", 32) == -1 or decl.get("warp_n", 32) == -1
    expand_pipeline = decl.get("pipeline", "compv4") == "*"
    expand_scheduler = decl.get("scheduler", "intrawave") == "*"

    # Build combinations
    wave_options = (
        valid_wave_combos
        if expand_wave
        else [[decl.get("wave_m", 2), decl.get("wave_n", 2), decl.get("wave_k", 1)]]
    )
    warp_options = (
        valid_warp_tiles
        if expand_warp
        else [[decl.get("warp_m", 32), decl.get("warp_n", 32), decl.get("warp_k", 16)]]
    )
    pipeline_options = (
        valid_pipelines if expand_pipeline else [decl.get("pipeline", "compv4")]
    )
    scheduler_options = (
        valid_schedulers if expand_scheduler else [decl.get("scheduler", "intrawave")]
    )

    expanded = []
    for wave in wave_options:
        for warp in warp_options:
            for pipeline in pipeline_options:
                for scheduler in scheduler_options:
                    # Skip known invalid combinations
                    if (pipeline, "cshuffle", scheduler) in arch_data[
                        "trait_unsupported"
                    ]:
                        continue

                    new_decl = decl.copy()
                    new_decl["wave_m"] = wave[0]
                    new_decl["wave_n"] = wave[1]
                    new_decl["wave_k"] = wave[2]
                    new_decl["warp_m"] = warp[0]
                    new_decl["warp_n"] = warp[1]
                    new_decl["warp_k"] = warp[2]
                    new_decl["pipeline"] = pipeline
                    new_decl["scheduler"] = scheduler

                    expanded.append(new_decl)

    # If no valid expansions, return original (will fail validation later)
    if not expanded:
        return [decl]

    # Return first valid config (or all if needed)
    return expanded[:1]  # Just use first valid config for conv


def validate_and_expand_conv_declarations(
    declarations: list, arch: str, verbose: bool = False
) -> list:
    """Validate declarations and auto-correct invalid ones via wildcard expansion."""
    print(f"\n    Validating against {arch} arch filter...")

    wildcard_count = 0
    invalid_count = 0
    auto_corrections = []

    for decl in declarations:
        decl_arch = decl.get("arch", arch)
        decl_name = (
            f"{decl['dtype']}_{decl['conv_type']}_{decl['tile_k']}x{decl['tile_c']}"
        )

        # Check for wildcards
        if is_conv_wildcard_declaration(decl):
            wildcard_count += 1
            continue

        is_valid, error_msg = validate_conv_kernel_config(decl, decl_arch)
        if not is_valid:
            print(f"\n    ⚠ Invalid conv configuration: {decl_name}")

            # Parse the error and show specific auto-corrections
            corrections = []
            original_values = {}

            if "wave configuration" in error_msg.lower():
                original_values["wave"] = (
                    f"[{decl.get('wave_m', 2)}, {decl.get('wave_n', 2)}, {decl.get('wave_k', 1)}]"
                )
                decl["wave_m"] = -1
                decl["wave_n"] = -1
                corrections.append(
                    f"wave: {original_values['wave']} → [wildcard expansion]"
                )

            if "warp tile" in error_msg.lower():
                original_values["warp"] = (
                    f"[{decl.get('warp_m', 32)}, {decl.get('warp_n', 32)}, {decl.get('warp_k', 16)}]"
                )
                decl["warp_m"] = -1
                decl["warp_n"] = -1
                corrections.append(
                    f"warp_tile: {original_values['warp']} → [wildcard expansion]"
                )

            if "trait combination" in error_msg.lower():
                original_values["pipeline"] = decl.get("pipeline", "compv4")
                original_values["scheduler"] = decl.get("scheduler", "intrawave")
                decl["pipeline"] = "*"
                decl["scheduler"] = "*"
                corrections.append(
                    f"pipeline: {original_values['pipeline']} → [wildcard expansion]"
                )
                corrections.append(
                    f"scheduler: {original_values['scheduler']} → [wildcard expansion]"
                )

            # Print the auto-corrections
            print("      AUTO-CORRECTION:")
            for corr in corrections:
                print(f"        • {corr}")
            auto_corrections.append((decl_name, corrections))

            invalid_count += 1
            wildcard_count += 1

    if invalid_count > 0:
        print(
            f"\n    ⚠ {invalid_count} invalid config(s) auto-corrected via wildcard expansion"
        )

    if wildcard_count > 0:
        print(
            f"    ✓ {len(declarations) - wildcard_count} explicit + {wildcard_count} wildcard (will expand)"
        )
    else:
        print(f"    ✓ All {len(declarations)} configurations valid")

    # Expand wildcards
    print("\n    Expanding wildcards to valid configurations...")
    expanded_declarations = []
    for decl in declarations:
        decl_arch = decl.get("arch", arch)
        decl_name = (
            f"{decl['dtype']}_{decl['conv_type']}_{decl['tile_k']}x{decl['tile_c']}"
        )

        expanded = expand_conv_declaration_with_arch_filter(decl, decl_arch)
        expanded_declarations.extend(expanded)

        if len(expanded) > 1:
            print(
                f"      {decl_name}: expanded to {len(expanded)} valid configurations"
            )
            for exp in expanded[:3]:
                wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
                warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
                print(
                    f"        → wave={wave_str}, warp={warp_str}, pipeline={exp['pipeline']}"
                )
            if len(expanded) > 3:
                print(f"        ... and {len(expanded) - 3} more")
        elif is_conv_wildcard_declaration(decl) and len(expanded) == 1:
            exp = expanded[0]
            wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
            warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
            print(f"      {decl_name}: → wave={wave_str}, warp={warp_str}")

    if len(expanded_declarations) != len(declarations):
        print(
            f"\n    Total: {len(declarations)} declarations → {len(expanded_declarations)} configurations"
        )

    return expanded_declarations


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

    # Phase 2: Validate and expand
    print_phase("\nPhase 2: Validating and expanding declarations...")
    declarations = validate_and_expand_conv_declarations(
        declarations, args.gpu_target, args.verbose
    )
    print()

    # Phase 3: Generate kernels
    print_phase("Phase 3: Generating kernels...")
    generated = generate_conv_kernels(declarations, kernel_dir)

    if not generated:
        print_error("  No kernels generated!")
        return 1

    print(f"  Generated {len(generated)} kernel file(s)")
    print()

    # Phase 4: Compile (optional)
    if args.no_compile:
        print_info("Skipping compilation (--no-compile)")
        print()
        print_success("=== Kernel Generation Complete ===")
        print(f"Kernels in: {kernel_dir}")
        return 0

    print_phase("Phase 4: Compiling example...")
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
