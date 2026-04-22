#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Self-contained build script for C++ grouped convolution examples.

Parses DECL_GROUPED_CONV_KERNEL_SET declarations from source files,
generates the needed kernels, and compiles the example.

Includes validation and auto-correction via wildcard expansion.

Usage:
    python3 compile_grouped_conv_examples.py examples/grouped_conv/cpp/02_grouped_conv_forward.cpp
    python3 compile_grouped_conv_examples.py examples/grouped_conv/cpp/03_grouped_conv_validation.cpp --no-compile
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CK_ROOT = DISPATCHER_DIR.parent

sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from dispatcher_common import (  # noqa: E402
    print_phase,
    print_success,
    print_error,
    print_info,
    find_hipcc,
    get_arch_filter_data,
    get_build_dir,
    get_ck_root,
    get_dispatcher_root,
    get_generated_kernels_dir,
)


def extract_grouped_conv_declarations(source_file: Path) -> list:
    """Extract DECL_GROUPED_CONV_KERNEL_SET declarations from C++ source."""
    content = source_file.read_text()
    declarations = []

    # Pattern: DECL_GROUPED_CONV_KERNEL_SET(name, .add(...).add(...))
    # Find all DECL_GROUPED_CONV_KERNEL_SET blocks by matching parentheses
    pattern_start = r"DECL_GROUPED_CONV_KERNEL_SET\s*\(\s*(\w+)\s*,"
    for match in re.finditer(pattern_start, content):
        set_name = match.group(1)
        start_pos = match.end()

        # Find matching closing paren by counting parens
        paren_count = 1  # We're already inside the first paren
        end_pos = start_pos
        for i, c in enumerate(content[start_pos:]):
            if c == "(":
                paren_count += 1
            elif c == ")":
                paren_count -= 1
                if paren_count == 0:
                    end_pos = start_pos + i
                    break

        set_body = content[start_pos:end_pos]

        # Pattern 1: Simple add("dtype", "layout", "conv_type", tile_k, tile_c)
        simple_add = (
            r'\.add\s*\(\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*(\d+)\s*,\s*(\d+)'
        )
        for add_match in re.finditer(simple_add, set_body):
            conv_type = add_match.group(3)
            default_pipeline = (
                "compv3" if conv_type in ("bwd_data", "bwd_weight") else "compv4"
            )
            declarations.append(
                {
                    "set": set_name,
                    "dtype": add_match.group(1),
                    "layout": add_match.group(2),
                    "conv_type": conv_type,
                    "tile_k": int(add_match.group(4)),
                    "tile_c": int(add_match.group(5)),
                    "num_dims": 2,
                    "pipeline": default_pipeline,
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
        # Find all .add( positions that start with ConvSig()
        full_add = r"\.add\s*\(\s*ConvSig\(\)"
        add_positions = [m.start() for m in re.finditer(full_add, set_body)]

        for pos in add_positions:
            # Find matching closing paren by counting parens
            paren_count = 0
            in_add = False
            end = pos
            for i, c in enumerate(set_body[pos:]):
                if c == "(":
                    paren_count += 1
                    in_add = True
                elif c == ")":
                    paren_count -= 1
                    if in_add and paren_count == 0:
                        end = pos + i + 1
                        break

            add_str = set_body[pos:end]

            # Extract signature part (between ConvSig() and ConvAlgo())
            sig_match = re.search(r"ConvSig\(\)(.*?)ConvAlgo\(\)", add_str, re.DOTALL)
            if not sig_match:
                continue
            sig_str = sig_match.group(1)

            # Extract algorithm part (between ConvAlgo() and arch string)
            algo_match = re.search(
                r'ConvAlgo\(\)(.*?),\s*"(\w+)"\s*\)', add_str, re.DOTALL
            )
            if not algo_match:
                continue
            algo_str = algo_match.group(1)
            arch = algo_match.group(2)

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

            # Parse additional parameters
            vector_a, vector_b, vector_c = 4, 8, 8
            vector_match = re.search(
                r"\.vector_sizes\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", algo_str
            )
            if vector_match:
                vector_a = int(vector_match.group(1))
                vector_b = int(vector_match.group(2))
                vector_c = int(vector_match.group(3))

            block_per_cu = 1
            block_per_cu_match = re.search(r"\.block_per_cu\s*\(\s*(\d+)", algo_str)
            if block_per_cu_match:
                block_per_cu = int(block_per_cu_match.group(1))

            memory_op = "set"
            memory_op_match = re.search(r'\.memory_op\s*\(\s*"(\w+)"', algo_str)
            if memory_op_match:
                memory_op = memory_op_match.group(1)

            epilogue = "cshuffle"
            epilogue_match = re.search(r'\.epilogue\s*\(\s*"(\w+)"', algo_str)
            if epilogue_match:
                epilogue = epilogue_match.group(1)

            # Parse num_wave_groups (for V5 pipeline)
            num_wave_groups = 1
            nwg_match = re.search(r"\.num_wave_groups\s*\(\s*(\d+)", algo_str)
            if nwg_match:
                num_wave_groups = int(nwg_match.group(1))

            # Parse num_groups_to_merge (for merged group grouped convolution)
            num_groups_to_merge = 1
            ngm_match = re.search(r"\.num_groups_to_merge\s*\(\s*(\d+)", algo_str)
            if ngm_match:
                num_groups_to_merge = int(ngm_match.group(1))

            # Parse double_smem_buffer (for V4 pipeline)
            double_smem_buffer = False
            dsb_match = re.search(
                r"\.double_smem_buffer\s*\(\s*(true|false)", algo_str, re.I
            )
            if dsb_match:
                double_smem_buffer = dsb_match.group(1).lower() == "true"

            # Parse padding flags
            pad_m, pad_n, pad_k = True, True, True
            padding_match = re.search(
                r"\.padding\s*\(\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)",
                algo_str,
                re.I,
            )
            if padding_match:
                pad_m = padding_match.group(1).lower() == "true"
                pad_n = padding_match.group(2).lower() == "true"
                pad_k = padding_match.group(3).lower() == "true"

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
                    "vector_a": vector_a,
                    "vector_b": vector_b,
                    "vector_c": vector_c,
                    "block_per_cu": block_per_cu,
                    "memory_op": memory_op,
                    "epilogue": epilogue,
                    "num_wave_groups": num_wave_groups,
                    "num_groups_to_merge": num_groups_to_merge,
                    "double_smem_buffer": double_smem_buffer,
                    "pad_m": pad_m,
                    "pad_n": pad_n,
                    "pad_k": pad_k,
                    "arch": arch,
                }
            )

    return declarations


# =============================================================================
# VALIDATION AND AUTO-CORRECTION
# =============================================================================


def is_grouped_conv_wildcard_declaration(decl: dict) -> bool:
    """Check if a declaration uses wildcards (-1 or '*')."""
    wildcard_fields = ["wave_m", "wave_n", "warp_m", "warp_n", "pipeline", "scheduler"]
    for field in wildcard_fields:
        val = decl.get(field)
        if val == -1 or val == "*":
            return True
    return False


def validate_grouped_conv_kernel_config(decl: dict, arch: str = "gfx942") -> tuple:
    """Validate a grouped conv kernel configuration against known supported combinations.

    Returns: (is_valid, error_message)
    """
    # Skip validation for wildcards - expansion will filter invalid combos
    if is_grouped_conv_wildcard_declaration(decl):
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
    acc_dtype = "int32" if dtype == "int8" else "fp32"
    dtype_key = f"{dtype}_{dtype}_{acc_dtype}"
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


def expand_grouped_conv_declaration_with_arch_filter(
    decl: dict, arch: str = "gfx942"
) -> list:
    """Expand a grouped conv declaration with wildcards into valid configurations.

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
    acc_dtype = "int32" if dtype == "int8" else "fp32"
    dtype_key = f"{dtype}_{dtype}_{acc_dtype}"
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
    return expanded[:1]  # Just use first valid config for grouped conv


def validate_and_expand_grouped_conv_declarations(
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
        if is_grouped_conv_wildcard_declaration(decl):
            wildcard_count += 1
            continue

        is_valid, error_msg = validate_grouped_conv_kernel_config(decl, decl_arch)
        if not is_valid:
            print(f"\n    WARNING Invalid grouped conv configuration: {decl_name}")

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
                    f"wave: {original_values['wave']} -> [wildcard expansion]"
                )

            if "warp tile" in error_msg.lower():
                original_values["warp"] = (
                    f"[{decl.get('warp_m', 32)}, {decl.get('warp_n', 32)}, {decl.get('warp_k', 16)}]"
                )
                decl["warp_m"] = -1
                decl["warp_n"] = -1
                corrections.append(
                    f"warp_tile: {original_values['warp']} -> [wildcard expansion]"
                )

            if "trait combination" in error_msg.lower():
                original_values["pipeline"] = decl.get("pipeline", "compv4")
                original_values["scheduler"] = decl.get("scheduler", "intrawave")
                decl["pipeline"] = "*"
                decl["scheduler"] = "*"
                corrections.append(
                    f"pipeline: {original_values['pipeline']} -> [wildcard expansion]"
                )
                corrections.append(
                    f"scheduler: {original_values['scheduler']} -> [wildcard expansion]"
                )

            # Print the auto-corrections
            print("      AUTO-CORRECTION:")
            for corr in corrections:
                print(f"        - {corr}")
            auto_corrections.append((decl_name, corrections))

            invalid_count += 1
            wildcard_count += 1

    if invalid_count > 0:
        print(
            f"\n    WARNING {invalid_count} invalid config(s) auto-corrected via wildcard expansion"
        )

    if wildcard_count > 0:
        print(
            f"    OK {len(declarations) - wildcard_count} explicit + {wildcard_count} wildcard (will expand)"
        )
    else:
        print(f"    OK All {len(declarations)} configurations valid")

    # Expand wildcards
    print("\n    Expanding wildcards to valid configurations...")
    expanded_declarations = []
    for decl in declarations:
        decl_arch = decl.get("arch", arch)
        decl_name = (
            f"{decl['dtype']}_{decl['conv_type']}_{decl['tile_k']}x{decl['tile_c']}"
        )

        expanded = expand_grouped_conv_declaration_with_arch_filter(decl, decl_arch)
        expanded_declarations.extend(expanded)

        if len(expanded) > 1:
            print(
                f"      {decl_name}: expanded to {len(expanded)} valid configurations"
            )
            for exp in expanded[:3]:
                wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
                warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
                print(
                    f"        -> wave={wave_str}, warp={warp_str}, pipeline={exp['pipeline']}"
                )
            if len(expanded) > 3:
                print(f"        ... and {len(expanded) - 3} more")
        elif is_grouped_conv_wildcard_declaration(decl) and len(expanded) == 1:
            exp = expanded[0]
            wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
            warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
            print(f"      {decl_name}: -> wave={wave_str}, warp={warp_str}")

    if len(expanded_declarations) != len(declarations):
        print(
            f"\n    Total: {len(declarations)} declarations -> {len(expanded_declarations)} configurations"
        )

    return expanded_declarations


def _generate_single_grouped_conv_kernel(args: tuple) -> tuple:
    """Generate one grouped conv kernel (picklable for ProcessPoolExecutor).

    Args: (decl, output_dir_str, gpu_target)
    Returns: (idx, filepath_str or None, error_str or None)
    """
    decl, output_dir_str, gpu_target = args
    output_dir = Path(output_dir_str)
    idx = decl.get("_idx", 0)

    try:
        from codegen_common import TileConfig
        from unified_grouped_conv_codegen import (
            GroupedConvKernelConfig,
            GroupedConvTraitConfig,
            GroupedConvVariant,
            UnifiedGroupedConvCodegen,
        )

        # Map conv_type to variant
        variant = GroupedConvVariant.FORWARD
        if decl["conv_type"] == "bwd_data":
            variant = GroupedConvVariant.BACKWARD_DATA
        elif decl["conv_type"] == "bwd_weight":
            variant = GroupedConvVariant.BACKWARD_WEIGHT

        pipeline = decl.get("pipeline", "compv4")
        adj_tile_k = 64 * 2 if pipeline == "compv4" else 64

        # Create tile config (tile_m=tile_k, tile_n=tile_c for conv GEMM view)
        tile = TileConfig(
            tile_m=decl["tile_k"],
            tile_n=decl["tile_c"],
            tile_k=adj_tile_k,
            warp_m=decl["wave_m"],
            warp_n=decl["wave_n"],
            warp_k=decl.get("wave_k", 1),
            warp_tile_m=decl["warp_m"],
            warp_tile_n=decl["warp_n"],
            warp_tile_k=decl["warp_k"],
        )

        trait = GroupedConvTraitConfig(
            pipeline=pipeline,
            scheduler=decl["scheduler"],
            epilogue=decl.get("epilogue", "cshuffle"),
            double_smem_buffer=decl.get("double_smem_buffer", False),
            pad_m=decl.get("pad_m", True),
            pad_n=decl.get("pad_n", True),
            pad_k=decl.get("pad_k", True),
            num_groups_to_merge=decl.get("num_groups_to_merge", 1),
        )

        config = GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=variant,
            ndim_spatial=decl["num_dims"],
            arch=decl.get("arch", gpu_target),
            vector_size_a=decl.get("vector_a", 4),
            vector_size_b=decl.get("vector_b", 8),
            vector_size_c=decl.get("vector_c", 8),
            block_per_cu=decl.get("block_per_cu", 1),
            num_wave_groups=decl.get("num_wave_groups", 1),
            num_groups_to_merge=decl.get("num_groups_to_merge", 1),
            double_smem_buffer=decl.get("double_smem_buffer", False),
        )

        codegen = UnifiedGroupedConvCodegen(output_dir, gpu_target=gpu_target)
        kernel_path, _ = codegen.generate_kernel(config, decl["dtype"], variant)
        return (idx, str(kernel_path), None)

    except Exception as e:
        return (idx, None, str(e))


def generate_grouped_conv_kernels(
    declarations: list,
    output_dir: Path,
    gpu_target: str = "gfx942",
    max_workers: Optional[int] = None,
) -> list:
    """Generate grouped convolution kernels using unified_grouped_conv_codegen.

    Uses ProcessPoolExecutor for parallel kernel generation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare work items (add _idx for ordering)
    work_items = []
    for idx, decl in enumerate(declarations):
        decl_copy = decl.copy()
        decl_copy["_idx"] = idx
        work_items.append((decl_copy, str(output_dir), gpu_target))

    max_workers = max_workers or min(len(work_items), os.cpu_count() or 4)
    generated = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single_grouped_conv_kernel, w): w[0]["_idx"]
            for w in work_items
        }
        for future in as_completed(futures):
            idx, path, err = future.result()
            if path:
                generated.append(Path(path))
                print_info(f"    Generated: {Path(path).name}")
            else:
                failed.append((idx, err))
                print_error(f"    Failed kernel {idx + 1}: {err}")

    if failed:
        for idx, err in failed[:3]:
            print_error(f"  Kernel {idx + 1}: {err[:200]}")
        if len(failed) > 3:
            print_error(f"  ... and {len(failed) - 3} more failures")

    return generated


def compile_grouped_conv_example(
    source_file: Path,
    output_bin: Path,
    kernel_headers: list,
    hipcc: str,
    gpu_target: str,
) -> bool:
    """Compile the C++ example with generated kernels."""
    kernel_dir = get_generated_kernels_dir()
    ck_root = get_ck_root()
    dispatcher_dir = get_dispatcher_root()

    includes = [
        f"-I{ck_root / 'include'}",
        f"-I{dispatcher_dir / 'include'}",
        f"-I{kernel_dir}",
    ]

    # Build include flags for generated kernels
    kernel_includes = []
    for header in kernel_headers:
        kernel_includes.extend(["-include", str(header)])

    # Add define to indicate kernels are available
    defines = ["-DGROUPED_CONV_KERNEL_AVAILABLE=1"]

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

    print_info(f"  Compiling: {source_file.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if result.stderr:
            lines = result.stderr.split("\n")
            errors = [line for line in lines if "error:" in line.lower()][:5]
            for err_line in errors:
                print_error(f"    {err_line}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build C++ grouped convolution example with self-contained kernel generation"
    )
    parser.add_argument("source", help="Source file (.cpp)")
    parser.add_argument("--output", "-o", help="Output binary name")
    parser.add_argument("--gpu-target", default="gfx942", help="GPU target")
    parser.add_argument(
        "--no-compile", action="store_true", help="Only generate kernels, don't compile"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Parallel jobs for kernel generation (default: cpu_count)",
    )
    args = parser.parse_args()

    # Resolve source file
    source_file = Path(args.source)
    if not source_file.is_absolute():
        candidates = [
            get_dispatcher_root() / args.source,
            Path.cwd() / args.source,
        ]
        for c in candidates:
            if c.exists():
                source_file = c
                break

    if not source_file.exists():
        print_error(f"Source file not found: {source_file}")
        return 1

    build_dir = get_build_dir()
    kernel_dir = get_generated_kernels_dir()
    output_name = args.output or source_file.stem
    output_bin = build_dir / output_name

    print_success("=== Grouped Conv Example Builder (Self-Contained) ===")

    # Phase 1: Extract declarations
    print_phase(1, "Scanning for DECL_GROUPED_CONV_KERNEL_SET...")
    declarations = extract_grouped_conv_declarations(source_file)

    if not declarations:
        print_error("  No DECL_GROUPED_CONV_KERNEL_SET declarations found!")
        return 1

    print(f"  Found {len(declarations)} kernel declaration(s):")
    for decl in declarations:
        name = f"{decl['dtype']}_{decl['conv_type']}_{decl['num_dims']}d_{decl['tile_k']}x{decl['tile_c']}"
        print(f"    [{decl['set']}] {name}")

    # Phase 2: Validate and expand
    print_phase(2, "Validating and expanding declarations...")
    declarations = validate_and_expand_grouped_conv_declarations(
        declarations, args.gpu_target, args.verbose
    )
    print()

    # Phase 3: Generate kernels
    print_phase(3, "Generating kernels...")
    generated = generate_grouped_conv_kernels(
        declarations, kernel_dir, args.gpu_target, max_workers=args.jobs
    )

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

    print_phase(4, "Compiling example...")
    hipcc_path = find_hipcc()

    if not hipcc_path:
        print_error("  hipcc not found. Install ROCm or set HIPCC env var.")
        print("  To compile manually:")
        ck_root = get_dispatcher_root().parent
        print(
            f"    hipcc -std=c++20 -O2 -I{ck_root / 'include'} -I{get_dispatcher_root() / 'include'} \\"
        )
        print(f"          -I{kernel_dir} \\")
        for h in generated[:1]:
            print(f"          -include {h} \\")
        print("          -DGROUPED_CONV_KERNEL_AVAILABLE=1 \\")
        print(f"          --offload-arch={args.gpu_target} \\")
        print(f"          {source_file} -o {output_bin}")
        return 1

    build_dir.mkdir(parents=True, exist_ok=True)

    if not compile_grouped_conv_example(
        source_file, output_bin, generated, hipcc_path, args.gpu_target
    ):
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
