#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generates dispatcher-based kernels for the CK Profiler (all directions).
#
# This script:
# 1. Generates GEMM + depthwise configs from tile_math rules via get_default_configs()
# 2. Calls UnifiedGroupedConvCodegen.generate_all() to emit C++ kernel headers
# 3. Generates include_all_grouped_conv_<variant>_kernels.hpp
# 4. Generates chunked register_*_chunk_N.cpp files + register_all_grouped_conv_kernels.cpp
#
# Usage:
#   python3 generate_profiler_kernels.py \
#     --variant {fwd,bwd_data,bwd_weight} \
#     --output-dir <generated-kernel-output-dir> \
#     --arch gfx950 \
#     [--mode tests|profiler]

import argparse
import sys
from pathlib import Path

from registration_codegen import generate_chunked_registration

VARIANT_CONFIG = {
    "fwd": {
        "glob_pattern": "grouped_conv_fwd_*.hpp",
        "include_all_header": "include_all_grouped_conv_fwd_kernels.hpp",
        "description": "forward",
        "op_enum": "GroupedConvOp::Forward",
        "run_fn_maker": "backends::make_conv_fwd_run_fn",
        "is_supported_fn_maker": "backends::make_conv_fwd_is_supported_fn",
        "register_fn_name": "register_all_grouped_conv_fwd_kernels",
    },
    "bwd_data": {
        "glob_pattern": "grouped_conv_bwd_data_*.hpp",
        "include_all_header": "include_all_grouped_conv_bwd_data_kernels.hpp",
        "description": "backward data",
        "op_enum": "GroupedConvOp::BackwardData",
        "run_fn_maker": "backends::make_conv_bwd_data_run_fn",
        "is_supported_fn_maker": "backends::make_conv_bwd_data_is_supported_fn",
        "register_fn_name": "register_all_grouped_conv_bwd_data_kernels",
    },
    "bwd_weight": {
        "glob_pattern": "grouped_conv_bwd_weight_*.hpp",
        "include_all_header": "include_all_grouped_conv_bwd_weight_kernels.hpp",
        "description": "backward weight",
        "op_enum": "GroupedConvOp::BackwardWeight",
        "run_fn_maker": "backends::make_conv_bwd_weight_run_fn",
        "is_supported_fn_maker": "backends::make_conv_bwd_weight_is_supported_fn",
        "register_fn_name": "register_all_grouped_conv_bwd_weight_kernels",
    },
}

VARIANT_MAP = {
    "fwd": "FORWARD",
    "bwd_data": "BACKWARD_DATA",
    "bwd_weight": "BACKWARD_WEIGHT",
}


def _ensure_codegen_importable():
    """Ensure the codegen directory is on sys.path."""
    codegen_dir = str(Path(__file__).resolve().parent.parent / "codegen")
    if codegen_dir not in sys.path:
        sys.path.insert(0, codegen_dir)


def collect_kernel_headers(output_dir, glob_pattern):
    """Collect all generated .hpp kernel headers matching the variant pattern."""
    headers = sorted(Path(output_dir).glob(glob_pattern))
    return headers


def generate_include_all_header(headers, output_dir, header_filename, description):
    """Generate include_all_grouped_conv_<variant>_kernels.hpp."""
    lines = [
        "// Auto-generated \u2014 do not edit",
        f"// Includes all generated {description} kernel headers.",
        "#pragma once",
        "",
    ]
    for h in headers:
        lines.append(f'#include "{h.name}"')
    lines.append("")

    path = Path(output_dir) / header_filename
    path.write_text("\n".join(lines))
    print(f"Generated {path} ({len(headers)} includes)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate dispatcher-based kernels for CK Profiler."
    )
    parser.add_argument("--variant", required=True, choices=list(VARIANT_CONFIG.keys()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--rule-set", default="tests",
                        choices=["profiler", "tests", "full", "full-tests", "tiny", "default"],
                        help="Rule set: 'profiler'/'tests' (CK Builder "
                             "profiler/tests instance sets generated in memory "
                             "from the .conf configs), 'full' (full rule-derived "
                             "per-(variant,ndim,datatype) set), 'full-tests' "
                             "(~20% stratified subset of 'full'), 'tiny' "
                             "(minimal >=10-config subset of 'full-tests'), or "
                             "'default' (original hand-curated heuristics)")

    args = parser.parse_args()
    cfg = VARIANT_CONFIG[args.variant]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_codegen_importable()
    from unified_grouped_conv_codegen import (
        UnifiedGroupedConvCodegen,
        GroupedConvVariant,
        get_default_configs,
    )

    variant_enum = GroupedConvVariant[VARIANT_MAP[args.variant]]
    datatypes = ["fp16", "bf16", "fp32"]

    # --- Step 1: Generate configs from rules ---
    print(f"Generating configs from rules (variant={args.variant}, "
          f"arch={args.arch}, rule_set={args.rule_set})...")

    configs = get_default_configs(
        arch=args.arch,
        variants=[variant_enum],
        ndims=[2, 3],
        datatypes=datatypes,
        rule_set=args.rule_set,
    )
    print(f"Generated {len(configs)} configs from rules")

    if not configs:
        print("ERROR: No configs generated from rules", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Generate kernel headers ---
    codegen = UnifiedGroupedConvCodegen(
        output_dir=output_dir, gpu_target=args.arch, enable_arch_filter=False,
    )
    codegen.generate_all(configs, datatypes=datatypes)

    # --- Step 3: Collect headers and generate registration ---
    headers = collect_kernel_headers(output_dir, cfg["glob_pattern"])
    print(f"Found {len(headers)} generated kernel headers")

    if not headers:
        print("ERROR: No kernel headers generated", file=sys.stderr)
        sys.exit(1)

    generate_include_all_header(headers, output_dir, cfg["include_all_header"], cfg["description"])
    generate_chunked_registration(
        headers, output_dir,
        variant=args.variant,
        op_enum=cfg["op_enum"],
        run_fn_maker=cfg["run_fn_maker"],
        is_supported_fn_maker=cfg["is_supported_fn_maker"],
        register_fn_name=cfg["register_fn_name"],
    )

    print(f"\nDone. {len(headers)} kernels ready in {output_dir}")


if __name__ == "__main__":
    main()
