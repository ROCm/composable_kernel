#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generates dispatcher-based kernels for the CK Profiler (all directions).
#
# This script:
# 1. Reads JSON config files
# 2. Calls load_configs_from_json + UnifiedGroupedConvCodegen.generate_all() for each JSON
# 3. Generates include_all_grouped_conv_<variant>_kernels.hpp
# 4. Generates chunked register_*_chunk_N.cpp files + register_all_grouped_conv_kernels.cpp
#
# Usage:
#   python3 generate_profiler_kernels.py \
#     --variant {fwd,bwd_data,bwd_weight} \
#     --config-dir <path-to-json-configs> \
#     --codegen <path-to-unified_grouped_conv_codegen.py> \
#     --output-dir <generated-kernel-output-dir> \
#     --arch gfx950 \
#     [--config-set tests|profiler]

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


def generate_kernels_from_config(config_file, output_dir, arch):
    """Generate kernels for a single JSON config via direct Python API."""
    import json
    from unified_grouped_conv_codegen import UnifiedGroupedConvCodegen, load_configs_from_json

    try:
        configs = load_configs_from_json(config_file, arch=arch)
        # Extract datatype from JSON config (matches old --config-file behavior)
        with open(config_file, "r") as f:
            config_data = json.load(f)
        datatype = config_data["datatype"]
        # The JSON configs are valid for all architectures.
        # Hence, disable the arch_filter.
        codegen = UnifiedGroupedConvCodegen(output_dir=output_dir, gpu_target=arch, enable_arch_filter=False)
        codegen.generate_all(configs, datatypes=[datatype])
        return True
    except Exception as e:
        print(f"ERROR generating from {config_file}: {e}", file=sys.stderr)
        return False


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
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--codegen", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--config-set", default="tests", choices=["tests", "profiler"])

    args = parser.parse_args()
    cfg = VARIANT_CONFIG[args.variant]

    config_dir = Path(args.config_dir) / args.config_set
    codegen_path = Path(args.codegen)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add the codegen directory to sys.path so unified_grouped_conv_codegen
    # and its siblings are importable regardless of working directory.
    codegen_dir = str(codegen_path.parent.resolve())
    if codegen_dir not in sys.path:
        sys.path.insert(0, codegen_dir)

    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    json_configs = sorted(config_dir.glob("*.json"))
    if not json_configs:
        print(f"ERROR: No JSON config files in {config_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_configs)} config files in {config_dir}")

    success = True
    for config_file in json_configs:
        print(f"Generating from {config_file.name}...")
        if not generate_kernels_from_config(config_file, output_dir, args.arch):
            success = False

    if not success:
        print("ERROR: Some kernel generations failed", file=sys.stderr)
        sys.exit(1)

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
