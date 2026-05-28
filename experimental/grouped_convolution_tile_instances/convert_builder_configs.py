#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Converts CK Builder .conf config files to JSON format for the CK Dispatcher codegen.
#
# CK Builder instances are parameterized with Seq() thread block cluster lengths
# and k0/k1 decompositions that control thread-to-data mappings at a level of
# detail the dispatcher codegen does not model.  Multiple Builder instances that
# differ only in these parameters produce identical dispatcher configurations
# (same tile/warp/vector sizes, pipeline, scheduler, specialization).  The
# converter therefore deduplicates the output so each unique dispatcher config
# appears exactly once in the JSON.
#
# Two categories of Builder instances are skipped because of hardware or
# architecture limitations (the Builder's generate_instances.py also skips them):
#   1. Irregular vector sizes (odd values other than 1) — AMD GPUs only have
#      vector load instructions for widths 1, 2, 4, 8, 16
#   2. Multi-warp per continuous tile dimension
#      (tile_m > warp_size * vec_a, or tile_n > warp_size * vec_b) — the
#      codegen assumes single-warp coverage per tile dimension for data loading
#
# Usage example:
#   python3 convert_builder_configs.py convert \
#     --input configs/backward_weight/profiler/nhwgc_bf16.conf \
#     --output ../../dispatcher/codegen/configs/grouped_conv/backward_weight/profiler/nhwgc_bf16.json \
#     --variant bwd_weight --layout nhwgc --datatype bf16 --ndim 2
#
# Or convert all configs at once:
#   python3 convert_builder_configs.py convert-all

import argparse
import json
import sys
from pathlib import Path
from enum import Enum

# generate_instances.py lives is the authoritative source for parsing CK Builder .conf files.
# Import from it directly such that this converter doesn't duplicates the logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_instances import (  # noqa: E402
    ConvInstanceTemplateParams,
    DEPTHWISE_CONFIGS,
    fwd_configs,
    bwd_data_configs,
    bwd_weight_configs,
    parse_fwd_instances,
    parse_depthwise_config,
    parse_bwd_weight_instances,
    parse_bwd_data_instances,
)


def map_pipeline_version(version_str):
    """Map CK Builder pipeline version to dispatcher pipeline string."""
    mapping = {
        "V1": "compv1",
        "V2": "mem",
        "V3": "compv3",
        "V4": "compv4",
        "V5": "compv5",
        "V6": "compv6",
        "ASYNC_V1": "basic_async_v1",
        "ASYNC_V4": "mem",
    }
    return mapping.get(version_str, version_str.lower())


def map_scheduler(scheduler_str):
    """Map CK Builder scheduler to dispatcher scheduler string."""
    if "Intrawave" in scheduler_str:
        return "intrawave"
    elif "Interwave" in scheduler_str:
        return "interwave"
    return scheduler_str.lower()


def map_specialization(spec_str):
    """Map CK Builder specialization to dispatcher specialization string."""
    mapping = {
        "Default": "default",
        "OddC": "default",
        "Filter1x1Pad0": "filter1x1_pad0",
        "Filter1x1Stride1Pad0": "filter1x1_stride1_pad0",
        "Filter3x3": "filter3x3",
    }
    return mapping.get(spec_str, spec_str.lower())

def conv_depthwise_params_to_dict(p: list, index: int) -> dict:
    if len(p) != 12:
        raise ValueError(f"Expected 12 parameters for depthwise conv, got {len(p)}: {p}")
    return {
        "id": index,
        "tile_h":   p[0],
        "tile_w":   p[1],
        "filt":     p[2],
        "str_h":    p[3],
        "str_w":    p[4],
        "pad_h":    p[5],
        "pad_w":    p[6],
        "nbatch":   p[7],
        "sub_h":    p[8],
        "sub_w":    p[9],
        "in_vec":   p[10],
        "out_vec":  p[11],
    }

def conv_params_to_dict(p: ConvInstanceTemplateParams) -> dict:
    """Convert a ConvInstanceTemplateParams (CK Builder) to a Dispatcher JSON dict."""
    return {
        "id":                           p.id,
        "tile_m":                       p.tile_size[0],
        "tile_n":                       p.tile_size[1],
        "tile_k":                       p.tile_size[2],
        "warp_m":                       p.warps[0],
        "warp_n":                       p.warps[1],
        "warp_k":                       p.warps[2],
        "warp_tile_m":                  p.warp_tile[0],
        "warp_tile_n":                  p.warp_tile[1],
        "warp_tile_k":                  p.warp_tile[2],
        "vector_size_a":                p.scalar_per_vector[0],
        "vector_size_b":                p.scalar_per_vector[1],
        "vector_size_c":                p.scalar_per_vector[2],
        "pipeline":                     map_pipeline_version(p.pipeline_version),
        "scheduler":                    map_scheduler(p.scheduler),
        "epilogue":                     "cshuffle",
        "double_smem_buffer":           p.double_smem_buffer,
        "num_groups_to_merge":          p.num_groups_to_merge,
        "num_wave_groups":              p.num_wave_groups,
        "specialization":               map_specialization(p.specialization),
        "two_stage":                    p.is_two_stage_instance,
        "explicit_gemm":                p.explicit_gemm,
        "split_image":                  p.split_image,
        "streamk_enabled":              p.streamk_enabled,
        "streamk_reduction_strategy":   p.streamk_reduction_strategy,
        "streamk_persistent":           p.streamk_persistent,
    }


def convert_config_file(input_path, variant, layout, datatype, ndim, specialization) -> dict:
    """Convert a single CK Builder .conf file to JSON format."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    # problem_name is used only for dtype detection (fp32/fp16/bf16 substring match)
    problem_name = f"grouped_convolution_{variant}_tile_{layout}_{datatype}"

    if variant == "bwd_weight":
        raw = parse_bwd_weight_instances(lines, problem_name)
    elif variant == "forward" and specialization == Specialization.Default:
        raw = parse_fwd_instances(lines, problem_name)
    elif variant == "forward" and specialization == Specialization.Depthwise:
        raw = parse_depthwise_config(input_path)
    elif variant == "bwd_data":
        raw = parse_bwd_data_instances(lines, problem_name)
    else:
        raise RuntimeError(f"Variant '{variant}' with specialization '{specialization}' is not yet implemented.")

    instances = [
        conv_params_to_dict(p) if isinstance(p, ConvInstanceTemplateParams) else conv_depthwise_params_to_dict(p, i)
        for (i, p) in enumerate(raw)
    ]

    # Deduplicate: Builder instances that differ only in Seq() thread block
    # cluster lengths or k0/k1 decomposition produce identical dispatcher
    # configs because the converter discards these parameters.
    seen = set()
    unique_instances = []
    for inst in instances:
        key = tuple(sorted((k, str(v)) for k, v in inst.items() if k != "id"))
        if key not in seen:
            seen.add(key)
            unique_instances.append(inst)
    if len(unique_instances) < len(instances):
        print(f"  Deduplicated: {len(instances)} -> {len(unique_instances)} "
              f"({len(instances) - len(unique_instances)} duplicates removed)")
    instances = unique_instances

    output_variant = "forward_depthwise" if variant == "forward" and specialization == Specialization.Depthwise else variant
    output = {
        "variant": output_variant,
        "ndim_spatial": ndim,
        "layout": layout,
        "datatype": datatype,
        "instances": instances,
    }

    print(f"Converted {len(instances)} instances from {input_path}")
    return output

class Specialization(Enum):
    Default = "Default"
    StreamK = "Streamk"
    Depthwise = "Depthwise"

def parse_config(config: str, variant: str) -> tuple[str, str, str, str, str, int, Specialization]:
    """Parse a config name like 'nhwgc_bf16' into its components."""
    parts = config.split("_")
    if len(parts) > 3:
        raise ValueError(f"Unsupported config: {config}")
    layout = parts[0]
    datatype = parts[1]

    specialization = Specialization.Default

    if datatype == "depthwise":
        datatype = parts[2]
        specialization = Specialization.Depthwise

    if len(parts) == 3 and parts[2] == "streamk":
        specialization = Specialization.StreamK

    if layout not in ["nhwgc", "ndhwgc"]:
        raise ValueError(f"Invalid layout: {layout}")
    
    if datatype not in ["fp32", "fp16", "bf16"]:
        raise ValueError(f"Invalid datatype: {datatype}")
    
    ndim = 2 if layout == "nhwgc" else 3

    source_cfg = config
    target_cfg = config
    if specialization == Specialization.StreamK:
        target_cfg = config.replace("_streamk", "")
    elif specialization == Specialization.Depthwise:
        target_cfg = config.replace("_depthwise", "")

    return variant, source_cfg, target_cfg, layout, datatype, ndim, specialization

def parse_config_depthwise() -> list[tuple[str, str, str, str, str, int, Specialization]]:
    configs = []
    for config in DEPTHWISE_CONFIGS:
        parts = config["name"].split("_")
        if len(parts) != 3:
            raise ValueError(f"Unsupported depthwise config: {config}")
        layout = parts[0]
        datatype = parts[2]
        specialization = Specialization.Depthwise

        if layout not in ["ngchw", "ngcdhw"]:
            raise ValueError(f"Invalid layout: {layout}")
        
        if datatype not in ["fp32", "fp16", "bf16"]:
            raise ValueError(f"Invalid datatype: {datatype}")
        
        ndim = 2 if layout in ("nhwgc", "ngchw") else 3

        source_cfg = config["conf"].replace(".conf", "")
        target_cfg = f"{layout}_{datatype}"

        configs.append(("forward", source_cfg, target_cfg, layout, datatype, ndim, specialization))

    return configs

def get_configs() -> list[tuple[str, str, str, str, str, int, Specialization]]:
    # Get the configs from the parent generator script
    config_fwd_depthwise = parse_config_depthwise()
    config_fwd = [parse_config(cfg, "forward") for cfg in fwd_configs]
    config_bwd_weight = [parse_config(cfg, "bwd_weight") for cfg in bwd_weight_configs]
    config_bwd_data = [parse_config(cfg, "bwd_data") for cfg in bwd_data_configs]
    configs = config_fwd_depthwise + config_fwd + config_bwd_weight + config_bwd_data
    return configs

def write_or_append_json(output_path, result):
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        start_index = max((inst["id"] for inst in existing["instances"]), default=-1) + 1

        # Increase the id of new instances to avoid duplicates with existing ones
        for inst in result["instances"]:
            inst["id"] += start_index

        existing["instances"].extend(result["instances"])
        result = existing

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

def convert_all(builder_configs_dir, output_dir):
    """Convert all config files for all variants."""
    builder_dir = Path(builder_configs_dir)
    out_dir = Path(output_dir)
    configs = get_configs()

    # Directory, variant name pairs
    variants = [
        ("backward_weight", "bwd_weight"),
        ("forward", "forward"),
        ("backward_data", "bwd_data"),
    ]

    # Clean-up precvious config files as they will be appended to if they exist
    for variant_dir, _ in variants:
        for prefix in ["tests", "profiler"]:
            output_path = out_dir / variant_dir / prefix
            if output_path.exists():
                for file in output_path.glob("*.json"):
                    file.unlink()

    for variant_dir, variant_name in variants:
        for prefix in ["tests", "profiler"]:
            for var_name, source_config_name, target_config_name, layout, datatype, ndim, specialization in configs:

                if var_name == variant_name:
                    input_path = builder_dir / variant_dir / prefix / f"{source_config_name}.conf"
                    if not input_path.exists():
                        print(f"Skipping {input_path} (not found)")
                        continue

                    output_path = out_dir / variant_dir / prefix / f"{target_config_name}.json"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    print(f"Variant: {variant_name}, Config: {source_config_name}, Specialization: {specialization.value}")
                    result = convert_config_file(input_path, variant_name, layout, datatype, ndim, specialization)
                    write_or_append_json(output_path, result)
                    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CK Builder .conf config files to JSON for CK Dispatcher codegen."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Single file conversion
    single = subparsers.add_parser("convert", help="Convert a single config file")
    single.add_argument("--input", required=True, help="Input .conf file path")
    single.add_argument("--output", required=True, help="Output .json file path")
    single.add_argument("--variant", required=True, choices=["bwd_weight", "forward", "bwd_data"])
    single.add_argument("--layout", required=True, choices=["nhwgc", "ndhwgc"])
    single.add_argument("--datatype", required=True, choices=["fp32", "fp16", "bf16"])
    single.add_argument("--ndim", required=True, type=int, choices=[2, 3])

    # Batch conversion
    batch = subparsers.add_parser("convert-all", help="Convert all configs")
    batch.add_argument(
        "--builder-configs-dir",
        default=str(Path(__file__).resolve().parent / "configs"),
        help="Path to CK Builder configs directory",
    )
    batch.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "dispatcher/codegen/configs/grouped_conv"),
        help="Output directory for JSON configs",
    )

    args = parser.parse_args()

    if args.command == "convert":
        result = convert_config_file(args.input, args.variant, args.layout, args.datatype, args.ndim, Specialization.Default)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {output_path}")
    elif args.command == "convert-all":
        convert_all(args.builder_configs_dir, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
