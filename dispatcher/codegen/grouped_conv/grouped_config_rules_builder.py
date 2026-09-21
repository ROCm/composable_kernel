#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Builder-derived rule sets ("profiler" and "tests") for Grouped Convolution Tile
Configurations.

Unlike the rule-derived sets, these configs are produced directly from the CK
Builder ``.conf`` configurations in
``experimental/grouped_convolution_tile_instances/configs/<variant>/<subset>/``.

CK Builder instances are parameterized with Seq() thread block cluster lengths
and k0/k1 decompositions that control thread-to-data mappings at a level of
detail the dispatcher codegen does not model. Multiple Builder instances that
differ only in these parameters produce identical dispatcher configurations
(same tile/warp/vector sizes, pipeline, scheduler, specialization). The
conversion therefore deduplicates so each unique dispatcher config appears
exactly once.

Selected via ``get_default_configs(rule_set="profiler")`` (profiler subset) or
``get_default_configs(rule_set="tests")`` (tests subset).
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Path setup — allow importing sibling codegen modules and the CK Builder's
# authoritative .conf parser (generate_instances.py).
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).resolve().parent.parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

_BUILDER_DIR = (
    _CODEGEN_DIR.parent.parent
    / "experimental"
    / "grouped_convolution_tile_instances"
)
if str(_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILDER_DIR))

# CK Builder .conf source directory.
_BUILDER_CONFIGS_DIR = _BUILDER_DIR / "configs"

from generate_instances import (
    ConvInstanceTemplateParams,
    DEPTHWISE_CONFIGS,
    fwd_configs,
    bwd_data_configs,
    bwd_weight_configs,
    get_warp_size,
    parse_fwd_instances,
    parse_depthwise_config,
    parse_bwd_weight_instances,
    parse_bwd_data_instances,
)

log = logging.getLogger(__name__)


# =============================================================================
# CK Builder field mapping helpers
# =============================================================================
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
        "WAVELET": "wavelet",
    }
    mapped = mapping.get(version_str)
    if mapped is None:
        log.warning(
            "Unknown pipeline version %r; falling back to %r",
            version_str, version_str.lower(),
        )
        return version_str.lower()
    return mapped


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


class Specialization(Enum):
    Default = "Default"
    StreamK = "Streamk"
    Depthwise = "Depthwise"


def _conv_depthwise_params_to_dict(p: list, index: int) -> dict:
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


def _conv_params_to_dict(p: ConvInstanceTemplateParams) -> dict:
    """Convert a ConvInstanceTemplateParams (CK Builder) to a dispatcher dict."""
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


def _build_data(input_path, variant, layout, datatype, ndim, specialization, warp_size, verbose=False) -> dict:
    """Parse a single CK Builder .conf file into an in-memory config dict.

    Equivalent to the old ``convert_config_file`` but returns the dict directly
    instead of writing JSON. The dict shape matches what the loaders below
    expect: ``{variant, ndim_spatial, layout, datatype, instances}``.

    ``warp_size`` must match the target architecture's warp size (64 for CDNA
    gfx9, 32 for RDNA). 
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # problem_name is used only for dtype detection (fp32/fp16/bf16 substring match)
    problem_name = f"grouped_convolution_{variant}_tile_{layout}_{datatype}"

    if variant == "bwd_weight":
        raw = parse_bwd_weight_instances(lines, problem_name, warp_size=warp_size, verbose=verbose)
    elif variant == "forward" and specialization == Specialization.Default:
        raw = parse_fwd_instances(lines, problem_name, warp_size=warp_size, verbose=verbose)
    elif variant == "forward" and specialization == Specialization.Depthwise:
        raw = parse_depthwise_config(input_path, verbose=verbose)
    elif variant == "bwd_data":
        raw = parse_bwd_data_instances(lines, problem_name, warp_size=warp_size, verbose=verbose)
    else:
        raise RuntimeError(
            f"Variant '{variant}' with specialization '{specialization}' is not yet implemented."
        )

    instances = [
        _conv_params_to_dict(p) if isinstance(p, ConvInstanceTemplateParams)
        else _conv_depthwise_params_to_dict(p, i)
        for (i, p) in enumerate(raw)
    ]

    # Deduplicate: Builder instances that differ only in Seq() thread block
    # cluster lengths or k0/k1 decomposition produce identical dispatcher
    # configs because the conversion discards these parameters.
    seen = set()
    unique_instances = []
    for inst in instances:
        key = tuple(sorted((k, str(v)) for k, v in inst.items() if k != "id"))
        if key not in seen:
            seen.add(key)
            unique_instances.append(inst)
    if len(unique_instances) < len(instances):
        log.debug(
            f"Deduplicated: {len(instances)} -> {len(unique_instances)} "
            f"({len(instances) - len(unique_instances)} duplicates removed)"
        )
    instances = unique_instances

    output_variant = (
        "forward_depthwise"
        if variant == "forward" and specialization == Specialization.Depthwise
        else variant
    )
    log.debug(f"Parsed {len(instances)} instances from {input_path}")
    return {
        "variant": output_variant,
        "ndim_spatial": ndim,
        "layout": layout,
        "datatype": datatype,
        "instances": instances,
    }


# =============================================================================
# Config-name parsing (which .conf files exist, and how to interpret them)
# =============================================================================
def _parse_config(config: str, variant: str):
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
    elif len(parts) == 3 and parts[2] == "streamk":
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


def _parse_config_depthwise():
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


def _builder_config_list():
    """List every (variant, source_cfg, target_cfg, layout, datatype, ndim,
    specialization) tuple known to the CK Builder."""
    config_fwd_depthwise = _parse_config_depthwise()
    config_fwd = [_parse_config(cfg, "forward") for cfg in fwd_configs]
    config_bwd_weight = [_parse_config(cfg, "bwd_weight") for cfg in bwd_weight_configs]
    config_bwd_data = [_parse_config(cfg, "bwd_data") for cfg in bwd_data_configs]
    return config_fwd_depthwise + config_fwd + config_bwd_weight + config_bwd_data


# =============================================================================
# In-memory loaders: config dict -> dispatcher config objects
# =============================================================================
def _load_depthwise_configs(data: dict, arch: str) -> List:
    """Build DepthwiseConvKernelConfig objects from an in-memory config dict."""
    from unified_grouped_conv_codegen import DepthwiseConvKernelConfig

    ndim_spatial = data["ndim_spatial"]
    layout = data["layout"]
    datatype = data["datatype"]

    configs = []
    for inst in data["instances"]:
        configs.append(DepthwiseConvKernelConfig(
            tile_h=inst["tile_h"],
            tile_w=inst["tile_w"],
            filt=inst["filt"],
            str_h=inst["str_h"],
            str_w=inst["str_w"],
            pad_h=inst["pad_h"],
            pad_w=inst["pad_w"],
            nbatch=inst["nbatch"],
            sub_h=inst["sub_h"],
            sub_w=inst["sub_w"],
            in_vec=inst["in_vec"],
            out_vec=inst["out_vec"],
            ndim_spatial=ndim_spatial,
            arch=arch,
            layout=layout,
            datatype=datatype,
        ))

    log.debug(f"Loaded {len(configs)} depthwise configs (layout={layout}, dtype={datatype})")
    return configs


def _load_gemm_configs(data: dict, arch: str) -> List:
    """Build GroupedConvKernelConfig objects from an in-memory config dict."""
    from unified_grouped_conv_codegen import (
        GroupedConvVariant,
        GroupedConvTraitConfig,
        GroupedConvKernelConfig,
        TileConfig,
        StreamKConfig,
        StreamKReductionStrategy,
    )

    variant_map = {
        "forward": GroupedConvVariant.FORWARD,
        "fwd": GroupedConvVariant.FORWARD,
        "forward_depthwise": GroupedConvVariant.FORWARD_DEPTHWISE,
        "bwd_data": GroupedConvVariant.BACKWARD_DATA,
        "bwd_weight": GroupedConvVariant.BACKWARD_WEIGHT,
    }
    variant = variant_map.get(data["variant"])
    if variant is None:
        raise ValueError(f"Unknown variant: {data['variant']}")

    ndim_spatial = data["ndim_spatial"]
    layout = data["layout"]
    datatype = data["datatype"]

    configs = []
    for inst in data["instances"]:
        trait = GroupedConvTraitConfig(
            pipeline=inst["pipeline"],
            scheduler=inst["scheduler"],
            epilogue=inst["epilogue"],
            pad_m=True,
            pad_n=True,
            pad_k=True,
            double_smem_buffer=inst.get("double_smem_buffer", False),
            num_groups_to_merge=inst.get("num_groups_to_merge", 1),
            split_image=inst.get("split_image", False),
            explicit_gemm=inst.get("explicit_gemm", False),
            two_stage=inst.get("two_stage", False),
            specialization=inst.get("specialization", "default"),
            streamk_config=StreamKConfig(
                streamk_enabled=inst.get("streamk_enabled", False),
                strategy=StreamKReductionStrategy(inst.get("streamk_reduction_strategy", "TREE")),
                streamk_persistent=inst.get("streamk_persistent", False)
            ) if inst.get("streamk_enabled", False) else StreamKConfig()
        )

        # compv2/basic_v2 (GemmPipelineAGmemBGmemCRegV2) is not compatible with
        # CK Tile's GroupedConvolutionBackwardWeightKernel. The builder maps
        # PipelineVersion::V2 to GemmPipelineAgBgCrMem (i.e. "mem"), not to
        # GemmPipelineAGmemBGmemCRegV2. Skip if any config somehow has compv2.
        if variant == GroupedConvVariant.BACKWARD_WEIGHT and trait.pipeline in ("compv2", "basic_v2"):
            log.info(f"Skipping instance {inst['id']}: compv2/basic_v2 pipeline not compatible with CK Tile bwd_weight")
            continue

        config = GroupedConvKernelConfig(
            tile=TileConfig(
                tile_m=inst["tile_m"],
                tile_n=inst["tile_n"],
                tile_k=inst["tile_k"],
                warp_m=inst["warp_m"],
                warp_n=inst["warp_n"],
                warp_k=inst["warp_k"],
                warp_tile_m=inst["warp_tile_m"],
                warp_tile_n=inst["warp_tile_n"],
                warp_tile_k=inst["warp_tile_k"],
            ),
            trait=trait,
            variant=variant,
            ndim_spatial=ndim_spatial,
            arch=arch,
            layout=layout,
            vector_size_a=inst["vector_size_a"],
            vector_size_b=inst["vector_size_b"],
            vector_size_c=inst["vector_size_c"],
            num_wave_groups=inst.get("num_wave_groups", 1),
        )
        # Tag each config with its concrete datatype so that generate_all emits
        # the kernel only for that datatype (an untagged config is compiled for
        # every datatype).
        config.datatype = datatype
        configs.append(config)

    log.debug(
        f"Loaded {len(configs)} configs (variant={data['variant']}, layout={layout}, dtype={datatype})"
    )
    return configs


# =============================================================================
# Unified rule-set entry point
# =============================================================================
def get_configs(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str],
    subset: str = "profiler",
    verbose: bool = False,
) -> List:
    """Build all configs for a builder-derived rule set by parsing the CK
    Builder ``.conf`` files in memory.

    ``subset`` selects the on-disk config subset (``"profiler"`` or ``"tests"``).
    Each requested (variant, ndim, datatype) is filtered against the Builder's
    config list, the matching ``.conf`` file is parsed and converted into
    dispatcher config objects, with no intermediate JSON written.

    ``verbose`` controls whether the underlying CK Builder parsers print their
    "Skipping instance ..." diagnostics. It defaults to ``False`` so the
    dispatcher rule set stays quiet; the standalone CK Builder script keeps
    printing them (its own default is ``True``).
    """
    from unified_grouped_conv_codegen import GroupedConvVariant

    # Builder variant name -> on-disk variant directory name.
    variant_dir_map = {
        "forward": "forward",
        "bwd_weight": "backward_weight",
        "bwd_data": "backward_data",
    }

    def to_enum(variant_name, specialization):
        if variant_name == "forward":
            return (GroupedConvVariant.FORWARD_DEPTHWISE
                    if specialization == Specialization.Depthwise
                    else GroupedConvVariant.FORWARD)
        if variant_name == "bwd_weight":
            return GroupedConvVariant.BACKWARD_WEIGHT
        if variant_name == "bwd_data":
            return GroupedConvVariant.BACKWARD_DATA
        return None

    want_variants = set(variants)
    want_ndims = set(ndims) if ndims else None
    want_dtypes = set(datatypes) if datatypes else None

    warp_size = get_warp_size(arch)

    configs: List = []
    for (variant_name, source_cfg, target_cfg, layout, datatype, ndim, spec) in _builder_config_list():
        if to_enum(variant_name, spec) not in want_variants:
            continue
        if want_ndims is not None and ndim not in want_ndims:
            continue
        if want_dtypes is not None and datatype not in want_dtypes:
            continue

        input_path = _BUILDER_CONFIGS_DIR / variant_dir_map[variant_name] / subset / f"{source_cfg}.conf"
        if not input_path.exists():
            log.warning(f"Builder config not found: {input_path}")
            continue

        data = _build_data(input_path, variant_name, layout, datatype, ndim, spec, warp_size, verbose=verbose)
        if data["variant"] == "forward_depthwise":
            configs.extend(_load_depthwise_configs(data, arch))
        else:
            configs.extend(_load_gemm_configs(data, arch))

    log.info(f"builder rule set ({subset}): generated {len(configs)} configs")
    return configs

def get_configs_profiler(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str],
) -> List:
    return get_configs(arch, variants, ndims, datatypes, subset="profiler")

def get_configs_tests(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str],
) -> List:
    return get_configs(arch, variants, ndims, datatypes, subset="tests")