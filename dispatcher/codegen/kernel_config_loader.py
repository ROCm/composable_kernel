#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Kernel Configuration Loader

Load kernel configurations from JSON files for generating specific kernel sets.
Compatible with tile_engine JSON format.

Usage:
    from kernel_config_loader import load_kernel_configs, KernelConfigSet

    # Load configs from JSON
    config_set = load_kernel_configs("my_kernels.json")

    # Get all configurations (cartesian product of all parameter values)
    for config in config_set.generate_configs():
        print(config)

    # Use with codegen
    from unified_gemm_codegen import UnifiedGemmCodegen
    codegen = UnifiedGemmCodegen(...)
    codegen.generate_from_configs(config_set.generate_configs())
"""

import json
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator


@dataclass
class TileConfig:
    """Tile configuration for a kernel"""

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32
    warp_m: int = 2
    warp_n: int = 2
    warp_k: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16


@dataclass
class TraitConfig:
    """Trait configuration for a kernel (order matches GEMM/Conv TraitConfig)"""

    pipeline: str = "compv4"
    epilogue: str = "cshuffle"
    scheduler: str = "intrawave"
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False


@dataclass
class KernelConfig:
    """Complete kernel configuration"""

    tile: TileConfig = field(default_factory=TileConfig)
    trait: TraitConfig = field(default_factory=TraitConfig)
    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"
    layout: str = "rcr"
    gpu_target: str = "gfx942"
    variant: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for codegen"""
        return {
            "tile_m": self.tile.tile_m,
            "tile_n": self.tile.tile_n,
            "tile_k": self.tile.tile_k,
            "warp_m": self.tile.warp_m,
            "warp_n": self.tile.warp_n,
            "warp_k": self.tile.warp_k,
            "warp_tile_m": self.tile.warp_tile_m,
            "warp_tile_n": self.tile.warp_tile_n,
            "warp_tile_k": self.tile.warp_tile_k,
            "pipeline": self.trait.pipeline,
            "scheduler": self.trait.scheduler,
            "epilogue": self.trait.epilogue,
            "pad_m": self.trait.pad_m,
            "pad_n": self.trait.pad_n,
            "pad_k": self.trait.pad_k,
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "dtype_c": self.dtype_c,
            "dtype_acc": self.dtype_acc,
            "layout": self.layout,
            "gpu_target": self.gpu_target,
            "variant": self.variant,
        }

    def kernel_name(self) -> str:
        """Generate kernel name from config"""
        name = f"gemm_{self.dtype_a}_{self.layout}_{self.trait.pipeline}"
        name += f"_{self.trait.epilogue}_{self.trait.scheduler}"
        name += f"_{str(self.trait.pad_m).capitalize()}"
        name += f"_{str(self.trait.pad_n).capitalize()}"
        name += f"_{str(self.trait.pad_k).capitalize()}"
        name += "_False"  # preshuffle
        name += f"_{self.tile.tile_m}x{self.tile.tile_n}x{self.tile.tile_k}"
        name += f"_{self.tile.warp_m}x{self.tile.warp_n}x{self.tile.warp_k}"
        name += (
            f"_{self.tile.warp_tile_m}x{self.tile.warp_tile_n}x{self.tile.warp_tile_k}"
        )
        return name


@dataclass
class KernelConfigSet:
    """A set of kernel configurations loaded from JSON"""

    name: str = "default"
    configs: List[KernelConfig] = field(default_factory=list)

    # Parameter ranges for generation
    tile_m_values: List[int] = field(default_factory=lambda: [128])
    tile_n_values: List[int] = field(default_factory=lambda: [128])
    tile_k_values: List[int] = field(default_factory=lambda: [32])
    warp_m_values: List[int] = field(default_factory=lambda: [2])
    warp_n_values: List[int] = field(default_factory=lambda: [2])
    warp_k_values: List[int] = field(default_factory=lambda: [1])
    warp_tile_m_values: List[int] = field(default_factory=lambda: [32])
    warp_tile_n_values: List[int] = field(default_factory=lambda: [32])
    warp_tile_k_values: List[int] = field(default_factory=lambda: [16])

    pipeline_values: List[str] = field(default_factory=lambda: ["compv4"])
    scheduler_values: List[str] = field(default_factory=lambda: ["intrawave"])
    epilogue_values: List[str] = field(default_factory=lambda: ["cshuffle"])
    pad_m_values: List[bool] = field(default_factory=lambda: [False])
    pad_n_values: List[bool] = field(default_factory=lambda: [False])
    pad_k_values: List[bool] = field(default_factory=lambda: [False])

    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"
    layout: str = "rcr"
    gpu_targets: List[str] = field(default_factory=lambda: ["gfx942"])
    variant: str = "standard"

    def generate_configs(self) -> Iterator[KernelConfig]:
        """Generate all kernel configurations (cartesian product)"""
        # Tile parameters
        tile_params = itertools.product(
            self.tile_m_values,
            self.tile_n_values,
            self.tile_k_values,
            self.warp_m_values,
            self.warp_n_values,
            self.warp_k_values,
            self.warp_tile_m_values,
            self.warp_tile_n_values,
            self.warp_tile_k_values,
        )

        # Trait parameters
        trait_params = itertools.product(
            self.pipeline_values,
            self.scheduler_values,
            self.epilogue_values,
            self.pad_m_values,
            self.pad_n_values,
            self.pad_k_values,
        )

        # Convert to lists for reuse
        tile_list = list(tile_params)
        trait_list = list(trait_params)

        # Generate for each GPU target
        for gpu_target in self.gpu_targets:
            for tile in tile_list:
                for trait in trait_list:
                    tile_cfg = TileConfig(
                        tile_m=tile[0],
                        tile_n=tile[1],
                        tile_k=tile[2],
                        warp_m=tile[3],
                        warp_n=tile[4],
                        warp_k=tile[5],
                        warp_tile_m=tile[6],
                        warp_tile_n=tile[7],
                        warp_tile_k=tile[8],
                    )
                    trait_cfg = TraitConfig(
                        pipeline=trait[0],
                        scheduler=trait[1],
                        epilogue=trait[2],
                        pad_m=trait[3],
                        pad_n=trait[4],
                        pad_k=trait[5],
                    )
                    yield KernelConfig(
                        tile=tile_cfg,
                        trait=trait_cfg,
                        dtype_a=self.dtype_a,
                        dtype_b=self.dtype_b,
                        dtype_c=self.dtype_c,
                        dtype_acc=self.dtype_acc,
                        layout=self.layout,
                        gpu_target=gpu_target,
                        variant=self.variant,
                    )

    def config_count(self) -> int:
        """Get total number of configurations"""
        tile_count = (
            len(self.tile_m_values)
            * len(self.tile_n_values)
            * len(self.tile_k_values)
            * len(self.warp_m_values)
            * len(self.warp_n_values)
            * len(self.warp_k_values)
            * len(self.warp_tile_m_values)
            * len(self.warp_tile_n_values)
            * len(self.warp_tile_k_values)
        )
        trait_count = (
            len(self.pipeline_values)
            * len(self.scheduler_values)
            * len(self.epilogue_values)
            * len(self.pad_m_values)
            * len(self.pad_n_values)
            * len(self.pad_k_values)
        )
        return tile_count * trait_count * len(self.gpu_targets)


def _get_values(config: Dict, key: str, default: List) -> List:
    """Extract values from config dict, handling range specifications"""
    if key not in config:
        return default

    item = config[key]

    # Explicit values list
    if "values" in item:
        return item["values"]

    # Range specification (min, max, step)
    if "min" in item and "max" in item:
        min_val = item["min"]
        max_val = item["max"]
        step = item.get("step", 1)
        return list(range(min_val, max_val + 1, step))

    return default


def load_kernel_configs(json_path: str | Path) -> KernelConfigSet:
    """
    Load kernel configurations from a JSON file.

    Supports both tile_engine format and dispatcher format.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        KernelConfigSet with all parameter values loaded
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    config_set = KernelConfigSet()

    # Name
    config_set.name = data.get("kernel_set_name", json_path.stem)

    # Data types
    if "datatype" in data:
        dt = data["datatype"]
        config_set.dtype_a = dt.get("a", "fp16")
        config_set.dtype_b = dt.get("b", "fp16")
        config_set.dtype_c = dt.get("c", "fp16")
        config_set.dtype_acc = dt.get("acc", "fp32")

    # Layout
    config_set.layout = data.get("layout", "rcr")

    # GPU targets
    if "gpu_targets" in data:
        config_set.gpu_targets = data["gpu_targets"]
    elif "gpu_target" in data:
        config_set.gpu_targets = [data["gpu_target"]]

    # Variant
    config_set.variant = data.get("variant", "standard")

    # Tile config
    tile_cfg = data.get("tile_config", {})
    config_set.tile_m_values = _get_values(tile_cfg, "tile_m", [128])
    config_set.tile_n_values = _get_values(tile_cfg, "tile_n", [128])
    config_set.tile_k_values = _get_values(tile_cfg, "tile_k", [32])
    config_set.warp_m_values = _get_values(tile_cfg, "warp_m", [2])
    config_set.warp_n_values = _get_values(tile_cfg, "warp_n", [2])
    config_set.warp_k_values = _get_values(tile_cfg, "warp_k", [1])
    config_set.warp_tile_m_values = _get_values(tile_cfg, "warp_tile_m", [32])
    config_set.warp_tile_n_values = _get_values(tile_cfg, "warp_tile_n", [32])
    config_set.warp_tile_k_values = _get_values(tile_cfg, "warp_tile_k", [16])

    # Trait config
    trait_cfg = data.get("trait_config", {})
    config_set.pipeline_values = _get_values(trait_cfg, "pipeline", ["compv4"])
    config_set.scheduler_values = _get_values(trait_cfg, "scheduler", ["intrawave"])
    config_set.epilogue_values = _get_values(trait_cfg, "epilogue", ["cshuffle"])
    config_set.pad_m_values = _get_values(trait_cfg, "pad_m", [False])
    config_set.pad_n_values = _get_values(trait_cfg, "pad_n", [False])
    config_set.pad_k_values = _get_values(trait_cfg, "pad_k", [False])

    return config_set


# =============================================================================
# Convolution Configuration Classes
# =============================================================================


@dataclass
class ConvTileConfig:
    """Tile configuration for a convolution kernel"""

    tile_m: int = 128  # M dimension (N * spatial_out for fwd)
    tile_n: int = 128  # N dimension (K output channels for fwd)
    tile_k: int = 32  # K dimension (C * filter for fwd)
    warp_m: int = 2
    warp_n: int = 2
    warp_k: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16


@dataclass
class ConvTraitConfig:
    """Trait configuration for a convolution kernel"""

    pipeline: str = "compv3"
    scheduler: str = "intrawave"
    epilogue: str = "cshuffle"
    pad_m: bool = True
    pad_n: bool = True
    pad_k: bool = True
    double_smem_buffer: bool = False
    num_groups_to_merge: int = 1


@dataclass
class GroupedConvKernelConfig:
    """Complete grouped convolution kernel configuration"""

    tile: ConvTileConfig = field(default_factory=ConvTileConfig)
    trait: ConvTraitConfig = field(default_factory=ConvTraitConfig)
    dtype_input: str = "fp16"
    dtype_weight: str = "fp16"
    dtype_output: str = "fp16"
    dtype_acc: str = "fp32"
    variant: str = "forward"  # forward, bwd_data, bwd_weight
    ndim: int = 2  # 1, 2, or 3
    layout: str = "nhwgc"
    gpu_target: str = "gfx942"

    # Vector sizes
    vector_size_a: int = 4
    vector_size_b: int = 8
    vector_size_c: int = 8

    # Occupancy
    block_per_cu: int = 1
    num_wave_groups: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for codegen"""
        return {
            "tile_m": self.tile.tile_m,
            "tile_n": self.tile.tile_n,
            "tile_k": self.tile.tile_k,
            "warp_m": self.tile.warp_m,
            "warp_n": self.tile.warp_n,
            "warp_k": self.tile.warp_k,
            "warp_tile_m": self.tile.warp_tile_m,
            "warp_tile_n": self.tile.warp_tile_n,
            "warp_tile_k": self.tile.warp_tile_k,
            "pipeline": self.trait.pipeline,
            "scheduler": self.trait.scheduler,
            "epilogue": self.trait.epilogue,
            "pad_m": self.trait.pad_m,
            "pad_n": self.trait.pad_n,
            "pad_k": self.trait.pad_k,
            "double_smem_buffer": self.trait.double_smem_buffer,
            "num_groups_to_merge": self.trait.num_groups_to_merge,
            "dtype_input": self.dtype_input,
            "dtype_weight": self.dtype_weight,
            "dtype_output": self.dtype_output,
            "dtype_acc": self.dtype_acc,
            "variant": self.variant,
            "ndim": self.ndim,
            "layout": self.layout,
            "gpu_target": self.gpu_target,
            "vector_size_a": self.vector_size_a,
            "vector_size_b": self.vector_size_b,
            "vector_size_c": self.vector_size_c,
            "block_per_cu": self.block_per_cu,
            "num_wave_groups": self.num_wave_groups,
        }

    def kernel_name(self) -> str:
        """Generate kernel name from config"""
        variant_map = {
            "forward": "fwd",
            "bwd_data": "bwd_data",
            "bwd_weight": "bwd_weight",
        }
        var_str = variant_map.get(self.variant, self.variant)

        name = f"conv_{var_str}_{self.dtype_input}_{self.ndim}d"
        name += f"_{self.trait.pipeline}_{self.trait.epilogue}_{self.trait.scheduler}"
        name += f"_{self.tile.tile_m}x{self.tile.tile_n}x{self.tile.tile_k}"
        name += f"_{self.tile.warp_m}x{self.tile.warp_n}x{self.tile.warp_k}"
        name += (
            f"_{self.tile.warp_tile_m}x{self.tile.warp_tile_n}x{self.tile.warp_tile_k}"
        )
        return name


@dataclass
class GroupedConvKernelConfigSet:
    """A set of convolution kernel configurations loaded from JSON"""

    name: str = "default"
    configs: List[GroupedConvKernelConfig] = field(default_factory=list)

    # Tile parameter ranges
    tile_m_values: List[int] = field(default_factory=lambda: [128])
    tile_n_values: List[int] = field(default_factory=lambda: [128])
    tile_k_values: List[int] = field(default_factory=lambda: [32])
    warp_m_values: List[int] = field(default_factory=lambda: [2])
    warp_n_values: List[int] = field(default_factory=lambda: [2])
    warp_k_values: List[int] = field(default_factory=lambda: [1])
    warp_tile_m_values: List[int] = field(default_factory=lambda: [32])
    warp_tile_n_values: List[int] = field(default_factory=lambda: [32])
    warp_tile_k_values: List[int] = field(default_factory=lambda: [16])

    # Trait parameter ranges
    pipeline_values: List[str] = field(default_factory=lambda: ["compv3"])
    scheduler_values: List[str] = field(default_factory=lambda: ["intrawave"])
    epilogue_values: List[str] = field(default_factory=lambda: ["cshuffle"])
    pad_m_values: List[bool] = field(default_factory=lambda: [True])
    pad_n_values: List[bool] = field(default_factory=lambda: [True])
    pad_k_values: List[bool] = field(default_factory=lambda: [True])
    double_smem_buffer_values: List[bool] = field(default_factory=lambda: [False])
    num_groups_to_merge_values: List[int] = field(default_factory=lambda: [1])

    # Vector sizes
    vector_size_a_values: List[int] = field(default_factory=lambda: [4])
    vector_size_b_values: List[int] = field(default_factory=lambda: [8])
    vector_size_c_values: List[int] = field(default_factory=lambda: [8])

    # Occupancy
    block_per_cu_values: List[int] = field(default_factory=lambda: [1])
    num_wave_groups_values: List[int] = field(default_factory=lambda: [1])

    # Data types
    dtype_input: str = "fp16"
    dtype_weight: str = "fp16"
    dtype_output: str = "fp16"
    dtype_acc: str = "fp32"

    # Conv specific
    variant: str = "forward"
    ndim: int = 2
    layout: str = "nhwgc"
    gpu_targets: List[str] = field(default_factory=lambda: ["gfx942"])

    def generate_configs(self) -> Iterator[GroupedConvKernelConfig]:
        """Generate all kernel configurations (cartesian product)"""
        # Tile parameters
        tile_params = itertools.product(
            self.tile_m_values,
            self.tile_n_values,
            self.tile_k_values,
            self.warp_m_values,
            self.warp_n_values,
            self.warp_k_values,
            self.warp_tile_m_values,
            self.warp_tile_n_values,
            self.warp_tile_k_values,
        )

        # Trait parameters
        trait_params = itertools.product(
            self.pipeline_values,
            self.scheduler_values,
            self.epilogue_values,
            self.pad_m_values,
            self.pad_n_values,
            self.pad_k_values,
            self.double_smem_buffer_values,
            self.num_groups_to_merge_values,
        )

        # Vector/occupancy parameters
        extra_params = itertools.product(
            self.vector_size_a_values,
            self.vector_size_b_values,
            self.vector_size_c_values,
            self.block_per_cu_values,
            self.num_wave_groups_values,
        )

        # Convert to lists for reuse
        tile_list = list(tile_params)
        trait_list = list(trait_params)
        extra_list = list(extra_params)

        # Generate for each GPU target
        for gpu_target in self.gpu_targets:
            for tile in tile_list:
                for trait in trait_list:
                    for extra in extra_list:
                        tile_cfg = ConvTileConfig(
                            tile_m=tile[0],
                            tile_n=tile[1],
                            tile_k=tile[2],
                            warp_m=tile[3],
                            warp_n=tile[4],
                            warp_k=tile[5],
                            warp_tile_m=tile[6],
                            warp_tile_n=tile[7],
                            warp_tile_k=tile[8],
                        )
                        trait_cfg = ConvTraitConfig(
                            pipeline=trait[0],
                            scheduler=trait[1],
                            epilogue=trait[2],
                            pad_m=trait[3],
                            pad_n=trait[4],
                            pad_k=trait[5],
                            double_smem_buffer=trait[6],
                            num_groups_to_merge=trait[7],
                        )
                        yield GroupedConvKernelConfig(
                            tile=tile_cfg,
                            trait=trait_cfg,
                            dtype_input=self.dtype_input,
                            dtype_weight=self.dtype_weight,
                            dtype_output=self.dtype_output,
                            dtype_acc=self.dtype_acc,
                            variant=self.variant,
                            ndim=self.ndim,
                            layout=self.layout,
                            gpu_target=gpu_target,
                            vector_size_a=extra[0],
                            vector_size_b=extra[1],
                            vector_size_c=extra[2],
                            block_per_cu=extra[3],
                            num_wave_groups=extra[4],
                        )

    def config_count(self) -> int:
        """Get total number of configurations"""
        tile_count = (
            len(self.tile_m_values)
            * len(self.tile_n_values)
            * len(self.tile_k_values)
            * len(self.warp_m_values)
            * len(self.warp_n_values)
            * len(self.warp_k_values)
            * len(self.warp_tile_m_values)
            * len(self.warp_tile_n_values)
            * len(self.warp_tile_k_values)
        )
        trait_count = (
            len(self.pipeline_values)
            * len(self.scheduler_values)
            * len(self.epilogue_values)
            * len(self.pad_m_values)
            * len(self.pad_n_values)
            * len(self.pad_k_values)
            * len(self.double_smem_buffer_values)
            * len(self.num_groups_to_merge_values)
        )
        extra_count = (
            len(self.vector_size_a_values)
            * len(self.vector_size_b_values)
            * len(self.vector_size_c_values)
            * len(self.block_per_cu_values)
            * len(self.num_wave_groups_values)
        )
        return tile_count * trait_count * extra_count * len(self.gpu_targets)


def load_grouped_conv_kernel_configs(
    json_path: str | Path,
) -> GroupedConvKernelConfigSet:
    """
    Load convolution kernel configurations from a JSON file.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        GroupedConvKernelConfigSet with all parameter values loaded
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    config_set = GroupedConvKernelConfigSet()

    # Name
    config_set.name = data.get("kernel_set_name", json_path.stem)

    # Data types
    if "datatype" in data:
        dt = data["datatype"]
        config_set.dtype_input = dt.get("input", "fp16")
        config_set.dtype_weight = dt.get("weight", "fp16")
        config_set.dtype_output = dt.get("output", "fp16")
        config_set.dtype_acc = dt.get("acc", "fp32")

    # Conv specific
    config_set.variant = data.get("variant", "forward")
    config_set.ndim = data.get("ndim", 2)
    config_set.layout = data.get("layout", "nhwgc")

    # GPU targets
    if "gpu_targets" in data:
        config_set.gpu_targets = data["gpu_targets"]
    elif "gpu_target" in data:
        config_set.gpu_targets = [data["gpu_target"]]

    # Tile config
    tile_cfg = data.get("tile_config", {})
    config_set.tile_m_values = _get_values(tile_cfg, "tile_m", [128])
    config_set.tile_n_values = _get_values(tile_cfg, "tile_n", [128])
    config_set.tile_k_values = _get_values(tile_cfg, "tile_k", [32])
    config_set.warp_m_values = _get_values(tile_cfg, "warp_m", [2])
    config_set.warp_n_values = _get_values(tile_cfg, "warp_n", [2])
    config_set.warp_k_values = _get_values(tile_cfg, "warp_k", [1])
    config_set.warp_tile_m_values = _get_values(tile_cfg, "warp_tile_m", [32])
    config_set.warp_tile_n_values = _get_values(tile_cfg, "warp_tile_n", [32])
    config_set.warp_tile_k_values = _get_values(tile_cfg, "warp_tile_k", [16])

    # Trait config
    trait_cfg = data.get("trait_config", {})
    config_set.pipeline_values = _get_values(trait_cfg, "pipeline", ["compv3"])
    config_set.scheduler_values = _get_values(trait_cfg, "scheduler", ["intrawave"])
    config_set.epilogue_values = _get_values(trait_cfg, "epilogue", ["cshuffle"])
    config_set.pad_m_values = _get_values(trait_cfg, "pad_m", [True])
    config_set.pad_n_values = _get_values(trait_cfg, "pad_n", [True])
    config_set.pad_k_values = _get_values(trait_cfg, "pad_k", [True])
    config_set.double_smem_buffer_values = _get_values(
        trait_cfg, "double_smem_buffer", [False]
    )
    config_set.num_groups_to_merge_values = _get_values(
        trait_cfg, "num_groups_to_merge", [1]
    )

    # Vector config
    vec_cfg = data.get("vector_config", {})
    config_set.vector_size_a_values = _get_values(vec_cfg, "vector_size_a", [4])
    config_set.vector_size_b_values = _get_values(vec_cfg, "vector_size_b", [8])
    config_set.vector_size_c_values = _get_values(vec_cfg, "vector_size_c", [8])

    # Occupancy config
    occ_cfg = data.get("occupancy_config", {})
    config_set.block_per_cu_values = _get_values(occ_cfg, "block_per_cu", [1])
    config_set.num_wave_groups_values = _get_values(occ_cfg, "num_wave_groups", [1])

    return config_set


def generate_cpp_conv_kernel_set_declaration(
    config_set: GroupedConvKernelConfigSet,
    set_name: Optional[str] = None,
) -> str:
    """
    Generate C++ DECL_GROUPED_CONV_KERNEL_SET code from a GroupedConvKernelConfigSet.
    """
    name = set_name or config_set.name

    lines = [f"DECL_GROUPED_CONV_KERNEL_SET({name},"]

    for config in config_set.generate_configs():
        line = f'    .add("{config.dtype_input}", "{config.variant}", {config.ndim}, '
        line += f"{config.tile.tile_m}, {config.tile.tile_n}, {config.tile.tile_k})"
        lines.append(line)

    lines.append(");")

    return "\n".join(lines)


# =============================================================================
# GEMM Configuration Export Functions
# =============================================================================


def generate_cpp_kernel_set_declaration(
    config_set: KernelConfigSet,
    set_name: Optional[str] = None,
) -> str:
    """
    Generate C++ DECL_KERNEL_SET code from a KernelConfigSet.

    Args:
        config_set: The kernel configuration set
        set_name: Optional name override for the kernel set

    Returns:
        C++ code string with DECL_KERNEL_SET declaration
    """
    name = set_name or config_set.name

    lines = [f"DECL_KERNEL_SET({name},"]

    for config in config_set.generate_configs():
        # Generate .add() call for each config
        line = f'    .add("{config.dtype_a}", "{config.layout}", '
        line += f"{config.tile.tile_m}, {config.tile.tile_n}, {config.tile.tile_k})"
        lines.append(line)

    lines.append(");")

    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kernel_config_loader.py <config.json>")
        print("\nLoads kernel configurations from JSON and prints summary.")
        sys.exit(1)

    json_path = sys.argv[1]

    try:
        config_set = load_kernel_configs(json_path)

        print(f"Kernel Set: {config_set.name}")
        print(
            f"Data Types: A={config_set.dtype_a}, B={config_set.dtype_b}, C={config_set.dtype_c}, Acc={config_set.dtype_acc}"
        )
        print(f"Layout: {config_set.layout}")
        print(f"GPU Targets: {config_set.gpu_targets}")
        print(f"Variant: {config_set.variant}")
        print()
        print("Tile Configurations:")
        print(f"  tile_m: {config_set.tile_m_values}")
        print(f"  tile_n: {config_set.tile_n_values}")
        print(f"  tile_k: {config_set.tile_k_values}")
        print(f"  warp_m: {config_set.warp_m_values}")
        print(f"  warp_n: {config_set.warp_n_values}")
        print(f"  warp_k: {config_set.warp_k_values}")
        print(
            f"  warp_tile: {config_set.warp_tile_m_values}x{config_set.warp_tile_n_values}x{config_set.warp_tile_k_values}"
        )
        print()
        print("Trait Configurations:")
        print(f"  pipeline: {config_set.pipeline_values}")
        print(f"  scheduler: {config_set.scheduler_values}")
        print(f"  epilogue: {config_set.epilogue_values}")
        print(
            f"  padding: m={config_set.pad_m_values}, n={config_set.pad_n_values}, k={config_set.pad_k_values}"
        )
        print()
        print(f"Total configurations: {config_set.config_count()}")
        print()

        # Print first few config names
        print("Sample kernel names:")
        for i, config in enumerate(config_set.generate_configs()):
            if i >= 5:
                print(f"  ... and {config_set.config_count() - 5} more")
                break
            print(f"  {config.kernel_name()}")
        print()

        # Generate C++ code
        if "--cpp" in sys.argv:
            print("C++ Declaration:")
            print("-" * 60)
            print(generate_cpp_kernel_set_declaration(config_set))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
