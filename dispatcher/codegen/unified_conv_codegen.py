#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unified Convolution Code Generator

This is the unified code generator for all convolution kernel variants:
- Forward convolution
- Backward data convolution
- Backward weight convolution

Generates both CK Tile kernels AND dispatcher wrappers.
Based on the GEMM codegen pattern.
"""

import argparse
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass
from enum import Enum
import concurrent.futures

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Configuration and Data Structures
# ============================================================================


class ConvVariant(Enum):
    """Convolution kernel variants"""

    FORWARD = "forward"
    BACKWARD_DATA = "bwd_data"
    BACKWARD_WEIGHT = "bwd_weight"


class ConvLayout(Enum):
    """Convolution data layouts"""

    # 1D
    NWGC = "NWGC"  # Input/Output: N W G C
    GKXC = "GKXC"  # Weight: G K X C
    NWGK = "NWGK"  # Output: N W G K

    # 2D
    NHWGC = "NHWGC"  # Input: N H W G C
    GKYXC = "GKYXC"  # Weight: G K Y X C
    NHWGK = "NHWGK"  # Output: N H W G K

    # 3D
    NDHWGC = "NDHWGC"  # Input: N D H W G C
    GKZYXC = "GKZYXC"  # Weight: G K Z Y X C
    NDHWGK = "NDHWGK"  # Output: N D H W G K


@dataclass
class TileConfig:
    """Tile configuration parameters"""

    tile_m: int  # Output (N * spatial_out)
    tile_n: int  # K (output channels)
    tile_k: int  # C * filter_spatial (input channels * filter)
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    def is_valid(self) -> bool:
        """Validate tile configuration"""
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
            and self.tile_m > 0
            and self.tile_n > 0
            and self.tile_k > 0
        )


@dataclass
class TraitConfig:
    """Kernel trait configuration"""

    pipeline: str  # mem, compv3, compv4, compv5
    scheduler: str  # intrawave, interwave
    epilogue: str = "cshuffle"  # cshuffle, default
    double_smem_buffer: bool = False
    pad_m: bool = True  # Padding for M dimension
    pad_n: bool = True  # Padding for N dimension
    pad_k: bool = True  # Padding for K dimension
    num_groups_to_merge: int = 1

    def is_valid(self) -> bool:
        """Check if trait combination is valid"""
        # Unsupported combinations (same as GEMM)
        unsupported = {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave"),
        }
        return (self.pipeline, self.epilogue, self.scheduler) not in unsupported


@dataclass
class ConvKernelConfig:
    """Complete convolution kernel configuration"""

    tile: TileConfig
    trait: TraitConfig
    variant: ConvVariant = ConvVariant.FORWARD
    ndim_spatial: int = 2  # 1D, 2D, or 3D
    arch: str = "gfx942"  # Target architecture

    # Vector sizes
    vector_size_a: int = 4
    vector_size_b: int = 8
    vector_size_c: int = 8

    # Fixed parameters
    block_per_cu: int = 1
    num_wave_groups: int = 1

    def name(self, datatype: str) -> str:
        """Generate kernel name"""
        t = self.tile
        tr = self.trait

        variant_str = {
            ConvVariant.FORWARD: "fwd",
            ConvVariant.BACKWARD_DATA: "bwdd",
            ConvVariant.BACKWARD_WEIGHT: "bwdw",
        }[self.variant]

        name = f"conv_{variant_str}_{datatype}_{self.ndim_spatial}d"
        name += f"_{tr.pipeline}_{tr.epilogue}_{tr.scheduler}"
        name += f"_{t.tile_m}x{t.tile_n}x{t.tile_k}"
        name += f"_{t.warp_m}x{t.warp_n}x{t.warp_k}"

        # Add padding suffix if not all enabled
        if not (tr.pad_m and tr.pad_n and tr.pad_k):
            name += f"_pad{int(tr.pad_m)}{int(tr.pad_n)}{int(tr.pad_k)}"

        return name

    def is_valid_for_arch(self) -> bool:
        """Check if configuration is valid for target architecture"""
        # Check trait validity
        if not self.trait.is_valid():
            return False

        # Check warp configuration (from arch_specs)
        try:
            from arch_specs_generated import WARP_SUPPORTED_COMBINATIONS

            supported = WARP_SUPPORTED_COMBINATIONS.get(self.arch, [])
            warp_cfg = [self.tile.warp_m, self.tile.warp_n, self.tile.warp_k]
            if supported and warp_cfg not in supported:
                return False
        except ImportError:
            pass  # Allow if arch_specs not available

        return True


# ============================================================================
# Type Mappings
# ============================================================================


class TypeMappings:
    """Centralized type mappings for code generation"""

    DTYPE_TO_CK = {
        "fp16": "half_t",
        "bf16": "bf16_t",
        "fp32": "float",
    }

    PIPELINE_TO_CK = {
        "mem": "GemmPipeline::MEMORY",
        "compv3": "GemmPipeline::COMPUTE_V3",
        "compv4": "GemmPipeline::COMPUTE_V4",
        "compv5": "GemmPipeline::COMPUTE_V5",
    }

    SCHEDULER_TO_CK = {
        "intrawave": "GemmPipelineScheduler::Intrawave",
        "interwave": "GemmPipelineScheduler::Interwave",
    }

    LAYOUT_1D = {
        "in": "tensor_layout::convolution::NWGC",
        "wei": "tensor_layout::convolution::GKXC",
        "out": "tensor_layout::convolution::NWGK",
    }

    LAYOUT_2D = {
        "in": "tensor_layout::convolution::NHWGC",
        "wei": "tensor_layout::convolution::GKYXC",
        "out": "tensor_layout::convolution::NHWGK",
    }

    LAYOUT_3D = {
        "in": "tensor_layout::convolution::NDHWGC",
        "wei": "tensor_layout::convolution::GKZYXC",
        "out": "tensor_layout::convolution::NDHWGK",
    }

    @classmethod
    def get_layouts(cls, ndim: int) -> dict:
        if ndim == 1:
            return cls.LAYOUT_1D
        elif ndim == 2:
            return cls.LAYOUT_2D
        else:
            return cls.LAYOUT_3D


# ============================================================================
# CK Tile Conv Kernel Generator
# ============================================================================


class CKTileConvKernelGenerator:
    """Generates CK Tile convolution kernel instance code"""

    def __init__(self, datatype: str, variant: ConvVariant = ConvVariant.FORWARD):
        self.datatype = datatype
        self.variant = variant
        self.tm = TypeMappings()

    def generate(self, config: ConvKernelConfig) -> str:
        """Generate complete CK Tile convolution kernel"""
        kernel_name = config.name(self.datatype)
        return f"""{self._header(kernel_name)}
{self._config_struct(config, kernel_name)}
{self._kernel_instance(config, kernel_name)}
"""

    def _header(self, kernel_name: str) -> str:
        """Generate header includes based on variant"""
        if self.variant == ConvVariant.BACKWARD_DATA:
            kernel_header = "grouped_convolution_backward_data_kernel.hpp"
        elif self.variant == ConvVariant.BACKWARD_WEIGHT:
            kernel_header = "grouped_convolution_backward_weight_kernel.hpp"
        else:
            kernel_header = "grouped_convolution_forward_kernel.hpp"

        return f"""// SPDX-License-Identifier: MIT
// Auto-generated CK Tile Convolution kernel: {kernel_name}
// Variant: {self.variant.value}
#pragma once

#include <cstdint>
#include <numeric>
#include <functional>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/{kernel_header}"

using namespace ck_tile;
"""

    def _config_struct(self, config: ConvKernelConfig, kernel_name: str) -> str:
        """Generate config struct"""
        t = config.tile
        tr = config.trait
        layouts = self.tm.get_layouts(config.ndim_spatial)

        return f"""
// Kernel configuration
struct {kernel_name}_Config {{
    // Data types
    using InDataType = {self.tm.DTYPE_TO_CK[self.datatype]};
    using WeiDataType = {self.tm.DTYPE_TO_CK[self.datatype]};
    using AccDataType = float;
    using OutDataType = {self.tm.DTYPE_TO_CK[self.datatype]};
    
    // Layouts
    using InLayout = {layouts["in"]};
    using WeiLayout = {layouts["wei"]};
    using OutLayout = {layouts["out"]};
    
    // Tile shape
    static constexpr index_t M_Tile = {t.tile_m};
    static constexpr index_t N_Tile = {t.tile_n};
    static constexpr index_t K_Tile = {t.tile_k};
    
    static constexpr index_t M_Warp = {t.warp_m};
    static constexpr index_t N_Warp = {t.warp_n};
    static constexpr index_t K_Warp = {t.warp_k};
    
    static constexpr index_t M_Warp_Tile = {t.warp_tile_m};
    static constexpr index_t N_Warp_Tile = {t.warp_tile_n};
    static constexpr index_t K_Warp_Tile = {t.warp_tile_k};
    
    // Vector sizes
    static constexpr index_t VectorSizeA = {config.vector_size_a};
    static constexpr index_t VectorSizeB = {config.vector_size_b};
    static constexpr index_t VectorSizeC = {config.vector_size_c};
    
    // Padding
    static constexpr bool kPadM = {str(tr.pad_m).lower()};
    static constexpr bool kPadN = {str(tr.pad_n).lower()};
    static constexpr bool kPadK = {str(tr.pad_k).lower()};
    
    // Pipeline & Epilogue
    static constexpr auto Pipeline = {self.tm.PIPELINE_TO_CK[tr.pipeline]};
    static constexpr auto Scheduler = {self.tm.SCHEDULER_TO_CK[tr.scheduler]};
    static constexpr bool DoubleSmemBuffer = {str(tr.double_smem_buffer).lower()};
    static constexpr bool UseCShuffleEpilogue = {str(tr.epilogue == "cshuffle").lower()};
    
    // Other params
    static constexpr int kBlockPerCu = {config.block_per_cu};
    static constexpr index_t NumWaveGroups = {config.num_wave_groups};
    static constexpr index_t NumGroupsToMerge = {tr.num_groups_to_merge};
    static constexpr index_t NDimSpatial = {config.ndim_spatial};
    
    // Target architecture
    static constexpr const char* TargetArch = "{config.arch}";
}};
"""

    def _kernel_instance(self, config: ConvKernelConfig, kernel_name: str) -> str:
        """Generate kernel instantiation code with launch function"""
        tr = config.trait

        # Variant-specific configuration
        if self.variant == ConvVariant.BACKWARD_DATA:
            host_args_type = "GroupedConvBwdDataHostArgs"
            kernel_type = "GroupedConvolutionBackwardDataKernel"
            gemm_traits = "GroupedConvImplicitGemmTraitsBwdData"
            layout_suffix = "BwdData"
            # For bwd_data: A=dOutput, B=Weight, C=dInput
            a_dtype = "OutDataType"
            b_dtype = "WeiDataType"
            c_dtype = "InDataType"
            gemm_k_calc = "args.K_ * std::accumulate(args.filter_spatial_lengths_.begin(), args.filter_spatial_lengths_.end()"
            direction_prefix = "BWD_DATA"
            launcher_alias = "SelectedConvBwdDataLauncher"
        elif self.variant == ConvVariant.BACKWARD_WEIGHT:
            host_args_type = "GroupedConvBwdWeightHostArgs"
            kernel_type = "GroupedConvolutionBackwardWeightKernel"
            gemm_traits = "GroupedConvImplicitGemmTraitsBwdWeight"
            layout_suffix = "BwdWeight"
            # For bwd_weight: A=dOutput, B=Input, C=dWeight (per CK Tile invoker)
            a_dtype = "OutDataType"
            b_dtype = "InDataType"
            c_dtype = "WeiDataType"
            gemm_k_calc = "args.N_ * std::accumulate(args.output_spatial_lengths_.begin(), args.output_spatial_lengths_.end()"
            direction_prefix = "BWD_WEIGHT"
            launcher_alias = "SelectedConvBwdWeightLauncher"
        else:  # Forward
            host_args_type = "GroupedConvFwdHostArgs<>"
            kernel_type = "GroupedConvolutionForwardKernel"
            gemm_traits = "GroupedConvImplicitGemmTraitsFwd"
            layout_suffix = "Fwd"
            a_dtype = "InDataType"
            b_dtype = "WeiDataType"
            c_dtype = "OutDataType"
            gemm_k_calc = "args.C_ * std::accumulate(args.filter_spatial_lengths_.begin(), args.filter_spatial_lengths_.end()"
            direction_prefix = "FWD"
            launcher_alias = "SelectedConvKernelLauncher"

        return f"""
// Kernel name for identification
constexpr const char* CONV_{direction_prefix}_KERNEL_NAME = "{kernel_name}";

// Selected kernel alias
using SelectedConv{direction_prefix.title()}Kernel = {kernel_name}_Config;

// =============================================================================
// Kernel Launch Implementation ({self.variant.value})
// =============================================================================

struct {kernel_name}_Launcher {{
    using Config = {kernel_name}_Config;
    using InDataType = typename Config::InDataType;
    using WeiDataType = typename Config::WeiDataType;
    using OutDataType = typename Config::OutDataType;
    using AccDataType = typename Config::AccDataType;
    using InLayout = typename Config::InLayout;
    using WeiLayout = typename Config::WeiLayout;
    using OutLayout = typename Config::OutLayout;
    
    static constexpr index_t NDimSpatial = Config::NDimSpatial;
    
    // Implicit GEMM shape
    using GemmShape = TileGemmShape<
        sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
        sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
        sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;
    
    // Convolution traits
    static constexpr auto ConvSpec = ConvolutionSpecialization::Default;
    using GroupedConvTraitsType = GroupedConvTraits<
        NDimSpatial, ConvSpec, InLayout, WeiLayout, tuple<>, OutLayout,
        Config::VectorSizeA, Config::VectorSizeB, Config::VectorSizeC,
        Config::NumGroupsToMerge>;
    
    // Tile partitioner
    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;
    
    // Universal traits - layout suffix changes per variant
    using GemmUniversalTraits = TileGemmUniversalTraits<
        GroupedConvTraitsType::FixedGemmParams::kPadM,
        GroupedConvTraitsType::FixedGemmParams::kPadN,
        GroupedConvTraitsType::FixedGemmParams::kPadK,
        Config::DoubleSmemBuffer,
        typename GroupedConvTraitsType::AsLayout{layout_suffix},
        typename GroupedConvTraitsType::BsLayout{layout_suffix},
        typename GroupedConvTraitsType::CLayout{layout_suffix},
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
        GroupedConvTraitsType::FixedGemmParams::Persistent,
        Config::NumWaveGroups>;
    
    // Pipeline problem - data types change per variant
    using GemmPipelineProblem = GemmPipelineProblem<
        {a_dtype}, {b_dtype}, AccDataType, GemmShape,
        typename GroupedConvTraitsType::template {gemm_traits}<Config::NumWaveGroups>,
        element_wise::PassThrough, element_wise::PassThrough, {c_dtype},
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeA, GroupedConvTraitsType::VectorSizeB>;
    
    // Base pipeline for tail handling
    using BaseGemmPipeline = {self._get_base_pipeline(tr.pipeline)}<GemmPipelineProblem>;
    
    static float launch(const {host_args_type}& args, const stream_config& s) {{
        const index_t gemm_k = {gemm_k_calc}, 1, std::multiplies<index_t>());
        
        const index_t k_grain = args.k_batch * Config::K_Tile;
        const index_t K_split = (gemm_k + k_grain - 1) / k_grain * Config::K_Tile;
        const index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_, 
                             const auto memory_operation_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;
            constexpr auto scheduler = Config::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;
            
            using UniversalGemmProblem = UniversalGemmPipelineProblem<
                {a_dtype}, {b_dtype}, AccDataType, GemmShape, GemmUniversalTraits,
                scheduler, has_hot_loop_v, tail_number_v,
                element_wise::PassThrough, element_wise::PassThrough, {c_dtype},
                GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                GroupedConvTraitsType::VectorSizeA, GroupedConvTraitsType::VectorSizeB>;
            
            using GemmPipeline = {self._get_pipeline(tr.pipeline)}<UniversalGemmProblem>;
            
            using ConvEpilogue = CShuffleEpilogue<CShuffleEpilogueProblem<
                {a_dtype}, {b_dtype}, tuple<>, AccDataType, {c_dtype},
                typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                typename GroupedConvTraitsType::FixedGemmParams::ELayout,
                element_wise::PassThrough,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                Config::M_Warp, Config::N_Warp, Config::M_Warp_Tile, 
                Config::N_Warp_Tile, Config::K_Warp_Tile,
                GroupedConvTraitsType::FixedGemmParams::TransposeC,
                memory_operation, Config::NumWaveGroups,
                GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                Config::VectorSizeC>>;
            
            using Kernel = {kernel_type}<
                GroupedConvTraitsType, TilePartitioner, GemmPipeline, ConvEpilogue>;
            
            auto kargs = Kernel::MakeKernelArgs(args);
            
            if (!Kernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Arguments not supported for conv kernel");
            }}
            
            const dim3 grids = Kernel::GridSize(kargs);
            const dim3 blocks = Kernel::BlockSize();
            
            ave_time = launch_kernel(s, make_kernel<Config::kBlockPerCu>(
                Kernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};
        
        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {{
            if (args.k_batch == 1) {{
                Run(has_hot_loop_, tail_number_,
                    integral_constant<memory_operation_enum, memory_operation_enum::set>{{}});
            }} else {{
                Run(has_hot_loop_, tail_number_,
                    integral_constant<memory_operation_enum, memory_operation_enum::atomic_add>{{}});
            }}
        }};
        
        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        return ave_time;
    }}
}};

// Launcher alias for examples
using {launcher_alias} = {kernel_name}_Launcher;
"""

    def _get_pipeline(self, pipeline: str) -> str:
        """Get pipeline class name"""
        pipelines = {
            "mem": "GemmPipelineAgBgCrMem",
            "compv3": "GemmPipelineAgBgCrCompV3",
            "compv4": "GemmPipelineAgBgCrCompV4",
            "compv5": "GemmPipelineAgBgCrCompV5",
        }
        return pipelines.get(pipeline, "GemmPipelineAgBgCrCompV3")

    def _get_base_pipeline(self, pipeline: str) -> str:
        """Get base pipeline class name"""
        pipelines = {
            "mem": "BaseGemmPipelineAgBgCrMem",
            "compv3": "BaseGemmPipelineAgBgCrCompV3",
            "compv4": "BaseGemmPipelineAgBgCrCompV4",
            "compv5": "BaseGemmPipelineAgBgCrCompV5",
        }
        return pipelines.get(pipeline, "BaseGemmPipelineAgBgCrCompV3")


# ============================================================================
# Dispatcher Wrapper Generator
# ============================================================================


class DispatcherWrapperGenerator:
    """Generates dispatcher integration wrapper"""

    def __init__(self, datatype: str):
        self.datatype = datatype

    def generate(self, config: ConvKernelConfig) -> str:
        """Generate dispatcher wrapper - empty for now, launcher is sufficient"""
        # The launcher struct already provides all needed functionality
        # Dispatcher integration can be added later if needed
        return ""


# ============================================================================
# Configuration Parser
# ============================================================================


def get_default_configs(
    arch: str = "gfx942", variants: List[ConvVariant] = None, ndims: List[int] = None
) -> List[ConvKernelConfig]:
    """Get default convolution configurations for target architecture"""
    configs = []

    if variants is None:
        variants = [ConvVariant.FORWARD]
    if ndims is None:
        ndims = [2]

    # Valid configurations per variant (based on CK Tile example configs)
    # Forward and Backward Data: standard GEMM-like tiles
    fwd_bwd_data_tiles = [
        # (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_m, warp_tile_n, warp_tile_k)
        (128, 128, 32, 2, 2, 32, 32, 16),  # Standard 128x128
        (256, 256, 32, 2, 2, 32, 32, 16),  # Large 256x256
        (64, 64, 32, 1, 4, 16, 16, 16),  # Small 64x64
        (128, 64, 32, 2, 2, 32, 32, 16),  # Rectangular
        (16, 64, 64, 1, 4, 16, 16, 32),  # Tall and narrow
    ]

    # Backward Weight: specific tile configs that work with CK Tile's bwd_weight kernel
    # Based on ConvConfigComputeV3 from CK Tile examples
    bwd_weight_tiles = [
        # (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_m, warp_tile_n, warp_tile_k)
        (16, 64, 64, 1, 4, 16, 16, 32),  # ConvConfigComputeV3 compatible
        (32, 64, 64, 2, 2, 16, 16, 32),  # Alternative small
        (64, 128, 32, 2, 2, 32, 32, 16),  # Medium
    ]

    for variant in variants:
        # Select tile configs based on variant
        if variant == ConvVariant.BACKWARD_WEIGHT:
            tile_configs = bwd_weight_tiles
        else:
            tile_configs = fwd_bwd_data_tiles
        for ndim in ndims:
            for pipeline, epilogue in [("compv3", "cshuffle"), ("compv4", "cshuffle")]:
                for (
                    tile_m,
                    tile_n,
                    tile_k,
                    warp_m,
                    warp_n,
                    warp_tile_m,
                    warp_tile_n,
                    warp_tile_k,
                ) in tile_configs:
                    # Adjust tile_k for compv4 (needs larger K for double buffering)
                    adj_tile_k = tile_k * 2 if pipeline == "compv4" else tile_k

                    trait = TraitConfig(
                        pipeline=pipeline,
                        scheduler="intrawave",
                        epilogue=epilogue,
                        double_smem_buffer=(pipeline == "compv4"),
                        pad_m=True,
                        pad_n=True,
                        pad_k=True,
                    )

                    # Skip invalid combinations
                    if not trait.is_valid():
                        continue

                    config = ConvKernelConfig(
                        tile=TileConfig(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            tile_k=adj_tile_k,
                            warp_m=warp_m,
                            warp_n=warp_n,
                            warp_k=1,
                            warp_tile_m=warp_tile_m,
                            warp_tile_n=warp_tile_n,
                            warp_tile_k=warp_tile_k,
                        ),
                        trait=trait,
                        variant=variant,
                        ndim_spatial=ndim,
                        arch=arch,
                    )

                    # Validate for target arch
                    if config.is_valid_for_arch():
                        configs.append(config)

    return configs


def get_arch_filter():
    """Get arch filter if available"""
    try:
        from arch_filter import ArchFilter

        return ArchFilter
    except ImportError:
        return None


# ============================================================================
# Main Generator
# ============================================================================


class UnifiedConvCodegen:
    """Main convolution code generator"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_files: List[Path] = []

    def generate_kernel(
        self,
        config: ConvKernelConfig,
        datatype: str,
        variant: ConvVariant = ConvVariant.FORWARD,
    ) -> Path:
        """Generate a single kernel file"""
        kernel_gen = CKTileConvKernelGenerator(datatype, variant)
        wrapper_gen = DispatcherWrapperGenerator(datatype)

        kernel_name = config.name(datatype)
        filename = f"{kernel_name}.hpp"
        filepath = self.output_dir / filename

        content = kernel_gen.generate(config)
        content += wrapper_gen.generate(config)

        filepath.write_text(content)
        self.generated_files.append(filepath)

        log.info(f"Generated: {filename}")
        return filepath

    def generate_all(
        self,
        configs: List[ConvKernelConfig],
        datatypes: List[str],
        parallel: bool = True,
    ) -> List[Path]:
        """Generate all kernel files (optionally in parallel)"""

        tasks = [
            (config, datatype, config.variant)
            for datatype in datatypes
            for config in configs
        ]

        if parallel and len(tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.generate_kernel, config, dtype, variant)
                    for config, dtype, variant in tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Collect results
                    except Exception as e:
                        log.error(f"Failed to generate kernel: {e}")
        else:
            for config, dtype, variant in tasks:
                self.generate_kernel(config, dtype, variant)

        return self.generated_files


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Unified Convolution Code Generator")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("build/generated_kernels"),
        help="Output directory",
    )
    parser.add_argument(
        "--datatype",
        "-d",
        type=str,
        nargs="+",
        default=["fp16"],
        choices=["fp16", "bf16", "fp32"],
        help="Data types to generate",
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        nargs="+",
        default=["forward"],
        choices=["forward", "bwd_data", "bwd_weight"],
        help="Convolution variants",
    )
    parser.add_argument(
        "--ndim",
        "-n",
        type=int,
        nargs="+",
        default=[2],
        choices=[1, 2, 3],
        help="Spatial dimensions",
    )
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        default="gfx942",
        choices=["gfx90a", "gfx942", "gfx950", "gfx1201"],
        help="Target GPU architecture",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List configurations without generating",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Map variant strings to enums
    variant_map = {
        "forward": ConvVariant.FORWARD,
        "bwd_data": ConvVariant.BACKWARD_DATA,
        "bwd_weight": ConvVariant.BACKWARD_WEIGHT,
    }
    requested_variants = [variant_map[v] for v in args.variant]

    # Get configurations for target arch with requested variants and ndims
    filtered_configs = get_default_configs(
        arch=args.arch, variants=requested_variants, ndims=args.ndim
    )

    if args.list_configs:
        print(f"Convolution configurations for {args.arch}:")
        print(f"  Datatypes: {args.datatype}")
        print(f"  Variants: {args.variant}")
        print(f"  Spatial dims: {args.ndim}")
        print(f"\nConfigurations ({len(filtered_configs)}):")
        for cfg in filtered_configs:
            print(f"  - {cfg.name('fp16')}")
            print(f"      Tile: {cfg.tile.tile_m}x{cfg.tile.tile_n}x{cfg.tile.tile_k}")
            print(f"      Warp: {cfg.tile.warp_m}x{cfg.tile.warp_n}x{cfg.tile.warp_k}")
            print(
                f"      WarpTile: {cfg.tile.warp_tile_m}x{cfg.tile.warp_tile_n}x{cfg.tile.warp_tile_k}"
            )
            print(
                f"      Pipeline: {cfg.trait.pipeline}, Epilogue: {cfg.trait.epilogue}, Scheduler: {cfg.trait.scheduler}"
            )
            print(
                f"      Padding: M={cfg.trait.pad_m}, N={cfg.trait.pad_n}, K={cfg.trait.pad_k}"
            )
        return

    # Generate
    codegen = UnifiedConvCodegen(args.output)
    files = codegen.generate_all(filtered_configs, args.datatype)

    print(
        f"\nGenerated {len(files)} convolution kernel files for {args.arch} in {args.output}"
    )


if __name__ == "__main__":
    main()
