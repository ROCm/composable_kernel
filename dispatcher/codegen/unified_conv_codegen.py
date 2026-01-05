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
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import concurrent.futures

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# Import architecture filter for GPU-specific validation
try:
    from arch_filter import ArchFilter, OperatorType

    HAS_ARCH_FILTER = True
except ImportError:
    HAS_ARCH_FILTER = False
    ArchFilter = None
    OperatorType = None


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
    """Kernel trait configuration (same order as GEMM for consistency)"""

    pipeline: str  # mem, compv3, compv4, compv5
    epilogue: str  # cshuffle, default
    scheduler: str  # intrawave, interwave
    pad_m: bool = True  # Padding for M dimension
    pad_n: bool = True  # Padding for N dimension
    pad_k: bool = True  # Padding for K dimension
    double_smem_buffer: bool = False
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
    layout: str = "nhwgc"  # Data layout (e.g., "nhwgc", "nchw", "ndhwgc")

    # Vector sizes
    vector_size_a: int = 4
    vector_size_b: int = 8
    vector_size_c: int = 8

    # Occupancy parameters
    block_per_cu: int = 1
    num_wave_groups: int = 1
    num_groups_to_merge: int = 1  # For group merged convolution

    # Double buffering
    double_smem_buffer: bool = False

    def name(self, datatype: str) -> str:
        """
        Generate kernel name that uniquely identifies the kernel configuration.

        Format: conv_{variant}_{dtype}_{layout}_{ndim}d_{pipeline}_{epilogue}_{scheduler}
                _{tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}
                _{warp_tile_m}x{warp_tile_n}x{warp_tile_k}
                [_vec{a}_{b}_{c}][_bpc{n}][_wg{n}][_gm{n}][_dsb][_pad{mnk}]

        All parameters that affect kernel behavior MUST be included to ensure
        unique names for unique configurations:
        - Variant (fwd/bwdd/bwdw)
        - Data type
        - Layout (nhwgc, nchw, ndhwgc, etc.)
        - Spatial dimensions (2d/3d)
        - Pipeline, epilogue, scheduler
        - Tile, warp, warp_tile dimensions
        - Vector sizes, occupancy hints (if non-default)
        - Double SMEM buffer, padding flags
        """
        t = self.tile
        tr = self.trait

        variant_str = {
            ConvVariant.FORWARD: "fwd",
            ConvVariant.BACKWARD_DATA: "bwdd",
            ConvVariant.BACKWARD_WEIGHT: "bwdw",
        }[self.variant]

        # Core identity: variant, dtype, layout, dims
        name = f"conv_{variant_str}_{datatype}_{self.layout}_{self.ndim_spatial}d"

        # Pipeline configuration
        name += f"_{tr.pipeline}_{tr.epilogue}_{tr.scheduler}"

        # Block tile dimensions (M_Tile x N_Tile x K_Tile)
        name += f"_{t.tile_m}x{t.tile_n}x{t.tile_k}"

        # Wave distribution (M_Warp x N_Warp x K_Warp)
        name += f"_{t.warp_m}x{t.warp_n}x{t.warp_k}"

        # Warp tile dimensions (M_Warp_Tile x N_Warp_Tile x K_Warp_Tile)
        name += f"_{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}"

        # Vector sizes (only if non-default)
        if (self.vector_size_a, self.vector_size_b, self.vector_size_c) != (4, 8, 8):
            name += (
                f"_vec{self.vector_size_a}_{self.vector_size_b}_{self.vector_size_c}"
            )

        # Occupancy hints (only if non-default)
        if self.block_per_cu != 1:
            name += f"_bpc{self.block_per_cu}"

        if self.num_wave_groups != 1:
            name += f"_wg{self.num_wave_groups}"

        if self.num_groups_to_merge != 1:
            name += f"_gm{self.num_groups_to_merge}"

        # Double SMEM buffer (for compute V4+)
        if self.double_smem_buffer or tr.double_smem_buffer:
            name += "_dsb"

        # Padding suffix (only if not all enabled)
        if not (tr.pad_m and tr.pad_n and tr.pad_k):
            name += f"_pad{int(tr.pad_m)}{int(tr.pad_n)}{int(tr.pad_k)}"

        return name

    def is_valid_for_arch(self) -> bool:
        """Check if configuration is valid for target architecture"""
        # Check trait validity
        if not self.trait.is_valid():
            return False

        # Backward operations have stricter pipeline requirements:
        # - Backward weight: compv4/compv5 have transpose_tile2d issues
        # - Backward data: compv4 has get_length issues in bwd_data kernel
        # Both backward operations ONLY support compv3 and mem pipelines
        if self.variant in (ConvVariant.BACKWARD_WEIGHT, ConvVariant.BACKWARD_DATA):
            if self.trait.pipeline not in ("compv3", "mem"):
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

        # Create valid C++ namespace name
        ns_name = "ns_" + kernel_name.replace("-", "_")

        return f"""
// Unique namespace for this kernel to avoid conflicts when including multiple kernels
namespace {ns_name} {{

// Bring Config into namespace
using Config = {kernel_name}_Config;

// Kernel name for identification
constexpr const char* CONV_{direction_prefix}_KERNEL_NAME = "{kernel_name}";

// Selected kernel alias
using SelectedConv{direction_prefix.title()}Kernel = Config;

// =============================================================================
// Kernel Launch Implementation ({self.variant.value})
// =============================================================================

struct {kernel_name}_Launcher {{
    using KernelConfig = Config; // Use the Config alias from namespace
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
        
        constexpr auto scheduler = Config::Scheduler;
        
        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            {a_dtype}, {b_dtype}, AccDataType, GemmShape, GemmUniversalTraits,
            scheduler,
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
            Config::NumWaveGroups,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            Config::VectorSizeC, false, 1, Config::DoubleSmemBuffer>>;
        
        using Kernel = {kernel_type}<
            GroupedConvTraitsType, TilePartitioner, GemmPipeline, ConvEpilogue>;
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
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
        
        BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
        return ave_time;
    }}
}};

// Launcher alias for examples
using {launcher_alias} = {kernel_name}_Launcher;
// Launcher alias
using {launcher_alias} = {kernel_name}_Launcher;

}} // namespace {ns_name}

// Export specific launcher to global namespace
using {kernel_name}_Launcher = {ns_name}::{kernel_name}_Launcher;
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
    """Generates dispatcher integration wrapper following GEMM pattern"""

    # Static mappings for pipeline and scheduler enum names (matches kernel_key.hpp)
    PIPELINE_TO_DISPATCHER = {
        "mem": "Pipeline::Mem",
        "compv3": "Pipeline::CompV3",
        "compv4": "Pipeline::CompV4",
        "compv5": "Pipeline::CompV5",
        "preshufflev1": "Pipeline::PreShuffleV1",
        "preshufflev2": "Pipeline::PreShuffleV2",
    }

    SCHEDULER_TO_DISPATCHER = {
        "default": "Scheduler::Default",
        "intrawave": "Scheduler::Intrawave",
        "interwave": "Scheduler::Interwave",
    }

    def __init__(self, datatype: str, variant: ConvVariant):
        self.datatype = datatype
        self.variant = variant

    def _pipeline_to_dispatcher(self, pipeline: str) -> str:
        """Convert pipeline string to dispatcher enum value"""
        return self.PIPELINE_TO_DISPATCHER.get(
            pipeline.lower(), f"Pipeline::{pipeline.capitalize()}"
        )

    def _scheduler_to_dispatcher(self, scheduler: str) -> str:
        """Convert scheduler string to dispatcher enum value"""
        return self.SCHEDULER_TO_DISPATCHER.get(
            scheduler.lower(), f"Scheduler::{scheduler.capitalize()}"
        )

    def generate(
        self, config: ConvKernelConfig, kernel_path: Path, output_dir: Path
    ) -> str:
        """Generate dispatcher wrapper with factory function for registry"""
        kernel_name = config.name(self.datatype)
        rel_path = kernel_path.relative_to(output_dir)

        # Determine launcher type based on variant
        if self.variant == ConvVariant.FORWARD:
            launcher_alias = "SelectedConvKernelLauncher"
            host_args_type = "GroupedConvFwdHostArgs<>"
            conv_type_str = "forward"
        elif self.variant == ConvVariant.BACKWARD_DATA:
            launcher_alias = "SelectedConvBwdDataLauncher"
            host_args_type = "GroupedConvBwdDataHostArgs"
            conv_type_str = "bwd_data"
        else:  # BACKWARD_WEIGHT
            launcher_alias = "SelectedConvBwdWeightLauncher"
            host_args_type = "GroupedConvBwdWeightHostArgs"
            conv_type_str = "bwd_weight"

        return f"""// SPDX-License-Identifier: MIT
// Auto-generated dispatcher wrapper for: {kernel_name}
#pragma once

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/conv_utils.hpp"
#include "../{rel_path}"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

using ::ck_tile::dispatcher::ConvKernelInstancePtr;
using ::ck_tile::dispatcher::ConvKernelKey;
using ::ck_tile::dispatcher::DataType;
using ::ck_tile::dispatcher::LayoutTag;
using ::ck_tile::dispatcher::Pipeline;
using ::ck_tile::dispatcher::Scheduler;
using ::ck_tile::dispatcher::Epilogue;
using Priority = ::ck_tile::dispatcher::ConvRegistry::Priority;

// Factory function to create kernel instance for registry
inline ConvKernelInstancePtr make_{kernel_name}(const std::string& gfx_arch = "gfx942") {{
    ConvKernelKey key;
    key.signature.dtype_in = DataType::FP16;
    key.signature.dtype_wei = DataType::FP16;
    key.signature.dtype_out = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    key.signature.layout = "nhwgc";
    key.signature.conv_type = "{conv_type_str}";
    key.signature.num_dims = {config.ndim_spatial};
    key.signature.groups = 1;
    
    key.algorithm.tile_shape = {{{config.tile.tile_m}, {config.tile.tile_n}, {config.tile.tile_k}}};
    key.algorithm.wave_shape = {{{config.tile.warp_m}, {config.tile.warp_n}, 1}};
    key.algorithm.warp_tile_shape = {{{config.tile.warp_tile_m}, {config.tile.warp_tile_n}, {config.tile.warp_tile_k}}};
    key.algorithm.pipeline = {self._pipeline_to_dispatcher(config.trait.pipeline)};
    key.algorithm.scheduler = {self._scheduler_to_dispatcher(config.trait.scheduler)};
    key.algorithm.epilogue = Epilogue::CShuffle;
    key.gfx_arch = gfx_arch;
    
    // Create kernel instance that wraps the launcher
    return std::make_shared<ConvKernelInstance>(
        key,
        "{kernel_name}",
        []({host_args_type}& args, const stream_config& cfg) -> float {{
            return {kernel_name}_Launcher::launch(args, cfg);
        }}
    );
}}

}}  // namespace generated
}}  // namespace dispatcher
}}  // namespace ck_tile

// Export launcher alias to global namespace for direct use
using {launcher_alias} = {kernel_name}_Launcher;
"""


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

    # Backward Weight: VERY specific tile configs that work with CK Tile's bwd_weight kernel
    # Based on ConvConfigComputeV3 from CK Tile examples (example/ck_tile/20_grouped_convolution/)
    # Note: Backward weight has strict constraints on warp configurations due to transpose_tile2d
    # Only specific warp configs work: (1, 4, 1) and (4, 1, 1) are known to work
    bwd_weight_tiles = [
        # (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_m, warp_tile_n, warp_tile_k)
        # ConvConfigComputeV3: The primary working config for backward weight
        (16, 64, 64, 1, 4, 16, 16, 32),
        # ConvConfigMem style - uses mem pipeline (handled separately)
        # (128, 32, 64, 4, 1, 32, 32, 16) - for mem pipeline only
    ]

    for variant in variants:
        # Select tile configs based on variant
        if variant == ConvVariant.BACKWARD_WEIGHT:
            tile_configs = bwd_weight_tiles
            # Backward weight ONLY supports compv3 (compv4/compv5 have transpose_tile2d issues)
            pipelines = [("compv3", "cshuffle")]
        elif variant == ConvVariant.BACKWARD_DATA:
            tile_configs = fwd_bwd_data_tiles
            # Backward data ONLY supports compv3 (compv4 has get_length issues in bwd_data kernel)
            pipelines = [("compv3", "cshuffle")]
        else:
            tile_configs = fwd_bwd_data_tiles
            # Only forward convolution supports both compv3 and compv4
            pipelines = [("compv3", "cshuffle"), ("compv4", "cshuffle")]
        for ndim in ndims:
            for pipeline, epilogue in pipelines:
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

    def __init__(
        self,
        output_dir: Path,
        gpu_target: str = "gfx942",
        enable_arch_filter: bool = True,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create wrapper directory for dispatcher integration
        self.wrapper_dir = self.output_dir / "dispatcher_wrappers"
        self.wrapper_dir.mkdir(parents=True, exist_ok=True)

        self.generated_files: List[Path] = []
        self.generated_wrappers: List[Path] = []
        self.gpu_target = gpu_target

        # Initialize architecture filter for GPU-specific validation
        self.arch_filter = None
        if enable_arch_filter and HAS_ARCH_FILTER:
            try:
                self.arch_filter = ArchFilter(gpu_target, strict_mode=False)
                log.info(f"Architecture filter enabled for {gpu_target}")
            except ValueError as e:
                log.warning(f"Could not create arch filter: {e}")

    def _get_operator_type(self, variant: "ConvVariant") -> Optional["OperatorType"]:
        """Map ConvVariant to OperatorType for arch validation"""
        if OperatorType is None:
            return None

        variant_to_operator = {
            ConvVariant.FORWARD: OperatorType.CONV_FWD,
            ConvVariant.BACKWARD_DATA: OperatorType.CONV_BWD_DATA,
            ConvVariant.BACKWARD_WEIGHT: OperatorType.CONV_BWD_WEIGHT,
        }
        return variant_to_operator.get(variant, OperatorType.CONV_FWD)

    def is_config_valid(self, config: ConvKernelConfig, datatype: str = "fp16") -> bool:
        """Validate configuration against architecture constraints"""
        if not self.arch_filter or not HAS_ARCH_FILTER:
            return True

        operator = self._get_operator_type(config.variant)

        return self.arch_filter.is_kernel_valid(
            datatype_a=datatype,
            datatype_b=datatype,
            datatype_c=datatype,
            tile_m=config.tile.tile_m,
            tile_n=config.tile.tile_n,
            tile_k=config.tile.tile_k,
            warp_m=config.tile.warp_m,
            warp_n=config.tile.warp_n,
            warp_k=1,  # Conv typically uses warp_k=1
            warp_tile_m=config.tile.warp_tile_m,
            warp_tile_n=config.tile.warp_tile_n,
            warp_tile_k=config.tile.warp_tile_k,
            pipeline=config.trait.pipeline,
            epilogue=config.trait.epilogue,
            scheduler=config.trait.scheduler,
            operator=operator,
        )

    def generate_kernel(
        self,
        config: ConvKernelConfig,
        datatype: str,
        variant: ConvVariant = ConvVariant.FORWARD,
    ) -> Path:
        """Generate a single kernel file and dispatcher wrapper"""
        kernel_gen = CKTileConvKernelGenerator(datatype, variant)
        wrapper_gen = DispatcherWrapperGenerator(datatype, variant)

        kernel_name = config.name(datatype)
        filename = f"{kernel_name}.hpp"
        filepath = self.output_dir / filename

        # Generate kernel header
        content = kernel_gen.generate(config)
        filepath.write_text(content)
        self.generated_files.append(filepath)

        # Generate dispatcher wrapper
        wrapper_content = wrapper_gen.generate(config, filepath, self.output_dir)
        wrapper_path = self.wrapper_dir / f"dispatcher_wrapper_{kernel_name}.hpp"
        wrapper_path.write_text(wrapper_content)
        self.generated_wrappers.append(wrapper_path)

        # Generate .cpp compilation unit for per-kernel parallel builds
        cpp_filename = f"{kernel_name}.cpp"
        cpp_filepath = self.output_dir / cpp_filename
        cpp_content = f'''// SPDX-License-Identifier: MIT
// Auto-generated compilation unit for: {kernel_name}
// Enables per-kernel parallel compilation with make -j

#include "{filename}"

namespace ck_tile {{ namespace generated {{
    volatile bool _{kernel_name.replace("-", "_")}_loaded = true;
}} }}
'''
        cpp_filepath.write_text(cpp_content)

        log.info(f"Generated: {filename}")
        return filepath

    def generate_all(
        self,
        configs: List[ConvKernelConfig],
        datatypes: List[str],
        parallel: bool = True,
    ) -> List[Path]:
        """Generate all kernel files (optionally in parallel)

        Configs are filtered using architecture validation before generation.
        """
        # Filter configs using arch validation
        valid_tasks = []
        rejected_count = 0

        for datatype in datatypes:
            for config in configs:
                if self.is_config_valid(config, datatype):
                    valid_tasks.append((config, datatype, config.variant))
                else:
                    rejected_count += 1
                    log.debug(
                        f"Rejected config for {self.gpu_target}: "
                        f"{config.tile.tile_m}x{config.tile.tile_n}x{config.tile.tile_k} "
                        f"variant={config.variant.value}"
                    )

        if rejected_count > 0:
            log.info(
                f"Filtered {rejected_count} configs for {self.gpu_target}, "
                f"{len(valid_tasks)} remaining"
            )

        if parallel and len(valid_tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.generate_kernel, config, dtype, variant)
                    for config, dtype, variant in valid_tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Collect results
                    except Exception as e:
                        log.error(f"Failed to generate kernel: {e}")
        else:
            for config, dtype, variant in valid_tasks:
                self.generate_kernel(config, dtype, variant)

        # Generate include_all_*.hpp headers for Python ctypes libraries
        self._generate_include_all_headers()

        return self.generated_files

    def _generate_include_all_headers(self):
        """Generate include_all_conv_*.hpp headers and registration header"""
        # Scan output directory for ALL kernel files (not just this run's generated_files)
        # This handles the case where fwd and bwd kernels are generated in separate make targets
        fwd_headers = []
        bwdd_headers = []
        bwdw_headers = []
        fwd_kernels = []
        bwdd_kernels = []
        bwdw_kernels = []

        for filepath in self.output_dir.glob("conv_*.hpp"):
            name = filepath.name
            kernel_name = name[:-4]  # Remove .hpp
            if name.startswith("conv_fwd_"):
                fwd_headers.append(name)
                fwd_kernels.append(kernel_name)
            elif name.startswith("conv_bwdd_"):
                bwdd_headers.append(name)
                bwdd_kernels.append(kernel_name)
            elif name.startswith("conv_bwdw_"):
                bwdw_headers.append(name)
                bwdw_kernels.append(kernel_name)

        # Generate include_all headers (for simple include-all usage)
        headers_to_generate = [
            ("include_all_conv_fwd_kernels.hpp", fwd_headers, "forward"),
            ("include_all_conv_bwdd_kernels.hpp", bwdd_headers, "backward data"),
            ("include_all_conv_bwdw_kernels.hpp", bwdw_headers, "backward weight"),
        ]

        for header_name, kernel_headers, variant_desc in headers_to_generate:
            header_path = self.output_dir / header_name
            includes = "\n".join(f'#include "{h}"' for h in sorted(kernel_headers))

            # Pick the first kernel as the default Selected*Launcher
            if kernel_headers:
                first_kernel = sorted(kernel_headers)[0][:-4]  # Remove .hpp
                if variant_desc == "forward":
                    launcher_alias = (
                        f"using SelectedConvKernelLauncher = {first_kernel}_Launcher;"
                    )
                elif variant_desc == "backward data":
                    launcher_alias = (
                        f"using SelectedConvBwdDataLauncher = {first_kernel}_Launcher;"
                    )
                else:  # backward weight
                    launcher_alias = f"using SelectedConvBwdWeightLauncher = {first_kernel}_Launcher;"
            else:
                launcher_alias = "// No kernels generated for this variant"

            content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated header for conv {variant_desc} kernels
#pragma once

{includes}

// Default launcher alias (uses first kernel)
{launcher_alias}
"""
            header_path.write_text(content)
            if kernel_headers:
                log.info(f"Generated: {header_name} ({len(kernel_headers)} kernels)")

        # Generate registration header (following GEMM pattern)
        self._generate_registration_header(fwd_kernels, bwdd_kernels, bwdw_kernels)

    def _generate_registration_header(
        self, fwd_kernels: List[str], bwdd_kernels: List[str], bwdw_kernels: List[str]
    ):
        """Generate master registration header for all conv kernels"""
        # Scan wrapper directory for ALL wrapper files
        all_wrappers = []
        for wrapper_path in self.wrapper_dir.glob("dispatcher_wrapper_conv_*.hpp"):
            all_wrappers.append(wrapper_path.name)

        wrapper_includes = "\n".join(f'#include "{w}"' for w in sorted(all_wrappers))

        # Generate registration calls
        fwd_registrations = "\n        ".join(
            f"registry.register_kernel(generated::make_{k}(gfx_arch), priority);"
            for k in sorted(fwd_kernels)
        )
        bwdd_registrations = "\n        ".join(
            f"registry.register_kernel(generated::make_{k}(gfx_arch), priority);"
            for k in sorted(bwdd_kernels)
        )
        bwdw_registrations = "\n        ".join(
            f"registry.register_kernel(generated::make_{k}(gfx_arch), priority);"
            for k in sorted(bwdw_kernels)
        )

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated master registration header for conv kernels
#pragma once

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/conv_utils.hpp"

{wrapper_includes}

namespace ck_tile {{
namespace dispatcher {{

using Priority = ConvRegistry::Priority;

inline void register_all_conv_fwd_kernels(
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
    auto& registry = ConvRegistry::instance();
    {fwd_registrations if fwd_registrations else "// No forward kernels"}
}}

inline void register_all_conv_bwdd_kernels(
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
    auto& registry = ConvRegistry::instance();
    {bwdd_registrations if bwdd_registrations else "// No backward data kernels"}
}}

inline void register_all_conv_bwdw_kernels(
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
    auto& registry = ConvRegistry::instance();
    {bwdw_registrations if bwdw_registrations else "// No backward weight kernels"}
}}

inline void register_all_conv_kernels(
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
    register_all_conv_fwd_kernels(gfx_arch, priority);
    register_all_conv_bwdd_kernels(gfx_arch, priority);
    register_all_conv_bwdw_kernels(gfx_arch, priority);
}}

inline std::size_t get_conv_fwd_kernel_count() {{ return {len(fwd_kernels)}; }}
inline std::size_t get_conv_bwdd_kernel_count() {{ return {len(bwdd_kernels)}; }}
inline std::size_t get_conv_bwdw_kernel_count() {{ return {len(bwdw_kernels)}; }}
inline std::size_t get_conv_kernel_count() {{ return {len(fwd_kernels) + len(bwdd_kernels) + len(bwdw_kernels)}; }}

}}  // namespace dispatcher
}}  // namespace ck_tile
"""
        reg_path = self.wrapper_dir / "register_all_conv_kernels.hpp"
        reg_path.write_text(content)
        log.info(f"Generated registration header: {reg_path}")


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

    # Individual kernel configuration (when not using predefined configs)
    parser.add_argument("--tile-m", type=int, help="Block tile M dimension")
    parser.add_argument("--tile-n", type=int, help="Block tile N dimension")
    parser.add_argument("--tile-k", type=int, help="Block tile K dimension")
    parser.add_argument("--warp-m", type=int, help="Wave distribution M")
    parser.add_argument("--warp-n", type=int, help="Wave distribution N")
    parser.add_argument("--warp-k", type=int, default=1, help="Wave distribution K")
    parser.add_argument("--warp-tile-m", type=int, help="Warp tile M")
    parser.add_argument("--warp-tile-n", type=int, help="Warp tile N")
    parser.add_argument("--warp-tile-k", type=int, default=16, help="Warp tile K")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["mem", "compv3", "compv4", "compv5"],
        help="Pipeline type",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["intrawave", "interwave"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--epilogue",
        type=str,
        default="cshuffle",
        choices=["cshuffle", "default"],
        help="Epilogue type",
    )
    parser.add_argument("--pad-m", type=bool, default=True, help="Pad M dimension")
    parser.add_argument("--pad-n", type=bool, default=True, help="Pad N dimension")
    parser.add_argument("--pad-k", type=bool, default=True, help="Pad K dimension")
    parser.add_argument("--vector-a", type=int, default=4, help="Vector size A")
    parser.add_argument("--vector-b", type=int, default=8, help="Vector size B")
    parser.add_argument("--vector-c", type=int, default=8, help="Vector size C")
    parser.add_argument("--block-per-cu", type=int, default=1, help="Blocks per CU")
    parser.add_argument("--num-wave-groups", type=int, default=1, help="Wave groups")
    parser.add_argument(
        "--num-groups-to-merge", type=int, default=1, help="Groups to merge"
    )
    parser.add_argument(
        "--double-smem-buffer",
        type=str,
        default=None,
        help="Double SMEM buffer (true/false)",
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

    # Check if user specified custom configuration
    custom_config = (
        args.tile_m is not None or args.tile_n is not None or args.pipeline is not None
    )

    if custom_config:
        # Build custom config from CLI arguments
        tile = TileConfig(
            tile_m=args.tile_m or 128,
            tile_n=args.tile_n or 128,
            tile_k=args.tile_k or 64,
            warp_m=args.warp_m or 2,
            warp_n=args.warp_n or 2,
            warp_k=args.warp_k or 1,
            warp_tile_m=args.warp_tile_m or 32,
            warp_tile_n=args.warp_tile_n or 32,
            warp_tile_k=args.warp_tile_k or 16,
        )
        pipeline = args.pipeline or "compv4"
        # Determine double_smem_buffer: use CLI arg if given, else default based on pipeline
        if args.double_smem_buffer is not None:
            dsb = args.double_smem_buffer.lower() == "true"
        else:
            dsb = pipeline == "compv4"  # compv4 requires double buffer

        trait = TraitConfig(
            pipeline=pipeline,
            scheduler=args.scheduler or "intrawave",
            epilogue=args.epilogue or "cshuffle",
            double_smem_buffer=dsb,
            pad_m=args.pad_m,
            pad_n=args.pad_n,
            pad_k=args.pad_k,
            num_groups_to_merge=args.num_groups_to_merge,
        )
        config = ConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=requested_variants[0]
            if requested_variants
            else ConvVariant.FORWARD,
            ndim_spatial=args.ndim[0] if args.ndim else 2,
            arch=args.arch,
            vector_size_a=args.vector_a,
            vector_size_b=args.vector_b,
            vector_size_c=args.vector_c,
            block_per_cu=args.block_per_cu,
            num_wave_groups=args.num_wave_groups,
        )
        filtered_configs = [config]
    else:
        # Get predefined configurations for target arch with requested variants and ndims
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
    codegen = UnifiedConvCodegen(
        output_dir=args.output,
        gpu_target=args.arch,
        enable_arch_filter=True,
    )
    files = codegen.generate_all(filtered_configs, args.datatype)

    print(
        f"\nGenerated {len(files)} convolution kernel files for {args.arch} in {args.output}"
    )


if __name__ == "__main__":
    main()
