#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unified GEMM Code Generator - Single Source of Truth

This is THE unified code generator for all GEMM kernel variants:
- Standard GEMM (C = A × B)
- Preshuffle GEMM (optimized weight access)
- Multi-D GEMM (element-wise fusion)

Generates both CK Tile kernels AND dispatcher wrappers in one pass.
Replaces all tile_engine GEMM codegen.
"""

import json
import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures

# Import architecture filter for GPU-specific validation
try:
    from arch_filter import ArchFilter, KernelConfig as ArchKernelConfig

    HAS_ARCH_FILTER = True
except ImportError:
    HAS_ARCH_FILTER = False
    ArchFilter = None
    ArchKernelConfig = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

log = logging.getLogger(__name__)


# ============================================================================
# Configuration and Data Structures
# ============================================================================


class GemmVariant(Enum):
    """GEMM kernel variants"""

    STANDARD = "standard"
    PRESHUFFLE = "preshuffle"
    MULTI_D = "multi_d"


@dataclass
class TileConfig:
    """Tile configuration parameters"""

    tile_m: int
    tile_n: int
    tile_k: int
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

    pipeline: str  # mem, compv3, compv4
    epilogue: str  # default, cshuffle
    scheduler: str  # intrawave, interwave
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool

    def is_valid(self) -> bool:
        """Check if trait combination is valid"""
        # Unsupported combinations
        unsupported = {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave"),
        }
        return (self.pipeline, self.epilogue, self.scheduler) not in unsupported


@dataclass
class KernelConfig:
    """Complete kernel configuration"""

    tile: TileConfig
    trait: TraitConfig
    variant: GemmVariant = GemmVariant.STANDARD

    # Variant-specific
    preshuffle: bool = False
    elementwise_op: str = "PassThrough"
    num_d_tensors: int = 0

    # Fixed parameters
    block_size: int = 256
    k_block_per_cu: int = 1
    num_wave_groups: int = 1

    def name(self, datatype: str, layout: str) -> str:
        """C++ alias for template instance"""
        return f"ck_tile_gemm_{self.key_name(datatype, layout)}"

    def key_name(self, datatype: str, layout: str) -> str:
        """Unique identifier for this kernel configuration"""
        parts = []
        parts.append(f"dt_{datatype}")
        parts.append(f"ly_{layout}")
        parts.append(f"tile_{self.tile.tile_m}x{self.tile.tile_n}x{self.tile.tile_k}")
        parts.append(f"warp_{self.tile.warp_m}x{self.tile.warp_n}x{self.tile.warp_k}")
        parts.append(
            f"wtile_{self.tile.warp_tile_m}x{self.tile.warp_tile_n}x{self.tile.warp_tile_k}"
        )
        parts.append(f"pipe_{self.trait.pipeline}")
        parts.append(f"epi_{self.trait.epilogue}")
        parts.append(f"sched_{self.trait.scheduler}")
        if self.trait.persistent:
            parts.append("persist")
        if self.preshuffle:
            parts.append("preshuffle")
        if self.variant == GemmVariant.MULTI_D:
            parts.append(f"ew_{self.elementwise_op}_d{self.num_d_tensors}")
        return "_".join(parts)

    def dict_items(self):
        """Iterator over (field, value) pairs"""
        return asdict(self).items()


# ============================================================================
# Type Mappings
# ============================================================================


class TypeMappings:
    """Centralized type mappings for code generation"""

    DTYPE_TO_CK = {
        "fp16": "fp16_t",
        "bf16": "bf16_t",
        "fp32": "float",
        "fp8": "fp8_t",
        "bf8": "bf8_t",
        "int8": "int8_t",
    }

    DTYPE_TO_DISPATCHER = {
        "fp16": "DataType::FP16",
        "bf16": "DataType::BF16",
        "fp32": "DataType::FP32",
        "fp8": "DataType::FP8",
        "bf8": "DataType::BF8",
        "int8": "DataType::INT8",
    }

    LAYOUT_TO_CK = {
        "r": "tensor_layout::gemm::RowMajor",
        "c": "tensor_layout::gemm::ColumnMajor",
    }

    LAYOUT_TO_DISPATCHER = {
        "r": "LayoutTag::RowMajor",
        "c": "LayoutTag::ColMajor",
    }

    PIPELINE_TO_CK = {
        "mem": "GemmPipelineAgBgCrMem",
        "compv3": "GemmPipelineAgBgCrCompV3",
        "compv4": "GemmPipelineAgBgCrCompV4",
    }

    PIPELINE_TO_BASE = {
        "mem": "BaseGemmPipelineAgBgCrMem",
        "compv3": "BaseGemmPipelineAgBgCrCompV3",
        "compv4": "BaseGemmPipelineAgBgCrCompV4",
    }

    PIPELINE_TO_DISPATCHER = {
        "mem": "Pipeline::Mem",
        "compv3": "Pipeline::CompV3",
        "compv4": "Pipeline::CompV4",
    }

    SCHEDULER_TO_CK = {
        "intrawave": "GemmPipelineScheduler::Intrawave",
        "interwave": "GemmPipelineScheduler::Interwave",
        "default": "GemmPipelineScheduler::Default",
    }

    SCHEDULER_TO_DISPATCHER = {
        "intrawave": "Scheduler::Intrawave",
        "interwave": "Scheduler::Interwave",
        "default": "Scheduler::Auto",
    }

    EPILOGUE_TO_DISPATCHER = {
        "cshuffle": "Epilogue::CShuffle",
        "default": "Epilogue::Default",
    }

    @staticmethod
    def get_output_dtype(dtype: str) -> str:
        """Get output datatype (fp8/bf8 -> fp16)"""
        return "fp16" if dtype in ["fp8", "bf8"] else dtype


# ============================================================================
# Kernel Name Generator
# ============================================================================


class KernelNaming:
    """Unified kernel naming"""

    @staticmethod
    def generate(config: KernelConfig, datatype: str, layout: str) -> str:
        """Generate kernel name following tile_engine convention"""
        t = config.tile
        tr = config.trait

        name = f"gemm_{datatype}_{layout}_{tr.pipeline}_{tr.epilogue}_{tr.scheduler}"
        name += f"_{str(tr.pad_m).capitalize()}_{str(tr.pad_n).capitalize()}"
        name += f"_{str(tr.pad_k).capitalize()}_{str(tr.persistent).capitalize()}"
        name += f"_{t.tile_m}x{t.tile_n}x{t.tile_k}"
        name += f"_{t.warp_m}x{t.warp_n}x{t.warp_k}"
        name += f"_{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}"

        # Add variant suffix
        if config.variant == GemmVariant.PRESHUFFLE:
            name += "_preshuffle"
        elif config.variant == GemmVariant.MULTI_D:
            name += f"_multid_{config.elementwise_op}_d{config.num_d_tensors}"

        return name


# ============================================================================
# CK Tile Kernel Generator
# ============================================================================


class CKTileKernelGenerator:
    """Generates CK Tile kernel instance code"""

    def __init__(self, datatype: str, layout: str):
        self.datatype = datatype
        self.layout = layout
        self.tm = TypeMappings()

    def generate(self, config: KernelConfig) -> str:
        """Generate complete CK Tile kernel"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)

        return f"""{self._header(kernel_name, config)}
{self._types(config, kernel_name)}
{self._selected_kernel_struct(config, kernel_name)}
"""

    def _header(self, kernel_name: str, config: KernelConfig) -> str:
        """Generate header includes"""
        includes = """// SPDX-License-Identifier: MIT
// Auto-generated CK Tile GEMM kernel
#pragma once

#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

"""

        if config.variant == GemmVariant.MULTI_D:
            includes += (
                '\n#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"'
            )

        return includes

    def _types(self, config: KernelConfig, kernel_name: str) -> str:
        """Generate type definitions"""
        output_dtype = self.tm.get_output_dtype(self.datatype)

        types = f"""
// Use ck_tile namespace for generated code
using namespace ck_tile;

// Data types
using ADataType = {self.tm.DTYPE_TO_CK[self.datatype]};
using BDataType = {self.tm.DTYPE_TO_CK[self.datatype]};
using AccDataType = float;
using CDataType = {self.tm.DTYPE_TO_CK[output_dtype]};

// Layouts
using ALayout = {self.tm.LAYOUT_TO_CK[self.layout[0]]};
using BLayout = {self.tm.LAYOUT_TO_CK[self.layout[1]]};
using CLayout = {self.tm.LAYOUT_TO_CK[self.layout[2]]};
"""

        if config.variant == GemmVariant.MULTI_D:
            d_types = ", ".join(["CDataType"] * config.num_d_tensors)
            d_layouts = ", ".join(["CLayout"] * config.num_d_tensors)
            types += f"""
// Multi-D types
using DsDataType = tuple<{d_types}>;
using DsLayout = tuple<{d_layouts}>;
using ElementWiseFn = element_wise::{config.elementwise_op};
"""

        return types

    def _selected_kernel_struct(self, config: KernelConfig, kernel_name: str) -> str:
        """Generate SelectedKernel struct with unique name"""
        t = config.tile
        tr = config.trait

        # Generate unique struct name from kernel name
        struct_name = f"Kernel_{kernel_name}"

        return f"""
constexpr const char* KERNEL_NAME = "{kernel_name}";

struct {struct_name} {{
    // Data types (required by backend as member types)
    using ADataType = ::ADataType;
    using BDataType = ::BDataType;
    using CDataType = ::CDataType;
    using AccDataType = ::AccDataType;
    
    // Configuration
    static constexpr index_t BlockSize = {config.block_size};
    static constexpr index_t TileM = {t.tile_m};
    static constexpr index_t TileN = {t.tile_n};
    static constexpr index_t TileK = {t.tile_k};
    static constexpr index_t WarpPerBlock_M = {t.warp_m};
    static constexpr index_t WarpPerBlock_N = {t.warp_n};
    static constexpr index_t WarpPerBlock_K = {t.warp_k};
    static constexpr index_t WarpTileM = {t.warp_tile_m};
    static constexpr index_t WarpTileN = {t.warp_tile_n};
    static constexpr index_t WarpTileK = {t.warp_tile_k};
    
    // Traits
    static constexpr bool kPadM = {str(tr.pad_m).lower()};
    static constexpr bool kPadN = {str(tr.pad_n).lower()};
    static constexpr bool kPadK = {str(tr.pad_k).lower()};
    static constexpr bool TransposeC = false;
    static constexpr bool UsePersistentKernel = {str(tr.persistent).lower()};
    static constexpr bool DoubleSmemBuffer = {str(tr.pipeline == "compv4").lower()};
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle = {str(config.preshuffle).lower()};
    static constexpr index_t NumWaveGroups = {config.num_wave_groups};
    
    {self._tile_types(config)}
    {self._launch_function(config)}
}};

// Alias for tile_engine style compatibility (when used with -include)
using SelectedKernel = {struct_name};
"""

    def _tile_types(self, config: KernelConfig) -> str:
        """Generate tile type definitions"""
        return (
            """// Tile shape
    using TileShape = TileGemmShape<
        sequence<TileM, TileN, TileK>,
        sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;
    
    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits = TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, NumWaveGroups>;
    using GemmPipelineProblem = GemmPipelineProblem<ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = """
            + self.tm.PIPELINE_TO_BASE[config.trait.pipeline]
            + """<GemmPipelineProblem>;"""
        )

    def _launch_function(self, config: KernelConfig) -> str:
        """Generate launch function"""
        return f"""
    static float launch(const GemmHostArgs& args, const stream_config& stream) {{
        const index_t k_grain = args.k_batch * TileK;
        const index_t K_split = (args.K + k_grain - 1) / k_grain * TileK;
        const index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;
            constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};
            [[maybe_unused]] constexpr auto memory_operation = memory_operation_.value;
            
            using UniversalGemmProblem = UniversalGemmPipelineProblem<
                ADataType, BDataType, AccDataType, TileShape,
                TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, CLayout, TransposeC,
                                                UseStructuredSparsity, UsePersistentKernel,
                                                NumWaveGroups, Preshuffle>,
                scheduler, has_hot_loop_v, tail_number_v>;
            
            using GemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<UniversalGemmProblem>;
            {self._epilogue_code(config)}
            
            using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Arguments not supported!");
            }}
            
            const dim3 grids = {"GemmKernel::MaxOccupancyGridSize(stream)" if config.trait.persistent else "GemmKernel::GridSize(args.M, args.N, args.k_batch)"};
            const dim3 blocks = GemmKernel::BlockSize();
            
            constexpr int kBlockPerCu = {config.k_block_per_cu};
            ave_time = launch_kernel(stream,
                make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};
        
        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {{
            if(args.k_batch == 1) {{
                Run(has_hot_loop_,
                    tail_number_,
                    integral_constant<memory_operation_enum,
                                            memory_operation_enum::set>{{}});
            }} else {{
                Run(has_hot_loop_,
                    tail_number_,
                    integral_constant<memory_operation_enum,
                                            memory_operation_enum::atomic_add>{{}});
            }}
        }};

        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        return ave_time;
    }}"""

    def _epilogue_code(self, config: KernelConfig) -> str:
        """Generate epilogue code"""
        if config.variant == GemmVariant.MULTI_D:
            return """
            using EpilogueProblem = CShuffleEpilogueProblem<
                ADataType, BDataType, DsDataType, AccDataType, CDataType,
                DsLayout, CLayout, ElementWiseFn,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
                TransposeC, memory_operation, NumWaveGroups>;
            using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>;"""
        elif config.trait.epilogue == "cshuffle":
            return """
            using EpilogueProblem = CShuffleEpilogueProblem<
                ADataType, BDataType, tuple<>, AccDataType, CDataType,
                tuple<>, CLayout, element_wise::PassThrough,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
                TransposeC, memory_operation, NumWaveGroups>;
            using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>;"""
        else:
            return """
            using EpilogueProblem = DefaultGemm2DEpilogueProblem<
                ADataType, BDataType, tuple<>, AccDataType, CDataType,
                tuple<>, CLayout, element_wise::PassThrough,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                kPadM, kPadN, WarpTileM, WarpTileN, WarpTileK, TransposeC>;
            using GemmEpilogue = DefaultGemm2DEpilogue<EpilogueProblem>;"""


# ============================================================================
# Dispatcher Wrapper Generator
# ============================================================================


class DispatcherWrapperGenerator:
    """Generates dispatcher wrapper code"""

    def __init__(self, datatype: str, layout: str):
        self.datatype = datatype
        self.layout = layout
        self.tm = TypeMappings()

    def generate(
        self, config: KernelConfig, kernel_path: Path, output_dir: Path
    ) -> str:
        """Generate dispatcher wrapper"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)
        output_dtype = self.tm.get_output_dtype(self.datatype)
        rel_path = kernel_path.relative_to(output_dir)

        return f"""// SPDX-License-Identifier: MIT
// Auto-generated dispatcher wrapper
#pragma once

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/backends/generated_kernel_backend.hpp"
#include "{rel_path}"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

using ::ck_tile::dispatcher::KernelInstancePtr;
using ::ck_tile::dispatcher::KernelKey;
using ::ck_tile::dispatcher::DataType;
using ::ck_tile::dispatcher::LayoutTag;
using ::ck_tile::dispatcher::Pipeline;
using ::ck_tile::dispatcher::Scheduler;
using ::ck_tile::dispatcher::Epilogue;
using Priority = ::ck_tile::dispatcher::Registry::Priority;
namespace backends = ::ck_tile::dispatcher::backends;

inline KernelInstancePtr make_{kernel_name}(const std::string& gfx_arch = "gfx942") {{
    // Use the unique kernel struct name
    using KernelStruct = Kernel_{kernel_name};
    
    KernelKey key;
    
    // Signature
    key.signature.dtype_a = {self.tm.DTYPE_TO_DISPATCHER[self.datatype]};
    key.signature.dtype_b = {self.tm.DTYPE_TO_DISPATCHER[self.datatype]};
    key.signature.dtype_c = {self.tm.DTYPE_TO_DISPATCHER[output_dtype]};
    key.signature.dtype_acc = DataType::FP32;
    key.signature.layout_a = {self.tm.LAYOUT_TO_DISPATCHER[self.layout[0]]};
    key.signature.layout_b = {self.tm.LAYOUT_TO_DISPATCHER[self.layout[1]]};
    key.signature.layout_c = {self.tm.LAYOUT_TO_DISPATCHER[self.layout[2]]};
    key.signature.transpose_a = false;
    key.signature.transpose_b = false;
    key.signature.grouped = false;
    key.signature.split_k = 1;
    key.signature.elementwise_op = "{config.elementwise_op}";
    key.signature.num_d_tensors = {config.num_d_tensors};
    key.signature.structured_sparsity = false;
    
    // Algorithm
    key.algorithm.tile_shape = {{{config.tile.tile_m}, {config.tile.tile_n}, {config.tile.tile_k}}};
    key.algorithm.wave_shape = {{{config.tile.warp_m}, {config.tile.warp_n}, {config.tile.warp_k}}};
    key.algorithm.warp_tile_shape = {{{config.tile.warp_tile_m}, {config.tile.warp_tile_n}, {config.tile.warp_tile_k}}};
    key.algorithm.pipeline = {self.tm.PIPELINE_TO_DISPATCHER[config.trait.pipeline]};
    key.algorithm.scheduler = {self.tm.SCHEDULER_TO_DISPATCHER[config.trait.scheduler]};
    key.algorithm.epilogue = {self.tm.EPILOGUE_TO_DISPATCHER[config.trait.epilogue]};
    key.algorithm.block_size = {config.block_size};
    key.algorithm.double_buffer = {str(config.trait.pipeline == "compv4").lower()};
    key.algorithm.persistent = {str(config.trait.persistent).lower()};
    key.algorithm.preshuffle = {str(config.preshuffle).lower()};
    key.algorithm.transpose_c = false;
    key.algorithm.num_wave_groups = {config.num_wave_groups};
    
    key.gfx_arch = gfx_arch;
    
    return std::make_shared<backends::GeneratedKernelInstance<KernelStruct>>(key, "{kernel_name}");
}}

}}}}}}
"""


# ============================================================================
# Main Unified Generator
# ============================================================================


class UnifiedGemmCodegen:
    """Unified GEMM code generator - single entry point"""

    def __init__(
        self,
        output_dir: Path,
        datatype: str,
        layout: str,
        gpu_target: str = "gfx942",
        config_file: Optional[Path] = None,
        variants: List[GemmVariant] = None,
        use_preselected: Optional[str] = None,
        enable_arch_filter: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.datatype = datatype
        self.layout = layout
        self.gpu_target = gpu_target
        self.variants = variants or [GemmVariant.STANDARD]
        self.use_preselected = use_preselected

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wrapper_dir = self.output_dir / "dispatcher_wrappers"
        self.wrapper_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize architecture filter for GPU-specific validation
        self.arch_filter = None
        if enable_arch_filter and HAS_ARCH_FILTER:
            try:
                self.arch_filter = ArchFilter(gpu_target, strict_mode=False)
                log.info(f"Architecture filter enabled for {gpu_target}")
            except ValueError as e:
                log.warning(f"Could not create arch filter: {e}")

        # Initialize generators
        self.ck_gen = CKTileKernelGenerator(datatype, layout)
        self.disp_gen = DispatcherWrapperGenerator(datatype, layout)

    def _load_config(self, config_file: Optional[Path]) -> Dict:
        """Load or create default configuration"""
        if config_file and config_file.exists():
            with open(config_file) as f:
                return json.load(f)

        return {
            "tile_config": {
                "tile_m": [128, 256],
                "tile_n": [128, 256],
                "tile_k": [32, 64],
                "warp_m": [2, 4],
                "warp_n": [2, 4],
                "warp_k": [1],
                "warp_tile_m": [16, 32],
                "warp_tile_n": [16, 32],
                "warp_tile_k": [16],
            },
            "trait_config": {
                "pipeline": ["compv3", "compv4"],
                "epilogue": ["cshuffle", "default"],
                "scheduler": ["intrawave"],
                "pad_m": [False],
                "pad_n": [False],
                "pad_k": [False],
                "persistent": [False, True],
            },
            "multi_d_config": {
                "elementwise_ops": ["MultiDAdd", "MultiDMultiply", "Relu", "Gelu"],
                "num_d_tensors": [1, 2],
            },
        }

    def generate_all(self, parallel: bool = True) -> Dict:
        """Generate all kernels"""
        log.info("Generating GEMM kernels:")
        log.info(f"  Datatype: {self.datatype}")
        log.info(f"  Layout: {self.layout}")
        log.info(f"  Variants: {[v.value for v in self.variants]}")
        if self.use_preselected:
            log.info(f"  Using preselected set: {self.use_preselected}")

        results = {"kernels": [], "wrappers": [], "failed": []}

        # Get configurations
        if self.use_preselected:
            configs = self._get_preselected_configs()
            log.info(f"  Total configurations: {len(configs)}")
        else:
            for variant in self.variants:
                log.info(f"\nGenerating {variant.value} kernels...")
                configs = self._get_configs_for_variant(variant)
                log.info(f"  Configurations: {len(configs)}")

                if parallel:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(self._generate_one, cfg) for cfg in configs
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                k, w = future.result()
                                results["kernels"].append(k)
                                results["wrappers"].append(w)
                            except Exception as e:
                                results["failed"].append(str(e))
                                log.error(f"Failed: {e}")
                else:
                    for cfg in configs:
                        try:
                            k, w = self._generate_one(cfg)
                            results["kernels"].append(k)
                            results["wrappers"].append(w)
                        except Exception as e:
                            results["failed"].append(str(e))
                            log.error(f"Failed: {e}")

            # Generate registration header
            if results["wrappers"]:
                self._generate_registration_header(results["wrappers"])

            return results

        # Generate from preselected set
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._generate_one, cfg) for cfg in configs]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        k, w = future.result()
                        results["kernels"].append(k)
                        results["wrappers"].append(w)
                    except Exception as e:
                        results["failed"].append(str(e))
                        log.error(f"Failed: {e}")
        else:
            for cfg in configs:
                try:
                    k, w = self._generate_one(cfg)
                    results["kernels"].append(k)
                    results["wrappers"].append(w)
                except Exception as e:
                    results["failed"].append(str(e))
                    log.error(f"Failed: {e}")

        # Generate registration header
        if results["wrappers"]:
            self._generate_registration_header(results["wrappers"])

        return results

    def _get_preselected_configs(self) -> List[KernelConfig]:
        """Get preselected kernel configurations"""
        try:
            from preselected_kernels import get_preselected_set

            return get_preselected_set(self.use_preselected)
        except ImportError:
            log.warning(
                "preselected_kernels module not found, falling back to config-based generation"
            )
            return []
        except ValueError as e:
            log.error(f"Invalid preselected set: {e}")
            return []

    def _get_configs_for_variant(self, variant: GemmVariant) -> List[KernelConfig]:
        """Get all configurations for a variant"""
        configs = []

        # Get base configs
        tile_configs = self._get_tile_configs()
        trait_configs = self._get_trait_configs()

        for tile, trait in itertools.product(tile_configs, trait_configs):
            if variant == GemmVariant.STANDARD:
                configs.append(KernelConfig(tile=tile, trait=trait, variant=variant))

            elif variant == GemmVariant.PRESHUFFLE:
                configs.append(
                    KernelConfig(
                        tile=tile, trait=trait, variant=variant, preshuffle=True
                    )
                )

            elif variant == GemmVariant.MULTI_D:
                multi_d = self.config.get("multi_d_config", {})
                for ew_op, num_d in itertools.product(
                    multi_d.get("elementwise_ops", ["MultiDAdd"]),
                    multi_d.get("num_d_tensors", [1]),
                ):
                    configs.append(
                        KernelConfig(
                            tile=tile,
                            trait=trait,
                            variant=variant,
                            elementwise_op=ew_op,
                            num_d_tensors=num_d,
                        )
                    )

        return configs

    def _get_tile_configs(self) -> List[TileConfig]:
        """Get valid tile configurations, filtered by architecture constraints"""
        tc = self.config["tile_config"]
        configs = []
        rejected_count = 0

        for params in itertools.product(
            tc["tile_m"],
            tc["tile_n"],
            tc["tile_k"],
            tc["warp_m"],
            tc["warp_n"],
            tc["warp_k"],
            tc["warp_tile_m"],
            tc["warp_tile_n"],
            tc["warp_tile_k"],
        ):
            tile = TileConfig(*params)

            # Basic validation
            if not tile.is_valid():
                rejected_count += 1
                continue

            # Architecture-specific validation
            if self.arch_filter and HAS_ARCH_FILTER:
                if not self._is_tile_arch_valid(tile):
                    rejected_count += 1
                    continue

            configs.append(tile)

        if rejected_count > 0:
            log.debug(f"Rejected {rejected_count} tile configs for {self.gpu_target}")

        return configs

    def _is_tile_arch_valid(self, tile: TileConfig) -> bool:
        """Check if tile configuration is valid for target architecture"""
        if not self.arch_filter or not HAS_ARCH_FILTER:
            return True

        # Determine data types based on self.datatype
        dtype_map = {
            "fp16": ("fp16", "fp16", "fp16"),
            "bf16": ("bf16", "bf16", "bf16"),
            "fp8": ("fp8", "fp8", "fp16"),
            "bf8": ("bf8", "bf8", "fp16"),
            "int8": ("int8", "int8", "int32"),
        }
        dtype_a, dtype_b, dtype_c = dtype_map.get(
            self.datatype, ("fp16", "fp16", "fp16")
        )

        return self.arch_filter.is_kernel_valid(
            datatype_a=dtype_a,
            datatype_b=dtype_b,
            datatype_c=dtype_c,
            tile_m=tile.tile_m,
            tile_n=tile.tile_n,
            tile_k=tile.tile_k,
            warp_m=tile.warp_m,
            warp_n=tile.warp_n,
            warp_k=tile.warp_k,
            warp_tile_m=tile.warp_tile_m,
            warp_tile_n=tile.warp_tile_n,
            warp_tile_k=tile.warp_tile_k,
            layout=self.layout,
        )

    def _get_trait_configs(self) -> List[TraitConfig]:
        """Get valid trait configurations, filtered by architecture constraints"""
        tc = self.config["trait_config"]
        configs = []
        rejected_count = 0

        for params in itertools.product(
            tc["pipeline"],
            tc["epilogue"],
            tc["scheduler"],
            tc["pad_m"],
            tc["pad_n"],
            tc["pad_k"],
            tc["persistent"],
        ):
            trait = TraitConfig(*params)

            # Basic trait validation (unsupported combinations)
            if not trait.is_valid():
                rejected_count += 1
                continue

            configs.append(trait)

        if rejected_count > 0:
            log.debug(f"Rejected {rejected_count} trait configs")

        return configs

    def _generate_one(self, config: KernelConfig) -> Tuple[str, str]:
        """Generate one kernel and wrapper"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)

        # Generate CK Tile kernel
        kernel_code = self.ck_gen.generate(config)
        kernel_path = self.output_dir / f"{kernel_name}.hpp"
        kernel_path.write_text(kernel_code)

        # Generate dispatcher wrapper
        wrapper_code = self.disp_gen.generate(config, kernel_path, self.output_dir)
        wrapper_path = self.wrapper_dir / f"dispatcher_wrapper_{kernel_name}.hpp"
        wrapper_path.write_text(wrapper_code)

        return str(kernel_path), str(wrapper_path)

    def _generate_registration_header(self, wrapper_paths: List[str]):
        """Generate master registration header"""
        kernel_names = [
            Path(w).stem.replace("dispatcher_wrapper_", "") for w in wrapper_paths
        ]

        includes = "\n".join(
            [f'#include "dispatcher_wrapper_{n}.hpp"' for n in kernel_names]
        )
        registrations = "\n        ".join(
            [
                f"registry.register_kernel(generated::make_{n}(gfx_arch), priority);"
                for n in kernel_names
            ]
        )

        content = f"""// SPDX-License-Identifier: MIT
// Auto-generated master registration
#pragma once

#include "ck_tile/dispatcher.hpp"
{includes}

namespace ck_tile {{
namespace dispatcher {{

using ::ck_tile::dispatcher::Registry;
using Priority = ::ck_tile::dispatcher::Registry::Priority;

inline void register_all_tile_gemm_kernels(
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
    auto& registry = Registry::instance();
    {registrations}
}}

inline std::size_t get_tile_gemm_kernel_count() {{ return {len(kernel_names)}; }}

}}}}
"""

        reg_path = self.wrapper_dir / "register_all_kernels.hpp"
        reg_path.write_text(content)
        logging.info(f"Generated registration header: {reg_path}")


# ============================================================================
# CLI
# ============================================================================


def _show_arch_info(gpu_target: str, datatype: str):
    """Display supported configurations for a GPU architecture"""
    if not HAS_ARCH_FILTER:
        print("Architecture filter module not available")
        return

    try:
        from arch_filter import (
            get_supported_archs,
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            LDS_CAPACITY_LIMITS,
            TRAIT_UNSUPPORTED_COMBINATIONS,
        )

        print(f"\n=== Architecture Info for {gpu_target} ===\n")

        # Supported architectures
        print(f"Supported GPUs: {get_supported_archs()}")

        # Warp configurations
        warp_cfgs = WARP_SUPPORTED_COMBINATIONS.get(gpu_target, [])
        print("\nWarp configurations [warp_m, warp_n, warp_k]:")
        for cfg in warp_cfgs:
            print(f"  {cfg}")

        # Warp tile configurations for data type
        dtype_map = {
            "fp16": "fp16_fp16_fp16",
            "bf16": "bf16_bf16_bf16",
            "fp8": "fp8_fp8_fp16",
            "bf8": "bf8_bf8_fp16",
            "int8": "int8_int8_int32",
        }
        dtype_key = dtype_map.get(datatype, "fp16_fp16_fp16")

        gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(gpu_target, {})
        warp_tiles = gpu_combos.get(dtype_key, [])
        print(
            f"\nWarp tile configurations for {dtype_key} [warp_tile_m, warp_tile_n, warp_tile_k]:"
        )
        for cfg in warp_tiles:
            print(f"  {cfg}")

        # All supported data types
        print(f"\nAll supported data types on {gpu_target}:")
        for dtype in gpu_combos.keys():
            print(f"  {dtype}")

        # LDS limits
        print("\nLDS capacity limits:")
        for pipeline, limit in LDS_CAPACITY_LIMITS.items():
            print(f"  {pipeline}: {limit // 1024}KB")

        # Unsupported trait combinations
        print("\nUnsupported trait combinations (pipeline, epilogue, scheduler):")
        for combo in TRAIT_UNSUPPORTED_COMBINATIONS:
            print(f"  {combo}")

        print()

    except Exception as e:
        print(f"Error showing arch info: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified GEMM Code Generator - Single Source of Truth"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32", "fp8", "bf8", "int8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout", type=str, default="rcr", help="Layout (e.g., rcr for row-col-row)"
    )
    parser.add_argument(
        "--gpu-target",
        type=str,
        default="gfx942",
        help="Target GPU (gfx90a, gfx942, gfx950, gfx1201)",
    )
    parser.add_argument("--config", type=Path, help="Configuration JSON file")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["standard", "preshuffle", "multi_d"],
        default=["standard"],
        help="Variants to generate",
    )
    parser.add_argument(
        "--preselected",
        type=str,
        help="Use preselected kernel set (e.g., fp16_rcr_essential)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel generation"
    )
    parser.add_argument(
        "--register", action="store_true", help="Generate dispatcher registration code"
    )
    parser.add_argument(
        "--no-arch-filter",
        action="store_true",
        help="Disable architecture-specific kernel filtering",
    )
    parser.add_argument(
        "--show-arch-info",
        action="store_true",
        help="Show supported configurations for target GPU and exit",
    )

    args = parser.parse_args()

    # Show architecture info if requested
    if args.show_arch_info:
        _show_arch_info(args.gpu_target, args.datatype)
        return 0

    variants = [GemmVariant(v) for v in args.variants] if not args.preselected else None

    codegen = UnifiedGemmCodegen(
        output_dir=args.output_dir,
        datatype=args.datatype,
        layout=args.layout,
        gpu_target=args.gpu_target,
        config_file=args.config,
        variants=variants,
        use_preselected=args.preselected,
        enable_arch_filter=not args.no_arch_filter,
    )

    results = codegen.generate_all(parallel=not args.no_parallel)

    logging.info("\n✅ Generation complete!")
    logging.info(f"  Kernels: {len(results['kernels'])}")
    logging.info(f"  Wrappers: {len(results['wrappers'])}")
    logging.info(f"  Failed: {len(results['failed'])}")

    if results["failed"]:
        logging.error(f"\nFailed kernels: {len(results['failed'])}")
        for err in results["failed"][:5]:
            logging.error(f"  {err}")

    # Generate dispatcher registration if requested
    if args.register:
        logging.info("\n📝 Generating dispatcher registration code...")
        try:
            from generate_dispatcher_registration import (
                scan_generated_headers,
                generate_registration_header,
                generate_registration_cpp,
            )

            kernels = scan_generated_headers(args.output_dir)
            reg_dir = args.output_dir / "registration"
            reg_dir.mkdir(exist_ok=True)

            generate_registration_header(
                kernels, reg_dir / "dispatcher_registration.hpp"
            )
            generate_registration_cpp(kernels, reg_dir / "dispatcher_registration.cpp")

            logging.info(f"✓ Generated registration code for {len(kernels)} kernels")
        except Exception as e:
            logging.error(f"Failed to generate registration code: {e}")
            return 1

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    exit(main())
