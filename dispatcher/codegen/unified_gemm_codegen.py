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
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import lru_cache
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

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
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0 and
            self.tile_n % (self.warp_n * self.warp_tile_n) == 0 and
            self.tile_k % (self.warp_k * self.warp_tile_k) == 0 and
            self.tile_m > 0 and self.tile_n > 0 and self.tile_k > 0
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
        parts.append(f"wtile_{self.tile.warp_tile_m}x{self.tile.warp_tile_n}x{self.tile.warp_tile_k}")
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
        'fp16': 'ck_tile::half_t',
        'bf16': 'ck_tile::bf16_t',
        'fp32': 'float',
        'fp8': 'ck_tile::fp8_t',
        'bf8': 'ck_tile::bf8_t',
        'int8': 'ck_tile::int8_t',
    }
    
    DTYPE_TO_DISPATCHER = {
        'fp16': 'DataType::FP16',
        'bf16': 'DataType::BF16',
        'fp32': 'DataType::FP32',
        'fp8': 'DataType::FP8',
        'bf8': 'DataType::BF8',
        'int8': 'DataType::INT8',
    }
    
    LAYOUT_TO_CK = {
        'r': 'ck_tile::tensor_layout::gemm::RowMajor',
        'c': 'ck_tile::tensor_layout::gemm::ColumnMajor',
    }
    
    LAYOUT_TO_DISPATCHER = {
        'r': 'LayoutTag::RowMajor',
        'c': 'LayoutTag::ColMajor',
    }
    
    PIPELINE_TO_CK = {
        'mem': 'ck_tile::GemmPipelineAgBgCrMem',
        'compv3': 'ck_tile::GemmPipelineAgBgCrCompV3',
        'compv4': 'ck_tile::GemmPipelineAgBgCrCompV4',
    }
    
    PIPELINE_TO_BASE = {
        'mem': 'ck_tile::BaseGemmPipelineAgBgCrMem',
        'compv3': 'ck_tile::BaseGemmPipelineAgBgCrCompV3',
        'compv4': 'ck_tile::BaseGemmPipelineAgBgCrCompV4',
    }
    
    PIPELINE_TO_DISPATCHER = {
        'mem': 'Pipeline::Mem',
        'compv3': 'Pipeline::CompV3',
        'compv4': 'Pipeline::CompV4',
    }
    
    SCHEDULER_TO_CK = {
        'intrawave': 'ck_tile::GemmPipelineScheduler::Intrawave',
        'interwave': 'ck_tile::GemmPipelineScheduler::Interwave',
        'default': 'ck_tile::GemmPipelineScheduler::Default',
    }
    
    SCHEDULER_TO_DISPATCHER = {
        'intrawave': 'Scheduler::Intrawave',
        'interwave': 'Scheduler::Interwave',
        'default': 'Scheduler::Auto',
    }
    
    EPILOGUE_TO_DISPATCHER = {
        'cshuffle': 'Epilogue::CShuffle',
        'default': 'Epilogue::Default',
    }
    
    @staticmethod
    def get_output_dtype(dtype: str) -> str:
        """Get output datatype (fp8/bf8 -> fp16)"""
        return 'fp16' if dtype in ['fp8', 'bf8'] else dtype


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
{self._types(config)}
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
            includes += '\n#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"'
        
        return includes
    
    def _types(self, config: KernelConfig) -> str:
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
using DsDataType = ck_tile::tuple<{d_types}>;
using DsLayout = ck_tile::tuple<{d_layouts}>;
using ElementWiseFn = ck_tile::element_wise::{config.elementwise_op};
"""
        
        return types
    
    def _selected_kernel_struct(self, config: KernelConfig, kernel_name: str) -> str:
        """Generate SelectedKernel struct"""
        t = config.tile
        tr = config.trait
        
        return f"""
constexpr const char* KERNEL_NAME = "{kernel_name}";

struct SelectedKernel {{
    // Configuration
    static constexpr ck_tile::index_t BlockSize = {config.block_size};
    static constexpr ck_tile::index_t TileM = {t.tile_m};
    static constexpr ck_tile::index_t TileN = {t.tile_n};
    static constexpr ck_tile::index_t TileK = {t.tile_k};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t.warp_m};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t.warp_n};
    static constexpr ck_tile::index_t WarpPerBlock_K = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK = {t.warp_tile_k};
    
    // Traits
    static constexpr bool kPadM = {str(tr.pad_m).lower()};
    static constexpr bool kPadN = {str(tr.pad_n).lower()};
    static constexpr bool kPadK = {str(tr.pad_k).lower()};
    static constexpr bool TransposeC = false;
    static constexpr bool UsePersistentKernel = {str(tr.persistent).lower()};
    static constexpr bool DoubleSmemBuffer = {str(tr.pipeline == "compv4").lower()};
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle = {str(config.preshuffle).lower()};
    static constexpr ck_tile::index_t NumWaveGroups = {config.num_wave_groups};
    
    {self._tile_types(config)}
    {self._launch_function(config)}
}};
"""
    
    def _tile_types(self, config: KernelConfig) -> str:
        """Generate tile type definitions"""
        return """// Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;
    
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, NumWaveGroups>;
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = """ + self.tm.PIPELINE_TO_BASE[config.trait.pipeline] + """<GemmPipelineProblem>;"""
    
    def _launch_function(self, config: KernelConfig) -> str:
        """Generate launch function"""
        return f"""
    static float launch(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {{
        const ck_tile::index_t k_grain = args.k_batch * TileK;
        const ck_tile::index_t K_split = (args.K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;
            constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};
            [[maybe_unused]] constexpr auto memory_operation = memory_operation_.value;
            
            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType, BDataType, AccDataType, TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
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
            ave_time = ck_tile::launch_kernel(stream,
                ck_tile::make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};
        
        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {{
            if(args.k_batch == 1) {{
                Run(has_hot_loop_, tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum, ck_tile::memory_operation_enum::set>{{}});
            }} else {{
                Run(has_hot_loop_, tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum, ck_tile::memory_operation_enum::atomic_add>{{}});
            }}
            return ave_time;
        }};
        
        if(has_hot_loop) {{
            if(tail_num == ck_tile::TailNumber::One) {{
                RunSplitk(ck_tile::bool_constant<true>{{}}, 
                         ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{{}});
            }} else if(tail_num == ck_tile::TailNumber::Full) {{
                RunSplitk(ck_tile::bool_constant<true>{{}}, 
                         ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{{}});
            }}
        }} else {{
            if(tail_num == ck_tile::TailNumber::One) {{
                RunSplitk(ck_tile::bool_constant<false>{{}}, 
                         ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{{}});
            }} else if(tail_num == ck_tile::TailNumber::Full) {{
                RunSplitk(ck_tile::bool_constant<false>{{}}, 
                         ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{{}});
            }}
        }}
        
        return ave_time;
    }}"""
    
    def _epilogue_code(self, config: KernelConfig) -> str:
        """Generate epilogue code"""
        if config.variant == GemmVariant.MULTI_D:
            return """
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType, BDataType, DsDataType, AccDataType, CDataType,
                DsLayout, CLayout, ElementWiseFn,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
                TransposeC, memory_operation, NumWaveGroups>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        elif config.trait.epilogue == "cshuffle":
            return """
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType, BDataType, ck_tile::tuple<>, AccDataType, CDataType,
                ck_tile::tuple<>, CLayout, ck_tile::element_wise::PassThrough,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
                TransposeC, memory_operation, NumWaveGroups>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        else:
            return """
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType, BDataType, ck_tile::tuple<>, AccDataType, CDataType,
                ck_tile::tuple<>, CLayout, ck_tile::element_wise::PassThrough,
                TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                kPadM, kPadN, WarpTileM, WarpTileN, WarpTileK, TransposeC>;
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""


# ============================================================================
# Dispatcher Wrapper Generator
# ============================================================================

class DispatcherWrapperGenerator:
    """Generates dispatcher wrapper code"""
    
    def __init__(self, datatype: str, layout: str):
        self.datatype = datatype
        self.layout = layout
        self.tm = TypeMappings()
    
    def generate(self, config: KernelConfig, kernel_path: Path, output_dir: Path) -> str:
        """Generate dispatcher wrapper"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)
        output_dtype = self.tm.get_output_dtype(self.datatype)
        rel_path = kernel_path.relative_to(output_dir)
        
        return f"""// SPDX-License-Identifier: MIT
// Auto-generated dispatcher wrapper
#pragma once

#include "ck_tile/dispatcher.hpp"
#include "{rel_path}"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

inline KernelInstancePtr make_{kernel_name}(std::uint16_t gfx_arch = 942) {{
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
    key.structured_sparsity = false;
    
    return std::make_shared<TileKernelInstance<SelectedKernel>>(key, "{kernel_name}");
}}

}}}}
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
        use_preselected: Optional[str] = None
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
                "num_d_tensors": [1, 2]
            }
        }
    
    def generate_all(self, parallel: bool = True) -> Dict:
        """Generate all kernels"""
        log.info(f"Generating GEMM kernels:")
        log.info(f"  Datatype: {self.datatype}")
        log.info(f"  Layout: {self.layout}")
        log.info(f"  Variants: {[v.value for v in self.variants]}")
        if self.use_preselected:
            log.info(f"  Using preselected set: {self.use_preselected}")
        
        results = {'kernels': [], 'wrappers': [], 'failed': []}
        
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
                        futures = [executor.submit(self._generate_one, cfg) for cfg in configs]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                k, w = future.result()
                                results['kernels'].append(k)
                                results['wrappers'].append(w)
                            except Exception as e:
                                results['failed'].append(str(e))
                                log.error(f"Failed: {e}")
                else:
                    for cfg in configs:
                        try:
                            k, w = self._generate_one(cfg)
                            results['kernels'].append(k)
                            results['wrappers'].append(w)
                        except Exception as e:
                            results['failed'].append(str(e))
                            log.error(f"Failed: {e}")
            
            # Generate registration header
            if results['wrappers']:
                self._generate_registration_header(results['wrappers'])
            
            return results
        
        # Generate from preselected set
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._generate_one, cfg) for cfg in configs]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        k, w = future.result()
                        results['kernels'].append(k)
                        results['wrappers'].append(w)
                    except Exception as e:
                        results['failed'].append(str(e))
                        log.error(f"Failed: {e}")
        else:
            for cfg in configs:
                try:
                    k, w = self._generate_one(cfg)
                    results['kernels'].append(k)
                    results['wrappers'].append(w)
                except Exception as e:
                    results['failed'].append(str(e))
                    log.error(f"Failed: {e}")
        
        # Generate registration header
        if results['wrappers']:
            self._generate_registration_header(results['wrappers'])
        
        return results
    
    def _get_preselected_configs(self) -> List[KernelConfig]:
        """Get preselected kernel configurations"""
        try:
            from preselected_kernels import get_preselected_set
            return get_preselected_set(self.use_preselected)
        except ImportError:
            log.warning("preselected_kernels module not found, falling back to config-based generation")
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
                configs.append(KernelConfig(
                    tile=tile, trait=trait, variant=variant, preshuffle=True))
            
            elif variant == GemmVariant.MULTI_D:
                multi_d = self.config.get('multi_d_config', {})
                for ew_op, num_d in itertools.product(
                    multi_d.get('elementwise_ops', ['MultiDAdd']),
                    multi_d.get('num_d_tensors', [1])
                ):
                    configs.append(KernelConfig(
                        tile=tile, trait=trait, variant=variant,
                        elementwise_op=ew_op, num_d_tensors=num_d))
        
        return configs
    
    def _get_tile_configs(self) -> List[TileConfig]:
        """Get valid tile configurations"""
        tc = self.config['tile_config']
        configs = []
        
        for params in itertools.product(
            tc['tile_m'], tc['tile_n'], tc['tile_k'],
            tc['warp_m'], tc['warp_n'], tc['warp_k'],
            tc['warp_tile_m'], tc['warp_tile_n'], tc['warp_tile_k']
        ):
            tile = TileConfig(*params)
            if tile.is_valid():
                configs.append(tile)
        
        return configs
    
    def _get_trait_configs(self) -> List[TraitConfig]:
        """Get valid trait configurations"""
        tc = self.config['trait_config']
        configs = []
        
        for params in itertools.product(
            tc['pipeline'], tc['epilogue'], tc['scheduler'],
            tc['pad_m'], tc['pad_n'], tc['pad_k'], tc['persistent']
        ):
            trait = TraitConfig(*params)
            if trait.is_valid():
                configs.append(trait)
        
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
            Path(w).stem.replace('dispatcher_wrapper_', '')
            for w in wrapper_paths
        ]
        
        includes = "\n".join([f'#include "dispatcher_wrapper_{n}.hpp"' for n in kernel_names])
        registrations = "\n        ".join([f'registry.register_kernel(generated::make_{n}(gfx_arch), priority);' for n in kernel_names])
        
        content = f"""// SPDX-License-Identifier: MIT
// Auto-generated master registration
#pragma once

#include "ck_tile/dispatcher.hpp"
{includes}

namespace ck_tile {{
namespace dispatcher {{

inline void register_all_tile_gemm_kernels(
    std::uint16_t gfx_arch = 942,
    Registry::Priority priority = Registry::Priority::Normal)
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

def main():
    parser = argparse.ArgumentParser(
        description='Unified GEMM Code Generator - Single Source of Truth')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--datatype', type=str, default='fp16',
                       choices=['fp16', 'bf16', 'fp32', 'fp8', 'bf8', 'int8'],
                       help='Data type')
    parser.add_argument('--layout', type=str, default='rcr',
                       help='Layout (e.g., rcr for row-col-row)')
    parser.add_argument('--gpu-target', type=str, default='gfx942',
                       help='Target GPU')
    parser.add_argument('--config', type=Path,
                       help='Configuration JSON file')
    parser.add_argument('--variants', nargs='+',
                       choices=['standard', 'preshuffle', 'multi_d'],
                       default=['standard'],
                       help='Variants to generate')
    parser.add_argument('--preselected', type=str,
                       help='Use preselected kernel set (e.g., fp16_rcr_essential)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel generation')
    parser.add_argument('--register', action='store_true',
                       help='Generate dispatcher registration code')
    
    args = parser.parse_args()
    
    variants = [GemmVariant(v) for v in args.variants] if not args.preselected else None
    
    codegen = UnifiedGemmCodegen(
        output_dir=args.output_dir,
        datatype=args.datatype,
        layout=args.layout,
        gpu_target=args.gpu_target,
        config_file=args.config,
        variants=variants,
        use_preselected=args.preselected
    )
    
    results = codegen.generate_all(parallel=not args.no_parallel)
    
    logging.info(f"\n✅ Generation complete!")
    logging.info(f"  Kernels: {len(results['kernels'])}")
    logging.info(f"  Wrappers: {len(results['wrappers'])}")
    logging.info(f"  Failed: {len(results['failed'])}")
    
    if results['failed']:
        logging.error(f"\nFailed kernels: {len(results['failed'])}")
        for err in results['failed'][:5]:
            logging.error(f"  {err}")
    
    # Generate dispatcher registration if requested
    if args.register:
        logging.info("\n📝 Generating dispatcher registration code...")
        try:
            from generate_dispatcher_registration import (
                scan_generated_headers,
                generate_registration_header,
                generate_registration_cpp
            )
            
            kernels = scan_generated_headers(args.output_dir)
            reg_dir = args.output_dir / "registration"
            reg_dir.mkdir(exist_ok=True)
            
            generate_registration_header(kernels, reg_dir / "dispatcher_registration.hpp")
            generate_registration_cpp(kernels, reg_dir / "dispatcher_registration.cpp")
            
            logging.info(f"✓ Generated registration code for {len(kernels)} kernels")
        except Exception as e:
            logging.error(f"Failed to generate registration code: {e}")
            return 1
    
    return 0 if not results['failed'] else 1


if __name__ == '__main__':
    exit(main())

