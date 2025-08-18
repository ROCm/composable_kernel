#!/usr/bin/env python

import os
import json
import argparse
import itertools
from pathlib import Path
import logging
from validation_utils import is_tile_config_valid, is_trait_combination_valid

logging.basicConfig(level=logging.INFO)


class GemmKernelBuilder:
    def __init__(self, working_path, datatype, layout, config_json=None):
        self.working_path = Path(working_path)
        self.datatype = datatype
        self.layout = layout
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()

    def _get_default_config(self):
        """Return default configuration if no config file is provided"""
        return {
            "tile_configs": {
                "fp16": {
                    "rcr": [
                        {
                            "tile_m": 256,
                            "tile_n": 256,
                            "tile_k": 32,
                            "warp_m": 2,
                            "warp_n": 2,
                            "warp_k": 1,
                            "warp_tile_m": 32,
                            "warp_tile_n": 32,
                            "warp_tile_k": 32,
                        },
                        {
                            "tile_m": 256,
                            "tile_n": 128,
                            "tile_k": 32,
                            "warp_m": 2,
                            "warp_n": 2,
                            "warp_k": 1,
                            "warp_tile_m": 32,
                            "warp_tile_n": 32,
                            "warp_tile_k": 16,
                        },
                    ]
                },
                "fp8": {
                    "rcr": [
                        {
                            "tile_m": 256,
                            "tile_n": 256,
                            "tile_k": 32,
                            "warp_m": 4,
                            "warp_n": 1,
                            "warp_k": 1,
                            "warp_tile_m": 32,
                            "warp_tile_n": 32,
                            "warp_tile_k": 32,
                        },
                        {
                            "tile_m": 256,
                            "tile_n": 128,
                            "tile_k": 32,
                            "warp_m": 1,
                            "warp_n": 4,
                            "warp_k": 1,
                            "warp_tile_m": 16,
                            "warp_tile_n": 16,
                            "warp_tile_k": 32,
                        },
                    ]
                },
            },
            "traits": {
                "pipelines": ["mem", "compv3", "compv4"],
                "epilogues": ["default", "cshuffle"],
                "schedulers": ["intrawave", "interwave"],
            },
            "structured_sparsity": ["false"],
            "padding": {"pad_m": ["false"], "pad_n": ["false"], "pad_k": ["false"]},
            "persistent": ["false"],
        }

    def _get_tile_configs(self):
        """Get tile configurations for the current datatype and layout"""
        if "tile_configs" in self.config:
            # Old format
            return (
                self.config["tile_configs"].get(self.datatype, {}).get(self.layout, [])
            )
        elif "tile_config" in self.config:
            # New format - generate combinations from individual parameter values
            tile_config = self.config["tile_config"]

            # Get all possible values for each parameter
            tile_m_values = tile_config.get("tile_m", {}).get("values", [256])
            tile_n_values = tile_config.get("tile_n", {}).get("values", [256])
            tile_k_values = tile_config.get("tile_k", {}).get("values", [32])
            warp_m_values = tile_config.get("warp_m", {}).get("values", [2])
            warp_n_values = tile_config.get("warp_n", {}).get("values", [2])
            warp_k_values = tile_config.get("warp_k", {}).get("values", [1])
            warp_tile_m_values = tile_config.get("warp_tile_m", {}).get("values", [32])
            warp_tile_n_values = tile_config.get("warp_tile_n", {}).get("values", [32])
            warp_tile_k_values = tile_config.get("warp_tile_k", {}).get("values", [32])

            # Generate all combinations
            configs = []
            for tile_m in tile_m_values:
                for tile_n in tile_n_values:
                    for tile_k in tile_k_values:
                        for warp_m in warp_m_values:
                            for warp_n in warp_n_values:
                                for warp_k in warp_k_values:
                                    for warp_tile_m in warp_tile_m_values:
                                        for warp_tile_n in warp_tile_n_values:
                                            for warp_tile_k in warp_tile_k_values:
                                                # Validate configuration
                                                if self._validate_tile_config(
                                                    tile_m,
                                                    tile_n,
                                                    tile_k,
                                                    warp_m,
                                                    warp_n,
                                                    warp_k,
                                                    warp_tile_m,
                                                    warp_tile_n,
                                                    warp_tile_k,
                                                ):
                                                    configs.append(
                                                        {
                                                            "tile_m": tile_m,
                                                            "tile_n": tile_n,
                                                            "tile_k": tile_k,
                                                            "warp_m": warp_m,
                                                            "warp_n": warp_n,
                                                            "warp_k": warp_k,
                                                            "warp_tile_m": warp_tile_m,
                                                            "warp_tile_n": warp_tile_n,
                                                            "warp_tile_k": warp_tile_k,
                                                        }
                                                    )
            return configs
        else:
            # Fallback to default
            return []

    def _validate_tile_config(
        self,
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        pipeline="mem",  # Default pipeline for validation
    ):
        """Validate that tile configuration is reasonable"""
        # Determine data types for validation
        a_datatype = self.datatype
        b_datatype = self.datatype
        c_datatype = self.datatype

        # Special handling for certain data types
        if self.datatype in ["fp8", "bf8"]:
            c_datatype = "fp16"

        # Use the comprehensive validation function
        return is_tile_config_valid(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            a_datatype,
            b_datatype,
            c_datatype,
            pipeline,
        )

    def _generate_trait_combinations(self):
        """Generate all combinations of traits"""
        if "traits" in self.config:
            # Old format
            traits = self.config["traits"]
            pipelines = traits["pipelines"]
            epilogues = traits["epilogues"]
            schedulers = traits["schedulers"]

            padding = self.config["padding"]
            structured_sparsity = self.config["structured_sparsity"]
            persistent = self.config["persistent"]

            all_combinations = list(
                itertools.product(
                    pipelines,
                    epilogues,
                    schedulers,
                    structured_sparsity,
                    padding["pad_m"],
                    padding["pad_n"],
                    padding["pad_k"],
                    persistent,
                )
            )

            # Filter out unsupported trait combinations
            combinations = []
            for combo in all_combinations:
                pipeline, epilogue, scheduler = combo[:3]
                if is_trait_combination_valid(pipeline, epilogue, scheduler):
                    combinations.append(combo)
                else:
                    logging.debug(
                        f"Skipping unsupported trait combination: {pipeline}-{epilogue}-{scheduler}"
                    )

        elif "trait_config" in self.config:
            # New format
            trait_config = self.config["trait_config"]

            pipelines = trait_config.get("pipeline", {}).get("values", ["mem"])
            epilogues = trait_config.get("epilogue", {}).get("values", ["default"])
            schedulers = trait_config.get("scheduler", {}).get("values", ["intrawave"])
            pad_m_values = trait_config.get("pad_m", {}).get("values", [False])
            pad_n_values = trait_config.get("pad_n", {}).get("values", [False])
            pad_k_values = trait_config.get("pad_k", {}).get("values", [False])
            persistent_values = trait_config.get("persistent", {}).get(
                "values", [False]
            )

            # For structured sparsity, use a default since it's not in the config
            structured_sparsity = [False]

            all_combinations = list(
                itertools.product(
                    pipelines,
                    epilogues,
                    schedulers,
                    structured_sparsity,
                    pad_m_values,
                    pad_n_values,
                    pad_k_values,
                    persistent_values,
                )
            )

            # Filter out unsupported trait combinations
            combinations = []
            for combo in all_combinations:
                pipeline, epilogue, scheduler = combo[:3]
                if is_trait_combination_valid(pipeline, epilogue, scheduler):
                    combinations.append(combo)
                else:
                    logging.debug(
                        f"Skipping unsupported trait combination: {pipeline}-{epilogue}-{scheduler}"
                    )
        else:
            # Fallback to minimal default
            combinations = [
                ("mem", "default", "intrawave", False, False, False, False, False)
            ]

        return combinations

    def _get_dtype_string(self):
        """Get C++ type string for datatype"""
        dtype_map = {
            "fp16": "ck_tile::fp16_t",
            "fp8": "ck_tile::fp8_t",
            "bf16": "ck_tile::bf16_t",
            "fp32": "float",
            "fp64": "double",
        }
        return dtype_map.get(self.datatype, "float")

    def _get_layout_string(self):
        """Get C++ layout string"""
        layout_map = {
            "rcr": "ck_tile::tensor_layout::gemm::RowMajor",
            "rrr": "ck_tile::tensor_layout::gemm::RowMajor",
            "rcm": "ck_tile::tensor_layout::gemm::ColumnMajor",
        }
        return layout_map.get(self.layout, "ck_tile::tensor_layout::gemm::RowMajor")

    def _generate_kernel_instance(self, tile_config, trait_combo, is_header=True):
        """Generate a single kernel instance"""
        (
            pipeline,
            epilogue,
            scheduler,
            structured_sparsity,
            pad_m,
            pad_n,
            pad_k,
            persistent,
        ) = trait_combo

        # Create kernel name
        kernel_name = f"gemm_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{structured_sparsity}_{pad_m}_{pad_n}_{pad_k}_{persistent}"

        # Create tile configuration string
        tile_str = (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
        )
        tile_str += (
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
        )
        tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

        kernel_name += f"_{tile_str}"

        # Map pipeline names to the correct pipeline implementation
        pipeline_impl_map = {
            "mem": "ck_tile::GemmPipelineAgBgCrMem",
            "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
            "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
        }

        # Map pipeline names to base pipeline for hot loop detection
        base_pipeline_map = {
            "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
            "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
            "compv4": "ck_tile::BaseGemmPipelineAgBgCrCompV4",
        }

        # Map scheduler names to the correct enum values
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        # Determine accumulator type based on datatype
        acc_type = "float"
        if self.datatype in ["int8", "int4"]:
            acc_type = "int32_t"

        # Determine output type
        c_type = self._get_dtype_string()
        if self.datatype in ["fp8", "bf8"]:
            c_type = "ck_tile::fp16_t"

        # Determine layouts based on self.layout
        a_layout = "ck_tile::tensor_layout::gemm::RowMajor"
        b_layout = "ck_tile::tensor_layout::gemm::ColumnMajor"
        c_layout = "ck_tile::tensor_layout::gemm::RowMajor"

        if self.layout == "rrr":
            b_layout = "ck_tile::tensor_layout::gemm::RowMajor"
        elif self.layout == "rcm":
            c_layout = "ck_tile::tensor_layout::gemm::ColumnMajor"

        # Generate kernel instance code using the correct API
        pragma_line = "#pragma once\n" if is_header else ""
        instance_code = f"""// Generated kernel instance for {kernel_name}
{pragma_line}
#include <cstdint>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

using ADataType = {self._get_dtype_string()};
using BDataType = {self._get_dtype_string()};
using AccDataType = {acc_type};
using CDataType = {c_type};

using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};

// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";

// Wrapper for simplified launch interface
struct SelectedKernel {{
    // Tile configuration
    static constexpr ck_tile::index_t BlockSize = 256;
    static constexpr ck_tile::index_t TileM = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t TileN = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t TileK = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t WarpTileM = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t WarpTileK = {tile_config["warp_tile_k"]};

    // Traits
    static constexpr bool kPadM = {"true" if pad_m == "true" else "false"};
    static constexpr bool kPadN = {"true" if pad_n == "true" else "false"};
    static constexpr bool kPadK = {"true" if pad_k == "true" else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool UsePersistentKernel = {"true" if persistent == "true" else "false"};
    static constexpr bool DoubleSmemBuffer = {"true" if pipeline == "compv4" else "false"};
    static constexpr bool UseStructuredSparsity = {"true" if structured_sparsity == "true" else "false"};
    static constexpr bool Preshuffle = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;
    
    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    
    // Traits
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, NumWaveGroups>;
    
    // Pipeline problem
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        TileShape,
        Traits>;
    
    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline, "ck_tile::BaseGemmPipelineAgBgCrMem")}<GemmPipelineProblem>;

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
            constexpr auto scheduler = {scheduler_type_map.get(scheduler, "ck_tile::GemmPipelineScheduler::Intrawave")};
            [[maybe_unused]] constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, CLayout, TransposeC,
                                                UseStructuredSparsity, UsePersistentKernel,
                                                NumWaveGroups, Preshuffle>,
                scheduler,
                has_hot_loop_v,
                tail_number_v>;
            
            using GemmPipeline = {pipeline_impl_map.get(pipeline, "ck_tile::GemmPipelineAgBgCrCompV3")}<UniversalGemmProblem>;
            
            // Epilogue
"""

        # Add epilogue configuration based on type
        if epilogue == "cshuffle":
            instance_code += """            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                UniversalGemmProblem::kBlockSize,
                TilePartitioner::MPerBlock,
                TilePartitioner::NPerBlock,
                WarpPerBlock_M,
                WarpPerBlock_N,
                WarpTileM,
                WarpTileN,
                WarpTileK,
                TransposeC,
                memory_operation,
                NumWaveGroups>;
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
"""
        else:  # default epilogue
            instance_code += """            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                AccDataType,
                CDataType,
                CLayout,
                kPadM,
                kPadN,
                WarpTileM,
                WarpTileN,
                WarpTileK,
                TransposeC>;
            
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
"""

        instance_code += f"""
            
            // Kernel type
            using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            
            // Make kernel arguments
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
            }}
            
            // Get grid and block sizes
            const dim3 grids = {"GemmKernel::MaxOccupancyGridSize(stream)" if persistent == "true" else "GemmKernel::GridSize(args.M, args.N, args.k_batch)"};
            constexpr dim3 blocks = GemmKernel::BlockSize();
            
            if(stream.log_level_ > 0) {{
                std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\\n'
                          << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }}
            
            // Launch kernel
            constexpr int kBlockPerCu = 1;
            ave_time = ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<blocks.x, kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {{
            if(args.k_batch == 1) {{
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::set>{{}});
            }} else {{
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::atomic_add>{{}});
            }}
        }};

        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        return ave_time;
    }}
}};
"""

        return kernel_name, instance_code

    def list_kernels(self):
        """List all kernel instances that will be generated"""
        kernels = []
        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                kernel_name, _ = self._generate_kernel_instance(
                    tile_config, trait_combo
                )
                kernels.append(kernel_name)

        return kernels

    def generate_blobs(self):
        """Generate blob files for monolithic build"""
        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        # Group by trait for blob generation
        trait_groups = {}
        for trait_combo in trait_combos:
            pipeline, epilogue, scheduler = trait_combo[:3]
            trait_key = f"{pipeline}_{epilogue}_{scheduler}"
            if trait_key not in trait_groups:
                trait_groups[trait_key] = []
            trait_groups[trait_key].append(trait_combo)

        blob_files = []
        blob_ranges = []

        blob_index = 0
        for trait_key, trait_list in trait_groups.items():
            start_index = blob_index

            for trait_combo in trait_list:
                for tile_config in tile_configs:
                    kernel_name, instance_code = self._generate_kernel_instance(
                        tile_config, trait_combo, is_header=False
                    )

                    # Write instance file
                    instance_file = self.working_path / f"{kernel_name}.cpp"
                    with open(instance_file, "w") as f:
                        f.write(instance_code)

                    blob_files.append(str(instance_file))
                    blob_index += 1

            end_index = blob_index
            blob_ranges.append(f"{trait_key} {start_index} {end_index}")

        # Write blob list files
        with open(self.working_path / "gemm_instance_blobs.txt", "w") as f:
            f.write("\n".join(blob_files))

        with open(self.working_path / "gemm_instance_blobs_range.txt", "w") as f:
            f.write("\n".join(blob_ranges))

        # Generate dispatcher header
        self._generate_dispatcher_header(trait_groups, tile_configs)

    def _generate_dispatcher_header(self, trait_groups, tile_configs):
        """Generate the dispatcher header for monolithic build"""
        dispatcher_code = f"""
// Generated GEMM dispatcher for {self.datatype} {self.layout}
#pragma once

#include <functional>
#include <vector>
#include <string>
#include <tuple>
#include "gemm_common.hpp"
#include <ck_tile/host.hpp>

// Forward declarations
"""

        # Add forward declarations for all kernels
        for trait_key, trait_list in trait_groups.items():
            for trait_combo in trait_list:
                for tile_config in tile_configs:
                    kernel_name, _ = self._generate_kernel_instance(
                        tile_config, trait_combo
                    )
                    dispatcher_code += f"extern std::tuple<std::string, float> {kernel_name}(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&);\n"

        dispatcher_code += """
class GemmDispatcher {
public:
    static std::vector<std::function<std::tuple<std::string, float>(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>
    dispatch(bool structured_sparsity, const KernelTraits& trait) {
        std::vector<std::function<std::tuple<std::string, float>(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>> kernels;
        
"""

        # Add dispatcher logic
        for trait_key, trait_list in trait_groups.items():
            for trait_combo in trait_list:
                pipeline, epilogue, scheduler, ss, pad_m, pad_n, pad_k, persistent = (
                    trait_combo
                )

                # Create condition
                conditions = []
                conditions.append(f'trait.pipeline == "{pipeline}"')
                conditions.append(f'trait.epilogue == "{epilogue}"')
                conditions.append(f'trait.scheduler == "{scheduler}"')
                conditions.append(
                    f"structured_sparsity == {'true' if ss == 'true' else 'false'}"
                )
                conditions.append(
                    f"trait.pad_m == {'true' if pad_m == 'true' else 'false'}"
                )
                conditions.append(
                    f"trait.pad_n == {'true' if pad_n == 'true' else 'false'}"
                )
                conditions.append(
                    f"trait.pad_k == {'true' if pad_k == 'true' else 'false'}"
                )
                conditions.append(
                    f"trait.persistent == {'true' if persistent == 'true' else 'false'}"
                )

                condition_str = " && ".join(conditions)

                dispatcher_code += f"        if ({condition_str}) {{\n"

                for tile_config in tile_configs:
                    kernel_name, _ = self._generate_kernel_instance(
                        tile_config, trait_combo
                    )
                    dispatcher_code += (
                        f"            kernels.push_back({kernel_name});\n"
                    )

                dispatcher_code += "        }\n"

        dispatcher_code += """
        return kernels;
    }
};
"""

        # Write dispatcher header
        with open(self.working_path / "gemm_dispatcher.hpp", "w") as f:
            f.write(dispatcher_code)

    def generate_individual(self):
        """Generate individual kernel files for separate compilation"""
        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        kernel_list = []

        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                kernel_name, instance_code = self._generate_kernel_instance(
                    tile_config, trait_combo
                )

                # Create simplified filename without the "gemm_" prefix
                # Remove "gemm_" from the beginning of kernel_name for the filename
                simplified_name = kernel_name
                if simplified_name.startswith("gemm_"):
                    simplified_name = simplified_name[5:]  # Remove "gemm_" prefix

                # Write individual header file
                header_file = self.working_path / f"gemm_single_{simplified_name}.hpp"
                with open(header_file, "w") as f:
                    f.write(instance_code)

                kernel_list.append((kernel_name, trait_combo, tile_config))

        # Generate CMake include file for individual targets
        self._generate_cmake_individual_targets(kernel_list)

    def _generate_cmake_individual_targets(self, kernel_list):
        """Generate CMake include file that creates individual targets"""
        cmake_code = f"""# Generated CMake file for individual GEMM targets
# Datatype: {self.datatype}, Layout: {self.layout}

"""

        for kernel_name, trait_combo, tile_config in kernel_list:
            pipeline, epilogue, scheduler = trait_combo[:3]

            # Format tile config for CMake function
            tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
            tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
            tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

            trait_str = f"{pipeline}_{epilogue}_{scheduler}_" + "_".join(
                str(x) for x in trait_combo[3:]
            )

            cmake_code += f'create_individual_gemm_target("{self.datatype}" "{self.layout}" "{trait_str}" "{tile_str}")\n'

        # Write CMake include file
        with open(self.working_path / "gemm_individual_targets.cmake", "w") as f:
            f.write(cmake_code)

    def run(self, mode):
        """Run the builder in the specified mode"""
        if mode == "list_blobs":
            # Generate the list of blob files that will be created
            tile_configs = self._get_tile_configs()
            trait_combos = self._generate_trait_combinations()

            # Group by trait for blob generation
            trait_groups = {}
            for trait_combo in trait_combos:
                pipeline, epilogue, scheduler = trait_combo[:3]
                trait_key = f"{pipeline}_{epilogue}_{scheduler}"
                if trait_key not in trait_groups:
                    trait_groups[trait_key] = []
                trait_groups[trait_key].append(trait_combo)

            # Generate list of blob files
            blob_files = []
            for trait_key, trait_list in trait_groups.items():
                for trait_combo in trait_list:
                    for tile_config in tile_configs:
                        kernel_name, _ = self._generate_kernel_instance(
                            tile_config, trait_combo
                        )
                        blob_file = str(self.working_path / f"{kernel_name}.cpp")
                        blob_files.append(blob_file)

            # Sort blob files for consistent ordering
            blob_files.sort()

            # Generate blob ranges
            blob_ranges = []
            blob_index = 0
            for trait_key, trait_list in trait_groups.items():
                start_index = blob_index

                # Count kernels for this trait group
                kernel_count = 0
                for trait_combo in trait_list:
                    kernel_count += len(tile_configs)

                blob_index += kernel_count
                end_index = blob_index
                blob_ranges.append(f"{trait_key} {start_index} {end_index}")

            # Write blob list files
            with open(self.working_path / "gemm_instance_blobs.txt", "w") as f:
                f.write("\n".join(blob_files))

            with open(self.working_path / "gemm_instance_blobs_range.txt", "w") as f:
                f.write("\n".join(blob_ranges))

            print(f"Listed {len(blob_files)} kernel blobs to be generated")

        elif mode == "gen_blobs":
            # Generate blob files for monolithic build
            self.generate_blobs()
            print(f"Generated blob files in {self.working_path}")

        elif mode == "gen_individual":
            # Generate individual kernel files
            self.generate_individual()
            print(f"Generated individual kernel files in {self.working_path}")

        else:
            raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="GEMM kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "fp8", "bf16", "fp32", "fp64"],
        help="Data type",
    )
    parser.add_argument(
        "--layout", required=True, choices=["rcr", "rrr", "rcm"], help="Matrix layout"
    )
    parser.add_argument("--config_json", help="Configuration JSON file")

    # Mode selection
    parser.add_argument(
        "--list_blobs", action="store_true", help="List all kernel blobs"
    )
    parser.add_argument("--gen_blobs", action="store_true", help="Generate blob files")
    parser.add_argument(
        "--gen_individual", action="store_true", help="Generate individual kernel files"
    )

    args = parser.parse_args()

    # Determine mode
    mode = None
    if args.list_blobs:
        mode = "list_blobs"
    elif args.gen_blobs:
        mode = "gen_blobs"
    elif args.gen_individual:
        mode = "gen_individual"
    else:
        parser.error("Must specify one of: --list_blobs, --gen_blobs, --gen_individual")

    # Create builder and run
    builder = GemmKernelBuilder(
        args.working_path, args.datatype, args.layout, args.config_json
    )
    builder.run(mode)


if __name__ == "__main__":
    main()
