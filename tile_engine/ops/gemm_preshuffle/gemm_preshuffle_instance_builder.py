import argparse
import os
import json
import itertools
import logging
import multiprocessing
import concurrent.futures

from pathlib import Path

from commons.validation_utils import (
    is_tile_config_valid,
    is_trait_combination_valid,
    get_dtype_string,
    get_abc_layouts,
)


class GemmPreshuffleKernelBuilder:
    def __init__(self, working_path, gpu_target, datatype, layout, config_json=None):
        self.working_path = Path(working_path)
        self.gpu_target = gpu_target
        self.datatype = datatype
        self.layout = layout
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)

    def write_kernel_list(self):
        """Write kernel list to file for CMake to read (with comprehensive validation)"""
        # Get configurations using comprehensive validation
        tile_configs = self._get_tile_configs(fast_mode=False)
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                (
                    pipeline,
                    epilogue,
                    scheduler,
                    pad_m,
                    pad_n,
                    pad_k,
                    persistent,
                ) = trait_combo

                # Create kernel name with proper boolean capitalization
                kernel_name = f"gemm_preshuffle_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

                # Create tile configuration string
                tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

                kernel_name += f"_{tile_str}"

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                    }
                )

        # Write kernel count
        with open(self.working_path / "gemm_preshuffle_kernel_count.txt", "w") as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(self.working_path / "gemm_preshuffle_kernel_list.txt", "w") as f:
            for kernel in kernel_list:
                # Format: kernel_name|tile_config|trait_combo
                tile_config = kernel["tile_config"]
                trait_combo = kernel["trait_combo"]

                tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

                trait_str = (
                    f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_"
                    + "_".join(str(x) for x in trait_combo[3:])
                )

                f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _get_tile_configs(self, fast_mode=False):
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
                                                    fast_mode=fast_mode,
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

    def _generate_trait_combinations(self):
        """Generate all combinations of traits"""
        if "traits" in self.config:
            # Old format
            traits = self.config["traits"]
            pipelines = traits["pipelines"]
            epilogues = traits["epilogues"]
            schedulers = traits["schedulers"]

            padding = self.config["padding"]
            persistent = self.config["persistent"]

            all_combinations = list(
                itertools.product(
                    pipelines,
                    epilogues,
                    schedulers,
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

            pipelines = trait_config.get("pipeline", {}).get("values", ["preshufflev2"])
            epilogues = trait_config.get("epilogue", {}).get("values", ["default"])
            schedulers = trait_config.get("scheduler", {}).get("values", ["default"])
            pad_m_values = trait_config.get("pad_m", {}).get("values", [False])
            pad_n_values = trait_config.get("pad_n", {}).get("values", [False])
            pad_k_values = trait_config.get("pad_k", {}).get("values", [False])
            persistent_values = trait_config.get("persistent", {}).get(
                "values", [False]
            )

            all_combinations = list(
                itertools.product(
                    pipelines,
                    epilogues,
                    schedulers,
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
                ("preshufflev2", "default", "default", False, False, False, False)
            ]

        return combinations

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
        pipeline="preshufflev2",  # Default pipeline for validation
        fast_mode=False,  # Add fast mode option
    ):
        """Validate that tile configuration is reasonable"""
        if fast_mode:
            # Fast validation for listing - only basic sanity checks
            if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
                return False
            if warp_m <= 0 or warp_n <= 0 or warp_k <= 0:
                return False
            if warp_tile_m <= 0 or warp_tile_n <= 0 or warp_tile_k <= 0:
                return False

            # Basic divisibility check
            if tile_m % (warp_m * warp_tile_m) != 0:
                return False
            if tile_n % (warp_n * warp_tile_n) != 0:
                return False
            if tile_k % (warp_k * warp_tile_k) != 0:
                return False

            return True
        else:
            # Full validation for generation
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
                self.gpu_target,
            )

    def _generate_kernel_instance(
        self, tile_config, trait_combo, k_block_per_cu, is_header=True
    ):
        """Generate a single kernel instance"""
        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            persistent,
        ) = trait_combo

        # Create kernel name with proper boolean capitalization
        kernel_name = f"gemm_preshuffle_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

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
            "preshufflev1": "ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV1",
            "preshufflev2": "ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2",
        }

        # Map pipeline names to base pipeline for hot loop detection
        base_pipeline_map = {
            "preshufflev1": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV1",
            "preshufflev2": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
        }

        # Map scheduler names to the correct enum values
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        # Determine accumulator type based on datatype
        acc_type = "float"

        # Determine output type
        c_type = get_dtype_string(self.datatype)
        if self.datatype in ["fp8", "bf8"]:
            c_type = "ck_tile::fp16_t"

        # Determine layouts based on self.layout
        a_layout, b_layout, c_layout = get_abc_layouts(self.layout)

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

using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
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
    static constexpr bool DoubleSmemBuffer = {"true" if pipeline == "preshufflev2" else "false"};
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle = true;
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
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline, "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2")}<GemmPipelineProblem>;

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
            constexpr auto scheduler = {scheduler_type_map.get(scheduler, "ck_tile::GemmPipelineScheduler::Default")};
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
            
            using GemmPipeline = {pipeline_impl_map.get(pipeline, "ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2")}<UniversalGemmProblem>;
            
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
                TilePartitioner::MPerBlock,  // kM_
                TilePartitioner::NPerBlock,  // kN_
                WarpPerBlock_M,              // MWave_
                WarpPerBlock_N,              // NWave_
                WarpTileM,                   // MPerXdl_
                WarpTileN,                   // NPerXdl_
                WarpTileK,                   // KPerXdl_
                TransposeC,                  // isCTransposed_
                memory_operation,            // MemoryOperation_
                NumWaveGroups>;              // kNumWaveGroups_
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
"""
        else:  # default epilogue
            instance_code += """            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TilePartitioner::MPerBlock,  // kM_
                TilePartitioner::NPerBlock,  // kN_
                kPadM,
                kPadN,
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_
            
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
            const dim3 blocks = GemmKernel::BlockSize();
            
            if(stream.log_level_ > 0) {{
                std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\\n'
                          << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }}
            
            // Launch kernel
            constexpr int kBlockPerCu = {k_block_per_cu};
            ave_time = ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
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

    def run(self, num_workers=None):
        """Run the builder to generate individual kernel files"""
        # Generate individual kernel files
        self.generate_individual(num_workers)

    def generate_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(), 8
            )  # Limit to avoid memory issues

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()
        k_block_per_cu = self.config.get("k_block_per_cu")

        # Prepare work items for parallel processing
        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (
                        tile_config,
                        trait_combo,
                        k_block_per_cu,
                        self.working_path,
                        self.datatype,
                        self.layout,
                    )
                )

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        # Show first few work items for debugging
        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")  # Show first 3 traits

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit all work items
            print(f"  Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("  All tasks submitted, waiting for completion...")

            # Collect results with progress reporting
            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 100 == 0 or completed == len(work_items):
                    print(
                        f"  Progress: {completed}/{len(work_items)} kernels generated"
                    )

                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Kernel generation failed for {item}: {exc}")

        # Sort kernel list for consistent ordering
        kernel_list.sort(key=lambda x: x[0])  # Sort by kernel name

        # Generate CMake include file for individual targets
        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )

    def _generate_cmake_individual_targets(self, kernel_list):
        """Generate CMake include file that creates individual targets"""
        cmake_code = f"""# Generated CMake file for individual GEMM Preshuffle targets
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

            cmake_code += f'create_individual_gemm_preshuffle_target("{self.datatype}" "{self.layout}" "{trait_str}" "{tile_str}")\n'

        # Write CMake include file
        with open(
            self.working_path / "gemm_preshuffle_individual_targets.cmake", "w"
        ) as f:
            f.write(cmake_code)


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    tile_config, trait_combo, k_block_per_cu, working_path, datatype, layout = work_item

    # Create a temporary builder instance for this worker
    builder = GemmPreshuffleKernelBuilder(working_path, datatype, layout)

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo, k_block_per_cu
        )

        # Create simplified filename without the "gemm_" prefix
        # Remove "gemm_" from the beginning of kernel_name for the filename
        simplified_name = kernel_name
        if simplified_name.startswith("gemm_"):
            simplified_name = simplified_name[5:]  # Remove "gemm_" prefix

        # Write individual header file
        header_file = working_path / f"gemm_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="GEMM kernel instance builder with parallel support"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--gpu_target",
        required=True,
        help="GPU target architecture",
    )
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "fp8", "bf16", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_individual", action="store_true", help="Generate individual kernel files"
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument(
        "--tile_config", help="Tile configuration string for single generation"
    )
    parser.add_argument(
        "--trait_combo", help="Trait combination string for single generation"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )

    args = parser.parse_args()

    assert args.datatype in ["fp16", "bf16", "fp8", "bf8"], (
        f"Invalid datatype string: {args.datatype} (supported datatypes are [fp16, bf16, fp8, and bf8])"
    )

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 3, (
        f"Invalid layout string: {args.layout} (must be 3 characters like 'rcr' where r stands for row major and c stands for column major)"
    )
    assert layout_parts[0] == "r" and layout_parts[1] == "c", (
        f"Invalid matrix_a layout : {layout_parts[0]} or matrix_b layout: {layout_parts[1]} (matrix_a must be 'r' for row major and matrix_b must be 'c' for column major as it is the only supported layout for preshuffle)"
    )
    assert layout_parts[2] == "r", (
        f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
    )

    # Create builder
    builder = GemmPreshuffleKernelBuilder(
        args.working_path, args.gpu_target, args.datatype, args.layout, args.config_json
    )

    if args.list_kernels:
        # Fast listing mode - just write kernel list without generating files
        builder.write_kernel_list()
        pass
    elif args.gen_single:
        # Generate a single kernel file
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )
        # Parse tile config
        tile_parts = args.tile_config.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")

        tile_config = {
            "tile_m": int(tile_dims[0]),
            "tile_n": int(tile_dims[1]),
            "tile_k": int(tile_dims[2]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_k": int(warp_dims[2]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "warp_tile_k": int(warp_tile_dims[2]),
        }

        # Parse trait combo
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            trait_parts[0],  # pipeline
            trait_parts[1],  # epilogue
            trait_parts[2],  # scheduler
            trait_parts[3] == "True",  # pad_m
            trait_parts[4] == "True",  # pad_n
            trait_parts[5] == "True",  # pad_k
            trait_parts[6] == "True",  # persistent
        )

        k_block_per_cu = builder.config.get("k_block_per_cu")

        # Generate the kernel
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo, k_block_per_cu
        )

        # Write the file
        simplified_name = kernel_name
        if simplified_name.startswith("gemm_preshuffle_"):
            simplified_name = simplified_name[16:]

        header_file = (
            builder.working_path / f"gemm_preshuffle_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

    elif args.gen_individual:
        # Generate all individual kernel files
        builder.run(args.num_workers)
        pass
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
