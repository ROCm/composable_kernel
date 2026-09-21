# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Import GemmKernelBuilder from the parent gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up: grouped_gemm_rowcolquant -> grouped_gemm_quant -> gemm
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(gemm_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


GemmKernelBuilder = _import_gemm_kernel_builder()


def _import_validation_utils():
    """Import validation utilities from the parent gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))
    spec = importlib.util.spec_from_file_location(
        "validation_utils",
        os.path.join(gemm_dir, "gemm_validation_utils.py"),
    )
    validation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_module)
    return validation_module


_validation_utils = _import_validation_utils()


class GroupedRowColQuantGemmKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json=None,
        max_instances=None,
        seed=None,
        tier=None,
        manifest_path=None,
    ):
        super().__init__(
            kernel_name_prefix,
            working_path,
            gpu_target,
            datatype,
            layout,
            config_json,
            max_instances=max_instances,
            seed=seed,
            tier=tier,
            manifest_path=manifest_path,
        )

    def populate_kernel_header(self, kernel_name):
        return f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
"""

    def populate_kernel_dtype_layout(self):
        d_type = "ck_tile::fp8_t" if self.datatype == "fp8" else "ck_tile::bf8_t"

        return f"""
using ADataType = {d_type};
using BDataType = {d_type};
using AQDataType = float;
using BQDataType = float;
using AccDataType = float;
using CDataType = ck_tile::half_t;
using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;
using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
"""

    def populate_trait_config(self, trait_combo):
        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            persistent,
        ) = trait_combo

        return f"""

    // Traits configurations
    static constexpr bool kPadM = {"true" if pad_m in [True, "true"] else "false"};
    static constexpr bool kPadN = {"true" if pad_n in [True, "true"] else "false"};
    static constexpr bool kPadK = {"true" if pad_k in [True, "true"] else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB = false;
    static constexpr bool UsePersistentKernel = {"true" if persistent in [True, "true"] else "false"};
"""

    def populate_initialization(self, base_pipeline_map, pipeline):
        return """

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;

    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    // RowColQuant Traits
    using GemmQuantTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant,
        BPreshuffleQuant,
        PreshuffleB,
        ALayout, BLayout, CLayout,
        ck_tile::QuantType::RowColQuant,
        AQLayout, BQLayout,
        TransposeC,
        DoubleSmemBuffer,
        UsePersistentKernel>;

    // Standard pipeline traits for hot loop detection
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;
"""

    def populate_launch(
        self,
        scheduler_type_map,
        scheduler,
        pipeline_impl_map,
        pipeline,
        epilogue,
        k_block_per_cu,
        persistent,
    ):
        scheduler_val = scheduler_type_map.get(scheduler)

        persistent_bool = persistent in [True, "true"]

        grid_size_expr = (
            "Kernel::MaxOccupancyGridSize(stream)"
            if persistent_bool
            else "Kernel::GridSize(gemm_descs)"
        )

        instance_code = f"""

    // Launch function
    static float launch(const std::vector<ck_tile::QuantGroupedGemmHostArgs>& gemm_descs,
                        const ck_tile::stream_config& stream,
                        void* kargs_ptr) {{

        constexpr auto scheduler = {scheduler_val};

        // Compute hot loop and tail number at runtime
        const ck_tile::index_t k_grain = gemm_descs[0].k_batch * TileShape::kK;
        const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * TileShape::kK;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{{0}};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType, BDataType, AccDataType, AccDataType,
                TileShape, GemmQuantTraits,
                TransposeC, BDataType,
                scheduler,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<QuantGemmProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<
                    ADataType, BDataType, ck_tile::tuple<>,
                    AccDataType, CDataType, ck_tile::tuple<>,
                    CLayout, ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpPerBlock_M, WarpPerBlock_N,
                    WarpTileM, WarpTileN, WarpTileK,
                    QuantGemmProblem::TransposeC>>;

            using Kernel = ck_tile::QuantGroupedGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                GemmQuantTraits::kQuantType>;

            auto kargs = Kernel::MakeKargs(gemm_descs);
            if(!Kernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping grouped rowcolquant gemm!");
            }}

            const dim3 grids = {grid_size_expr};
            const dim3 blocks = Kernel::BlockSize();

            HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                                kargs.data(),
                                                kargs.size() * sizeof(ck_tile::QuantGemmTransKernelArg),
                                                hipMemcpyHostToDevice,
                                                stream.stream_id_));

            if(stream.log_level_ > 0) {{
                std::cout << "Launching kernel: " << Kernel::GetName() << " with args:"
                          << " grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }}

            constexpr int kBlockPerCu = {k_block_per_cu};
            return ave_time = ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(
                    Kernel{{}},
                    grids,
                    blocks,
                    0,
                    ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                    gemm_descs.size()));
        }};

        return ave_time = BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}
}};
"""
        return instance_code

    def _generate_kernel_instance(self, tile_config, trait_combo):
        """Generate a single kernel instance"""

        k_block_per_cu = self.config.get("k_block_per_cu", 1)

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
        kernel_name = f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

        # Create tile configuration string
        tile_str = (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
        )
        tile_str += (
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
        )
        tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

        kernel_name += f"_{tile_str}"

        # RowColQuant only uses compv3
        pipeline_impl_map = {
            "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
        }
        base_pipeline_map = {
            "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
        }
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        instance_code = self.populate_kernel_header(kernel_name)
        instance_code += self.populate_kernel_dtype_layout()
        instance_code += self.populate_strut_begin(kernel_name)
        instance_code += self.populate_tile_config(tile_config)
        instance_code += self.populate_trait_config(trait_combo)
        instance_code += self.populate_initialization(base_pipeline_map, pipeline)
        instance_code += self.populate_launch(
            scheduler_type_map,
            scheduler,
            pipeline_impl_map,
            pipeline,
            epilogue,
            k_block_per_cu,
            persistent,
        )

        # Write into a file
        simplified_name = kernel_name
        if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
            simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

        header_file = (
            self.working_path / f"{self.kernel_name_prefix}_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

        return kernel_name, instance_code

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        # Prepare work items for parallel processing
        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (
                        tile_config,
                        trait_combo,
                        self.kernel_name_prefix,
                        self.working_path,
                        self.gpu_target,
                        self.datatype,
                        self.layout,
                        self.config_json,
                    )
                )

        # Apply RFC-compliant sampling (Sobol + LHS + maximin)
        if self.max_instances is not None and len(work_items) > self.max_instances:
            kernel_dicts = [
                {"tile_config": item[0], "trait_combo": item[1], "_work_item": item}
                for item in work_items
            ]
            sampled = self._apply_sampling(kernel_dicts)
            work_items = [k["_work_item"] for k in sampled]

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            print(f"  Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("  All tasks submitted, waiting for completion...")

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
        kernel_list.sort(key=lambda x: x[0])

        # Generate CMake include file for individual targets
        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ) = work_item

    # Create a temporary builder instance for this worker
    builder = GroupedRowColQuantGemmKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        # Create simplified filename
        simplified_name = kernel_name
        if simplified_name.startswith(f"{kernel_name_prefix}_"):
            simplified_name = simplified_name[len(kernel_name_prefix) + 1 :]

        # Write individual header file
        header_file = working_path / f"{kernel_name_prefix}_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Grouped RowColQuant GEMM kernel instance builder with parallel support"
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
        choices=["fp8", "bf8"],
        help="Data type (fp8 or bf8 only for RowColQuant)",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
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
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap on number of kernel instances per (dtype, layout) combo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for deterministic sampling; if omitted, derived from today's date",
    )
    parser.add_argument("--tier", default=None, help="Sampling tier (daily/weekly)")
    parser.add_argument(
        "--manifest-path", default=None, help="Directory for chosen_instances.json"
    )

    args = parser.parse_args()

    if args.datatype not in ["fp8", "bf8"]:
        parser.error(
            f"Invalid datatype string: {args.datatype} (RowColQuant supports fp8 and bf8 only)"
        )

    layout_str = args.layout.lower()
    if len(layout_str) != 3:
        parser.error(
            f"Invalid layout string: {args.layout} (must be 3 characters like 'rcr')"
        )

    matrix_a_layout = layout_str[0]
    matrix_b_layout = layout_str[1]
    matrix_c_layout = layout_str[2]

    if matrix_a_layout not in ["r", "c"] or matrix_b_layout not in ["r", "c"]:
        parser.error(
            f"Invalid matrix_a layout: {matrix_a_layout} or matrix_b layout: {matrix_b_layout}"
        )

    if matrix_c_layout != "r":
        parser.error(
            f"Invalid matrix_c layout: {matrix_c_layout} (must be 'r' - row major only)"
        )

    kernel_name_prefix = "grouped_gemm_rowcolquant"
    builder = GroupedRowColQuantGemmKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
        max_instances=args.max_instances,
        seed=args.seed,
        tier=args.tier,
        manifest_path=args.manifest_path,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
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

        builder._generate_kernel_instance(tile_config, trait_combo)
    elif args.gen_all_individual:
        builder._generate_all_individual(num_workers=args.num_workers)
    else:
        parser.error(
            "Must specify --list_kernels, --gen_single, or --gen_all_individual"
        )


if __name__ == "__main__":
    main()
