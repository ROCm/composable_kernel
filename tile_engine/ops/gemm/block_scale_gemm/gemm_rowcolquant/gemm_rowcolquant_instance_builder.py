# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import multiprocessing
import os
import argparse
import importlib.util

import concurrent.futures


def _import_gemm_kernel_builder():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


GemmKernelBuilder = _import_gemm_kernel_builder()


def _import_validation_utils():
    """Import validation utilities."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))
    spec = importlib.util.spec_from_file_location(
        "gemm_validation_utils",
        os.path.join(gemm_dir, "gemm_validation_utils.py"),
    )
    validation_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_utils)
    return validation_utils


_validation_utils = _import_validation_utils()
is_trait_combination_valid = _validation_utils.is_trait_combination_valid
is_tile_config_valid = _validation_utils.is_tile_config_valid


class GemmRowColQuantKernelBuilder(GemmKernelBuilder):
    SUPPORTED_DTYPES = {"fp8", "bf8"}
    SUPPORTED_LAYOUT = "rcr"
    SUPPORTED_TARGETS = {"gfx90a", "gfx942", "gfx950", "gfx1201"}

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

        if datatype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported datatype: {datatype}")
        if layout != self.SUPPORTED_LAYOUT:
            raise ValueError(f"Unsupported layout: {layout}")

        self.gpu_targets = [target for target in str(gpu_target).split(";") if target]
        if not self.gpu_targets:
            raise ValueError("gpu_target must contain at least one architecture")
        unsupported_targets = [
            target
            for target in self.gpu_targets
            if target not in self.SUPPORTED_TARGETS
        ]
        if unsupported_targets:
            raise ValueError(f"Unsupported GPU targets: {unsupported_targets}")

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(), 8
            )  # Limit to avoid memory issues

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
                        self.gpu_targets,
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
            f"Generating {len(work_items)} individual kernel files with {num_workers} workers..."
        )
        print(f"    Trait configs: {len(tile_configs)}")
        print(f"    Trait combinations: {len(trait_combos)}")
        print(f"    Total kernels: {len(work_items)}")

        if work_items:
            print(" First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            print(f"    Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("    All tasks submitted, waiting for completion...")

            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 100 == 0 or completed == len(work_items):
                    print(
                        f"    Progress: {completed}/{len(work_items)} kernels generated"
                    )
                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"    Kernel generation failed for {item}: {exc}")

            # Sort kernel list for consistent ordering
            kernel_list.sort(key=lambda x: x[0])

            # Generate CMake include file for individual targets
            self._generate_cmake_individual_targets(kernel_list)

            print(
                f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
            )

    def _tile_config_string(self, tile_config):
        return (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
            f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"
        )

    def _build_kernel_name(self, tile_config, trait_combo):
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = trait_combo
        return (
            f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_"
            f"{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}_"
            f"{self._tile_config_string(tile_config)}"
        )

    def _parse_tile_config(self, tile_config_arg):
        tile_parts = tile_config_arg.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")
        return {
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

    def _parse_trait_combo(self, trait_combo_arg):
        trait_parts = trait_combo_arg.split("_")
        return (
            trait_parts[0],
            trait_parts[1],
            trait_parts[2],
            trait_parts[3] == "True",
            trait_parts[4] == "True",
            trait_parts[5] == "True",
            trait_parts[6] == "True",
        )

    def populate_kernel_header(self, kernel_name):
        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <iostream>
#include <stdexcept>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp"
"""
        return instance_code

    def populate_kernel_dtype_layout(self):
        dtype = "ck_tile::fp8_t" if self.datatype == "fp8" else "ck_tile::bf8_t"
        instance_code = f"""
using ADataType = {dtype};
using AQDataType = float;
using BDataType = {dtype};
using BQDataType = float;
using AccDataType = float;
using CDataType = ck_tile::half_t;

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;
"""
        return instance_code

    def populate_tile_config(self, tile_config):
        instance_code = f"""
    static constexpr ck_tile::index_t kBlockPerCu = 1;

    static constexpr ck_tile::index_t TileM = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t TileN = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t TileK = {tile_config["tile_k"]};

    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};

    static constexpr ck_tile::index_t WarpTileM = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t WarpTileK = {tile_config["warp_tile_k"]};"""
        return instance_code

    def populate_trait_config(self, trait_combo):
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = trait_combo
        # RowColQuant currently fixes preshuffle-related flags to false.
        # We still thread trait_combo values for pad/persistent correctness
        instance_code = f"""

    static constexpr bool kPadM = {str(pad_m).lower()};
    static constexpr bool kPadN = {str(pad_n).lower()};
    static constexpr bool kPadK = {str(pad_k).lower()};
    static constexpr bool TransposeC = false;
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB = false;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool UsePersistentKernel = {str(persistent).lower()};"""
        return instance_code

    def populate_initialization(self):
        instance_code = """

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& stream)
    {
        using ComputeDataType = ADataType;

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<TileM, TileN, TileK>,
            ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
            ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
        using GemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                               kPadN,
                                                               kPadK,
                                                               APreshuffleQuant,
                                                               BPreshuffleQuant,
                                                               PreshuffleB,
                                                               ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               ck_tile::QuantType::RowColQuant,
                                                               AQLayout,
                                                               BQLayout,
                                                               TransposeC,
                                                               DoubleSmemBuffer,
                                                               UsePersistentKernel,
                                                               16>;

        using BasePipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
            ADataType,
            BDataType,
            AccDataType,
            AccDataType,
            GemmShape,
            GemmTraits,
            TransposeC,
            ComputeDataType,
            ck_tile::GemmPipelineScheduler::Intrawave>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<BasePipelineProblem>;

        const ck_tile::index_t k_split = ck_tile::integer_least_multiple(args.K, TileK);
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);"""
        return instance_code

    def populate_launch(self):
        instance_code = """

        const auto run = [&](const auto has_hot_loop_, const auto tail_number_)
        {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                AccDataType,
                GemmShape,
                GemmTraits,
                TransposeC,
                ComputeDataType,
                ck_tile::GemmPipelineScheduler::Intrawave,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 WarpPerBlock_M,
                                                 WarpPerBlock_N,
                                                 WarpTileM,
                                                 WarpTileN,
                                                 WarpTileK,
                                                 TransposeC>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::RowColQuant>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for RowColQuant kernel");
            }

            const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(stream.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: " << Kernel::GetName() << std::endl
                          << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }

            return ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(run, has_hot_loop, tail_num);
    }
};
"""
        return instance_code

    def _generate_kernel_instance(self, tile_config, trait_combo):
        kernel_name = self._build_kernel_name(tile_config, trait_combo)
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = trait_combo
        if pipeline != "compv3" or epilogue != "cshuffle" or scheduler != "intrawave":
            raise ValueError("RowColQuant only supports compv3/cshuffle/intrawave")

        instance_code = self.populate_kernel_header(kernel_name)
        instance_code += self.populate_kernel_dtype_layout()
        instance_code += self.populate_strut_begin(kernel_name)
        instance_code += self.populate_tile_config(tile_config)
        instance_code += self.populate_trait_config(trait_combo)
        instance_code += self.populate_initialization()
        instance_code += self.populate_launch()

        simplified_name = kernel_name
        if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
            simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

        header_file = (
            self.working_path
            / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")
        return kernel_name, instance_code


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
    builder = GemmRowColQuantKernelBuilder(
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
        header_file = (
            working_path / f"{kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="GEMM RowColQuant kernel instance builder"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype", required=True, choices=["fp8", "bf8"], help="Data type"
    )
    parser.add_argument(
        "--layout", required=True, choices=["rcr"], help="Matrix layout"
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")
    parser.add_argument(
        "--num-workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate all individual kernel files",
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string")
    parser.add_argument("--trait_combo", help="Trait combination string")
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

    builder = GemmRowColQuantKernelBuilder(
        "gemm_rowcolquant",
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
        builder._generate_kernel_instance(
            builder._parse_tile_config(args.tile_config),
            builder._parse_trait_combo(args.trait_combo),
        )
    elif args.gen_all_individual:
        # Generate all individual kernel files
        builder._generate_all_individual(num_workers=args.num_workers)
    else:
        parser.error("Must specify one of: --list_kernels or --gen_single")


if __name__ == "__main__":
    main()
