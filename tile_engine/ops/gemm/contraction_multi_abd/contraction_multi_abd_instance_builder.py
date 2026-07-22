# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import multiprocessing
import concurrent.futures
import logging


def _import_gemm_kernel_builder():
    """Import GemmKernelBuilder from parent gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder, gemm_builder_module.get_dtype_string


GemmKernelBuilder, get_dtype_string = _import_gemm_kernel_builder()

LAYOUT_MAP = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}


class ContractionMultiABDKernelBuilder(GemmKernelBuilder):
    """Builder for BatchedContractionMultiABDKernel instances."""

    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        a_elementwise="PassThrough",
        b_elementwise="PassThrough",
        cde_elementwise="AddDs",
        num_a_tensor=2,
        num_b_tensor=1,
        num_d_tensor=1,
        num_dim_g=1,
        num_dim_m=1,
        num_dim_n=1,
        num_dim_k=1,
        config_json=None,
    ):
        super().__init__(
            kernel_name_prefix,
            working_path,
            gpu_target,
            datatype,
            layout,
            config_json,
        )
        self.a_elementwise = a_elementwise
        self.b_elementwise = b_elementwise
        self.cde_elementwise = cde_elementwise
        self.num_a_tensor = int(num_a_tensor)
        self.num_b_tensor = int(num_b_tensor)
        self.num_d_tensor = int(num_d_tensor)
        self.num_dim_g = int(num_dim_g)
        self.num_dim_m = int(num_dim_m)
        self.num_dim_n = int(num_dim_n)
        self.num_dim_k = int(num_dim_k)

    def _get_tile_configs(self):
        """Override to set default_pipeline for contraction_multi_abd."""
        tile_config = self.config["tile_config"]

        if tile_config.get("tile_m").get("values") is None:
            tile_config.get("tile_m")["values"] = self._generate_values(
                tile_config.get("tile_m").get("min"),
                tile_config.get("tile_m").get("max"),
                tile_config.get("tile_m").get("step"),
            )
        if tile_config.get("tile_n").get("values") is None:
            tile_config.get("tile_n")["values"] = self._generate_values(
                tile_config.get("tile_n").get("min"),
                tile_config.get("tile_n").get("max"),
                tile_config.get("tile_n").get("step"),
            )
        if tile_config.get("tile_k").get("values") is None:
            tile_config.get("tile_k")["values"] = self._generate_values(
                tile_config.get("tile_k").get("min"),
                tile_config.get("tile_k").get("max"),
                tile_config.get("tile_k").get("step"),
            )

        tile_m_values = tile_config.get("tile_m").get("values")
        tile_n_values = tile_config.get("tile_n").get("values")
        tile_k_values = tile_config.get("tile_k").get("values")
        warp_m_values = tile_config.get("warp_m").get("values")
        warp_n_values = tile_config.get("warp_n").get("values")
        warp_k_values = tile_config.get("warp_k").get("values")
        warp_tile_m_values = tile_config.get("warp_tile_m").get("values")
        warp_tile_n_values = tile_config.get("warp_tile_n").get("values")
        warp_tile_k_values = tile_config.get("warp_tile_k").get("values")

        default_pipeline = "compv3"

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
                                                default_pipeline,
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

    def _generate_kernel_instance(self, tile_config, trait_combo):
        """Generate a contraction multi-ABD kernel instance as C++ code."""
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

        kernel_name = (
            f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}"
            f"_{pipeline}_{epilogue}_{scheduler}"
            f"_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}"
            f"_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"
        )

        tile_str = (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
            f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"
        )
        kernel_name += f"_{tile_str}"

        pipeline_impl_map = {
            "mem": "ck_tile::GemmPipelineAgBgCrMem",
            "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
            "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
        }
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        is_double_smem = pipeline in ["compv4"]
        is_pad_m = pad_m in [True, "true"]
        is_pad_n = pad_n in [True, "true"]
        is_pad_k = pad_k in [True, "true"]

        dtype_str = get_dtype_string(self.datatype)

        # Build A/B/D type lists
        as_types = [f"A{i}DataType" for i in range(self.num_a_tensor)]
        bs_types = [f"B{i}DataType" for i in range(self.num_b_tensor)]
        ds_types = [f"D{i}DataType" for i in range(self.num_d_tensor)]

        a_layout_str = LAYOUT_MAP[self.layout[0]]
        b_layout_str = LAYOUT_MAP[self.layout[1]]
        e_layout_str = LAYOUT_MAP[self.layout[2]]

        # Build D layouts (same as E layout for contraction)
        ds_layout_entries = [
            f"using D{i}Layout = {e_layout_str};"
            for i in range(self.num_d_tensor)
        ]
        ds_layout_names = [f"D{i}Layout" for i in range(self.num_d_tensor)]

        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <utility>
#include <tuple>
#include <array>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
"""

        # Data type aliases
        for i in range(self.num_a_tensor):
            instance_code += f"using A{i}DataType = {dtype_str};\n"
        for i in range(self.num_b_tensor):
            instance_code += f"using B{i}DataType = {dtype_str};\n"
        for i in range(self.num_d_tensor):
            instance_code += f"using D{i}DataType = {dtype_str};\n"

        instance_code += f"""
using EDataType  = {dtype_str};
using AccDataType = float;

using AsDataType = ck_tile::tuple<{", ".join(as_types)}>;
using BsDataType = ck_tile::tuple<{", ".join(bs_types)}>;
using DsDataType = ck_tile::tuple<{", ".join(ds_types)}>;

static constexpr ck_tile::index_t NumATensor = {self.num_a_tensor};
static constexpr ck_tile::index_t NumBTensor = {self.num_b_tensor};
static constexpr ck_tile::index_t NumDTensor = {self.num_d_tensor};
static constexpr ck_tile::index_t NumDimG    = {self.num_dim_g};
static constexpr ck_tile::index_t NumDimM    = {self.num_dim_m};
static constexpr ck_tile::index_t NumDimN    = {self.num_dim_n};
static constexpr ck_tile::index_t NumDimK    = {self.num_dim_k};

using ALayout = {a_layout_str};
using BLayout = {b_layout_str};
using ELayout = {e_layout_str};
"""
        for entry in ds_layout_entries:
            instance_code += f"{entry}\n"

        instance_code += f"""using DsLayout = ck_tile::tuple<{", ".join(ds_layout_names)}>;

using AsLayout = ALayout;
using BsLayout = BLayout;

constexpr const char* KERNEL_NAME = "{kernel_name}";

struct SelectedKernel
{{
    static constexpr ck_tile::index_t BlockSize      = 256;
    static constexpr ck_tile::index_t TileM          = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t TileN          = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t TileK          = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t WarpTileM      = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN      = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t WarpTileK      = {tile_config["warp_tile_k"]};

    static constexpr bool kPadM            = {"true" if is_pad_m else "false"};
    static constexpr bool kPadN            = {"true" if is_pad_n else "false"};
    static constexpr bool kPadK            = {"true" if is_pad_k else "false"};
    static constexpr bool TransposeC       = false;
    static constexpr bool DoubleSmemBuffer = {"true" if is_double_smem else "false"};

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                 kPadN,
                                                                 kPadK,
                                                                 DoubleSmemBuffer,
                                                                 AsLayout,
                                                                 BsLayout,
                                                                 ELayout,
                                                                 TransposeC>;

    using Problem = ck_tile::BatchedContractionMultiABDProblem<AsDataType,
                                                               BsDataType,
                                                               DsDataType,
                                                               EDataType,
                                                               NumDimG,
                                                               NumDimM,
                                                               NumDimN,
                                                               NumDimK>;

    using AElementWise = ck_tile::element_wise::{self.a_elementwise};
    using BElementWise = ck_tile::element_wise::{self.b_elementwise};

    constexpr static auto scheduler = {scheduler_type_map.get(scheduler)};

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<AsDataType,
                                                                       BsDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       scheduler,
                                                                       AElementWise,
                                                                       BElementWise>;

    using GemmPipeline = {pipeline_impl_map.get(pipeline)}<UniversalGemmProblem>;
"""

        # Epilogue
        if epilogue == "cshuffle":
            instance_code += f"""
    using CDEElementWise = ck_tile::element_wise::{self.cde_elementwise};

    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<AsDataType,
                                                             BsDataType,
                                                             DsDataType,
                                                             AccDataType,
                                                             EDataType,
                                                             DsLayout,
                                                             ELayout,
                                                             CDEElementWise,
                                                             TileM,
                                                             TileN,
                                                             WarpPerBlock_M,
                                                             WarpPerBlock_N,
                                                             WarpTileM,
                                                             WarpTileN,
                                                             WarpTileK,
                                                             TransposeC>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
"""
        else:
            instance_code += f"""
    using CDEElementWise = ck_tile::element_wise::{self.cde_elementwise};

    using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<AsDataType,
                                                                  BsDataType,
                                                                  DsDataType,
                                                                  AccDataType,
                                                                  EDataType,
                                                                  DsLayout,
                                                                  ELayout,
                                                                  CDEElementWise,
                                                                  TileM,
                                                                  TileN,
                                                                  kPadM,
                                                                  kPadN,
                                                                  WarpTileM,
                                                                  WarpTileN,
                                                                  WarpTileK,
                                                                  TransposeC>;

    using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
"""

        instance_code += f"""    using Kernel = ck_tile::BatchedContractionMultiABDKernel<
        Problem,
        TilePartitioner, GemmPipeline, GemmEpilogue>;

    static float launch(
        const ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                          NumDimM,
                                                          NumDimN,
                                                          NumDimK,
                                                          NumATensor,
                                                          NumBTensor,
                                                          NumDTensor>& args,
        const ck_tile::stream_config& stream)
    {{
        auto kargs = Kernel::MakeKernelArgs(args);

        if(!Kernel::IsSupportedArguments(kargs))
        {{
            throw std::runtime_error(
                "Wrong! Arguments not supported! Skipping contraction!");
        }}

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::GetBlockSize();

        if(stream.log_level_ > 0)
        {{
            std::cout << "Launching kernel: " << Kernel::GetKernelName() << '\\n'
                      << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z
                      << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", "
                      << blocks.z << "}}" << std::endl;
        }}

        constexpr int kBlockPerCu = {k_block_per_cu};
        float ave_time            = ck_tile::launch_kernel(
            stream, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));

        return ave_time;
    }}
}};
"""

        # Write into file
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

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation."""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

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
                        self.a_elementwise,
                        self.b_elementwise,
                        self.cde_elementwise,
                        self.num_a_tensor,
                        self.num_b_tensor,
                        self.num_d_tensor,
                        self.num_dim_g,
                        self.num_dim_m,
                        self.num_dim_n,
                        self.num_dim_k,
                        self.config_json,
                    )
                )

        print(
            f"Generating {len(work_items)} contraction multi-ABD kernel files "
            f"using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }

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
                    print(f"Kernel generation failed for {item[:2]}: {exc}")

        kernel_list.sort(key=lambda x: x[0])
        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single contraction multi-ABD kernel file."""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        a_elementwise,
        b_elementwise,
        cde_elementwise,
        num_a_tensor,
        num_b_tensor,
        num_d_tensor,
        num_dim_g,
        num_dim_m,
        num_dim_n,
        num_dim_k,
        config_json,
    ) = work_item

    builder = ContractionMultiABDKernelBuilder(
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        a_elementwise,
        b_elementwise,
        cde_elementwise,
        num_a_tensor,
        num_b_tensor,
        num_d_tensor,
        num_dim_g,
        num_dim_m,
        num_dim_n,
        num_dim_k,
        config_json,
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )
        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Contraction Multi-ABD kernel instance builder"
    )
    parser.add_argument(
        "--working_path", required=True, help="Working directory path"
    )
    parser.add_argument(
        "--gpu_target", required=True, help="GPU target architecture"
    )
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout (A, B, E)",
    )
    parser.add_argument(
        "--a_elementwise", default="PassThrough", help="A elementwise function"
    )
    parser.add_argument(
        "--b_elementwise", default="PassThrough", help="B elementwise function"
    )
    parser.add_argument(
        "--cde_elementwise", default="AddDs", help="CDE elementwise function"
    )
    parser.add_argument(
        "--num_a_tensor", type=int, default=2, help="Number of A tensors"
    )
    parser.add_argument(
        "--num_b_tensor", type=int, default=1, help="Number of B tensors"
    )
    parser.add_argument(
        "--num_d_tensor", type=int, default=1, help="Number of D tensors"
    )
    parser.add_argument(
        "--num_dim_g", type=int, default=1, help="Number of G dimensions"
    )
    parser.add_argument(
        "--num_dim_m", type=int, default=1, help="Number of M dimensions"
    )
    parser.add_argument(
        "--num_dim_n", type=int, default=1, help="Number of N dimensions"
    )
    parser.add_argument(
        "--num_dim_k", type=int, default=1, help="Number of K dimensions"
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers"
    )
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument(
        "--kernel_name", help="Kernel name for single generation"
    )
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

    builder = ContractionMultiABDKernelBuilder(
        "contraction_multi_abd",
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.a_elementwise,
        args.b_elementwise,
        args.cde_elementwise,
        args.num_a_tensor,
        args.num_b_tensor,
        args.num_d_tensor,
        args.num_dim_g,
        args.num_dim_m,
        args.num_dim_n,
        args.num_dim_k,
        args.config_json,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

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

        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            trait_parts[0],
            trait_parts[1],
            trait_parts[2],
            trait_parts[3],
            trait_parts[4],
            trait_parts[5],
            trait_parts[6],
        )

        builder._generate_kernel_instance(tile_config, trait_combo)
    elif args.gen_all_individual:
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
