#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import os


def _import_gemm_builder_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(gemm_dir, "gemm_instance_builder.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gemm_builder_module = _import_gemm_builder_module()
GemmKernelBuilder = gemm_builder_module.GemmKernelBuilder
get_dtype_string = gemm_builder_module.get_dtype_string
get_abc_layouts = gemm_builder_module.get_abc_layouts


class GemmTensorQuantKernelBuilder(GemmKernelBuilder):
    def _parse_trait_string(self, trait_string):
        trait_parts = trait_string.split("_")
        return (
            trait_parts[0],  # pipeline
            trait_parts[1],  # epilogue
            trait_parts[2],  # scheduler
            trait_parts[3] == "True",  # pad_m
            trait_parts[4] == "True",  # pad_n
            trait_parts[5] == "True",  # pad_k
            trait_parts[6] == "True",  # persistent
        )

    def _generate_kernel_instance(self, tile_config, trait_combo, kernel_name):
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = trait_combo
        if epilogue != "cshuffle" or persistent:
            raise ValueError("Unsupported gemm_tensor_quant epilogue/persistent configuration")

        if not self._validate_tile_config(
            tile_config["tile_m"],
            tile_config["tile_n"],
            tile_config["tile_k"],
            tile_config["warp_m"],
            tile_config["warp_n"],
            tile_config["warp_k"],
            tile_config["warp_tile_m"],
            tile_config["warp_tile_n"],
            tile_config["warp_tile_k"],
            pipeline,
        ):
            raise ValueError("Unsupported gemm_tensor_quant configuration")

        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        a_layout, b_layout, c_layout = get_abc_layouts(self.layout)

        q_dtype = "float"

        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
using AQDataType = {q_dtype};
using BQDataType = {q_dtype};
using AccDataType = float;
using CDataType = ck_tile::half_t;

using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};

constexpr const char* KERNEL_NAME = "{kernel_name}";

struct SelectedKernel {{
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::{scheduler.capitalize()};

    static constexpr bool kPadM = {"true" if pad_m else "false"};
    static constexpr bool kPadN = {"true" if pad_n else "false"};
    static constexpr bool kPadK = {"true" if pad_k else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB = false;
    static constexpr bool DoubleSmemBuffer = false;

    static constexpr ck_tile::index_t M_Tile = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t N_Tile = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t K_Tile = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t M_Warp = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t N_Warp = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t K_Warp = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t M_Warp_Tile = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t N_Warp_Tile = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t K_Warp_Tile = {tile_config["warp_tile_k"]};

    static float launch(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
            ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
            ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

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
                                                        ck_tile::QuantType::TensorQuant,
                                                        ck_tile::tensor_layout::gemm::RowMajor,
                                                        ck_tile::tensor_layout::gemm::ColumnMajor,
                                                        TransposeC,
                                                        DoubleSmemBuffer>;

        using BaseProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             GemmShape,
                                                             GemmTraits,
                                                             void>;

        using BaseGemmPipeline = ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<BaseProblem>;

        const ck_tile::index_t k_split = ck_tile::integer_least_multiple(args.K, K_Tile);
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const auto tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                                                  BDataType,
                                                                                  AccDataType,
                                                                                  AccDataType,
                                                                                  GemmShape,
                                                                                  GemmTraits,
                                                                                  false,
                                                                                  void,
                                                                                  Scheduler,
                                                                                  has_hot_loop_v,
                                                                                  tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<typename PipelineProblem::AComputeDataType,
                                                 typename PipelineProblem::BComputeDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 M_Warp,
                                                 N_Warp,
                                                 M_Warp_Tile,
                                                 N_Warp_Tile,
                                                 K_Warp_Tile,
                                                 false>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::TensorQuant>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
            {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm_tensor_quant!");
            }}

            const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            constexpr int kBlockPerCu = {k_block_per_cu};
            return ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(run, has_hot_loop, tail_num);
    }}
}};
"""

        simplified_name = kernel_name[len(self.kernel_name_prefix) + 1 :]
        header_file = (
            self.working_path / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")
        return kernel_name, instance_code


def main():
    parser = argparse.ArgumentParser(description="GEMM Tensor Quant kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp8", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")
    parser.add_argument("--gen_single", action="store_true", help="Generate a single kernel file")
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string for single generation")
    parser.add_argument("--trait_combo", help="Trait combination string for single generation")
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
    parser.add_argument(
        "--tier",
        default=None,
        help="Sampling tier (daily/weekly)",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Directory for chosen_instances.json",
    )

    args = parser.parse_args()

    builder = GemmTensorQuantKernelBuilder(
        "gemm_tensor_quant",
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
        return

    if args.gen_single:
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
        trait_combo = builder._parse_trait_string(args.trait_combo)
        builder._generate_kernel_instance(tile_config, trait_combo, args.kernel_name)
        return

    parser.error("Must specify either --list_kernels or --gen_single")


if __name__ == "__main__":
    main()
