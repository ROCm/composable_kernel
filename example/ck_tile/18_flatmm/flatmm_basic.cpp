// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "flatmm_basic.hpp"

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
float flatmm_calc(const ck_tile::FlatmmHostArgs& args, const ck_tile::stream_config& s)
{
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    constexpr ck_tile::index_t M_Tile = GemmConfig<BDataType>::M_Tile;
    constexpr ck_tile::index_t N_Tile = GemmConfig<BDataType>::N_Tile;
    constexpr ck_tile::index_t K_Tile = GemmConfig<BDataType>::K_Tile;

    constexpr ck_tile::index_t M_Warp = GemmConfig<BDataType>::M_Warp;
    constexpr ck_tile::index_t N_Warp = GemmConfig<BDataType>::N_Warp;
    constexpr ck_tile::index_t K_Warp = GemmConfig<BDataType>::K_Warp;

    constexpr ck_tile::index_t M_Warp_Tile = GemmConfig<BDataType>::M_Warp_Tile;
    constexpr ck_tile::index_t N_Warp_Tile = GemmConfig<BDataType>::N_Warp_Tile;
    constexpr ck_tile::index_t K_Warp_Tile = GemmConfig<BDataType>::K_Warp_Tile;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    
    using CodegenFlatmmShape =
        ck_tile::TileFlatmmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                 ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                 ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenFlatmmShape>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenFlatmmShape, Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    float ave_time{0};
    
    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto memory_operation = memory_operation_.value;
        using CodegenPipelineProblem    = ck_tile::FlatmmPipelineProblem<ADataType,
                                                                         BDataType,
                                                                         AccDataType,
                                                                         CodegenFlatmmShape,
                                                                         CodegenGemmTraits,
                                                                         has_hot_loop_v,
                                                                         tail_number_v>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             AccDataType,
                                             CDataType,
                                             CLayout,
                                             CodegenPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation>>;

        using CodegenFlatmmPolicy = ck_tile::UniversalFlatmmPipelineAgBgCrPolicy;
        using CodegenFlatmmPipeline =
            ck_tile::FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem, CodegenFlatmmPolicy>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::FlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };

    if(has_hot_loop)
    {
        if(tail_num == ck_tile::TailNumber::Odd)
        {
            RunSplitk(ck_tile::bool_constant<true>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else if(tail_num == ck_tile::TailNumber::Even)
        {
            RunSplitk(ck_tile::bool_constant<true>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
        }
        else
        {
            std::ostringstream err;
            err << "For compute pipeline tail number should always be Full, but have \"" << tail_num
                << "\" which is not supported! PrefetchStages: " << BaseGemmPipeline::PrefetchStages
                << "\n File: " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
            throw std::runtime_error(err.str());
        }
    }
    else
    {
        if(tail_num == ck_tile::TailNumber::Odd)
        {
            RunSplitk(ck_tile::bool_constant<false>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else if(tail_num == ck_tile::TailNumber::Even)
        {
            RunSplitk(ck_tile::bool_constant<false>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
        }
        else
        {
            std::ostringstream err;
            err << "Num K loop must be larger than number of prefetech stages."
                << "\n PrefetchStages: " << BaseGemmPipeline::PrefetchStages
                << "\n File: " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
            throw std::runtime_error(err.str());
        }
    }

    return ave_time;
}

#include "run_flatmm_example.inc"

int run_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "C")
    {
        if(data_type == "fp16")
        {
            //run_flatmm_example_with_layouts<ck_tile::half_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf16")
        {
            //run_flatmm_example_with_layouts<ck_tile::bf16_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "fp8")
        {
            run_flatmm_example_with_layouts<ck_tile::fp8_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf8")
        {
            //run_flatmm_example_with_layouts<ck_tile::bf8_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

int main(int argc, char* argv[]) { return !run_flatmm_example(argc, argv); }
