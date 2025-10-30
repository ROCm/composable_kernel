// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "gemm_multi_d_fp16.hpp"
#include "utils.hpp"

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
auto gemm_multi_d(const gemm_multi_d_kargs& args, const ck_tile::stream_config& s) -> float
{
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
    // Memory friendly for Interwave scheduler
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 32;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 4;
    constexpr ck_tile::index_t N_Warp = 1;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    constexpr bool DoubleSmemBuffer = false;
#endif
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
    // Compute friendly for Intrawave scheduler
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    constexpr bool DoubleSmemBuffer = false;
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V4)
    // Compute friendly for Intrawave scheduler
    // Using the ping pong reader in the lds level
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    constexpr bool DoubleSmemBuffer = true;
#endif

    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr bool TransposeC = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                 kPadN,
                                                                 kPadK,
                                                                 DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 TransposeC>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = UNIVERSAL_GEMM_PIPELINE<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run =
        [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GEMM_PIPELINE_SCHEDULER;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 EDataType,
                                                 DsLayout,
                                                 CLayout,
                                                 CDEElementWise,
                                                 GemmPipelineProblem::kBlockSize,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 M_Warp,
                                                 N_Warp,
                                                 M_Warp_Tile,
                                                 N_Warp_Tile,
                                                 K_Warp_Tile,
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation>>;

            using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args:" << " grid: {" << grids.x << ", "
                          << grids.y << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", "
                          << blocks.y << ", " << blocks.z << "}" << std::endl;
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
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
        if(tail_num == ck_tile::TailNumber::Full)
        {
            RunSplitk(ck_tile::bool_constant<true>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else if(tail_num == ck_tile::TailNumber::Odd)
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
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
        if(tail_num == ck_tile::TailNumber::One)
        {
            RunSplitk(ck_tile::bool_constant<true>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
        }
        else if(tail_num == ck_tile::TailNumber::Full)
        {
            RunSplitk(ck_tile::bool_constant<true>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }

        auto check_tail = [&](auto... TNs) {
            (try_run<BaseGemmPipeline, decltype(TNs)::value>(tail_num), ...);
        };

        check_tail(ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{},
                   ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{},
                   ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{},
                   ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{},
                   ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{},
                   ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});

#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V4)
        if(tail_num == ck_tile::TailNumber::Three)
        {
            RunSplitk(
                ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
        }
        else
        {
            RunSplitk(ck_tile::bool_constant<true>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
        }
#endif
    }
    else
    {
        if(tail_num == ck_tile::TailNumber::Full)
        {
            RunSplitk(ck_tile::bool_constant<false>{},
                      ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else if(tail_num == ck_tile::TailNumber::Odd)
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

#include "run_gemm_multi_d_fp16_example.inc"

int main(int argc, char* argv[]) { return !run_multiple_d_gemm_example(argc, argv); }
