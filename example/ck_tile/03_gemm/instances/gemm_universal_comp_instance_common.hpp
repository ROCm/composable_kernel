// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include <ck_tile/core.hpp>
#include <iostream>
#include "gemm_basic.hpp"

using A = ck_tile::GemmHostArgs;
using S = ck_tile::stream_config;

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          ck_tile::index_t M_Tile_,
          ck_tile::index_t N_Tile_,
          ck_tile::index_t K_Tile_,
          ck_tile::index_t M_Warp_,
          ck_tile::index_t N_Warp_,
          ck_tile::index_t K_Warp_,
          ck_tile::index_t M_Warp_Tile_,
          ck_tile::index_t N_Warp_Tile_,
          ck_tile::index_t K_Warp_Tile_,
          bool kPadM_,
          bool kPadN_,
          bool kPadK_>
using trait_ = gemm_traits_<ADataType_,
                            BDataType_,
                            AccDataType_,
                            CDataType_,
                            ALayout_,
                            BLayout_,
                            CLayout_,
                            M_Tile_,
                            N_Tile_,
                            K_Tile_,
                            M_Warp_,
                            N_Warp_,
                            K_Warp_,
                            M_Warp_Tile_,
                            N_Warp_Tile_,
                            K_Warp_Tile_,
                            kPadM_,
                            kPadN_,
                            kPadK_>;

template <typename Traits_>
float gemm_(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Traits_::M_Tile, Traits_::N_Tile, Traits_::K_Tile>,
        ck_tile::sequence<Traits_::M_Warp, Traits_::N_Warp, Traits_::K_Warp>,
        ck_tile::sequence<Traits_::M_Warp_Tile, Traits_::N_Warp_Tile, Traits_::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmTilePartitioner<GemmShape>;

    using GemmEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename Traits_::AccDataType,
                                                                     typename Traits_::CDataType,
                                                                     Traits_::kPadM,
                                                                     Traits_::kPadN>>;
    using GemmTraits       = ck_tile::TileGemmTraits<Traits_::kPadM,
                                               Traits_::kPadN,
                                               Traits_::kPadK,
                                               typename Traits_::ALayout,
                                               typename Traits_::BLayout,
                                               typename Traits_::CLayout>;
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<
        ck_tile::GemmPipelineProblem<typename Traits_::ADataType,
                                     typename Traits_::BDataType,
                                     typename Traits_::AccDataType,
                                     GemmShape,
                                     GemmTraits>>;

    constexpr int kBlockPerCu = 1;

    const ck_tile::index_t k_grain     = args.k_batch * Traits_::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * Traits_::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<
            ck_tile::UniversalGemmPipelineProblem<typename Traits_::ADataType,
                                                  typename Traits_::BDataType,
                                                  typename Traits_::AccDataType,
                                                  GemmShape,
                                                  GemmTraits,
                                                  ck_tile::GemmPipelineScheduler::Intrawave,
                                                  has_hot_loop_v,
                                                  tail_number_v>>;
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

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

    if(has_hot_loop)
    {
        // Tail pipeline One to Seven
        if(tail_num == ck_tile::TailNumber::One)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
        }
        else if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }

        if constexpr(BaseGemmPipeline::PrefetchStages > 2)
        {
            if(tail_num == ck_tile::TailNumber::Two)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 3)
        {
            static_assert(BaseGemmPipeline::PrefetchStages > 3);
            if(tail_num == ck_tile::TailNumber::Three)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 4)
        {
            if(tail_num == ck_tile::TailNumber::Four)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 5)
        {
            if(tail_num == ck_tile::TailNumber::Five)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 6)
        {
            if(tail_num == ck_tile::TailNumber::Six)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 7)
        {
            if(tail_num == ck_tile::TailNumber::Seven)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
            }
        }
    }
    else
    {
        // Tail number always Full - #PrefetchStages
        if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else
        {
            std::ostringstream err;
            err << "When there's no hot loop, this tail number \"" << tail_num
                << "\" is not supported! PrefetchStages: " << BaseGemmPipeline::PrefetchStages
                << "\n File: " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
            throw std::runtime_error(err.str());
        }
    }

    return ave_time;
}
