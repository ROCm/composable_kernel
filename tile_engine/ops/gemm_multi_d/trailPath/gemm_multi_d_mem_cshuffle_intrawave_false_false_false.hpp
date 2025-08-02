// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "gemm_multi_d_common.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host.hpp"

namespace mem_cshuffle_intrawave_false_false_false {


template <int TileM, int TileN, int TileK,
          int WarpM, int WarpN, int WarpK,
          int WarpTileM, int WarpTileN, int WarpTileK>
struct GemmKernel {
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static float launch(ck_tile::GemmHostArgs<>& args, const ck_tile::stream_config& stream) {
        static constexpr bool DoubleSmemBuffer =false;
        
        static constexpr bool TransposeC = false;

        static constexpr int kBlockPerCu                         = 1;
        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                   ck_tile::sequence<WarpM, WarpN, WarpK>,
                                   ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;


        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                      TileParitionerGroupNum,
                                                      TileParitionerM01>;

        using Traits  =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, ELayout>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                             ALayout, BLayout, ELayout, TransposeC>;

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>;

        const ck_tile::index_t k_grain     = args.k_batch * TileK;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{0};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr auto scheduler      = ck_tile::GemmPipelineScheduler::Intrawave;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem =
                ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      GemmShape,
                                                      GemmUniversalTraits,
                                                      scheduler,
                                                      has_hot_loop_v,
                                                      tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem>;
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             DsDataType,
                                                             AccDataType,
                                                             EDataType,
                                                             DsLayout,
                                                             ELayout,
                                                             ck_tile::element_wise::PassThrough,
                                                             GemmPipelineProblem::kBlockSize,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             WarpM,
                                                             WarpN,
                                                             WarpTileM,
                                                             WarpTileN,
                                                             WarpTileK,
                                                             UniversalGemmProblem::TransposeC,
                                                             memory_operation>>;

            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
            }

            if(stream.log_level_ > 0)
            {
                std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
            }

            ave_time = ck_tile::launch_kernel(stream,
                                          ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                                              Kernel{}, grids, blocks, 0, kargs));
                
            return ave_time;

        };

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
            if(args.k_batch == 1) {
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::set>{});
            } else {
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::atomic_add>{});
            }
        };

        if(has_hot_loop) {
            
            // Handle One and Full cases directly
            if (tail_num == ck_tile::TailNumber::One) {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
            } else if (tail_num == ck_tile::TailNumber::Full) {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            
            auto check_tail = [&](auto... TNs) {
                ([&]{
                    if constexpr(BaseGemmPipeline::PrefetchStages > static_cast<int>(decltype(TNs)::value)) {
                        if(tail_num == decltype(TNs)::value) {
                            RunSplitk(ck_tile::bool_constant<true>{},
                                    ck_tile::integral_constant<ck_tile::TailNumber, decltype(TNs)::value>{});
                        }
                    }
                }(), ...);
            };

            check_tail(
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{}
            );

        } else {
            
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
                throw std::runtime_error("Num K loop must be larger than number of prefetech stages.");
            }

        }

        return ave_time;
    }

    static std::string get_name() {
        return std::string("gemm_multi_d_") + std::to_string(TileM) + "x" + std::to_string(TileN) + "x" + std::to_string(TileK) +
                "_" + std::to_string(WarpM) + "x" + std::to_string(WarpN) + "x" + std::to_string(WarpK) + "_" +
                std::to_string(WarpTileM) + "x" + std::to_string(WarpTileN) + "x" + std::to_string(WarpTileK) + "_" +
                "false" + "_" +
                "false" + "_" +
                "false" + "_" +
                "mem" + "_" +
                "cshuffle" + "_" +
                "intrawave";
    }
};

} // namespace mem_cshuffle_intrawave_false_false_false
