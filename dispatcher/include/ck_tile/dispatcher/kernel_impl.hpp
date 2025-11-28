// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file kernel_impl.hpp
 * @brief Template implementation - included ONLY in instantiation .cpp files
 *
 * DO NOT include this in headers! Only include in .cpp files that
 * explicitly instantiate a specific kernel configuration.
 *
 * This separation allows:
 * 1. Parallel compilation - each .cpp is independent
 * 2. Incremental builds - change one kernel, rebuild one file
 * 3. Distributed builds - spread across machines
 */

#pragma once

#include "ck_tile/dispatcher/kernel_template.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// GemmKernel::launch() implementation
// =============================================================================

template <typename AType_,
          typename BType_,
          typename CType_,
          typename AccType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          index_t TileM_,
          index_t TileN_,
          index_t TileK_,
          index_t WaveM_,
          index_t WaveN_,
          index_t WaveK_,
          index_t WarpM_,
          index_t WarpN_,
          index_t WarpK_,
          bool PadM_,
          bool PadN_,
          bool PadK_,
          index_t BlockSize_>
float GemmKernel<AType_,
                 BType_,
                 CType_,
                 AccType_,
                 ALayout_,
                 BLayout_,
                 CLayout_,
                 TileM_,
                 TileN_,
                 TileK_,
                 WaveM_,
                 WaveN_,
                 WaveK_,
                 WarpM_,
                 WarpN_,
                 WarpK_,
                 PadM_,
                 PadN_,
                 PadK_,
                 BlockSize_>::launch(const GemmHostArgs& args, const stream_config& stream)
{
    // Internal type aliases
    using TileShape = TileGemmShape<sequence<TileM, TileN, TileK>,
                                    sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
                                    sequence<WarpTileM, WarpTileN, WarpTileK>,
                                    false,
                                    false>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits          = TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, 1>;
    using PipelineProblem =
        GemmPipelineProblem<ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BasePipeline = BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;

    const index_t k_grain     = args.k_batch * TileK;
    const index_t K_split     = (args.K + k_grain - 1) / k_grain * TileK;
    const index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop   = BasePipeline::BlockHasHotloop(num_loop);
    const TailNumber tail_num = BasePipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    // Lambda to run with specific compile-time parameters
    const auto Run = [&](auto has_hot_loop_v, auto tail_number_v) {
        constexpr bool has_hot_loop_val = decltype(has_hot_loop_v)::value;
        constexpr auto tail_number_val  = decltype(tail_number_v)::value;
        constexpr auto scheduler        = GemmPipelineScheduler::Intrawave;

        using UniversalProblem = UniversalGemmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              TileShape,
                                                              TileGemmUniversalTraits<kPadM,
                                                                                      kPadN,
                                                                                      kPadK,
                                                                                      true,
                                                                                      ALayout,
                                                                                      BLayout,
                                                                                      CLayout,
                                                                                      false,
                                                                                      false,
                                                                                      false,
                                                                                      1,
                                                                                      false>,
                                                              scheduler,
                                                              has_hot_loop_val,
                                                              tail_number_val>;

        using Pipeline        = GemmPipelineAgBgCrCompV4<UniversalProblem>;
        using EpilogueProblem = CShuffleEpilogueProblem<CDataType, CDataType, CLayout>;
        using Epilogue        = CShuffleEpilogue<EpilogueProblem>;
        using Kernel          = ck_tile::GemmKernel<TilePartitioner, Pipeline, Epilogue>;

        const dim3 grids              = Kernel::GridSize(args.M, args.N, 1);
        const dim3 blocks             = Kernel::BlockSize();
        constexpr index_t kBlockPerCu = 1;

        ave_time = launch_kernel(
            stream,
            make_kernel<blocks.x, kBlockPerCu>(Kernel{},
                                               grids,
                                               blocks,
                                               static_cast<const ADataType*>(args.a_ptr),
                                               static_cast<const BDataType*>(args.b_ptr),
                                               static_cast<CDataType*>(args.e_ptr),
                                               args.M,
                                               args.N,
                                               K_split,
                                               args.stride_A,
                                               args.stride_B,
                                               args.stride_E));
    };

    // Dispatch based on runtime conditions
    if(has_hot_loop)
    {
        if(tail_num == TailNumber::Odd)
        {
            Run(std::true_type{}, std::integral_constant<TailNumber, TailNumber::Odd>{});
        }
        else
        {
            Run(std::true_type{}, std::integral_constant<TailNumber, TailNumber::Even>{});
        }
    }
    else
    {
        Run(std::false_type{}, std::integral_constant<TailNumber, TailNumber::Even>{});
    }

    return ave_time;
}

// =============================================================================
// Macro for explicit instantiation in .cpp files
// =============================================================================

/**
 * @brief Explicitly instantiate a kernel type
 *
 * Usage in a .cpp file:
 *   #include "kernel_impl.hpp"
 *   CK_TILE_INSTANTIATE_KERNEL(Kernel_fp16_rcr_128x128x32)
 *
 * This creates a separate compilation unit for this kernel.
 */
#define CK_TILE_INSTANTIATE_KERNEL(KernelType) \
    template float KernelType::launch(const GemmHostArgs&, const stream_config&)

} // namespace dispatcher
} // namespace ck_tile
