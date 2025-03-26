// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#define INSTRUCTION_SCHEDULE

#ifdef INSTRUCTION_SCHEDULE
#include "instruction_schedule/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "instruction_schedule/gemm_pipeline_problem.hpp"
#include "instruction_schedule/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "instruction_schedule/tile_gemm_shape.hpp"
#include "instruction_schedule/tile_gemm_traits.hpp"
#endif

#pragma once

namespace ck_tile {

template <typename Problem, typename Policy>
struct GridGemm
{
    using ADataType        = typename Problem::ADataType;
    using BDataType        = typename Problem::BDataType;
    using CDataType        = typename Problem::CDataType;
    using AccDataType      = typename Problem::AccDataType;
    using CElementFunction = typename Problem::CElementFunction;

    static constexpr auto kMPerBlock = Policy::kMPerBlock;
    static constexpr auto kNPerBlock = Policy::kNPerBlock;
    static constexpr auto kKPerBlock = Policy::kKPerBlock;

    template <typename AGridTensorView, typename BGridTensorView, typename CGridTensorView>
    CK_TILE_DEVICE void operator()(const AGridTensorView& a_grid,
                                   const BGridTensorView& b_grid,
                                   CGridTensorView& c_grid,
                                   const CElementFunction& c_element_func) const
    {
        const auto M = a_grid.get_tensor_descriptor().get_length(number<0>{});
        const auto N = c_grid.get_tensor_descriptor().get_length(number<1>{});
        const auto K = a_grid.get_tensor_descriptor().get_length(number<1>{});

        // divide problem
        const auto id_block = get_block_id();

        const auto num_tile_m = integer_divide_ceil(M, kMPerBlock);
        const auto num_tile_n = integer_divide_ceil(N, kNPerBlock);

        const auto block2tile = Policy::template MakeBlock2TileMap<Problem>(num_tile_m, num_tile_n);

        const auto id_tile = block2tile(id_block);

        const auto iM = __builtin_amdgcn_readfirstlane(id_tile.template get(number<0>{}) * kMPerBlock);
        const auto iN = __builtin_amdgcn_readfirstlane(id_tile.template get(number<1>{}) * kNPerBlock);

        // A block window
        auto a_block_window = make_tile_window(
            a_grid, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {iM, 0});

        // B block window
        auto b_block_window = make_tile_window(
            b_grid, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {iN, 0});

#ifndef INSTRUCTION_SCHEDULE
#pragma message ("disable instruction scheduling")
        // Block GEMM pipeline w/o instruction scheduling
        constexpr auto block_gemm_pipeline = Policy::template GetBlockGemmPipeline<Problem>();

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];
#else
#pragma message ("enable instruction scheduling")
        // Block GEMM pipeline w/ instruction scheduling
        static constexpr index_t M_Tile = 128;
        static constexpr index_t N_Tile = 128;
        static constexpr index_t K_Tile = 64;
        static constexpr index_t M_Warp = 2;
        static constexpr index_t N_Warp = 2;
        static constexpr index_t K_Warp = 1;
        static constexpr index_t M_Warp_Tile = 32;
        static constexpr index_t N_Warp_Tile = 32;
        static constexpr index_t K_Warp_Tile = 16;
        static constexpr bool DoubleSmemBuffer = false;
        static constexpr bool kPadM = false;
        static constexpr bool kPadN = false;
        static constexpr bool kPadK = false;
        static constexpr bool PermuteA = false;
        static constexpr bool PermuteB = false;
        static constexpr bool TransposeC = false;

        // static constexpr int kBlockPerCu                = 1;
        // static constexpr index_t TileParitionerGroupNum = 8;
        // static constexpr index_t TileParitionerM01      = 4;

        using GemmShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
                                        sequence<M_Warp, N_Warp, K_Warp>,
                                        sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>,
                                        PermuteA,
                                        PermuteB>;

        using GemmUniversalTraits = TileGemmUniversalTraits<kPadM,
                                                            kPadN,
                                                            kPadK,
                                                            DoubleSmemBuffer,
                                                            /* ALayout */ tensor_layout::gemm::RowMajor,
                                                            /* BLayout */ tensor_layout::gemm::ColumnMajor,
                                                            /* CLayout */ tensor_layout::gemm::RowMajor,
                                                            TransposeC>;

        using UniversalGemmProblem = UniversalGemmPipelineProblem<ADataType,
                                                                  BDataType,
                                                                  AccDataType,
                                                                  GemmShape,
                                                                  GemmUniversalTraits,
                                                                  GemmPipelineScheduler::Intrawave,
                                                                  /* Has hot loop */ true,
                                                                  TailNumber::Full>;

        constexpr auto block_gemm_pipeline = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>();

        __shared__ char p_smem_char[block_gemm_pipeline.GetSmemSize()];
#endif
        const auto acc_block_tile = block_gemm_pipeline(a_block_window,
                                                        b_block_window,
                                                        K / kKPerBlock,
                                                        p_smem_char);

        // cast to CDataType and apply CElementFunction
        const auto c_block_tile = tile_elementwise_in(
            [&](const auto& acc) { return c_element_func(type_convert<CDataType>(acc)); },
            acc_block_tile);

        // store C
        auto c_window = make_tile_window(
            c_grid, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {iM, iN});

        store_tile(c_window, c_block_tile);
    }
};

} // namespace ck_tile
