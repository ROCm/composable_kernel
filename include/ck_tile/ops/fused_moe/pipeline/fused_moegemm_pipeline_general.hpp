// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_general_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = FusedMoeGemmPipelineGeneralPolicy>
struct FusedMoeGemmPipeline_General
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "flatmm_gl";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // matrix a or tokens smem
        constexpr index_t smem_mat_a =
            BlockShape::Block_M0 * BlockShape::Block_K0 * sizeof(ADataType);
        // shuffle C matrix
        constexpr index_t smem_bridge =
            BlockShape::Block_M0 * BlockShape::Block_N0 * sizeof(YDataType);

        return max(smem_mat_a, smem_bridge);
        // return Policy::template GetSmemSize<Problem>();
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }
    template <typename AWindow, typename GWindow, typename DWindow, typename OWindow>
    CK_TILE_DEVICE auto operator()(const AWindow& a_window_,
                                   const GWindow& g_window_,
                                   const DWindow& d_window_,
                                   OWindow& o_window_,
                                   TopkWeightDataType /*topk_weight*/,
                                   CK_TILE_LDS_ADDR void* smem,
                                   index_t hidden_size,
                                   index_t intermediate_size)
    {
        ignore = d_window_;
        ignore = o_window_;
        ignore = hidden_size;

        CK_TILE_LDS_ADDR ADataType* smem_0 = reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem);
        auto a_lds_view                    = make_tensor_view<address_space_enum::lds>(
            smem_0, Policy::template MakeLdsBlockDesc_A<Problem>());
        auto a_lds_win = make_tile_window(
            a_lds_view,
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            {0, 0});

        auto a_global_to_dram_window = make_tile_window(
            a_window_.get_bottom_tensor_view(),
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            a_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_A<Problem>());

        auto g_global_to_dram_window = make_tile_window(
            g_window_.get_bottom_tensor_view(),
            make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
            g_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_G<Problem>());

        // Block GEMM
        constexpr auto gemm_0   = Policy::template GetBlockGemm0<Problem>();
        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // save tokens to lds
        auto a_dram_block = load_tile(a_global_to_dram_window);
        store_tile(a_lds_win, a_dram_block);

        // load g to register
        auto g_dram_block = load_tile(g_global_to_dram_window);

        clear_tile(s_acc); // initialize C
        constexpr index_t kK0 = BlockShape::Block_K0;
        const index_t k0_loops = ck_tile::integer_divide_ceil(intermediate_size, kK0);
        index_t iCounter = k0_loops - 1;
        //gemm 0
        while(iCounter > 0)
        {
            block_sync_lds();

                gemm_0(s_acc, a_lds_win, g_dram_block);

                block_sync_lds();
                move_tile_window(a_global_to_dram_window, {0, kK0});
                move_tile_window(g_global_to_dram_window, {0, kK0});

                a_dram_block = load_tile(a_global_to_dram_window);
                g_dram_block = load_tile(g_global_to_dram_window);

                store_tile(a_lds_win, a_dram_block);
                iCounter--;
        }
        // tail
        {
            block_sync_lds();
            gemm_0(s_acc, a_lds_win, g_dram_block);
        }
        
        //move sacc to LDS
        
        ignore = g_dram_block;

        store_tile(o_window_, a_dram_block);
#if 0
        //check a matrix gather right or not
        constexpr auto a_spans = decltype(a_dram_block)::get_distributed_spans();
        int counter            = 0;
        sweep_tile_span(a_spans[number<0>{}], [&](auto idxm) {
            sweep_tile_span(a_spans[number<1>{}], [&](auto idxk) {
                constexpr auto i_j_idx = make_tuple(idxm, idxk);
                if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
                {
                    counter       = counter + 1;
                    index_t idm_0 = idxm.impl_.at(0);
                    index_t idn_0 = idxk.impl_.at(0);
                    printf("in A idm is %d , idn_ is %d , counter is %d, value is: %f \n",
                           idm_0,
                           idn_0,
                           counter,
                           ck_tile::type_convert<float>(a_dram_block(i_j_idx)));
                }
            });
        });
#endif
    }
};

} // namespace ck_tile
