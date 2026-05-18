// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "block_gemm_pipeline_agmem_bgmem_creg_policy.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = BlockGemmPipelineAGmemBGmemCRegPolicy>
struct BlockGemmPipelineAGmemBGmemCReg
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // -----------------------------------------------------------------------------------------
        // Definitions of all needed tiles

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             a_copy_dram_window.get_tile_distribution());

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.get_tile_distribution());

        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        auto block_gemm = BlockGemm();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

        using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
        using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

        using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
        using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

        ABlockTile a_block_tile;
        BBlockTile b_block_tile;
        using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
        constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, kKPerBlock);
        constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, kKPerBlock);

        // -------------------------------------------------------------------------------------
        // Gemm pipeline start

        // Initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        // non-prefetch
        index_t iCounter = num_loop;

        while(iCounter > 0)
        {
            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
            store_tile(a_copy_lds_window, a_block_tile);
            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            block_sync_lds();

            iCounter--;
        }

        return c_block_tile;
    }
};

} // namespace ck_tile
