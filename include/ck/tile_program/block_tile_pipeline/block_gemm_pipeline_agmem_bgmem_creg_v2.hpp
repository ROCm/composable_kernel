// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2_default_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>
struct BlockGemmPipelineAGmemBGmemCRegV2
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return ck::math::integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().GetElementSpaceSize(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().GetElementSpaceSize();
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    __host__ __device__ auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
                      "wrong!");

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            math::integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.GetElementSpaceSize(),
                                      16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             a_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             a_copy_dram_window.GetTileDistribution());

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             b_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.GetTileDistribution());

        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

        // prefetch
        // global read 0
        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);

        {
            // move to 1
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // Initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);
            // global read 1
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write 0
            const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
            store_tile(b_copy_lds_window, b_block_tile_tmp);
            // global read 1
            b_block_tile = load_tile(b_copy_dram_window);
        }

        index_t iCounter = num_loop - 2;

        do
        {
            block_sync_lds();

            // GEMM i
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);
            // global read i + 2
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write i + 1
            const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
            store_tile(b_copy_lds_window, b_block_tile_tmp);
            // global read i + 2
            b_block_tile = load_tile(b_copy_dram_window);

            iCounter--;

        } while(iCounter > 0);

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // LDS write num_loop - 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
            store_tile(b_copy_lds_window, b_block_tile_tmp);

            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    __device__ auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                               const BDramBlockWindowTmp& b_dram_block_window_tmp,
                               index_t num_loop,
                               void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
