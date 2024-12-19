// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/cross_gpu_reduce/pipeline/reduce_receive_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename DataType_,
          typename ODataType_,
          typename ReduceShape_,
          typename Policy = ReduceReceivePipelineDefaultPolicy>
struct CrossReduceReceivePipelineScaleUp
{
    using DataType    = remove_cvref_t<DataType_>;
    using ODataType   = remove_cvref_t<ODataType_>;
    using ReduceShape = remove_cvref_t<ReduceShape_>;

    static constexpr index_t Block_M = ReduceShape::Block_M;
    static constexpr index_t Block_N = ReduceShape::Block_N;

    static constexpr index_t Vector_M = ReduceShape::Vector_M;
    static constexpr index_t Vector_N = ReduceShape::Vector_N;

    static constexpr index_t BlockSize = ReduceShape::NumWarps * get_warp_size();

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(2 * sizeof(DataType) *
                                       Policy::template MakeLdsBlockDescriptor<ReduceShape>()
                                           .get_element_space_size(),
                                   16) *
               16;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<DataType, ReduceShape>();
    }

    template <typename InDramBlockWindowTmp,
              typename ReceiveDramBlockWindowTmp,
              typename OutDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const InDramBlockWindowTmp& input_dram_block_window_tmp,
               const ReceiveDramBlockWindowTmp& receive_dram_block_window_tmp,
               const OutDramBlockWindowTmp& output_dram_block_window_tmp,
               void* p_smem) const
    {
        DataType* p_lds               = static_cast<DataType*>(p_smem);
        constexpr auto lds_block_desc = Policy::template MakeLdsBlockDescriptor<ReduceShape>();
        auto lds_block = make_tensor_view<address_space_enum::lds>(p_lds, lds_block_desc);
        constexpr index_t lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(DataType) * lds_block_desc.get_element_space_size(), 16) *
            16;

        // DRAM tile window for load
        auto copy_dram_window =
            make_tile_window(input_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<Block_M>{}, number<Block_N>{}),
                             input_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeDramTileDistribution<ReduceShape>());

        auto copy_lds_window = make_tile_window(lds_block,
                                                make_tuple(number<Block_M>{}, number<Block_N>{}),
                                                {0, 0},
                                                copy_dram_window.get_tile_distribution());

        auto host_block_tile = load_tile(copy_dram_window);

        // Receive tile window initialization
        DataType* p_receive_lds = static_cast<DataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + lds_block_space_size_aligned));

        auto receive_dram_window =
            make_tile_window(receive_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<Block_M>{}, number<Block_N>{}),
                             receive_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeDramTileDistribution<ReduceShape>());

        auto receive_lds_block =
            make_tensor_view<address_space_enum::lds>(p_receive_lds, lds_block_desc);
        auto receive_lds_window = make_tile_window(receive_lds_block,
                                                   make_tuple(number<Block_M>{}, number<Block_N>{}),
                                                   {0, 0},
                                                   receive_dram_window.get_tile_distribution());
        auto receive_block_tile = load_tile(receive_dram_window);

        const auto host_block_tile_tmp =
            tile_elementwise_in([](const DataType& a) { return a; }, host_block_tile);
        store_tile(copy_lds_window, host_block_tile_tmp);

        const auto receive_block_tile_tmp =
            tile_elementwise_in([](const DataType& a) { return a; }, receive_block_tile);
        store_tile(receive_lds_window, receive_block_tile_tmp);

        __syncthreads();
    }
};

} // namespace ck_tile
