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
               OutDramBlockWindowTmp& output_dram_block_window_tmp) const
    {
        // DRAM tile window for load
        auto copy_dram_window =
            make_tile_window(input_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<Block_M>{}, number<Block_N>{}),
                             input_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeDramTileDistribution<ReduceShape>());

        auto host_block_tile = load_tile(copy_dram_window);

        auto receive_dram_window =
            make_tile_window(receive_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<Block_M>{}, number<Block_N>{}),
                             receive_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeDramTileDistribution<ReduceShape>());

        auto receive_block_tile = load_tile(receive_dram_window);

        auto acc = cast_tile<ODataType>(host_block_tile);

        __syncthreads();

        sweep_tile(receive_block_tile, [&](auto idx) {
            acc(idx) =type_convert<DataType>(receive_block_tile(idx)) + acc(idx);
        });

        store_tile(const_cast<OutDramBlockWindowTmp&>(output_dram_block_window_tmp), acc);
    }
};

} // namespace ck_tile
