// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/cross_gpu_reduce/pipeline/reduce_send_pipeline_default_policy.hpp"
#include "ck_tile/ops/cross_gpu_reduce/kernel/cross_gpu_connect.hpp"

__constant__ mscclpp::DeviceHandle<mscclpp::SmChannel> constMasterSmChannel;

namespace ck_tile {
template <typename DataType_,
          typename ReduceShape_,
          typename Policy = ReduceSendPipelineDefaultPolicy>
struct CrossReduceSendPipelineScaleUp
{
    using DataType    = remove_cvref_t<DataType_>;
    using ReduceShape = remove_cvref_t<ReduceShape_>;

    static constexpr index_t Block_M = ReduceShape::Block_M;
    static constexpr index_t Block_N = ReduceShape::Block_N;

    static constexpr index_t Vector_N = ReduceShape::Vector_N;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(sizeof(DataType) *
                                       Policy::template MakeLdsBlockDescriptor<ReduceShape>()
                                           .get_element_space_size(),
                                   16) *
               16;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<DataType, ReduceShape>();
    }

    template <typename InDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto operator()(const InDramBlockWindowTmp& input_dram_block_window_tmp,
                                        void* p_smem,
                                        uint32_t threadId,
                                        uint32_t numThreads) const
    {
        DataType* p_lds               = static_cast<DataType*>(p_smem);
        constexpr auto lds_block_desc = Policy::template MakeLdsBlockDescriptor<ReduceShape>();
        auto lds_block = make_tensor_view<address_space_enum::lds>(p_lds, lds_block_desc);
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

        const auto block_tile_tmp =
            tile_elementwise_in([](const DataType& a) { return a; }, host_block_tile);
        store_tile(copy_lds_window, block_tile_tmp);
        uint64_t totalBytes = static_cast<uint64_t>(Block_M * Block_N * sizeof(DataType));
        constMasterSmChannel.put(0, totalBytes, threadId, numThreads);
        move_tile_window(copy_lds_window, {0, Block_N});

        __syncthreads();
        // send the data.
    }
};
} // namespace ck_tile
