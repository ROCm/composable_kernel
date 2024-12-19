// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/cross_gpu_reduce/kernel/cross_gpu_connect.hpp"

__constant__ mscclpp::DeviceHandle<mscclpp::SmChannel> constMasterSmChannel;

namespace ck_tile {
template <typename CrossReducePartitioner, typename ReduceSendPipeline_>
struct ReduceSendKernel
{
    using ReduceSendPipeline = remove_cvref_t<ReduceSendPipeline_>;
    using DataType           = remove_cvref_t<typename ReduceSendPipeline::DataType>;

    struct ReduceSendKargs
    {
        const void* reduce_ptr;
        index_t M;
        index_t N;
    };

    CK_TILE_HOST static constexpr ReduceSendKargs
    MakeKargs(const void* reduce_ptr, index_t M, index_t N)
    {
        return ReduceSendKargs{reduce_ptr, M, N};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return ReduceSendPipeline::GetSmemSize();
    }

    __host__ static constexpr auto GridSize(index_t M_size, index_t N_size)
    {
        return CrossReducePartitioner::GridSize(M_size, N_size);
    }

    CK_TILE_DEVICE void operator()(ReduceSendKargs kargs) const
    {
        const auto [i_m, i_n]              = CrossReducePartitioner{}();
        const DataType* reduce_start = static_cast<const DataType*>(kargs.reduce_ptr);
        auto transfer_tensor_view    = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                reduce_start,
                make_tuple(kargs.M, kargs.N),
                make_tuple(kargs.N, 1),
                number<ReduceSendPipeline::Vector_N>{},
                number<1>{});
        }();
        auto transfer_block_window =
            make_tile_window(transfer_tensor_view,
                             make_tuple(number<ReduceSendPipeline::Block_M>{},
                                        number<ReduceSendPipeline::Block_N>{}),
                             {i_m, i_n});

        __shared__ char smem_ptr[ReduceSendPipeline::GetSmemSize()];

        ReduceSendPipeline{}(transfer_block_window, kargs.send_ptr, smem_ptr);

        return;
    }
};
} // namespace ck_tile
