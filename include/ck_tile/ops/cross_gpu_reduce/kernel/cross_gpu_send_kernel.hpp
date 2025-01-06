// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {
template <typename CrossReducePartitioner, typename ReduceSendPipeline_>
struct ReduceSendKernel
{
    using ReduceSendPipeline = remove_cvref_t<ReduceSendPipeline_>;
    using DataType           = remove_cvref_t<typename ReduceSendPipeline::DataType>;

    struct ReduceSendKargs
    {
        void* reduce_ptr;
        index_t M;
        index_t N;
    };

    CK_TILE_HOST static constexpr ReduceSendKargs MakeKargs(void* reduce_ptr, index_t M, index_t N)
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
        const auto [i_m, i_n]        = CrossReducePartitioner{}();
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

        uint32_t numThreads =
            static_cast<uint32_t>(CrossReducePartitioner::NumThreads(kargs.M, kargs.N));

        uint32_t threadId = static_cast<uint32_t>(
            i_m + i_n * (kargs.M + ReduceSendPipeline::Block_M - 1) / ReduceSendPipeline::Block_M);

        kargs.reduce_ptr = smem_ptr;
        ReduceSendPipeline{}(transfer_block_window, smem_ptr, threadId, numThreads);

        return;
    }
};
} // namespace ck_tile
