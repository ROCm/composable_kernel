// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/cross_gpu_reduce/kernel/cross_gpu_connect.hpp"

__constant__ DeviceHandle<mscclpp::SmChannel> constSlaveSmChannels[8]; // For SmChannel

namespace ck_tile {
template <typename CrossReducePartitioner, typename ReduceReceivePipeline_>
struct ReduceReceiveKernel
{
    using ReduceReceivePipeline                = remove_cvref_t<ReduceReceivePipeline_>;
    static constexpr index_t TransferBlockSize = ReduceReceivePipeline::BlockSize;
    using DataType  = remove_cvref_t<typename ReduceReceivePipeline::DataType>;
    using ODataType = remove_cvref_t<typename ReduceReceivePipeline::ODataType>;

    struct ReduceReceiveKargs
    {
        const void* reduce_ptr;
        std::array<const void*, MaxSendGPUNum> receive_ptr_list;
        const void* output_ptr;
        index_t M;
        index_t N;
    };

    CK_TILE_HOST static constexpr ReduceReceiveKargs
    MakeKargs(const void* reduce_ptr,
              std::array<const void*, MaxSendGPUNum> receive_ptr_list,
              const void* output_ptr,
              index_t M,
              index_t N)
    {
        return ReduceReceiveKargs{reduce_ptr, receive_ptr_list, output_ptr, M, N};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return ReduceReceivePipeline::GetSmemSize();
    }

    __host__ static constexpr auto GridSize(index_t M_size, index_t N_size)
    {
        return CrossReducePartitioner::GridSize(M_size, N_size);
    }

    CK_TILE_DEVICE void operator()(ReduceReceiveKargs kargs) const
    {
        auto channel                 = constSlaveSmChannels[0];
        const auto [i_m, i_n]        = CrossReducePartitioner{}();
        const DataType* reduce_start = static_cast<const DataType*>(kargs.reduce_ptr);
        auto transfer_tensor_view    = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                reduce_start,
                make_tuple(kargs.M, kargs.N),
                make_tuple(kargs.N, 1),
                number<ReduceReceivePipeline::Vector_N>{},
                number<1>{});
        }();
        auto transfer_block_window =
            make_tile_window(transfer_tensor_view,
                             make_tuple(number<ReduceReceivePipeline::Block_M>{},
                                        number<ReduceReceivePipeline::Block_N>{}),
                             {i_m, i_n});

        uint32_t numThreads =
            static_cast<uint32_t>(CrossReducePartitioner::NumThreads(kargs.M, kargs.N));
        uint32_t threadId =
            static_cast<uint32_t>(i_m + i_n * (kargs.M + ReduceReceivePipeline::Block_M - 1) /
                                            ReduceReceivePipeline::Block_M);
        uint64_t totalBytes = static_cast<uint64_t>(
            ReduceReceivePipeline::Block_M * ReduceReceivePipeline::Block_N * sizeof(DataType));
        channel.get(0, totalBytes, threadId, numThreads);

        // After the channel get, start the memory block preparation for the receiving window
        const DataType* receive_start =
            static_cast<const DataType*>(kargs.receive_ptr_list[0]);
        auto receive_tensor_view = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                receive_start,
                make_tuple(kargs.M, kargs.N),
                make_tuple(kargs.N, 1),
                number<ReduceReceivePipeline::Vector_N>{},
                number<1>{});
        }();
        auto receive_block_window =
            make_tile_window(receive_tensor_view,
                             make_tuple(number<ReduceReceivePipeline::Block_M>{},
                                        number<ReduceReceivePipeline::Block_N>{}),
                             {i_m, i_n});

        const ODataType* output_start = static_cast<const ODataType*>(kargs.output_ptr);
        auto output_tensor_view       = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                output_start,
                make_tuple(kargs.M, kargs.N),
                make_tuple(kargs.N, 1),
                number<ReduceReceivePipeline::Vector_N>{},
                number<1>{});
        }();
        auto output_block_window =
            make_tile_window(output_tensor_view,
                             make_tuple(number<ReduceReceivePipeline::Block_M>{},
                                        number<ReduceReceivePipeline::Block_N>{}),
                             {i_m, i_n});

        __shared__ char smem_ptr[ReduceReceivePipeline::GetSmemSize()];

        ReduceReceivePipeline{}(
            transfer_block_window, receive_block_window, output_block_window, smem_ptr);
        return;
    }
};
} // namespace ck_tile
