// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"

namespace ck {

// clang-format off
// Assume:
//  1) work_buffer is buffer (typically LDS) allocated outside as workspace
//  2) work_buffer has AccDataType elements, and space size is no less than 3*BlockSize
//  3) mean_value, var_value and count is the input data in vgpr from each thread
//  4) mean_value, var_value and count is the over-written reduced output in vgpr for each thread
//  5) Merge mean and M from ThreadwiseWelford
// clang-format on
template <typename AccDataType,
          index_t BlockSize,
          typename ThreadClusterLengths_M_K,
          typename ThreadClusterArrangeOrder,
          bool GetActualVariance = true>
struct BlockwiseWelford
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    __device__ static inline void Merge(AccDataType& mean_a,
                                        AccDataType& var_a,
                                        int& count_a,
                                        AccDataType mean_b,
                                        AccDataType var_b,
                                        int count_b)
    {
        int count                      = count_a + count_b;
        AccDataType count_b_over_count = count_b / count;
        AccDataType delta              = mean_b - mean_a;
        mean_a += delta * count_b_over_count;
        var_a += var_b + delta * delta * count_a * count_b_over_count;
        count_a = count;
    }

    __device__ static void
    Run(AccDataType& work_buffer, AccDataType& mean_value, AccDataType& var_value, int& count)
    {
        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        index_t mean_offset1  = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);
        index_t var_offset1   = mean_offset1 + BlockSize;
        index_t count_offset1 = var_offset1 + BlockSize;

        work_buffer(mean_offset1)  = mean_value;
        work_buffer(var_offset1)   = var_value;
        work_buffer(count_offset1) = count;

        __syncthreads();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << (cluster_len_shift - 1 - I());

            if(thread_k_cluster_id < indOffset)
            {
                index_t mean_offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx) +
                                       make_tuple(0, indOffset);
                index_t var_offset2   = mean_offset2 + BlockSize;
                index_t count_offset2 = var_offset2 + BlockSize;

                AccDataType mean1  = work_buffer[mean_offset1];
                AccDataType var1   = work_buffer[var_offset1];
                AccDataType count1 = work_buffer[count_offset1];

                AccDataType mean2  = work_buffer[mean_offset2];
                AccDataType var2   = work_buffer[var_offset2];
                AccDataType count2 = work_buffer[count_offset2];

                Merge(mean1, var1, count1, mean2, var2, count2);

                work_buffer(mean_offset1)  = mean1;
                work_buffer(var_offset1)   = var1;
                work_buffer(count_offset1) = count1;
            }

            __syncthreads();
        });

        index_t mean_offset =
            block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));
        index_t var_offset   = mean_offset + BlockSize;
        index_t count_offset = var_offset + BlockSize;

        mean_value = work_buffer[mean_offset];
        if constexpr(GetActualVariance)
            var_value = work_buffer[var_offset] / count;
        else
            var_value = work_buffer[var_offset];
        count = work_buffer[count_offset];
    };
};
} // namespace ck
