/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_REDUCTION_FUNCTIONS_PARTITIONED_BLOCKWISE_HPP
#define CK_REDUCTION_FUNCTIONS_PARTITIONED_BLOCKWISE_HPP

#include "data_type.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_binop.hpp"

namespace ck {

template <typename buffer1dDescType,
          index_t BlockSize,
          index_t dim0_thread_cluster_length,
          index_t dim1_thread_cluster_length,
          bool reorder_thread_clusters,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct PartitionedBlockwiseReduction_1d_block_buffer
{
    using compType = typename opReduce::dataType;

    static constexpr auto buffer1dDesc = buffer1dDescType{};

    static_assert(BlockSize == dim0_thread_cluster_length * dim1_thread_cluster_length,
                  "The product of cluster lengths should be same as BlockSize!");
    static_assert(dim1_thread_cluster_length > 1,
                  "Parallel reduction need work on at least two elements");

    static_assert(buffer1dDesc.GetElementSize() == BlockSize,
                  "The buffer size should be the same as BlockSize!");

    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    template <typename BufferType>
    __device__ static void Reduce(BufferType& block_buffer,
                                  compType& accuData,
                                  index_t thread_dim0_cluster_id,
                                  index_t thread_dim1_cluster_id)
    {
        constexpr auto cluster_len_shift = get_shift<dim1_thread_cluster_length>();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << (cluster_len_shift - 1 - I());

            if(thread_dim1_cluster_id < indOffset)
            {
                // consider the thread clusters order, ensure the contiguous locations are accessed
                // by contiguous Thread-ID
                index_t offset1 =
                    reorder_thread_clusters
                        ? buffer1dDesc.CalculateOffset(
                              make_tuple(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                         thread_dim0_cluster_id))
                        : buffer1dDesc.CalculateOffset(
                              make_tuple(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         thread_dim1_cluster_id));
                index_t offset2 =
                    reorder_thread_clusters
                        ? buffer1dDesc.CalculateOffset(make_tuple(
                              (thread_dim1_cluster_id + indOffset) * dim0_thread_cluster_length +
                              thread_dim0_cluster_id))
                        : buffer1dDesc.CalculateOffset(
                              make_tuple(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         (thread_dim1_cluster_id + indOffset)));

                compType opData1 = type_convert<compType>{}(block_buffer[offset1]);
                compType opData2 = type_convert<compType>{}(block_buffer[offset2]);
                binop::calculate(opData1, opData2);
                block_buffer(offset1) = type_convert<compType>{}(opData1);
            }

            __syncthreads();
        });

        index_t offset = reorder_thread_clusters
                             ? buffer1dDesc.CalculateOffset(make_tuple(thread_dim0_cluster_id))
                             : buffer1dDesc.CalculateOffset(
                                   make_tuple(thread_dim0_cluster_id * dim1_thread_cluster_length));

        accuData = type_convert<compType>{}(block_buffer[offset]);
    };

    // This interface accumulates on both data values and indices
    template <typename BufferType, typename IdxBufferType>
    __device__ static void Reduce2(BufferType& block_val_buffer,
                                   IdxBufferType& block_idx_buffer,
                                   compType& accuData,
                                   int& accuIndex,
                                   index_t thread_dim0_cluster_id,
                                   index_t thread_dim1_cluster_id)
    {
        constexpr auto cluster_len_shift = get_shift<dim1_thread_cluster_length>();

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << I();

            if(thread_dim1_cluster_id % (indOffset * 2) == 0)
            {
                // consider the thread clusters order, ensure the contiguous locations are accessed
                // by contiguous Thread-ID
                index_t offset1 =
                    reorder_thread_clusters
                        ? buffer1dDesc.CalculateOffset(
                              make_tuple(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                         thread_dim0_cluster_id))
                        : buffer1dDesc.CalculateOffset(
                              make_tuple(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         thread_dim1_cluster_id));
                index_t offset2 =
                    reorder_thread_clusters
                        ? buffer1dDesc.CalculateOffset(make_tuple(
                              (thread_dim1_cluster_id + indOffset) * dim0_thread_cluster_length +
                              thread_dim0_cluster_id))
                        : buffer1dDesc.CalculateOffset(
                              make_tuple(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         (thread_dim1_cluster_id + indOffset)));

                compType opData1 = type_convert<compType>{}(block_val_buffer[offset1]);
                compType opData2 = type_convert<compType>{}(block_val_buffer[offset2]);
                int currIndex1   = block_idx_buffer[offset1];
                int currIndex2   = block_idx_buffer[offset2];

                binop::calculate(opData1, opData2, currIndex1, currIndex2);
                block_val_buffer(offset1) = type_convert<compType>{}(opData1);
                block_idx_buffer(offset1) = currIndex1;
            }

            __syncthreads();
        });

        index_t offset = reorder_thread_clusters
                             ? buffer1dDesc.CalculateOffset(make_tuple(thread_dim0_cluster_id))
                             : buffer1dDesc.CalculateOffset(
                                   make_tuple(thread_dim0_cluster_id * dim1_thread_cluster_length));

        accuData  = type_convert<compType>{}(block_val_buffer[offset]);
        accuIndex = block_idx_buffer[offset];
    }
};

}; // end of namespace ck

#endif
