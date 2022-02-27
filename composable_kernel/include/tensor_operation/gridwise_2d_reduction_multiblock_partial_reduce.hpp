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
#ifndef CK_GRIDWISE_2D_REDUCTION_MULTIBLOCK_TWO_CALL_HPP
#define CK_GRIDWISE_2D_REDUCTION_MULTIBLOCK_TWO_CALL_HPP

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_accumulate.hpp"
#include "reduction_functions_blockwise.hpp"

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseReduction,
          bool NeedIndices,
          typename InDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename WorkspaceDesc_M_K,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
__global__ void
kernel_partial_reduce_multiblock(const InGridDesc_M_K in_grid_desc_m_k,
                                 const WorkspaceDesc_M_K workspace_desc_m_k,
                                 const InElementwiseOperation in_elementwise_op,
                                 const AccElementwiseOperation acc_elementwise_op,
                                 index_t BlkGroupSize,
                                 index_t kBlockTileIterations,
                                 const InDataType* const __restrict__ p_src_global,
                                 AccDataType* const __restrict__ p_ws_values_global,
                                 IndexDataType* const __restrict__ p_ws_indices_global)

{
    if constexpr(!NeedIndices)
    {
        GridwiseReduction::Run(in_grid_desc_m_k,
                               workspace_desc_m_k,
                               in_elementwise_op,
                               acc_elementwise_op,
                               BlkGroupSize,
                               kBlockTileIterations,
                               p_src_global,
                               p_ws_values_global,
                               p_ws_indices_global);
    }
    else
    {
        GridwiseReduction::RunWithIndices(in_grid_desc_m_k,
                                          workspace_desc_m_k,
                                          in_elementwise_op,
                                          acc_elementwise_op,
                                          BlkGroupSize,
                                          kBlockTileIterations,
                                          p_src_global,
                                          p_ws_values_global,
                                          p_ws_indices_global);
    };
};

template <typename InDataType,
          typename dstDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename WorkspaceDesc_M_K,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct GridwiseReduction_mk_to_mk_multiblock_partial_reduce
{
    static constexpr bool reorder_thread_cluster = (InSrcVectorDim == 0);

    static constexpr auto buffer1dDesc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<BlockSize>{}));

    using BlockwiseReduce = PartitionedBlockwiseReduction_1d_block_buffer<decltype(buffer1dDesc),
                                                                          AccDataType,
                                                                          BlockSize,
                                                                          MThreadClusterSize,
                                                                          KThreadClusterSize,
                                                                          reorder_thread_cluster,
                                                                          ReduceOperation,
                                                                          PropagateNan>;

    using BlockwiseReduceWithIndices =
        PartitionedBlockwiseReductionWithIndices_1d_block_buffer<decltype(buffer1dDesc),
                                                                 AccDataType,
                                                                 IndexDataType,
                                                                 BlockSize,
                                                                 MThreadClusterSize,
                                                                 KThreadClusterSize,
                                                                 reorder_thread_cluster,
                                                                 ReduceOperation,
                                                                 PropagateNan>;

    template <typename T>
    using PassThroughOp = tensor_operation::element_wise::UnaryIdentic<T, T>;

    static constexpr auto I0 = Number<0>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    using Accumulation =
        detail::accumulate_with_nan_check<PropagateNan, ReduceOperation, AccDataType>;

    using AccumulationWithIndices = detail::accumulate_with_indices_with_nan_check<PropagateNan,
                                                                                   ReduceOperation,
                                                                                   AccDataType,
                                                                                   IndexDataType>;

    __device__ static void Run(const InGridDesc_M_K& in_grid_desc_m_k,
                               const WorkspaceDesc_M_K& workspace_desc_m_k,
                               const InElementwiseOperation& in_elementwise_op,
                               const AccElementwiseOperation& acc_elementwise_op,
                               index_t BlkGroupSize,
                               index_t kBlockTileIterations,
                               const InDataType* const __restrict__ p_src_global,
                               AccDataType* const __restrict__ p_ws_values_global,
                               IndexDataType* const __restrict__ p_ws_indices_global)
    {
        (void)p_ws_indices_global;
        (void)acc_elementwise_op;

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        // LDS
        __shared__ AccDataType p_block_reduce_buffer[BlockSize];

        const auto in_global_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Global>(p_src_global,
                                                            in_grid_desc_m_k.GetElementSpaceSize(),
                                                            type_convert<InDataType>(zeroVal));
        auto workspace_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_ws_values_global, workspace_desc_m_k.GetElementSpaceSize());

        auto block_reduce_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accuValue_buf(I) = zeroVal; });

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;
        const index_t thread_m_cluster_id =
            reorder_thread_cluster ? thread_local_id % MThreadClusterSize
                                   : ((thread_local_id / KThreadClusterSize) % MThreadClusterSize);
        const index_t thread_k_cluster_id =
            reorder_thread_cluster ? ((thread_local_id / MThreadClusterSize) % KThreadClusterSize)
                                   : thread_local_id % KThreadClusterSize;

        const index_t reduceSizePerBlock = K_BlockTileSize * kBlockTileIterations;

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<
            InDataType,
            AccDataType,
            InGridDesc_M_K,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<InSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            InSrcVectorDim,
            InSrcVectorSize,
            1,
            false>(
            in_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             block_local_id * reduceSizePerBlock +
                                 thread_k_cluster_id * KThreadSliceSize));

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        index_t reducedTiles = 0;
        do
        {
            threadwise_src_load.Run(in_grid_desc_m_k,
                                    in_global_buf,
                                    ThreadBufferDesc,
                                    make_tuple(I0, I0),
                                    in_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                // do element-wise pre-reduction operation
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;
                    in_elementwise_op(in_thread_buf(offset), in_thread_buf(offset));
                });

                // reduce on each thread-local slice
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;
                    Accumulation::calculate(accuValue_buf(I), in_thread_buf[offset]);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            reducedTiles++;
        } while(reducedTiles < kBlockTileIterations);

        constexpr auto ReducedDataDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));

        // Each block executes multiple parallel reductions on the LDS, and due to the using of
        // vector_load, each block/thread is involved into multiple invarirant dimensions.
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(reorder_thread_cluster)
            {
                block_reduce_buf(thread_k_cluster_id * MThreadClusterSize + thread_m_cluster_id) =
                    accuValue_buf[I];
            }
            else
                block_reduce_buf(thread_m_cluster_id * KThreadClusterSize + thread_k_cluster_id) =
                    accuValue_buf[I];

            accuValue_buf(I) = zeroVal;

            __syncthreads();

            BlockwiseReduce::Reduce(
                block_reduce_buf, accuValue_buf(I), thread_m_cluster_id, thread_k_cluster_id);
        });

        if(thread_k_cluster_id == 0)
        {
            auto threadwise_workspace_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   AccDataType,
                                                   decltype(ReducedDataDesc),
                                                   WorkspaceDesc_M_K,
                                                   PassThroughOp<AccDataType>,
                                                   Sequence<MThreadSliceSize, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(
                    workspace_desc_m_k,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     block_local_id),
                    PassThroughOp<AccDataType>{});

            threadwise_workspace_store.Run(ReducedDataDesc,
                                           make_tuple(I0, I0),
                                           accuValue_buf,
                                           workspace_desc_m_k,
                                           workspace_global_buf);
        }
    };

    __device__ static void RunWithIndices(const InGridDesc_M_K& in_grid_desc_m_k,
                                          const WorkspaceDesc_M_K& workspace_desc_m_k,
                                          const InElementwiseOperation& in_elementwise_op,
                                          const AccElementwiseOperation& acc_elementwise_op,
                                          index_t BlkGroupSize,
                                          index_t kBlockTileIterations,
                                          const InDataType* const __restrict__ p_src_global,
                                          AccDataType* const __restrict__ p_ws_values_global,
                                          IndexDataType* const __restrict__ p_ws_indices_global)
    {
        (void)acc_elementwise_op;

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        // LDS
        __shared__ AccDataType p_block_reduce_val_buffer[BlockSize];
        __shared__ index_t p_block_reduce_idx_buffer[BlockSize];

        const auto in_global_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Global>(p_src_global,
                                                            in_grid_desc_m_k.GetElementSpaceSize(),
                                                            type_convert<InDataType>(zeroVal));
        auto workspace_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_ws_values_global, workspace_desc_m_k.GetElementSpaceSize());
        auto workspace_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_ws_indices_global, workspace_desc_m_k.GetElementSpaceSize());

        auto block_reduce_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_val_buffer, BlockSize);
        auto block_reduce_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_idx_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_val_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     IndexDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, IndexDataType, MThreadSliceSize, true> accuIndex_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;
        const index_t thread_m_cluster_id =
            reorder_thread_cluster ? thread_local_id % MThreadClusterSize
                                   : ((thread_local_id / KThreadClusterSize) % MThreadClusterSize);
        const index_t thread_k_cluster_id =
            reorder_thread_cluster ? ((thread_local_id / MThreadClusterSize) % KThreadClusterSize)
                                   : thread_local_id % KThreadClusterSize;

        const index_t reduceSizePerBlock = K_BlockTileSize * kBlockTileIterations;

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<
            InDataType,
            AccDataType,
            InGridDesc_M_K,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<InSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            InSrcVectorDim,
            InSrcVectorSize,
            1,
            false>(
            in_grid_desc_m_k,
            make_multi_index(blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                             block_local_id * reduceSizePerBlock +
                                 thread_k_cluster_id * KThreadSliceSize));

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        index_t indexOffset = block_local_id * reduceSizePerBlock;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        index_t reducedTiles = 0;
        do
        {
            // load the thread slice
            threadwise_src_load.Run(in_grid_desc_m_k,
                                    in_global_buf,
                                    ThreadBufferDesc,
                                    make_tuple(I0, I0),
                                    in_thread_val_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;

                    // initialize the indices for the per-thread to-reduce values
                    in_thread_idx_buf(offset) =
                        indexOffset + thread_k_cluster_id * KThreadSliceSize + J();

                    // do element-wise pre-reduction operation
                    in_elementwise_op(in_thread_val_buf(offset), in_thread_val_buf(offset));
                });

                AccDataType tmpValue   = zeroVal;
                IndexDataType tmpIndex = 0;

                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;

                    // reduce on the dim1 thread slice
                    AccumulationWithIndices::calculate(
                        tmpValue, in_thread_val_buf[offset], tmpIndex, in_thread_idx_buf[offset]);
                });

                // store thread local value to LDS for parallel reduction
                if constexpr(reorder_thread_cluster)
                {
                    block_reduce_val_buf(thread_k_cluster_id * MThreadClusterSize +
                                         thread_m_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_k_cluster_id * MThreadClusterSize +
                                         thread_m_cluster_id) = tmpIndex;
                }
                else
                {
                    block_reduce_val_buf(thread_m_cluster_id * KThreadClusterSize +
                                         thread_k_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_m_cluster_id * KThreadClusterSize +
                                         thread_k_cluster_id) = tmpIndex;
                }

                __syncthreads();

                BlockwiseReduceWithIndices::Reduce(block_reduce_val_buf,
                                                   block_reduce_idx_buf,
                                                   tmpValue,
                                                   tmpIndex,
                                                   thread_m_cluster_id,
                                                   thread_k_cluster_id);

                AccumulationWithIndices::calculate(
                    accuValue_buf(I), tmpValue, accuIndex_buf(I), tmpIndex);
            });

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            indexOffset += K_BlockTileSize;

            reducedTiles++;
        } while(reducedTiles < kBlockTileIterations);

        constexpr auto ReducedDataDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));

        if(thread_k_cluster_id == 0)
        {
            auto threadwise_workspace_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   AccDataType,
                                                   decltype(ReducedDataDesc),
                                                   WorkspaceDesc_M_K,
                                                   PassThroughOp<AccDataType>,
                                                   Sequence<MThreadSliceSize, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(
                    workspace_desc_m_k,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     block_local_id),
                    PassThroughOp<AccDataType>{});

            auto threadwise_workspace_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<IndexDataType,
                                                   IndexDataType,
                                                   decltype(ReducedDataDesc),
                                                   WorkspaceDesc_M_K,
                                                   PassThroughOp<IndexDataType>,
                                                   Sequence<MThreadSliceSize, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(
                    workspace_desc_m_k,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     block_local_id),
                    PassThroughOp<IndexDataType>{});

            threadwise_workspace_val_store.Run(ReducedDataDesc,
                                               make_tuple(I0, I0),
                                               accuValue_buf,
                                               workspace_desc_m_k,
                                               workspace_global_val_buf);
            threadwise_workspace_idx_store.Run(ReducedDataDesc,
                                               make_tuple(I0, I0),
                                               accuIndex_buf,
                                               workspace_desc_m_k,
                                               workspace_global_idx_buf);
        }
    };
};

} // namespace ck
#endif
