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
#ifndef CK_GRIDWISE_2D_REDUCTION_MULTIBLOCK_ATOMIC_ADD_HPP
#define CK_GRIDWISE_2D_REDUCTION_MULTIBLOCK_ATOMIC_ADD_HPP

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_binop.hpp"
#include "reduction_functions_blockwise.hpp"

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseReduction,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename In2dDescType,
          typename Out1dDescType,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
__global__ void
kernel_reduce_multiblock_atocmi_add(const In2dDescType in2dDesc,
                                    const Out1dDescType out1dDesc,
                                    const InElementwiseOperation inElementwiseOp,
                                    const AccElementwiseOperation accElementwiseOp,
                                    int BlkGroupSize,
                                    AccDataType alpha,
                                    const InDataType* const __restrict__ p_src_global,
                                    OutDataType* const __restrict__ p_dst_global)
{
    GridwiseReduction::Run(in2dDesc,
                           out1dDesc,
                           inElementwiseOp,
                           accElementwiseOp,
                           BlkGroupSize,
                           alpha,
                           p_src_global,
                           p_dst_global);
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename In2dDescType,
          typename Out1dDescType,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InVectorDim,
          index_t InVectorSize,
          index_t OutVectorSize>
struct GridwiseReduction_xy_to_x_multiblock_atomic_add
{
    static constexpr bool reorder_thread_cluster = (InVectorDim == 0);

    static constexpr auto buffer1dDesc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<BlockSize>{}));

    using blockwise_reduce = PartitionedBlockwiseReduction_1d_block_buffer<decltype(buffer1dDesc),
                                                                           BlockSize,
                                                                           MThreadClusterSize,
                                                                           KThreadClusterSize,
                                                                           reorder_thread_cluster,
                                                                           ReduceOperation,
                                                                           PropagateNan>;

    template <typename T>
    using PassThroughOp = reduce::unary_identic<T, T>;

    static constexpr auto I0 = Number<0>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    using BinaryOperation =
        detail::binop_with_nan_check<PropagateNan, ReduceOperation, AccDataType>;

    __device__ static void Run(const In2dDescType& in2dDesc,
                               const Out1dDescType& out1dDesc,
                               const InElementwiseOperation& inElementwiseOp,
                               const AccElementwiseOperation& accElementwiseOp,
                               int BlkGroupSize,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_src_global,
                               OutDataType* const __restrict__ p_dst_global)
    {
        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        // LDS
        __shared__ AccDataType p_block_reduce_buffer[BlockSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, in2dDesc.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, out1dDesc.GetElementSpaceSize());

        auto block_reduce_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accuValue_buf(I) = zeroVal; });

        const auto toReduceLength = in2dDesc.GetLength(Number<1>{});

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;
        const index_t thread_dim0_cluster_id =
            reorder_thread_cluster ? thread_local_id % MThreadClusterSize
                                   : ((thread_local_id / KThreadClusterSize) % MThreadClusterSize);
        const index_t thread_dim1_cluster_id =
            reorder_thread_cluster ? ((thread_local_id / MThreadClusterSize) % KThreadClusterSize)
                                   : thread_local_id % KThreadClusterSize;

        const index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + K_BlockTileSize - 1) /
             K_BlockTileSize) *
            K_BlockTileSize;

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<
            InDataType,
            AccDataType,
            In2dDescType,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<InVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            InVectorDim,
            InVectorSize,
            1,
            false>(
            in2dDesc,
            make_multi_index(
                blkgroup_id * M_BlockTileSize + thread_dim0_cluster_id * MThreadSliceSize,
                block_local_id * reduceSizePerBlock + thread_dim1_cluster_id * KThreadSliceSize));

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        const index_t toReduceTiles = reduceSizePerBlock / K_BlockTileSize;

        index_t reducedTiles = 0;
        do
        {
            threadwise_src_load.Run(
                in2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                // do element-wise pre-reduction operation
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;
                    inElementwiseOp(in_thread_buf(offset), in_thread_buf(offset));
                });

                // reduce on each thread-local slice
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;
                    BinaryOperation::calculate(accuValue_buf(I), in_thread_buf[offset]);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(in2dDesc, in_thread_copy_step);

            reducedTiles++;
        } while(reducedTiles < toReduceTiles);

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

        // Each block executes multiple parallel reductions on the LDS, and by atomic-adding its
        // reduced output to the global location corresponding to each invariant dimension to get a
        // consistent reduced result for that invariant dimension. due to the using of vector_load,
        // each block/thread is involved into multiple invarirant dimensions.
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(reorder_thread_cluster)
                block_reduce_buf(thread_dim1_cluster_id * MThreadClusterSize +
                                 thread_dim0_cluster_id) = accuValue_buf[I];
            else
                block_reduce_buf(thread_dim0_cluster_id * KThreadClusterSize +
                                 thread_dim1_cluster_id) = accuValue_buf[I];

            accuValue_buf(I) = zeroVal;

            __syncthreads();

            blockwise_reduce::Reduce(
                block_reduce_buf, accuValue_buf(I), thread_dim0_cluster_id, thread_dim1_cluster_id);
        });

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if(thread_dim1_cluster_id == 0)
            {
                accElementwiseOp(accuValue_buf(I), accuValue_buf(I));

                accuValue_buf(I) *= alpha;
            }
        });

        if(thread_dim1_cluster_id == 0)
        {
            auto threadwise_dst_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   OutDataType,
                                                   decltype(ReducedDataDesc),
                                                   Out1dDescType,
                                                   PassThroughOp<AccDataType>,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutVectorSize,
                                                   InMemoryDataOperationEnum_t::AtomicAdd,
                                                   1,
                                                   true>(
                    out1dDesc,
                    make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_dim0_cluster_id * MThreadSliceSize),
                    PassThroughOp<AccDataType>{});

            threadwise_dst_store.Run(
                ReducedDataDesc, make_tuple(I0), accuValue_buf, out1dDesc, dst_global_buf);
        }
    };
};

template <index_t BlockSize, typename DataType, typename Global1dBufferDescType>
__global__ void kernel_buffer_set_value(const Global1dBufferDescType global1dBufferDesc,
                                        DataType* const __restrict__ p_global,
                                        DataType value)

{
    using PassThroughOp = reduce::unary_identic<DataType, DataType>;

    constexpr auto I0 = Number<0>{};

    const index_t thread_local_id = get_thread_local_1d_id();
    const index_t block_global_id = get_block_1d_id();

    const index_t thread_global_id = block_global_id * BlockSize + thread_local_id;

    StaticBuffer<AddressSpaceEnum_t::Vgpr, DataType, 1, true> value_buf;

    value_buf(I0) = value;

    constexpr auto valueBuffDesc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    auto global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_global, global1dBufferDesc.GetElementSpaceSize());

    if(thread_global_id < global1dBufferDesc.GetElementSize())
    {
        auto threadwise_store = ThreadwiseTensorSliceTransfer_v1r3<DataType,
                                                                   DataType,
                                                                   decltype(valueBuffDesc),
                                                                   Global1dBufferDescType,
                                                                   PassThroughOp,
                                                                   Sequence<1>,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperationEnum_t::Set,
                                                                   1,
                                                                   true>(
            global1dBufferDesc, make_multi_index(thread_global_id), PassThroughOp{});

        threadwise_store.Run(
            valueBuffDesc, make_tuple(I0), value_buf, global1dBufferDesc, global_buf);
    }
};

} // namespace ck
#endif
