/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef CK_GRIDWISE_2D_REDUCTION_BLOCKWISE_HPP
#define CK_GRIDWISE_2D_REDUCTION_BLOCKWISE_HPP

#include "data_type.hpp"
#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_blockwise.hpp"

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseReduction,
          bool NeedIndices,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename In2dDescType,
          typename Out1dDescType,
          typename InElementwiseOperation,
          typename OutElementwiseOperation>
__global__ void kernel_reduce_blockwise(const In2dDescType in2dDesc,
                                        const Out1dDescType out1dDesc,
                                        const InElementwiseOperation inElementwiseOp,
                                        const OutElementwiseOperation accElementwiseOp,
                                        AccDataType alpha,
                                        const InDataType* const __restrict__ p_src_global,
                                        OutDataType beta,
                                        OutDataType* const __restrict__ p_dst_global,
                                        const int* const __restrict__ ws_indices_global,
                                        int* const __restrict__ indices_global)
{
    if constexpr(!NeedIndices)
    {
        GridwiseReduction::Run(in2dDesc,
                               out1dDesc,
                               inElementwiseOp,
                               accElementwiseOp,
                               alpha,
                               p_src_global,
                               beta,
                               p_dst_global,
                               ws_indices_global,
                               indices_global);
    }
    else
    {
        GridwiseReduction::RunWithIndices(in2dDesc,
                                          out1dDesc,
                                          inElementwiseOp,
                                          accElementwiseOp,
                                          alpha,
                                          p_src_global,
                                          beta,
                                          p_dst_global,
                                          ws_indices_global,
                                          indices_global);
    };
};

template <typename GridwiseReduction,
          bool NeedIndices,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename In2dDescType,
          typename Out1dDescType,
          typename InElementwiseOperation,
          typename OutElementwiseOperation>
__global__ void
kernel_reduce_blockwise_second_call(const In2dDescType in2dDesc,
                                    const Out1dDescType out1dDesc,
                                    const InElementwiseOperation inElementwiseOp,
                                    const OutElementwiseOperation accElementwiseOp,
                                    AccDataType alpha,
                                    const InDataType* const __restrict__ p_src_global,
                                    OutDataType beta,
                                    OutDataType* const __restrict__ p_dst_global,
                                    const int* const __restrict__ ws_indices_global,
                                    int* const __restrict__ indices_global)
{
    if constexpr(!NeedIndices)
    {
        GridwiseReduction::Run(in2dDesc,
                               out1dDesc,
                               inElementwiseOp,
                               accElementwiseOp,
                               alpha,
                               p_src_global,
                               beta,
                               p_dst_global,
                               ws_indices_global,
                               indices_global);
    }
    else
    {
        GridwiseReduction::RunSecondCallWithIndices(in2dDesc,
                                                    out1dDesc,
                                                    inElementwiseOp,
                                                    accElementwiseOp,
                                                    alpha,
                                                    p_src_global,
                                                    beta,
                                                    p_dst_global,
                                                    ws_indices_global,
                                                    indices_global);
    };
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename In2dDescType,
          typename Out1dDescType,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename OutElementwiseOperation,
          bool PropagateNan,
          bool BetaIsZero,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InVectorDim,
          index_t InVectorSize,
          index_t OutVectorSize>
struct GridwiseReduction_xy_to_x_blockwise
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
                               const OutElementwiseOperation& accElementwiseOp,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_src_global,
                               OutDataType beta,
                               OutDataType* const __restrict__ p_dst_global,
                               const int* const __restrict__ ws_indices_global,
                               int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;
        (void)indices_global;

        // LDS
        __shared__ AccDataType p_block_reduce_buffer[BlockSize];

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

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

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();
        const index_t thread_dim0_cluster_id =
            reorder_thread_cluster ? thread_local_id % MThreadClusterSize
                                   : ((thread_local_id / KThreadClusterSize) % MThreadClusterSize);
        const index_t thread_dim1_cluster_id =
            reorder_thread_cluster ? ((thread_local_id / MThreadClusterSize) % KThreadClusterSize)
                                   : thread_local_id % KThreadClusterSize;

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
            false>(in2dDesc,
                   make_multi_index(block_global_1d_id * M_BlockTileSize +
                                        thread_dim0_cluster_id * MThreadSliceSize,
                                    thread_dim1_cluster_id * KThreadSliceSize));

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        const index_t toReduceTiles = (toReduceLength + K_BlockTileSize - 1) / K_BlockTileSize;

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
            if constexpr(!BetaIsZero)
            {
                if(!float_equal_zero{}(beta))
                {
                    StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                        priorDstValue_buf;

                    auto threadwise_dst_load =
                        ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                         OutDataType,
                                                         Out1dDescType,
                                                         decltype(ReducedDataDesc),
                                                         Sequence<MThreadSliceSize>,
                                                         Sequence<0>,
                                                         0,
                                                         OutVectorSize,
                                                         1,
                                                         false>(
                            out1dDesc,
                            make_multi_index(block_global_1d_id * M_BlockTileSize +
                                             thread_dim0_cluster_id * MThreadSliceSize));

                    threadwise_dst_load.Run(out1dDesc,
                                            dst_global_buf,
                                            ReducedDataDesc,
                                            make_tuple(I0),
                                            priorDstValue_buf);

                    static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                        accuValue_buf(I) += type_convert<AccDataType>(priorDstValue_buf[I] * beta);
                    });
                };
            };

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
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(
                    out1dDesc,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_dim0_cluster_id * MThreadSliceSize),
                    PassThroughOp<AccDataType>{});

            threadwise_dst_store.Run(
                ReducedDataDesc, make_tuple(I0), accuValue_buf, out1dDesc, dst_global_buf);
        }
    };

    __device__ static void RunWithIndices(const In2dDescType& in2dDesc,
                                          const Out1dDescType& out1dDesc,
                                          const InElementwiseOperation& inElementwiseOp,
                                          const OutElementwiseOperation& accElementwiseOp,
                                          AccDataType alpha,
                                          const InDataType* const __restrict__ p_src_global,
                                          OutDataType beta,
                                          OutDataType* const __restrict__ p_dst_global,
                                          const int* const __restrict__ ws_indices_global,
                                          int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;

        // LDS
        __shared__ AccDataType p_block_reduce_val_buffer[BlockSize];
        __shared__ int p_block_reduce_idx_buffer[BlockSize];

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, in2dDesc.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, out1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, out1dDesc.GetElementSpaceSize());

        auto block_reduce_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_val_buffer, BlockSize);
        auto block_reduce_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_idx_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_val_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, MThreadSliceSize, true> accuIndex_buf;

        const auto toReduceLength = in2dDesc.GetLength(Number<1>{});

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();
        const index_t thread_dim0_cluster_id =
            reorder_thread_cluster ? thread_local_id % MThreadClusterSize
                                   : ((thread_local_id / KThreadClusterSize) % MThreadClusterSize);
        const index_t thread_dim1_cluster_id =
            reorder_thread_cluster ? ((thread_local_id / MThreadClusterSize) % KThreadClusterSize)
                                   : thread_local_id % KThreadClusterSize;

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
            false>(in2dDesc,
                   make_multi_index(block_global_1d_id * M_BlockTileSize +
                                        thread_dim0_cluster_id * MThreadSliceSize,
                                    thread_dim1_cluster_id * KThreadSliceSize));

        int indexOffset = 0;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        const index_t toReduceTiles = (toReduceLength + K_BlockTileSize - 1) / K_BlockTileSize;

        index_t reducedTiles = 0;
        do
        {
            // load the thread slice
            threadwise_src_load.Run(
                in2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_val_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;

                    // initialize the indices for the per-thread to-reduce values
                    in_thread_idx_buf(offset) =
                        indexOffset + thread_dim1_cluster_id * KThreadSliceSize + J();

                    // do element-wise pre-reduction operation
                    inElementwiseOp(in_thread_val_buf(offset), in_thread_val_buf(offset));
                });

                AccDataType tmpValue = zeroVal;
                int tmpIndex         = 0;

                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;

                    // reduce on the dim1 thread slice
                    BinaryOperation::calculate(
                        tmpValue, in_thread_val_buf[offset], tmpIndex, in_thread_idx_buf[offset]);
                });

                // store thread local value to LDS for parallel reduction
                if constexpr(reorder_thread_cluster)
                {
                    block_reduce_val_buf(thread_dim1_cluster_id * MThreadClusterSize +
                                         thread_dim0_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim1_cluster_id * MThreadClusterSize +
                                         thread_dim0_cluster_id) = tmpIndex;
                }
                else
                {
                    block_reduce_val_buf(thread_dim0_cluster_id * KThreadClusterSize +
                                         thread_dim1_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim0_cluster_id * KThreadClusterSize +
                                         thread_dim1_cluster_id) = tmpIndex;
                }

                __syncthreads();

                blockwise_reduce::Reduce2(block_reduce_val_buf,
                                          block_reduce_idx_buf,
                                          tmpValue,
                                          tmpIndex,
                                          thread_dim0_cluster_id,
                                          thread_dim1_cluster_id);

                BinaryOperation::calculate(accuValue_buf(I), tmpValue, accuIndex_buf(I), tmpIndex);
            });

            threadwise_src_load.MoveSrcSliceWindow(in2dDesc, in_thread_copy_step);

            indexOffset += K_BlockTileSize;
            reducedTiles++;
        } while(reducedTiles < toReduceTiles);

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if(thread_dim1_cluster_id == 0)
            {
                // for indiced operation, accElementwiseOp shoud do nothing
                accElementwiseOp(accuValue_buf(I), accuValue_buf(I));

                accuValue_buf(I) *= alpha;
            }
        });

        if(thread_dim1_cluster_id == 0)
        {
            if constexpr(!BetaIsZero)
            {
                if(!float_equal_zero{}(beta))
                {
                    StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                        priorDstValue_buf;

                    auto threadwise_dst_load =
                        ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                         OutDataType,
                                                         Out1dDescType,
                                                         decltype(ReducedDataDesc),
                                                         Sequence<MThreadSliceSize>,
                                                         Sequence<0>,
                                                         0,
                                                         OutVectorSize,
                                                         1,
                                                         false>(
                            out1dDesc,
                            make_multi_index(block_global_1d_id * M_BlockTileSize +
                                             thread_dim0_cluster_id * MThreadSliceSize));

                    threadwise_dst_load.Run(out1dDesc,
                                            dst_global_val_buf,
                                            ReducedDataDesc,
                                            make_tuple(I0),
                                            priorDstValue_buf);

                    static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                        accuValue_buf(I) += type_convert<AccDataType>(priorDstValue_buf[I] * beta);
                    });
                };
            };

            auto threadwise_dst_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   OutDataType,
                                                   decltype(ReducedDataDesc),
                                                   Out1dDescType,
                                                   PassThroughOp<AccDataType>,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutVectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   false>(
                    out1dDesc,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_dim0_cluster_id * MThreadSliceSize),
                    PassThroughOp<AccDataType>{});

            auto threadwise_dst_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<int,
                                                   int,
                                                   decltype(ReducedDataDesc),
                                                   Out1dDescType,
                                                   PassThroughOp<int>,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutVectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   false>(
                    out1dDesc,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_dim0_cluster_id * MThreadSliceSize),
                    PassThroughOp<int>{});

            threadwise_dst_val_store.Run(
                ReducedDataDesc, make_tuple(I0), accuValue_buf, out1dDesc, dst_global_val_buf);
            threadwise_dst_idx_store.Run(
                ReducedDataDesc, make_tuple(I0), accuIndex_buf, out1dDesc, dst_global_idx_buf);
        }
    };

    __device__ static void
    RunSecondCallWithIndices(const In2dDescType& in2dDesc,
                             const Out1dDescType& out1dDesc,
                             const InElementwiseOperation inElementwiseOp,
                             const OutElementwiseOperation accElementwiseOp,
                             AccDataType alpha,
                             const InDataType* const __restrict__ ws_values_global,
                             OutDataType beta,
                             OutDataType* const __restrict__ p_dst_global,
                             const int* const __restrict__ ws_indices_global,
                             int* const __restrict__ indices_global)
    {
        (void)inElementwiseOp;

        // LDS
        __shared__ AccDataType p_block_reduce_val_buffer[BlockSize];
        __shared__ int p_block_reduce_idx_buffer[BlockSize];

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        const auto src_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_values_global, in2dDesc.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        const auto src_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_indices_global, in2dDesc.GetElementSpaceSize());
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, out1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, out1dDesc.GetElementSpaceSize());

        auto block_reduce_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_val_buffer, BlockSize);
        auto block_reduce_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_idx_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_val_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, MThreadSliceSize, true> accuIndex_buf;

        const auto toReduceLength = in2dDesc.GetLength(Number<1>{});

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();
        const index_t thread_dim0_cluster_id =
            reorder_thread_cluster ? thread_local_id % MThreadClusterSize
                                   : ((thread_local_id / KThreadClusterSize) % MThreadClusterSize);
        const index_t thread_dim1_cluster_id =
            reorder_thread_cluster ? ((thread_local_id / MThreadClusterSize) % KThreadClusterSize)
                                   : thread_local_id % KThreadClusterSize;

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto threadwise_src_val_load = ThreadwiseTensorSliceTransfer_v2<
            InDataType,
            AccDataType,
            In2dDescType,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<InVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            InVectorDim,
            InVectorSize,
            1,
            false>(in2dDesc,
                   make_multi_index(block_global_1d_id * M_BlockTileSize +
                                        thread_dim0_cluster_id * MThreadSliceSize,
                                    thread_dim1_cluster_id * KThreadSliceSize));

        auto threadwise_src_idx_load = ThreadwiseTensorSliceTransfer_v2<
            int,
            int,
            In2dDescType,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<InVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            InVectorDim,
            InVectorSize,
            1,
            false>(in2dDesc,
                   make_multi_index(block_global_1d_id * M_BlockTileSize +
                                        thread_dim0_cluster_id * MThreadSliceSize,
                                    thread_dim1_cluster_id * KThreadSliceSize));

        int indexOffset = 0;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        const index_t toReduceTiles = (toReduceLength + K_BlockTileSize - 1) / K_BlockTileSize;

        index_t reducedTiles = 0;
        do
        {
            // load the thread slice
            threadwise_src_val_load.Run(in2dDesc,
                                        src_global_val_buf,
                                        ThreadBufferDesc,
                                        make_tuple(I0, I0),
                                        in_thread_val_buf);
            threadwise_src_idx_load.Run(in2dDesc,
                                        src_global_idx_buf,
                                        ThreadBufferDesc,
                                        make_tuple(I0, I0),
                                        in_thread_idx_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                AccDataType tmpValue = zeroVal;
                int tmpIndex         = 0;

                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;

                    // reduce on the dim1 thread slice
                    BinaryOperation::calculate(
                        tmpValue, in_thread_val_buf[offset], tmpIndex, in_thread_idx_buf[offset]);
                });

                // store thread local value to LDS for parallel reduction
                if constexpr(reorder_thread_cluster)
                {
                    block_reduce_val_buf(thread_dim1_cluster_id * MThreadClusterSize +
                                         thread_dim0_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim1_cluster_id * MThreadClusterSize +
                                         thread_dim0_cluster_id) = tmpIndex;
                }
                else
                {
                    block_reduce_val_buf(thread_dim0_cluster_id * KThreadClusterSize +
                                         thread_dim1_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim0_cluster_id * KThreadClusterSize +
                                         thread_dim1_cluster_id) = tmpIndex;
                }

                __syncthreads();

                blockwise_reduce::Reduce2(block_reduce_val_buf,
                                          block_reduce_idx_buf,
                                          tmpValue,
                                          tmpIndex,
                                          thread_dim0_cluster_id,
                                          thread_dim1_cluster_id);

                BinaryOperation::calculate(accuValue_buf(I), tmpValue, accuIndex_buf(I), tmpIndex);
            });

            threadwise_src_val_load.MoveSrcSliceWindow(in2dDesc, in_thread_copy_step);
            threadwise_src_idx_load.MoveSrcSliceWindow(in2dDesc, in_thread_copy_step);

            indexOffset += K_BlockTileSize;
            reducedTiles++;
        } while(reducedTiles < toReduceTiles);

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if(thread_dim1_cluster_id == 0)
            {
                // for indiced operation, accElementwiseOp shoud do nothing
                accElementwiseOp(accuValue_buf(I), accuValue_buf(I));

                accuValue_buf(I) *= alpha;
            }
        });

        if(thread_dim1_cluster_id == 0)
        {
            if constexpr(!BetaIsZero)
            {
                if(!float_equal_zero{}(beta))
                {
                    StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                        priorDstValue_buf;

                    auto threadwise_dst_load =
                        ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                         OutDataType,
                                                         Out1dDescType,
                                                         decltype(ReducedDataDesc),
                                                         Sequence<MThreadSliceSize>,
                                                         Sequence<0>,
                                                         0,
                                                         OutVectorSize,
                                                         1,
                                                         true>(
                            out1dDesc,
                            make_multi_index(block_global_1d_id * M_BlockTileSize +
                                             thread_dim0_cluster_id * MThreadSliceSize));

                    threadwise_dst_load.Run(out1dDesc,
                                            dst_global_val_buf,
                                            ReducedDataDesc,
                                            make_tuple(I0),
                                            priorDstValue_buf);

                    static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                        accuValue_buf(I) += type_convert<AccDataType>(priorDstValue_buf[I] * beta);
                    });
                };
            };

            auto threadwise_dst_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   OutDataType,
                                                   decltype(ReducedDataDesc),
                                                   Out1dDescType,
                                                   PassThroughOp<AccDataType>,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutVectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(
                    out1dDesc,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_dim0_cluster_id * MThreadSliceSize),
                    PassThroughOp<AccDataType>{});

            auto threadwise_dst_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<int,
                                                   int,
                                                   decltype(ReducedDataDesc),
                                                   Out1dDescType,
                                                   PassThroughOp<int>,
                                                   Sequence<MThreadSliceSize>,
                                                   Sequence<0>,
                                                   0,
                                                   OutVectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(
                    out1dDesc,
                    make_multi_index(block_global_1d_id * M_BlockTileSize +
                                     thread_dim0_cluster_id * MThreadSliceSize),
                    PassThroughOp<int>{});

            threadwise_dst_val_store.Run(
                ReducedDataDesc, make_tuple(I0), accuValue_buf, out1dDesc, dst_global_val_buf);
            threadwise_dst_idx_store.Run(
                ReducedDataDesc, make_tuple(I0), accuIndex_buf, out1dDesc, dst_global_idx_buf);
        }
    };
};

} // namespace ck
#endif
