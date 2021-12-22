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
#include "reduction_functions_threadwise.hpp"
#include "reduction_functions_blockwise.hpp"

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template<typename GridwiseReduction, int RunId, typename inType, typename outType, typename src2dDescType, typename dst1dDescType>
__global__ void kernel_reduce_blockwise( const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               int origReduceLen,
                               inType alpha,
                               const inType* const __restrict__ p_src_global,
                               inType beta,
                               inType* const __restrict__ p_dst_global,
                               const int* const __restrict__ ws_indices_global,
                               int* const __restrict__ indices_global)
{
     GridwiseReduction::template Run<RunId>(src2dDesc, dst1dDesc, origReduceLen, alpha, p_src_global, beta, p_dst_global, ws_indices_global, indices_global);
};

template <typename srcDataType,
          typename dstDataType,
          typename compType,
          typename src2dDescType,
          typename dst1dDescType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          index_t BlockSize,
          index_t dim0_thread_cluster_length,
          index_t dim1_thread_cluster_length,
          index_t dim0_thread_slice_length,
          index_t dim1_thread_slice_length,
          index_t dim0_max_vector_size,
          index_t dim1_max_vector_size,
          bool isFirstCall,
          bool isLastCall>
struct GridwiseReduction_xy_to_x_blockwise
{
    static constexpr index_t dim0_VectorSize =
        math::gcd(dim0_thread_slice_length, dim0_max_vector_size);
    static constexpr index_t dim1_VectorSize =
        math::gcd(dim1_thread_slice_length, dim1_max_vector_size);

    static constexpr bool dim0_is_fastest = ((dim1_max_vector_size == 1) && (dim0_max_vector_size > 1));
    static constexpr bool reorder_thread_cluster = dim0_is_fastest;

    using opReduce = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType =
        typename reduce_unary_operator<compType, op, isFirstCall, isLastCall>::preUnaryOp;
    using posUnaryOpType =
        typename reduce_unary_operator<compType, op, isFirstCall, isLastCall>::posUnaryOp;

    static constexpr auto buffer1dDesc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<BlockSize>{}));

    using blockwise_reduce =
        PartitionedBlockwiseReduction_1d_block_buffer<decltype(buffer1dDesc),
                                                      BlockSize,
                                                      dim0_thread_cluster_length,
                                                      dim1_thread_cluster_length,
                                                      reorder_thread_clusters,
                                                      opReduce,
                                                      nanPropaOpt>;
    
    static constexpr auto I0 = Number<0>{};

    static constexpr index_t dim0_BlockTileSize = dim0_thread_cluster_length * dim0_thread_slice_length;
    static constexpr index_t dim1_BlockTileSize = dim1_thread_cluster_length * dim1_thread_slice_length;

    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    template <int RunId>
    __device__ static void Run(const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               int origReduceLen,
                               srcDataType alpha,
                               const srcDataType* const __restrict__ p_src_global,
                               dstDataType beta,
                               dstDataType* const __restrict__ p_dst_global,
                               const int* const __restrict__ ws_indices_global,
                               int* const __restrict__ indices_global);

    template <>
    __device__ static void Run<1>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;
        (void)indices_global;

        // LDS
        __shared__ compType p_block_reduce_buffer[BlockSize];

        const auto zeroVal = opReduce::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_length * dim1_thread_slice_length, true> in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_length, true> accuValue_buf;

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) { accuValue_buf(I) = zeroVal; });

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);
        const posUnaryOpType posUnaryOp(divider);

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();
        const index_t thread_dim0_cluster_id = reorder_thread_clusters
                ? thread_local_id % dim0_thread_cluster_length
                : ((thread_local_id / dim1_thread_cluster_length) % dim0_thread_cluster_length);
        const index_t thread_dim1_cluster_id = reorder_thread_clusters
                ? ((thread_local_id / dim0_thread_cluster_length) % dim1_thread_cluster_length)
                : thread_local_id % dim1_thread_cluster_length;

        using ThreadBufferLengths = Sequence<dim0_thread_slice_length, dim1_thread_slice_length>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<dim0_thread_slice_length>{}, Number<dim1_thread_slice_length>{}));
	
        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2 < srcDataType, compType,
             src2dDescType, decltype(ThreadBufferDesc), ThreadBufferLengths,
             typename conditional<dim0_is_fastest, Sequence<1, 0>, Sequence<0, 1>>::type,
             dim0_is_fastest ? 0 : 1, dim0_is_fastest ? dim0_VectorSize : dim1_VectorSize, 1,
             false > (src2dDesc, make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length, thread_dim1_cluster_id * dim1_thread_slice_length));

        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_BlockTileSize);

        const index_t toReduceTiles = (toReduceLength + dim1_BlockTileSize - 1) / dim1_BlockTileSize;

        for(index_t reducedTiles = 0; reducedTiles < toReduceTiles; reducedTiles++)
        {
            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
                // do element-wise pre-reduction operation
                static_for<0, dim1_thread_slice_length, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_length>{} + J;
                    in_thread_buf(offset) = preUnaryOp(in_thread_buf[offset]);
                });

                // reduce on each thread-local slice
                static_for<0, dim1_thread_slice_length, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_length>{} + J;
                    binop::calculate(accuValue_buf(I), in_thread_buf[offset]);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);
        }

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<dim0_thread_slice_length>{}));

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
            if constexpr(reorder_thread_clusters)
                block_reduce_buf(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                 thread_dim0_cluster_id) = accuValue_buf[I];
            else
                block_reduce_buf(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                 thread_dim1_cluster_id) = accuValue_buf[I];

            accuValue_buf(I) = zeroVal;

            __syncthreads();

            blockwise_reduce::Reduce(
                block_reduce_buf, accuValue_buf(I), thread_dim0_cluster_id, thread_dim1_cluster_id);
        });

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
            if(thread_dim1_cluster_id == 0)
            {
                accuValue_buf(I) = posUnaryOp(accuValue_buf[I]);

                if(!float_equal_one{}(alpha))
                    accuValue_buf(I) *= type_convert<compType>{}(alpha);
            }
        });

        if(thread_dim0_cluster_id == 0)
        {
            if(!float_equal_zero{}(beta))
            {
                StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, dim0_thread_slice_length, true> priorDstValue_buf;

                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                     dstDataType,
                                                     dst1dDescType,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<dim0_thread_slice_length>,
                                                     Sequence<0>,
                                                     0,
                                                     dim0_VectorSize,
                                                     1,
                                                     false>(dst1dDesc,
                                                            make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length));

                threadwise_dst_load.Run(
                    dst1dDesc, dst_global_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

                static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
                     accuValue_buf(I) += type_convert<compType>(priorDstValue_buf[I] * beta);
		}); 
            }

            auto threadwise_dst_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   dstDataType,
                                                   decltype(ReducedDataDesc),
                                                   dst1dDescType,
                                                   Sequence<dim0_thread_slice_length>,
                                                   Sequence<0>,
                                                   0,
                                                   dim0_VectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   false>(dst1dDesc,
                                                          make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length));

            threadwise_dst_store.Run(
                ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_buf);
        }
    };

    template <>
    __device__ static void Run<2>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)ws_indices_global;

        // LDS
        __shared__ compType p_block_reduce_val_buffer[BlockSize];
        __shared__ int p_block_reduce_idx_buffer[BlockSize];

        const auto zeroVal = opReduce::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, dst1dDesc.GetElementSpaceSize());

        auto block_reduce_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_val_buffer, BlockSize);
        auto block_reduce_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_idx_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_length * dim1_thread_slice_length, true> in_thread_val_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, dim0_thread_slice_length * dim1_thread_slice_length, true> in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_length, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, dim0_thread_slice_length, true> accuIndex_buf;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();
        const index_t thread_dim0_cluster_id = reorder_thread_clusters
                ? thread_local_id % dim0_thread_cluster_length
                : ((thread_local_id / dim1_thread_cluster_length) % dim0_thread_cluster_length);
        const index_t thread_dim1_cluster_id = reorder_thread_clusters
                ? ((thread_local_id / dim0_thread_cluster_length) % dim1_thread_cluster_length)
                : thread_local_id % dim1_thread_cluster_length;

        using ThreadBufferLengths = Sequence<dim0_thread_slice_length, dim1_thread_slice_length>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<dim0_thread_slice_length>{}, Number<dim1_thread_slice_length>{}));

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2 < srcDataType, compType,
             src2dDescType, decltype(ThreadBufferDesc), ThreadBufferLengths,
             typename conditional<dim0_is_fastest, Sequence<1, 0>, Sequence<0, 1>>::type,
             dim0_is_fastest ? 0 : 1, dim0_is_fastest ? dim0_VectorSize : dim1_VectorSize, 1,
             false > (src2dDesc, make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length, thread_dim1_cluster_id * dim1_thread_slice_length));	

        int indexOffset = 0;

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_BlockTileSize);

        const index_t toReduceTiles = (toReduceLength + dim1_BlockTileSize - 1) / dim1_BlockTileSize;

        for(index_t reducedTiles = 0; reducedTiles < toReduceTiles; reducedTiles++)
        {
            // load the thread slice
            threadwise_src_load.Run(src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_val_buf);

            static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
                static_for<0, dim1_thread_slice_length, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_length>{} + J;

                    // initialize the indices for the per-thread to-reduce values
                    in_thread_idx_buf(offset) = indexOffset + thread_dim1_cluster_id * dim1_thread_slice_length + J();

                    // do element-wise pre-reduction operation
                    in_thread_val_buf(offset) = preUnaryOp(in_thread_val_buf[offset]);
                });

                compType tmpValue = zeroVal;
                int tmpIndex      = 0;

                static_for<0, dim1_thread_slice_length, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_length>{} + J;

                    // reduce on the dim1 thread slice
                    binop::calculate(tmpValue, in_thread_val_buf[offset], tmpIndex, in_thread_idx_buf[offset]);
                });

                // store thread local value to LDS for parallel reduction
                if constexpr(reorder_thread_clusters)
                {
                    block_reduce_val_buf(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                         thread_dim0_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                         thread_dim0_cluster_id) = tmpIndex;
                }
                else
                {
                    block_reduce_val_buf(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         thread_dim1_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         thread_dim1_cluster_id) = tmpIndex;
                }

                __syncthreads();

                blockwise_reduce::Reduce2(block_reduce_val_buf,
                                          block_reduce_idx_buf,
                                          tmpValue,
                                          tmpIndex,
                                          thread_dim0_cluster_id,
                                          thread_dim1_cluster_id);

                binop::calculate(accuValue_buf(I), tmpValue, accuIndex_buf(I), tmpIndex);
            });

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);

            indexOffset += dim1_BlockTileSize;
        }
	
        constexpr auto ReducedDataDesc = make_naive_tensor_descriptor_packed( make_tuple(Number<dim0_thread_slice_length>{}) );

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
            if(thread_dim1_cluster_id == 0)
            {
                accuValue_buf(I) = posUnaryOp(accuValue_buf[I]);

                if(!float_equal_one{}(alpha))
                    accuValue_buf(I) *= type_convert<compType>{}(alpha);
            }
        });

        if(thread_dim1_cluster_id == 0)
        {
            if(!float_equal_zero{}(beta))
            {
                StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, dim0_thread_cluster_length, true> priorDstValue_buf;

                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                     dstDataType,
                                                     dst1dDescType,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<dim0_thread_slice_length>,
                                                     Sequence<0>,
                                                     0,
                                                     dim0_VectorSize ,
                                                     1,
                                                     false>(dst1dDesc,
		                                            make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length)); 

                threadwise_dst_load.Run(dst1dDesc,
                                        dst_global_val_buf,
                                        ReducedDataDesc,
                                        make_tuple(I0),
                                        priorDstValue_buf);

                static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
                     accuValue_buf(I) += type_convert<compType>(priorDstValue_buf[I] * beta);
                });		
            }

            auto threadwise_dst_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   dstDataType,
                                                   decltype(ReducedDataDesc),
                                                   dst1dDescType,
                                                   Sequence<dim0_thread_slice_length>,
                                                   Sequence<0>,
                                                   0,
                                                   dim0_VectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   false>(dst1dDesc,
		                                            make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length)); 

            auto threadwise_dst_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<int,
                                                   int,
                                                   decltype(ReducedDataDesc),
                                                   dst1dDescType,
                                                   Sequence<dim0_thread_slice_length>,
                                                   Sequence<0>,
                                                   0,
                                                   dim0_VectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   false>(dst1dDesc,
		                                            make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length)); 

            threadwise_dst_val_store.Run(
                ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_val_buf);
            threadwise_dst_idx_store.Run(
                ReducedDataDesc, make_tuple(I0), accuIndex_buf, dst1dDesc, dst_global_idx_buf);
        }
    };

    template <>
    __device__ static void Run<3>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ ws_values_global,
                                  dstDataType beta,
                                  dstDataType* const __restrict__ p_dst_global,
                                  const int* const __restrict__ ws_indices_global,
                                  int* const __restrict__ indices_global)
    {
        (void)origReduceLen;

        // LDS
        __shared__ compType p_block_reduce_val_buffer[BlockSize];
        __shared__ int p_block_reduce_idx_buffer[BlockSize];

        const auto zeroVal = opReduce::GetReductionZeroVal();

       const auto src_global_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Global>(ws_values_global,
                                                            src2dDesc.GetElementSpaceSize(),
                                                            type_convert<srcDataType>{}(zeroVal));
        const auto src_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_indices_global, src2dDesc.GetElementSpaceSize());
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, dst1dDesc.GetElementSpaceSize());	

       auto block_reduce_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_val_buffer, BlockSize);
        auto block_reduce_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_idx_buffer, BlockSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_length * dim1_thread_slice_length, true> in_thread_val_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, dim0_thread_slice_length * dim1_thread_slice_length, true> in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_length, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, dim0_thread_slice_length, true> accuIndex_buf;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();
        const index_t thread_dim0_cluster_id = reorder_thread_clusters
                ? thread_local_id % dim0_thread_cluster_length
                : ((thread_local_id / dim1_thread_cluster_length) % dim0_thread_cluster_length);
        const index_t thread_dim1_cluster_id = reorder_thread_clusters
                ? ((thread_local_id / dim0_thread_cluster_length) % dim1_thread_cluster_length)
                : thread_local_id % dim1_thread_cluster_length;
	
        using ThreadBufferLengths = Sequence<dim0_thread_slice_length, dim1_thread_slice_length>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<dim0_thread_slice_length>{}, Number<dim1_thread_slice_length>{}));

        auto threadwise_src_val_load = ThreadwiseTensorSliceTransfer_v2 < srcDataType, compType,
             src2dDescType, decltype(ThreadBufferDesc), ThreadBufferLengths,
             typename conditional<dim0_is_fastest, Sequence<1, 0>, Sequence<0, 1>>::type,
             dim0_is_fastest ? 0 : 1, dim0_is_fastest ? dim0_VectorSize : dim1_VectorSize, 1,
             false > (src2dDesc, make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length, thread_dim1_cluster_id * dim1_thread_slice_length));

        auto threadwise_src_idx_load = ThreadwiseTensorSliceTransfer_v2<int, int,
             src2dDescType, decltype(ThreadBufferDesc), ThreadBufferLengths,
             typename conditional<dim0_is_fastest, Sequence<1, 0>, Sequence<0, 1>>::type,
             dim0_is_fastest ? 0 : 1, dim0_is_fastest ? dim0_VectorSize : dim1_VectorSize, 1,
             false > (src2dDesc, make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length, thread_dim1_cluster_id * dim1_thread_slice_length))	

        int indexOffset = 0;

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });
		     
        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_BlockTileSize);

        const index_t toReduceTiles = (toReduceLength + dim1_BlockTileSize - 1) / dim1_BlockTileSize;

        for(index_t reducedTiles = 0; reducedTiles < toReduceTiles; reducedTiles++)
        {
            // load the thread slice
            threadwise_src_val_load.Run(src2dDesc, src_global_val_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_val_buf);
            threadwise_src_idx_load.Run(src2dDesc, src_global_idx_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_idx_buf);

            static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
                compType tmpValue = zeroVal;
                int tmpIndex      = 0;

                static_for<0, dim1_thread_slice_length, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_length>{} + J;

                    // reduce on the dim1 thread slice
                    binop::calculate(tmpValue, in_thread_val_buf[offset], tmpIndex, in_thread_idx_buf[offset]);
                });

                // store thread local value to LDS for parallel reduction
                if constexpr(reorder_thread_clusters)
                {
                    block_reduce_val_buf(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                         thread_dim0_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim1_cluster_id * dim0_thread_cluster_length +
                                         thread_dim0_cluster_id) = tmpIndex;
                }
                else
                {
                    block_reduce_val_buf(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         thread_dim1_cluster_id) = tmpValue;
                    block_reduce_idx_buf(thread_dim0_cluster_id * dim1_thread_cluster_length +
                                         thread_dim1_cluster_id) = tmpIndex;
                }

                __syncthreads();

                blockwise_reduce::Reduce2(block_reduce_val_buf,
                                          block_reduce_idx_buf,
                                          tmpValue,
                                          tmpIndex,
                                          thread_dim0_cluster_id,
                                          thread_dim1_cluster_id);

                binop::calculate(accuValue_buf(I), tmpValue, accuIndex_buf(I), tmpIndex);
            });

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);

            indexOffset += dim1_BlockTileSize;
        }
		     
        constexpr auto ReducedDataDesc = make_naive_tensor_descriptor_packed( make_tuple(Number<dim0_thread_slice_length>{}) );

        static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
            if(thread_dim1_cluster_id == 0)
            {
                accuValue_buf(I) = posUnaryOp(accuValue_buf[I]);

                if(!float_equal_one{}(alpha))
                    accuValue_buf(I) *= type_convert<compType>{}(alpha);
            }
        });

        if(thread_dim1_cluster_id == 0)
        {
            if(!float_equal_zero{}(beta))
            {
                StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, dim0_thread_slice_length, true> priorDstValue_buf;

                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                     dstDataType,
                                                     dst1dDescType,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<dim0_thread_slice_length>,
                                                     Sequence<0>,
                                                     0,
                                                     dim0_VectorSize,
                                                     1,
                                                     true>(dst1dDesc,
		                                            make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length)); 


                threadwise_dst_load.Run(dst1dDesc,
                                        dst_global_val_buf,
                                        ReducedDataDesc,
                                        make_tuple(I0),
                                        priorDstValue_buf);

                static_for<0, dim0_thread_slice_length, 1>{}([&](auto I) {
                     accuValue_buf(I) += type_convert<compType>(priorDstValue_buf[I] * beta);
                });		
            }

            auto threadwise_dst_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   dstDataType,
                                                   decltype(ReducedDataDesc),
                                                   dst1dDescType,
                                                   Sequence<dim0_thread_slice_length>,
                                                   Sequence<0>,
                                                   0,
                                                   dim0_VectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(dst1dDesc,
		                                         make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length)); 

            auto threadwise_dst_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<int,
                                                   int,
                                                   decltype(ReducedDataDesc),
                                                   dst1dDescType,
                                                   Sequence<dim0_thread_slice_length>,
                                                   Sequence<0>,
                                                   0,
                                                   dim0_VectorSize,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(dst1dDesc,
		                                         make_multi_index(block_global_1d_id * dim0_BlockTileSize + thread_dim0_cluster_id * dim0_thread_slice_length)); 

            threadwise_dst_val_store.Run(ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_val_buf);
            threadwise_dst_idx_store.Run(ReducedDataDesc, make_tuple(I0), accuIndex_buf, dst1dDesc, dst_global_idx_buf);
        }
    };
};

} // namespace ck
#endif
