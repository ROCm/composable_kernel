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
#ifndef CK_GRIDWISE_2D_REDUCTION_THREADWISE_HPP
#define CK_GRIDWISE_2D_REDUCTION_THREADWISE_HPP

#include "data_type.hpp"
#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_threadwise.hpp"

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseReduction,
          bool need_indices,
          typename inType,
          typename outType,
          typename src2dDescType,
          typename dst1dDescType>
__global__ void kernel_reduce_threadwise(const src2dDescType src2dDesc,
                                         const dst1dDescType dst1dDesc,
                                         int origReduceLen,
                                         inType alpha,
                                         const inType* const __restrict__ p_src_global,
                                         outType beta,
                                         outType* const __restrict__ p_dst_global,
                                         int* const __restrict__ indices_global)
{
    if constexpr(!need_indices)
        GridwiseReduction::Run(src2dDesc,
                               dst1dDesc,
                               origReduceLen,
                               alpha,
                               p_src_global,
                               beta,
                               p_dst_global,
                               indices_global);
    else
        GridwiseReduction::RunWithIndices(src2dDesc,
                                          dst1dDesc,
                                          origReduceLen,
                                          alpha,
                                          p_src_global,
                                          beta,
                                          p_dst_global,
                                          indices_global);
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
          index_t dim0_thread_cluster_size,
          index_t dim1_thread_cluster_size,
          index_t dim0_thread_slice_size,
          index_t dim1_thread_slice_size,
          bool dim0_is_fastest,
          index_t dim0_vector_size,
          index_t dim1_vector_size>
struct GridwiseReduction_xy_to_x_threadwise
{
    using opReduce       = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, op, true, true>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, op, true, true>::posUnaryOp;

    template <typename T>
    using PassThroughOp = reduce::unary_identic<T, false>;

    static constexpr auto I0 = Number<0>{};

    __device__ static void Run(const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               int origReduceLen,
                               srcDataType alpha,
                               const srcDataType* const __restrict__ p_src_global,
                               dstDataType beta,
                               dstDataType* const __restrict__ p_dst_global,
                               int* const __restrict__ indices_global)
    {
        (void)indices_global;

        const auto zeroVal = opReduce::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim1_thread_slice_size, true>
            in_thread_buf;

        using threadwise_reduce = ThreadReduce<decltype(in_thread_buf), opReduce, nanPropaOpt>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;

        accuValue_buf(I0) = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);
        const posUnaryOpType posUnaryOp(divider);

        using ThreadBufferLengths       = Sequence<1, dim1_thread_slice_size>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<dim1_thread_slice_size>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<srcDataType,
                                                                    compType,
                                                                    src2dDescType,
                                                                    decltype(ThreadBufferDesc),
                                                                    ThreadBufferLengths,
                                                                    Sequence<0, 1>,
                                                                    1,
                                                                    dim1_vector_size,
                                                                    1,
                                                                    false>(
            src2dDesc, make_multi_index(thread_global_1d_id, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_thread_slice_size);

        index_t reducedLength = 0;
        do
        {
            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            // do element-wise pre-reduction operation
            threadwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the reduction on the Thread Buffer
            threadwise_reduce::Reduce(in_thread_buf, accuValue_buf(I0));

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);

            reducedLength += dim1_thread_slice_size;
        } while(reducedLength < toReduceLength);

        accuValue_buf(I0) = posUnaryOp(accuValue_buf[I0]);

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        if(!float_equal_one{}(alpha))
            accuValue_buf(I0) *= type_convert<compType>(alpha);

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                                        dstDataType,
                                                                        dst1dDescType,
                                                                        decltype(ReducedDataDesc),
                                                                        Sequence<1>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        true>(
                dst1dDesc, make_multi_index(thread_global_1d_id));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            accuValue_buf(I0) += priorDstValue_buf[I0] * beta;
        }

        auto threadwise_dst_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               PassThroughOp<dstDataType>,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               true>(
                dst1dDesc, make_multi_index(thread_global_1d_id), PassThroughOp<dstDataType>{});

        threadwise_dst_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_buf);
    };

    __device__ static void RunWithIndices(const src2dDescType& src2dDesc,
                                          const dst1dDescType& dst1dDesc,
                                          int origReduceLen,
                                          srcDataType alpha,
                                          const srcDataType* const __restrict__ p_src_global,
                                          dstDataType beta,
                                          dstDataType* const __restrict__ p_dst_global,
                                          int* const __restrict__ indices_global)
    {
        const auto zeroVal = opReduce::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim1_thread_slice_size, true>
            in_thread_buf;

        using threadwise_reduce = ThreadReduce<decltype(in_thread_buf), opReduce, nanPropaOpt>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, 1, true> accuIndex_buf;

        accuValue_buf(I0) = zeroVal;
        accuIndex_buf(I0) = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        using ThreadBufferLengths       = Sequence<1, dim1_thread_slice_size>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<dim1_thread_slice_size>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<srcDataType,
                                                                    compType,
                                                                    src2dDescType,
                                                                    decltype(ThreadBufferDesc),
                                                                    ThreadBufferLengths,
                                                                    Sequence<0, 1>,
                                                                    1,
                                                                    dim1_vector_size,
                                                                    1,
                                                                    false>(
            src2dDesc, make_multi_index(thread_global_1d_id, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_thread_slice_size);

        index_t indexStart    = 0;
        index_t reducedLength = 0;
        do
        {
            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            threadwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the reduction on the Thread Buffer
            threadwise_reduce::Reduce2(
                in_thread_buf, accuValue_buf(I0), accuIndex_buf(I0), indexStart);

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);

            indexStart += dim1_thread_slice_size;
            reducedLength += dim1_thread_slice_size;
        } while(reducedLength < toReduceLength);

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        if(!float_equal_one{}(alpha))
            accuValue_buf(I0) *= type_convert<compType>(alpha);

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                                        dstDataType,
                                                                        dst1dDescType,
                                                                        decltype(ReducedDataDesc),
                                                                        Sequence<1>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        false>(
                dst1dDesc, make_multi_index(thread_global_1d_id));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_val_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            accuValue_buf(I0) += type_convert<compType>(priorDstValue_buf[I0] * beta);
        }

        auto threadwise_dst_val_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               PassThroughOp<dstDataType>,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                dst1dDesc, make_multi_index(thread_global_1d_id), PassThroughOp<dstDataType>{});

        auto threadwise_dst_idx_store =
            ThreadwiseTensorSliceTransfer_v1r3<int,
                                               int,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               PassThroughOp<int>,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                dst1dDesc, make_multi_index(thread_global_1d_id), PassThroughOp<int>{});

        threadwise_dst_val_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_val_buf);
        threadwise_dst_idx_store.Run(
            ReducedDataDesc, make_tuple(I0), accuIndex_buf, dst1dDesc, dst_global_idx_buf);
    };
};

} // namespace ck
#endif
