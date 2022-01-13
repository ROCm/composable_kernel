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
#include "reduction_functions_binop.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseReduction,
          bool need_indices,
          typename inType,
          typename outType,
          typename src2dDescType,
          typename dst1dDescType,
          typename preUnaryOpType,
          typename posUnaryOpType>
__global__ void kernel_reduce_threadwise(const src2dDescType src2dDesc,
                                         const dst1dDescType dst1dDesc,
                                         const preUnaryOpType preUnaryOp,
                                         const posUnaryOpType posUnaryOp,
                                         inType alpha,
                                         const inType* const __restrict__ p_src_global,
                                         outType beta,
                                         outType* const __restrict__ p_dst_global,
                                         int* const __restrict__ indices_global)
{
    if constexpr(!need_indices)
        GridwiseReduction::Run(src2dDesc,
                               dst1dDesc,
                               preUnaryOp,
                               posUnaryOp,
                               alpha,
                               p_src_global,
                               beta,
                               p_dst_global,
                               indices_global);
    else
        GridwiseReduction::RunWithIndices(src2dDesc,
                                          dst1dDesc,
                                          preUnaryOp,
                                          posUnaryOp,
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
          typename opReduce,
          typename preUnaryOpType,
          typename posUnaryOpType,
          NanPropagation_t nanPropaOpt,
          index_t BlockSize,
          index_t dim0_thread_cluster_size,
          index_t dim1_thread_cluster_size,
          index_t dim0_thread_slice_size,
          index_t dim1_thread_slice_size,
          index_t vectorDim,
          index_t vectorSize>
struct GridwiseReduction_xy_to_x_threadwise
{
    template <typename T>
    using PassThroughOp = reduce::unary_identic<T, false>;

    static constexpr auto I0 = Number<0>{};

    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    __device__ static void Run(const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               const preUnaryOpType& preUnaryOp,
                               const posUnaryOpType& posUnaryOp,
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

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     compType,
                     dim0_thread_slice_size * dim1_thread_slice_size,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_size, true>
            accuValue_buf;

        static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) { accuValue_buf(I) = zeroVal; });

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<dim0_thread_slice_size, dim1_thread_slice_size>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<dim0_thread_slice_size>{}, Number<dim1_thread_slice_size>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<
            srcDataType,
            compType,
            src2dDescType,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<vectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            vectorDim,
            vectorSize,
            1,
            false>(src2dDesc, make_multi_index(thread_global_1d_id * dim0_thread_slice_size, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_thread_slice_size);

        index_t reducedLength = 0;
        do
        {
            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
                // do element-wise pre-reduction operation
                static_for<0, dim1_thread_slice_size, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_size>{} + J;
                    in_thread_buf(offset) = preUnaryOp(in_thread_buf[offset]);
                });

                // reduce on each thread-local slice
                static_for<0, dim1_thread_slice_size, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_size>{} + J;
                    binop::calculate(accuValue_buf(I), in_thread_buf[offset]);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);

            reducedLength += dim1_thread_slice_size;
        } while(reducedLength < toReduceLength);

        static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
            accuValue_buf(I) = posUnaryOp(accuValue_buf[I]);

            if(!float_equal_one{}(alpha))
                accuValue_buf(I) *= type_convert<compType>(alpha);
        });

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<dim0_thread_slice_size>{}));

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load =
                ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                 dstDataType,
                                                 dst1dDescType,
                                                 decltype(ReducedDataDesc),
                                                 Sequence<dim0_thread_slice_size>,
                                                 Sequence<0>,
                                                 0,
                                                 1,
                                                 1,
                                                 true>(
                    dst1dDesc, make_multi_index(thread_global_1d_id * dim0_thread_slice_size));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, dim0_thread_slice_size, true>
                priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
                accuValue_buf(I) += type_convert<compType>(priorDstValue_buf[I] * beta);
            });
        }

        auto threadwise_dst_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               PassThroughOp<dstDataType>,
                                               Sequence<dim0_thread_slice_size>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               true>(
                dst1dDesc,
                make_multi_index(thread_global_1d_id * dim0_thread_slice_size),
                PassThroughOp<dstDataType>{});

        threadwise_dst_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_buf);
    };

    __device__ static void RunWithIndices(const src2dDescType& src2dDesc,
                                          const dst1dDescType& dst1dDesc,
                                          const preUnaryOpType& preUnaryOp,
                                          const posUnaryOpType& posUnaryOp,
                                          srcDataType alpha,
                                          const srcDataType* const __restrict__ p_src_global,
                                          dstDataType beta,
                                          dstDataType* const __restrict__ p_dst_global,
                                          int* const __restrict__ indices_global)
    {
        (void)posUnaryOp;

        const auto zeroVal = opReduce::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, dst1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     compType,
                     dim0_thread_slice_size * dim1_thread_slice_size,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, dim0_thread_slice_size, true>
            accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, dim0_thread_slice_size, true> accuIndex_buf;

        static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<dim0_thread_slice_size, dim1_thread_slice_size>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<dim0_thread_slice_size>{}, Number<dim1_thread_slice_size>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<
            srcDataType,
            compType,
            src2dDescType,
            decltype(ThreadBufferDesc),
            ThreadBufferLengths,
            typename conditional<vectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type,
            vectorDim,
            vectorSize,
            1,
            false>(src2dDesc, make_multi_index(thread_global_1d_id * dim0_thread_slice_size, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, dim1_thread_slice_size);

        index_t indexStart    = 0;
        index_t reducedLength = 0;
        do
        {
            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
                // do element-wise pre-reduction operation
                static_for<0, dim1_thread_slice_size, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_size>{} + J;

                    in_thread_buf(offset) = preUnaryOp(in_thread_buf[offset]);
                });

                // reduce on each thread-local slice
                static_for<0, dim1_thread_slice_size, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<dim1_thread_slice_size>{} + J;
                    binop::calculate(
                        accuValue_buf(I), in_thread_buf[offset], accuIndex_buf(I), indexStart + J);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);

            indexStart += dim1_thread_slice_size;
            reducedLength += dim1_thread_slice_size;
        } while(reducedLength < toReduceLength);

        // for indiced operation, posUnaryOp shoud do nothing
        static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
            accuValue_buf(I) = posUnaryOp(accuValue_buf[I]);

            if(!float_equal_one{}(alpha))
                accuValue_buf(I) *= type_convert<compType>(alpha);
        });

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<dim0_thread_slice_size>{}));

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load =
                ThreadwiseTensorSliceTransfer_v2<dstDataType,
                                                 dstDataType,
                                                 dst1dDescType,
                                                 decltype(ReducedDataDesc),
                                                 Sequence<dim0_thread_slice_size>,
                                                 Sequence<0>,
                                                 0,
                                                 1,
                                                 1,
                                                 false>(
                    dst1dDesc, make_multi_index(thread_global_1d_id * dim0_thread_slice_size));

            StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, dim0_thread_slice_size, true>
                priorDstValue_buf;

            threadwise_dst_load.Run(
                dst1dDesc, dst_global_val_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

            static_for<0, dim0_thread_slice_size, 1>{}([&](auto I) {
                accuValue_buf(I) += type_convert<compType>(priorDstValue_buf[I] * beta);
            });
        }

        auto threadwise_dst_val_store =
            ThreadwiseTensorSliceTransfer_v1r3<compType,
                                               dstDataType,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               PassThroughOp<dstDataType>,
                                               Sequence<dim0_thread_slice_size>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                dst1dDesc,
                make_multi_index(thread_global_1d_id * dim0_thread_slice_size),
                PassThroughOp<dstDataType>{});

        auto threadwise_dst_idx_store =
            ThreadwiseTensorSliceTransfer_v1r3<int,
                                               int,
                                               decltype(ReducedDataDesc),
                                               dst1dDescType,
                                               PassThroughOp<int>,
                                               Sequence<dim0_thread_slice_size>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                dst1dDesc,
                make_multi_index(thread_global_1d_id * dim0_thread_slice_size),
                PassThroughOp<int>{});

        threadwise_dst_val_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, dst1dDesc, dst_global_val_buf);
        threadwise_dst_idx_store.Run(
            ReducedDataDesc, make_tuple(I0), accuIndex_buf, dst1dDesc, dst_global_idx_buf);
    };
};

} // namespace ck
#endif
