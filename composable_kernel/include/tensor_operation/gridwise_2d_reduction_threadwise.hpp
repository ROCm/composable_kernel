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
          bool NeedIndices,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename In2dDescType,
          typename Out1dDescType,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
__global__ void kernel_reduce_threadwise(const In2dDescType in2dDesc,
                                         const Out1dDescType out1dDesc,
                                         const InElementwiseOperation inElementwiseOp,
                                         const AccElementwiseOperation accElementwiseOp,
                                         AccDataType alpha,
                                         const InDataType* const __restrict__ p_src_global,
                                         OutDataType beta,
                                         OutDataType* const __restrict__ p_dst_global,
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
          typename AccElementwiseOperation,
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
struct GridwiseReduction_xy_to_x_threadwise
{
    template <typename T>
    using PassThroughOp = reduce::unary_identic<T, T>;

    static constexpr auto I0 = Number<0>{};

    using BinaryOperation =
        detail::binop_with_nan_check<PropagateNan, ReduceOperation, AccDataType>;

    __device__ static void Run(const In2dDescType& in2dDesc,
                               const Out1dDescType& out1dDesc,
                               const InElementwiseOperation& inElementwiseOp,
                               const AccElementwiseOperation& accElementwiseOp,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_src_global,
                               OutDataType beta,
                               OutDataType* const __restrict__ p_dst_global,
                               int* const __restrict__ indices_global)
    {
        (void)indices_global;

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, in2dDesc.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, out1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accuValue_buf(I) = zeroVal; });

        const auto toReduceLength = in2dDesc.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

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
            false>(in2dDesc, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t reducedLength = 0;
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

            reducedLength += KThreadSliceSize;
        } while(reducedLength < toReduceLength);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accElementwiseOp(accuValue_buf(I), accuValue_buf(I));

            accuValue_buf(I) *= alpha;
        });

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

        if constexpr(!BetaIsZero)
        {
            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                     OutDataType,
                                                     Out1dDescType,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<MThreadSliceSize>,
                                                     Sequence<0>,
                                                     0,
                                                     1,
                                                     1,
                                                     true>(
                        out1dDesc, make_multi_index(thread_global_1d_id * MThreadSliceSize));

                StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                    priorDstValue_buf;

                threadwise_dst_load.Run(
                    out1dDesc, dst_global_buf, ReducedDataDesc, make_tuple(I0), priorDstValue_buf);

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
                                               false>(
                out1dDesc,
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp<AccDataType>{});

        threadwise_dst_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, out1dDesc, dst_global_buf);
    };

    __device__ static void RunWithIndices(const In2dDescType& in2dDesc,
                                          const Out1dDescType& out1dDesc,
                                          const InElementwiseOperation& inElementwiseOp,
                                          const AccElementwiseOperation& accElementwiseOp,
                                          AccDataType alpha,
                                          const InDataType* const __restrict__ p_src_global,
                                          OutDataType beta,
                                          OutDataType* const __restrict__ p_dst_global,
                                          int* const __restrict__ indices_global)
    {
        (void)accElementwiseOp;

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, in2dDesc.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        auto dst_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_dst_global, out1dDesc.GetElementSpaceSize());
        auto dst_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            indices_global, out1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, MThreadSliceSize, true> accuIndex_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        const auto toReduceLength = in2dDesc.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

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
            false>(in2dDesc, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t indexStart    = 0;
        index_t reducedLength = 0;
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
                    BinaryOperation::calculate(
                        accuValue_buf(I), in_thread_buf[offset], accuIndex_buf(I), indexStart + J);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(in2dDesc, in_thread_copy_step);

            indexStart += KThreadSliceSize;
            reducedLength += KThreadSliceSize;
        } while(reducedLength < toReduceLength);

        // for indiced operation, accElementwiseOp shoud do nothing
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accElementwiseOp(accuValue_buf(I), accuValue_buf(I));

            accuValue_buf(I) *= alpha;
        });

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

        if constexpr(!BetaIsZero)
        {
            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load =
                    ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                     OutDataType,
                                                     Out1dDescType,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<MThreadSliceSize>,
                                                     Sequence<0>,
                                                     0,
                                                     1,
                                                     1,
                                                     false>(
                        out1dDesc, make_multi_index(thread_global_1d_id * MThreadSliceSize));

                StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                    priorDstValue_buf;

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
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
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
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp<int>{});

        threadwise_dst_val_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, out1dDesc, dst_global_val_buf);

        threadwise_dst_idx_store.Run(
            ReducedDataDesc, make_tuple(I0), accuIndex_buf, out1dDesc, dst_global_idx_buf);
    };
};

} // namespace ck
#endif
