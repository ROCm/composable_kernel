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
#include "reduction_functions_accumulate.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseReduction,
          bool NeedIndices,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
__global__ void kernel_reduce_threadwise(const InGridDesc_M_K in_grid_desc_m_k,
                                         const OutGridDesc_M out_grid_desc_m,
                                         const InElementwiseOperation in_elementwise_op,
                                         const AccElementwiseOperation acc_elementwise_op,
                                         AccDataType alpha,
                                         const InDataType* const __restrict__ p_in_global,
                                         OutDataType beta,
                                         OutDataType* const __restrict__ p_out_global,
                                         IndexDataType* const __restrict__ p_indices_global)
{
    if constexpr(!NeedIndices)
    {
        GridwiseReduction::Run(in_grid_desc_m_k,
                               out_grid_desc_m,
                               in_elementwise_op,
                               acc_elementwise_op,
                               alpha,
                               p_in_global,
                               beta,
                               p_out_global,
                               p_indices_global);
    }
    else
    {
        GridwiseReduction::RunWithIndices(in_grid_desc_m_k,
                                          out_grid_desc_m,
                                          in_elementwise_op,
                                          acc_elementwise_op,
                                          alpha,
                                          p_in_global,
                                          beta,
                                          p_out_global,
                                          p_indices_global);
    };
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M,
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
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct GridwiseReduction_mk_to_m_threadwise
{
    template <typename T>
    using PassThroughOp = tensor_operation::element_wise::UnaryIdentic<T, T>;

    static constexpr auto I0 = Number<0>{};

    using Accumulation =
        detail::accumulate_with_nan_check<PropagateNan, ReduceOperation, AccDataType>;
    using AccumulationWithIndices = detail::accumulate_with_indices_with_nan_check<PropagateNan,
                                                                                   ReduceOperation,
                                                                                   AccDataType,
                                                                                   IndexDataType>;

    __device__ static void Run(const InGridDesc_M_K& in_grid_desc_m_k,
                               const OutGridDesc_M& out_grid_desc_m,
                               const InElementwiseOperation& in_elementwise_op,
                               const AccElementwiseOperation& acc_elementwise_op,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_in_global,
                               OutDataType beta,
                               OutDataType* const __restrict__ p_out_global,
                               IndexDataType* const __restrict__ p_indices_global)
    {
        (void)p_indices_global;

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        const auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_in_global, in_grid_desc_m_k.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_out_global, out_grid_desc_m.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accuValue_buf(I) = zeroVal; });

        const auto toReduceLength = in_grid_desc_m_k.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

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
            false>(in_grid_desc_m_k, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t reducedLength = 0;
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

            reducedLength += KThreadSliceSize;
        } while(reducedLength < toReduceLength);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            acc_elementwise_op(accuValue_buf(I), accuValue_buf(I));

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
                                                     OutGridDesc_M,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<MThreadSliceSize>,
                                                     Sequence<0>,
                                                     0,
                                                     1,
                                                     1,
                                                     true>(
                        out_grid_desc_m, make_multi_index(thread_global_1d_id * MThreadSliceSize));

                StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                    priorDstValue_buf;

                threadwise_dst_load.Run(out_grid_desc_m,
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
                                               OutGridDesc_M,
                                               PassThroughOp<AccDataType>,
                                               Sequence<MThreadSliceSize>,
                                               Sequence<0>,
                                               0,
                                               OutDstVectorSize,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                out_grid_desc_m,
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp<AccDataType>{});

        threadwise_dst_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, out_grid_desc_m, dst_global_buf);
    };

    __device__ static void RunWithIndices(const InGridDesc_M_K& in_grid_desc_m_k,
                                          const OutGridDesc_M& out_grid_desc_m,
                                          const InElementwiseOperation& in_elementwise_op,
                                          const AccElementwiseOperation& acc_elementwise_op,
                                          AccDataType alpha,
                                          const InDataType* const __restrict__ p_in_global,
                                          OutDataType beta,
                                          OutDataType* const __restrict__ p_out_global,
                                          IndexDataType* const __restrict__ p_indices_global)
    {
        (void)acc_elementwise_op;

        const auto zeroVal = ReduceOperation::GetReductionZeroVal();

        const auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_in_global, in_grid_desc_m_k.GetElementSpaceSize(), type_convert<InDataType>(zeroVal));
        auto out_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_out_global, out_grid_desc_m.GetElementSpaceSize());
        auto out_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_indices_global, out_grid_desc_m.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     AccDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, MThreadSliceSize, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, IndexDataType, MThreadSliceSize, true> accuIndex_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accuValue_buf(I) = zeroVal;
            accuIndex_buf(I) = 0;
        });

        const auto toReduceLength = in_grid_desc_m_k.GetLength(Number<1>{});

        using ThreadBufferLengths       = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

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
            false>(in_grid_desc_m_k, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t indexStart    = 0;
        index_t reducedLength = 0;
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
                    AccumulationWithIndices::calculate(
                        accuValue_buf(I), in_thread_buf[offset], accuIndex_buf(I), indexStart + J);
                });
            });

            threadwise_src_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            indexStart += KThreadSliceSize;
            reducedLength += KThreadSliceSize;
        } while(reducedLength < toReduceLength);

        // for indiced operation, acc_elementwise_op shoud do nothing
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            acc_elementwise_op(accuValue_buf(I), accuValue_buf(I));

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
                                                     OutGridDesc_M,
                                                     decltype(ReducedDataDesc),
                                                     Sequence<MThreadSliceSize>,
                                                     Sequence<0>,
                                                     0,
                                                     1,
                                                     1,
                                                     false>(
                        out_grid_desc_m, make_multi_index(thread_global_1d_id * MThreadSliceSize));

                StaticBuffer<AddressSpaceEnum_t::Vgpr, OutDataType, MThreadSliceSize, true>
                    priorDstValue_buf;

                threadwise_dst_load.Run(out_grid_desc_m,
                                        out_global_val_buf,
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
                                               OutGridDesc_M,
                                               PassThroughOp<AccDataType>,
                                               Sequence<MThreadSliceSize>,
                                               Sequence<0>,
                                               0,
                                               OutDstVectorSize,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                out_grid_desc_m,
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp<AccDataType>{});

        auto threadwise_dst_idx_store =
            ThreadwiseTensorSliceTransfer_v1r3<IndexDataType,
                                               IndexDataType,
                                               decltype(ReducedDataDesc),
                                               OutGridDesc_M,
                                               PassThroughOp<IndexDataType>,
                                               Sequence<MThreadSliceSize>,
                                               Sequence<0>,
                                               0,
                                               OutDstVectorSize,
                                               InMemoryDataOperationEnum_t::Set,
                                               1,
                                               false>(
                out_grid_desc_m,
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp<IndexDataType>{});

        threadwise_dst_val_store.Run(
            ReducedDataDesc, make_tuple(I0), accuValue_buf, out_grid_desc_m, out_global_val_buf);

        threadwise_dst_idx_store.Run(
            ReducedDataDesc, make_tuple(I0), accuIndex_buf, out_grid_desc_m, out_global_idx_buf);
    };
};

} // namespace ck
#endif
