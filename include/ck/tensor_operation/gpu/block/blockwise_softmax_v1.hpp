// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"

namespace ck {
template <index_t BlockSize,
          typename AccDataType,
          index_t MPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t RegSizePerXdlops,
          index_t MRepeat,
          index_t NRepeat>
struct BlockwiseSoftmax_V1
{
    static_assert(MRepeat == 1, "Now MRepeat must equal 1");

    static constexpr auto I0                  = Number<0>{};
    static constexpr auto I1                  = Number<1>{};
    static constexpr index_t MThreadSliceSize = 1;
    static constexpr index_t WaveSize         = 64;

    static_assert(MPerBlock == MPerXDL * BlockSize / WaveSize, "wave is only m direction");

    struct BlockToMKMap_M0_K_M1Adapt
    {
        __host__ __device__ BlockToMKMap_M0_K_M1Adapt() = default;
        template <typename TopIdx>
        __host__ __device__ static constexpr auto CalculateBottomIndex(const TopIdx& idx_top)
        {
            const auto index = idx_top[I0];
            const auto m     = (index / WaveSize) * MPerXDL + index % MPerXDL;
            const auto k     = (index % WaveSize) / MPerXDL;
            return make_tuple(m, k);
        }
    };

    constexpr static auto in_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, Number<RegSizePerXdlops>{}));

    using ThreadReduceSrcDesc_M_K = decltype(
        make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}, Number<RegSizePerXdlops>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<1>{})));

    using ThreadwiseMaxReduce =
        ThreadwiseReduction<AccDataType,
                            ThreadReduceSrcDesc_M_K,
                            ThreadReduceDstDesc_M,
                            reduce::Max,
                            false, // param ignored
                            detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>;

    using ThreadClusterLengths_M_K = Sequence<MPerBlock, WaveSize / MPerXDL>;

    using BlockwiseMaxReduce =
        PartitionedBlockwiseReduction2<AccDataType,
                                       BlockSize,
                                       ThreadClusterLengths_M_K,
                                       BlockToMKMap_M0_K_M1Adapt,
                                       reduce::Max,
                                       false, // param ignored
                                       detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>;

    using BlockwiseSumReduce =
        PartitionedBlockwiseReduction2<AccDataType,
                                       BlockSize,
                                       ThreadClusterLengths_M_K,
                                       BlockToMKMap_M0_K_M1Adapt,
                                       reduce::Add,
                                       false, // ignored
                                       detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>;

    using ThreadwiseSumReduce =
        ThreadwiseReduction<AccDataType,
                            ThreadReduceSrcDesc_M_K,
                            ThreadReduceDstDesc_M,
                            reduce::Add,
                            false, // ignored
                            detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>;
    template <typename CThreadBuffer>
    __host__ __device__ static void
    Run(CThreadBuffer& in_thread_buf, float& f_sum, float& f_max, void* __restrict__ p_reduce)
    {
        auto reduce_work_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<AccDataType*>(p_reduce), BlockSize);
        //
        // find max value
        //
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> max_value_buf;
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            max_value_buf(I) = reduce::Max::template GetIdentityValue<AccDataType>();
        });

        // max value for one thread
        static_for<0, NRepeat, 1>{}([&](auto n) {
            constexpr index_t in_offset = in_thread_desc.CalculateOffset(make_tuple(0, n, 0));
            auto& xdlops_out            = in_thread_buf.GetVectorTypeReference(Number<in_offset>{});

            ThreadwiseMaxReduce::Reduce(xdlops_out.template AsType<float>(), max_value_buf);
        });

        // block reduce for max
        BlockwiseMaxReduce::Reduce(reduce_work_buf, max_value_buf(I0));
        block_sync_lds();
        // save max
        f_max = max_value_buf(I0);

        //
        // softmax
        //

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accu_value_buf(I) = reduce::Add::template GetIdentityValue<AccDataType>();
        });
        // calculate exp for elements, P=exp(s-max)
        static_for<0, NRepeat, 1>{}([&](auto n) {
            constexpr index_t in_offset = in_thread_desc.CalculateOffset(make_tuple(0, n, 0));
            auto& xdlops_out            = in_thread_buf.GetVectorTypeReference(Number<in_offset>{});

            static_for<0, RegSizePerXdlops, 1>{}([&](auto iK) {
                xdlops_out.template AsType<float>()(iK) =
                    math::exp(xdlops_out.template AsType<float>()[iK] - max_value_buf(I0));
            });
        });
        // sum data
        static_for<0, NRepeat, 1>{}([&](auto n) {
            constexpr index_t in_offset = in_thread_desc.CalculateOffset(make_tuple(0, n, 0));
            auto& xdlops_out            = in_thread_buf.GetVectorTypeReference(Number<in_offset>{});
            ThreadwiseSumReduce::Reduce(xdlops_out.template AsType<float>(), accu_value_buf);
            block_sync_lds();
        });
        BlockwiseSumReduce::Reduce(reduce_work_buf, accu_value_buf(I0));
        block_sync_lds();

        // save sum
        f_sum = accu_value_buf(I0);
    }
}; // namespace ck

} // namespace ck
