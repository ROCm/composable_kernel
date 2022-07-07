// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_threadwise.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_impl_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef QUICK_REDUCE_TEST
using reduce_configuration_2_instances_threadwise = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    ReductionConfiguration_2<0, 2, 2, 2, 1>,
    ReductionConfiguration_2<0, 1, 1, 2, 1>,
    ReductionConfiguration_2<1, 2, 1, 1, 2>,
    ReductionConfiguration_2<0, 1, 1, 3, 1>,
    ReductionConfiguration_2<1, 1, 1, 1, 3>
    // clang-format on
    >;
#else
using reduce_configuration_2_instances_threadwise = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    ReductionConfiguration_2<0, 4, 4, 8, 1>,
    ReductionConfiguration_2<0, 4, 4, 4, 1>,
    ReductionConfiguration_2<0, 2, 2, 2, 1>,

    ReductionConfiguration_2<1, 4, 1, 1, 8>,
    ReductionConfiguration_2<1, 4, 1, 1, 4>,
    ReductionConfiguration_2<1, 2, 1, 1, 2>,

    // special instances
    ReductionConfiguration_2<0, 1, 1, 3, 1>,
    ReductionConfiguration_2<0, 1, 1, 5, 1>,
    ReductionConfiguration_2<0, 1, 1, 7, 1>,
    ReductionConfiguration_2<0, 1, 1, 11, 1>,

    ReductionConfiguration_2<1, 1, 1, 1, 3>,
    ReductionConfiguration_2<1, 1, 1, 1, 5>,
    ReductionConfiguration_2<1, 1, 1, 1, 7>,
    ReductionConfiguration_2<1, 1, 1, 1, 11>
    // clang-format on
    >;
#endif

template <ReduceTensorOp ReduceOpId>
using deviceReduceThreadWisePtrType = DeviceReducePtr<
    typename reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation,
    typename reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation>;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim,
          ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool UseIndex>
void add_device_reduce_instance_threadwise(
    std::vector<deviceReduceThreadWisePtrType<ReduceOpId>>& device_op_instances)
{
    using ReduceOperation = typename reduce_binary_operator<ReduceOpId>::opType;
    using InElementwiseOperation =
        typename reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

    constexpr bool Indexable =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);
    constexpr bool OutputIndex = Indexable && UseIndex;

    using cfg1 = ReductionConfiguration_1<256, 256, 1>;

    static_for<0, std::tuple_size<reduce_configuration_2_instances_threadwise>::value, 1>{}(
        [&](auto j) {
            using cfg2 = remove_cvref_t<decltype(
                std::get<j.value>(reduce_configuration_2_instances_threadwise{}))>;

            using ReduceOpInstance = DeviceReduceThreadWise<InDataType,
                                                            AccDataType,
                                                            OutDataType,
                                                            Rank,
                                                            NumReduceDim,
                                                            ReduceOperation,
                                                            InElementwiseOperation,
                                                            AccElementwiseOperation,
                                                            PropagateNan,
                                                            OutputIndex,
                                                            false, // HaveIndexInputIfOutputIndex
                                                            cfg1::BlockSize_,
                                                            cfg2::MThreadSliceSize_,
                                                            cfg2::KThreadSliceSize_,
                                                            cfg2::InSrcVectorDim_,
                                                            cfg2::InSrcVectorSize_,
                                                            cfg2::OutDstVectorSize_>;

            device_op_instances.push_back(std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
        });
};

#define ADD_THREADWISE_INST_BY_TYPE(                                          \
    inT, compT, outT, ReduceOpId, PropagateNan, UseIndex, Rank, NumReduceDim) \
    template void add_device_reduce_instance_threadwise<inT,                  \
                                                        compT,                \
                                                        outT,                 \
                                                        Rank,                 \
                                                        NumReduceDim,         \
                                                        ReduceOpId,           \
                                                        PropagateNan,         \
                                                        UseIndex>(            \
        std::vector<deviceReduceThreadWisePtrType<ReduceOpId>> & device_op_instances)

#define ADD_THREADWISE_INST_BY_ID(                                        \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, NumReduceDim) \
    ADD_THREADWISE_INST_BY_TYPE(inT,                                      \
                                compT,                                    \
                                outT,                                     \
                                static_cast<ReduceTensorOp>(ReduceOpId),  \
                                static_cast<bool>(NanOpt),                \
                                static_cast<bool>(IndicesOpt),            \
                                Rank,                                     \
                                NumReduceDim)

#define ADD_THREADWISE_INST_REF_BY_TYPE(                                      \
    inT, compT, outT, ReduceOpId, PropagateNan, UseIndex, Rank, NumReduceDim) \
    extern template void add_device_reduce_instance_threadwise<inT,           \
                                                               compT,         \
                                                               outT,          \
                                                               Rank,          \
                                                               NumReduceDim,  \
                                                               ReduceOpId,    \
                                                               PropagateNan,  \
                                                               UseIndex>(     \
        std::vector<deviceReduceThreadWisePtrType<ReduceOpId>> & device_op_instances)

#define ADD_THREADWISE_INST_REF_BY_ID(                                       \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, NumReduceDim)    \
    ADD_THREADWISE_INST_REF_BY_TYPE(inT,                                     \
                                    compT,                                   \
                                    outT,                                    \
                                    static_cast<ReduceTensorOp>(ReduceOpId), \
                                    static_cast<bool>(NanOpt),               \
                                    static_cast<bool>(IndicesOpt),           \
                                    Rank,                                    \
                                    NumReduceDim)

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
