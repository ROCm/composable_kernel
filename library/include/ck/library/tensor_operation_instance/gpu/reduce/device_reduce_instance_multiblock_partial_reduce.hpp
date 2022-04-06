#ifndef DEVICE_REDUCE_INSTANCE_MULTIBLOCK_PARTIAL_REDUCE_HPP
#define DEVICE_REDUCE_INSTANCE_MULTIBLOCK_PARTIAL_REDUCE_HPP

#include "reduction_operator_mapping.hpp"
#include "device_reduce_instance_impl_common.hpp"
#include "device_reduce_multiblock_partial_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

#ifdef QUICK_REDUCE_TEST
using reduce_configuration_2_instances_multiblock_partial_reduce = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    ReductionConfiguration_2<0, 1, 1, 2, 1>,
    ReductionConfiguration_2<1, 2, 1, 1, 2>,
    ReductionConfiguration_2<0, 1, 1, 3, 1>,
    ReductionConfiguration_2<1, 1, 1, 1, 3>
    // clang-format on
    >;
#else
using reduce_configuration_2_instances_multiblock_partial_reduce = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    ReductionConfiguration_2<0, 4, 1, 8, 1>,
    ReductionConfiguration_2<0, 4, 1, 4, 1>,
    ReductionConfiguration_2<0, 2, 1, 2, 1>,

    ReductionConfiguration_2<1, 4, 1, 1, 8>,
    ReductionConfiguration_2<1, 4, 1, 1, 4>,
    ReductionConfiguration_2<1, 2, 1, 1, 2>,

    // special instances
    ReductionConfiguration_2<0, 1, 1, 3, 1>,
    ReductionConfiguration_2<0, 1, 1, 5, 1>,
    ReductionConfiguration_2<0, 1, 1, 7, 1>,
    ReductionConfiguration_2<0, 1, 1, 11, 1>,

    ReductionConfiguration_2<0, 1, 1, 1, 3>,
    ReductionConfiguration_2<0, 1, 1, 1, 5>,
    ReductionConfiguration_2<0, 1, 1, 1, 7>,
    ReductionConfiguration_2<0, 1, 1, 1, 11>
    // clang-format on
    >;
#endif

template <typename AccDataType, ReduceTensorOp ReduceOpId>
using deviceReduceMultiBlockPartialReducePtrType = DeviceReducePtr<
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, false>::InElementwiseOperation,
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, false>::AccElementwiseOperation>;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim,
          ReduceTensorOp ReduceOpId,
          NanPropagation NanOpt,
          ReduceTensorIndices IndicesOpt>
void add_device_reduce_instance_multiblock_partial_reduce(
    std::vector<deviceReduceMultiBlockPartialReducePtrType<AccDataType, ReduceOpId>>&
        device_op_instances)
{
    using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;
    using InElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, false>::
            InElementwiseOperation;
    using AccElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, false>::
            AccElementwiseOperation;

    constexpr bool Indexable =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);
    constexpr bool NeedIndices = Indexable && (IndicesOpt != ReduceTensorIndices::NO_INDICES);

    constexpr bool PropagateNan = (NanOpt == NanPropagation::NOT_PROPAGATE_NAN) ? false : true;

    static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}([&](auto i) {
        using cfg1 =
            remove_cvref_t<decltype(std::get<i.value>(reduce_configuration_1_instances{}))>;

        static_for<
            0,
            std::tuple_size<reduce_configuration_2_instances_multiblock_partial_reduce>::value,
            1>{}([&](auto j) {
            using cfg2 = remove_cvref_t<decltype(
                std::get<j.value>(reduce_configuration_2_instances_multiblock_partial_reduce{}))>;

            using ReduceOpInstance = DeviceReduceMultiBlockPartialReduce<InDataType,
                                                                         AccDataType,
                                                                         OutDataType,
                                                                         Rank,
                                                                         NumReduceDim,
                                                                         ReduceOperation,
                                                                         InElementwiseOperation,
                                                                         AccElementwiseOperation,
                                                                         PropagateNan,
                                                                         NeedIndices,
                                                                         cfg1::BlockSize_,
                                                                         cfg1::MThreadClusterSize_,
                                                                         cfg1::KThreadClusterSize_,
                                                                         cfg2::MThreadSliceSize_,
                                                                         cfg2::KThreadSliceSize_,
                                                                         cfg2::InSrcVectorDim_,
                                                                         cfg2::InSrcVectorSize_,
                                                                         cfg2::OutDstVectorSize_>;

            device_op_instances.push_back(std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
        });
    });
};

#define ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_TYPE(                                  \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, NumReduceDim)            \
    template void add_device_reduce_instance_multiblock_partial_reduce<inT,          \
                                                                       compT,        \
                                                                       outT,         \
                                                                       Rank,         \
                                                                       NumReduceDim, \
                                                                       ReduceOpId,   \
                                                                       NanOpt,       \
                                                                       IndicesOpt>(  \
        std::vector<deviceReduceMultiBlockPartialReducePtrType<compT, ReduceOpId>> & \
        device_op_instances)

#define ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(                                            \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, NumReduceDim)                    \
    ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_TYPE(inT,                                          \
                                               compT,                                        \
                                               outT,                                         \
                                               static_cast<ReduceTensorOp>(ReduceOpId),      \
                                               static_cast<NanPropagation>(NanOpt),          \
                                               static_cast<ReduceTensorIndices>(IndicesOpt), \
                                               Rank,                                         \
                                               NumReduceDim)

#define ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_TYPE(                                      \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, NumReduceDim)                    \
    extern template void add_device_reduce_instance_multiblock_partial_reduce<inT,           \
                                                                              compT,         \
                                                                              outT,          \
                                                                              Rank,          \
                                                                              NumReduceDim,  \
                                                                              ReduceOpId,    \
                                                                              NanOpt,        \
                                                                              IndicesOpt>(   \
        std::vector<                                                                         \
            DeviceReducePtr<typename reduce_unary_operator<compT, ReduceOpId, true, false>:: \
                                InElementwiseOperation,                                      \
                            typename reduce_unary_operator<compT, ReduceOpId, true, false>:: \
                                AccElementwiseOperation>> &                                  \
        device_op_instances)

#define ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(                                            \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, NumReduceDim)                        \
    ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_TYPE(inT,                                          \
                                                   compT,                                        \
                                                   outT,                                         \
                                                   static_cast<ReduceTensorOp>(ReduceOpId),      \
                                                   static_cast<NanPropagation>(NanOpt),          \
                                                   static_cast<ReduceTensorIndices>(IndicesOpt), \
                                                   Rank,                                         \
                                                   NumReduceDim)

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
