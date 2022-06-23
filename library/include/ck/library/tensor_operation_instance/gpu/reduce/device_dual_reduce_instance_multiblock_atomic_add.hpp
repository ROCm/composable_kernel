#ifndef DEVICE_DUAL_REDUCE_INSTANCE_MULTIBLOCK_ATOMIC_ADD_HPP
#define DEVICE_DUAL_REDUCE_INSTANCE_MULTIBLOCK_ATOMIC_ADD_HPP

#include "device_reduce_instance_impl_common.hpp"
#include "device_multiple_reduce_multiblock.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

using dual_reduce_configuration_1_instances_multiblock = std::tuple<
    // clang-format off
    // BlockSize | MThreadClusterSize | KThreadClusterSize
    ReductionConfiguration_1<256, 128, 2>,
    ReductionConfiguration_1<256, 64, 4>,
    ReductionConfiguration_1<256, 32, 8>,
    ReductionConfiguration_1<256, 16, 16>,
    ReductionConfiguration_1<256, 8, 32>,
    ReductionConfiguration_1<256, 4, 64>,
    ReductionConfiguration_1<256, 2, 128>,
    ReductionConfiguration_1<256, 1, 256>
    // clang-format on
    >;

#ifdef QUICK_REDUCE_TEST
using dual_reduce_configuration_2_instances_multiblock = std::tuple<
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
using dual_reduce_configuration_2_instances_multiblock = std::tuple<
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

template <typename InElementwiseOperation1,
          typename AccElementwiseOperation1,
          typename InElementwiseOperation2,
          typename AccElementwiseOperation2>
using deviceDualReduceMultiBlockAtomicAddPtrType =
    DeviceMultipleReducePtr<2,
                            Tuple<InElementwiseOperation1, InElementwiseOperation2>,
                            Tuple<AccElementwiseOperation1, AccElementwiseOperation2>>;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType1,
          typename OutDataType2,
          typename ReduceOperation,
          typename InElementwiseOperation1,
          typename InElementwiseOperation2,
          typename AccElementwiseOperation1,
          typename AccElementwiseOperation2,
          bool PropagateNan,
          int Rank,
          int NumReduceDim>
void add_device_dual_reduce_instance_multiblock_atomic_add(
    std::vector<deviceDualReduceMultiBlockAtomicAddPtrType<InElementwiseOperation1,
                                                           AccElementwiseOperation1,
                                                           InElementwiseOperation2,
                                                           AccElementwiseOperation2>>&
        device_op_instances)
{
    static_for<0, std::tuple_size<dual_reduce_configuration_1_instances_multiblock>::value, 1>{}(
        [&](auto i) {
            using cfg1 = remove_cvref_t<decltype(
                std::get<i.value>(dual_reduce_configuration_1_instances_multiblock{}))>;

            static_for<0,
                       std::tuple_size<dual_reduce_configuration_2_instances_multiblock>::value,
                       1>{}([&](auto j) {
                using cfg2 = remove_cvref_t<decltype(
                    std::get<j.value>(dual_reduce_configuration_2_instances_multiblock{}))>;

                using ReduceOpInstance = DeviceMultipleReduceMultiBlock<
                    2,
                    InDataType,
                    AccDataType,
                    Tuple<OutDataType1*, OutDataType2*>,
                    Rank,
                    NumReduceDim,
                    ReduceOperation,
                    Tuple<InElementwiseOperation1, InElementwiseOperation2>,
                    Tuple<AccElementwiseOperation1, AccElementwiseOperation2>,
                    InMemoryDataOperationEnum::AtomicAdd,
                    PropagateNan,
                    cfg1::BlockSize_,
                    cfg1::MThreadClusterSize_,
                    cfg1::KThreadClusterSize_,
                    cfg2::MThreadSliceSize_,
                    cfg2::KThreadSliceSize_,
                    cfg2::InSrcVectorDim_,
                    cfg2::InSrcVectorSize_,
                    cfg2::OutDstVectorSize_>;

                device_op_instances.push_back(
                    std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
            });
        });
};

#define ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST(inT,                                    \
                                                   compT,                                  \
                                                   outT1,                                  \
                                                   outT2,                                  \
                                                   ReduceOperation,                        \
                                                   InElementwiseOp1,                       \
                                                   AccElementwiseOp1,                      \
                                                   InElementwiseOp2,                       \
                                                   AccElementwiseOp2,                      \
                                                   PropagateNan,                           \
                                                   Rank,                                   \
                                                   NumReduceDim)                           \
    template void add_device_dual_reduce_instance_multiblock_atomic_add<inT,               \
                                                                        compT,             \
                                                                        outT1,             \
                                                                        outT2,             \
                                                                        ReduceOperation,   \
                                                                        InElementwiseOp1,  \
                                                                        InElementwiseOp2,  \
                                                                        AccElementwiseOp1, \
                                                                        AccElementwiseOp2, \
                                                                        PropagateNan,      \
                                                                        Rank,              \
                                                                        NumReduceDim>(     \
        std::vector<deviceDualReduceMultiBlockAtomicAddPtrType<InElementwiseOp1,           \
                                                               AccElementwiseOp1,          \
                                                               InElementwiseOp2,           \
                                                               AccElementwiseOp2>> &       \
        device_op_instances)

#define ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST_REF(inT,                                       \
                                                       compT,                                     \
                                                       outT1,                                     \
                                                       outT2,                                     \
                                                       ReduceOperation,                           \
                                                       InElementwiseOp1,                          \
                                                       AccElementwiseOp1,                         \
                                                       InElementwiseOp2,                          \
                                                       AccElementwiseOp2,                         \
                                                       PropagateNan,                              \
                                                       Rank,                                      \
                                                       NumReduceDim)                              \
    extern template void add_device_dual_reduce_instance_multiblock_atomic_add<inT,               \
                                                                               compT,             \
                                                                               outT1,             \
                                                                               outT2,             \
                                                                               ReduceOperation,   \
                                                                               InElementwiseOp1,  \
                                                                               InElementwiseOp2,  \
                                                                               AccElementwiseOp1, \
                                                                               AccElementwiseOp2, \
                                                                               PropagateNan,      \
                                                                               Rank,              \
                                                                               NumReduceDim>(     \
        std::vector<deviceDualReduceMultiBlockAtomicAddPtrType<InElementwiseOp1,                  \
                                                               AccElementwiseOp1,                 \
                                                               InElementwiseOp2,                  \
                                                               AccElementwiseOp2>> &              \
        device_op_instances)

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
