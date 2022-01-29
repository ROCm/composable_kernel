#include "device_reduce_instance_impl_common.hpp"
#include "device_reduce_threadwise.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename AccDataType, ReduceTensorOp_t ReduceOpId>
using deviceReducePtrType = DeviceReducePtr<
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation,
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::AccElementwiseOperation>;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          typename InnerDims,
          ReduceTensorOp_t ReduceOpId,
          NanPropagation_t NanOpt,
          ReduceTensorIndices_t IndicesOpt>
void add_device_reduce_instance_threadwise(
    std::vector<deviceReducePtrType<AccDataType, ReduceOpId>>& device_op_instances)
{
    using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;
    using InElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::
            AccElementwiseOperation;

    constexpr bool Indexable =
        (ReduceOpId == ReduceTensorOp_t::MIN || ReduceOpId == ReduceTensorOp_t::MAX ||
         ReduceOpId == ReduceTensorOp_t::AMAX);
    constexpr bool NeedIndices = Indexable && (IndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    constexpr bool PropagateNan = (NanOpt == NanPropagation_t::NOT_PROPAGATE_NAN) ? false : true;

    using cfg1 = ReductionConfiguration_1<256, 256, 1>;

    static_for<0, std::tuple_size<reduce_configuration_2_instances>::value, 1>{}([&](auto j) {
        using cfg2 =
            remove_cvref_t<decltype(std::get<j.value>(reduce_configuration_2_instances{}))>;

        using ReduceOpInstance = DeviceReduceThreadWise<InDataType,
                                                        AccDataType,
                                                        OutDataType,
                                                        Rank,
                                                        InnerDims,
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
                                                        cfg2::VectorDim_,
                                                        cfg2::VectorSize_>;

        device_op_instances.push_back(std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
    });
};

#define ADD_THREADWISE_INST_BY_TYPE(inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...) \
    template void add_device_reduce_instance_threadwise<inT,                                     \
                                                        compT,                                   \
                                                        outT,                                    \
                                                        Rank,                                    \
                                                        Sequence<__VA_ARGS__>,                   \
                                                        ReduceOpId,                              \
                                                        NanOpt,                                  \
                                                        IndicesOpt>(                             \
        std::vector<deviceReducePtrType<compT, ReduceOpId>> & device_op_instances)

#define ADD_THREADWISE_INST_BY_ID(inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...) \
    ADD_THREADWISE_INST_BY_TYPE(inT,                                                           \
                                compT,                                                         \
                                outT,                                                          \
                                static_cast<ReduceTensorOp_t>(ReduceOpId),                     \
                                static_cast<NanPropagation_t>(NanOpt),                         \
                                static_cast<ReduceTensorIndices_t>(IndicesOpt),                \
                                Rank,                                                          \
                                __VA_ARGS__)

// half, half, half
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 2, 1);       //

// half, float, half
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 0, 0, 0, 4, 0);
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 0, 0, 0, 2, 1);
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 5, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 5, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 7, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(half_t, float, half_t, 7, 0, 0, 2, 1);       //

// float, float, float
ADD_THREADWISE_INST_BY_ID(float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_THREADWISE_INST_BY_ID(float, float, float, 0, 0, 0, 4, 0);
ADD_THREADWISE_INST_BY_ID(float, float, float, 0, 0, 0, 2, 1);
ADD_THREADWISE_INST_BY_ID(float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_THREADWISE_INST_BY_ID(float, float, float, 5, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 5, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_THREADWISE_INST_BY_ID(float, float, float, 7, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 7, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_BY_ID(float, float, float, 2, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 2, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_BY_ID(float, float, float, 3, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 3, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_BY_ID(float, float, float, 4, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 4, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_BY_ID(float, float, float, 2, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 2, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_BY_ID(float, float, float, 3, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 3, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_BY_ID(float, float, float, 4, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, float, float, 4, 0, 1, 2, 1);       //

// float, double, float
ADD_THREADWISE_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_THREADWISE_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0);
ADD_THREADWISE_INST_BY_ID(float, double, float, 0, 0, 0, 2, 1);
ADD_THREADWISE_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_THREADWISE_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, double, float, 5, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_THREADWISE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, double, float, 7, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
