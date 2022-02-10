#include "reduction_operator_mapping.hpp"
#include "device_reduce_instance_impl_common.hpp"
#include "device_reduce_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

#ifdef QUICK_REDUCE_TEST
using reduce_configuration_2_instances_blockwise =
    std::tuple<ReductionConfiguration_2<0, 2, 2, 2, 1>,
               ReductionConfiguration_2<0, 1, 1, 2, 1>,
               ReductionConfiguration_2<1, 2, 1, 1, 2>,
               ReductionConfiguration_2<1, 2, 2, 1, 2>,
               ReductionConfiguration_2<0, 1, 1, 3, 1>,
               ReductionConfiguration_2<1, 1, 1, 1, 3>>;
#else
using reduce_configuration_2_instances_blockwise =
    std::tuple<ReductionConfiguration_2<0, 4, 4, 8, 1>,
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
               ReductionConfiguration_2<1, 1, 1, 1, 11>>;
#endif

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
void add_device_reduce_instance_blockwise(
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

    static_for<0, std::tuple_size<reduce_configuration_1_instances>::value, 1>{}([&](auto i) {
        using cfg1 =
            remove_cvref_t<decltype(std::get<i.value>(reduce_configuration_1_instances{}))>;

        static_for<0, std::tuple_size<reduce_configuration_2_instances_blockwise>::value, 1>{}(
            [&](auto j) {
                using cfg2 = remove_cvref_t<decltype(
                    std::get<j.value>(reduce_configuration_2_instances_blockwise{}))>;

                using ReduceOpInstance = DeviceReduceBlockWise<InDataType,
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
                                                               cfg2::InSrcVectorDim_,
                                                               cfg2::InSrcVectorSize_,
                                                               cfg2::OutDstVectorSize_>;

                device_op_instances.push_back(
                    std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
            });
    });
};

#define ADD_BLOCKWISE_INST_BY_TYPE(inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...) \
    template void add_device_reduce_instance_blockwise<inT,                                     \
                                                       compT,                                   \
                                                       outT,                                    \
                                                       Rank,                                    \
                                                       Sequence<__VA_ARGS__>,                   \
                                                       ReduceOpId,                              \
                                                       NanOpt,                                  \
                                                       IndicesOpt>(                             \
        std::vector<deviceReducePtrType<compT, ReduceOpId>> & device_op_instances)

#define ADD_BLOCKWISE_INST_BY_ID(inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...) \
    ADD_BLOCKWISE_INST_BY_TYPE(inT,                                                           \
                               compT,                                                         \
                               outT,                                                          \
                               static_cast<ReduceTensorOp_t>(ReduceOpId),                     \
                               static_cast<NanPropagation_t>(NanOpt),                         \
                               static_cast<ReduceTensorIndices_t>(IndicesOpt),                \
                               Rank,                                                          \
                               __VA_ARGS__)

// half, half, half
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 2, 1);       //

// half, float, half
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(half_t, float, half_t, 7, 0, 0, 2, 1);       //

// float, float, float
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 7, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 2, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 2, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 3, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 3, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 4, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 4, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 2, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 2, 0, 1, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 3, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 3, 0, 1, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 4, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, float, float, 4, 0, 1, 2, 1);       //

// float, double, float
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(float, double, float, 7, 0, 0, 2, 1);       //

// double, double, double
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 7, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 0, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 0, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 1, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 1, 2, 1);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 1, 4, 0);       //
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 1, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
