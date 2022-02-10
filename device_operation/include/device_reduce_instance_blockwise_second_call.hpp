#ifndef DEVICE_REDUCE_INSTANCE_BLOCKWISE_SECOND_CALL_HPP
#define DEVICE_REDUCE_INSTANCE_BLOCKWISE_SECOND_CALL_HPP

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          typename InnerDims,
          ReduceTensorOp_t ReduceOpId,
          NanPropagation_t NanOpt,
          ReduceTensorIndices_t IndicesOpt>
extern void add_device_reduce_instance_blockwise_second_call(
    std::vector<
        DeviceReducePtr<typename reduce_unary_operator<AccDataType, ReduceOpId, false, true>::
                            InElementwiseOperation,
                        typename reduce_unary_operator<AccDataType, ReduceOpId, false, true>::
                            AccElementwiseOperation>>& device_op_instances);

#define ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_TYPE(                                              \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...)                                 \
    extern template void add_device_reduce_instance_blockwise_second_call<inT,                   \
                                                                          compT,                 \
                                                                          outT,                  \
                                                                          Rank,                  \
                                                                          Sequence<__VA_ARGS__>, \
                                                                          ReduceOpId,            \
                                                                          NanOpt,                \
                                                                          IndicesOpt>(           \
        std::vector<                                                                             \
            DeviceReducePtr<typename reduce_unary_operator<compT, ReduceOpId, false, true>::     \
                                InElementwiseOperation,                                          \
                            typename reduce_unary_operator<compT, ReduceOpId, false, true>::     \
                                AccElementwiseOperation>> &                                      \
        device_op_instances)

#define ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(                                              \
    inT, compT, outT, ReduceOpId, NanOpt, IndicesOpt, Rank, ...)                               \
    ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_TYPE(inT,                                            \
                                               compT,                                          \
                                               outT,                                           \
                                               static_cast<ReduceTensorOp_t>(ReduceOpId),      \
                                               static_cast<NanPropagation_t>(NanOpt),          \
                                               static_cast<ReduceTensorIndices_t>(IndicesOpt), \
                                               Rank,                                           \
                                               __VA_ARGS__)

// half, half, half
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 2, 1);       //

// float, float, half
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, half_t, 7, 0, 0, 2, 1);       //

// float, float, float
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 7, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 1, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 1, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 1, 2, 1);       //

// double, double, float
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, float, 7, 0, 0, 2, 1);       //

// double, double, double
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 7, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 2, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 2, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 3, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 3, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 4, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 4, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 2, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 2, 0, 1, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 3, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 3, 0, 1, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 4, 0, 1, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(double, double, double, 4, 0, 1, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
