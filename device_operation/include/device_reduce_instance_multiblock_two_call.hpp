#ifndef DEVICE_REDUCE_INSTANCE_MULTIBLOCK_TWO_CALL_HPP
#define DEVICE_REDUCE_INSTANCE_MULTIBLOCK_TWO_CALL_HPP

#include "reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
extern void add_device_reduce_instance_multiblock_two_call(
    std::vector<DeviceReducePtr<
        typename reduce_unary_operator<compType, reduceOp, true, false>::preUnaryOp,
        typename reduce_unary_operator<compType, reduceOp, true, false>::posUnaryOp>>&
        device_op_instances);

#define ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_TYPE(                                              \
    inT, compT, outT, reduceOp, nanOpt, indicesOpt, rank, ...)                                 \
    extern template void add_device_reduce_instance_multiblock_two_call<inT,                   \
                                                                        compT,                 \
                                                                        outT,                  \
                                                                        rank,                  \
                                                                        Sequence<__VA_ARGS__>, \
                                                                        reduceOp,              \
                                                                        nanOpt,                \
                                                                        indicesOpt>(           \
        std::vector<DeviceReducePtr<                                                           \
            typename reduce_unary_operator<compT, reduceOp, true, false>::preUnaryOp,          \
            typename reduce_unary_operator<compT, reduceOp, true, false>::posUnaryOp>> &       \
        device_op_instances)

#define ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(                                              \
    inT, compT, outT, reduceOp, nanOpt, indicesOpt, rank, ...)                               \
    ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_TYPE(inT,                                            \
                                             compT,                                          \
                                             outT,                                           \
                                             static_cast<ReduceTensorOp_t>(reduceOp),        \
                                             static_cast<NanPropagation_t>(nanOpt),          \
                                             static_cast<ReduceTensorIndices_t>(indicesOpt), \
                                             rank,                                           \
                                             __VA_ARGS__)

// half, half, half
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 2, 1);       //

// half, float, half
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 0, 0, 0, 4, 0);
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 5, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 5, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 7, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(half_t, float, half_t, 7, 0, 0, 2, 1);       //

// float, float, float
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 2, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 3, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 4, 0, 1, 2, 1);       //

// float, float, float
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 7, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, float, float, 7, 0, 0, 2, 1);       //

// float, double, float
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, double, float, 7, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_REF_BY_ID(float, double, float, 7, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
