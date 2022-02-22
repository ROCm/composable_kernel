#ifndef DEVICE_REDUCE_INSTANCE_THREADWISE_F16_F16_F16_HPP
#define DEVICE_REDUCE_INSTANCE_THREADWISE_F16_F16_F16_HPP

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"
#include "device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | InnerDims 
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_REF_BY_ID(half_t, half_t, half_t, 4, 0, 1, 2, 1);       //
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
