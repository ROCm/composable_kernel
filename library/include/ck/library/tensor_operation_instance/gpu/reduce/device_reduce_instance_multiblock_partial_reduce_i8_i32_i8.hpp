#ifndef DEVICE_REDUCE_INSTANCE_MULTIBLOCK_PARTIAL_REDUCE_I8_I32_I8_HPP
#define DEVICE_REDUCE_INSTANCE_MULTIBLOCK_PARTIAL_REDUCE_I8_I32_I8_HPP

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"
#include "device_reduce_instance_multiblock_partial_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim 
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 0, 0, 0, 4, 3); // for ADD
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 0, 0, 0, 4, 4);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 0, 0, 0, 4, 1);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 5, 0, 0, 4, 3); // for AVG
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 5, 0, 0, 4, 4);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 5, 0, 0, 4, 1);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_REF_BY_ID(int8_t, int32_t, int8_t, 5, 0, 0, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
