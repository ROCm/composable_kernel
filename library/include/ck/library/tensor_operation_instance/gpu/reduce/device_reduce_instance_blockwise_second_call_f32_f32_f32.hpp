#ifndef DEVICE_REDUCE_INSTANCE_BLOCKWISE_SECOND_CALL_F32_F32_F32_HPP
#define DEVICE_REDUCE_INSTANCE_BLOCKWISE_SECOND_CALL_F32_F32_F32_HPP

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"
#include "device_reduce_instance_blockwise_second_call.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim 
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 0, 0, 0, 4, 3); // for ADD
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 0, 0, 0, 4, 4);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 0, 0, 0, 4, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 5, 0, 0, 4, 3); // for AVG
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 5, 0, 0, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 5, 0, 0, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 5, 0, 0, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 7, 0, 0, 4, 3); // for NORM2
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 7, 0, 0, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 7, 0, 0, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 7, 0, 0, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 0, 4, 3); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 0, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 0, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 0, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 0, 4, 3); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 0, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 0, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 0, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 0, 4, 3); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 0, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 0, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 0, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 1, 4, 3); // for MIN
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 1, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 1, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 2, 0, 1, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 1, 4, 3); // for MAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 1, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 1, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 3, 0, 1, 2, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 1, 4, 3); // for AMAX
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 1, 4, 4);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 1, 4, 1);       
ADD_BLOCKWISE_SECOND_CALL_INST_REF_BY_ID(float, float, 4, 0, 1, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
