#include "device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDims
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 4, 3); // for MIN
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 4, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 2, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 4, 3); // for MAX
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 4, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 2, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 4, 3); // for AMAX
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 4, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 2, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 4, 3); // for MIN
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 4, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 2, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 4, 3); // for MAX
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 4, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 2, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 4, 3); // for AMAX
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 4, 1);       
ADD_THREADWISE_INST_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
