#include "device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDims
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 0, 0, 0, 4, 3); // for ADD
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 0, 0, 0, 4, 1);
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 5, 0, 0, 4, 3); // for AVG
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 5, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 5, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 7, 0, 0, 4, 3); // for NORM2
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 7, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 7, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 0, 4, 3); // for MIN
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 0, 4, 3); // for MAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 0, 4, 3); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 1, 4, 3); // for MIN
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 1, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 2, 0, 1, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 1, 4, 3); // for MAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 1, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 3, 0, 1, 2, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 1, 4, 3); // for AMAX
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 1, 4, 1);       
ADD_BLOCKWISE_INST_BY_ID(double, double, double, 4, 0, 1, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
