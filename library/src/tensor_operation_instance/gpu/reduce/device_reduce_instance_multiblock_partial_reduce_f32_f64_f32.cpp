#include "device_reduce_instance_multiblock_partial_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 3); // for NORM2
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 4);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 1);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(float, double, float, 7, 0, 0, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
