#include "device_reduce_instance_multiblock_partial_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 0, 0, 0, 4, 3); // for ADD
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 0, 0, 0, 4, 4);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 0, 0, 0, 4, 1);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 5, 0, 0, 4, 3); // for AVG
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 5, 0, 0, 4, 4);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 5, 0, 0, 4, 1);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 5, 0, 0, 2, 1);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 7, 0, 0, 4, 3); // for NORM2
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 7, 0, 0, 4, 4);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 7, 0, 0, 4, 1);       
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, float, 7, 0, 0, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
