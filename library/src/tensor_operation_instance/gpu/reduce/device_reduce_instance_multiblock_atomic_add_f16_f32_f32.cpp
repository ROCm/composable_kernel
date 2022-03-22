#include "device_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 4, 3); // for ADD
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 4, 4);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 4, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 4, 3); // for AVG
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 4, 4);       
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 4, 1);       
ADD_MULTIBLOCK_ATOMIC_ADD_INST_BY_ID(half_t, float, float, 5, 0, 0, 2, 1);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
