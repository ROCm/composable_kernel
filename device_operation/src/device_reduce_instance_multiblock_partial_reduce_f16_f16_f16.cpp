#include "device_reduce_instance_multiblock_partial_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | InnerDims
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_PARTIAL_REDUCE_INST_BY_ID(half_t, half_t, half_t, 4, 0, 1, 2, 1);       //
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
