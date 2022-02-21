#include "device_reduce_instance_multiblock_two_call.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// float, float, float
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 2, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 2, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 3, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 3, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 4, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 4, 0, 0, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 2, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 2, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 3, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 3, 0, 1, 2, 1);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 4, 0, 1, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 4, 0, 1, 2, 1);       //

// float, float, float
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 7, 0, 0, 4, 0);       //
ADD_MULTIBLOCK_TWO_CALL_INST_BY_ID(float, float, float, 7, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
