#include "device_reduce_instance_ref_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// half, float, float
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 0, 0, 0, 4, 0);
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 1, 0, 0, 4, 0, 1, 2); // for MUL
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 1, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 5, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 6, 0, 0, 4, 0, 1, 2); // for NORM1
ADD_INST_REF_BY_ID(multiblock_atomic_add, half_t, float, float, 6, 0, 0, 4, 0);       //

// float, float, float
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 0, 0, 0, 4, 0);
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 1, 0, 0, 4, 0, 1, 2); // for MUL
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 1, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 5, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 6, 0, 0, 4, 0, 1, 2); // for NORM1
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, float, float, 6, 0, 0, 4, 0);       //

// float, double, float
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 0, 0, 0, 4, 0);
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 1, 0, 0, 4, 0, 1, 2); // for MUL
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 1, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 5, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 6, 0, 0, 4, 0, 1, 2); // for NORM1
ADD_INST_REF_BY_ID(multiblock_atomic_add, float, double, float, 6, 0, 0, 4, 0);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
