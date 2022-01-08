#include "device_reduce_instance_ref_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// half, half, half
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 2, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 2, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 3, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 3, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 4, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 4, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 2, 0, 1, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 2, 0, 1, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 3, 0, 1, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 3, 0, 1, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 4, 0, 1, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, half_t, half_t, 4, 0, 1, 2, 1);       //

// half, float, half
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 0, 0, 0, 4, 0);
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 0, 0, 0, 2, 1);
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 5, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 5, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 7, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, half_t, float, half_t, 7, 0, 0, 2, 1);       //

// float, float, float
ADD_INST_REF_BY_ID(blockwise, float, float, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_REF_BY_ID(blockwise, float, float, float, 0, 0, 0, 4, 0);
ADD_INST_REF_BY_ID(blockwise, float, float, float, 0, 0, 0, 2, 1);
ADD_INST_REF_BY_ID(blockwise, float, float, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_REF_BY_ID(blockwise, float, float, float, 5, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 5, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_INST_REF_BY_ID(blockwise, float, float, float, 7, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 7, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_INST_REF_BY_ID(blockwise, float, float, float, 2, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 2, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_INST_REF_BY_ID(blockwise, float, float, float, 3, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 3, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_INST_REF_BY_ID(blockwise, float, float, float, 4, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 4, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_INST_REF_BY_ID(blockwise, float, float, float, 2, 0, 1, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 2, 0, 1, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_INST_REF_BY_ID(blockwise, float, float, float, 3, 0, 1, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 3, 0, 1, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_INST_REF_BY_ID(blockwise, float, float, float, 4, 0, 1, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, float, float, 4, 0, 1, 2, 1);       //

// float, double, float
ADD_INST_REF_BY_ID(blockwise, float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_INST_REF_BY_ID(blockwise, float, double, float, 0, 0, 0, 4, 0);
ADD_INST_REF_BY_ID(blockwise, float, double, float, 0, 0, 0, 2, 1);
ADD_INST_REF_BY_ID(blockwise, float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_INST_REF_BY_ID(blockwise, float, double, float, 5, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, double, float, 5, 0, 0, 2, 1);       //
ADD_INST_REF_BY_ID(blockwise, float, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_INST_REF_BY_ID(blockwise, float, double, float, 7, 0, 0, 4, 0);       //
ADD_INST_REF_BY_ID(blockwise, float, double, float, 7, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
