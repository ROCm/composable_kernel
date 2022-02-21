#include "device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// float, double, float
ADD_THREADWISE_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_THREADWISE_INST_BY_ID(float, double, float, 0, 0, 0, 4, 0);
ADD_THREADWISE_INST_BY_ID(float, double, float, 0, 0, 0, 2, 1);
ADD_THREADWISE_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_THREADWISE_INST_BY_ID(float, double, float, 5, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, double, float, 5, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_THREADWISE_INST_BY_ID(float, double, float, 7, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(float, double, float, 7, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
