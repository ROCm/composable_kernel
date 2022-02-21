#include "device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// double, double, double
ADD_THREADWISE_INST_BY_ID(double, double, double, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_THREADWISE_INST_BY_ID(double, double, double, 0, 0, 0, 4, 0);
ADD_THREADWISE_INST_BY_ID(double, double, double, 0, 0, 0, 2, 1);
ADD_THREADWISE_INST_BY_ID(double, double, double, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_THREADWISE_INST_BY_ID(double, double, double, 5, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 5, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_THREADWISE_INST_BY_ID(double, double, double, 7, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 7, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 2, 0, 0, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_BY_ID(double, double, double, 2, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 2, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 3, 0, 0, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_BY_ID(double, double, double, 3, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 3, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 4, 0, 0, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_BY_ID(double, double, double, 4, 0, 0, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 4, 0, 0, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 2, 0, 1, 4, 0, 1, 2); // for MIN
ADD_THREADWISE_INST_BY_ID(double, double, double, 2, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 2, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 3, 0, 1, 4, 0, 1, 2); // for MAX
ADD_THREADWISE_INST_BY_ID(double, double, double, 3, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 3, 0, 1, 2, 1);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 4, 0, 1, 4, 0, 1, 2); // for AMAX
ADD_THREADWISE_INST_BY_ID(double, double, double, 4, 0, 1, 4, 0);       //
ADD_THREADWISE_INST_BY_ID(double, double, double, 4, 0, 1, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
