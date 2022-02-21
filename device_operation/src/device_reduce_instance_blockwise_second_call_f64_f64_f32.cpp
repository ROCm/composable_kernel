#include "device_reduce_instance_blockwise_second_call.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// double, double, float
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 0, 0, 0, 4, 0, 1, 2); // for ADD
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 0, 0, 0, 4, 0);
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 0, 0, 0, 2, 1);
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 5, 0, 0, 4, 0, 1, 2); // for AVG
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 5, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 5, 0, 0, 2, 1);       //
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 7, 0, 0, 4, 0, 1, 2); // for NORM2
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 7, 0, 0, 4, 0);       //
ADD_BLOCKWISE_SECOND_CALL_INST_BY_ID(double, double, float, 7, 0, 0, 2, 1);       //

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
