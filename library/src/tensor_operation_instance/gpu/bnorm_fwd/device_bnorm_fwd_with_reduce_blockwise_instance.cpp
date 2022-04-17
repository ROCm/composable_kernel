#include "device_bnorm_fwd_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_bnorm_fwd_instance {

// clang-format off
// InOutDataType | AccDataType 
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST(float, float); 
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST(double, double); 
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST(half_t, float); 
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST(bhalf_t, float); 
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST(int8_t, float);
// clang-format on

} // namespace device_bnorm_fwd_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
