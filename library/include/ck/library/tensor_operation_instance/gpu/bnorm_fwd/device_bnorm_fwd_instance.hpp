#ifndef DEVICE_BNORM_FWD_INSTANTCE_HPP
#define DEVICE_BNORM_FWD_INSTANTCE_HPP

#include "device_bnorm_fwd_with_reduce_blockwise_instance.hpp"
#include "device_bnorm_fwd_with_reduce_multiblock_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_bnorm_fwd_instance {

// clang-format off
// InOutDataType | AccDataType
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST_REF(float, float);  
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST_REF(double, double);  
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST_REF(half_t, float);  
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST_REF(bhalf_t, float);  
ADD_BNORM_FWD_WITH_REDUCE_BLOCKWISE_INST_REF(int8_t, float);  

ADD_BNORM_FWD_WITH_REDUCE_MULTIBLOCK_INST_REF(float, float);  
ADD_BNORM_FWD_WITH_REDUCE_MULTIBLOCK_INST_REF(double, double);  
ADD_BNORM_FWD_WITH_REDUCE_MULTIBLOCK_INST_REF(half_t, float);  
ADD_BNORM_FWD_WITH_REDUCE_MULTIBLOCK_INST_REF(bhalf_t, float);  
ADD_BNORM_FWD_WITH_REDUCE_MULTIBLOCK_INST_REF(int8_t, float);
// clang-format on

} // namespace device_bnorm_fwd_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
