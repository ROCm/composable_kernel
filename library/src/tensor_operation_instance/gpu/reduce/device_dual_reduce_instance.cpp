#include "ck/library/tensor_operation_instance/gpu/reduce/device_dual_reduce_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

// clang-format off
// InDataType | AccDataType | OutDataType1 | OutDataType2 | ReduceOp | InElementwiseOp1 | AccElementwiseOp1 | InElementwiseOp2 | AccElementwiseOp2 | PropagateNan | Rank | NumReduceDim 
ADD_DUAL_REDUCE_BLOCKWISE_INST(half_t, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3);  
ADD_DUAL_REDUCE_BLOCKWISE_INST(float, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3); 
ADD_DUAL_REDUCE_BLOCKWISE_INST(float, double, double, double, Sum, PassThrough, Divide, Square, Divide, false, 4, 3); 

ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST(half_t, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3);  
ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST(float, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3); 
ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST(float, double, double, double, Sum, PassThrough, Divide, Square, Divide, false, 4, 3);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
