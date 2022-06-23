#ifndef DEVICE_DUAL_REDUCE_INSTANTCE_HPP
#define DEVICE_DUAL_REDUCE_INSTANTCE_HPP

#include "device_dual_reduce_instance_blockwise.hpp"
#include "device_dual_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

using Sum = ck::reduce::Add;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using Square = ck::tensor_operation::element_wise::UnarySquare;

using Divide = ck::tensor_operation::element_wise::UnaryDivide;

// clang-format off
// InDataType | AccDataType | OutDataType1 | OutDataType2 | ReduceOp | InElementwiseOp1 | AccElementwiseOp1 | InElementwiseOp2 | AccElementwiseOp2 | PropagateNan | Rank | NumReduceDim 
ADD_DUAL_REDUCE_BLOCKWISE_INST_REF(half_t, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3);  
ADD_DUAL_REDUCE_BLOCKWISE_INST_REF(float, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3); 
ADD_DUAL_REDUCE_BLOCKWISE_INST_REF(float, double, double, double, Sum, PassThrough, Divide, Square, Divide, false, 4, 3); 

ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST_REF(half_t, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3);  
ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST_REF(float, float, float, float, Sum, PassThrough, Divide, Square, Divide, false, 4, 3); 
ADD_DUAL_REDUCE_MULTIBLOCK_ATOMIC_ADD_INST_REF(float, double, double, double, Sum, PassThrough, Divide, Square, Divide, false, 4, 3);
// clang-format on

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif
