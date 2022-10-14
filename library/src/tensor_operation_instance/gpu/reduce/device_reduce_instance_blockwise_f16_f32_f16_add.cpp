// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOpId | PropagateNan | UseIndex 
template void add_device_reduce_instance_blockwise<half_t, float, half_t, 4, 3, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 3, ReduceTensorOp::ADD>>&); 
template void add_device_reduce_instance_blockwise<half_t, float, half_t, 4, 4, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 4, ReduceTensorOp::ADD>>&); 
template void add_device_reduce_instance_blockwise<half_t, float, half_t, 4, 1, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 1, ReduceTensorOp::ADD>>&); 
template void add_device_reduce_instance_blockwise<half_t, float, half_t, 2, 1, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceBlockWisePtrType<2, 1, ReduceTensorOp::ADD>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
