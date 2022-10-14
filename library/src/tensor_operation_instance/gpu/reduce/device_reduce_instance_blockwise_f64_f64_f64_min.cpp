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
template void add_device_reduce_instance_blockwise<double, double, double, 4, 3, ReduceTensorOp::MIN, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 3, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 4, 4, ReduceTensorOp::MIN, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 4, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 4, 1, ReduceTensorOp::MIN, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 1, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 2, 1, ReduceTensorOp::MIN, false, false>(std::vector<deviceReduceBlockWisePtrType<2, 1, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 4, 3, ReduceTensorOp::MIN, false, true>(std::vector<deviceReduceBlockWisePtrType<4, 3, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 4, 4, ReduceTensorOp::MIN, false, true>(std::vector<deviceReduceBlockWisePtrType<4, 4, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 4, 1, ReduceTensorOp::MIN, false, true>(std::vector<deviceReduceBlockWisePtrType<4, 1, ReduceTensorOp::MIN>>&);
template void add_device_reduce_instance_blockwise<double, double, double, 2, 1, ReduceTensorOp::MIN, false, true>(std::vector<deviceReduceBlockWisePtrType<2, 1, ReduceTensorOp::MIN>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
