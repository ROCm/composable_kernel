// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOpId | PropagateNan | UseIndex 
template void add_device_reduce_instance_multiblock_atomic_add<bhalf_t, float, float, 4, 3, ReduceTensorOp::AVG, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<4, 3, ReduceTensorOp::AVG>>&); 
template void add_device_reduce_instance_multiblock_atomic_add<bhalf_t, float, float, 4, 4, ReduceTensorOp::AVG, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<4, 4, ReduceTensorOp::AVG>>&); 
template void add_device_reduce_instance_multiblock_atomic_add<bhalf_t, float, float, 4, 1, ReduceTensorOp::AVG, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<4, 1, ReduceTensorOp::AVG>>&); 
template void add_device_reduce_instance_multiblock_atomic_add<bhalf_t, float, float, 2, 1, ReduceTensorOp::AVG, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<2, 1, ReduceTensorOp::AVG>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
