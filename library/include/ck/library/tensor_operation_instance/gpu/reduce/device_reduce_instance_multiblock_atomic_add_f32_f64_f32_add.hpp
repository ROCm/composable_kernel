// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOpId | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_multiblock_atomic_add<float, double, float, 4, 3, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<4, 3, ReduceTensorOp::ADD>>&); 
extern template void add_device_reduce_instance_multiblock_atomic_add<float, double, float, 4, 4, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<4, 4, ReduceTensorOp::ADD>>&); 
extern template void add_device_reduce_instance_multiblock_atomic_add<float, double, float, 4, 1, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<4, 1, ReduceTensorOp::ADD>>&); 
extern template void add_device_reduce_instance_multiblock_atomic_add<float, double, float, 2, 1, ReduceTensorOp::ADD, false, false>(std::vector<deviceReduceMultiBlockAtomicAddPtrType<2, 1, ReduceTensorOp::ADD>>&);
// clang-format on
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
