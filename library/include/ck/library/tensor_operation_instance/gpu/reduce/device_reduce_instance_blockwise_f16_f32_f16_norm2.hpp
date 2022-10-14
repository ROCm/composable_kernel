// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOpId | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_blockwise<half_t, float, half_t, 4, 3, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 3, ReduceTensorOp::NORM2>>&); 
extern template void add_device_reduce_instance_blockwise<half_t, float, half_t, 4, 4, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 4, ReduceTensorOp::NORM2>>&); 
extern template void add_device_reduce_instance_blockwise<half_t, float, half_t, 4, 1, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceBlockWisePtrType<4, 1, ReduceTensorOp::NORM2>>&); 
extern template void add_device_reduce_instance_blockwise<half_t, float, half_t, 2, 1, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceBlockWisePtrType<2, 1, ReduceTensorOp::NORM2>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
