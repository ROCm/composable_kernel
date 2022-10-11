// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | PropagateNan | UseIndex | Rank | NumReduceDim
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::ADD, false, false, 4, 3); // for ADD
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::ADD, false, false, 4, 4);
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::ADD, false, false, 4, 1);
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::ADD, false, false, 2, 1);
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::AVG, false, false, 4, 3); // for AVG
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::AVG, false, false, 4, 4);
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::AVG, false, false, 4, 1);
ADD_BLOCKWISE_INST(int8_t, int32_t, int8_t, ReduceTensorOp::AVG, false, false, 2, 1);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
