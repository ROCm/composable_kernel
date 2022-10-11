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
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::ADD, false, false, 4, 3); // for ADD
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::ADD, false, false, 4, 4);
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::ADD, false, false, 4, 1);
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::ADD, false, false, 2, 1);
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AVG, false, false, 4, 3); // for AVG
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AVG, false, false, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AVG, false, false, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AVG, false, false, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::NORM2, false, false, 4, 3); // for NORM2
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::NORM2, false, false, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::NORM2, false, false, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::NORM2, false, false, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, false, 4, 3); // for MIN
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, false, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, false, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, false, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, false, 4, 3); // for MAX
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, false, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, false, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, false, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, false, 4, 3); // for AMAX
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, false, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, false, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, false, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, true, 4, 3); // for MIN
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, true, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, true, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MIN, false, true, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, true, 4, 3); // for MAX
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, true, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, true, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::MAX, false, true, 2, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, true, 4, 3); // for AMAX
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, true, 4, 4);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, true, 4, 1);       
ADD_BLOCKWISE_INST(float, float, float, ReduceTensorOp::AMAX, false, true, 2, 1);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
