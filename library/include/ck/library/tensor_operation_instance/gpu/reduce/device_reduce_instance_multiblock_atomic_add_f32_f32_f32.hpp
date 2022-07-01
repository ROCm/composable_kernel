// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim 
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 0, 0, 0, 4, 3); // for ADD
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 0, 0, 0, 4, 4);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 0, 0, 0, 4, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 0, 0, 0, 2, 1);
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 5, 0, 0, 4, 3); // for AVG
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 5, 0, 0, 4, 4);       
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 5, 0, 0, 4, 1);       
ADD_MULTIBLOCK_ATOMIC_ADD_INST_REF_BY_ID(float, float, float, 5, 0, 0, 2, 1);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
