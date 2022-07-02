// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | ReduceOpId | NanPropaOpt | IndicesOpt | Rank | NumReduceDim 
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 4, 3); // for MIN
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 4, 4);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 4, 3); // for MAX
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 4, 4);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 4, 3); // for AMAX
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 4, 4);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 4, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 0, 2, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 4, 3); // for MIN
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 4, 4);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 4, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 2, 0, 1, 2, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 4, 3); // for MAX
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 4, 4);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 4, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 3, 0, 1, 2, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 4, 3); // for AMAX
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 4, 4);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 4, 1);       
ADD_BLOCKWISE_INST_REF_BY_ID(int8_t, int8_t, int8_t, 4, 0, 1, 2, 1);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
