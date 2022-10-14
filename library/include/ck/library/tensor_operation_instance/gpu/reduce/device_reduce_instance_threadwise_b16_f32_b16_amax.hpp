// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOpId | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 4, 3, ReduceTensorOp::AMAX, false, false>(std::vector<deviceReduceThreadWisePtrType<4, 3, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 4, 4, ReduceTensorOp::AMAX, false, false>(std::vector<deviceReduceThreadWisePtrType<4, 4, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 4, 1, ReduceTensorOp::AMAX, false, false>(std::vector<deviceReduceThreadWisePtrType<4, 1, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 2, 1, ReduceTensorOp::AMAX, false, false>(std::vector<deviceReduceThreadWisePtrType<2, 1, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 4, 3, ReduceTensorOp::AMAX, false, true>(std::vector<deviceReduceThreadWisePtrType<4, 3, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 4, 4, ReduceTensorOp::AMAX, false, true>(std::vector<deviceReduceThreadWisePtrType<4, 4, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 4, 1, ReduceTensorOp::AMAX, false, true>(std::vector<deviceReduceThreadWisePtrType<4, 1, ReduceTensorOp::AMAX>>&);
extern template void add_device_reduce_instance_threadwise<bhalf_t, float, bhalf_t, 2, 1, ReduceTensorOp::AMAX, false, true>(std::vector<deviceReduceThreadWisePtrType<2, 1, ReduceTensorOp::AMAX>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
