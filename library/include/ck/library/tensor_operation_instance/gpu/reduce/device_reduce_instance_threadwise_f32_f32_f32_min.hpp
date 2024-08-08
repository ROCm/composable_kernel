// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 12, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 12, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 11, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 11, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 10, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 10, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 9, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 9, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 8, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 8, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 7, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 7, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 6, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 6, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 5, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 5, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 4, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 4, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 3, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 3, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 2, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 2, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 12, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 12, 1, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 4, 3, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 4, 4, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 4, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 2, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceMin, PassThrough, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 4, 3, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceMin, PassThrough, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 4, 4, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceMin, PassThrough, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 4, 1, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceMin, PassThrough, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_threadwise<F32, F32, F32, 2, 1, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceMin, PassThrough, PassThrough, false, true>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
