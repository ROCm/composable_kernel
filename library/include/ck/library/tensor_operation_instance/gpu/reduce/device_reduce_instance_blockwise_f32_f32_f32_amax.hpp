// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_blockwise<F32, F32, F32, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
