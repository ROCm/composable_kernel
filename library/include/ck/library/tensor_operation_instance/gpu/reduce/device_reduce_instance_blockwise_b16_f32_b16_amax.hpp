// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 12, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16,12, 12, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 11, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16,12, 11, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 10, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16,12, 10, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 9, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 9, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 8, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 8, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 7, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 7, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 6, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 6, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 5, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 5, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 2, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 2, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16,12, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
extern template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
