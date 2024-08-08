// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 12, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 12, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 11, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 11, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 10, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 10, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 9, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 9, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 8, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 8, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 7, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 7, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 6, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 6, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 5, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 5, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 4, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 4, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 3, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 3, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 2, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 2, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 12, 1, ReduceAdd, PassThrough, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 12, 1, ReduceAdd, PassThrough, UnarySqrt, false, false>>&);
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 3, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&); 
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 4, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&); 
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 4, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&); 
template void add_device_reduce_instance_blockwise<BF16, F32, BF16, 2, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
