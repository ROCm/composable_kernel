// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F32 = float;

template <index_t Rank, index_t Reduce>
using device_softmax_f32_f32_instances = std::tuple<
    // clang-format off
        // InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 8, 32, 1, 8, 1, 1, 1>, // fallback kernel
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 8, 32, 1, 8, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 4, 64, 1, 8, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 2, 128, 1, 8, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 2, 128, 1, 16, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 2, 128, 1, 32, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 4>,
        DeviceSoftmax<F32, F32, F32, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 4>
    // clang-format on
    >;

void add_device_softmax_f32_f32_rank3_instances(std::vector<DeviceNormalizationPtr>& instances)
{
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<3, 1>{});
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<3, 2>{});
}

void add_device_softmax_f32_f32_rank4_instances(std::vector<DeviceNormalizationPtr>& instances)
{
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<4, 1>{});
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<4, 2>{});
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<4, 3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
