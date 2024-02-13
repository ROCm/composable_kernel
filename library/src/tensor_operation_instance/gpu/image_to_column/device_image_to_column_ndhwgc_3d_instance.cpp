// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/conv_tensor_rearrange/device_image_to_column_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::conv_tensor_rearrange_op;

void add_device_image_to_column_ndhwgc_3d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, NDHWGC, BF16, BF16, ImageToColumn>>>&
        instances)
{
#ifdef CK_ENABLE_BF16
    add_device_operation_instances(instances, device_image_to_column_bf16_instances<3, NDHWGC>{});
#else
    ignore = instances;
#endif
}

void add_device_image_to_column_ndhwgc_3d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, NDHWGC, F16, F16, ImageToColumn>>>&
        instances)
{
#ifdef CK_ENABLE_FP16
    add_device_operation_instances(instances, device_image_to_column_f16_instances<3, NDHWGC>{});
#else
    ignore = instances;
#endif
}

void add_device_image_to_column_ndhwgc_3d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, NDHWGC, F32, F32, ImageToColumn>>>&
        instances)
{
#ifdef CK_ENABLE_FP32
    add_device_operation_instances(instances, device_image_to_column_f32_instances<3, NDHWGC>{});
#else
    ignore = instances;
#endif
}

void add_device_image_to_column_ndhwgc_3d_i8_instances(
    std::vector<
        std::unique_ptr<DeviceConvTensorRearrange<3, NDHWGC, int8_t, int8_t, ImageToColumn>>>&
        instances)
{
#ifdef CK_ENABLE_INT8
    add_device_operation_instances(instances, device_image_to_column_i8_instances<3, NDHWGC>{});
#else
    ignore = instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
