// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/image_to_column/device_image_to_column_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_image_to_column_nhwc_2d_bf16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, BF16, BF16>>>& instances)
{
    add_device_operation_instances(instances, device_image_to_column_bf16_instances<2, GNHWC>{});
}

void add_device_image_to_column_nhwc_2d_f16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, F16, F16>>>& instances)
{
    add_device_operation_instances(instances, device_image_to_column_f16_instances<2, GNHWC>{});
}

void add_device_image_to_column_nhwc_2d_f32_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, F32, F32>>>& instances)
{
    add_device_operation_instances(instances, device_image_to_column_f32_instances<2, GNHWC>{});
}

void add_device_image_to_column_nhwc_2d_i8_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, int8_t, int8_t>>>& instances)
{
    add_device_operation_instances(instances, device_image_to_column_i8_instances<2, GNHWC>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
