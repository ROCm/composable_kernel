// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "device_conv2d_dl_int8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_conv2d_dl_bias_perchannel_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                GK_GK_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                I32_F32_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Add_Mul2_Clamp>>>& instances)
{
    // dl
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Mul2_Clamp,
                                                                           ConvFwdDefault,
                                                                           4>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Mul2_Clamp,
                                                                           ConvFwd1x1P0,
                                                                           4>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Mul2_Clamp,
                                                                           ConvFwd1x1S1P0,
                                                                           4>{});
}

void add_device_conv2d_dl_bias_relu_perchannel_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                GK_GK_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                I32_F32_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Add_Relu_Mul2_Clamp>>>& instances)
{
    // dl
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Relu_Mul2_Clamp,
                                                                           ConvFwdDefault,
                                                                           4>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Relu_Mul2_Clamp,
                                                                           ConvFwd1x1P0,
                                                                           4>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Relu_Mul2_Clamp,
                                                                           ConvFwd1x1S1P0,
                                                                           4>{});
}

void add_device_conv2d_dl_bias_tanh_perchannel_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                GK_GK_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                I32_F32_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Add_Mul2_TanH_Mul_Clamp>>>&
        instances)
{
    // dl
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Mul2_TanH_Mul_Clamp,
                                                                           ConvFwdDefault,
                                                                           4>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Mul2_TanH_Mul_Clamp,
                                                                           ConvFwd1x1P0,
                                                                           4>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_dl_int8_instances<NHWGC,
                                                                           GKYXC,
                                                                           GK_GK_Tuple,
                                                                           NHWGK,
                                                                           I32_F32_Tuple,
                                                                           Add_Mul2_TanH_Mul_Clamp,
                                                                           ConvFwd1x1S1P0,
                                                                           4>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
