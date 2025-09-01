// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                        NHWGK,
                                                        GKYXC,
                                                        Tuple<>,
                                                        NHWGC,
                                                        F16,
                                                        F16,
                                                        Tuple<>,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_16x16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                        NHWGK,
                                                        GKYXC,
                                                        Tuple<>,
                                                        NHWGC,
                                                        F16,
                                                        F16,
                                                        Tuple<>,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                        NDHWGK,
                                                        GKZYXC,
                                                        Tuple<>,
                                                        NDHWGC,
                                                        F16,
                                                        F16,
                                                        Tuple<>,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f16_16x16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                        NDHWGK,
                                                        GKZYXC,
                                                        Tuple<>,
                                                        NDHWGC,
                                                        F16,
                                                        F16,
                                                        Tuple<>,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                        NHWGK,
                                                        GKYXC,
                                                        Tuple<>,
                                                        NHWGC,
                                                        BF16,
                                                        BF16,
                                                        Tuple<>,
                                                        BF16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_16x16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                        NHWGK,
                                                        GKYXC,
                                                        Tuple<>,
                                                        NHWGC,
                                                        BF16,
                                                        BF16,
                                                        Tuple<>,
                                                        BF16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                        NDHWGK,
                                                        GKZYXC,
                                                        Tuple<>,
                                                        NDHWGC,
                                                        BF16,
                                                        BF16,
                                                        Tuple<>,
                                                        BF16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_16x16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                        NDHWGK,
                                                        GKZYXC,
                                                        Tuple<>,
                                                        NDHWGC,
                                                        BF16,
                                                        BF16,
                                                        Tuple<>,
                                                        BF16,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                        NHWGK,
                                                        GKYXC,
                                                        Tuple<>,
                                                        NHWGC,
                                                        F32,
                                                        F32,
                                                        Tuple<>,
                                                        F32,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_16x16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                        NHWGK,
                                                        GKYXC,
                                                        Tuple<>,
                                                        NHWGC,
                                                        F32,
                                                        F32,
                                                        Tuple<>,
                                                        F32,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f32_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                        NDHWGK,
                                                        GKZYXC,
                                                        Tuple<>,
                                                        NDHWGC,
                                                        F32,
                                                        F32,
                                                        Tuple<>,
                                                        F32,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);

void add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f32_16x16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                        NDHWGK,
                                                        GKZYXC,
                                                        Tuple<>,
                                                        NDHWGC,
                                                        F32,
                                                        F32,
                                                        Tuple<>,
                                                        F32,
                                                        PassThrough,
                                                        PassThrough,
                                                        Clamp>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
