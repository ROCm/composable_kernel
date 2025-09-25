// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv2d backward data
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
