// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
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

#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
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

void add_device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f16_16_16_instances(
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
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkyxc_ngkhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkyxc_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkyxc_ngkhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_f16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_f16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_f32_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngchw_gkcyx_ngkhw_bf16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

// conv3d backward data
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(
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
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
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

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16_16_instances(
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
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgc_gkzyxc_ndhwgk_input_f16_comp_bf8_f8_instances(
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
                                                                  PassThrough,
                                                                  BF8,
                                                                  F8>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkzyxc_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_f32_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngcdhw_gkczyx_ngkdhw_bf16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
