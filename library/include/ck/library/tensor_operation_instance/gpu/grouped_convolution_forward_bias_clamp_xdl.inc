// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances);

#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
