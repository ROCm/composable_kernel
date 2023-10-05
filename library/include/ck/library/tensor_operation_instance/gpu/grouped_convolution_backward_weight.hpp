// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// xdl
// conv1d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// conv2d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// conv3d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
void add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           BF8,
                                                           F8>>>& instances);
#endif

#ifdef DL_KERNELS
// dl
// conv1d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// conv2d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// conv3d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#endif

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeTypeA,
    ComputeTypeB>>
{
    using DeviceOp = DeviceGroupedConvBwdWeight<NumDimSpatial,
                                                InLayout,
                                                WeiLayout,
                                                OutLayout,
                                                InDataType,
                                                WeiDataType,
                                                OutDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ComputeTypeA,
                                                ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 1)
        {
            if constexpr(is_same_v<InLayout, GNWC> && is_same_v<WeiLayout, GKXC> &&
                         is_same_v<OutLayout, GNWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instances(op_ptrs);
#endif
                    add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instances(op_ptrs);
#endif
                    add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<WeiDataType, float> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            else if constexpr(is_same_v<InLayout, NWGC> && is_same_v<WeiLayout, GKXC> &&
                              is_same_v<OutLayout, NWGK>)
            {
#ifdef DL_KERNELS
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<WeiDataType, float> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
#endif
            }
        }
        else if constexpr(NumDimSpatial == 2)
        {
            if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, GNHWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f32_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<WeiDataType, float> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            else if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
                              is_same_v<OutLayout, NHWGK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f32_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<WeiDataType, float> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
        }
        else if constexpr(NumDimSpatial == 3)
        {
            if constexpr(is_same_v<InLayout, GNDHWC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, GNDHWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<WeiDataType, float> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            else if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                              is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t> &&
                                  is_same_v<ComputeTypeA, half_t> &&
                                  is_same_v<ComputeTypeB, half_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<WeiDataType, float> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
#ifdef DL_KERNELS
                    add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
#endif
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                                  is_same_v<OutDataType, half_t> &&
                                  is_same_v<ComputeTypeA, bf8_t> && is_same_v<ComputeTypeB, f8_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
                        op_ptrs);
                }
#endif
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
