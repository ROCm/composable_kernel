// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv1d backward weight
void add_device_conv1d_bwd_weight_xdl_nwc_kxc_nwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<1,
                                                    NWC,
                                                    KXC,
                                                    NWK,
                                                    BF16,
                                                    F32,
                                                    BF16,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

void add_device_conv1d_bwd_weight_xdl_nwc_kxc_nwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<1,
                                                    NWC,
                                                    KXC,
                                                    NWK,
                                                    F16,
                                                    F16,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

void add_device_conv1d_bwd_weight_xdl_nwc_kxc_nwk_f32_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<1,
                                                    NWC,
                                                    KXC,
                                                    NWK,
                                                    F32,
                                                    F32,
                                                    F32,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

// conv2d backward weight
void add_device_conv2d_bwd_weight_xdl_nhwc_kyxc_nhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<2,
                                                    NHWC,
                                                    KYXC,
                                                    NHWK,
                                                    BF16,
                                                    F32,
                                                    BF16,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

void add_device_conv2d_bwd_weight_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<2,
                                                    NHWC,
                                                    KYXC,
                                                    NHWK,
                                                    F16,
                                                    F16,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

void add_device_conv2d_bwd_weight_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<2,
                                                    NHWC,
                                                    KYXC,
                                                    NHWK,
                                                    F32,
                                                    F32,
                                                    F32,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

// conv3d backward weight
void add_device_conv3d_bwd_weight_xdl_ndhwc_kzyxc_ndhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<3,
                                                    NDHWC,
                                                    KZYXC,
                                                    NDHWK,
                                                    BF16,
                                                    F32,
                                                    BF16,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

void add_device_conv3d_bwd_weight_xdl_ndhwc_kzyxc_ndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<3,
                                                    NDHWC,
                                                    KZYXC,
                                                    NDHWK,
                                                    F16,
                                                    F16,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

void add_device_conv3d_bwd_weight_xdl_ndhwc_kzyxc_ndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceConvBwdWeight<3,
                                                    NDHWC,
                                                    KZYXC,
                                                    NDHWK,
                                                    F32,
                                                    F32,
                                                    F32,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>>>& instances);

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceConvBwdWeight<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceConvBwdWeight<NumDimSpatial,
                                         InLayout,
                                         WeiLayout,
                                         OutLayout,
                                         InDataType,
                                         WeiDataType,
                                         OutDataType,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 1 && is_same_v<InLayout, NWC> && is_same_v<WeiLayout, KXC> &&
                     is_same_v<OutLayout, NWK>)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv1d_bwd_weight_xdl_nwc_kxc_nwk_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                              is_same_v<OutDataType, half_t>)
            {
                add_device_conv1d_bwd_weight_xdl_nwc_kxc_nwk_f16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                              is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv1d_bwd_weight_xdl_nwc_kxc_nwk_bf16_f32_bf16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWC> &&
                          is_same_v<WeiLayout, KYXC> && is_same_v<OutLayout, NHWK>)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv2d_bwd_weight_xdl_nhwc_kyxc_nhwk_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                              is_same_v<OutDataType, half_t>)
            {
                add_device_conv2d_bwd_weight_xdl_nhwc_kyxc_nhwk_f16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                              is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv2d_bwd_weight_xdl_nhwc_kyxc_nhwk_bf16_f32_bf16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWC> &&
                          is_same_v<WeiLayout, KZYXC> && is_same_v<OutLayout, NDHWK>)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv3d_bwd_weight_xdl_ndhwc_kzyxc_ndhwk_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                              is_same_v<OutDataType, half_t>)
            {
                add_device_conv3d_bwd_weight_xdl_ndhwc_kzyxc_ndhwk_f16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                              is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv3d_bwd_weight_xdl_ndhwc_kzyxc_ndhwk_bf16_f32_bf16_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
