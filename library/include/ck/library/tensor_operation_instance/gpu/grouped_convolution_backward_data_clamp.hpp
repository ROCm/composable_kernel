// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef CK_USE_XDL
#include "grouped_convolution_backward_data_clamp_xdl.inc"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename DLayouts,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename DDataTypes,
          typename OutDataType,
          typename AComputeType,
          typename BComputeType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    DLayouts,
    OutLayout,
    InDataType,
    WeiDataType,
    DDataTypes,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Clamp,
    AComputeType,
    BComputeType>>
{
    using DeviceOp = DeviceGroupedConvBwdDataMultipleD<
        NumDimSpatial,
        InLayout,
        WeiLayout,
        DLayouts,
        OutLayout,
        InDataType,
        WeiDataType,
        DDataTypes,
        OutDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Clamp,
        AComputeType,
        BComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        // layout NHWGC/GKYXC/NHWGK
        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWGK> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, NHWGC>)
        {
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_instances(op_ptrs);
                add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_bf16_16x16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_instances(op_ptrs);
                add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f16_16x16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_instances(op_ptrs);
                add_device_grouped_conv2d_bwd_data_clamp_xdl_nhwgk_gkyxc_nhwgc_f32_16x16_instances(op_ptrs);
            }
#endif
        }
        // layout NDHWGC/GKZYXC/NDHWGK
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> &&
                         is_same_v<OutDataType, ck::bhalf_t> &&
                         is_same_v<AComputeType, ck::bhalf_t> &&
                         is_same_v<BComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_instances(op_ptrs);
                add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_16x16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<AComputeType, half_t> &&
                         is_same_v<BComputeType, half_t>)
            {
                add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f16_instances(op_ptrs);
                add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f16_16x16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float> && is_same_v<AComputeType, float> &&
                         is_same_v<BComputeType, float>)
            {
                add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f32_instances(op_ptrs);
                add_device_grouped_conv3d_bwd_data_clamp_xdl_ndhwgk_gkzyxc_ndhwgc_f32_16x16_instances(op_ptrs);
            }
#endif
        }
#endif // CK_USE_XDL

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
