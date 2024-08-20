// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_outelementop_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using ConvScaleRelu = ck::tensor_operation::element_wise::ConvScaleRelu;

void add_device_grouped_conv3d_fwd_xdl_convscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvScaleRelu,
                                                                F8,
                                                                F8>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              ck::Tuple<>,
                                                              NDHWGK,
                                                              ConvFwdDefault,
                                                              ConvScaleRelu>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              ck::Tuple<>,
                                                              NDHWGK,
                                                              ConvFwd1x1P0,
                                                              ConvScaleRelu>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              ck::Tuple<>,
                                                              NDHWGK,
                                                              ConvFwd1x1S1P0,
                                                              ConvScaleRelu>{});
}

namespace ew            = ck::tensor_operation::element_wise;
using CombConvScaleRelu = ew::UnaryCombinedOp<ew::Scale, ew::Scale, ew::Relu>;

void add_device_grouped_conv3d_fwd_xdl_combconvscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_f8_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                ck::Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                CombConvScaleRelu,
                                                                F8,
                                                                F8>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_f8_f32_instances<3,
                                                                     NDHWGC,
                                                                     GKZYXC,
                                                                     ck::Tuple<>,
                                                                     NDHWGK,
                                                                     ConvFwdDefault,
                                                                     CombConvScaleRelu>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_f8_f32_instances<3,
                                                                     NDHWGC,
                                                                     GKZYXC,
                                                                     ck::Tuple<>,
                                                                     NDHWGK,
                                                                     ConvFwd1x1P0,
                                                                     CombConvScaleRelu>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_f8_f32_instances<3,
                                                                     NDHWGC,
                                                                     GKZYXC,
                                                                     ck::Tuple<>,
                                                                     NDHWGK,
                                                                     ConvFwd1x1S1P0,
                                                                     CombConvScaleRelu>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
