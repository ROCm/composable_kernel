// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_outelementop_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_relu.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F8            = ck::f8_t;
using ConvScaleRelu = ck::tensor_operation::element_wise::ConvScaleRelu;

void add_device_grouped_conv3d_fwd_wmma_cshufflev3_convscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_instances(
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
        device_grouped_conv_fwd_wmma_cshufflev3_outelementop_f8_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<>,
                                                                          NDHWGK,
                                                                          ConvFwdDefault,
                                                                          ConvScaleRelu>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
