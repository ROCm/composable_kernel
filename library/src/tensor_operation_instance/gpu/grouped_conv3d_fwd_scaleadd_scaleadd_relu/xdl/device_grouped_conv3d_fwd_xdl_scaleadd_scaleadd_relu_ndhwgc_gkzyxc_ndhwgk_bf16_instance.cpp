// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_xdl_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<BF16, BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                ScaleAddScaleAddRelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_bf16_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<NDHWGK, NDHWGK>,
                                                                          NDHWGK,
                                                                          ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_bf16_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<NDHWGK, NDHWGK>,
                                                                          NDHWGK,
                                                                          ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_bf16_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<NDHWGK, NDHWGK>,
                                                                          NDHWGK,
                                                                          ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
