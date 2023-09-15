// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

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
                                                           PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_instances<3,
                                                                   GNDHWC,
                                                                   GKZYXC,
                                                                   GNDHWK,
                                                                   ConvBwdWeightDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_instances<
                                       3,
                                       GNDHWC,
                                       GKZYXC,
                                       GNDHWK,
                                       ConvBwdWeightFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
