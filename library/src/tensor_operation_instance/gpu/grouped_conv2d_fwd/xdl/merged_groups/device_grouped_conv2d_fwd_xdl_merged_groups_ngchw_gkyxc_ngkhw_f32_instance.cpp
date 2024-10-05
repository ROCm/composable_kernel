// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkyxc_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_merged_groups_f32_instances<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                ConvFwdDefault>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_merged_groups_f32_instances<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                ConvFwd3x3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
