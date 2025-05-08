// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_bias_relu_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddRelu>>>& instances)
{
    if(ck::get_device_name() == "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Empty_Tuple,
                                                                        NHWGK,
                                                                        ConvFwdDefault,
                                                                        AddRelu>{});

        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Empty_Tuple,
                                                                        NHWGK,
                                                                        ConvFwd3x3,
                                                                        AddRelu>{});
    }
    else
    {
        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Empty_Tuple,
                                                                     NHWGK,
                                                                     ConvFwdDefault,
                                                                     AddRelu>{});

        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Empty_Tuple,
                                                                     NHWGK,
                                                                     ConvFwd3x3,
                                                                     AddRelu>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
