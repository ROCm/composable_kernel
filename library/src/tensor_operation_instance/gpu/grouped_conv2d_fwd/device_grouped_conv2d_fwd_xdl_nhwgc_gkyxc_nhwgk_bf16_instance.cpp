// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "device_grouped_conv2d_fwd_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<2,
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
                                                              PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_xdl_bf16_instances<NHWGC,
                                                                                GKYXC,
                                                                                Empty_Tuple,
                                                                                NHWGK,
                                                                                Empty_Tuple,
                                                                                PassThrough,
                                                                                ConvFwdDefault>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_xdl_bf16_instances<NHWGC,
                                                                                GKYXC,
                                                                                Empty_Tuple,
                                                                                NHWGK,
                                                                                Empty_Tuple,
                                                                                PassThrough,
                                                                                ConvFwd1x1P0>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_xdl_bf16_instances<NHWGC,
                                                                                GKYXC,
                                                                                Empty_Tuple,
                                                                                NHWGK,
                                                                                Empty_Tuple,
                                                                                PassThrough,
                                                                                ConvFwd1x1S1P0>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_xdl_bf16_instances<NHWGC,
                                                                                GKYXC,
                                                                                Empty_Tuple,
                                                                                NHWGK,
                                                                                Empty_Tuple,
                                                                                PassThrough,
                                                                                ConvFwdOddC>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
