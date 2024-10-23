// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_mem_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for out[n, di, hi, wi, g, c] * wei[g, k, z, y, x, c] = in[n, do, ho, wo,
// g, k]
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_mem_instances<3,
                                                           NDHWGK,
                                                           GKZYXC,
                                                           Empty_Tuple,
                                                           NDHWGC,
                                                           ConvBwdDataDefault,
                                                           Interwave>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_mem_instances<3,
                                                           NDHWGK,
                                                           GKZYXC,
                                                           Empty_Tuple,
                                                           NDHWGC,
                                                           ConvBwdDataFilter1x1Stride1Pad0,
                                                           Interwave>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
