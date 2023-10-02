// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for out[g, n, hi, wi, c] * wei[g, k, y, x, c] = in[g, n, ho, wo, k]
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_bf16_instances<2,
                                                        GNHWK,
                                                        GKYXC,
                                                        Empty_Tuple,
                                                        GNHWC,
                                                        ConvBwdDataDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_bf16_instances<2,
                                                        GNHWK,
                                                        GKYXC,
                                                        Empty_Tuple,
                                                        GNHWC,
                                                        ConvBwdDataFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
