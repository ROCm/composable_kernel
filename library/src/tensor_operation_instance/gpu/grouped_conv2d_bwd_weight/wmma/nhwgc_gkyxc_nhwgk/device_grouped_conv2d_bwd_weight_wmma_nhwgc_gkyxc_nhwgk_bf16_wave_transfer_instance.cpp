// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_wave_transfer_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_bf16_wave_transfer_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<2,
                                                                    NHWGC,
                                                                    GKYXC,
                                                                    NHWGK,
                                                                    Tuple<>,
                                                                    BF16,
                                                                    BF16,
                                                                    BF16,
                                                                    Tuple<>,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_bf16_wave_transfer_instances<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        NHWGK,
                                                                        ConvBwdWeightFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
