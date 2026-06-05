// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    NDHWGK,
                                                                    Tuple<GKZYXC>,
                                                                    BF16,
                                                                    BF16,
                                                                    BF16,
                                                                    Tuple<BF16>,
                                                                    PassThrough,
                                                                    Bilinear,
                                                                    PassThrough>>>& instances)
{
    // Default bwd weight bilinear
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_bilinear_instances<
            3,
            NDHWGC,
            GKZYXC,
            NDHWGK,
            ConvBwdWeightDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
