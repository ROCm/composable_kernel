// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, G_K>,
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
        device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_bf16_instances<
            3,
            NDHWGC,
            GKZYXC,
            ck::Tuple<NDHWGK, G_K>,
            NDHWGK,
            ConvFwdDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
