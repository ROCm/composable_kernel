// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BiasNormalizeInInferClamp = ck::tensor_operation::element_wise::BiasNormalizeInInferClamp;

void add_device_grouped_conv3d_fwd_bias_bn_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<
        DeviceGroupedConvFwdMultipleABD<3,
                                        NDHWGC,
                                        GKZYXC,
                                        Tuple<NDHWGK, NDHWGK, NDHWGK, NDHWGK, NDHWGK>,
                                        NDHWGK,
                                        BF16,
                                        BF16,
                                        Tuple<BF16, BF16, BF16, BF16, BF16>,
                                        BF16,
                                        PassThrough,
                                        PassThrough,
                                        BiasNormalizeInInferClamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_wmma_large_tensor_bf16_generic_instances<
                                       3,
                                       NDHWGC,
                                       GKZYXC,
                                       Tuple<NDHWGK, NDHWGK, NDHWGK, NDHWGK, NDHWGK>,
                                       NDHWGK,
                                       ConvFwdDefault,
                                       Tuple<BF16, BF16, BF16, BF16, BF16>,
                                       BiasNormalizeInInferClamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
