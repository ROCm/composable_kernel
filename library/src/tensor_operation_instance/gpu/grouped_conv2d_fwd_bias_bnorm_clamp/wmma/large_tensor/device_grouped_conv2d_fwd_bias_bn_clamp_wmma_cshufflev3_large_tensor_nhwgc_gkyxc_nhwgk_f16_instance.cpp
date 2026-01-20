// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BiasNormalizeInInferClamp = ck::tensor_operation::element_wise::BiasNormalizeInInferClamp;

void add_device_grouped_conv2d_fwd_bias_bn_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                        NHWGC,
                                                        GKYXC,
                                                        Tuple<NHWGK, NHWGK, NHWGK, NHWGK, NHWGK>,
                                                        NHWGK,
                                                        F16,
                                                        F16,
                                                        Tuple<F16, F16, F16, F16, F16>,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        BiasNormalizeInInferClamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_wmma_large_tensor_f16_generic_instances<
                                       2,
                                       NHWGC,
                                       GKYXC,
                                       Tuple<NHWGK, NHWGK, NHWGK, NHWGK, NHWGK>,
                                       NHWGK,
                                       ConvFwdDefault,
                                       Tuple<F16, F16, F16, F16, F16>,
                                       BiasNormalizeInInferClamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
