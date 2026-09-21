// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv3d_fwd_bias_bn_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<
        DeviceGroupedConvFwdMultipleABD<3,
                                        NDHWGC,
                                        GKZYXC,
                                        Tuple<NDHWGK, NDHWGK, NDHWGK, NDHWGK, NDHWGK>,
                                        NDHWGK,
                                        F16,
                                        F16,
                                        Tuple<F16, F16, F16, F16, F16>,
                                        F16,
                                        PassThrough,
                                        PassThrough,
                                        BiasNormalizeInInferClamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_wmma_cshufflev3_f16_generic_instances<
                                       3,
                                       NDHWGC,
                                       GKZYXC,
                                       Tuple<NDHWGK, NDHWGK, NDHWGK, NDHWGK, NDHWGK>,
                                       NDHWGK,
                                       ConvFwdDefault,
                                       Tuple<F16, F16, F16, F16, F16>,
                                       BiasNormalizeInInferClamp>{});

    // Note: Commented out temporarily , might be used later.

    // add_device_operation_instances(instances,
    //                                device_grouped_conv_fwd_wmma_cshufflev3_f16_generic_instances<
    //                                    3,
    //                                    NDHWGC,
    //                                    GKZYXC,
    //                                    Tuple<NDHWGK, NDHWGK, NDHWGK, NDHWGK, NDHWGK>,
    //                                    NDHWGK,
    //                                    ConvFwd1x1P0,
    //                                    Tuple<F16, F16, F16, F16, F16>,
    //                                    BiasNormalizeInInferClamp>{});

    // add_device_operation_instances(instances,
    //                                device_grouped_conv_fwd_wmma_cshufflev3_f16_generic_instances<
    //                                    3,
    //                                    NDHWGC,
    //                                    GKZYXC,
    //                                    Tuple<NDHWGK, NDHWGK, NDHWGK, NDHWGK, NDHWGK>,
    //                                    NDHWGK,
    //                                    ConvFwd1x1S1P0,
    //                                    Tuple<F16, F16, F16, F16, F16>,
    //                                    BiasNormalizeInInferClamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
