// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instances<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Empty_Tuple,
                                                                        NHWGK,
                                                                        ConvFwdDefault,
                                                                        GemmMNKPadding,
                                                                        F16>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instances<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Empty_Tuple,
                                                                        NHWGK,
                                                                        ConvFwd1x1S1P0,
                                                                        GemmDefault,
                                                                        F16>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
