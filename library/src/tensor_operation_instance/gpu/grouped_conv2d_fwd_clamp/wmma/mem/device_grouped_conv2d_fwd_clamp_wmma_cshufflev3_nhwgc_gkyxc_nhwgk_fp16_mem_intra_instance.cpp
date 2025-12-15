// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_mem_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_f16_mem_instances<2,
                                                                  NHWGC,
                                                                  GKYXC,
                                                                  Tuple<>,
                                                                  NHWGK,
                                                                  ConvFwdDefault,
                                                                  Intrawave,
                                                                  Tuple<>,
                                                                  Clamp>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_f16_mem_instances<2,
                                                                  NHWGC,
                                                                  GKYXC,
                                                                  Tuple<>,
                                                                  NHWGK,
                                                                  ConvFwd1x1P0,
                                                                  Intrawave,
                                                                  Tuple<>,
                                                                  Clamp>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_f16_mem_instances<2,
                                                                  NHWGC,
                                                                  GKYXC,
                                                                  Tuple<>,
                                                                  NHWGK,
                                                                  ConvFwd1x1S1P0,
                                                                  Intrawave,
                                                                  Tuple<>,
                                                                  Clamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
