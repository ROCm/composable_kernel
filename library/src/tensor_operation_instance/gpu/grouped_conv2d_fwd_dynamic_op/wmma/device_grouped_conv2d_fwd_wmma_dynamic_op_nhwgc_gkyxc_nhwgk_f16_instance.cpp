// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_dynamic_op_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_fwd_wmma_cshufflev3_dynamic_op_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                ck::Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                DynamicUnaryOp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_dynamic_op_f16_instances<2,
                                                                         NHWGC,
                                                                         GKYXC,
                                                                         Tuple<>,
                                                                         NHWGK,
                                                                         ConvFwdDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
