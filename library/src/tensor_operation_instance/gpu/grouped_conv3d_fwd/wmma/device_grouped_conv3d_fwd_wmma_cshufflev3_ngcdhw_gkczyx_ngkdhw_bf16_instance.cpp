// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_ngcdhw_gkczyx_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_ngchw_instances<3,
                                                                     NGCDHW,
                                                                     GKCZYX,
                                                                     Empty_Tuple,
                                                                     NGKDHW,
                                                                     ConvFwdDefault>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_ngchw_instances<3,
                                                                     NGCDHW,
                                                                     GKCZYX,
                                                                     Empty_Tuple,
                                                                     NGKDHW,
                                                                     ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_ngchw_instances<3,
                                                                     NGCDHW,
                                                                     GKCZYX,
                                                                     Empty_Tuple,
                                                                     NGKDHW,
                                                                     ConvFwd1x1S1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_ngchw_instances<3,
                                                                     NGCDHW,
                                                                     GKCZYX,
                                                                     Empty_Tuple,
                                                                     NGKDHW,
                                                                     ConvFwdOddC>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
