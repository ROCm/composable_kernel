// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_dl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_dl_f32_instances<GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              Empty_Tuple,
                                                                              PassThrough,
                                                                              ConvFwdDefault>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_dl_f32_instances<GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              Empty_Tuple,
                                                                              PassThrough,
                                                                              ConvFwd1x1P0>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_dl_f32_instances<GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              Empty_Tuple,
                                                                              PassThrough,
                                                                              ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
