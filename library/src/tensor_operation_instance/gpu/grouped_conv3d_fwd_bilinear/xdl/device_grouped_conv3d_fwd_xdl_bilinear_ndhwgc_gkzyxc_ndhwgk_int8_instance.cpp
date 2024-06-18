// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_bilinear_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                ck::Tuple<int8_t>,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Bilinear>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_bilinear_int8_instances<3,
                                                            NDHWGC,
                                                            GKZYXC,
                                                            Tuple<NDHWGK>,
                                                            NDHWGK,
                                                            ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_bilinear_int8_instances<3,
                                                            NDHWGC,
                                                            GKZYXC,
                                                            Tuple<NDHWGK>,
                                                            NDHWGK,
                                                            ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_bilinear_int8_instances<3,
                                                            NDHWGC,
                                                            GKZYXC,
                                                            Tuple<NDHWGK>,
                                                            NDHWGK,
                                                            ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
