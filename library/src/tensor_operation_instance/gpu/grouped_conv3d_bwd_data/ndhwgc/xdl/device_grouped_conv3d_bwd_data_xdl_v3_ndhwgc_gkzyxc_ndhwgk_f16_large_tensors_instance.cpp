// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_v3_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_bwd_data_xdl_v3_ndhwgk_gkzyxc_ndhwgc_f16_large_tensors_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
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
        device_grouped_conv_bwd_data_xdl_v3_f16_large_tensor_instances<3,
                                                                       NDHWGK,
                                                                       GKZYXC,
                                                                       Empty_Tuple,
                                                                       NDHWGC,
                                                                       ConvBwdDataDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
