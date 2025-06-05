// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_i8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv3d_bwd_data_wmma_gndhwk_gkzyxc_gndhwc_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  int8_t,
                                                                  int8_t,
                                                                  Empty_Tuple,
                                                                  int8_t,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
#if CK_BUILD_DEPRECATED
#pragma message "These instances are getting deprecated"
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_wmma_i8_instances<3,
                                                       GNDHWK,
                                                       GKZYXC,
                                                       Empty_Tuple,
                                                       GNDHWC,
                                                       Empty_Tuple,
                                                       PassThrough,
                                                       ConvBwdDataDefault>{});
#else
#pragma message "These instances were deprecated"
    std::ignore = instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
