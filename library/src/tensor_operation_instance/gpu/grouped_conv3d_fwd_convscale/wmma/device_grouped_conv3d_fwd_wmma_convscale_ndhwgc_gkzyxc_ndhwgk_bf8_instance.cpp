// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_outelementop_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using ConvScale = ck::tensor_operation::element_wise::ConvScale;

void add_device_grouped_conv3d_fwd_wmma_cshufflev3_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF8,
                                                                BF8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvScale,
                                                                BF8,
                                                                BF8>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_outelementop_bf8_instances<3,
                                                                           NDHWGC,
                                                                           GKZYXC,
                                                                           ck::Tuple<>,
                                                                           NDHWGK,
                                                                           ConvFwdDefault,
                                                                           ConvScale>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
