// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wavelet_xdl_instance.hpp"

#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, di, hi, wi, g, c] * wei[g, k, z, y, x, c] = out[n, do, ho, wo,
// g, k]
void add_device_grouped_conv3d_fwd_wavelet_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wavelet_xdl_c_shuffle_f32_instances<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    Empty_Tuple,
                                                                    NDHWGK,
                                                                    ConvFwdDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
