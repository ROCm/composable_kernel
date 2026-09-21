// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    // These instances are not code-generated for gfx908: they crash the "SI Form memory
    // clauses" pass in the ROCm 7.0.2.x backend, so gfx908 is excluded from this
    // translation unit in library/src/tensor_operation_instance/gpu/CMakeLists.txt.
    // Skip registration there as well, otherwise a heuristic could pick an instance that
    // has no device image and the launch would fail with hipErrorNoBinaryForGpu.
    if(ck::get_device_name() == "gfx908")
    {
        return;
    }

    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_optimized_loads_instances<2,
                                                                       NHWGK,
                                                                       GKYXC,
                                                                       Empty_Tuple,
                                                                       NHWGC,
                                                                       ConvBwdDataDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_data_xdl_f32_optimized_loads_instances<
                                       2,
                                       NHWGK,
                                       GKYXC,
                                       Empty_Tuple,
                                       NHWGC,
                                       ConvBwdDataFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
