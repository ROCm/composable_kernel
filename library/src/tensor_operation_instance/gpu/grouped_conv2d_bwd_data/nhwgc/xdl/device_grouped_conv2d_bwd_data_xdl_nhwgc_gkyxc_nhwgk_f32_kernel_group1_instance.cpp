// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_kernel_groups.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

namespace detail {

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_kernel_group1_instances(
    F32KernelSlots& slots)
{
    fill_f32_kernel_slots<GenericF32<ConvBwdDataDefault>, 0>(slots, GenericGroup1{});
    fill_f32_kernel_slots<GenericF32<ConvBwdDataFilter1x1Stride1Pad0>, GenericF32Count>(
        slots, GenericGroup1{});
    fill_f32_kernel_slots<NoshuffleF32<ConvBwdDataDefault>, 2 * GenericF32Count>(slots,
                                                                                 NoshuffleGroup1{});
    fill_f32_kernel_slots<NoshuffleF32<ConvBwdDataFilter1x1Stride1Pad0>,
                          2 * GenericF32Count + NoshuffleF32Count>(slots, NoshuffleGroup1{});
}

} // namespace detail

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
