// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_kernel_groups.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_instances(
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
    detail::F32KernelSlots slots;
    detail::fill_f32_kernel_slots<detail::GenericF32<ConvBwdDataDefault>, 0>(
        slots, detail::GenericGroup0{});
    detail::fill_f32_kernel_slots<detail::GenericF32<ConvBwdDataFilter1x1Stride1Pad0>,
                                  detail::GenericF32Count>(slots, detail::GenericGroup0{});
    detail::fill_f32_kernel_slots<detail::NoshuffleF32<ConvBwdDataDefault>,
                                  2 * detail::GenericF32Count>(slots, detail::NoshuffleGroup0{});
    detail::fill_f32_kernel_slots<detail::NoshuffleF32<ConvBwdDataFilter1x1Stride1Pad0>,
                                  2 * detail::GenericF32Count + detail::NoshuffleF32Count>(
        slots, detail::NoshuffleGroup0{});
    detail::add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_kernel_group1_instances(
        slots);

    // Restore the original family, specialization, and tuple order.
    ck::static_for<0, std::tuple_size_v<detail::F32KernelSlots>, 1>{}(
        [&](auto i) { instances.push_back(std::move(slots[i.value])); });
    detail::add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_nongrouped_match_instances(
        instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
