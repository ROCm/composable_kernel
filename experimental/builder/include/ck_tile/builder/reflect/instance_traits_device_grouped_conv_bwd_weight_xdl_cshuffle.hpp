// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The full InstanceTraits specialization is in:
//   ck_tile/builder/reflect/reflect_device_grouped_conv_bwd_weight_xdl_cshuffle.inc
//
// CRITICAL MAINTENANCE NOTE:
// Keep the specialization in the .inc file strictly in sync with:
// ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp.
// "In sync" means that the template parameter order, names, and types MUST EXACTLY MATCH. If they
// diverge, you may encounter compilation errors, subtle template instantiation mismatches, or
// silent runtime bugs that are difficult to diagnose. Always update both files together and review
// changes carefully.

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"

namespace ck_tile::reflect {

struct DeviceGroupedConvBwdWeight_Xdl_CShuffle_Tag
{
};

} // namespace ck_tile::reflect
