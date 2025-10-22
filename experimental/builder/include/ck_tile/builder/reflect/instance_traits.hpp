// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Compile-time reflection for CK device kernel instances.
//
// - This is the Lowest-level reflection primitive for higher-level semantic abstractions (e.g.,
//   ConvTraits).
// - Extracts raw template parameters (block sizes, data types, layouts, tuning params) from kernel
//   specializations.
// - Provides uniform interface to query kernel configuration without implementation knowledge
// - Other details about the device kernels can be manually added to template specializations.
// - Currently supports:
//   - DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3

#pragma once

#include <array>
#include <string>
#include <sstream>
#include <type_traits>
#include <ck/utility/data_type.hpp>
#include <ck/utility/sequence.hpp>
#include <ck/utility/blkgemmpipe_scheduler.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include "instance_traits_util.hpp"

namespace ck_tile::reflect {

// Primary template for InstanceTraits - extracts compile-time information directly from
// device kernel instances (e.g., DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3)
//
// This is an unspecialized template declaration. Actual specializations for specific
// device kernels are provided in separate header files (e.g.,
// instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp).
template <typename Instance>
struct InstanceTraits;

// Concept-based helper to detect if InstanceTraits<T> is specialized
// (i.e., has the instance_string() member function).
// This can be used for an informative static_assert in the device-op GetInstanceString in case the
// instance_string() template is broken.
template <typename T>
concept HasInstanceTraits = requires {
    { InstanceTraits<T>::instance_string() } -> std::convertible_to<std::string>;
};

// Free function that delegates to InstanceTraits static member function.
// Each InstanceTraits specialization provides its own instance_string() implementation.
template <typename T>
inline std::string instance_string()
{
    return InstanceTraits<T>::instance_string();
}

} // namespace ck_tile::reflect
