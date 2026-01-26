// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Implementation of the describe() function template for convolution kernels

#pragma once

#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck_tile/builder/reflect/instance_to_conv_traits.hpp"

namespace ck_tile::reflect {

/// @brief Concept to check if an Instance type has conv traits
template <typename Instance>
concept HasConvTraits = requires {
    { conv::instance_to_conv_traits<Instance>() };
};

/// Factory function to create ConvDescription from a convolution instance type
/// Instance The convolution instance type
/// A ConvDescription object populated with the instance's configuration details
///
/// TODO: Fix ConvDescription to just use the ConvTraits directly.
template <typename Instance>
    requires HasConvTraits<Instance>
conv::ConvDescription describe()
{
    const auto traits = conv::instance_to_conv_traits<Instance>();

    return conv::ConvDescription(
        traits, []<typename T = Instance>() { return reflect::instance_string<T>(); });
}

} // namespace ck_tile::reflect
