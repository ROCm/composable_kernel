// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {
/**********************************************
 * constexpr helper functions for optional parameters
 **********************************************/

template <auto Sig>
concept ProvidesElementwiseOperation = requires { Sig.elementwiseOperation; };

template <auto Sig>
concept ProvidesConvolutionDirection = requires { Sig.direction; };

template <auto Sig>
constexpr auto get_elementwise_operation()
{
    if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

template <auto Sig>
constexpr auto get_conv_direction()
{
    if constexpr(ProvidesConvolutionDirection<Sig>)
    {
        return Sig.direction;
    }
    else
    {
        return ConvDirection::FORWARD;
    }
}
} // namespace ck_tile::builder
