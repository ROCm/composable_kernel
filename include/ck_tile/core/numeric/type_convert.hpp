// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <stdint.h>
#include <tuple>
#include <type_traits>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"

namespace ck_tile {

template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr remove_cvref_t<Y> type_convert(const X& x)
{
    return static_cast<Y>(x);
}

#if 0
// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);
    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using non_const_y = std::remove_const_t<Y>;
    using non_const_x = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<non_const_y, non_const_x>(x));
}

#define CK_TILE_TYPE_CONVERT(dtype_, stype_)                                           \
    template <>                                                                        \
    inline CK_TILE_HOST_DEVICE constexpr dtype_ type_convert<dtype_, stype_>(stype_ x) \
    {                                                                                  \
        return stype_##_to_##dtype_(x);                                                \
    }

CK_TILE_TYPE_CONVERT(float, fp16_t)
CK_TILE_TYPE_CONVERT(float, bf16_t)
CK_TILE_TYPE_CONVERT(float, fp8_t)
CK_TILE_TYPE_CONVERT(float, bf8_t)

CK_TILE_TYPE_CONVERT(fp16_t, float)
CK_TILE_TYPE_CONVERT(bf16_t, float)
CK_TILE_TYPE_CONVERT(fp8_t, float)
CK_TILE_TYPE_CONVERT(bf8_t, float)

#undef CK_TILE_TYPE_CONVERT
#endif

} // namespace ck_tile
