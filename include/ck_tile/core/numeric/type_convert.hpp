// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#if CK_TILE_USE_CUSTOM_DATA_TYPE
#include "ck_tile/core/utility/type_traits.hpp"
#else
#if defined(__gfx125__)
#include "ck_tile/core/numeric/mxfp_scale.hpp"
#endif

#include <type_traits>
#endif

namespace ck_tile {

#if CK_TILE_USE_CUSTOM_DATA_TYPE
template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr remove_cvref_t<Y> type_convert(const X& x)
{
    return static_cast<Y>(x);
}
#else
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

template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr Y scaled_type_convert(X x, float scale);

#if defined(__gfx125__)
// Declare a template function for wave-wise scaled conversion
/* scale is packed 4 form, see details for FP8/BF8, FP4, FP6 */
template <typename Y, typename X, int Scale_sel>
struct pk4scaled_type_convert_impl
{
    CK_TILE_DEVICE static constexpr Y run(X x, Packed4Scale_E8M0 scale);
};

template <typename Y, typename X, int Scale_sel = 0>
CK_TILE_DEVICE constexpr Y pk4scaled_type_convert(X x, Packed4Scale_E8M0 scale)
{
    return pk4scaled_type_convert_impl<Y, X, Scale_sel>::run(x, scale);
}

// pk6scaled_type_convert for FP6 E2M3 and BF6 E3M2
template <typename Y, typename X, int Scale_sel>
struct pk6scaled_type_convert_impl
{
    CK_TILE_DEVICE static constexpr Y run(X x, Packed4Scale_E8M0 scale);
};

template <typename Y, typename X, int Scale_sel = 0>
CK_TILE_DEVICE constexpr Y pk6scaled_type_convert(X x, Packed4Scale_E8M0 scale)
{
    return pk6scaled_type_convert_impl<Y, X, Scale_sel>::run(x, scale);
}
#endif
#endif

} // namespace ck_tile
