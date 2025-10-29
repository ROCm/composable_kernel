// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

namespace ReduceOp {
// y = ReduceOp(y, x);
struct Add
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    template <typename T,
              typename = std::enable_if_t<is_any_of<T, float, double, int32_t, int8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return y + x;
    }

    template <typename T,
              typename = std::enable_if_t<is_any_of<T, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(T& y, T x) const
    {
        float y_ = type_convert<float>(y);
        float x_ = type_convert<float>(x);

        return type_convert<T>(y_ + x_);
    }
};

struct SquareAdd
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    template <typename T,
              typename = std::enable_if_t<is_any_of<T, float, double, int32_t, int8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return y + (x * x);
    }

    template <typename T,
              typename = std::enable_if_t<is_any_of<T, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(T& y, T x) const
    {
        float y_ = type_convert<float>(y);
        float x_ = type_convert<float>(x);
        return type_convert<T>(y_ + (x_ * x_));
    }
};

struct Max
{
    template <
        typename T,
        typename = std::enable_if_t<
            is_any_of<T, float, double, int32_t, int8_t, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return numeric<T>::lowest();
    };

    template <
        typename T,
        typename = std::enable_if_t<
            is_any_of<T, float, double, int32_t, int8_t, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return max(y, x);
    }

    // Overload with changed flag for index tracking
    template <
        typename T,
        typename = std::enable_if_t<
            is_any_of<T, float, double, int32_t, int8_t, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x, bool& changed) const
    {
        T new_max = max(y, x);
        if(x > y)
        {
            changed = true;
        }
        return new_max;
    }
};

struct AbsMax
{
    template <
        typename T,
        typename = std::enable_if_t<
            is_any_of<T, float, double, int32_t, int8_t, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return numeric<T>::zero();
    };

    template <
        typename T,
        typename = std::enable_if_t<
            is_any_of<T, float, double, int32_t, int8_t, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return max(y, abs(x));
    }

    // Overload with changed flag for index tracking
    template <
        typename T,
        typename = std::enable_if_t<
            is_any_of<T, float, double, int32_t, int8_t, half_t, bf16_t, fp8_t, bf8_t>::value>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x, bool& changed) const
    {
        T new_max = max(y, abs(x));
        if(abs(x) > y)
        {
            changed = true;
        }
        return new_max;
    }
};

} // namespace ReduceOp
} // namespace ck_tile
