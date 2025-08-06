// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

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
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return y + x;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(T& y, T x) const
    {
        float y_ = type_convert<float>(y);
        float x_ = type_convert<float>(x);

        return type_convert<T>(y_ + x_);
    }

    static constexpr bool requires_special_combine = false;
};

struct SquareAdd
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return y + (x * x);
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(T& y, T x) const
    {
        float y_ = type_convert<float>(y);
        float x_ = type_convert<float>(x);
        return type_convert<T>(y_ + (x_ * x_));
    }

    // For combining partial results
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T combine_partial_results(const T& partial1,
                                                            const T& partial2) const
    {
        return partial1 + partial2; // Just add the partial sums, don't square again
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE constexpr T combine_partial_results(T& partial1, T& partial2) const
    {
        float partial1_ = type_convert<float>(partial1);
        float partial2_ = type_convert<float>(partial2);
        return type_convert<T>(partial1_ + partial2_);
    }

    static constexpr bool requires_special_combine = true;
};

struct Max
{
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t> ||
                                          std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return numeric<T>::min();
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t> ||
                                          std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return max(y, x);
    }

    static constexpr bool requires_special_combine = false;
};

struct AbsMax
{
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t> ||
                                          std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return numeric<T>::min();
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t> ||
                                          std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
                                          std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return max(y, abs(x));
    }

    static constexpr bool requires_special_combine = false;
};

} // namespace ReduceOp
} // namespace ck_tile
