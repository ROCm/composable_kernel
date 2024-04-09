// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename F, typename... Fs>
struct composer
{
    composer(F f, Fs... fs) : f_(f), tail_(fs...) {}

    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& arg) const
    {
        return f_(tail_(arg));
    }

    F f_;
    composer<Fs...> tail_;
};

template <typename F>
struct composer<F>
{
    composer(F f) : f_(f) {}

    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& arg) const
    {
        return f_(arg);
    }

    F f_;
};

template <typename... F>
CK_TILE_HOST auto compose(F... f)
{
    return composer<F...>(f...);
}

// TODO: Overload numeric::min() and numeric::max()
struct saturate_f8
{
    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        T y = clamp(x, static_cast<T>(-448), static_cast<T>(448));
        return y;
    }
};

} // namespace ck_tile
