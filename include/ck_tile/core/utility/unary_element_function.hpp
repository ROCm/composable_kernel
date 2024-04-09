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

template <typename To>
struct saturates
{
    template <typename From>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const From& from) const
        -> std::enable_if_t<std::is_arithmetic_v<From>, From>
    {
        if constexpr(std::is_floating_point_v<To> || std::is_same_v<To, half_t> ||
                     std::is_same_v<To, bfloat16_t> || std::is_same_v<To, fp8_t> ||
                     std::is_same_v<To, bf8_t>)
        {
            return clamp(from,
                         type_convert<From>(numeric<To>::lowest()),
                         type_convert<From>(numeric<To>::max()));
        }
        else
        {
            return clamp(from,
                         type_convert<From>(numeric<To>::min()),
                         type_convert<From>(numeric<To>::max()));
        }
    }
};

} // namespace ck_tile
