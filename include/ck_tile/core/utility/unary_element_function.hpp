// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename F, typename... Fs>
struct composes : private composes<F>, private composes<Fs...>
{
    template <typename FirstArg, typename... RestArgs>
    CK_TILE_HOST_DEVICE constexpr explicit composes(FirstArg&& firstArg, RestArgs&&... restArgs)
        : composes<F>(std::forward<FirstArg>(firstArg)),
          composes<Fs...>(std::forward<RestArgs>(restArgs)...)
    {
    }

    template <typename Arg>
    CK_TILE_HOST_DEVICE constexpr decltype(auto) operator()(Arg&& arg) const
    {
        return static_cast<const composes<F>&>(*this)(
            static_cast<const composes<Fs...>&>(*this)(std::forward<Arg>(arg)));
    }
};

template <typename F>
struct composes<F>
{
    static_assert(!std::is_reference_v<F>);

    template <typename Arg, typename = std::enable_if_t<std::is_constructible_v<Arg, F>>>
    CK_TILE_HOST_DEVICE constexpr explicit composes(Arg&& arg) : f_(std::forward<Arg>(arg))
    {
    }

    template <typename Arg,
              typename = std::enable_if_t<std::is_invocable_v<std::add_const_t<F>&, Arg>>>
    CK_TILE_HOST_DEVICE constexpr decltype(auto) operator()(Arg&& arg) const
    {
        return f_(std::forward<Arg>(arg));
    }

    private:
    F f_;
};

template <typename... Ts>
__host__ __device__ composes(Ts&&...)->composes<remove_cvref_t<Ts>...>;

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
