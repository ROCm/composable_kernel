// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

namespace ck {
namespace utils {
namespace detail {

struct empty_array_result final
{
    template <typename T>
    operator std::array<T, 0>() const
    {
        return std::array<T, 0>{};
    }
};
} // namespace detail

inline constexpr auto empty_array() noexcept { return detail::empty_array_result{}; }

} // namespace utils
} // namespace ck
