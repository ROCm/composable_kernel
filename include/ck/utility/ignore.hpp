// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_IGNORE_HPP
#define CK_IGNORE_HPP

// https://en.cppreference.com/w/cpp/utility/tuple/ignore

namespace ck {

namespace detail {
struct ignore_t
{
    template <typename T>
    constexpr void operator=(T&&) const noexcept
    {
    }
};
} // namespace detail

inline constexpr detail::ignore_t ignore;

} // namespace ck
#endif
