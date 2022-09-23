// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <type_traits>

#include "ck/library/utility/ranges.hpp"

namespace ck {
namespace utils {

namespace detail {

template <typename Range, std::size_t Size>
class to_array_result;

template <typename FromT, std::size_t Size>
class to_array_result<FromT (&)[Size], Size> final
{
    public:
    explicit constexpr to_array_result(FromT (&source)[Size]) noexcept : source_(source) {}

    template <typename T>
    operator std::array<T, Size>() const
    {
        static_assert(std::is_convertible_v<FromT, T>);

        return copy_as_array<T>(source_, std::make_index_sequence<Size>{});
    }

    private:
    template <typename T, std::size_t... Indices>
    static std::array<T, Size> copy_as_array(FromT (&array)[Size], std::index_sequence<Indices...>)
    {
        return std::array<T, Size>{static_cast<T>(array[Indices])...};
    }

    private:
    FromT (&source_)[Size];
};

template <typename FromT, std::size_t Size>
class to_array_result<FromT(&&)[Size], Size> final
{
    public:
    explicit constexpr to_array_result(FromT(&&source)[Size]) noexcept : source_(std::move(source))
    {
    }

    template <typename T>
    operator std::array<T, Size>() &&
    {
        static_assert(std::is_convertible_v<FromT, T>);

        return move_as_array<T>(std::move(source_), std::make_index_sequence<Size>{});
    }

    private:
    template <typename T, std::size_t... Indices>
    static std::array<T, Size> move_as_array(FromT(&&array)[Size], std::index_sequence<Indices...>)
    {
        return std::array<T, Size>{static_cast<T>(std::move(array[Indices]))...};
    }

    private:
    FromT(&&source_)[Size];
};

template <typename Range>
class to_array_result<Range, std::numeric_limits<std::size_t>::max()> final
{
    public:
    explicit constexpr to_array_result(const Range& source) noexcept : source_(source) {}

    template <typename T, std::size_t Size>
    operator std::array<T, Size>() const
    {
        static_assert(std::is_convertible_v<ranges::range_value_t<Range>, T>);

        std::array<T, Size> destination;

        std::copy_n(std::begin(source_),
                    std::min<std::size_t>(Size, std::size(source_)),
                    std::begin(destination));

        return destination;
    }

    private:
    const Range& source_;
};

struct empty_array_result final
{
    template <typename T>
    operator std::array<T, 0>() const
    {
        return std::array<T, 0>{};
    }
};
} // namespace detail

template <typename T, std::size_t N>
inline constexpr auto to_array(T (&array)[N]) -> detail::to_array_result<decltype(array), N>
{
    return detail::to_array_result<decltype(array), N>{array};
}

template <typename T, std::size_t N>
inline constexpr auto to_array(T(&&array)[N]) -> detail::to_array_result<decltype(array), N>
{
    return detail::to_array_result<decltype(array), N>{std::move(array)};
}

template <typename Range>
inline constexpr auto to_array(const Range& range) noexcept
    -> detail::to_array_result<ck::remove_cvref_t<Range>, std::numeric_limits<std::size_t>::max()>
{
    return detail::to_array_result<ck::remove_cvref_t<Range>,
                                   std::numeric_limits<std::size_t>::max()>{range};
}

inline constexpr auto empty_array() noexcept { return detail::empty_array_result{}; }

} // namespace utils
} // namespace ck
