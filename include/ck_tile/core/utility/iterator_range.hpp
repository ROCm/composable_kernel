// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "ck_tile/core/utility/iterator.hpp"

namespace ck_tile {

template <typename ForwardIterator, typename Sentinel = ForwardIterator>
struct iterator_range
{
    using iterator        = ForwardIterator;
    using const_iterator  = ForwardIterator;
    using value_type      = iter_value_t<iterator>;
    using reference       = iter_reference_t<iterator>;
    using sentinel        = Sentinel;
    using difference_type = iter_difference_t<iterator>;
    using size_type       = std::make_unsigned_t<difference_type>;

    constexpr iterator_range(ForwardIterator first_, Sentinel last_) : first(first_), last(last_) {}
    iterator_range(const iterator_range&) = default;
    iterator_range(iterator_range&&)      = default;

    template <typename ForwardRange>
    explicit constexpr iterator_range(ForwardRange& range) : first(range.begin()), last(range.end())
    {
    }

    iterator_range& operator=(const iterator_range&) = default;
    iterator_range& operator=(iterator_range&&) = default;

    iterator begin() const { return first; }

    sentinel end() const { return last; }

    reference front() const { return *begin(); }

    template <bool Cond = is_bidirectional_iterator_v<sentinel>>
    reference back() const
    {
        return *std::prev(end());
    }

    template <bool Cond = is_random_access_iterator_v<iterator>>
    reference operator[](size_type pos) const
    {
        return *std::next(begin(), static_cast<difference_type>(pos));
    }

    template <typename = std::void_t<decltype(std::distance(std::declval<iterator>(),
                                                            std::declval<sentinel>()))>>
    size_type size() const
    {
        return static_cast<size_type>(std::distance(begin(), end()));
    }

    private:
    iterator first;
    sentinel last;
};

} // namespace ck_tile
