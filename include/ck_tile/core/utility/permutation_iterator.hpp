// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "ck_tile/core/utility/iterator.hpp"
#include "ck_tile/core/utility/iterator_range.hpp"

namespace ck_tile {

template <typename ElementIterator, typename IndexIterator>
struct permutation_iterator
{
    static_assert(is_random_access_iterator_v<ElementIterator>);

    using element_iterator = ElementIterator;
    using reference        = iter_reference_t<element_iterator>;
    using index_iterator   = IndexIterator;
    using difference_type  = iter_difference_t<index_iterator>;

    permutation_iterator() = delete;

    explicit constexpr permutation_iterator(element_iterator next_element_,
                                            index_iterator next_index_)
        : next_element(next_element_), next_index(next_index_)
    {
    }

    reference operator*() const { return *std::next(base(), *index()); }

    permutation_iterator& operator++()
    {
        ++next_index;
        return *this;
    }

    permutation_iterator operator++(int)
    {
        permutation_iterator result(*this);
        ++(*this);
        return result;
    }

    template <bool Cond = is_random_access_iterator_v<index_iterator>>
    std::enable_if_t<Cond, permutation_iterator> operator+(difference_type step) const
    {
        permutation_iterator result(*this);
        result += step;
        return result;
    }

    template <bool Cond = is_random_access_iterator_v<index_iterator>>
    std::enable_if_t<Cond, permutation_iterator&> operator+=(difference_type step)
    {
        std::advance(next_index, step);
        return *this;
    }

    element_iterator base() const { return next_element; }

    private:
    index_iterator index() const { return next_index; }

    friend bool operator==(const permutation_iterator& lhs, const permutation_iterator& rhs)
    {
        return lhs.base() == rhs.base() && lhs.index() == rhs.index();
    }

    friend bool operator!=(const permutation_iterator& lhs, const permutation_iterator& rhs)
    {
        return !(lhs == rhs);
    }

    element_iterator next_element;
    index_iterator next_index;
};

template <typename ElementIterator, typename IndexIterator>
permutation_iterator(ElementIterator, IndexIterator)
    -> permutation_iterator<ElementIterator, IndexIterator>;

} // namespace ck_tile
