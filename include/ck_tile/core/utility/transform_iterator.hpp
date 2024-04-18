// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <iterator>
#include <type_traits>

#include "ck_tile/core/utility/iterator.hpp"
#include "ck_tile/core/utility/iterator_range.hpp"

namespace ck_tile {

template <typename UnaryFunction, typename Iterator>
struct transform_iterator
{
    using element_iterator = Iterator;
    using transform_type   = UnaryFunction;
    using difference_type  = iter_difference_t<element_iterator>;
    using invoke_result_type =
        decltype(std::invoke(std::declval<transform_type>(), *std::declval<element_iterator>()));
    using value_type        = std::remove_cv_t<std::remove_reference_t<invoke_result_type>>;
    using reference         = std::conditional_t<std::is_lvalue_reference_v<invoke_result_type>,
                                         invoke_result_type,
                                         const value_type&>;
    using pointer           = std::remove_reference_t<reference>*;
    using iterator_category = typename std::iterator_traits<element_iterator>::iterator_category;

    transform_iterator() = delete;

    explicit constexpr transform_iterator(transform_type transform_, element_iterator next_element_)
        : transform(transform_), next_element(next_element_)
    {
    }

    invoke_result_type operator*() const { return std::invoke(transform, *base()); }

    transform_iterator& operator++()
    {
        ++next_element;
        return *this;
    }

    transform_iterator operator++(int)
    {
        transform_iterator result(*this);
        ++(*this);
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>>
    transform_iterator& operator--()
    {
        --next_element;
        return *this;
    }

    template <bool Cond = std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>>
    transform_iterator operator--(int)
    {
        transform_iterator result(*this);
        ++(*this);
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, transform_iterator&> operator+=(difference_type step)
    {
        std::advance(next_element, step);
        return *this;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, transform_iterator> operator-=(difference_type step)
    {
        return (*this) += (-step);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, reference> operator[](difference_type step) const
    {
        return std::invoke(transform, *std::next(base(), step));
        ;
    }

    element_iterator base() const { return next_element; }

    private:
    friend bool operator==(const transform_iterator& lhs, const transform_iterator& rhs)
    {
        return lhs.base() == rhs.base();
    }

    friend bool operator!=(const transform_iterator& lhs, const transform_iterator& rhs)
    {
        return !(lhs == rhs);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend std::enable_if_t<Cond, transform_iterator> operator+(const transform_iterator& lhs,
                                                                difference_type step)
    {
        transform_iterator result(lhs);
        result += step;
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend std::enable_if_t<Cond, transform_iterator> operator-(const transform_iterator& lhs,
                                                                difference_type step)
    {
        return lhs + (-step);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend difference_type operator-(const transform_iterator& lhs, const transform_iterator& rhs)
    {
        return lhs.base() - rhs.base();
    }

    transform_type transform;
    element_iterator next_element;
};

template <typename UnaryFunction, typename Iterator>
transform_iterator(UnaryFunction, Iterator) -> transform_iterator<UnaryFunction, Iterator>;

template <typename Range, typename UnaryFunction>
auto make_transform_range(Range&& range, UnaryFunction transform)
{
    using std::begin, std::end;

    return iterator_range(transform_iterator(transform, begin(range)),
                          transform_iterator(transform, end(range)));
}

} // namespace ck_tile
