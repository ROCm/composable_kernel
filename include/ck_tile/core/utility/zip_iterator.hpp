// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/iterator.hpp"
#include "ck_tile/core/utility/iterator_range.hpp"

namespace ck_tile {

template <typename IterTuple>
struct zip_iterator
{
    using iter_tuple_type = IterTuple;

    private:
    static inline constexpr auto indices_all =
        std::make_index_sequence<std::tuple_size_v<iter_tuple_type>>();

    template <template <typename...> class Outer,
              template <typename...>
              class Inner,
              std::size_t... Is>
    static auto unpack_and_apply_templates(std::index_sequence<Is...>)
        -> Outer<Inner<std::tuple_element_t<Is, iter_tuple_type>>...>;

    template <std::size_t... Is>
    static auto dereference(std::index_sequence<Is...>)
        -> std::tuple<iter_reference_t<std::tuple_element_t<Is, iter_tuple_type>>...>;

    public:
    using value_type = decltype(dereference(indices_all));
    using reference  = const value_type&;
    using pointer    = const value_type*;
    using difference_type =
        decltype(unpack_and_apply_templates<std::common_type_t, iter_difference_t>(indices_all));
    using iterator_category = std::conditional_t<
        decltype(unpack_and_apply_templates<std::conjunction, is_random_access_iterator>(
            indices_all))::value,
        std::random_access_iterator_tag,
        std::conditional_t<
            decltype(unpack_and_apply_templates<std::conjunction, is_bidirectional_iterator>(
                indices_all))::value,
            std::bidirectional_iterator_tag,
            std::input_iterator_tag>>;

    explicit zip_iterator(iter_tuple_type iters_) : iters(iters_) {}

    template <typename OtherIterTuple,
              typename = std::enable_if_t<std::is_convertible_v<OtherIterTuple, iter_tuple_type>>>
    explicit zip_iterator(const zip_iterator<OtherIterTuple>& other) : zip_iterator(other.iters)
    {
    }

    zip_iterator(const zip_iterator&) = default;

    zip_iterator& operator++()
    {
        tuple_inc_impl(iters, indices_all);
        return *this;
    }

    zip_iterator operator++(int)
    {
        zip_iterator result(*this);
        ++(*this);
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>>
    zip_iterator& operator--()
    {
        tuple_dec_impl(iters, indices_all);
        return *this;
    }

    template <bool Cond = std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>>
    zip_iterator operator--(int)
    {
        zip_iterator result(*this);
        --(*this);
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, zip_iterator&> operator+=(difference_type step)
    {
        tuple_inc_impl(iters, step, indices_all);
        return *this;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, zip_iterator&> operator-=(difference_type step)
    {
        return (*this) += (-step);
    }

    auto operator*() const { return get_impl(indices_all); }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    std::enable_if_t<Cond, value_type> operator[](difference_type step) const
    {
        return *((*this) + step);
    }

    private:
    friend bool operator==(const zip_iterator& lhs, const zip_iterator& rhs)
    {
        return tuple_equal_impl(lhs.iters, rhs.iters, indices_all);
    }

    friend bool operator!=(const zip_iterator& lhs, const zip_iterator& rhs)
    {
        return !(lhs == rhs);
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend std::enable_if_t<Cond, zip_iterator> operator+(const zip_iterator& lhs,
                                                          difference_type step)
    {
        zip_iterator result(lhs);
        result += step;
        return result;
    }

    template <bool Cond = std::is_base_of_v<std::random_access_iterator_tag, iterator_category>>
    friend std::enable_if_t<Cond, zip_iterator> operator-(const zip_iterator& lhs,
                                                          difference_type step)
    {
        return lhs + (-step);
    }

    template <typename Tuple, std::size_t... Is>
    static void tuple_inc_impl(Tuple& tuple, std::index_sequence<Is...>)
    {
        (++std::get<Is>(tuple), ...);
    }

    template <typename Tuple, std::size_t... Is>
    static void tuple_dec_impl(Tuple& tuple, std::index_sequence<Is...>)
    {
        (--std::get<Is>(tuple), ...);
    }

    template <typename Tuple, typename Step, std::size_t... Is>
    static void tuple_inc_impl(Tuple& tuple, Step step, std::index_sequence<Is...>)
    {
        ((std::get<Is>(tuple) = std::get<Is>(tuple) + step), ...);
    }

    template <std::size_t... Is>
    auto get_impl(std::index_sequence<Is...>) const
    {
        return std::tie(*std::get<Is>(iters)...);
    }

    template <typename LeftTuple, typename RightTuple, std::size_t First, std::size_t... Rest>
    static bool tuple_equal_impl(const LeftTuple& lhs,
                                 const RightTuple& rhs,
                                 std::index_sequence<First, Rest...>)
    {
        return (std::get<First>(lhs) == std::get<First>(rhs)) &&
               tuple_equal_impl(lhs, rhs, std::index_sequence<Rest...>());
    }

    template <typename LeftTuple, typename RightTuple, std::size_t I>
    static bool
    tuple_equal_impl(const LeftTuple& lhs, const RightTuple& rhs, std::index_sequence<I>)
    {
        return (std::get<I>(lhs) == std::get<I>(rhs));
    }

    private:
    IterTuple iters;
};

template <typename IterTuple>
zip_iterator(IterTuple&&) -> zip_iterator<remove_cvref_t<IterTuple>>;

template <typename... Ranges>
constexpr auto make_zip_range(Ranges&&... ranges)
{
    using std::begin, std::end;

    return iterator_range(zip_iterator(std::make_tuple(begin(ranges)...)),
                          zip_iterator(std::make_tuple(end(ranges)...)));
}

} // namespace ck_tile
