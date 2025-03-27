// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/multi_index.hpp"

namespace ck {

namespace detail {

// RemainLengths: Sequence<...>
// Orders: Sequence<...>
template <class RemainLengths, class Orders>
struct static_ford_impl
{
    __host__ __device__ constexpr static_ford_impl()
    {
        static_assert(RemainLengths::GetSize() > 0, "wrong! should not get here");
    }

    // F signature: F(Sequence<...>)
    // CurrentOrderedId: Sequence<...>
    template <class F, class CurrentOrderedId>
    __host__ __device__ constexpr void operator()(F f, CurrentOrderedId) const
    {
        static_for<0, RemainLengths::Front(), 1>{}([=](auto I) {
            static_ford_impl<decltype(RemainLengths::PopFront()), Orders>{}(
                f, CurrentOrderedId::PushBack(I));
        });
    }
};

template <class Orders>
struct static_ford_impl<Sequence<>, Orders>
{
    // F signature: F(Sequence<...>)
    // OrderedId: Sequence<...>
    template <class F, class OrderedId>
    __host__ __device__ constexpr void operator()(F f, OrderedId) const
    {
        // retrive unordered Id
        f(OrderedId::ReorderGivenOld2New(Orders{}));
    }
};

// RemainLengths: Sequence<...>
// Orders: Sequence<...>
template <class RemainLengths, class Orders>
struct ford_impl
{
    __host__ __device__ constexpr ford_impl()
    {
        static_assert(RemainLengths::GetSize() > 0, "wrong! should not get here");
    }

    // F signature: F(Array<...> multi_id)
    // CurrentOrderdId: Array<...>
    template <class F, class CurrentOrderedId>
    __host__ __device__ constexpr void operator()(F f, CurrentOrderedId current_ordered_id) const
    {
        for(index_t i = 0; i < RemainLengths::Front(); ++i)
        {
            ford_impl<decltype(RemainLengths::PopFront()), Orders>{}(
                f, container_push_back(current_ordered_id, i));
        }
    }
};

template <class Orders>
struct ford_impl<Sequence<>, Orders>
{
    // F signature: F(Array<...> multi_id)
    // CurrentOrderdId: Array<...>
    template <class F, class CurrentOrderedId>
    __host__ __device__ constexpr void operator()(F f, CurrentOrderedId current_ordered_id) const
    {
        // retrive unordered Id
        f(container_reorder_given_old2new(current_ordered_id, Orders{}));
    }
};

} // namespace detail

namespace detail {

// clang-format off

template <int32_t IDim0>
constexpr auto make_cumulative_product(ck::Number<IDim0>)
    -> ck::Sequence<IDim0>;

template <int32_t IDim0, int32_t IDim1>
constexpr auto make_cumulative_product(ck::Number<IDim0>, ck::Number<IDim1>)
    -> ck::Sequence<IDim0, 
                    IDim0 * IDim1>;

template <int32_t IDim0, int32_t IDim1, int32_t IDim2>
constexpr auto make_cumulative_product(ck::Number<IDim0>, ck::Number<IDim1>, ck::Number<IDim2>)
    -> ck::Sequence<IDim0, 
                    IDim0 * IDim1, 
                    IDim0 * IDim1 * IDim2>;

template <int32_t IDim0, int32_t IDim1, int32_t IDim2, int32_t IDim3>
constexpr auto make_cumulative_product(ck::Number<IDim0>, ck::Number<IDim1>, ck::Number<IDim2>, ck::Number<IDim3>)
    -> ck::Sequence<IDim0, 
                    IDim0 * IDim1, 
                    IDim0 * IDim1 * IDim2,
                    IDim0 * IDim1 * IDim2 * IDim3>;

template <int32_t IDim0, int32_t IDim1, int32_t IDim2, int32_t IDim3, int32_t IDim4>
constexpr auto make_cumulative_product(ck::Number<IDim0>, ck::Number<IDim1>, ck::Number<IDim2>, ck::Number<IDim3>, ck::Number<IDim4>)
    -> ck::Sequence<IDim0, 
                    IDim0 * IDim1, 
                    IDim0 * IDim1 * IDim2,
                    IDim0 * IDim1 * IDim2 * IDim3,
                    IDim0 * IDim1 * IDim2 * IDim3 * IDim4>;

template <int32_t IDim0, int32_t IDim1, int32_t IDim2, int32_t IDim3, int32_t IDim4, int32_t IDim5>
constexpr auto make_cumulative_product(ck::Number<IDim0>, ck::Number<IDim1>, ck::Number<IDim2>, ck::Number<IDim3>, ck::Number<IDim4>, ck::Number<IDim5>)
    -> ck::Sequence<IDim0, 
                    IDim0 * IDim1, 
                    IDim0 * IDim1 * IDim2,
                    IDim0 * IDim1 * IDim2 * IDim3,
                    IDim0 * IDim1 * IDim2 * IDim3 * IDim4,
                    IDim0 * IDim1 * IDim2 * IDim3 * IDim4 * IDim5>;
// clang-format on
template <int32_t... Dims>
struct convert_flat_to_multi_index
{
    using SDim                 = ck::Sequence<Dims...>;
    static constexpr auto Prod = (Dims * ...);
    using TCumProd             = decltype(make_cumulative_product(ck::Number<Dims>{}...));

    template<int32_t flat_idx, int32_t... values>
    static constexpr auto infer_size_from(ck::Sequence<values...>, ck::Number<flat_idx>)
      -> ck::Sequence<values * flat_idx / Prod ...>;

    template <ck::index_t flat_idx>
    using type = decltype(decltype(infer_size_from(TCumProd{}, ck::Number<flat_idx>{})){} % SDim{});
};

template <typename T, T... Is>
struct applier
{
    // F: code block parameterized by compile-time constant
    // IndexTransform: metafunction from int32_t to code block argument type
    // Result: side effect of executing the code block for each element in (Is...)
    template <typename F, template <int32_t> typename IndexTransform = ck::Number>
    __host__ __device__ constexpr void operator()(F f) const
    {
        static_assert(sizeof...(Is) <= 3136, "tweak -fbracket-depth");
        (f(IndexTransform<Is>{}), ...);
    }
};

template <int32_t Size>
using make_applier = __make_integer_seq<applier, ck::index_t, Size>;

} // namespace detail

// Lengths is Sequence<...>, it is the length of each dimension for
// N-dimensional loop
template <typename T>
struct static_ford;

template <template <int32_t...> typename T, int32_t... Dims>
struct static_ford<T<Dims...>> : detail::make_applier<(Dims * ...)>
{
    // `base` is the same as `applier<index_t, 0, ..., product of Dims>`
    using base = detail::make_applier<(Dims * ...)>;

    template <ck::index_t I>
    using convert_t = typename detail::convert_flat_to_multi_index<Dims...>::template type<I>;

    template <typename F, template <auto> typename IndexTransform = convert_t>
    __host__ __device__ constexpr void operator()(F f) const
    {
        base::template operator()<F, IndexTransform>(f);
    }
};

template <template <int32_t...> typename T, int32_t... Dims>
struct static_ford<const T<Dims...>> : static_ford<T<Dims...>>
{
    using static_ford<T<Dims...>>::operator();
};

// Lengths is Sequence<...>, it is the length of each dimension for
// N-dimensional loop
// Orders is Sequence<...>, it is the order of dimension in which ford will loop
// over each
// dimension
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::GetSize(), 1>::type>
struct ford
{
    __host__ __device__ constexpr ford()
    {
        static_assert(Lengths::GetSize() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::GetSize() == Orders::GetSize(), "wrong! inconsistent size");
    }

    // F signature: F(Array<...> multi_id)
    // multi_id is the unordered multi-index
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        constexpr auto ordered_lengths = Lengths::ReorderGivenNew2Old(Orders{});

        for(index_t i = 0; i < ordered_lengths.Front(); ++i)
        {
            detail::ford_impl<decltype(ordered_lengths.PopFront()), Orders>{}(f,
                                                                              make_multi_index(i));
        }
    }
};

} // namespace ck
