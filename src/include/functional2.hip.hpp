#pragma once
#include "Sequence.hip.hpp"

template <index_t RemainDim>
struct static_ford_impl
{
    // F signature: F(Sequence<...> multi_id)
    // CurrentMultiIndex: Sequence<...>
    // RemainLengths: Sequence<...>
    template <class F, class CurrentMultiIndex, class RemainLengths>
    __host__ __device__ void operator()(F f, CurrentMultiIndex, RemainLengths) const
    {
        static_assert(RemainLengths::GetSize() == RemainDim, "wrong!");
        static_assert(RemainDim > 1, "wrong!");

        constexpr auto next_length = RemainLengths{}.Front();

        static_for<0, next_length, 1>{}([=](auto I) {
            static_ford_impl<RemainDim - 1>{}(
                f, CurrentMultiIndex{}.PushBack(I), RemainLengths{}.PopFront());
        });
    }
};

template <>
struct static_ford_impl<1>
{
    // F signature: F(Sequence<Is...> multi_id)
    // CurrentMultiIndex: Sequence<...>
    // RemainLengths: Sequence<...>
    template <class F, class CurrentMultiIndex, class RemainLengths>
    __host__ __device__ void operator()(F f, CurrentMultiIndex, RemainLengths) const
    {
        static_assert(RemainLengths::GetSize() == 1, "wrong!");

        constexpr index_t last_length = RemainLengths{}.Front();

        static_for<0, last_length, 1>{}([=](auto I) { f(CurrentMultiIndex{}.PushBack(I)); });
    }
};

// Lengths is Sequence<...>
template <class Lengths>
struct static_ford
{
    // F signature: F(Sequence<Is...> multi_id)
    template <class F>
    __host__ __device__ void operator()(F f) const
    {
        constexpr index_t first_length = Lengths{}.Front();

        static_for<0, first_length, 1>{}([=](auto I) {
            static_ford_impl<Lengths::GetSize() - 1>{}(
                f, Sequence<I.Get()>{}, Lengths{}.PopFront());
        });
    }
};

template <index_t RemainDim>
struct ford_impl
{
    // F signature: F(Array<...> multi_id)
    // CurrentMultiIndex: Array<...>
    // RemainLengths: Sequence<...>
    template <class F, class CurrentMultiIndex, class RemainLengths>
    __host__ __device__ void
    operator()(F f, CurrentMultiIndex current_multi_id, RemainLengths) const
    {
        static_assert(RemainLengths::GetSize() == RemainDim, "wrong!");
        static_assert(RemainDim > 1, "wrong!");

        constexpr auto next_length = RemainLengths{}.Front();

        for(index_t i = 0; i < next_length; ++i)
        {
            ford_impl<RemainDim - 1>{}(f, current_multi_id.PushBack(i), RemainLengths{}.PopFront());
        }
    }
};

template <>
struct ford_impl<1>
{
    // F signature: F(Array<...> multi_id)
    // CurrentMultiIndex: Array<...>
    // RemainLengths: Sequence<...>
    template <class F, class CurrentMultiIndex, class RemainLengths>
    __host__ __device__ void
    operator()(F f, CurrentMultiIndex current_multi_id, RemainLengths) const
    {
        static_assert(RemainLengths::GetSize() == 1, "wrong!");

        constexpr index_t last_length = RemainLengths{}.Front();

        for(index_t i = 0; i < last_length; ++i)
        {
            f(current_multi_id.PushBack(i));
        }
    }
};

// Lengths is Sequence<...>
template <class Lengths>
struct ford
{
    // F signature: F(Array<...> multi_id)
    template <class F>
    __host__ __device__ void operator()(F f) const
    {
        constexpr index_t first_length = Lengths{}.Front();

        for(index_t i = 0; i < first_length; ++i)
        {
            ford_impl<Lengths::GetSize() - 1>{}(f, Array<index_t, 1>{i}, Lengths{}.PopFront());
        }
    }
};
