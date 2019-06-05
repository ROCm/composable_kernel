#pragma once
#include "functional.hip.hpp"
#include "Sequence.hip.hpp"

#if 0
template <index_t Iter, index_t Remaining, index_t Increment>
struct static_for_impl
{
    template <class F>
    constexpr __host__ __device__ void operator()(F f) const
    {
        static_assert(Remaining % Increment == 0, "wrong! Remaining % Increment != 0");
        static_assert(Increment <= Remaining, "will go out-of-range");

        f(Number<Iter>{});
        static_for_impl<Iter + Increment, Remaining - Increment, Increment>{}(f);
    }
};

template <index_t Iter, index_t Increment>
struct static_for_impl<Iter, 0, Increment>
{
    template <class F>
    constexpr __host__ __device__ void operator()(F) const
    {
        // no work left, just return
        return;
    }
};

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    template <class F>
    constexpr __host__ __device__ void operator()(F f) const
    {
        static_assert(NBegin <= NEnd, "wrongs! should have NBegin <= NEnd");

        static_assert((NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");

#if 0
        static_if<(NBegin < NEnd)>{}(
            [&](auto fwd) { static_for_impl<NBegin, NEnd - NBegin, fwd(Increment)>{}(f); });
#else
        static_for_impl<NBegin, NEnd - NBegin, Increment>{}(f);
#endif
    }
};
#else
template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<Sequence<Is...>>
{
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        swallow{(f(Number<Is>{}), 0)...};
    }
};

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        static_assert(NBegin <= NEnd, "wrongs! should have NBegin <= NEnd");

        static_assert((NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");

        static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::SeqType>{}(f);
    }
};
#endif
