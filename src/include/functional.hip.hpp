#pragma once
#include "constant_integral.hip.hpp"

template <index_t Iter, index_t Remaining, index_t Increment>
struct static_for_impl
{
    template <class F>
    __host__ __device__ void operator()(F f) const
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
    __host__ __device__ void operator()(F) const
    {
        // do nothing
        return;
    }
};

template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    template <class F>
    __host__ __device__ void operator()(F f) const
    {
        static_assert(NBegin < NEnd, "Wrong! we should have NBegin < NEnd");
        static_assert((NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_for_impl<NBegin, NEnd - NBegin, Increment>{}(f);
    }
};

template <index_t NLoop>
struct static_const_reduce_n
{
    template <class F, class Reduce>
    __host__ __device__ constexpr auto operator()(F f, Reduce r) const
    {
        static_assert(NLoop > 1, "out-of-range");

        constexpr auto a = f(Number<NLoop - 1>{});
        auto b = static_const_reduce_n<NLoop - 1>{}(f, r); // cannot use constexpr here, weird
        return r(a, b);
    }
};

template <>
struct static_const_reduce_n<1>
{
    template <class F, class Reduce>
    __host__ __device__ constexpr auto operator()(F f, Reduce) const
    {
        return f(Number<0>{});
    }
};

#if 0
template<class F>
__host__ __device__ constexpr auto unpacker(F f)
{
    return [=](auto xs_array){ f(xs...); };
}
#endif

namespace mod_conv {
template <class T>
struct multiplies
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a * b; }
};

template <class T>
struct plus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a + b; }
};

} // namespace mod_conv
