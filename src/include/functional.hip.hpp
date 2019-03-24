#pragma once
#include "constant_integral.hip.hpp"

template <index_t NLoop>
struct static_loop_n
{
    template <class F>
    __host__ __device__ void operator()(F f) const
    {
        static_assert(NLoop > 1, "out-of-range");

        f(Number<NLoop - 1>{});
        static_loop_n<NLoop - 1>{}(f);
    }
};

template <>
struct static_loop_n<1>
{
    template <class F>
    __host__ __device__ void operator()(F f) const
    {
        f(Number<0>{});
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