#pragma once
#include "integral_constant.hip.hpp"

struct forwarder
{
    template <typename T>
    __host__ __device__ constexpr T operator()(T&& x) const
    {
        return std::forward<T>(x);
    }
};

#if 0
template<class F>
__host__ __device__ constexpr auto unpacker(F f)
{
    return [=](auto xs_array){ f(xs...); };
}
#endif

// Emulate compile time if statement for C++14
//   Get the idea from
//   "https://baptiste-wicht.com/posts/2015/07/simulate-static_if-with-c11c14.html"
// TODO: use if constexpr, when C++17 is supported
template <bool Predicate>
struct static_if
{
};

template <>
struct static_if<true>
{
    using Type = static_if<true>;

    template <class F>
    __host__ __device__ constexpr auto operator()(F f) const
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and maks sure "f" will use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled until being
        //   instantiated here
        f(forwarder{});
        return Type{};
    }

    template <class F>
    __host__ __device__ static constexpr auto Else(F)
    {
        return Type{};
    }
};

template <>
struct static_if<false>
{
    using Type = static_if<false>;

    template <class F>
    __host__ __device__ constexpr auto operator()(F) const
    {
        return Type{};
    }

    template <class F>
    __host__ __device__ static constexpr auto Else(F f)
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and maks sure "f" will use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled until being
        //   instantiated here
        f(forwarder{});
        return Type{};
    }
};
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
        // no work left, just return
        return;
    }
};

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    template <class F>
    __host__ __device__ void operator()(F f) const
    {
        static_assert((NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");

        static_if<(NBegin < NEnd)>{}(
            [&](auto fwd) { static_for_impl<NBegin, NEnd - NBegin, fwd(Increment)>{}(f); });
    }
};

template <index_t NLoop>
struct static_const_reduce_n
{
    // signature of F: F(Number<I>)
    template <class F, class Reduce>
    __host__ __device__ constexpr auto operator()(F f, Reduce r) const
    {
        static_assert(NLoop > 1, "out-of-range");

        constexpr auto a = f(Number<NLoop - 1>{});
        auto b = static_const_reduce_n<NLoop - 1>{}(f, r); // TODO: cannot use constexpr here, weird
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
