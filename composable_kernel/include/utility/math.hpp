#ifndef CK_MATH_HPP
#define CK_MATH_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"

namespace ck {
namespace math {

template <class T, T s>
struct scales
{
    __host__ __device__ constexpr T operator()(T a) const { return s * a; }
};

template <class T>
struct plus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a + b; }
};

template <class T>
struct minus
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a - b; }
};

template <class T>
struct multiplies
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a * b; }
};

struct multiplies_v2
{
    template <typename A, typename B>
    __host__ __device__ constexpr auto operator()(const A& a, const B& b) const
    {
        return a * b;
    }
};

template <class T>
struct maximize
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
};

template <class T>
struct minimize
{
    __host__ __device__ constexpr T operator()(T a, T b) const { return a <= b ? a : b; }
};

template <class T>
struct integer_divide_ceiler
{
    __host__ __device__ constexpr T operator()(T a, T b) const
    {
        static_assert(is_same<T, index_t>{} || is_same<T, int>{}, "wrong type");

        return (a + b - 1) / b;
    }
};

template <class X, class Y>
__host__ __device__ constexpr auto integer_divide_floor(X x, Y y)
{
    return x / y;
}

template <class X, class Y>
__host__ __device__ constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - Number<1>{}) / y;
}

template <class X, class Y>
__host__ __device__ constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

template <class T>
__host__ __device__ constexpr T max(T x)
{
    return x;
}

template <class T, class... Ts>
__host__ __device__ constexpr T max(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = max(xs...);

    static_assert(is_same<decltype(y), T>{}, "not the same type");

    return x > y ? x : y;
}

template <class T>
__host__ __device__ constexpr T min(T x)
{
    return x;
}

template <class T, class... Ts>
__host__ __device__ constexpr T min(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = min(xs...);

    static_assert(is_same<decltype(y), T>{}, "not the same type");

    return x < y ? x : y;
}

// greatest common divisor, aka highest common factor
__host__ __device__ constexpr index_t gcd(index_t x, index_t y)
{
    if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x - y, y);
    }
    else
    {
        return gcd(x, y - x);
    }
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto gcd(Number<X>, Number<Y>)
{
    constexpr auto r = gcd(X, Y);

    return Number<r>{};
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
__host__ __device__ constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, ys...);
}

// least common multiple
template <typename X, typename Y>
__host__ __device__ constexpr auto lcm(X x, Y y)
{
    return (x * y) / gcd(x, y);
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
__host__ __device__ constexpr auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <class T>
struct equal
{
    __host__ __device__ constexpr bool operator()(T x, T y) const { return x == y; }
};

template <class T>
struct less
{
    __host__ __device__ constexpr bool operator()(T x, T y) const { return x < y; }
};

} // namespace math
} // namespace ck

#endif
