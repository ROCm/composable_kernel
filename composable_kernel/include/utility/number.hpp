#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"

namespace ck {

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator+(Number<X>, Number<Y>)
{
    return Number<X + Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator-(Number<X>, Number<Y>)
{
    static_assert(Y <= X, "wrong!");
    return Number<X - Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator*(Number<X>, Number<Y>)
{
    return Number<X * Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator/(Number<X>, Number<Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number<X / Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator%(Number<X>, Number<Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number<X % Y>{};
}

template <long_index_t N>
using LongNumber = integral_constant<long_index_t, N>;

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          enable_if_t<is_same<decltype(X + Y), long_index_t>::value, bool> = true>
__host__ __device__ constexpr auto operator+(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    return LongNumber<X + Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          enable_if_t<is_same<decltype(X - Y), long_index_t>::value, bool> = true>
__host__ __device__ constexpr auto operator-(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y <= X, "wrong!");
    return LongNumber<X - Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          enable_if_t<is_same<decltype(X * Y), long_index_t>::value, bool> = true>
__host__ __device__ constexpr auto operator*(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    return LongNumber<X * Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          enable_if_t<is_same<decltype(X / Y), long_index_t>::value, bool> = true>
__host__ __device__ constexpr auto operator/(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y > 0, "wrong!");
    return LongNumber<X / Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          enable_if_t<is_same<decltype(X % Y), long_index_t>::value, bool> = true>
__host__ __device__ constexpr auto operator%(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y > 0, "wrong!");
    return LongNumber<X % Y>{};
}

} // namespace ck
#endif
