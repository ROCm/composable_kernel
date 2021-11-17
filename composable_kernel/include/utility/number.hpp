#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"
#include <type_traits>

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
using Number64 = integral_constant<long_index_t, N>;

#if 1

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          std::enable_if_t<std::is_same<Index0, long_index_t>::value ||
                               std::is_same<Index1, long_index_t>::value,
                           bool> = true>
__host__ __device__ constexpr auto operator+(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    return Number64<X + Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          std::enable_if_t<std::is_same<Index0, long_index_t>::value ||
                               std::is_same<Index1, long_index_t>::value,
                           bool> = true>
__host__ __device__ constexpr auto operator-(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y <= X, "wrong!");
    return Number64<X - Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          std::enable_if_t<std::is_same<Index0, long_index_t>::value ||
                               std::is_same<Index1, long_index_t>::value,
                           bool> = true>
__host__ __device__ constexpr auto operator*(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    return Number64<X * Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          std::enable_if_t<std::is_same<Index0, long_index_t>::value ||
                               std::is_same<Index1, long_index_t>::value,
                           bool> = true>
__host__ __device__ constexpr auto operator/(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number64<X / Y>{};
}

template <typename Index0,
          Index0 X,
          typename Index1,
          Index1 Y,
          std::enable_if_t<std::is_same<Index0, long_index_t>::value ||
                               std::is_same<Index1, long_index_t>::value,
                           bool> = true>
__host__ __device__ constexpr auto operator%(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number64<X % Y>{};
}

#else

template <long_index_t X, long_index_t Y>
__host__ __device__ constexpr auto operator+(integral_constant<long_index_t, X>,
                                             integral_constant<long_index_t, Y>)
{
    return Number64<X + Y>{};
}

template <long_index_t X, index_t Y>
__host__ __device__ constexpr auto operator+(integral_constant<long_index_t, X>,
                                             integral_constant<index_t, Y>)
{
    return Number64<X + Y>{};
}

template <long_index_t X, long_index_t Y>
__host__ __device__ constexpr auto operator-(integral_constant<long_index_t, X>,
                                             integral_constant<long_index_t, Y>)
{
    return Number64<X - Y>{};
}

template <long_index_t X, index_t Y>
__host__ __device__ constexpr auto operator-(integral_constant<long_index_t, X>,
                                             integral_constant<index_t, Y>)
{
    return Number64<X - Y>{};
}

template <long_index_t X, long_index_t Y>
__host__ __device__ constexpr auto operator*(integral_constant<long_index_t, X>,
                                             integral_constant<long_index_t, Y>)
{
    return Number64<X * Y>{};
}

template <long_index_t X, index_t Y>
__host__ __device__ constexpr auto operator*(integral_constant<long_index_t, X>,
                                             integral_constant<index_t, Y>)
{
    return Number64<X * Y>{};
}

template <long_index_t X, long_index_t Y>
__host__ __device__ constexpr auto operator/(integral_constant<long_index_t, X>,
                                             integral_constant<long_index_t, Y>)
{
    return Number64<X / Y>{};
}

template <long_index_t X, index_t Y>
__host__ __device__ constexpr auto operator/(integral_constant<long_index_t, X>,
                                             integral_constant<index_t, Y>)
{
    return Number64<X / Y>{};
}

template <long_index_t X, long_index_t Y>
__host__ __device__ constexpr auto operator%(integral_constant<long_index_t, X>,
                                             integral_constant<long_index_t, Y>)
{
    return Number64<X % Y>{};
}

template <long_index_t X, index_t Y>
__host__ __device__ constexpr auto operator%(integral_constant<long_index_t, X>,
                                             integral_constant<index_t, Y>)
{
    return Number64<X % Y>{};
}
#endif

} // namespace ck
#endif
