#pragma once

template <class T, T N>
struct integral_constant
{
    static const T value = N;

    __host__ __device__ constexpr T Get() const { return value; }
};

template <class T, T X, T Y>
__host__ __device__ constexpr auto operator+(integral_constant<T, X>, integral_constant<T, Y>)
{
    return integral_constant<T, X + Y>{};
}

template <index_t N>
using Number = integral_constant<index_t, N>;
