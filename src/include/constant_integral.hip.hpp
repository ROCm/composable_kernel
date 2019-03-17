#pragma once

template <class T, T N>
struct integral_constant
{
    static const T value = N;

    __host__ __device__ constexpr T Get() const { return value; }
};

template <unsigned N>
using Number = integral_constant<unsigned, N>;
