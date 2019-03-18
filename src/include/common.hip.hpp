#pragma once
#include "constant_integral.hip.hpp"
#include "Sequence.hip.hpp"
#include "Array.hip.hpp"
#include "functional.hip.hpp"

__device__ unsigned get_thread_local_1d_id() { return threadIdx.x; }

__device__ unsigned get_block_1d_id() { return blockIdx.x; }

template <class T1, class T2>
struct is_same
{
    static const bool value = false;
};

template <class T>
struct is_same<T, T>
{
    static const bool value = true;
};

template <class T, unsigned N>
struct vector_type
{
};

template <>
struct vector_type<float, 1>
{
    using VectorType = float;
};

template <>
struct vector_type<float, 2>
{
    using VectorType = float2;
};

template <>
struct vector_type<float, 4>
{
    using VectorType = float4;
};

#if 0
template <>
struct vector_type<half_float::half, 1>
{
    using VectorType = half_float::half;
};

template <>
struct vector_type<half_float::half, 2>
{
    using VectorType = float;
};

template <>
struct vector_type<half_float::half, 4>
{
    using VectorType = float2;
};

template <>
struct vector_type<half_float::half, 8>
{
    using VectorType = float4;
};
#endif

#if 1
template <>
struct vector_type<half, 1>
{
    using VectorType = half;

    __host__ __device__ static VectorType pack(half s) { return s; }
};

template <>
struct vector_type<half, 2>
{
    using VectorType = half2;

    union Data
    {
        VectorType vector;
        half scalar[2];
    };

    __host__ __device__ static VectorType pack(half s0, half s1)
    {
        Data data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<half, 4>
{
    using VectorType = float2;
};

template <>
struct vector_type<half, 8>
{
    using VectorType = float4;
};
#endif

template <typename T>
__host__ __device__ constexpr T max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__host__ __device__ constexpr T min(T a, T b)
{
    return a < b ? a : b;
}

__host__ __device__ constexpr unsigned integer_divide_ceil(unsigned a, unsigned b)
{
    return (a + b - 1) / b;
}
