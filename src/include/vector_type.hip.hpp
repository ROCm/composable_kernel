#pragma once
#include "config.h"
#include "constant_integral.hip.hpp"

template <class T, index_t N>
struct vector_type
{
};

template <>
struct vector_type<float, 1>
{
    typedef float MemoryType;

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }
};

template <>
struct vector_type<float, 2>
{
    using MemoryType = float2_t;

    union Data
    {
        MemoryType vector;
        float scalar[2];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }

    __host__ __device__ static MemoryType Pack(float s0, float s1)
    {
        Data data;
        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<float, 4>
{
    using MemoryType = float4_t;

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }
};

#if 0
template <>
struct vector_type<half, 1>
{
    using MemoryType = half;

    __host__ __device__ static MemoryType Pack(half s) { return s; }
};

template <>
struct vector_type<half, 2>
{
    using MemoryType = half2;

    __host__ __device__ static MemoryType Pack(half s0, half s1)
    {
        union
        {
            MemoryType vector;
            half scalar[2];
        } data;

        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<half, 4>
{
    using MemoryType = float2;
};

template <>
struct vector_type<half, 8>
{
    using MemoryType = float4;
};

template <>
struct vector_type<char, 1>
{
    using MemoryType = char;

    __host__ __device__ static MemoryType Pack(char s) { return s; }
};

template <>
struct vector_type<char, 2>
{
    using MemoryType = int16_t;

    __host__ __device__ static MemoryType Pack(char s0, char s1)
    {
        union
        {
            MemoryType vector;
            char scalar[2];
        } data;

        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<char, 4>
{
    using MemoryType = int32_t;

    __host__ __device__ static MemoryType Pack(char s0, char s1, char s2, char s3)
    {
        union
        {
            MemoryType vector;
            char scalar[4];
        } data;

        data.scalar[0] = s0;
        data.scalar[1] = s1;
        data.scalar[2] = s2;
        data.scalar[3] = s3;
        return data.vector;
    }
};

template <>
struct vector_type<char, 8>
{
    using MemoryType = int64_t;
};

template <>
struct vector_type<int32_t, 2>
{
    using MemoryType = int64_t;
};

template <>
struct vector_type<char2, 2>
{
    using MemoryType = char4;
};

template <>
struct vector_type<char2, 4>
{
    using MemoryType = int64_t;
};

template <>
struct vector_type<char4, 1>
{
    using MemoryType = int;
};

template <>
struct vector_type<char4, 2>
{
    using MemoryType = int64_t;
};
#endif
