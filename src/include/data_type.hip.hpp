#pragma once
#include "config.h"

template <class T, index_t N>
struct vector_type
{
};

template <>
struct vector_type<float, 1>
{
    typedef float MemoryType;
};

template <>
struct vector_type<float, 2>
{
#if DEVICE_BACKEND_HIP
    // For some reason, HIP compiler need this definition to generate optimal load and store
    // instruction
    typedef float MemoryType __attribute__((ext_vector_type(2)));
#elif DEVICE_BACKEND_CUDA
    // For some reason, CUDA need this definition to, otherwise
    //   compiler won't generate optimal load and store instruction, and
    //   kernel would produce wrong result, indicating the compiler fail to generate correct
    //   instruction,
    using MemoryType = float2;
#endif

    __host__ __device__ static MemoryType Pack(float s0, float s1)
    {
        union
        {
            MemoryType vector;
            float scalar[2];
        } data;

        data.scalar[0] = s0;
        data.scalar[1] = s1;
        return data.vector;
    }
};

template <>
struct vector_type<float, 4>
{
#if DEVICE_BACKEND_HIP
    // For some reason, HIP compiler need this definition to generate optimal load and store
    // instruction
    typedef float MemoryType __attribute__((ext_vector_type(4)));
#elif DEVICE_BACKEND_CUDA
    // For some reason, CUDA need this definition to, otherwise
    //   compiler won't generate optimal load and store instruction, and
    //   kernel would produce wrong result, indicating the compiler fail to generate correct
    //   instruction,
    using MemoryType = float4;
#endif
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

__device__ void fused_multiply_accumulate(float& d, const float& s0, const float& s1)
{
    d += s0 * s1;
}

__device__ void fused_multiply_accumulate(float& d, const float2& s0, const float2& s1)
{
    d += s0.x * s1.x;
    d += s0.y * s1.y;
}

__device__ void fused_multiply_accumulate(float& d, const float4& s0, const float4& s1)
{
    d += s0.x * s1.x;
    d += s0.y * s1.y;
    d += s0.z * s1.z;
    d += s0.w * s1.w;
}

#if 0
__device__ void fused_multiply_accumulate(half& d, const half& s0, const half& s1) { d += s0 * s1; }

__device__ void fused_multiply_accumulate(half& d, const half2& s0, const half2& s1)
{
    d += s0.x * s1.x;
    d += s0.y * s1.y;
}

__device__ void fused_multiply_accumulate(float& d, const half2& s0, const half2& s1)
{
    d += s0.x * s1.x + s0.y * s1.y;
}

__device__ void fused_multiply_accumulate(char& d, const char& s0, const char& s1) { d += s0 * s1; }

// TODO:: this interface is misleading, s0, s1 are actually int8x4
//  need to make a better interface
__device__ void fused_multiply_accumulate(int32_t& d, const int32_t& s0, const int32_t& s1)
{
#if DEVICE_BACKEND_CUDA
    d = __dp4a(s0, s1, d);
#endif
}
#endif
