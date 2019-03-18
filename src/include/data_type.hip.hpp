#pragma once
#include "config.h"

#if DEVICE_BACKEND_CUDA
namespace CUDA {
#include "cuda_fp16.h"
}
#endif

using half  = CUDA::half;
using half2 = CUDA::half2;

struct half4
{
    half data[4];
};

struct half8
{
    half data[8];
};

template <class T, unsigned N>
struct vector_type
{
};

template <>
struct vector_type<float, 1>
{
    using MemoryType = float;
};

template <>
struct vector_type<float, 2>
{
    using MemoryType = float2;

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
    using MemoryType = float4;
};

template <>
struct vector_type<float2, 2>
{
    using MemoryType = float4;
};

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
struct vector_type<half2, 1>
{
    using MemoryType = half2;
};

template <>
struct vector_type<half2, 2>
{
    using MemoryType = float2;
};

template <>
struct vector_type<half2, 4>
{
    using MemoryType = float4;
};

template <class TDst, class TSrc0, class TSrc1, class TSrc2>
__device__ void fused_multiply_add(TDst& d, TSrc0 s0, TSrc1 s1, TSrc2 s2)
{
    printf("should not call into base");
    assert(false);
}

template <>
__device__ void fused_multiply_add(float& d, float s0, float s1, float s2)
{
    d = s0 * s1 + s2;
}

template <>
__device__ void fused_multiply_add(float& d, float2 s0, float2 s1, float s2)
{
    d = s0.x * s1.x + s0.y * s1.y + s2;
}

template <>
__device__ void fused_multiply_add(float& d, float4 s0, float4 s1, float s2)
{
    d = s0.x * s1.x + s0.y * s1.y + s0.z * s1.z + s0.w * s1.w + s2;
}

template <>
__device__ void fused_multiply_add(half& d, half s0, half s1, half s2)
{
    d = s0 * s1 + s2;
}

template <>
__device__ void fused_multiply_add(half& d, half2 s0, half2 s1, half s2)
{
    d = s0.x * s1.x + s0.y * s1.y + s2;
}