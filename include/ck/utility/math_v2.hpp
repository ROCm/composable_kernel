// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#include <cmath>
#endif

#include "ck/utility/data_type.hpp"
#include "ck/utility/type.hpp"

namespace ck {
namespace math {

// math functions for the host,  some are implemented by calling C++ std functions

static inline __host__ float abs(float x) { return std::abs(x); };

static inline __host__ double abs(double x) { return std::abs(x); };

static inline __host__ int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

static inline __host__ int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

static inline __host__ half_t abs(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    uint16_t abs_xx = xx & 0x7fff;

    half_t abs_x = ck::bit_cast<half_t>(abs_xx);

    return abs_x;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __host__ int4_t abs(int4_t x)
{
    int4_t sgn = x >> (4 - 1);
    return (x ^ sgn) - sgn;
}
#endif

static inline __host__ bool isnan(float x) { return std::isnan(x); };

static inline __host__ bool isnan(double x) { return std::isnan(x); };

static inline __host__ bool isnan(int8_t x)
{
    (void)x;
    return false;
};

static inline __host__ bool isnan(int32_t x)
{
    (void)x;
    return false;
};

static inline __host__ bool isnan(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    return (xx & 0x7FFF) > 0x7C00;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __host__ bool isnan(int4_t x)
{
    (void)x;
    return false;
};
#endif

static inline __host__ half_t sqrt(half_t x)
{
    return static_cast<half_t>(std::sqrt(static_cast<float>(x)));
};

static inline __host__ float sqrt(float x) { return std::sqrt(x); };

static inline __host__ double sqrt(double x) { return std::sqrt(x); };

static inline __host__ half_t tanh(half_t x)
{
    return static_cast<half_t>(std::tanh(static_cast<float>(x)));
};

static inline __host__ float tanh(float x) { return std::tanh(x); };

static inline __host__ double tanh(double x) { return std::tanh(x); };

// math functions for the HIP kernel,  some are implemented by calling hip builtin functions

static inline __device__ float abs(float x) { return ::abs(x); };

static inline __device__ double abs(double x) { return ::abs(x); };

static inline __device__ int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

static inline __device__ int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __device__ int4_t abs(int4_t x)
{
    int4_t sgn = x >> (4 - 1);

    return (x ^ sgn) - sgn;
};
#endif

static inline __device__ half_t abs(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    uint16_t abs_xx = xx & 0x7fff;

    half_t abs_x = ck::bit_cast<half_t>(abs_xx);

    return abs_x;
};

static inline __device__ bool isnan(float x) { return ::isnan(x); };

static inline __device__ bool isnan(double x) { return ::isnan(x); };

static inline __device__ bool isnan(int8_t x)
{
    (void)x;
    return false;
};

static inline __device__ bool isnan(int32_t x)
{
    (void)x;
    return false;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __device__ bool isnan(int4_t x)
{
    (void)x;
    return false;
};
#endif

static inline __device__ bool isnan(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    return (xx & 0x7FFF) > 0x7C00;
};

static inline __device__ half_t sqrt(half_t x)
{
    return static_cast<half_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

static inline __device__ float sqrt(float x) { return __builtin_amdgcn_sqrtf(x); };

static inline __device__ double sqrt(double x) { return __builtin_amdgcn_sqrt(x); };

static inline __device__ half_t tanh(half_t x)
{
    return static_cast<half_t>(::tanhf(static_cast<float>(x)));
};

static inline __device__ float tanh(float x) { return ::tanhf(x); };

static inline __device__ double tanh(double x) { return ::tanh(x); };

} // namespace math
} // namespace ck
