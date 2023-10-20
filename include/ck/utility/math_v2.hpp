// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#include <cmath>
#endif

#include "ck/utility/data_type.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/type_convert.hpp"

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

template <typename T>
inline __host__ T tanh(T x)
{
    return ck::type_convert<T>(std::tanhf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float tanh<float>(float x)
{
    return std::tanhf(x);
};

template <>
inline __host__ double tanh<double>(double x)
{
    return std::tanh(x);
};

template <typename T>
inline __host__ T exp(T x)
{
    return ck::type_convert<T>(std::expf(ck::type_convert<float>(x)));
}

template <>
inline __host__ float exp<float>(float x)
{
    return std::expf(x);
}

template <>
inline __host__ double exp<double>(double x)
{
    return std::exp(x);
}

template <typename T>
inline __host__ T log(T x)
{
    return ck::type_convert<T>(std::logf(ck::type_convert<float>(x)));
}

template <>
inline __host__ float log<float>(float x)
{
    return std::logf(x);
}

template <>
inline __host__ double log<double>(double x)
{
    return std::log(x);
}

template <typename T>
inline __host__ T pow(T x, T gamma)
{
    return ck::type_convert<T>(
        std::powf(ck::type_convert<float>(x), ck::type_convert<float>(gamma)));
}

template <>
inline __host__ float pow<float>(float x, float gamma)
{
    return std::powf(x, gamma);
}

template <>
inline __host__ double pow<double>(double x, double gamma)
{
    return std::pow(x, gamma);
}

template <typename T>
inline __host__ T expm1(T x)
{
    return ck::type_convert<T>(std::expm1f(ck::type_convert<float>(x)));
}

template <>
inline __host__ float expm1<float>(float x)
{
    return std::expm1f(x);
}

template <>
inline __host__ double expm1<double>(double x)
{
    return std::expm1(x);
}

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

template <typename T>
inline __device__ T tanh(T x)
{
    return ck::type_convert<T>(::tanhf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float tanh<float>(float x)
{
    return ::tanhf(x);
};

template <>
inline __device__ double tanh<double>(double x)
{
    return ::tanh(x);
};

template <typename T>
inline __device__ T exp(T x)
{
    return ck::type_convert<T>(__expf(ck::type_convert<float>(x)));
};

template <>
inline __device__ half_t exp<half_t>(half_t x)
{
    return hexp(x);
};

template <>
inline __device__ float exp<float>(float x)
{
    return __expf(x);
};

template <>
inline __device__ double exp<double>(double x)
{
    return exp(x);
};

template <typename T>
inline __device__ T log(T x)
{
    return ck::type_convert<T>(__logf(ck::type_convert<float>(x)));
};

template <>
inline __device__ half_t log<half_t>(half_t x)
{
    return hlog(x);
};

template <>
inline __device__ float log<float>(float x)
{
    return __logf(x);
};

template <>
inline __device__ double log<double>(double x)
{
    return log(x);
};

template <typename T>
inline __device__ T pow(T x, T gamma)
{
    return ck::type_convert<T>(powf(ck::type_convert<float>(x), ck::type_convert<float>(gamma)));
};

template <>
inline __device__ float pow<float>(float x, float gamma)
{
    return powf(x, gamma);
};

template <>
inline __device__ double pow<double>(double x, double gamma)
{
    return pow(x, gamma);
};

template <typename T>
inline __device__ T expm1(T x)
{
    return ck::type_convert<T>(expm1f(ck::type_convert<float>(x)));
};

template <>
inline __device__ float expm1<float>(float x)
{
    return expm1f(x);
};

template <>
inline __device__ double expm1<double>(double x)
{
    return expm1(x);
};

} // namespace math
} // namespace ck
