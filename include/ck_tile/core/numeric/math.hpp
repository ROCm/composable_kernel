// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include <type_traits>
#include <stdint.h>

namespace ck_tile {

template <typename T, T s>
struct scales
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a) const { return s * a; }
};

template <typename T>
struct plus
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct minus
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a - b; }
};

struct multiplies
{
    template <typename A, typename B>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const A& a, const B& b) const
    {
        return a * b;
    }
};

template <typename T>
struct maximize
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
};

template <typename T>
struct minimize
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a <= b ? a : b; }
};

template <typename T>
struct integer_divide_ceiler
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const
    {
        static_assert(std::is_same<T, index_t>{} || std::is_same<T, int>{}, "wrong type");
        return (a + b - number<1>{}) / b;
    }
};

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_divide_floor(X x, Y y)
{
    return x / y;
}

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - number<1>{}) / y;
}

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T max(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t max(number<X>, index_t y)
{
    return X > y ? X : y;
}

template <index_t Y>
CK_TILE_HOST_DEVICE constexpr index_t max(index_t x, number<Y>)
{
    return x > Y ? x : Y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto max(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return max(x, max(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T min(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t min(number<X>, index_t y)
{
    return X < y ? X : y;
}

template <index_t Y>
CK_TILE_HOST_DEVICE constexpr index_t min(index_t x, number<Y>)
{
    return x < Y ? x : Y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto min(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return min(x, min(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T clamp(const T& x, const T& lowerbound, const T& upperbound)
{
    return min(max(x, lowerbound), upperbound);
}

// greatest common divisor, aka highest common factor
CK_TILE_HOST_DEVICE constexpr index_t gcd(index_t x, index_t y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
}

template <index_t X, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto gcd(number<X>, number<Y>)
{
    constexpr auto r = gcd(X, Y);

    return number<r>{};
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto lcm(X x, Y y)
{
    return (x * y) / gcd(x, y);
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename T>
struct equal
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(T x, T y) const { return x == y; }
};

template <typename T>
struct less
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(T x, T y) const { return x < y; }
};

CK_TILE_HOST_DEVICE constexpr int32_t next_power_of_two(int32_t x)
{
    // TODO: x need to be 2 ~ 0x7fffffff. 0, 1, or larger than 0x7fffffff will compile fail
    return 1 << (32 - __builtin_clz(x - 1));
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr auto next_power_of_two()
{
    constexpr index_t y = next_power_of_two(X);
    return number<y>{};
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr auto next_power_of_two(number<X>)
{
    constexpr index_t y = next_power_of_two(X);
    return number<y>{};
}

CK_TILE_HOST_DEVICE constexpr int32_t integer_log2_floor(int32_t x)
{
    // TODO: x need to be 1 ~ 0x7fffffff
    // __builtin_clz will produce unexpected result if x is 0;
    return 31 - __builtin_clz(x);
}

CK_TILE_HOST_DEVICE constexpr bool is_power_of_two_integer(int32_t x)
{
    // TODO: x need to be 1 ~ 0x7fffffff
    return x == (1 << integer_log2_floor(x));
}

#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif

template <typename T>
struct log2e;

template <>
struct log2e<double>
{
    static constexpr double value = C_LOG2E;
};

template <>
struct log2e<float>
{
    static constexpr float value = C_LOG2E;
};

template <typename T = double>
inline constexpr T log2e_v = log2e<T>::value;

// math
CK_TILE_HOST_DEVICE
float abs(const float& x)
{
    union
    {
        float f32;
        uint32_t u32;
    } y;
    y.f32 = x;
    y.u32 = y.u32 & 0x7fffffff;
    return y.f32;
}

CK_TILE_HOST_DEVICE
bool isnan(const float& x)
{
    uint32_t xx = bit_cast<uint32_t>(x);
    return (xx & 0x7fffffff) > 0x7F800000;
}

CK_TILE_DEVICE
float sqrt(float x) { return __builtin_amdgcn_sqrtf(x); };

CK_TILE_DEVICE
float exp(float x) { return __expf(x); };

CK_TILE_DEVICE
float exp2(float x) { return exp2f(x); };

CK_TILE_DEVICE
float log(float x) { return __logf(x); };

} // namespace ck_tile
