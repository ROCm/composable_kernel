// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

//using ext_vector_type = std::vector;

template <typename T, std::size_t N>
struct ext_vector_type {
    union {
        std::array<T, N> data;
        struct {
            T x;
            T y;
            T z;
            T w;
        };
        struct {
            T hi;
            T lo;
        };
    };
    
    ext_vector_type() {
        for (std::size_t i = 0; i < N; i++) {
            data[i++] = static_cast<T>(0);
        }
    }
    
    ext_vector_type(std::initializer_list<T> init) {
        std::size_t i = 0;
        for (const auto& value : init) {
            if (i < N) {
                data[i++] = value;
            }
        }
    }
    
    T& operator[](std::size_t index) { return data[index]; }
    const T& operator[](std::size_t index) const { return data[index]; }

    static constexpr std::size_t size() { return N; }
    
    /*ext_vector_type operator+(const ext_vector_type& other) const {
        ext_vector_type result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }
    ext_vector_type operator-(const ext_vector_type& other) const {
        ext_vector_type result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    ext_vector_type operator*(T scalar) const {
        ext_vector_type result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
    ext_vector_type operator/(T scalar) const {
        ext_vector_type result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }
    T dot(const ext_vector_type& other) const {
        T result = 0;
        for (std::size_t i = 0; i < N; ++i) {
            result += data[i] * other.data[i];
        }
        return result;
    }*/
};

// fp64
using fp64_t   = double;
using fp64x2_t = ext_vector_type<double,  2>;
using fp64x4_t = ext_vector_type<double,  4>;

// fp32
using fp32_t    = float;
using fp32x2_t  = ext_vector_type<float,  2>;
using fp32x4_t  = ext_vector_type<float,  4>;
using fp32x8_t  = ext_vector_type<float,  8>;
using fp32x16_t = ext_vector_type<float, 16>;
using fp32x32_t = ext_vector_type<float, 32>;
using fp32x64_t = ext_vector_type<float, 64>;

// fp16
// using fp16_t = half_t;
// using fp16x2_t = __half2; // defined in hip_fp16_gcc.h
using fp16x2_t  = ext_vector_type<fp16_t,  2>;
using fp16x4_t  = ext_vector_type<fp16_t,  4>;
using fp16x8_t  = ext_vector_type<fp16_t,  8>;
using fp16x16_t = ext_vector_type<fp16_t, 16>;
using fp16x32_t = ext_vector_type<fp16_t, 32>;
using fp16x64_t = ext_vector_type<fp16_t, 64>;

// bf16
// using bf16_t = ...
using bf16x2_t  = ext_vector_type<bf16_t,  2>;
using bf16x4_t  = ext_vector_type<bf16_t,  4>;
using bf16x8_t  = ext_vector_type<bf16_t,  8>;
using bf16x16_t = ext_vector_type<bf16_t, 16>;
using bf16x32_t = ext_vector_type<bf16_t, 32>;
using bf16x64_t = ext_vector_type<bf16_t, 64>;

/*#if CK_TILE_USE_CUSTOM_DATA_TYPE
// f8
// using fp8_t
using fp8x2_t  = ext_vector_type<fp8_raw_t,  2>;
using fp8x4_t  = ext_vector_type<fp8_raw_t,  4>;
using fp8x8_t  = ext_vector_type<fp8_raw_t,  8>;
using fp8x16_t = ext_vector_type<fp8_raw_t, 16>;
using fp8x32_t = ext_vector_type<fp8_raw_t, 32>;
using fp8x64_t = ext_vector_type<fp8_raw_t, 64>;

// bf8
// using bf8_t
using bf8x2_t  = ext_vector_type<bf8_raw_t,  2>;
using bf8x4_t  = ext_vector_type<bf8_raw_t,  4>;
using bf8x8_t  = ext_vector_type<bf8_raw_t,  8>;
using bf8x16_t = ext_vector_type<bf8_raw_t, 16>;
using bf8x32_t = ext_vector_type<bf8_raw_t, 32>;
using bf8x64_t = ext_vector_type<bf8_raw_t, 64>;
#else*/
// f8
// using fp8_t
using fp8x2_t  = ext_vector_type<fp8_t,  2>;
using fp8x4_t  = ext_vector_type<fp8_t,  4>;
using fp8x8_t  = ext_vector_type<fp8_t,  8>;
using fp8x16_t = ext_vector_type<fp8_t, 16>;
using fp8x32_t = ext_vector_type<fp8_t, 32>;
using fp8x64_t = ext_vector_type<fp8_t, 64>;

// bf8
// using bf8_t
using bf8x2_t  = ext_vector_type<bf8_t,  2>;
using bf8x4_t  = ext_vector_type<bf8_t,  4>;
using bf8x8_t  = ext_vector_type<bf8_t,  8>;
using bf8x16_t = ext_vector_type<bf8_t, 16>;
using bf8x32_t = ext_vector_type<bf8_t, 32>;
using bf8x64_t = ext_vector_type<bf8_t, 64>;
//#endif

// pk_int4_t
// using pk_int4_t
using pk_int4x2_t  = ext_vector_type<int8_t,  2>;
using pk_int4x4_t  = ext_vector_type<int8_t,  4>;
using pk_int4x8_t  = ext_vector_type<int8_t,  8>;
using pk_int4x16_t = ext_vector_type<int8_t, 16>;
using pk_int4x32_t = ext_vector_type<int8_t, 32>;
using pk_int4x64_t = ext_vector_type<int8_t, 64>;

CK_TILE_HOST fp16x2_t pk_add_f16(const fp16x2_t& x, const fp16x2_t& y)
{
    fp16x2_t vector_res;

    vector_res.x = x.x + y.x;
    vector_res.y = x.y + y.y;

    return vector_res;
}

CK_TILE_HOST_DEVICE fp32x2_t pk_int4_t_to_fp32x2_t(const pk_int4_t& x)
{
    uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    fp32x2_t res = {x_h, x_l};
#elif
    fp32x2_t res = {x_l, x_h};
#endif
    return res;
}

CK_TILE_HOST_DEVICE fp16x2_t pk_int4_t_to_halfx2_t(const pk_int4_t& x)
{
    uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);
#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    uint32_t i4s = ((x_u8 & 0x0f) << 16) | ((x_u8 & 0xf0) >> 4);
#elif
    uint32_t i4s = ((x_u8 & 0xf0) << 12) | (x_u8 & 0xf);
#endif
    const int EX  = 0x64006400;
    const int SUB = 0xE408E408; //-8

    int lo = i4s | EX;

    return pk_add_f16(bit_cast<fp16x2_t>(lo), bit_cast<fp16x2_t>(SUB));
}

CK_TILE_HOST_DEVICE bf16x2_t pk_int4_t_to_bfloat16x2_t(const pk_int4_t& x)
{
    uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    bf16x2_t res = {static_cast<bf16_t>(x_h), static_cast<bf16_t>(x_l)};
#elif
    bf16x2_t res = {static_cast<bf16_t>(x_l), static_cast<bf16_t>(x_h)};
#endif
    return res;
}


} // namespace ck_tile
