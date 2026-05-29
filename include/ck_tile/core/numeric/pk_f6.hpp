// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/numeric/mxfp_scale.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
#define CK_TILE_FP6_CVT_DEVICE 1
#else
#define CK_TILE_FP6_CVT_DEVICE 0
#endif

namespace ck_tile {
using uint32x3_t = uint32_t __attribute__((ext_vector_type(3)));
using fp32x8_t   = float __attribute__((ext_vector_type(8)));
using fp32x16_t  = float __attribute__((ext_vector_type(16)));
using fp16x16_t  = _Float16 __attribute__((ext_vector_type(16)));
using bf16x16_t  = bfloat16_t __attribute__((ext_vector_type(16)));

// Helpers for 16-element vector access
namespace detail {
struct fp16x16_repr
{
    _Float16 e[16];
};
struct bf16x16_repr
{
    bfloat16_t e[16];
};
struct fp32x16_repr
{
    float e[16];
};

// Lane accessors for 16-element vectors
CK_TILE_HOST_DEVICE constexpr _Float16 lane(const fp16x16_t& v, index_t i)
{
    return ck_tile::bit_cast<fp16x16_repr>(v).e[i];
}
CK_TILE_HOST_DEVICE constexpr bfloat16_t lane(const bf16x16_t& v, index_t i)
{
    return ck_tile::bit_cast<bf16x16_repr>(v).e[i];
}
CK_TILE_HOST_DEVICE constexpr float lane(const fp32x16_t& v, index_t i)
{
    return ck_tile::bit_cast<fp32x16_repr>(v).e[i];
}
} // namespace detail

// Forward declarations
struct pk_float6_e2m3_t;
struct pk_float6_e3m2_t;

CK_TILE_HOST_DEVICE constexpr pk_float6_e2m3_t float_to_pk_fp6(const float& x, float scale = 1.f);
CK_TILE_HOST_DEVICE constexpr pk_float6_e3m2_t float_to_pk_bf6(const float& x, float scale = 1.f);

// FP6 E2M3 (2-bit exponent, 3-bit mantissa)
// Packed format: 16 fp6 values in 96 bits (3 x uint32_t)
struct pk_float6_e2m3_t
{
    using raw_type     = uint8_t;    // Type for a single unpacked fp6 element
    using storage_type = uint32x3_t; // Type for storing 16 fp6 values (96 bits)
    storage_type data;

    CK_TILE_HOST_DEVICE constexpr pk_float6_e2m3_t() : data{} {}
    CK_TILE_HOST_DEVICE constexpr pk_float6_e2m3_t(raw_type init);
    CK_TILE_HOST_DEVICE constexpr pk_float6_e2m3_t(storage_type init);
    CK_TILE_HOST_DEVICE explicit pk_float6_e2m3_t(float init, float scale = 1.f);
    CK_TILE_HOST_DEVICE constexpr storage_type& get() [[clang::lifetimebound]] { return data; }
    CK_TILE_HOST_DEVICE constexpr storage_type get() const { return data; }

    CK_TILE_HOST_DEVICE constexpr float to_float(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16_t to_fp16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16x16_t to_fp16x16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16_t to_bf16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16x16_t to_bf16x16(float scale = 1.f) const;

    CK_TILE_HOST_DEVICE constexpr operator float() const { return to_float(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16_t() const { return to_fp16(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16x16_t() const { return to_fp16x16(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16_t() const { return to_bf16(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16x16_t() const { return to_bf16x16(); }

#if !CK_TILE_AVX512F_WA
    CK_TILE_HOST_DEVICE constexpr fp32x16_t to_fp32x16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr operator fp32x16_t() const { return to_fp32x16(); }
#endif

    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr raw_type unpack(number<I>) const
    {
        static_assert(I < 16, "Index is out of range.");
        constexpr index_t bit_offset  = I * 6;
        constexpr index_t word_idx    = bit_offset / 32;
        constexpr index_t bit_in_word = bit_offset % 32;

        if constexpr(bit_in_word <= 26)
        {
            // Value fits entirely in one word
            return (data[word_idx] >> bit_in_word) & 0b00111111;
        }
        else
        {
            // Value spans two words
            constexpr index_t bits_in_first  = 32 - bit_in_word;
            constexpr index_t bits_in_second = 6 - bits_in_first;
            uint8_t low_bits  = (data[word_idx] >> bit_in_word) & ((1u << bits_in_first) - 1);
            uint8_t high_bits = (data[word_idx + 1] & ((1u << bits_in_second) - 1))
                                << bits_in_first;
            return low_bits | high_bits;
        }
    }

    CK_TILE_HOST_DEVICE constexpr uint8_t get_element(index_t idx) const
    {
        const index_t bit_offset  = idx * 6;
        const index_t word_idx    = bit_offset / 32;
        const index_t bit_in_word = bit_offset % 32;

        if(bit_in_word <= 26)
        {
            // Value fits entirely in one word
            return (data[word_idx] >> bit_in_word) & 0b00111111;
        }
        else
        {
            // Value spans two words
            const index_t bits_in_first  = 32 - bit_in_word;
            const index_t bits_in_second = 6 - bits_in_first;
            uint8_t low_bits  = (data[word_idx] >> bit_in_word) & ((1u << bits_in_first) - 1);
            uint8_t high_bits = (data[word_idx + 1] & ((1u << bits_in_second) - 1))
                                << bits_in_first;
            return low_bits | high_bits;
        }
    }

    CK_TILE_HOST_DEVICE constexpr void set_element(index_t idx, uint8_t value)
    {
        const index_t bit_offset  = idx * 6;
        const index_t word_idx    = bit_offset / 32;
        const index_t bit_in_word = bit_offset % 32;

        value &= 0b00111111; // Ensure only 6 bits

        if(bit_in_word <= 26)
        {
            // Value fits entirely in one word
            uint32_t mask = 0b00111111u << bit_in_word;
            data[word_idx] =
                (data[word_idx] & ~mask) | (static_cast<uint32_t>(value) << bit_in_word);
        }
        else
        {
            // Value spans two words
            const index_t bits_in_first  = 32 - bit_in_word;
            const index_t bits_in_second = 6 - bits_in_first;

            uint32_t mask1 = ((1u << bits_in_first) - 1) << bit_in_word;
            uint32_t mask2 = (1u << bits_in_second) - 1;

            data[word_idx] =
                (data[word_idx] & ~mask1) | ((value & ((1u << bits_in_first) - 1)) << bit_in_word);
            data[word_idx + 1] = (data[word_idx + 1] & ~mask2) | ((value >> bits_in_first) & mask2);
        }
    }
};

// BF6 E3M2 (3-bit exponent, 2-bit mantissa)
// Packed format: 16 bf6 values in 96 bits (3 x uint32_t)
struct pk_float6_e3m2_t
{
    using raw_type     = uint8_t;    // Type for a single unpacked bf6 element
    using storage_type = uint32x3_t; // Type for storing 16 bf6 values (96 bits)
    storage_type data;

    CK_TILE_HOST_DEVICE constexpr pk_float6_e3m2_t() : data{} {}
    CK_TILE_HOST_DEVICE constexpr pk_float6_e3m2_t(raw_type init);
    CK_TILE_HOST_DEVICE constexpr pk_float6_e3m2_t(storage_type init);
    CK_TILE_HOST_DEVICE explicit pk_float6_e3m2_t(float init, float scale = 1.f);
    CK_TILE_HOST_DEVICE constexpr storage_type& get() [[clang::lifetimebound]] { return data; }
    CK_TILE_HOST_DEVICE constexpr storage_type get() const { return data; }

    CK_TILE_HOST_DEVICE constexpr float to_float(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16_t to_fp16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16x16_t to_fp16x16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16_t to_bf16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16x16_t to_bf16x16(float scale = 1.f) const;

    CK_TILE_HOST_DEVICE constexpr operator float() const { return to_float(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16_t() const { return to_fp16(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16x16_t() const { return to_fp16x16(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16_t() const { return to_bf16(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16x16_t() const { return to_bf16x16(); }

#if !CK_TILE_AVX512F_WA
    CK_TILE_HOST_DEVICE constexpr fp32x16_t to_fp32x16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr operator fp32x16_t() const { return to_fp32x16(); }
#endif

    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr uint8_t unpack(number<I>) const
    {
        static_assert(I < 16, "Index is out of range.");
        constexpr index_t bit_offset  = I * 6;
        constexpr index_t word_idx    = bit_offset / 32;
        constexpr index_t bit_in_word = bit_offset % 32;

        if constexpr(bit_in_word <= 26)
        {
            // Value fits entirely in one word
            return (data[word_idx] >> bit_in_word) & 0b00111111;
        }
        else
        {
            // Value spans two words
            constexpr index_t bits_in_first  = 32 - bit_in_word;
            constexpr index_t bits_in_second = 6 - bits_in_first;
            uint8_t low_bits  = (data[word_idx] >> bit_in_word) & ((1u << bits_in_first) - 1);
            uint8_t high_bits = (data[word_idx + 1] & ((1u << bits_in_second) - 1))
                                << bits_in_first;
            return low_bits | high_bits;
        }
    }

    CK_TILE_HOST_DEVICE constexpr uint8_t get_element(index_t idx) const
    {
        const index_t bit_offset  = idx * 6;
        const index_t word_idx    = bit_offset / 32;
        const index_t bit_in_word = bit_offset % 32;

        if(bit_in_word <= 26)
        {
            // Value fits entirely in one word
            return (data[word_idx] >> bit_in_word) & 0b00111111;
        }
        else
        {
            // Value spans two words
            const index_t bits_in_first  = 32 - bit_in_word;
            const index_t bits_in_second = 6 - bits_in_first;
            uint8_t low_bits  = (data[word_idx] >> bit_in_word) & ((1u << bits_in_first) - 1);
            uint8_t high_bits = (data[word_idx + 1] & ((1u << bits_in_second) - 1))
                                << bits_in_first;
            return low_bits | high_bits;
        }
    }

    CK_TILE_HOST_DEVICE constexpr void set_element(index_t idx, uint8_t value)
    {
        const index_t bit_offset  = idx * 6;
        const index_t word_idx    = bit_offset / 32;
        const index_t bit_in_word = bit_offset % 32;

        value &= 0b00111111; // Ensure only 6 bits

        if(bit_in_word <= 26)
        {
            // Value fits entirely in one word
            uint32_t mask = 0b00111111u << bit_in_word;
            data[word_idx] =
                (data[word_idx] & ~mask) | (static_cast<uint32_t>(value) << bit_in_word);
        }
        else
        {
            // Value spans two words
            const index_t bits_in_first  = 32 - bit_in_word;
            const index_t bits_in_second = 6 - bits_in_first;

            uint32_t mask1 = ((1u << bits_in_first) - 1) << bit_in_word;
            uint32_t mask2 = (1u << bits_in_second) - 1;

            data[word_idx] =
                (data[word_idx] & ~mask1) | ((value & ((1u << bits_in_first) - 1)) << bit_in_word);
            data[word_idx + 1] = (data[word_idx + 1] & ~mask2) | ((value >> bits_in_first) & mask2);
        }
    }
};

using pk_fp6_t         = pk_float6_e2m3_t;
using pk_bf6_t         = pk_float6_e3m2_t;
using pk_fp6_raw_t     = typename pk_fp6_t::raw_type;
using pk_bf6_raw_t     = typename pk_bf6_t::raw_type;
using pk_fp6_storage_t = typename pk_fp6_t::storage_type;
using pk_bf6_storage_t = typename pk_bf6_t::storage_type;
// Numeric traits for FP6 E2M3
template <>
struct numeric_traits<pk_fp6_t>
{
    using bitwise_type = uint8_t; // uint8_t for single element operations

    static constexpr int exp        = 2;
    static constexpr int mant       = 3;
    static constexpr int bias       = 1;
    static constexpr int PackedSize = 16; // 16 values packed in 96 bits
};

// Numeric traits for BF6 E3M2
template <>
struct numeric_traits<pk_bf6_t>
{
    using bitwise_type = uint8_t; // uint8_t for single element operations

    static constexpr int exp        = 3;
    static constexpr int mant       = 2;
    static constexpr int bias       = 3;
    static constexpr int PackedSize = 16; // 16 values packed in 96 bits
};

// Numeric limits for FP6 E2M3
template <>
struct numeric<pk_fp6_t>
{
    // E2M3: sign(1) + exp(2) + mant(3) = 6 bits
    // bias = 1
    // Values: exp=00,mant=000 -> 0
    //         exp=00,mant=111 -> 0.875 (subnormal max)
    //         exp=01,mant=000 -> 1.0 (normal min)
    //         exp=11,mant=111 -> 7.0 (normal max)

    static constexpr uint8_t binary_min_normal    = 0b001000; // 1.0
    static constexpr uint8_t binary_max_normal    = 0b011111; // 7.5
    static constexpr uint8_t binary_lowest_normal = 0b111111; // -7.5
    static constexpr uint8_t binary_min_subnorm   = 0b000001; // 0.125
    static constexpr uint8_t binary_max_subnorm   = 0b000111; // largest subnormal
    static constexpr uint8_t binary_zero          = 0b000000; // 0

    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t min()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_min_normal);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t max()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_max_normal);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t lowest()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_lowest_normal);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t epsilon()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_min_subnorm);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t round_error()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_min_subnorm);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t zero()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_zero);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t denorm_min()
    {
        pk_fp6_t ret;
        ret.set_element(0, binary_min_subnorm);
        return ret;
    }

    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t infinity() { return max(); }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t quiet_NaN() { return max(); }
    CK_TILE_HOST_DEVICE static constexpr pk_fp6_t signaling_NaN() { return max(); }
};

// Numeric limits for BF6 E3M2
template <>
struct numeric<pk_bf6_t>
{
    // E3M2: sign(1) + exp(3) + mant(2) = 6 bits
    // bias = 3
    // Value layout (positive):
    //   exp=000,mant=00 -> 0 (zero)
    //   exp=000,mant=01 -> smallest positive subnormal
    //   exp=000,mant=11 -> largest positive subnormal (≈ 0.0625)
    //   exp=001,mant=00 -> smallest positive normal (≈ 0.25)
    //   exp=111,mant=11 -> largest positive normal (≈ 28.0)

    static constexpr uint8_t binary_min_normal    = 0b000100; // smallest positive normal (≈ 0.25)
    static constexpr uint8_t binary_max_normal    = 0b011111; // largest positive normal (≈ 28.0)
    static constexpr uint8_t binary_lowest_normal = 0b111111; // most negative normal (≈ -28.0)
    static constexpr uint8_t binary_min_subnorm   = 0b000001; // smallest positive subnormal
    static constexpr uint8_t binary_max_subnorm = 0b000011; // largest positive subnormal (≈ 0.0625)
    static constexpr uint8_t binary_zero = 0b000000;        // zero

    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t min()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_min_normal);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t max()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_max_normal);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t lowest()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_lowest_normal);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t epsilon()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_min_subnorm);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t round_error()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_min_subnorm);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t zero()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_zero);
        return ret;
    }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t denorm_min()
    {
        pk_bf6_t ret;
        ret.set_element(0, binary_min_subnorm);
        return ret;
    }

    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t infinity() { return max(); }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t quiet_NaN() { return max(); }
    CK_TILE_HOST_DEVICE static constexpr pk_bf6_t signaling_NaN() { return max(); }
};

#if CK_TILE_FP6_CVT_DEVICE
namespace impl {
#if defined(__gfx125__)
// Device conversion functions for FP6 E2M3 with pkscale and Opsel
template <typename T, int Opsel>
CK_TILE_DEVICE T _from_fp6x16_pkscale(pk_fp6_storage_t src, uint32_t scale)
{
    if constexpr(std::is_same_v<T, fp32x16_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk16_f32_fp6(src, scale, Opsel);
    }
    else if constexpr(std::is_same_v<T, fp16x16_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk16_f16_fp6(src, scale, Opsel);
    }
    else if constexpr(std::is_same_v<T, bf16x16_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk16_bf16_fp6(src, scale, Opsel);
    }
    else
    {
        static_assert(false_type::value, "Unsupported type.");
    }
}

template <typename T, bool stochastic_rounding = false>
CK_TILE_DEVICE uint32x3_t _to_fp6_pk16(T src, float scale = 1.0f)
{
    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        auto thread_gid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (thread_gid + 1));

        if constexpr(std::is_same_v<T, fp32x16_t>)
            return __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f32(src, rng, scale);
        else if constexpr(std::is_same_v<T, fp16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f16(src, rng, scale);
        else if constexpr(std::is_same_v<T, bf16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_bf16(src, rng, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<T, fp32x16_t>)
            return __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32(src, scale);
        else if constexpr(std::is_same_v<T, fp16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_pk16_fp6_f16(src, scale);
        else if constexpr(std::is_same_v<T, bf16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16(src, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
}

// Overload for scalar and small vectors (size 1 or 16)
template <typename T,
          bool stochastic_rounding            = false,
          typename std::enable_if<vector_traits<T>::vector_size == 1 ||
                                      vector_traits<T>::vector_size == 16,
                                  bool>::type = true>
CK_TILE_DEVICE pk_fp6_raw_t _to_pk_fp6(T src, float scale = 1.0f)
{
    const int N = vector_traits<T>::vector_size;
    using BaseT = typename vector_traits<T>::scalar_type;
    using T16   = ext_vector_t<BaseT, 16>;
    union
    {
        uint32x3_t u32x3;
        pk_fp6_raw_t pf6[16];
    } cvt;

    if constexpr(N == 1)
        cvt.u32x3 = _to_fp6_pk16<T16, stochastic_rounding>(T16(src), scale);
    else if constexpr(N == 16)
        cvt.u32x3 = _to_fp6_pk16<T, stochastic_rounding>(src, scale);
    else
        static_assert(false_type::value, "Unsupported type.");

    return cvt.pf6[0];
}

// Device conversion functions for BF6 E3M2 with pkscale and Opsel
template <typename T, int Opsel>
CK_TILE_DEVICE T _from_bf6x16_pkscale(pk_bf6_storage_t src, uint32_t scale)
{
    if constexpr(std::is_same_v<T, fp32x16_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk16_f32_bf6(src, scale, Opsel);
    }
    else if constexpr(std::is_same_v<T, fp16x16_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk16_f16_bf6(src, scale, Opsel);
    }
    else if constexpr(std::is_same_v<T, bf16x16_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk16_bf16_bf6(src, scale, Opsel);
    }
    else
    {
        static_assert(false_type::value, "Unsupported type.");
    }
}

template <typename T, bool stochastic_rounding = false>
CK_TILE_DEVICE uint32x3_t _to_bf6_pk16(T src, float scale = 1.0f)
{
    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        auto thread_gid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (thread_gid + 1));

        if constexpr(std::is_same_v<T, fp32x16_t>)
            return __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f32(src, rng, scale);
        else if constexpr(std::is_same_v<T, fp16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f16(src, rng, scale);
        else if constexpr(std::is_same_v<T, bf16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_bf16(src, rng, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<T, fp32x16_t>)
            return __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32(src, scale);
        else if constexpr(std::is_same_v<T, fp16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_pk16_bf6_f16(src, scale);
        else if constexpr(std::is_same_v<T, bf16x16_t>)
            return __builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16(src, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
}

// Overload for scalar and small vectors (size 1 or 16)
template <typename T,
          bool stochastic_rounding            = false,
          typename std::enable_if<vector_traits<T>::vector_size == 1 ||
                                      vector_traits<T>::vector_size == 16,
                                  bool>::type = true>
CK_TILE_DEVICE pk_bf6_raw_t _to_pk_bf6(T src, float scale = 1.0f)
{
    const int N = vector_traits<T>::vector_size;
    using BaseT = typename vector_traits<T>::scalar_type;
    using T16   = ext_vector_t<BaseT, 16>;
    union
    {
        uint32x3_t u32x3;
        pk_bf6_raw_t pbf6[16];
    } cvt;

    if constexpr(N == 1)
        cvt.u32x3 = _to_bf6_pk16<T16, stochastic_rounding>(T16(src), scale);
    else if constexpr(N == 16)
        cvt.u32x3 = _to_bf6_pk16<T, stochastic_rounding>(src, scale);
    else
        static_assert(false_type::value, "Unsupported type.");

    return cvt.pbf6[0];
}
#endif
} // namespace impl
#endif

// Conversion functions: FP6 E2M3
CK_TILE_HOST_DEVICE constexpr float pk_fp6_t::to_float(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return detail::lane(impl::_from_fp6x16_pkscale<fp32x16_t, 0>(data, pkscale.data()), 0);
#else
    pk_fp6_raw_t val = unpack(number<0>{});
    return convert_to_float<pk_fp6_t>(val, scale);
#endif
}

#if !CK_TILE_AVX512F_WA
CK_TILE_HOST_DEVICE constexpr fp32x16_t pk_fp6_t::to_fp32x16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_fp6x16_pkscale<fp32x16_t, 0>(data, pkscale.data());
#else
    fp32x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = convert_to_float<pk_fp6_t>(get_element(i), scale);
    }
    return result;
#endif
}
#endif

CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp6_t::to_fp16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return detail::lane(impl::_from_fp6x16_pkscale<fp16x16_t, 0>(data, pkscale.data()), 0);
#else
    return fp16_t{type_convert<fp16_t>(convert_to_float<pk_fp6_t>(unpack(number<0>{}), scale))};
#endif
}

CK_TILE_HOST_DEVICE constexpr fp16x16_t pk_fp6_t::to_fp16x16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_fp6x16_pkscale<fp16x16_t, 0>(data, pkscale.data());
#else
    fp16x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = type_convert<fp16_t>(convert_to_float<pk_fp6_t>(get_element(i), scale));
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr bf16_t pk_fp6_t::to_bf16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return detail::lane(impl::_from_fp6x16_pkscale<bf16x16_t, 0>(data, pkscale.data()), 0);
#else
    return bf16_t{type_convert<bf16_t>(convert_to_float<pk_fp6_t>(unpack(number<0>{}), scale))};
#endif
}

CK_TILE_HOST_DEVICE constexpr bf16x16_t pk_fp6_t::to_bf16x16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_fp6x16_pkscale<bf16x16_t, 0>(data, pkscale.data());
#else
    bf16x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = type_convert<bf16_t>(convert_to_float<pk_fp6_t>(get_element(i), scale));
    }
    return result;
#endif
}

// Conversion functions: BF6 E3M2
CK_TILE_HOST_DEVICE constexpr float pk_bf6_t::to_float(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return detail::lane(impl::_from_bf6x16_pkscale<fp32x16_t, 0>(data, pkscale.data()), 0);
#else
    uint8_t val = unpack(number<0>{});
    return convert_to_float<pk_bf6_t>(val, scale);
#endif
}

#if !CK_TILE_AVX512F_WA
CK_TILE_HOST_DEVICE constexpr fp32x16_t pk_bf6_t::to_fp32x16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_bf6x16_pkscale<fp32x16_t, 0>(data, pkscale.data());
#else
    fp32x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = convert_to_float<pk_bf6_t>(get_element(i), scale);
    }
    return result;
#endif
}
#endif

CK_TILE_HOST_DEVICE constexpr fp16_t pk_bf6_t::to_fp16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return detail::lane(impl::_from_bf6x16_pkscale<fp16x16_t, 0>(data, pkscale.data()), 0);
#else
    return fp16_t{type_convert<fp16_t>(convert_to_float<pk_bf6_t>(unpack(number<0>{}), scale))};
#endif
}

CK_TILE_HOST_DEVICE constexpr fp16x16_t pk_bf6_t::to_fp16x16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_bf6x16_pkscale<fp16x16_t, 0>(data, pkscale.data());
#else
    fp16x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = type_convert<fp16_t>(convert_to_float<pk_bf6_t>(get_element(i), scale));
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr bf16_t pk_bf6_t::to_bf16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return detail::lane(impl::_from_bf6x16_pkscale<bf16x16_t, 0>(data, pkscale.data()), 0);
#else
    return bf16_t{type_convert<bf16_t>(convert_to_float<pk_bf6_t>(unpack(number<0>{}), scale))};
#endif
}

CK_TILE_HOST_DEVICE constexpr bf16x16_t pk_bf6_t::to_bf16x16(float scale) const
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_bf6x16_pkscale<bf16x16_t, 0>(data, pkscale.data());
#else
    bf16x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = type_convert<bf16_t>(convert_to_float<pk_bf6_t>(get_element(i), scale));
    }
    return result;
#endif
}

// Conversion to FP6/BF6
CK_TILE_HOST_DEVICE constexpr uint8_t float_to_fp6(float x, float scale)
{
    return bit_cast<uint8_t>(convert_to_type<pk_fp6_t>(x, scale));
}

CK_TILE_HOST_DEVICE constexpr uint8_t float_to_bf6(float x, float scale)
{
    return bit_cast<uint8_t>(convert_to_type<pk_bf6_t>(x, scale));
}

CK_TILE_HOST_DEVICE constexpr pk_float6_e2m3_t::pk_float6_e2m3_t(uint32x3_t init) : data{}
{
    data = init;
}

CK_TILE_HOST_DEVICE constexpr pk_float6_e2m3_t::pk_float6_e2m3_t(uint8_t init) : data{}
{
    set_element(0, init);
}

CK_TILE_HOST_DEVICE pk_float6_e2m3_t::pk_float6_e2m3_t(float init, float scale) : data{}
{
    auto res = bit_cast<uint8_t>(convert_to_type<pk_fp6_t>(init, scale));
    for(index_t i = 0; i < 16; ++i)
    {
        set_element(i, res);
    }
}

CK_TILE_HOST_DEVICE constexpr pk_float6_e3m2_t::pk_float6_e3m2_t(uint32x3_t init) : data{}
{
    data = init;
}

CK_TILE_HOST_DEVICE constexpr pk_float6_e3m2_t::pk_float6_e3m2_t(uint8_t init) : data{}
{
    set_element(0, init);
}

CK_TILE_HOST_DEVICE pk_float6_e3m2_t::pk_float6_e3m2_t(float init, float scale) : data{}
{
    auto res = bit_cast<uint8_t>(convert_to_type<pk_bf6_t>(init, scale));
    for(index_t i = 0; i < 16; ++i)
    {
        set_element(i, res);
    }
}

// Conversion between FP6/BF6 and FP32/FP16/BF16
// ====== FP32 Conversions ======
// PK_FP6/PK_BF6 <-> FP32 Array
#if CK_TILE_AVX512F_WA
// Workaround for host CPU without AVX-512F support. Using fp32x8_t x2 insead of fp32x16_t
CK_TILE_HOST constexpr pk_fp6_t fp32x8x2_to_pk_fp6(const fp32x8_t x[2], float scale)
{
    pk_fp6_t result{};
    for(index_t i = 0; i < 8; ++i)
    {
        result.set_element(i, float_to_fp6(x[0][i], scale));
        result.set_element(i + 8, float_to_fp6(x[1][i], scale));
    }
    return result;
}

CK_TILE_HOST constexpr pk_bf6_t fp32x8x2_to_pk_bf6(const fp32x8_t x[2], float scale)
{
    pk_bf6_t result{};
    for(index_t i = 0; i < 8; ++i)
    {
        result.set_element(i, float_to_bf6(x[0][i], scale));
        result.set_element(i + 8, float_to_bf6(x[1][i], scale));
    }
    return result;
}

CK_TILE_HOST_DEVICE constexpr void
pk_fp6_to_fp32x8(const pk_fp6_t& x, float scale, fp32x8_t (&result)[2])
{
    for(index_t i = 0; i < 8; ++i)
    {
        result[0][i] = convert_to_float<pk_fp6_t>(x.get_element(i), scale);
        result[1][i] = convert_to_float<pk_fp6_t>(x.get_element(i + 8), scale);
    }
}

CK_TILE_HOST_DEVICE constexpr void
pk_bf6_to_fp32x8(const pk_bf6_t& x, float scale, fp32x8_t (&result)[2])
{
    for(index_t i = 0; i < 8; ++i)
    {
        result[0][i] = convert_to_float<pk_bf6_t>(x.get_element(i), scale);
        result[1][i] = convert_to_float<pk_bf6_t>(x.get_element(i + 8), scale);
    }
}
#else
CK_TILE_HOST_DEVICE constexpr pk_fp6_t fp32x16_to_pk_fp6(const fp32x16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_fp6_t(impl::_to_fp6_pk16(x, scale));
#else
    pk_fp6_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result.set_element(i, float_to_fp6(x[i], scale));
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr pk_bf6_t fp32x16_to_pk_bf6(const fp32x16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_bf6_t(impl::_to_bf6_pk16(x, scale));
#else
    pk_bf6_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result.set_element(i, float_to_bf6(x[i], scale));
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr fp32x16_t pk_fp6_to_fp32x16(const pk_fp6_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_fp6x16_pkscale<fp32x16_t, 0>(x.data, pkscale.data());
#else
    fp32x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = convert_to_float<pk_fp6_t>(x.get_element(i), scale);
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr fp32x16_t pk_bf6_to_fp32x16(const pk_bf6_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::_from_bf6x16_pkscale<fp32x16_t, 0>(x.data, pkscale.data());
#else
    fp32x16_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result[i] = convert_to_float<pk_bf6_t>(x.get_element(i), scale);
    }
    return result;
#endif
}
#endif

// PK_FP6/PK_BF6 <-> FP32
CK_TILE_HOST_DEVICE constexpr float pk_fp6_to_float(const pk_fp6_t& x, float scale)
{
    return x.to_float(scale);
}

CK_TILE_HOST_DEVICE constexpr float pk_bf6_to_float(const pk_bf6_t& x, float scale)
{
    return x.to_float(scale);
}

CK_TILE_HOST_DEVICE constexpr pk_fp6_t float_to_pk_fp6(const float& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_fp6_t(impl::_to_pk_fp6(x, scale));
#else
    return pk_fp6_t(convert_to_type<pk_fp6_t>(x, scale));
#endif
}

CK_TILE_HOST_DEVICE constexpr pk_bf6_t float_to_pk_bf6(const float& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return impl::_to_pk_bf6(x, scale);
#else
    return pk_bf6_t(convert_to_type<pk_bf6_t>(x, scale));
#endif
}

// ====== FP16 Conversions ======
// PK_FP6/PK_BF6 <-> FP16
CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp6_to_fp16(const pk_fp6_t& x, float scale)
{
    return x.to_fp16(scale);
}

CK_TILE_HOST_DEVICE constexpr fp16_t pk_bf6_to_fp16(const pk_bf6_t& x, float scale)
{
    return x.to_fp16(scale);
}

CK_TILE_HOST_DEVICE constexpr pk_fp6_t fp16_to_pk_fp6(const fp16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return impl::_to_pk_fp6(x, scale);
#else
    return pk_fp6_t(convert_to_type<pk_fp6_t>(type_convert<float>(x), scale));
#endif
}

CK_TILE_HOST_DEVICE constexpr pk_bf6_t fp16_to_pk_bf6(const bf16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return impl::_to_pk_bf6(x, scale);
#else
    return pk_bf6_t(convert_to_type<pk_bf6_t>(type_convert<float>(x), scale));
#endif
}

// PK_FP6/PK_BF6 -> FP16x16
CK_TILE_HOST_DEVICE constexpr fp16x16_t pk_fp6_to_fp16x16(const pk_fp6_t& x, float scale)
{
    return x.to_fp16x16(scale);
}

CK_TILE_HOST_DEVICE constexpr fp16x16_t pk_bf6_to_fp16x16(const pk_bf6_t& x, float scale)
{
    return x.to_fp16x16(scale);
}

CK_TILE_HOST_DEVICE constexpr pk_fp6_t fp16x16_to_pk_fp6(const fp16x16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_fp6_t(impl::_to_fp6_pk16(x, scale));
#else
    pk_fp6_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result.set_element(i, float_to_fp6(detail::lane(x, i), scale));
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr pk_bf6_t fp16x16_to_pk_bf6(const fp16x16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_bf6_t(impl::_to_bf6_pk16(x, scale));
#else
    pk_bf6_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result.set_element(i, float_to_bf6(detail::lane(x, i), scale));
    }
    return result;
#endif
}

// ====== BF16x16 Conversions ======
// PK_FP6/PK_BF6 <-> BF16
CK_TILE_HOST_DEVICE constexpr bf16_t pk_fp6_to_bf16(const pk_fp6_t& x, float scale)
{
    return x.to_bf16(scale);
}

CK_TILE_HOST_DEVICE constexpr bf16_t pk_bf6_to_bf16(const pk_bf6_t& x, float scale)
{
    return x.to_bf16(scale);
}

CK_TILE_HOST_DEVICE constexpr pk_fp6_t bf16_to_pk_fp6(const bf16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_fp6_t(impl::_to_pk_fp6(x, scale));
#else
    return pk_fp6_t(convert_to_type<pk_fp6_t>(type_convert<float>(x), scale));
#endif
}

CK_TILE_HOST_DEVICE constexpr pk_bf6_t bf16_to_pk_bf6(const bf16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_bf6_t(impl::_to_pk_bf6(x, scale));
#else
    return pk_bf6_t(convert_to_type<pk_bf6_t>(type_convert<float>(x), scale));
#endif
}

// PK_FP6/PK_BF6 -> BF16x16
CK_TILE_HOST_DEVICE constexpr bf16x16_t pk_fp6_to_bf16x16(const pk_fp6_t& x, float scale)
{
    return x.to_bf16x16(scale);
}

CK_TILE_HOST_DEVICE constexpr bf16x16_t pk_bf6_to_bf16x16(const pk_bf6_t& x, float scale)
{
    return x.to_bf16x16(scale);
}

CK_TILE_HOST_DEVICE constexpr pk_fp6_t bf16x16_to_pk_fp6(const bf16x16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_fp6_t(impl::_to_fp6_pk16(x, scale));
#else
    pk_fp6_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result.set_element(i, float_to_fp6(detail::lane(x, i), scale));
    }
    return result;
#endif
}

CK_TILE_HOST_DEVICE constexpr pk_bf6_t bf16x16_to_pk_bf6(const bf16x16_t& x, float scale)
{
#if CK_TILE_FP6_CVT_DEVICE
    return pk_bf6_t(impl::_to_bf6_pk16(x, scale));
#else
    pk_bf6_t result{};
    for(index_t i = 0; i < 16; ++i)
    {
        result.set_element(i, float_to_bf6(detail::lane(x, i), scale));
    }
    return result;
#endif
}

#if CK_TILE_AVX512F_WA
// Overloaded wrapper functions for fp32x8_t[2] conversions (for non-AVX512 hosts)
// Use wrapper approach since template specialization doesn't work well with array types
template <typename T>
CK_TILE_HOST constexpr std::enable_if_t<std::is_same_v<T, pk_fp6_t> || std::is_same_v<T, pk_bf6_t>,
                                        T>
scaled_type_convert(const fp32x8_t (&x)[2], float scale)
{
    if constexpr(std::is_same_v<T, pk_fp6_t>)
        return fp32x8x2_to_pk_fp6(x, scale);
    else if constexpr(std::is_same_v<T, pk_bf6_t>)
        return fp32x8x2_to_pk_bf6(x, scale);
}

template <typename T>
CK_TILE_HOST constexpr std::enable_if_t<std::is_same_v<T, pk_fp6_t> || std::is_same_v<T, pk_bf6_t>,
                                        T>
type_convert(const fp32x8_t (&x)[2])
{
    return scaled_type_convert<T>(x, 1.f);
}

template <typename T, typename S>
CK_TILE_HOST constexpr std::enable_if_t<
    std::is_same_v<T, fp32x8_t[2]> && (std::is_same_v<S, pk_fp6_t> || std::is_same_v<S, pk_bf6_t>),
    void>
scaled_type_convert(const S& x, float scale, fp32x8_t (&result)[2])
{
    if constexpr(std::is_same_v<S, pk_fp6_t>)
        pk_fp6_to_fp32x8(x, scale, result);
    else if constexpr(std::is_same_v<S, pk_bf6_t>)
        pk_bf6_to_fp32x8(x, scale, result);
}

template <typename T, typename S>
CK_TILE_HOST constexpr std::enable_if_t<
    std::is_same_v<T, fp32x8_t[2]> && (std::is_same_v<S, pk_fp6_t> || std::is_same_v<S, pk_bf6_t>),
    void>
type_convert(const S& x, fp32x8_t (&result)[2])
{
    scaled_type_convert<T>(x, 1.f, result);
}
#endif

enum class f6_kind
{
    fp6,
    bf6
};

// Generic packed type for fp6 and bf6.
template <index_t pk_size, f6_kind kind>
struct pk_f6_legacy_t
{
    static constexpr index_t num_bits_elem = 6;
    using element_type                     = int32_t; // element storage fundamental type
    static constexpr index_t packed_size   = pk_size;
    static constexpr index_t num_bits_vec_elem =
        sizeof(element_type) * 8; // 32-bit uint for storage
    static_assert((packed_size * num_bits_elem) % num_bits_vec_elem == 0,
                  "Packed elements must fit exactly into the element storage.");
    static constexpr index_t vector_size = (packed_size * num_bits_elem) / num_bits_vec_elem;
    element_type data_[vector_size]; // packed data
    using type = pk_f6_legacy_t<packed_size, kind>;

    CK_TILE_HOST_DEVICE constexpr pk_f6_legacy_t() : data_{element_type{}} {}

    CK_TILE_HOST_DEVICE constexpr explicit pk_f6_legacy_t(int value)
    {
        for(size_t i = 0; i < vector_size; ++i)
        {
            data_[i] = value;
        }
    }

    CK_TILE_HOST_DEVICE void pack(const int32_t x, const index_t i)
    {
        int32_t bits         = static_cast<int32_t>(x) & 0x3F;
        const int bit_pos    = i * num_bits_elem;
        const int arr_index  = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;
        int32_t old_value    = data_[arr_index];

        // insert bits into the current 32-bit block
        old_value |= (bits << bit_offset);
        data_[arr_index] = old_value;

        // if it crosses into the next block, shift the remainder
        if(overhang > 0 && (arr_index + 1) < vector_size)
        {
            int32_t next_value = data_[arr_index + 1];
            next_value |= (bits >> (num_bits_elem - overhang));
            data_[arr_index + 1] = next_value;
        }
    }

    template <typename T>
    CK_TILE_HOST_DEVICE static int32_t unpack(const T& pk, const index_t i)
    {
        const int bit_pos    = i * num_bits_elem;
        const int arr_idx    = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;

        // Cast through uint32_t before shifting: data_[arr_idx] is int32_t, and a
        // negative value (high bit set) would otherwise sign-extend on right shift,
        // corrupting any element that places its low bits in the upper region of a
        // word (e.g. fp6 1.0 = 0x08 at idx=10 puts bit 31 of data_[1] high; an
        // arithmetic >> 28 returns 0xFFFFFFF8 instead of 0x8 -> read back as -4).
        int32_t bits = static_cast<int32_t>(static_cast<uint32_t>(pk.data_[arr_idx]) >> bit_offset);
        if(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (pk.data_[arr_idx + 1] & ((1u << overhang) - 1)) << (num_bits_elem - overhang);
        }

        return bits & 0x3F;
    }

    CK_TILE_HOST_DEVICE int32_t unpack(const index_t i) const { return unpack(*this, i); }

    CK_TILE_HOST_DEVICE int32_t operator[](index_t i) const { return data_[i]; }

    // Element-wise comparison of the packed storage. Two packed vectors are equal
    // iff their underlying int32 storage arrays are bit-identical.
    CK_TILE_HOST_DEVICE friend constexpr bool operator==(const pk_f6_legacy_t& x,
                                                         const pk_f6_legacy_t& y)
    {
        for(index_t i = 0; i < vector_size; ++i)
        {
            if(x.data_[i] != y.data_[i])
            {
                return false;
            }
        }
        return true;
    }

    CK_TILE_HOST_DEVICE friend constexpr bool operator!=(const pk_f6_legacy_t& x,
                                                         const pk_f6_legacy_t& y)
    {
        return !(x == y);
    }

    CK_TILE_HOST_DEVICE static float fp6_e2m3_to_float(int32_t fp6_bits)
    {
        fp6_bits = fp6_bits & 0x3F;

        uint32_t sign     = (fp6_bits >> 5) & 0x1; // bit 5
        uint32_t exponent = (fp6_bits >> 3) & 0x3; // bits 4-3
        uint32_t mantissa = fp6_bits & 0x7;        // bits 2-0

        float result;
        if(exponent == 0 && mantissa == 0)
        {
            result = 0.f;
        }
        else if(exponent != 0)
        {
            result               = std::exp2f(static_cast<int>(exponent) - 1);
            float mantissa_value = 1.0f + mantissa / 8.0f;
            result *= mantissa_value;
        }
        else
        {
            result = mantissa / 8.0f;
        }
        return sign == 1 ? -1 * result : result;
    }
};

using pk_fp6x16_t = pk_f6_legacy_t<16, f6_kind::fp6>;
using pk_fp6x32_t = pk_f6_legacy_t<32, f6_kind::fp6>;
using pk_bf6x16_t = pk_f6_legacy_t<16, f6_kind::bf6>;
using pk_bf6x32_t = pk_f6_legacy_t<32, f6_kind::bf6>;

template <int N, f6_kind kind>
struct numeric_traits<pk_f6_legacy_t<N, kind>>
{
    static constexpr int PackedSize = N;
};

template <int N, f6_kind kind>
struct f6x16xN_tt // Underlying type for ext_vector_t<fp6x16, N> and ext_vector_t<bf6x16, N> impls
{
    int32_t data[N * 3];

    // Proxy reference structure is necessary for setting values by subscript.
    struct reference
    {
        f6x16xN_tt& vec;
        index_t i;

        CK_TILE_HOST_DEVICE reference& operator=(const pk_f6_legacy_t<16, kind>& val)
        {
            vec.data[i * 3]     = val[0];
            vec.data[i * 3 + 1] = val[1];
            vec.data[i * 3 + 2] = val[2];
            return *this;
        }

        CK_TILE_HOST_DEVICE operator pk_f6_legacy_t<16, kind>() const
        {
            pk_f6_legacy_t<16, kind> result;
            result.data_[0] = vec.data[i * 3];
            result.data_[1] = vec.data[i * 3 + 1];
            result.data_[2] = vec.data[i * 3 + 2];
            return result;
        }
    };

    CK_TILE_HOST_DEVICE reference operator[](index_t i) { return reference{*this, i}; }

    CK_TILE_HOST_DEVICE pk_f6_legacy_t<16, kind> operator[](index_t i) const
    {
        pk_f6_legacy_t<16, kind> result;
        result.data_[0] = data[i * 3];
        result.data_[1] = data[i * 3 + 1];
        result.data_[2] = data[i * 3 + 2];
        return result;
    }
};

template <int N, f6_kind kind>
struct vector_traits<f6x16xN_tt<N, kind>>
{
    using scalar_type                    = pk_f6_legacy_t<16, kind>;
    static constexpr index_t vector_size = N;
};

template <>
struct impl::ext_vector<pk_fp6x16_t, 1>
{
    static constexpr index_t N = 1;
    using value_type           = f6x16xN_tt<1, f6_kind::fp6>;
    using type                 = f6x16xN_tt<1, f6_kind::fp6>;
};

template <>
struct impl::ext_vector<pk_fp6x16_t, 2>
{
    static constexpr index_t N = 2;
    using value_type           = f6x16xN_tt<2, f6_kind::fp6>;
    using type                 = f6x16xN_tt<2, f6_kind::fp6>;
};

template <>
struct impl::ext_vector<pk_bf6x16_t, 1>
{
    static constexpr index_t N = 1;
    using value_type           = f6x16xN_tt<1, f6_kind::bf6>;
    using type                 = f6x16xN_tt<1, f6_kind::bf6>;
};

template <>
struct impl::ext_vector<pk_bf6x16_t, 2>
{
    static constexpr index_t N = 2;
    using value_type           = f6x16xN_tt<2, f6_kind::bf6>;
    using type                 = f6x16xN_tt<2, f6_kind::bf6>;
};

// Used as AVecType / BVecType for the gfx1250 16x16x128 mx-scale wmma kernel
template <>
struct impl::ext_vector<pk_fp6x16_t, 4>
{
    static constexpr index_t N = 4;
    using value_type           = f6x16xN_tt<4, f6_kind::fp6>;
    using type                 = f6x16xN_tt<4, f6_kind::fp6>;
};

// Arithmetic operations using float conversion
// Note: Arithmetic operations on packed types containing 32 elements
// may not be semantically meaningful for element-wise operations
// CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, pk_fp6_t)
// CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, pk_bf6_t)

} // namespace ck_tile
