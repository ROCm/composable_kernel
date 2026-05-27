// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"
#include "ck_tile/core/numeric/mxfp_scale.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
#if defined(__gfx950__) || defined(__gfx125__)
#define CK_TILE_FP4_CVT_DEVICE 1
#else
#define CK_TILE_FP4_CVT_DEVICE 0
#endif

#define TEST_convert_with_table 0

namespace ck_tile {

using fp32_t   = float;
using fp32x2_t = float __attribute__((ext_vector_type(2)));
using fp16x2_t = _Float16 __attribute__((ext_vector_type(2)));
using bf16x2_t = bfloat16_t __attribute__((ext_vector_type(2)));

#if CK_TILE_USE_CUSTOM_DATA_TYPE
using fp8x2_t = fp8_raw_t __attribute__((ext_vector_type(2)));
#else
using fp8x2_t = fp8_t __attribute__((ext_vector_type(2)));
#endif

// Helpers: constexpr-safe access to elements of ext_vector_type(2)
// Some compilers don't allow operator[] in constant expressions for vector types.
// We use bit_cast to a trivially copyable representation to extract lanes.
namespace detail {
template <int idx, typename VecT>
CK_TILE_HOST_DEVICE constexpr auto get_from_lane(const VecT& v)
{
    const int N = vector_traits<VecT>::vector_size;
    static_assert(idx < N);
    using scalar_t = typename vector_traits<VecT>::scalar_type;
    struct repr
    {
        scalar_t e[N];
    };
    return ck_tile::bit_cast<repr>(v).e[idx];
}
} // namespace detail

struct pk_float4_e2m1_t;
CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t float_to_pk_fp4(const float& x, float scale = 1.f);

// TODO: Add stochastic method
struct pk_float4_e2m1_t
{
    // TODO: Can we merge raw_type and type?
    using raw_type = uint8_t;
    using type     = raw_type;
    type data;
    static constexpr int packed_size = 2;

    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t() : data{type{}} {}
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t(T init) : data{static_cast<type>(init)}
    {
    }
    CK_TILE_HOST_DEVICE explicit constexpr pk_float4_e2m1_t(float init, float scale = 1.f)
        : data{float_to_pk_fp4(init, scale)}
    {
    }
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr type& get() { return data; }
    CK_TILE_HOST_DEVICE constexpr type get() const { return data; }

    CK_TILE_HOST_DEVICE constexpr float to_float(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp32x2_t to_fp32x2(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16_t to_fp16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16x2_t to_fp16x2(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16_t to_bf16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16x2_t to_bf16x2(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp8_t to_fp8(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp8x2_t to_fp8x2(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf8_t to_bf8(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf8x2_t to_bf8x2(float scale = 1.f) const;

    CK_TILE_HOST_DEVICE constexpr operator float() const { return to_float(); }
    CK_TILE_HOST_DEVICE constexpr operator fp32x2_t() const { return to_fp32x2(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16_t() const { return to_fp16(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16x2_t() const { return to_fp16x2(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16_t() const { return to_bf16(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16x2_t() const { return to_bf16x2(); }
    CK_TILE_HOST_DEVICE constexpr operator fp8_t() const { return to_fp8(); }
    CK_TILE_HOST_DEVICE constexpr operator fp8x2_t() const { return to_fp8x2(); }
    CK_TILE_HOST_DEVICE constexpr operator bf8_t() const { return to_bf8(); }
    CK_TILE_HOST_DEVICE constexpr operator bf8x2_t() const { return to_bf8x2(); }
    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t unpack(number<I>) const
    {
        return _unpack(number<I>{});
    }
    CK_TILE_HOST_DEVICE constexpr static pk_float4_e2m1_t pack(const pk_float4_e2m1_t& x0,
                                                               const pk_float4_e2m1_t& x1)
    {
        return _pack(x0.get(), x1.get());
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr type _unpack(number<I>) const;
    CK_TILE_HOST_DEVICE constexpr static type _pack(const type x0, const type x1)
    {
        return (x1 << 4) | (x0 & 0b00001111);
    }

#if TEST_convert_with_table
    static constexpr float e2m1_to_fp32_table[16] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6};
    static constexpr fp16_t e2m1_to_fp16_table[16] = {
        bit_cast<fp16_t>(static_cast<uint16_t>(0x0000)), //  0
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3800)), //  0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3C00)), //  1
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3E00)), //  1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4000)), //  2
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4200)), //  3
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4400)), //  4
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4600)), //  6
        bit_cast<fp16_t>(static_cast<uint16_t>(0x8000)), // -0
        bit_cast<fp16_t>(static_cast<uint16_t>(0xB800)), // -0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBC00)), // -1
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBE00)), // -1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC000)), // -2
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC200)), // -3
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC400)), // -4
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC600))  // -6
    };

#if CK_TILE_USE_OCP_FP8
    // FP8 EM4E3 (OCP) representation
    static constexpr fp8_t e2m1_to_fp8_table[16] = {
        fp8_t(static_cast<uint8_t>(0x00)), //  0
        fp8_t(static_cast<uint8_t>(0x30)), //  0.5
        fp8_t(static_cast<uint8_t>(0x38)), //  1
        fp8_t(static_cast<uint8_t>(0x3C)), //  1.5
        fp8_t(static_cast<uint8_t>(0x40)), //  2
        fp8_t(static_cast<uint8_t>(0x44)), //  3
        fp8_t(static_cast<uint8_t>(0x48)), //  4
        fp8_t(static_cast<uint8_t>(0x4C)), //  6
        fp8_t(static_cast<uint8_t>(0x00)), // -0
        fp8_t(static_cast<uint8_t>(0xB0)), // -0.5
        fp8_t(static_cast<uint8_t>(0xB8)), // -1
        fp8_t(static_cast<uint8_t>(0xBC)), // -1.5
        fp8_t(static_cast<uint8_t>(0xC0)), // -2
        fp8_t(static_cast<uint8_t>(0xC4)), // -3
        fp8_t(static_cast<uint8_t>(0xC8)), // -4
        fp8_t(static_cast<uint8_t>(0xCC))  // -6
    };
#else // CK_TILE_USE_FNUZ_FP8
    // FP8 E4M3 FNUZ
    static constexpr fp8_t e2m1_to_fp8_table[16] = {
        fp8_t(static_cast<uint8_t>(0x00)), //  0
        fp8_t(static_cast<uint8_t>(0x38)), //  0.5
        fp8_t(static_cast<uint8_t>(0x40)), //  1
        fp8_t(static_cast<uint8_t>(0x44)), //  1.5
        fp8_t(static_cast<uint8_t>(0x48)), //  2
        fp8_t(static_cast<uint8_t>(0x4C)), //  3
        fp8_t(static_cast<uint8_t>(0x50)), //  4
        fp8_t(static_cast<uint8_t>(0x54)), //  6
        fp8_t(static_cast<uint8_t>(0x00)), // -0
        fp8_t(static_cast<uint8_t>(0xB8)), // -0.5
        fp8_t(static_cast<uint8_t>(0xC0)), // -1
        fp8_t(static_cast<uint8_t>(0xC4)), // -1.5
        fp8_t(static_cast<uint8_t>(0xC4)), // -2
        fp8_t(static_cast<uint8_t>(0xCC)), // -3
        fp8_t(static_cast<uint8_t>(0xD0)), // -4
        fp8_t(static_cast<uint8_t>(0xD4))  // -6
    };
#endif

#endif
};

using pk_fp4_t     = pk_float4_e2m1_t;
using pk_fp4_raw_t = typename pk_fp4_t::type;

template <>
struct numeric_traits<pk_fp4_t>
{
    using bitwise_type = pk_fp4_raw_t;

    static constexpr int exp        = 2;
    static constexpr int mant       = 1;
    static constexpr int bias       = 1;
    static constexpr int PackedSize = 2;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<pk_fp4_t>
{
    static constexpr pk_fp4_raw_t binary_min_normal    = 0b00100010; // 1
    static constexpr pk_fp4_raw_t binary_max_normal    = 0b01110111; // 6
    static constexpr pk_fp4_raw_t binary_lowest_normal = 0b11111111; // -6
    static constexpr pk_fp4_raw_t binary_min_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_max_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_zero          = 0b00000000; // 0
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t min() { return binary_min_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t max() { return binary_max_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t lowest() { return binary_lowest_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t epsilon() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t round_error() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t zero() { return binary_zero; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t denorm_min() { return binary_min_subnorm; }

    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t infinity() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t quiet_NaN() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t signaling_NaN() { return max(); }
};

// Specialize vector_traits for pk_fp4_t to map to uint8_t scalar type
template <>
struct vector_traits<pk_fp4_t>
{
    using scalar_type                    = uint8_t;
    static constexpr index_t vector_size = 1;
};

template <index_t I>
CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t pk_fp4_t::_unpack(number<I>) const
{
    static_assert(I < 2, "Index is out of range.");
    if constexpr(I == 1)
        return (data >> 4);
    else
        return data & 0b00001111;
}
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, pk_fp4_t)
// TODO: consider replace this macro to improve performance

#if CK_TILE_FP4_CVT_DEVICE
namespace impl {
#if defined(__gfx950__)
template <typename T>
CK_TILE_DEVICE T _from_f4(pk_fp4_raw_t src, float scale = 1.0f)
{
    if constexpr(std::is_same_v<T, fp32_t>)
    {
        fp32x2_t tmp = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0);
        return detail::get_from_lane<0>(tmp);
    }
    else if constexpr(std::is_same_v<T, fp32x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0);
    else if constexpr(std::is_same_v<T, fp16_t>)
    {
        fp16x2_t tmp = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, 0);
        return detail::get_from_lane<0>(tmp);
    }
    else if constexpr(std::is_same_v<T, fp16x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, 0);
    else if constexpr(std::is_same_v<T, bf16_t>)
    {
        bf16x2_t tmp = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, 0);
        return detail::get_from_lane<0>(tmp);
    }
    else if constexpr(std::is_same_v<T, bf16x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, 0);
    else
        static_assert(false_type::value, "Unsupported type.");
    return T{};
}
template <typename T>
CK_TILE_DEVICE pk_fp4_raw_t _to_f4(T src, float scale = 1.0f)
{
    union
    {
        uint32_t u32;
        pk_fp4_raw_t pf4[4];
    } cvt{0};
    if constexpr(std::is_same_v<T, fp32_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(cvt.u32, src, src, scale, 0);
    else if constexpr(std::is_same_v<T, fp32x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            cvt.u32, detail::get_from_lane<0>(src), detail::get_from_lane<1>(src), scale, 0);
    else if constexpr(std::is_same_v<T, fp16_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(cvt.u32, fp16x2_t{src, src}, scale, 0);
    else if constexpr(std::is_same_v<T, fp16x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(cvt.u32, src, scale, 0);
    else if constexpr(std::is_same_v<T, bf16_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, bf16x2_t{src, src}, scale, 0);
    else if constexpr(std::is_same_v<T, bf16x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, src, scale, 0);
    else
        static_assert(false_type::value, "Unsupported type.");
    return cvt.pf4[0];
}
#elif defined(__gfx125__)
template <typename T, int Opsel>
CK_TILE_DEVICE T _from_f4x8_pkscale(uint32_t src, uint32_t scale)
{
    if constexpr(std::is_same_v<T, fp32x8_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk8_f32_fp4(src, scale, Opsel);
    }
    else if constexpr(std::is_same_v<T, fp16x8_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk8_f16_fp4(src, scale, Opsel);
    }
    else if constexpr(std::is_same_v<T, bf16x8_t>)
    {
        return __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(src, scale, Opsel);
    }
    else
    {
        static_assert(false_type::value, "Unsupported type.");
    }
}

template <typename T>
CK_TILE_DEVICE T _from_f4(pk_fp4_raw_t src, float scale)
{
    const int N = vector_traits<T>::vector_size;
    using BaseT = typename vector_traits<T>::scalar_type;
    using T8    = ext_vector_t<BaseT, 8>;

    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);

    T8 vec8 = _from_f4x8_pkscale<T8, 0>(static_cast<uint32_t>(src), pkscale.data());
    if constexpr(N == 1)
        return detail::get_from_lane<0>(vec8);
    else if constexpr(N == 2)
        return T{detail::get_from_lane<0>(vec8), detail::get_from_lane<1>(vec8)};
    else
        static_assert(false_type::value, "Unsupported type.");
}
template <typename T>
CK_TILE_DEVICE T _from_f4(pk_fp4x4_t src, float scale)
{
    static_assert(vector_traits<T>::vector_size == 8, "Unsupported type.");
    using BaseT = typename vector_traits<T>::scalar_type;
    using T8    = ext_vector_t<BaseT, 8>;

    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);

    return _from_f4x8_pkscale<T8, 0>(bit_cast<uint32_t>(src), pkscale.data());
}

template <typename T, bool stochastic_rounding = false>
CK_TILE_DEVICE uint32_t _to_f4_pk8(T src, float scale = 1.0f)
{
    uint32_t bitwise;
    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        auto thread_gid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (thread_gid + 1));

        if constexpr(std::is_same_v<T, fp32x8_t>)
            bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f32(src, rng, scale);
        else if constexpr(std::is_same_v<T, fp16x8_t>)
            bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f16(src, rng, scale);
        else if constexpr(std::is_same_v<T, bf16x8_t>)
            bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_bf16(src, rng, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<T, fp32x8_t>)
            bitwise = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(src, scale);
        else if constexpr(std::is_same_v<T, fp16x8_t>)
            bitwise = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16(src, scale);
        else if constexpr(std::is_same_v<T, bf16x8_t>)
            bitwise = __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16(src, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    return bitwise;
}
// Overload for scalar and small vectors (size <= 2)
template <typename T,
          bool stochastic_rounding                                                = false,
          typename std::enable_if<vector_traits<T>::vector_size <= 2, bool>::type = true>
CK_TILE_DEVICE pk_fp4_raw_t _to_f4(T src, float scale = 1.0f)
{
    const int N = vector_traits<T>::vector_size;
    using BaseT = typename vector_traits<T>::scalar_type;
    using T8    = ext_vector_t<BaseT, 8>;
    union
    {
        uint32_t u32;
        pk_fp4_raw_t pf4[4];
    } cvt{0};

    if constexpr(N == 1)
        cvt.u32 = _to_f4_pk8<T8, stochastic_rounding>(T8(src), scale);
    else if constexpr(N == 2)
        cvt.u32 = _to_f4_pk8<T8, stochastic_rounding>(
            T8{
                detail::get_from_lane<0>(src),
                detail::get_from_lane<1>(src),
                0,
                0,
                0,
                0,
                0,
                0,
            },
            scale);
    else
        static_assert(false_type::value, "Unsupported type.");

    return cvt.pf4[0];
}
// Overload for 8-element vectors
template <typename T,
          bool stochastic_rounding                                                = false,
          typename std::enable_if<vector_traits<T>::vector_size == 8, bool>::type = true>
CK_TILE_DEVICE pk_fp4x4_t _to_f4(T src, float scale = 1.0f)
{
    uint32_t result = _to_f4_pk8<T, stochastic_rounding>(src, scale);
    return bit_cast<pk_fp4x4_t>(result);
}
#endif

} // namespace impl
#endif

CK_TILE_HOST_DEVICE constexpr bf16_t pk_fp4_t::to_bf16(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<bf16_t>(data, scale);
#else
    return bf16_t{type_convert<bf16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale))};
#endif
}

CK_TILE_HOST_DEVICE constexpr bf16x2_t pk_fp4_t::to_bf16x2(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<bf16x2_t>(data, scale);
#else
    return bf16x2_t{type_convert<bf16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale)),
                    type_convert<bf16_t>(convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale))};
#endif
}

// TODO: make it generic so that we can convert from directrly.
CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t float_to_mxfp4(float x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return convert_to_type<pk_fp4_t>(x, scale);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t float_to_pk_fp4(const float& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    auto res = convert_to_type<pk_fp4_t>(x, scale);
    return pk_fp4_t::_pack(res, res);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp16_to_pk_fp4(const fp16_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    auto res = float_to_mxfp4(type_convert<float>(x), scale);
    return pk_fp4_t::_pack(res, res);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t bf16_to_pk_fp4(const bf16_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    auto res = float_to_mxfp4(type_convert<float>(x), scale);
    return pk_fp4_t::_pack(res, res);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp16x2_to_pk_fp4(const fp16x2_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return pk_fp4_t::_pack(float_to_mxfp4(detail::get_from_lane<0>(x), scale),
                           float_to_mxfp4(detail::get_from_lane<1>(x), scale));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t bf16x2_to_pk_fp4(const bf16x2_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return pk_fp4_t::_pack(float_to_mxfp4(detail::get_from_lane<0>(x), scale),
                           float_to_mxfp4(detail::get_from_lane<1>(x), scale));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp32x2_to_pk_fp4(const fp32x2_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return pk_fp4_t::_pack(float_to_mxfp4(detail::get_from_lane<0>(x), scale),
                           float_to_mxfp4(detail::get_from_lane<1>(x), scale));
#endif
}

CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_to_fp32x2(const pk_fp4_t& x, float scale)
{
    return x.to_fp32x2(scale);
}
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_to_fp16x2(const pk_fp4_t& x, float scale)
{
    return x.to_fp16x2(scale);
}
CK_TILE_HOST_DEVICE constexpr bf16x2_t pk_fp4_to_bf16x2(const pk_fp4_t& x, float scale)
{
    return x.to_bf16x2(scale);
}
CK_TILE_HOST_DEVICE constexpr float pk_fp4_to_float(const pk_fp4_t& x, float scale)
{
    return x.to_float(scale);
}
CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp4_to_fp16(const pk_fp4_t& x, float scale)
{
    return x.to_fp16(scale);
}
CK_TILE_HOST_DEVICE constexpr bf16_t pk_fp4_to_bf16(const pk_fp4_t& x, float scale)
{
    return x.to_bf16(scale);
}
CK_TILE_HOST_DEVICE constexpr pk_fp4x4_t fp32x8_to_pk_fp4(const fp32x8_t& x, float scale)
{
#if defined(__gfx125__)
    return bit_cast<pk_fp4x4_t>(impl::_to_f4(x, scale));
#else
    // Pack 8 floats into 4 pk_fp4_t values using fp32x2_to_pk_fp4
    // note: consider using get_from_lane if met compiler errors with x[]
    auto p0 = fp32x2_to_pk_fp4(fp32x2_t{x[0], x[1]}, scale);
    auto p1 = fp32x2_to_pk_fp4(fp32x2_t{x[2], x[3]}, scale);
    auto p2 = fp32x2_to_pk_fp4(fp32x2_t{x[4], x[5]}, scale);
    auto p3 = fp32x2_to_pk_fp4(fp32x2_t{x[6], x[7]}, scale);
    return pk_fp4x4_t{p0, p1, p2, p3};
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4x4_t fp16x8_to_pk_fp4(const fp16x8_t& x, float scale)
{
#if defined(__gfx125__)
    return bit_cast<pk_fp4x4_t>(impl::_to_f4(x, scale));
#else
    // Pack 8 fp16 values into 4 pk_fp4_t values using fp16x2_to_pk_fp4
    auto p0 = fp16x2_to_pk_fp4(fp16x2_t{x[0], x[1]}, scale);
    auto p1 = fp16x2_to_pk_fp4(fp16x2_t{x[2], x[3]}, scale);
    auto p2 = fp16x2_to_pk_fp4(fp16x2_t{x[4], x[5]}, scale);
    auto p3 = fp16x2_to_pk_fp4(fp16x2_t{x[6], x[7]}, scale);
    return pk_fp4x4_t{p0, p1, p2, p3};
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4x4_t bf16x8_to_pk_fp4(const bf16x8_t& x, float scale)
{
#if defined(__gfx125__)
    return bit_cast<pk_fp4x4_t>(impl::_to_f4(x, scale));
#else
    // Pack 8 bf16 values into 4 pk_fp4_t values using bf16x2_to_pk_fp4
    auto p0 = bf16x2_to_pk_fp4(bf16x2_t{x[0], x[1]}, scale);
    auto p1 = bf16x2_to_pk_fp4(bf16x2_t{x[2], x[3]}, scale);
    auto p2 = bf16x2_to_pk_fp4(bf16x2_t{x[4], x[5]}, scale);
    auto p3 = bf16x2_to_pk_fp4(bf16x2_t{x[6], x[7]}, scale);
    return pk_fp4x4_t{p0, p1, p2, p3};
#endif
}
CK_TILE_HOST_DEVICE constexpr fp32x8_t pk_fp4_to_fp32x8(const pk_fp4x4_t& x, float scale)
{
#if defined(__gfx125__)
    return impl::_from_f4<fp32x8_t>(x, scale);
#else
    auto v0 = pk_fp4_to_fp32x2(pk_fp4_t{x[0]}, scale);
    auto v1 = pk_fp4_to_fp32x2(pk_fp4_t{x[1]}, scale);
    auto v2 = pk_fp4_to_fp32x2(pk_fp4_t{x[2]}, scale);
    auto v3 = pk_fp4_to_fp32x2(pk_fp4_t{x[3]}, scale);
    return fp32x8_t{v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], v3[0], v3[1]};
#endif
}
CK_TILE_HOST_DEVICE constexpr fp16x8_t pk_fp4_to_fp16x8(const pk_fp4x4_t& x, float scale)
{
#if defined(__gfx125__)
    return impl::_from_f4<fp16x8_t>(x, scale);
#else
    auto v0 = pk_fp4_to_fp16x2(pk_fp4_t{x[0]}, scale);
    auto v1 = pk_fp4_to_fp16x2(pk_fp4_t{x[1]}, scale);
    auto v2 = pk_fp4_to_fp16x2(pk_fp4_t{x[2]}, scale);
    auto v3 = pk_fp4_to_fp16x2(pk_fp4_t{x[3]}, scale);
    return fp16x8_t{v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], v3[0], v3[1]};
#endif
}
CK_TILE_HOST_DEVICE constexpr bf16x8_t pk_fp4_to_bf16x8(const pk_fp4x4_t& x, float scale)
{
#if defined(__gfx125__)
    return impl::_from_f4<bf16x8_t>(x, scale);
#else
    auto v0 = pk_fp4_to_bf16x2(pk_fp4_t{x[0]}, scale);
    auto v1 = pk_fp4_to_bf16x2(pk_fp4_t{x[1]}, scale);
    auto v2 = pk_fp4_to_bf16x2(pk_fp4_t{x[2]}, scale);
    auto v3 = pk_fp4_to_bf16x2(pk_fp4_t{x[3]}, scale);
    return bf16x8_t{v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], v3[0], v3[1]};
#endif
}

#if TEST_convert_with_table == 0
CK_TILE_HOST_DEVICE constexpr float pk_fp4_t::to_float(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp32_t>(data, scale);
#else
    return convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale);
#endif
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_t::to_fp32x2(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp32x2_t>(data, scale);
#else
    return fp32x2_t{convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale),
                    convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale)};
#endif
}

CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp4_t::to_fp16(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp16_t>(data, scale);
#else
    return fp16_t{type_convert<fp16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale))};
#endif
}
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_t::to_fp16x2(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp16x2_t>(data, scale);
#else
    return fp16x2_t{type_convert<fp16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale)),
                    type_convert<fp16_t>(convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale))};
#endif
}
CK_TILE_HOST_DEVICE constexpr fp8_t pk_fp4_t::to_fp8(float scale) const
{
    // NOTE: No specialized fp4 to fp8 instructions are available. Unsure whether fp4 to fp16 to fp8
    // would be better than the naive implementation below
    // #if CK_TILE_FP4_CVT_DEVICE
    //    return impl::_from_f4<fp8_t>(data, scale);
    // #else
    return fp8_t{type_convert<fp8_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale))};
    // #endif
}
CK_TILE_HOST_DEVICE constexpr fp8x2_t pk_fp4_t::to_fp8x2(float scale) const
{
    // NOTE: No specialized fp4 to fp8 instructions are available. Unsure whether fp4 to fp16 to fp8
    // would be better than the naive implementation below
    // #if CK_TILE_FP4_CVT_DEVICE
    //    return impl::_from_f4<fp8x2_t>(data, scale);
    // #else
    return fp8x2_t{type_convert<fp8_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale)),
                   type_convert<fp8_t>(convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale))};
    // #endif
}
#else
CK_TILE_HOST_DEVICE constexpr float pk_fp4_t::to_float(float scale) const
{
    return e2m1_to_fp32_table[_unpack(number<0>{})] * scale;
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_t::to_fp32x2(float scale) const
{
    return fp32x2_t{e2m1_to_fp32_table[_unpack(number<0>{})] * scale,
                    e2m1_to_fp32_table[_unpack(number<1>{})] * scale};
}
CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp4_t::to_fp16(float scale) const
{
    return type_convert<float>(e2m1_to_fp16_table[_unpack(number<0>{})]) * scale;
}
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_t::to_fp16x2(float scale) const
{
    return fp16x2_t{
        type_convert<fp16_t>(type_convert<float>(e2m1_to_fp16_table[_unpack(number<0>{})]) * scale),
        type_convert<fp16_t>(type_convert<float>(e2m1_to_fp16_table[_unpack(number<1>{})]) *
                             scale)};
}
CK_TILE_HOST_DEVICE constexpr fp8_t pk_fp4_t::to_fp8(float scale) const
{
    return type_convert<float>(e2m1_to_fp8_table[_unpack(number<0>{})]) * scale;
}
CK_TILE_HOST_DEVICE constexpr fp8x2_t pk_fp4_t::to_fp8x2(float scale) const
{
    return fp8x2_t{
        type_convert<fp8_t>(type_convert<float>(e2m1_to_fp8_table[_unpack(number<0>{})]) * scale),
        type_convert<fp8_t>(type_convert<float>(e2m1_to_fp8_table[_unpack(number<1>{})]) * scale)};
}
#endif

CK_TILE_HOST_DEVICE constexpr bf8_t pk_fp4_t::to_bf8(float scale) const
{
    // NOTE: No specialized fp4 to fp8 instructions are available. Unsure whether fp4 to fp16 to fp8
    // would be better than the naive implementation below
    // #if CK_TILE_FP4_CVT_DEVICE
    //    return impl::_from_f4<fp8_t>(data, scale);
    // #else
    return bf8_t{type_convert<bf8_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale))};
    // #endif
}

CK_TILE_HOST_DEVICE constexpr bf8x2_t pk_fp4_t::to_bf8x2(float scale) const
{
    // NOTE: No specialized fp4 to fp8 instructions are available. Unsure whether fp4 to fp16 to fp8
    // would be better than the naive implementation below
    // #if CK_TILE_FP4_CVT_DEVICE
    //    return impl::_from_f4<fp8x2_t>(data, scale);
    // #else
    return bf8x2_t{type_convert<bf8_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale)),
                   type_convert<bf8_t>(convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale))};
    // #endif
}

} // namespace ck_tile
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
