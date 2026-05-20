// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>
#include <tuple>
#include <type_traits>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck_tile {

#if CK_TILE_USE_CUSTOM_DATA_TYPE
template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr remove_cvref_t<Y> type_convert(const X& x)
{
    return static_cast<Y>(x);
}
#else
// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);
    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using non_const_y = std::remove_const_t<Y>;
    using non_const_x = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<non_const_y, non_const_x>(x));
}

#define CK_TILE_TYPE_CONVERT(dtype_, dname_, stype_, sname_)                    \
    template <>                                                                 \
    CK_TILE_HOST_DEVICE constexpr dtype_ type_convert<dtype_, stype_>(stype_ x) \
    {                                                                           \
        return sname_##_to_##dname_(x);                                         \
    }

CK_TILE_TYPE_CONVERT(float, float, fp16_t, fp16)
CK_TILE_TYPE_CONVERT(float, float, bf16_t, bf16)
CK_TILE_TYPE_CONVERT(float, float, fp8_t, fp8)
CK_TILE_TYPE_CONVERT(float, float, bf8_t, bf8)

static constexpr uint32_t float32_exponent_mask = 0x7f800000u;

enum class tf32_rounding_mode
{
    trunc = 0, // truncate
    rne   = 1, // round to nearest even (RTNE)
};

template <tf32_rounding_mode rounding = tf32_rounding_mode::trunc>
CK_TILE_HOST_DEVICE constexpr float float_to_tf32(float x)
{
    uint32_t i = bit_cast<uint32_t>(x);
    if constexpr(rounding == tf32_rounding_mode::rne)
    {
        // RTNE rounding.
        if((i & float32_exponent_mask) != float32_exponent_mask)
        {
            // Add rounding bias for round-to-nearest-even (RTNE) before truncating:
            //  - 0xfff is the rounding bias corresponding to the 13 fraction bits that
            //    will be discarded.
            //  - (i >> 13) & 1 extracts the least significant of those discarded bits and
            //    adding it implements "ties to even" (round half-way cases to even).
            i += 0xfff + ((i >> 13) & 1);
        }
    }
    // Zero out the lowest 13 fraction bits to form the TF32-like value.
    i &= 0xFFFFE000u;
    return bit_cast<float>(i);
}

template <typename Y,
          tf32_rounding_mode rounding                       = tf32_rounding_mode::trunc,
          std::enable_if_t<std::is_same_v<Y, tf32_t>, bool> = false>
CK_TILE_HOST_DEVICE constexpr float type_convert(float x)
{
    return float_to_tf32<rounding>(x);
}

CK_TILE_TYPE_CONVERT(fp16_t, fp16, float, float)
CK_TILE_TYPE_CONVERT(bf16_t, bf16, float, float)
CK_TILE_TYPE_CONVERT(fp8_t, fp8, float, float)
CK_TILE_TYPE_CONVERT(bf8_t, bf8, float, float)

CK_TILE_TYPE_CONVERT(float, float, int8_t, int8)
CK_TILE_TYPE_CONVERT(int8_t, int8, float, float)

CK_TILE_TYPE_CONVERT(fp8_t, fp8, fp16_t, fp16)
CK_TILE_TYPE_CONVERT(bf8_t, bf8, fp16_t, fp16)
CK_TILE_TYPE_CONVERT(fp16_t, fp16, fp8_t, fp8)
CK_TILE_TYPE_CONVERT(fp16_t, fp16, bf8_t, bf8)

CK_TILE_TYPE_CONVERT(fp16x2_t, fp16x2, fp32x2_t, fp32x2)
CK_TILE_TYPE_CONVERT(bf16x2_t, bf16x2, fp32x2_t, fp32x2)
#undef CK_TILE_TYPE_CONVERT

} // namespace ck_tile

#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/float8_ext.hpp"

namespace ck_tile {

template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr Y scaled_type_convert(X x, float scale);

#define CK_TILE_SCALED_TYPE_CONVERT(dtype_, dname_, stype_, sname_)                       \
    template <>                                                                           \
    CK_TILE_HOST_DEVICE constexpr dtype_ scaled_type_convert<dtype_, stype_>(stype_ x,    \
                                                                             float scale) \
    {                                                                                     \
        return sname_##_to_##dname_(x, scale);                                            \
    }                                                                                     \
    template <>                                                                           \
    CK_TILE_HOST_DEVICE constexpr dtype_ type_convert<dtype_, stype_>(stype_ x)           \
    {                                                                                     \
        return sname_##_to_##dname_(x, 1.f);                                              \
    }

CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp32x2_t, fp32x2)
CK_TILE_SCALED_TYPE_CONVERT(fp32x2_t, fp32x2, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16x2_t, fp16x2)
CK_TILE_SCALED_TYPE_CONVERT(fp16x2_t, fp16x2, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16x2_t, bf16x2)
CK_TILE_SCALED_TYPE_CONVERT(bf16x2_t, bf16x2, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, float, float)
CK_TILE_SCALED_TYPE_CONVERT(float, float, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16_t, bf16)
CK_TILE_SCALED_TYPE_CONVERT(bf16_t, bf16, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16_t, fp16)
CK_TILE_SCALED_TYPE_CONVERT(fp16_t, fp16, pk_fp4_t, pk_fp4)

// 8-element vector conversions for pk_fp4x4_t
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4x4_t, pk_fp4, fp32x8_t, fp32x8)
CK_TILE_SCALED_TYPE_CONVERT(fp32x8_t, fp32x8, pk_fp4x4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4x4_t, pk_fp4, fp16x8_t, fp16x8)
CK_TILE_SCALED_TYPE_CONVERT(fp16x8_t, fp16x8, pk_fp4x4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4x4_t, pk_fp4, bf16x8_t, bf16x8)
CK_TILE_SCALED_TYPE_CONVERT(bf16x8_t, bf16x8, pk_fp4x4_t, pk_fp4)

CK_TILE_SCALED_TYPE_CONVERT(pk_fp6_t, pk_fp6, float, float)
CK_TILE_SCALED_TYPE_CONVERT(float, float, pk_fp6_t, pk_fp6)
CK_TILE_SCALED_TYPE_CONVERT(pk_bf6_t, pk_bf6, float, float)
CK_TILE_SCALED_TYPE_CONVERT(float, float, pk_bf6_t, pk_bf6)

// 16-element vector conversions for pk_fp6_t and pk_bf6_t
CK_TILE_SCALED_TYPE_CONVERT(pk_fp6_t, pk_fp6, fp16x16_t, fp16x16)
CK_TILE_SCALED_TYPE_CONVERT(fp16x16_t, fp16x16, pk_fp6_t, pk_fp6)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp6_t, pk_fp6, bf16x16_t, bf16x16)
CK_TILE_SCALED_TYPE_CONVERT(bf16x16_t, bf16x16, pk_fp6_t, pk_fp6)
CK_TILE_SCALED_TYPE_CONVERT(pk_bf6_t, pk_bf6, fp16x16_t, fp16x16)
CK_TILE_SCALED_TYPE_CONVERT(fp16x16_t, fp16x16, pk_bf6_t, pk_bf6)
CK_TILE_SCALED_TYPE_CONVERT(pk_bf6_t, pk_bf6, bf16x16_t, bf16x16)
CK_TILE_SCALED_TYPE_CONVERT(bf16x16_t, bf16x16, pk_bf6_t, pk_bf6)
#if !CK_TILE_AVX512F_WA
CK_TILE_SCALED_TYPE_CONVERT(pk_fp6_t, pk_fp6, fp32x16_t, fp32x16)
CK_TILE_SCALED_TYPE_CONVERT(fp32x16_t, fp32x16, pk_fp6_t, pk_fp6)
CK_TILE_SCALED_TYPE_CONVERT(pk_bf6_t, pk_bf6, fp32x16_t, fp32x16)
CK_TILE_SCALED_TYPE_CONVERT(fp32x16_t, fp32x16, pk_bf6_t, pk_bf6)
#endif

// 8-element vector conversions for fp8x8_t, bf8x8_t
CK_TILE_SCALED_TYPE_CONVERT(fp8x8_t, fp8x8, fp32x8_t, fp32x8)
CK_TILE_SCALED_TYPE_CONVERT(bf8x8_t, bf8x8, fp32x8_t, fp32x8)
CK_TILE_SCALED_TYPE_CONVERT(fp8x8_t, fp8x8, fp16x8_t, fp16x8)
CK_TILE_SCALED_TYPE_CONVERT(bf8x8_t, bf8x8, fp16x8_t, fp16x8)
CK_TILE_SCALED_TYPE_CONVERT(fp8x8_t, fp8x8, bf16x8_t, bf16x8)
CK_TILE_SCALED_TYPE_CONVERT(bf8x8_t, bf8x8, bf16x8_t, bf16x8)

CK_TILE_SCALED_TYPE_CONVERT(fp32x8_t, fp32x8, fp8x8_t, fp8x8)
CK_TILE_SCALED_TYPE_CONVERT(fp32x8_t, fp32x8, bf8x8_t, bf8x8)
CK_TILE_SCALED_TYPE_CONVERT(fp16x8_t, fp16x8, fp8x8_t, fp8x8)
CK_TILE_SCALED_TYPE_CONVERT(fp16x8_t, fp16x8, bf8x8_t, bf8x8)
CK_TILE_SCALED_TYPE_CONVERT(bf16x8_t, bf16x8, fp8x8_t, fp8x8)
CK_TILE_SCALED_TYPE_CONVERT(bf16x8_t, bf16x8, bf8x8_t, bf8x8)
#undef CK_TILE_SCALED_TYPE_CONVERT

#if defined(__gfx125__)
// Declare a template function for wave-wise scaled conversion
/* scale is packed 4 form, see details for FP8/BF8, FP4, FP6 */
template <typename Y, typename X, int Scale_sel>
struct pk4scaled_type_convert_impl
{
    CK_TILE_DEVICE static constexpr Y run(X x, Packed4Scale_E8M0 scale);
};

template <typename Y, typename X, int Scale_sel = 0>
CK_TILE_DEVICE constexpr Y pk4scaled_type_convert(X x, Packed4Scale_E8M0 scale)
{
    return pk4scaled_type_convert_impl<Y, X, Scale_sel>::run(x, scale);
}

/* scale is packed 4 form [FP4]
 * Scale_sel: select different scale set and apply to the tensor[16x16] represented by a wave,
 *            th[0-15]: 16x8 and th[16-31]: 16x8
 *      Block 32 :
 *      0(000): src[th[0-15]]  * scale[th[0-15]][7:0]
                src[th[16-31]] * scale[th[0-15]][15:8]
 *      1(001): src[th[0-15]]  * scale[th[16-31]][7:0]
                src[th[16-31]] * scale[th[16-31]][15:8]
 *      2(010): src[th[0-15]]  * scale[th[0-15]][23:16]
                src[th[16-31]] * scale[th[0-15]][31:24]
 *      3(011): src[th[0-15]]  * scale[th[16-31]][23:16]
                src[th[16-31]] * scale[th[16-31]][31:24]
 *      Block 16 : Available for certain revision
 *      4(100): src[th[0-15]]  * scale[th[0-15]][7:0]
                src[th[16-31]] * scale[th[0-15]][23:16]
 *      5(101): src[th[0-15]]  * scale[th[16-31]][7:0]
                src[th[16-31]] * scale[th[16-31]][23:16]
 *      6(110): src[th[0-15]]  * scale[th[0-15]][15:8]
                src[th[16-31]] * scale[th[0-15]][31:24]
 *      7(111): src[th[0-15]]  * scale[th[16-31]][15:8]
                src[th[16-31]] * scale[th[16-31]][31:24]
 */
template <typename Y, int Scale_sel>
struct pk4scaled_type_convert_impl<Y, pk_fp4x4_t, Scale_sel>
{
    CK_TILE_DEVICE static Y run(pk_fp4x4_t x, Packed4Scale_E8M0 scale)
    {
        return impl::_from_f4x8_pkscale<Y, Scale_sel>(bit_cast<uint32_t>(x), scale.data());
    }
};

// pk6scaled_type_convert for FP6 E2M3 and BF6 E3M2
template <typename Y, typename X, int Scale_sel>
struct pk6scaled_type_convert_impl
{
    CK_TILE_DEVICE static constexpr Y run(X x, Packed4Scale_E8M0 scale);
};

template <typename Y, typename X, int Scale_sel = 0>
CK_TILE_DEVICE constexpr Y pk6scaled_type_convert(X x, Packed4Scale_E8M0 scale)
{
    return pk6scaled_type_convert_impl<Y, X, Scale_sel>::run(x, scale);
}

template <typename Y, int Scale_sel>
struct pk6scaled_type_convert_impl<Y, pk_fp6_t, Scale_sel>
{
    CK_TILE_DEVICE static Y run(pk_fp6_t x, Packed4Scale_E8M0 scale)
    {
        return impl::_from_fp6x16_pkscale<Y, Scale_sel>(x.get(), scale.data());
    }
};

template <typename Y, int Scale_sel>
struct pk6scaled_type_convert_impl<Y, pk_bf6_t, Scale_sel>
{
    CK_TILE_DEVICE static Y run(pk_bf6_t x, Packed4Scale_E8M0 scale)
    {
        return impl::_from_bf6x16_pkscale<Y, Scale_sel>(x.get(), scale.data());
    }
};

/* scale is packed 4 form [FP8/BF8]
 * Scale_sel: select different scale set and apply to the tensor[16x16] represented by a wave,
 *            th[0-15]: 16x8 and th[16-31]: 16x8
 *      Block 32 :
 *      0(0000): src[th[0:31]]  * scale[th[0:15]][7:0]
 *      1(0001): src[th[0:31]]  * scale[th[16:31]][7:0]
 *      2(0010): src[th[0:31]]  * scale[th[0:15]][23:16]
 *      3(0011): src[th[0:31]]  * scale[th[16:31]][23:16]
 *      4(0100): src[th[0:31]]  * scale[th[0:15]][15:8]
 *      5(0101): src[th[0:31]]  * scale[th[16:31]][15:8]
 *      6(0110): src[th[0:31]]  * scale[th[0:15]][31:24]
 *      7(0111): src[th[0:31]]  * scale[th[16:31]][31:24]
 *      Block 16 : Available for certain revision
 *      8(1000) : src[th[0:15]]  * scale[th[0:15]][7:0]
 *                src[th[16:31]] * scale[th[0:15]][15:8]
 *      9(1001) : src[th[0:15]]  * scale[th[16:31]][7:0]
 *                src[th[16:31]] * scale[th[16:31]][15:8]
 *      10(1010): src[th[0:15]]  * scale[th[0:15]][23:16]
 *                src[th[16:31]] * scale[th[0:15]][31:24]
 *      11(1011): src[th[0:15]]  * scale[th[16:31]][23:16]
 *                src[th[16:31]] * scale[th[16:31]][31:24] */
template <typename Y, int Scale_sel>
struct pk4scaled_type_convert_impl<Y, fp8x8_t, Scale_sel>
{
    CK_TILE_DEVICE static Y run(fp8x8_t x, Packed4Scale_E8M0 scale)
    {
        return impl::cast_from_f8x8_scaled<Y, numeric_traits<fp8_t>::f8_interpret, Scale_sel>(
            bit_cast<impl::fp8x8_storage_t>(x), scale.data());
    }
};
template <typename Y, int Scale_sel>
struct pk4scaled_type_convert_impl<Y, bf8x8_t, Scale_sel>
{
    CK_TILE_DEVICE static Y run(bf8x8_t x, Packed4Scale_E8M0 scale)
    {
        return impl::cast_from_f8x8_scaled<Y, numeric_traits<bf8_t>::f8_interpret, Scale_sel>(
            bit_cast<impl::fp8x8_storage_t>(x), scale.data());
    }
};
#endif

#endif

} // namespace ck_tile
#pragma clang diagnostic pop
