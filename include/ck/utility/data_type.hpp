// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/statically_indexed_array.hpp"
#include "ck/utility/random_gen.hpp"

#ifdef CK_USE_FNUZ_FP8
#define CK_USE_FNUZ_FP8 1
#else
#define CK_USE_FNUZ_FP8 0
#endif

#ifdef CK_USE_OCP_FP8
#define CK_USE_OCP_FP8 1
#else
#define CK_USE_OCP_FP8 0
#endif

namespace ck {

using bhalf_t = ushort;
using half_t  = _Float16;
using int4_t  = _BitInt(4);

using f8_fnuz_t  = _BitInt(8);
using bf8_fnuz_t = unsigned _BitInt(8);

#if(defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx1200__) || \
    defined(__gfx1201__) || defined(__gfx950__)) &&                                              \
    __HIP_DEVICE_COMPILE__
#define CK_FP8_CVT_FAST_PATH 1
#else
#define CK_FP8_CVT_FAST_PATH 0
#endif

typedef unsigned char fp8_storage_t;

namespace internal {

/**
 * \brief Describes FP8 interpretation
 */
enum ck_fp8_interpretation_t
{
    CK_E4M3_OCP  = 0, // OCP E4M3
    CK_E5M2_OCP  = 1, // OCP E5M2
    CK_E4M3_FNUZ = 2, // FP8
    CK_E5M2_FNUZ = 3, // BF8
};

/**
 * \brief Describes saturation behavior
 */
enum ck_saturation_t
{
    CK_NOSAT     = 0, // No saturation - replace with NaN or Inf
    CK_SATFINITE = 1, // Saturate to finite
};

// Assertions to check for supported conversion types
#define __assert_ocp_support(interp)                                               \
    {                                                                              \
        if(interp != CK_E4M3_OCP && interp != CK_E5M2_OCP)                         \
        {                                                                          \
            __hip_assert(false && "type is unsupported by current target device"); \
        }                                                                          \
    }
#define __assert_fnuz_support(interp)                                              \
    {                                                                              \
        if(interp != CK_E4M3_FNUZ && interp != CK_E5M2_FNUZ)                       \
        {                                                                          \
            __hip_assert(false && "type is unsupported by current target device"); \
        }                                                                          \
    }

__host__ __device__ static inline void
__is_interpret_supported([[maybe_unused]] ck_fp8_interpretation_t interp)
{
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if CK_USE_OCP_FP8
    __assert_ocp_support(interp);
#endif
#if CK_USE_FNUZ_FP8
    __assert_fnuz_support(interp);
#endif
#endif
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L39
// This has been modified to add double types conversion as well
template <typename T, int wm, int we, bool is_fnuz, bool clip = false, bool stoch = false>
__host__ __device__ static inline fp8_storage_t cast_to_f8(T _x, unsigned int rng = 0)
{
    constexpr bool is_half   = __hip_internal::is_same<T, _Float16>::value;
    constexpr bool is_float  = __hip_internal::is_same<T, float>::value;
    constexpr bool is_double = __hip_internal::is_same<T, double>::value;
    static_assert(is_half || is_float || is_double,
                  "Only half, float and double can be cast to f8");

    constexpr int mfmt = (sizeof(T) == 8) ? 52 : ((sizeof(T) == 4) ? 23 : 10);

    using T_bitwise = typename __hip_internal::conditional<
        sizeof(T) == 2,
        unsigned short int,
        typename __hip_internal::conditional<sizeof(T) == 4, unsigned int, unsigned long long>::
            type>::type;
    T_bitwise x_bitwise = bit_cast<T_bitwise>(_x);

    unsigned long long x{x_bitwise};

    unsigned long long head, mantissa;
    int exponent, bias;
    unsigned int sign;
    unsigned long long fInf, mask;

    if constexpr(sizeof(T) == 8)
    {
        head     = x & 0xFFF0000000000000ull;
        mantissa = x & 0xFFFFFFFFFFFFFull;
        exponent = (head >> 52) & 0x7FF;
        sign     = head >> 63;
        bias     = 1023;
        fInf     = 0x7FF0000000000000ull;
        mask     = 0x7FFFFFFFFFFFFFFFull;
    }
    else if(sizeof(T) == 4)
    {
        head     = x & 0xFF800000;
        mantissa = x & 0x7FFFFF;
        exponent = (head >> 23) & 0xFF;
        sign     = head >> 31;
        bias     = 127;
        fInf     = 0x7F800000;
        mask     = 0x7FFFFFFF;
    }
    else
    {
        head     = x & 0xFC00;
        mantissa = x & 0x3FF;
        exponent = (head >> 10) & 0x1F;
        sign     = head >> 15;
        bias     = 15;
        fInf     = 0x7C00;
        mask     = 0x7FFF;
    }
    unsigned int signed_inf = 0;
    unsigned int nan        = 0;
    if constexpr(is_fnuz)
    {
        signed_inf = clip ? ((sign << 7) + 0x7f) : 0x80;
        nan        = 0x80;
    }
    else
    {
        if(we == 4)
        { // e4m3
            signed_inf = (sign << 7) + (clip ? 0x7e : 0x7f);
        }
        else
        { // e5m2
            signed_inf = (sign << 7) + (clip ? 0x7b : 0x7c);
        }
        nan = (sign << 7) + 0x7f;
    }
    // Max values
    unsigned long long ifmax = 0;
    if constexpr(sizeof(T) == 8)
    {
        if(we == 5)
        { // 57344
            ifmax = 0x40EC000000000000ull;
        }
        else
        {
            if(is_fnuz)
            { // 240
                ifmax = 0x406E000000000000ull;
            }
            else
            { // 448
                ifmax = 0x407C000000000000ull;
            }
        }
    }
    else if(sizeof(T) == 4)
    {
        if(we == 5)
        {
            ifmax = 0x47600000;
        }
        else
        {
            if(is_fnuz)
            {
                ifmax = 0x43700000;
            }
            else
            {
                ifmax = 0x43E00000;
            }
        }
    }
    else
    {
        if(we == 5)
        {
            ifmax = 0x7B00;
        }
        else
        {
            if(is_fnuz)
            {
                ifmax = 0x5B80;
            }
            else
            {
                ifmax = 0x5F00;
            }
        }
    }
    // Deal with inf and NaNs
    if((x & fInf) == fInf)
    {
        if(is_fnuz)
            return signed_inf;

        return mantissa != 0 ? nan : signed_inf;
    }

    if((x & mask) > ifmax)
    {
        return signed_inf;
    }

    if(x == 0)
    {
        return 0;
    }

    // First need to check if it is normal or denorm as there is a difference of
    // implicit 1 Then need to adjust the exponent to align with the F8 exponent,
    // in the meanwhile, shift The mantissa. Then for stochastic rounding, add rng
    // to mantissa and truncate. And for RNE, no need to add rng. Then probably
    // need to check whether there is carry and adjust exponent and mantissa again

    // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent
    // bits
    const int f8_bias                  = (1 << (we - 1)) - 1 + (is_fnuz ? 1 : 0);
    const int f8_denormal_act_exponent = 1 - f8_bias; // actual exponent of f8 denormal
    // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    // f8_exponent is the converted f8 exponent with bias encoding
    // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    // the difference needs to be adjusted and mantissa shifted
    int act_exponent, f8_exponent, exponent_diff;

    if(exponent == 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we
    mostly concern fp16 here. In this case, f8 is usually in denormal. But there
    could be exceptions. fp16 denormal has exponent bias 15 while bf8 with NANOO has
    exponent bias 16. It means that there are some numbers in fp16 denormal but they
    are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
    where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8
    (NANOO) normal. In this case, the fp16 mantissa should be shift left by 1  */
        act_exponent  = exponent - bias + 1;
        exponent_diff = f8_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= f8_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal
      range. For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
      actual exponent is -7, it is actually larger due to the implicit 1,
      Therefore it needs to be adjust to -6 and mantissa shift right by 1.
      So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        }
        else
        {                      // both fp32/fp16 and f8 are in normal range
            exponent_diff = 0; // exponent_diff=0 does not mean there is no difference
                               // for this case, act_exponent could be larger. Just
                               // that it does not need shift mantissa
        }
        mantissa += (1ull << mfmt); // Add the implicit 1 into mantissa
    }

    bool midpoint = (mantissa & ((1ull << (mfmt - wm + exponent_diff)) - 1)) ==
                    (1ull << (mfmt - wm + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be
  done before we shift right as shift right could rip off some residual part and
  make something not midpoint look like midpoint. For example, the fp16 number
  0x1002 (0 00100 0000000010), it is larger than midpoint, but after shift right
  by 4 bits, it would look like midpoint.
  */

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1ull << mfmt);
    // if there is no implicit 1, it  means the f8 is denormal and need to adjust
    // to denorm exponent
    f8_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    unsigned long long drop_mask = (1ull << (mfmt - wm)) - 1;
    bool odd =
        mantissa & (1ull << (mfmt - wm)); // if the least significant bit that is not truncated is 1
    mantissa +=
        (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(f8_exponent == 0)
    {
        if((1ull << mfmt) & mantissa)
        {
            f8_exponent = 1; // denormal overflow to become normal, promote exponent
        }
    }
    else
    {
        if((1ull << (mfmt + 1)) & mantissa)
        {
            mantissa >>= 1;
            f8_exponent++;
        }
    }

    mantissa >>= (mfmt - wm);

    // above range: quantize to maximum possible float of the same sign
    const int max_exp = (1 << we) - 1;
    if(f8_exponent > max_exp)
    {
        if(clip)
        {
            mantissa    = (1 << wm) - 1;
            f8_exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    if(f8_exponent == 0 && mantissa == 0)
        return is_fnuz ? 0 : (sign << 7);
    mantissa &= (1 << wm) - 1;
    return (sign << 7) | (f8_exponent << wm) | mantissa;
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L220
// This has been modified to handle double types as well
template <typename T, int wm, int we, bool is_fnuz, bool clip = false>
__host__ __device__ static inline T cast_from_f8(fp8_storage_t x)
{
    constexpr bool is_half   = __hip_internal::is_same<T, _Float16>::value;
    constexpr bool is_float  = __hip_internal::is_same<T, float>::value;
    constexpr bool is_double = __hip_internal::is_same<T, double>::value;
    static_assert(is_half || is_float || is_double, "only half, float and double are supported");

    constexpr int weo = is_half ? 5 : (is_float ? 8 : 11);
    constexpr int wmo = is_half ? 10 : (is_float ? 23 : 52);

    T fInf, fNegInf, fNaN, fNeg0, fmax, fmin;
    if constexpr(is_half)
    {
        const unsigned short int ihInf    = 0x7C00;
        const unsigned short int ihNegInf = 0xFC00;
        const unsigned short int ihNaN    = 0x7C01;
        const unsigned short int ihNeg0   = 0x8000;
        /* Max number in e5m2 57344*/
        const unsigned short int ifmax = 0x7B00;
        const unsigned short int ifmin = 0xFB00;

        fInf    = bit_cast<_Float16>(ihInf);
        fNegInf = bit_cast<_Float16>(ihNegInf);
        fNaN    = bit_cast<_Float16>(ihNaN);
        fNeg0   = bit_cast<_Float16>(ihNeg0);
        fmax    = bit_cast<_Float16>(ifmax);
        fmin    = bit_cast<_Float16>(ifmin);
    }
    else if(is_float)
    {
        const unsigned int ifInf    = 0x7F800000;
        const unsigned int ifNegInf = 0xFF800000;
        const unsigned int ifNaN    = 0x7F800001;
        const unsigned int ifNeg0   = 0x80000000;
        /* Max number in e5m2 57344*/
        const unsigned int ifmax = 0x47600000;
        const unsigned int ifmin = 0xC7600000;

        fInf    = bit_cast<float>(ifInf);
        fNegInf = bit_cast<float>(ifNegInf);
        fNaN    = bit_cast<float>(ifNaN);
        fNeg0   = bit_cast<float>(ifNeg0);
        fmax    = bit_cast<float>(ifmax);
        fmin    = bit_cast<float>(ifmin);
    }
    else if(is_double)
    {
        const unsigned long long ifInf    = 0x7FF0000000000000ull;
        const unsigned long long ifNegInf = 0xFFF0000000000000ull;
        const unsigned long long ifNaN    = 0x7FF0000000000001ull;
        const unsigned long long ifNeg0   = 0x8000000000000000ull;
        /* Max number in e5m2 57344*/
        const unsigned long long ifmax = 0x40EC000000000000ull;
        const unsigned long long ifmin = 0xC0EC000000000000ull;

        fInf    = bit_cast<double>(ifInf);
        fNegInf = bit_cast<double>(ifNegInf);
        fNaN    = bit_cast<double>(ifNaN);
        fNeg0   = bit_cast<double>(ifNeg0);
        fmax    = bit_cast<double>(ifmax);
        fmin    = bit_cast<double>(ifmin);
    }

    if(x == 0)
    {
        return 0;
    }

    unsigned long long sign     = x >> 7;
    unsigned long long mantissa = x & ((1 << wm) - 1);
    int exponent                = (x & 0x7F) >> wm;
    if constexpr(is_fnuz)
    {
        if(x == 0x80)
        {
            return fNaN;
        }
    }
    else
    {
        if(x == 0x80)
        {
            return fNeg0;
        }
        if(we == 4)
        { // e4m3
            if((x & 0x7F) == 0x7F)
            {
                return fNaN;
            }
        }
        else if((x & 0x7C) == 0x7C)
        { // e5m2
            if((x & 0x3) == 0)
            {
                if(clip)
                {
                    return sign ? fmin : fmax;
                }
                return sign ? fNegInf : fInf;
            }
            return fNaN;
        }
    }

    typename __hip_internal::conditional<
        sizeof(T) == 2,
        unsigned short int,
        typename __hip_internal::conditional<sizeof(T) == 4, unsigned int, unsigned long long>::
            type>::type retval;

    if(we == 5 && is_half && !is_fnuz)
    {
        retval = x << 8;
        return bit_cast<T>(retval);
    }

    const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (is_fnuz ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __clz(mantissa) - (32 - wm);
#else
        int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
#endif
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1ull << wm) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= wmo - wm;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << wmo;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    if constexpr(sizeof(T) == 2)
        retval = (sign << 15) | (exponent << 10) | mantissa;
    else if(sizeof(T) == 4)
        retval = (sign << 31) | (exponent << 23) | mantissa;
    else
        retval = (sign << 63) | (static_cast<unsigned long long>(exponent) << 52) | mantissa;
    return bit_cast<T>(retval);
}

#if CK_FP8_CVT_FAST_PATH
template <ck_fp8_interpretation_t interpret>
static __device__ float cast_to_f32_from_f8(fp8_storage_t v)
{
    union
    {
        unsigned int i32val;
        unsigned char i8val[4];
    } val;
    val.i8val[0] = v;

    static_assert(interpret == CK_E4M3_FNUZ || interpret == CK_E4M3_OCP ||
                      interpret == CK_E5M2_FNUZ || interpret == CK_E5M2_OCP,
                  "Only FNUZ and OCP interpretations are supported");

    if constexpr((interpret == internal::CK_E4M3_FNUZ) || (interpret == internal::CK_E4M3_OCP))
    {
        return __builtin_amdgcn_cvt_f32_fp8(val.i32val, 0);
    }
    else
    {
        return __builtin_amdgcn_cvt_f32_bf8(val.i32val, 0);
    }
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_float8.h#L79
template <ck_fp8_interpretation_t interpret, bool saturate, bool stochastic_rounding = false>
static __device__ fp8_storage_t cast_to_f8_from_f32(float v, unsigned int rng = 0)
{
    fp8_storage_t i8data;
    union
    {
        float fval;
        unsigned int i32val;
        unsigned char i8val[4]; // NOTE: not endian independent
    } val;

    unsigned int ival = 0;
    val.fval          = v;

    if constexpr(saturate)
    {
        if constexpr(interpret == CK_E4M3_FNUZ)
        {
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
            }
        }
        else if(interpret == CK_E4M3_OCP)
        { // OCP type
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 448.0, -448.0);
            }
        }
        else
        {
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
            }
        }
    }

    if constexpr(stochastic_rounding)
    {
        ival       = (interpret == CK_E4M3_FNUZ) || (interpret == CK_E4M3_OCP)
                         ? __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0)
                         : __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
        val.i32val = ival;
        i8data     = val.i8val[0]; // little endian
    }
    else
    { // RNE CVT
        ival       = (interpret == CK_E4M3_FNUZ) || (interpret == CK_E4M3_OCP)
                         ? __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false)
                         : __builtin_amdgcn_cvt_pk_bf8_f32(val.fval,
                                                     val.fval,
                                                     ival,
                                                     false); // false -> WORD0
        val.i32val = ival;
        i8data     = val.i8val[0];
    }
    return i8data;
}

#endif // CK_FP8_CVT_FAST_PATH

/**
 * \brief convert float to @p fp8_storage_t
 *
 * \tparam interp interpretation of fp8
 * \tparam sat saturation of fp8
 * \param f float number
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp,
          ck_saturation_t sat      = CK_SATFINITE,
          bool stochastic_rounding = false>
#if CK_FP8_CVT_FAST_PATH
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8(const float f)
{
    internal::__is_interpret_supported(interp);
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f);
    }
    return internal::cast_to_f8_from_f32<interp, sat == CK_SATFINITE, stochastic_rounding>(f, rng);
#else
#if CK_USE_OCP_FP8
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8(const float f)
{
#else
__host__ static inline fp8_storage_t cvt_float_to_fp8(const float f)
{
#endif
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        constexpr int seed = 1254739;
        rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f);
    }

    if constexpr(interp == CK_E4M3_FNUZ)
    {
        return internal::cast_to_f8<float, 3, 4, true, sat == CK_SATFINITE, stochastic_rounding>(
            f, rng);
    }
    else if(interp == CK_E5M2_FNUZ)
    {
        return internal::cast_to_f8<float, 2, 5, true, sat == CK_SATFINITE, stochastic_rounding>(
            f, rng);
    }
    else if(interp == CK_E4M3_OCP)
    {
        return internal::cast_to_f8<float, 3, 4, false, sat == CK_SATFINITE, stochastic_rounding>(
            f, rng);
    }
    else if(interp == CK_E5M2_OCP)
    {
        return internal::cast_to_f8<float, 2, 5, false, sat == CK_SATFINITE, stochastic_rounding>(
            f, rng);
    }
    else
    {
        __hip_assert(false && "FP8 type is not supported by current target device");
        return 0;
    }
#endif // CK_FP8_CVT_FAST_PATH
}

/**
 * \brief convert half_t to @p fp8_storage_t
 *
 * \tparam sat saturation of fp8
 * \tparam interp interpretation of fp8
 * \tparam stochastic_rounding switch between RNE and SR
 * \param x half_t value
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp,
          ck_saturation_t sat      = CK_SATFINITE,
          bool stochastic_rounding = false>
#if CK_FP8_CVT_FAST_PATH
__host__ __device__ static inline fp8_storage_t cvt_half_t_to_fp8(const half_t x)
{
    internal::__is_interpret_supported(interp);
#elif CK_USE_OCP_FP8
__host__ __device__ static inline fp8_storage_t cvt_half_t_to_fp8(const half_t x)
{
#else
__host__ static inline fp8_storage_t cvt_half_t_to_fp8(const half_t x)
{
#endif
    return cvt_float_to_fp8<interp, sat, stochastic_rounding>(static_cast<float>(x));
}

/* For fp8 fnuz types, finite and NaN values are supported. Zero is unsigned.
Inf are not supported. This gives us one additional number to represent.
NaN are represented by 1-0000-000 or 1-00000-00 */
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, bf8_fnuz_t> || std::is_same_v<T, f8_fnuz_t>, bool> = true>
__host__ __device__ inline constexpr bool fnuz_fp8_is_nan(T a)
{
    return static_cast<unsigned char>(a) == 0x80;
}
__host__ __device__ static inline constexpr bool fnuz_f8_is_nan(f8_fnuz_t a)
{
    return static_cast<unsigned char>(a) == 0x80;
}
__host__ __device__ static inline constexpr bool fnuz_bf8_is_nan(bf8_fnuz_t a)
{
    return static_cast<unsigned char>(a) == 0x80;
}

__host__ __device__ static inline constexpr bool ocp_f8_is_nan(fp8_storage_t a)
{
    return (a & 0x7f) == 0x7f;
}
__host__ __device__ static inline constexpr bool ocp_bf8_is_nan(fp8_storage_t a)
{
    return (a & 0x7f) > 0x7c;
}

} // namespace internal

struct f8_ocp_t
{
    using data_type = fp8_storage_t;
    data_type data;

    static constexpr internal::ck_saturation_t default_saturation        = internal::CK_SATFINITE;
    static constexpr internal::ck_fp8_interpretation_t default_interpret = internal::CK_E4M3_OCP;

    static constexpr unsigned int we = 4; // exponent width
    static constexpr unsigned int wm = 3; // mantissa width

    __host__ __device__ constexpr bool operator==(const f8_ocp_t& other) const
    {
        return (data == other.data) && (internal::ocp_f8_is_nan(data) == false); // NaN != NaN
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator float() const
    {
#else
    __host__ explicit operator float() const
    {
#endif
#if CK_FP8_CVT_FAST_PATH
        return internal::cast_to_f32_from_f8<default_interpret>(this->data);
#else
        return internal::cast_from_f8<float, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator half_t
#endif
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator half_t() const {
#else
    __host__ explicit operator half_t() const
    {
#endif
#if CK_FP8_CVT_FAST_PATH
        return static_cast<half_t>(internal::cast_to_f32_from_f8<default_interpret>(this->data));
#else
        return internal::cast_from_f8<half_t, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator float
#endif
}
}; // namespace ck

struct bf8_ocp_t
{
    using data_type = fp8_storage_t;
    data_type data;

    static constexpr internal::ck_saturation_t default_saturation        = internal::CK_SATFINITE;
    static constexpr internal::ck_fp8_interpretation_t default_interpret = internal::CK_E5M2_OCP;

    static constexpr unsigned int we = 5; // exponent width
    static constexpr unsigned int wm = 2; // mantissa width

    __host__ __device__ constexpr bool operator==(const bf8_ocp_t& other) const
    {
        return (data == other.data) && (internal::ocp_bf8_is_nan(data) == false); // NaN != NaN
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator float() const {
#else
    __host__ explicit operator float() const
    {
#endif
#if CK_FP8_CVT_FAST_PATH
        return internal::cast_to_f32_from_f8<default_interpret>(this->data);
#else
        return internal::cast_from_f8<float, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator half_t
#endif
}
}
;

namespace internal {
template <typename T,
          std::enable_if_t<std::is_same_v<T, bf8_ocp_t> || std::is_same_v<T, f8_ocp_t> ||
                               std::is_same_v<T, bf8_fnuz_t> || std::is_same_v<T, f8_fnuz_t>,
                           bool> = true>
__host__ __device__ static inline constexpr bool fp8_is_inf(T)
{
    return false;
}
template <>
__host__ __device__ inline constexpr bool fp8_is_inf(bf8_ocp_t a)
{
    return (a.data & 0x7f) == 0x7c;
}
} // namespace internal

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_rne(X x);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        internal::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{
        internal::cvt_float_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(x)};
}

// convert half_t to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, half_t>(half_t x)
{
    return f8_ocp_t{
        internal::cvt_half_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}
// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        internal::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation, true>(
            x)};
}
// convert half_t to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, half_t>(half_t x)
{
    return f8_ocp_t{internal::cvt_half_t_to_fp8<f8_ocp_t::default_interpret,
                                                f8_ocp_t::default_saturation,
                                                true>(x)};
}

#if CK_USE_OCP_FP8
using f8_t  = f8_ocp_t;
using bf8_t = bf8_ocp_t;
#define CK_FP8_TYPE_FNUZ 0
#define CK_FP8_TYPE_OCP 1
#else
using f8_t = f8_fnuz_t;
using bf8_t = bf8_fnuz_t;
#define CK_FP8_TYPE_FNUZ 1
#define CK_FP8_TYPE_OCP 0
#endif

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}

// native types: double, float, _Float16, ushort, int32_t, int8_t, uint8_t, f8_fnuz_t, bf8_fnuz_t,
// native types: bool
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value || is_same<T, int8_t>::value ||
           is_same<T, uint8_t>::value || is_same<T, f8_fnuz_t>::value ||
           is_same<T, bf8_fnuz_t>::value || is_same<T, bool>::value;
}

// vector_type
template <typename T, index_t N, typename Enable = void>
struct vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint8_t>
{
    using type                           = uint8_t;
    static constexpr index_t vector_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<f8_fnuz_t>
{
    using type                           = f8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_fnuz_t>
{
    using type                           = bf8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_ocp_t>
{
    using type                           = f8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_ocp_t>
{
    using type                           = bf8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

template <typename T>
struct vector_type<T, 1, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    using type = d1_t;

    union
    {
        T d1_;
        StaticallyIndexedArray<T, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }
};

__device__ int static err = 0;
template <typename T>
struct vector_type<T, 2, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 128, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 256, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N>
struct non_native_vector_base
{
    using type = non_native_vector_base<T, N>;

    __host__ __device__ non_native_vector_base() = default;

    typedef char data_v __attribute__((ext_vector_type(sizeof(T) * N)));
    data_v d;
};

// non-native vector_type implementation
template <typename T>
struct vector_type<T, 1, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using type = d1_t;

    union alignas(next_pow2(1 * sizeof(T)))
    {
        d1_t d1_;
        StaticallyIndexedArray<d1_t, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }
};

template <typename T>
struct vector_type<T, 2, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;

    using type = d2_t;

    union alignas(next_pow2(2 * sizeof(T)))
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;
    using d4_t = non_native_vector_base<T, 4>;

    using type = d4_t;

    union alignas(next_pow2(4 * sizeof(T)))
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;
    using d4_t = non_native_vector_base<T, 4>;
    using d8_t = non_native_vector_base<T, 8>;

    using type = d8_t;

    union alignas(next_pow2(8 * sizeof(T)))
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;

    using type = d16_t;

    union alignas(next_pow2(16 * sizeof(T)))
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;

    using type = d32_t;

    union alignas(next_pow2(32 * sizeof(T)))
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;
    using d64_t = non_native_vector_base<T, 64>;

    using type = d64_t;

    union alignas(next_pow2(64 * sizeof(T)))
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

using int64_t = long;

// fp64
using double2_t = typename vector_type<double, 2>::type;
using double4_t = typename vector_type<double, 4>::type;

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// f8
using f8x2_fnuz_t  = typename vector_type<f8_fnuz_t, 2>::type;
using f8x4_fnuz_t  = typename vector_type<f8_fnuz_t, 4>::type;
using f8x8_fnuz_t  = typename vector_type<f8_fnuz_t, 8>::type;
using f8x16_fnuz_t = typename vector_type<f8_fnuz_t, 16>::type;
using f8x32_fnuz_t = typename vector_type<f8_fnuz_t, 32>::type;
using f8x64_fnuz_t = typename vector_type<f8_fnuz_t, 64>::type;

// bf8
using bf8x2_fnuz_t  = typename vector_type<bf8_fnuz_t, 2>::type;
using bf8x4_fnuz_t  = typename vector_type<bf8_fnuz_t, 4>::type;
using bf8x8_fnuz_t  = typename vector_type<bf8_fnuz_t, 8>::type;
using bf8x16_fnuz_t = typename vector_type<bf8_fnuz_t, 16>::type;
using bf8x32_fnuz_t = typename vector_type<bf8_fnuz_t, 32>::type;
using bf8x64_fnuz_t = typename vector_type<bf8_fnuz_t, 64>::type;

// f8
using f8x2_ocp_t  = typename vector_type<f8_ocp_t, 2>::type;
using f8x4_ocp_t  = typename vector_type<f8_ocp_t, 4>::type;
using f8x8_ocp_t  = typename vector_type<f8_ocp_t, 8>::type;
using f8x16_ocp_t = typename vector_type<f8_ocp_t, 16>::type;
using f8x32_ocp_t = typename vector_type<f8_ocp_t, 32>::type;
using f8x64_ocp_t = typename vector_type<f8_ocp_t, 64>::type;

// bf8
using bf8x2_ocp_t  = typename vector_type<bf8_ocp_t, 2>::type;
using bf8x4_ocp_t  = typename vector_type<bf8_ocp_t, 4>::type;
using bf8x8_ocp_t  = typename vector_type<bf8_ocp_t, 8>::type;
using bf8x16_ocp_t = typename vector_type<bf8_ocp_t, 16>::type;
using bf8x32_ocp_t = typename vector_type<bf8_ocp_t, 32>::type;
using bf8x64_ocp_t = typename vector_type<bf8_ocp_t, 64>::type;

#if CK_FP8_TYPE_OCP
// f8
using f8x2_t  = f8x2_ocp_t;
using f8x4_t  = f8x4_ocp_t;
using f8x8_t  = f8x8_ocp_t;
using f8x16_t = f8x16_ocp_t;
using f8x32_t = f8x32_ocp_t;
using f8x64_t = f8x64_ocp_t;

// bf8
using bf8x2_t  = bf8x2_ocp_t;
using bf8x4_t  = bf8x4_ocp_t;
using bf8x8_t  = bf8x8_ocp_t;
using bf8x16_t = bf8x16_ocp_t;
using bf8x32_t = bf8x32_ocp_t;
using bf8x64_t = bf8x64_ocp_t;
#elif CK_FP8_TYPE_FNUZ
// f8
using f8x2_t = f8x2_fnuz_t;
using f8x4_t = f8x4_fnuz_t;
using f8x8_t = f8x8_fnuz_t;
using f8x16_t = f8x16_fnuz_t;
using f8x32_t = f8x32_fnuz_t;
using f8x64_t = f8x64_fnuz_t;

// bf8
using bf8x2_t = bf8x2_fnuz_t;
using bf8x4_t = bf8x4_fnuz_t;
using bf8x8_t = bf8x8_fnuz_t;
using bf8x16_t = bf8x16_fnuz_t;
using bf8x32_t = bf8x32_fnuz_t;
using bf8x64_t = bf8x64_fnuz_t;
#endif

// u8
// i8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;
using uint8x32_t = typename vector_type<uint8_t, 32>::type;
using uint8x64_t = typename vector_type<uint8_t, 64>::type;

template <typename T>
struct NumericLimits
{
    __host__ __device__ static constexpr T Min() { return std::numeric_limits<T>::min(); }

    __host__ __device__ static constexpr T Max() { return std::numeric_limits<T>::max(); }

    __host__ __device__ static constexpr T Lowest() { return std::numeric_limits<T>::lowest(); }

    __host__ __device__ static constexpr T QuietNaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    __host__ __device__ static constexpr T Infinity() { return std::numeric_limits<T>::infinity(); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_fnuz_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 7
    // static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    // static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    // static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=0

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 15
    // static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    // static constexpr uint8_t binary_max    = 0x7B; // 0b01111011
    // static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};

template <typename T>
struct NumericUtils
{
};

template <>
struct NumericUtils<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    using bitwise_type                  = uint32_t;
};

template <>
struct NumericUtils<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    using bitwise_type                  = uint16_t;
};

template <>
struct NumericUtils<f8_fnuz_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 8; // negative zero nan mode
    // static constexpr int bias = 7; // ieee mode
};

template <>
struct NumericUtils<bf8_fnuz_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 16; // negative zero nan mode
    // static constexpr int bias = 15; // ieee mode
};
template <>
struct NumericUtils<f8_ocp_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 7;
};

template <>
struct NumericUtils<bf8_ocp_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 15;
};

} // namespace ck
