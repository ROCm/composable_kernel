// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "numeric_limits.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "tuple.hpp"

namespace ck {

// magic number division
// Caution:
//   1. For uint32_t as dividend: magic number division implementation being used would produce
//   correct result if the dividend is uint32_t and its value is within 31-bit value range.
//   2. For int32_t as dividendd: magic number division for int32_t dividened has not been
//   implemented, the int32_t dividend would be bit-wise interpreted as uint32_t and magic number
//   division implementation for uint32_t is then used. Therefore, dividend value need to be
//   non-negative.
// TODO:
//   1. Implement magic number divison for int32_t
//   2. Implement magic number divison for unit32_t with 32-bit value range
struct MagicDivision
{
    // uint32_t
    __host__ __device__ static constexpr auto CalculateMagicNumbers(uint32_t divisor)
    {
        // WARNING: magic division is only applicable for division inside this range.
        // You should use the return value of CalculateMagicNumbers, if division is not inside this
        // range. The "else" logic below is to quiet down run-time error.
        if(divisor >= 1 && divisor <= ck::NumericLimits<int32_t>::Max())
        {
            uint32_t shift = 0;
            for(shift = 0; shift < 32; ++shift)
            {
                if((1U << shift) >= divisor)
                {
                    break;
                }
            }

            uint64_t one        = 1;
            uint64_t multiplier = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
            // assert(multiplier <= 0xffffffffUL);

            return make_tuple(uint32_t(multiplier), shift);
        }
        else
        {
            return make_tuple(uint32_t(0), uint32_t(0));
        }
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicMultiplier(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<0>{}];
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicShift(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<1>{}];
    }

    // integral_constant<uint32_t, .>
    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
    CalculateMagicNumbers(integral_constant<uint32_t, Divisor>)
    {
        constexpr auto tmp = CalculateMagicNumbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[Number<0>{}];
        constexpr uint32_t shift      = tmp[Number<1>{}];

        return make_tuple(integral_constant<uint32_t, multiplier>{},
                          integral_constant<uint32_t, shift>{});
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
    CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t multiplier = CalculateMagicMultiplier(uint32_t{Divisor});

        return integral_constant<uint32_t, multiplier>{};
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
    CalculateMagicShift(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t shift = CalculateMagicShift(uint32_t{Divisor});

        return integral_constant<uint32_t, shift>{};
    }

    // integral_constant<int32_t, .>
    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
    CalculateMagicNumbers(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicNumbers(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
    CalculateMagicMultiplier(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
    CalculateMagicShift(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicShift(integral_constant<uint32_t, Divisor>{});
    }

    // magic division for uint32_t
    __device__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = __umulhi(dividend, multiplier);
        return (tmp + dividend) >> shift;
    }

    __host__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = static_cast<uint64_t>(dividend) * multiplier >> 32;
        return (tmp + dividend) >> shift;
    }

    // magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    __device__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = __umulhi(dividend_u32, multiplier);
        return (tmp + dividend_u32) >> shift;
    }

    __host__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = static_cast<uint64_t>(dividend_u32) * multiplier >> 32;
        return (tmp + dividend_u32) >> shift;
    }

    // 64-bit magic number computation for a 32-bit divisor.
    // Returns a (uint64_t multiplier, uint32_t shift) pair such that:
    //   floor(dividend / divisor) == (umulhi64(dividend, multiplier) + dividend) >> shift
    // for any uint64_t dividend, provided divisor >= 1 and divisor <= INT32_MAX.
    // The ConvBwdDataImplicitGemmOutTransform struct is always constructed on the host,
    // so the __uint128_t arithmetic below is only ever executed on the host side.
    __host__ __device__ static constexpr auto CalculateMagicNumbers64(uint32_t divisor)
    {
        uint64_t out_multiplier = 0;
        uint32_t out_shift      = 0;

        if(divisor >= 1 && divisor <= ck::NumericLimits<int32_t>::Max())
        {
            uint32_t shift = 0;
            for(shift = 0; shift < 64; ++shift)
            {
                if((uint64_t{1} << shift) >= divisor)
                {
                    break;
                }
            }
            out_shift = shift;

// __uint128_t is only available on host (CPU) compilers.
// On device, this path is never actually invoked at runtime because
// ConvBwdDataImplicitGemmOutTransform is always constructed on the host.
#ifndef __HIP_DEVICE_COMPILE__
            __uint128_t one = 1;
            out_multiplier  = uint64_t(((one << 64) * ((one << shift) - divisor)) / divisor + 1);
#endif
        }

        return make_tuple(out_multiplier, out_shift);
    }

    __host__ __device__ static constexpr uint64_t CalculateMagicMultiplier64(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers64(divisor);
        return tmp[Number<0>{}];
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicShift64(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers64(divisor);
        return tmp[Number<1>{}];
    }

    // magic division for uint64_t dividend using 64-bit magic multiplier
    __device__ static constexpr uint64_t
    DoMagicDivision(uint64_t dividend, uint64_t multiplier, uint32_t shift)
    {
        uint64_t tmp = __umul64hi(dividend, multiplier);
        return (tmp + dividend) >> shift;
    }

    __host__ static constexpr uint64_t
    DoMagicDivision(uint64_t dividend, uint64_t multiplier, uint32_t shift)
    {
        uint64_t tmp = static_cast<__uint128_t>(dividend) * multiplier >> 64;
        return (tmp + dividend) >> shift;
    }

    // magic division for int64_t dividend (dividend must be non-negative)
    __device__ static constexpr int64_t
    DoMagicDivision(int64_t dividend_i64, uint64_t multiplier, uint32_t shift)
    {
        uint64_t dividend_u64 = static_cast<uint64_t>(dividend_i64);
        uint64_t tmp          = __umul64hi(dividend_u64, multiplier);
        return static_cast<int64_t>((tmp + dividend_u64) >> shift);
    }

    __host__ static constexpr int64_t
    DoMagicDivision(int64_t dividend_i64, uint64_t multiplier, uint32_t shift)
    {
        uint64_t dividend_u64 = static_cast<uint64_t>(dividend_i64);
        uint64_t tmp          = static_cast<__uint128_t>(dividend_u64) * multiplier >> 64;
        return static_cast<int64_t>((tmp + dividend_u64) >> shift);
    }
};

struct MDiv
{
    // 1 dword -> 3 dword storage
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift; // TODO: 8 bit is enough

    // prefer construct on host
    __host__ __device__ MDiv(uint32_t divisor_) : divisor(divisor_)
    {
        auto tmp = MagicDivision::CalculateMagicNumbers(divisor_);

        multiplier = tmp[Number<0>{}];
        shift      = tmp[Number<1>{}];
    }

    __host__ __device__ MDiv() : divisor(0), multiplier(0), shift(0) {}

    __host__ __device__ void update(uint32_t divisor_)
    {
        divisor  = divisor_;
        auto tmp = MagicDivision::CalculateMagicNumbers(divisor_);

        multiplier = tmp[Number<0>{}];
        shift      = tmp[Number<1>{}];
    }

    __host__ __device__ uint32_t div(uint32_t dividend_) const
    {
        return MagicDivision::DoMagicDivision(dividend_, multiplier, shift);
    }

    __host__ __device__ void
    divmod(uint32_t dividend_, uint32_t& quotient_, uint32_t& remainder_) const
    {
        quotient_  = div(dividend_);
        remainder_ = dividend_ - (quotient_ * divisor);
    }

    __host__ __device__ uint32_t get() const { return divisor; }
};

struct MDiv2
{
    // 1 dword -> 2 dword storage, divisor need compute from runtime
    uint32_t multiplier;
    uint32_t shift; // TODO: 8 bit is enough

    // prefer construct on host
    __host__ __device__ MDiv2(uint32_t divisor_)
    {
        auto tmp = MagicDivision::CalculateMagicNumbers(divisor_);

        multiplier = tmp[Number<0>{}];
        shift      = tmp[Number<1>{}];
    }

    __host__ __device__ MDiv2() : multiplier(0), shift(0) {}

    __host__ __device__ uint32_t div(uint32_t dividend_) const
    {
        return MagicDivision::DoMagicDivision(dividend_, multiplier, shift);
    }

    __host__ __device__ void
    divmod(uint32_t dividend_, uint32_t divisor_, uint32_t& quotient_, uint32_t& remainder_) const
    {
        quotient_  = div(dividend_);
        remainder_ = dividend_ - (quotient_ * divisor_);
    }
};

} // namespace ck
