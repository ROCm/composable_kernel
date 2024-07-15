// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#pragma once

namespace ck {

template <index_t MPerWave, index_t NPerWave>
struct intrin_smfmac_f32_16x16x32f16;

template <>
struct intrin_smfmac_f32_16x16x32f16<16, 16>
{
    template <class FloatC>
    __device__ static void
    Run(const half4_t& reg_a, const half8_t& reg_b, const int32_t& reg_idx, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], reg_idx, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_idx;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_smfmac_f32_16x16x32bf16;

template <>
struct intrin_smfmac_f32_16x16x32bf16<16, 16>
{
    template <class FloatC>
    __device__ static void
    Run(const bhalf4_t& reg_a, const bhalf8_t& reg_b, const int32_t& reg_idx, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], reg_idx, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_idx;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_smfmac_f32_32x32x16f16;

template <>
struct intrin_smfmac_f32_32x32x16f16<32, 32>
{
    template <class FloatC>
    __device__ static void
    Run(const half4_t& reg_a, const half8_t& reg_b, const int32_t& reg_idx, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_smfmac_f32_32x32x16_f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], reg_idx, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_idx;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_smfmac_f32_32x32x16bf16;

template <>
struct intrin_smfmac_f32_32x32x16bf16<32, 32>
{
    template <class FloatC>
    __device__ static void
    Run(const bhalf4_t& reg_a, const bhalf8_t& reg_b, const int32_t& reg_idx, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], reg_idx, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_idx;
#endif
    }
};

} // namespace ck
