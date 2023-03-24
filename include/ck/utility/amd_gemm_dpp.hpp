// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_GEMM_DPP_HPP
#define CK_AMD_GEMM_DPP_HPP

#include "data_type.hpp"

namespace ck {

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave, index_t KPerWave>
struct intrin_dpp_f32_8x8x8_f16;

template <>
struct intrin_dpp_f32_8x8x8_f16<8, 8, 8>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx1030__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_update_dpp(
            reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: int8, dst: int32
template <index_t MPerWave, index_t NPerWave, index_t KPerWave>
struct intrin_dpp_i32_8x8x8_i8;

template <>
struct intrin_dpp_i32_8x8x8_i8<8, 8, 8>
{
    template <class IntC>
    __device__ static void Run(const int8_t& reg_a, const int8_t& reg_b, IntC& reg_c)
    {
#if defined(__gfx1030__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) = __builtin_amdgcn_update_dpp(
            reg_a, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

} // namespace ck
#endif
