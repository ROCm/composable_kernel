// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_AMD_WMMA_HPP
#define CK_AMD_WMMA_HPP

#include "ck/utility/amd_inline_asm.hpp"
#include "data_type.hpp"
#include "dtype_fp64.hpp"
// TODO: Add arch limitation
namespace ck {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx11_generic__)
#define __gfx11__
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx120__
#endif

#if defined(__gfx1250__)
#define __gfx125__
#endif

/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half16_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<half16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<bhalf16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

/********************************WAVE64 MODE***********************************************/

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w64;

template <>
struct intrin_wmma_f32_16x16x16_f16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w64;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x4_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx12
/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx125x
/********************************WAVE32 MODE***********************************************/
// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x32_f16;

template <>
struct intrin_wmma_f16_16x16x32_f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x32_f16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<half8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16_16x16x32_bf16;

template <>
struct intrin_wmma_bf16_16x16x32_bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx125__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<bhalf8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x32_f16;

template <>
struct intrin_wmma_f32_16x16x32_f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x32_bf16;

template <>
struct intrin_wmma_f32_16x16x32_bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16f32_16x16x32_bf16;

template <>
struct intrin_wmma_bf16f32_16x16x32_bf16<16, 16>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx125__)
        reg_d
            .template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b>
struct intrin_wmma_i32_16x16x64_iu8;

template <bool neg_a, bool neg_b>
struct intrin_wmma_i32_16x16x64_iu8<16, 16, neg_a, neg_b>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_a, const int8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x64_iu8(neg_a,
                                                   bit_cast<int32x8_t>(reg_a),
                                                   neg_b,
                                                   bit_cast<int32x8_t>(reg_b),
                                                   reg_c.template AsType<int32x8_t>()[Number<0>{}],
                                                   false,
                                                   false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_f8f8;
template <>
struct intrin_wmma_f32_16x16x64_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_f8bf8;
template <>
struct intrin_wmma_f32_16x16x64_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_bf8f8;
template <>
struct intrin_wmma_f32_16x16x64_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_bf8bf8;
template <>
struct intrin_wmma_f32_16x16x64_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_f8f8;
template <>
struct intrin_wmma_f16_16x16x64_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_f8bf8;
template <>
struct intrin_wmma_f16_16x16x64_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_bf8f8;
template <>
struct intrin_wmma_f16_16x16x64_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_bf8bf8;
template <>
struct intrin_wmma_f16_16x16x64_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_f8f8;
template <>
struct intrin_wmma_f32_16x16x128_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_f8bf8;
template <>
struct intrin_wmma_f32_16x16x128_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_bf8f8;
template <>
struct intrin_wmma_f32_16x16x128_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_bf8bf8;
template <>
struct intrin_wmma_f32_16x16x128_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_f8f8;
template <>
struct intrin_wmma_f16_16x16x128_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {

#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_f8bf8;
template <>
struct intrin_wmma_f16_16x16x128_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_bf8f8;
template <>
struct intrin_wmma_f16_16x16x128_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_bf8bf8;
template <>
struct intrin_wmma_f16_16x16x128_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x4_f32;

template <>
struct intrin_wmma_f32_16x16x4_f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float2_t& reg_a, const float2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x4_f32(0,
                                                  bit_cast<float2_t>(reg_a),
                                                  0,
                                                  bit_cast<float2_t>(reg_b),
                                                  0,
                                                  reg_c.template AsType<float8_t>()[Number<0>{}],
                                                  false,
                                                  false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

namespace wmma_impl {
#ifndef CK_CODE_GEN_RTC
// utils for f8f6f4 instructions
template <typename T>
struct ScaleTypeSelector
{
};

// use int32_t for backward compatibility
template <>
struct ScaleTypeSelector<int32_t>
{
    static constexpr int value = 0x0;
};

template <>
struct ScaleTypeSelector<e8m0x4_bexp_t>
{
    static constexpr int value = 0x0;
};

template <>
struct ScaleTypeSelector<e8m0x8_bexp_t>
{
    static constexpr int value = 0x0;
};

template <>
struct ScaleTypeSelector<e5m3x4_scale_t>
{
    static constexpr int value = 0x1;
};

template <>
struct ScaleTypeSelector<e5m3x8_scale_t>
{
    static constexpr int value = 0x1;
};

template <>
struct ScaleTypeSelector<e4m3x4_scale_t>
{
    static constexpr int value = 0x2;
};

template <>
struct ScaleTypeSelector<e4m3x8_scale_t>
{
    static constexpr int value = 0x2;
};

enum InputFormat : uint8_t
{
    E4M3 = 0x0,
    E5M2 = 0x1,
    E2M3 = 0x2,
    E3M2 = 0x3,
    E2M1 = 0x4
};

template <typename T>
struct MxTypeSelector
{
};

template <>
struct MxTypeSelector<f8x64_t>
{
    static constexpr InputFormat value = InputFormat::E4M3;
};

template <>
struct MxTypeSelector<bf8x64_t>
{
    static constexpr InputFormat value = InputFormat::E5M2;
};

template <>
struct MxTypeSelector<f6x64_t>
{
    static constexpr InputFormat value = InputFormat::E2M3;
};

template <>
struct MxTypeSelector<f6x16x4_t>
{
    static constexpr InputFormat value = InputFormat::E2M3;
};

template <>
struct MxTypeSelector<bf6x64_t>
{
    static constexpr InputFormat value = InputFormat::E3M2;
};

template <>
struct MxTypeSelector<bf6x16x4_t>
{
    static constexpr InputFormat value = InputFormat::E3M2;
};

template <>
struct MxTypeSelector<f4x64_t>
{
    static constexpr InputFormat value = InputFormat::E2M1;
};

template <typename MxType>
constexpr auto bit_cast_mx_reg(const MxType& reg_mx)
{
    if constexpr(sizeof(MxType) == sizeof(f8x64_t))
    {
        return bit_cast<int32x16_t>(reg_mx);
    }
    else if constexpr(sizeof(MxType) == sizeof(f4x64_t))
    {
        return int32x16_t{
            bit_cast<int32x8_t>(reg_mx)[0],
            bit_cast<int32x8_t>(reg_mx)[1],
            bit_cast<int32x8_t>(reg_mx)[2],
            bit_cast<int32x8_t>(reg_mx)[3],
            bit_cast<int32x8_t>(reg_mx)[4],
            bit_cast<int32x8_t>(reg_mx)[5],
            bit_cast<int32x8_t>(reg_mx)[6],
            bit_cast<int32x8_t>(reg_mx)[7],
        };
    }
    else
    {
        static_assert(0);
    }
}

template <>
constexpr auto bit_cast_mx_reg(const bf6x64_t& reg_mx)
{
    int32x6_t arg_mx_0 = bit_cast<int32x6_t>(reg_mx.AsType<bf6x32_pk_t>()[Number<0>{}]);
    int32x6_t arg_mx_1 = bit_cast<int32x6_t>(reg_mx.AsType<bf6x32_pk_t>()[Number<1>{}]);
    return int32x16_t{arg_mx_0[0],
                      arg_mx_0[1],
                      arg_mx_0[2],
                      arg_mx_0[3],
                      arg_mx_0[4],
                      arg_mx_0[5],
                      arg_mx_1[0],
                      arg_mx_1[1],
                      arg_mx_1[2],
                      arg_mx_1[3],
                      arg_mx_1[4],
                      arg_mx_1[5]};
}

template <>
constexpr auto bit_cast_mx_reg(const f6x64_t& reg_mx)
{
    int32x6_t arg_mx_0 = bit_cast<int32x6_t>(reg_mx.AsType<f6x32_pk_t>()[Number<0>{}]);
    int32x6_t arg_mx_1 = bit_cast<int32x6_t>(reg_mx.AsType<f6x32_pk_t>()[Number<1>{}]);
    return int32x16_t{arg_mx_0[0],
                      arg_mx_0[1],
                      arg_mx_0[2],
                      arg_mx_0[3],
                      arg_mx_0[4],
                      arg_mx_0[5],
                      arg_mx_1[0],
                      arg_mx_1[1],
                      arg_mx_1[2],
                      arg_mx_1[3],
                      arg_mx_1[4],
                      arg_mx_1[5]};
}

template <>
constexpr auto bit_cast_mx_reg(const f6x16x4_t& reg_mx)
{
    auto a0 = reg_mx.template AsType<f6x16_pk_t>()[Number<0>{}].data_;
    auto a1 = reg_mx.template AsType<f6x16_pk_t>()[Number<1>{}].data_;
    auto a2 = reg_mx.template AsType<f6x16_pk_t>()[Number<2>{}].data_;
    auto a3 = reg_mx.template AsType<f6x16_pk_t>()[Number<3>{}].data_;
    return int32x16_t{static_cast<int32_t>(a0[0]),
                      static_cast<int32_t>(a0[1]),
                      static_cast<int32_t>(a0[2]),
                      static_cast<int32_t>(a1[0]),
                      static_cast<int32_t>(a1[1]),
                      static_cast<int32_t>(a1[2]),
                      static_cast<int32_t>(a2[0]),
                      static_cast<int32_t>(a2[1]),
                      static_cast<int32_t>(a2[2]),
                      static_cast<int32_t>(a3[0]),
                      static_cast<int32_t>(a3[1]),
                      static_cast<int32_t>(a3[2])};
}

template <>
constexpr auto bit_cast_mx_reg(const bf6x16x4_t& reg_mx)
{
    auto a0 = reg_mx.template AsType<bf6x16_pk_t>()[Number<0>{}].data_;
    auto a1 = reg_mx.template AsType<bf6x16_pk_t>()[Number<1>{}].data_;
    auto a2 = reg_mx.template AsType<bf6x16_pk_t>()[Number<2>{}].data_;
    auto a3 = reg_mx.template AsType<bf6x16_pk_t>()[Number<3>{}].data_;
    return int32x16_t{static_cast<int32_t>(a0[0]),
                      static_cast<int32_t>(a0[1]),
                      static_cast<int32_t>(a0[2]),
                      static_cast<int32_t>(a1[0]),
                      static_cast<int32_t>(a1[1]),
                      static_cast<int32_t>(a1[2]),
                      static_cast<int32_t>(a2[0]),
                      static_cast<int32_t>(a2[1]),
                      static_cast<int32_t>(a2[2]),
                      static_cast<int32_t>(a3[0]),
                      static_cast<int32_t>(a3[1]),
                      static_cast<int32_t>(a3[2])};
}
#endif // #ifndef CK_CODE_GEN_RTC
} // namespace wmma_impl

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselA,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale_f32_16x16x128_f8f6f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselA, index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale_f32_16x16x128_f8f6f4<16,
                                              16,
                                              ScaleOpselA,
                                              ScaleOpselB,
                                              ScaleTypeA,
                                              ScaleTypeB>
{
    template <typename TypeA, typename TypeB, class FloatC>
    __device__ static void Run(const TypeA& reg_a,
                               const ScaleTypeA& scale_a,
                               const TypeB& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility

#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::MxTypeSelector<TypeA>::value, // OPSEL
                wmma_impl::bit_cast_mx_reg(reg_a),
                wmma_impl::MxTypeSelector<TypeB>::value, // OPSEL_HI
                wmma_impl::bit_cast_mx_reg(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                // M=laneId % 16 [7:0] K=0..31; [15:8] K=32..63; [23:16] K=64..95; [31:24] K=96..127
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                // N=laneId % 16 [7:0] K=0..31; [15:8] K=32..63; [23:16] K=64..95; [31:24] K=96..127
                bit_cast<int32_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselA,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale16_f32_16x16x128_f8f6f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselA, index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale16_f32_16x16x128_f8f6f4<16,
                                                16,
                                                ScaleOpselA,
                                                ScaleOpselB,
                                                ScaleTypeA,
                                                ScaleTypeB>
{
    template <typename TypeA, typename TypeB, class FloatC>
    __device__ static void Run(const TypeA& reg_a,
                               const ScaleTypeA& scale_a,
                               const TypeB& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::MxTypeSelector<TypeA>::value, // OPSEL
                wmma_impl::bit_cast_mx_reg(reg_a),
                wmma_impl::MxTypeSelector<TypeB>::value, // OPSEL_HI
                wmma_impl::bit_cast_mx_reg(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale_f32_32x16x128_f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale_f32_32x16x128_f4<32, 16, ScaleOpselB, ScaleTypeA, ScaleTypeB>
{
    template <class FloatC>
    __device__ static void Run(const f4x128_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> ||
                          is_same_v<ScaleTypeA, e5m3x4_scale_t> ||
                          is_same_v<ScaleTypeA, e4m3x4_scale_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t, e5m3x4_scale_t, or e4m3x4_scale_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> ||
                          is_same_v<ScaleTypeB, e5m3x4_scale_t> ||
                          is_same_v<ScaleTypeB, e4m3x4_scale_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t, e5m3x4_scale_t, or e4m3x4_scale_t");
#if defined(__gfx125__)
        int32x16_t arg_a = bit_cast<int32x16_t>(reg_a);
        int32x8_t arg_b  = bit_cast<int32x8_t>(reg_b);
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(
                arg_a,
                arg_b,
                0,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // fix ScaleOpselA as 1
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value,
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value,
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale16_f32_32x16x128_f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale16_f32_32x16x128_f4<32, 16, ScaleOpselB, ScaleTypeA, ScaleTypeB>
{
    template <class FloatC>
    __device__ static void Run(const f4x128_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeA, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeA, e4m3x8_scale_t>,
                      "ScaleTypeA must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeB, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeB, e4m3x8_scale_t>,
                      "ScaleTypeB must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
#if defined(__gfx125__)
        int32x16_t arg_a = bit_cast<int32x16_t>(reg_a);
        int32x8_t arg_b  = bit_cast<int32x8_t>(reg_b);
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(
                arg_a,
                arg_b,
                0,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // fix ScaleOpselA as 1
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value,
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value,
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

} // namespace ck
#endif
