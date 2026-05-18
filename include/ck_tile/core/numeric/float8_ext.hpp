// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/numeric/mxfp_scale.hpp"

namespace ck_tile {

/* vector scaled type conversion */
namespace impl {

using fp8x2_storage_t = ext_vector_t<float8_storage_t, 2>;
using fp8x8_storage_t = ext_vector_t<float8_storage_t, 8>;

#if CK_TILE_FP8_CVT_DEVICE
#if defined(__gfx950__)
// fp8 -> fp32 packed 2 vector instruction
template <typename VDstT, fp8_interpretation interpret>
CK_TILE_DEVICE VDstT cast_from_f8x2_scaled(fp8x2_storage_t v, float scale)
{
    static_assert(interpret == fp8_interpretation::E4M3_OCP ||
                      interpret == fp8_interpretation::E5M2_OCP,
                  "Do not support FNUZ FP8 type");

    union
    {
        fp8x2_storage_t v2f8x2[2];
        uint32_t i32val;
        uint16_t i16val[2];
    } val{{v, v}};

    if constexpr(interpret == fp8_interpretation::E4M3_OCP)
    {
        if constexpr(std::is_same_v<VDstT, fp32x2_t>)
            return __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(val.i16val[0], scale, 0);
        else if constexpr(std::is_same_v<VDstT, fp16x2_t>)
            return __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(val.i32val, scale, false);
        else if constexpr(std::is_same_v<VDstT, bf16x2_t>)
            return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(val.i32val, scale, false);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<VDstT, fp32x2_t>)
            return __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(val.i16val[0], scale, 0);
        else if constexpr(std::is_same_v<VDstT, fp16x2_t>)
            return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(val.i32val, scale, false);
        else if constexpr(std::is_same_v<VDstT, bf16x2_t>)
            return __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(val.i32val, scale, false);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
}
// fp32 -> fp8 sr
template <typename SrcT, fp8_interpretation interpret>
CK_TILE_DEVICE float8_storage_t cast_to_f8_scaled_sr(SrcT v, float scale)
{
    static_assert(interpret == fp8_interpretation::E4M3_OCP ||
                      interpret == fp8_interpretation::E5M2_OCP,
                  "Do not support FNUZ FP8 type");
    union
    {
        uint32_t ival;
        float8_storage_t v8f8[2];
    } ret{};

    // use HW clock for stochastic input multiply by incremented thread id
    auto thread_gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rng    = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (thread_gid + 1));

    if constexpr(interpret == fp8_interpretation::E4M3_OCP)
    {
        if constexpr(std::is_same_v<SrcT, fp32_t>)
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, v, rng, scale, 0);
        else if constexpr(std::is_same_v<SrcT, fp16_t>)
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_f16(ret.ival, v, rng, scale, 0);
        else if constexpr(std::is_same_v<SrcT, bf16_t>)
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16(ret.ival, v, rng, scale, 0);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<SrcT, fp32_t>)
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, v, rng, scale, 0);
        else if constexpr(std::is_same_v<SrcT, fp16_t>)
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_f16(ret.ival, v, rng, scale, 0);
        else if constexpr(std::is_same_v<SrcT, bf16_t>)
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16(ret.ival, v, rng, scale, 0);
        else
            static_assert(false_type::value, "Unsupported type.");
    }

    return ret.v8f8[0];
}
// fp32 -> fp8 rtn packed 2 vector instruction
template <typename VSrcT, fp8_interpretation interpret>
CK_TILE_DEVICE fp8x2_storage_t cast_to_f8x2_scaled_rtn(VSrcT v, float scale)
{
    static_assert(interpret == fp8_interpretation::E4M3_OCP ||
                      interpret == fp8_interpretation::E5M2_OCP,
                  "Do not support FNUZ FP8 type");
    typedef short shortx2_t __attribute__((ext_vector_type(2)));
    union
    {
        shortx2_t v2i16;
        fp8x2_storage_t v8f8[2];
    } ret{};

    if constexpr(interpret == fp8_interpretation::E4M3_OCP)
    {
        if constexpr(std::is_same_v<VSrcT, fp32x2_t>)
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(/*old_vdst*/ ret.v2i16,
                                                                 v[0],
                                                                 v[1],
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        else if constexpr(std::is_same_v<VSrcT, fp16x2_t>)
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(ret.v2i16, v, scale, false);
        else if constexpr(std::is_same_v<VSrcT, bf16x2_t>)
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(ret.v2i16, v, scale, false);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<VSrcT, fp32x2_t>)
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(/*old_vdst*/ ret.v2i16,
                                                                 v[0],
                                                                 v[1],
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        else if constexpr(std::is_same_v<VSrcT, fp16x2_t>)
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(ret.v2i16, v, scale, false);
        else if constexpr(std::is_same_v<VSrcT, bf16x2_t>)
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(ret.v2i16, v, scale, false);
        else
            static_assert(false_type::value, "Unsupported type.");
    }

    return ret.v8f8[0];
}
#elif defined(__gfx125__)
// fp8 -> fp32 packed 8 vector instruction
template <typename VDstT, fp8_interpretation interpret, int Opsel = 0>
CK_TILE_DEVICE VDstT cast_from_f8x8_scaled(fp8x8_storage_t v, uint32_t scale)
{
    static_assert(interpret == fp8_interpretation::E4M3_OCP ||
                      interpret == fp8_interpretation::E5M2_OCP,
                  "Do not support FNUZ FP8 type");

    if constexpr(interpret == fp8_interpretation::E4M3_OCP)
    {
        if constexpr(std::is_same_v<VDstT, fp32x8_t>)
            return __builtin_amdgcn_cvt_scale_pk8_f32_fp8(bit_cast<uint32x2_t>(v), scale, Opsel);
        else if constexpr(std::is_same_v<VDstT, fp16x8_t>)
            return __builtin_amdgcn_cvt_scale_pk8_f16_fp8(bit_cast<uint32x2_t>(v), scale, Opsel);
        else if constexpr(std::is_same_v<VDstT, bf16x8_t>)
            return __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(bit_cast<uint32x2_t>(v), scale, Opsel);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(std::is_same_v<VDstT, fp32x8_t>)
            return __builtin_amdgcn_cvt_scale_pk8_f32_bf8(bit_cast<uint32x2_t>(v), scale, Opsel);
        else if constexpr(std::is_same_v<VDstT, fp16x8_t>)
            return __builtin_amdgcn_cvt_scale_pk8_f16_bf8(bit_cast<uint32x2_t>(v), scale, Opsel);
        else if constexpr(std::is_same_v<VDstT, bf16x8_t>)
            return __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(bit_cast<uint32x2_t>(v), scale, Opsel);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
}

// fp32 -> fp8 packed 8 vector instruction
template <fp8_interpretation interpret, bool stochastic_rounding = false, typename VSrcT>
CK_TILE_DEVICE fp8x8_storage_t cast_to_f8x8_scaled(VSrcT v, float scale)
{
    static_assert(interpret == fp8_interpretation::E4M3_OCP ||
                      interpret == fp8_interpretation::E5M2_OCP,
                  "Do not support FNUZ FP8 type");
    uint32x2_t ival;

    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        auto thread_gid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (thread_gid + 1));

        if constexpr(interpret == fp8_interpretation::E4M3_OCP)
        {
            if constexpr(std::is_same_v<VSrcT, fp32x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32(v, rng, scale);
            else if constexpr(std::is_same_v<VSrcT, fp16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f16(v, rng, scale);
            else if constexpr(std::is_same_v<VSrcT, bf16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_bf16(v, rng, scale);
            else
                static_assert(false_type::value, "Unsupported type.");
        }
        else
        {
            if constexpr(std::is_same_v<VSrcT, fp32x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32(v, rng, scale);
            else if constexpr(std::is_same_v<VSrcT, fp16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f16(v, rng, scale);
            else if constexpr(std::is_same_v<VSrcT, bf16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_bf16(v, rng, scale);
            else
                static_assert(false_type::value, "Unsupported type.");
        }
    }
    else
    {
        if constexpr(interpret == fp8_interpretation::E4M3_OCP)
        {
            if constexpr(std::is_same_v<VSrcT, fp32x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v, scale);
            else if constexpr(std::is_same_v<VSrcT, fp16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f16(v, scale);
            else if constexpr(std::is_same_v<VSrcT, bf16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(v, scale);
            else
                static_assert(false_type::value, "Unsupported type.");
        }
        else
        {
            if constexpr(std::is_same_v<VSrcT, fp32x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32(v, scale);
            else if constexpr(std::is_same_v<VSrcT, fp16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_pk8_bf8_f16(v, scale);
            else if constexpr(std::is_same_v<VSrcT, bf16x8_t>)
                ival = __builtin_amdgcn_cvt_scalef32_pk8_bf8_bf16(v, scale);
            else
                static_assert(false_type::value, "Unsupported type.");
        }
    }

    return bit_cast<fp8x8_storage_t>(ival);
}
#endif
#endif // CK_TILE_FP8_CVT_DEVICE

template <fp8_interpretation interpret, typename VDstT>
CK_TILE_HOST_DEVICE VDstT from_float8x8(fp8x8_storage_t x, float scale)
{
    [[maybe_unused]] constexpr int N = vector_traits<VDstT>::vector_size;
#if defined(__gfx950__)
    using DstT   = typename vector_traits<VDstT>::scalar_type;
    using V2DstT = ext_vector_t<DstT, 2>;
    static_assert(N % 2 == 0, "Unsupport vector size");
    constexpr int N_v2 = N / 2;
    union
    {
        VDstT v8;
        V2DstT v2[N_v2];
    } res{};

    if constexpr(N_v2 >= 1)
        res.v2[0] =
            impl::cast_from_f8x2_scaled<V2DstT, interpret>(fp8x2_storage_t{x[0], x[1]}, scale);
    if constexpr(N_v2 >= 2)
        res.v2[1] =
            impl::cast_from_f8x2_scaled<V2DstT, interpret>(fp8x2_storage_t{x[2], x[3]}, scale);
    if constexpr(N_v2 >= 3)
        res.v2[2] =
            impl::cast_from_f8x2_scaled<V2DstT, interpret>(fp8x2_storage_t{x[4], x[5]}, scale);
    if constexpr(N_v2 >= 4)
        res.v2[3] =
            impl::cast_from_f8x2_scaled<V2DstT, interpret>(fp8x2_storage_t{x[6], x[7]}, scale);

    return res.v8;
#elif defined(__gfx125__)
    Packed4Scale_E8M0 pkscale(0, 0, 0, scale);
    return impl::cast_from_f8x8_scaled<VDstT, interpret>(x, pkscale.data());
#else
    using DstT = typename vector_traits<VDstT>::scalar_type;
    union
    {
        VDstT v8;
        DstT v[8];
    } res{};
    using SrcT = std::conditional_t<interpret == fp8_interpretation::E4M3_OCP ||
                                        interpret == fp8_interpretation::E4M3_FNUZ,
                                    fp8_t,
                                    bf8_t>;
#pragma unroll
    for(int i = 0; i < N; ++i)
    {
        res.v[i] = impl::run_cast_from_f8<SrcT, DstT>(bit_cast<SrcT>(x[i])) * scale;
    }
    return res.v8;
#endif
}

template <fp8_interpretation interpret, bool stochastic_rounding = false, typename VSrcT>
CK_TILE_HOST_DEVICE fp8x8_storage_t to_float8x8(VSrcT x, [[maybe_unused]] float scale)
{
    [[maybe_unused]] constexpr int N = vector_traits<VSrcT>::vector_size;

#if defined(__gfx950__)
    using SrcT = typename vector_traits<VSrcT>::scalar_type;
    static_assert(N % 2 == 0, "Unsupport vector size");
    using V2SrcT   = ext_vector_t<SrcT, 2>;
    const int N_v2 = N / 2;
    union
    {
        fp8x8_storage_t v8;
        fp8x2_storage_t v2[N_v2];
        float8_storage_t v[N];
    } res{};
    union
    {
        VSrcT v8;
        V2SrcT v2[N_v2];
        SrcT v[N];
    } in{x};

    if constexpr(stochastic_rounding)
    {
        if constexpr(N >= 1)
            res.v[0] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[0], scale);
        if constexpr(N >= 2)
            res.v[1] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[1], scale);
        if constexpr(N >= 3)
            res.v[2] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[2], scale);
        if constexpr(N >= 4)
            res.v[3] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[3], scale);
        if constexpr(N >= 5)
            res.v[4] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[4], scale);
        if constexpr(N >= 6)
            res.v[5] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[5], scale);
        if constexpr(N >= 7)
            res.v[6] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[6], scale);
        if constexpr(N >= 8)
            res.v[7] = impl::cast_to_f8_scaled_sr<SrcT, interpret>(in.v[7], scale);
    }
    else
    {
        if constexpr(N_v2 >= 1)
            res.v2[0] = impl::cast_to_f8x2_scaled_rtn<V2SrcT, interpret>(in.v2[0], scale);
        if constexpr(N_v2 >= 2)
            res.v2[1] = impl::cast_to_f8x2_scaled_rtn<V2SrcT, interpret>(in.v2[1], scale);
        if constexpr(N_v2 >= 3)
            res.v2[2] = impl::cast_to_f8x2_scaled_rtn<V2SrcT, interpret>(in.v2[2], scale);
        if constexpr(N_v2 >= 4)
            res.v2[3] = impl::cast_to_f8x2_scaled_rtn<V2SrcT, interpret>(in.v2[3], scale);
    }
    return res.v8;
#elif defined(__gfx125__)
    return impl::cast_to_f8x8_scaled<interpret, stochastic_rounding>(x, scale);
#else
    using SrcT         = typename vector_traits<VSrcT>::scalar_type;
    constexpr int seed = 42;
    uint32_t rng       = prand_generator_t<SrcT, seed>{}(reinterpret_cast<uintptr_t>(&x),
                                                   static_cast<SrcT>(detail::get_from_lane<0>(x)));
    union
    {
        fp8x8_storage_t v8;
        float8_storage_t v[8];
    } res{};
    constexpr bool clip = true;
    using DstT          = std::conditional_t<interpret == fp8_interpretation::E4M3_OCP ||
                                                 interpret == fp8_interpretation::E4M3_FNUZ,
                                             fp8_t,
                                             bf8_t>;

#pragma unroll
    for(int i = 0; i < N; ++i)
    {
        float scaled_val = static_cast<float>(x[i]) / scale;
        res.v[i] = impl::run_cast_to_f8<float, DstT, clip, stochastic_rounding>(scaled_val, rng);
    }

    return res.v8;
#endif
}

} // namespace impl

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr fp8x8_t fp32x8_to_fp8x8(const fp32x8_t& x, float scale)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return bit_cast<fp8x8_t>(
            impl::to_float8x8<numeric_traits<fp8_t>::f8_interpret, false>(x, scale));
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return bit_cast<fp8x8_t>(
            impl::to_float8x8<numeric_traits<fp8_t>::f8_interpret, true>(x, scale));
    }
    else
    {
        return bit_cast<fp8x8_t>(impl::fp8x8_storage_t{0});
    }
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr bf8x8_t fp32x8_to_bf8x8(const fp32x8_t& x, float scale)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return bit_cast<bf8x8_t>(
            impl::to_float8x8<numeric_traits<bf8_t>::f8_interpret, false>(x, scale));
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return bit_cast<bf8x8_t>(
            impl::to_float8x8<numeric_traits<bf8_t>::f8_interpret, true>(x, scale));
    }
    else
    {
        return bit_cast<bf8x8_t>(impl::fp8x8_storage_t{0});
    }
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr fp8x8_t fp16x8_to_fp8x8(const fp16x8_t& x, float scale)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return bit_cast<fp8x8_t>(
            impl::to_float8x8<numeric_traits<fp8_t>::f8_interpret, false>(x, scale));
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return bit_cast<fp8x8_t>(
            impl::to_float8x8<numeric_traits<fp8_t>::f8_interpret, true>(x, scale));
    }
    else
    {
        return bit_cast<fp8x8_t>(impl::fp8x8_storage_t{0});
    }
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr bf8x8_t fp16x8_to_bf8x8(const fp16x8_t& x, float scale)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return bit_cast<bf8x8_t>(
            impl::to_float8x8<numeric_traits<bf8_t>::f8_interpret, false>(x, scale));
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return bit_cast<bf8x8_t>(
            impl::to_float8x8<numeric_traits<bf8_t>::f8_interpret, true>(x, scale));
    }
    else
    {
        return bit_cast<bf8x8_t>(impl::fp8x8_storage_t{0});
    }
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr fp8x8_t bf16x8_to_fp8x8(const bf16x8_t& x, float scale)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return bit_cast<fp8x8_t>(
            impl::to_float8x8<numeric_traits<fp8_t>::f8_interpret, false>(x, scale));
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return bit_cast<fp8x8_t>(
            impl::to_float8x8<numeric_traits<fp8_t>::f8_interpret, true>(x, scale));
    }
    else
    {
        return bit_cast<fp8x8_t>(impl::fp8x8_storage_t{0});
    }
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr bf8x8_t bf16x8_to_bf8x8(const bf16x8_t& x, float scale)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return bit_cast<bf8x8_t>(
            impl::to_float8x8<numeric_traits<bf8_t>::f8_interpret, false>(x, scale));
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return bit_cast<bf8x8_t>(
            impl::to_float8x8<numeric_traits<bf8_t>::f8_interpret, true>(x, scale));
    }
    else
    {
        return bit_cast<bf8x8_t>(impl::fp8x8_storage_t{0});
    }
}

CK_TILE_HOST_DEVICE constexpr fp32x8_t fp8x8_to_fp32x8(const fp8x8_t& x, float scale)
{
    impl::fp8x8_storage_t x_in = bit_cast<impl::fp8x8_storage_t>(x);
    return impl::from_float8x8<numeric_traits<fp8_t>::f8_interpret, fp32x8_t>(x_in, scale);
}

CK_TILE_HOST_DEVICE constexpr fp32x8_t bf8x8_to_fp32x8(const bf8x8_t& x, float scale)
{
    impl::fp8x8_storage_t x_in = bit_cast<impl::fp8x8_storage_t>(x);
    return impl::from_float8x8<numeric_traits<bf8_t>::f8_interpret, fp32x8_t>(x_in, scale);
}

CK_TILE_HOST_DEVICE constexpr fp16x8_t fp8x8_to_fp16x8(const fp8x8_t& x, float scale)
{
    impl::fp8x8_storage_t x_in = bit_cast<impl::fp8x8_storage_t>(x);
    return impl::from_float8x8<numeric_traits<fp8_t>::f8_interpret, fp16x8_t>(x_in, scale);
}

CK_TILE_HOST_DEVICE constexpr fp16x8_t bf8x8_to_fp16x8(const bf8x8_t& x, float scale)
{
    impl::fp8x8_storage_t x_in = bit_cast<impl::fp8x8_storage_t>(x);
    return impl::from_float8x8<numeric_traits<bf8_t>::f8_interpret, fp16x8_t>(x_in, scale);
}

CK_TILE_HOST_DEVICE constexpr bf16x8_t fp8x8_to_bf16x8(const fp8x8_t& x, float scale)
{
    impl::fp8x8_storage_t x_in = bit_cast<impl::fp8x8_storage_t>(x);
    return impl::from_float8x8<numeric_traits<fp8_t>::f8_interpret, bf16x8_t>(x_in, scale);
}

CK_TILE_HOST_DEVICE constexpr bf16x8_t bf8x8_to_bf16x8(const bf8x8_t& x, float scale)
{
    impl::fp8x8_storage_t x_in = bit_cast<impl::fp8x8_storage_t>(x);
    return impl::from_float8x8<numeric_traits<bf8_t>::f8_interpret, bf16x8_t>(x_in, scale);
}
} // namespace ck_tile
