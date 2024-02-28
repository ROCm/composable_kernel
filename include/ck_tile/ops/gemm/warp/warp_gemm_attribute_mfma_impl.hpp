// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// FP16
struct WarpGemmAttributeMfmaImplF16F16F32M32N32K8
{
    using ADataType = half_t;
    using BDataType = half_t;
    using CDataType = float;

    using AVecType = typename vector_type<half_t, 4>::type;
    using BVecType = typename vector_type<half_t, 4>::type;
    using CVecType = typename vector_type<float, 16>::type;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 8;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

struct WarpGemmAttributeMfmaImplF16F16F32M16N16K16
{
    using ADataType = half_t;
    using BDataType = half_t;
    using CDataType = float;

    using AVecType = typename vector_type<half_t, 4>::type;
    using BVecType = typename vector_type<half_t, 4>::type;
    using CVecType = typename vector_type<float, 4>::type;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

// Bf16
struct WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8
{
    using ADataType = bhalf_t;
    using BDataType = bhalf_t;
    using CDataType = float;

    using AVecType = typename vector_type<bhalf_t, 4>::type;
    using BVecType = typename vector_type<bhalf_t, 4>::type;
    using CVecType = typename vector_type<float, 16>::type;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 8;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

struct WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16
{
    using ADataType = bhalf_t;
    using BDataType = bhalf_t;
    using CDataType = float;

    using AVecType = typename vector_type<bhalf_t, 4>::type;
    using BVecType = typename vector_type<bhalf_t, 4>::type;
    using CVecType = typename vector_type<float, 4>::type;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

// FP8
template <typename AType_, typename BType_>
struct WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base
{
    using ADataType = AType_;
    using BDataType = BType_;
    using CDataType = float;

    using AVecType = typename vector_type<ADataType, 8>::type;
    using BVecType = typename vector_type<BDataType, 8>::type;
    using CVecType = typename vector_type<CDataType, 16>::type;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        if constexpr(is_same_v<ADataType, fp8_t> && is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
        else if constexpr(is_same_v<ADataType, fp8_t> && is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
        else if constexpr(is_same_v<ADataType, bf8_t> && is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
        else if constexpr(is_same_v<ADataType, bf8_t> && is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), c_vec, 0, 0, 0);
#else
        vector_type<ADataType, 8> a_(a_vec);
        vector_type<BDataType, 8> b_(b_vec);

        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 = type_convert<float>(a_.template AsType<ADataType>()[number<k>{}]);
            float b_f32 = type_convert<float>(b_.template AsType<BDataType>()[number<k>{}]);

            c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c_vec, 0, 0, 0);
        });
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(is_same_v<ADataType, fp8_t> && is_same_v<BDataType, fp8_t>)
            return __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0);
        else if constexpr(is_same_v<ADataType, fp8_t> && is_same_v<BDataType, bf8_t>)
            return __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0);
        else if constexpr(is_same_v<ADataType, bf8_t> && is_same_v<BDataType, fp8_t>)
            return __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0);
        else if constexpr(is_same_v<ADataType, bf8_t> && is_same_v<BDataType, bf8_t>)
            return __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<long>(a_vec), bit_cast<long>(b_vec), CVecType{0.f}, 0, 0, 0);
    }
};

using WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, fp8_t>;
using WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, bf8_t>;
using WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<bf8_t, fp8_t>;
using WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<bf8_t, bf8_t>;

} // namespace ck_tile
