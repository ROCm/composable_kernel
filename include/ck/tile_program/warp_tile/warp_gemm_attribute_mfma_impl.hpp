// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tile_program {
namespace warp {

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
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // FIXME: Is this correct?
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
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // FIXME: Is this correct?
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
