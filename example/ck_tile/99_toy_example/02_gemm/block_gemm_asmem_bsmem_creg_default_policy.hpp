// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

#include "config.h"

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCReg
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
#if defined(NAIVE_IMPLEMENTATION)
#pragma message ("mfma m32 n32 k8")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, 2, 2);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution{}, 2, 2);
        }
#elif defined(USING_MFMA_32x32x_8x2)
#pragma message ("mfma m32 n32 k16")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, 2, 2);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution{}, 2, 2);
        }
#elif defined(USING_MFMA_16x16x16)
#pragma message ("mfma m16 n16 k16")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{}, 2, 2);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{}, 2, 2);
        }
#elif defined(USING_MFMA_16x16x_16x2)
#pragma message ("mfma m16 n16 k32")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{}, 2, 2);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution{}, 2, 2);
        }
#endif
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
