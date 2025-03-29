// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

#define mfma_m32_n32_k8 0
#define mfma_m32_n32_k16 0
#define mfma_m16_n16_k16 0
#define mfma_m16_n16_k32 1

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCReg
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
#if mfma_m32_n32_k8
#pragma message ("mfma m32 n32 k8")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
        return make_tuple(WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, 4, 1);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
        return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution{}, 4, 1);
        }
#elif mfma_m32_n32_k16
#pragma message ("mfma m32 n32 k16")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, 4, 1);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution{}, 4, 1);
        }
#elif mfma_m16_n16_k16
#pragma message("mfma m16 n16 k16")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
        return make_tuple(WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{}, 4, 1);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
        return make_tuple(WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{}, 4, 1);
        }
#elif mfma_m16_n16_k32
#pragma message("mfma m16 n16 k32")
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
        return make_tuple(WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{}, 4, 1);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
        return make_tuple(WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution{}, 4, 1);
        }
#endif
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
