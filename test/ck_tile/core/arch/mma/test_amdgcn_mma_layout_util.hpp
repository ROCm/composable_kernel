// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"

#include <cassert>
#include <cstdint>

namespace {

using namespace ck_tile;

/**
 * @class RegisterMapTraits
 * @brief Traits class that defines tile_distribution_encoding for each MmaOp
 * @tparam MmaOp amdgcn_mma specialization
 */
template <typename MmaOp>
struct RegisterMapTraits
{
    static_assert(sizeof(MmaOp) == 0, "RegisterMapTraits requires a specialization");
};

/**
 * @class RegisterMap
 * @brief Uses specialized RegisterMapTraits to get the encoding
 * @tparam MmaOp amdgcn_mma specialization
 */
template <typename MmaOp>
struct RegisterMap
{
    using Traits = RegisterMapTraits<MmaOp>;

    using AMap = core::arch::mma::TileDistrEncRegMap<typename Traits::AWarpDstrEncoding>;
    using BMap = core::arch::mma::TileDistrEncRegMap<typename Traits::BWarpDstrEncoding>;
    using CMap = core::arch::mma::TileDistrEncRegMap<typename Traits::CWarpDstrEncoding>;

    CK_TILE_HOST_DEVICE static auto Register2AMap(const uint32_t lane, const uint32_t vecIdx)
    {
        return AMap::calc_matrix_indices_from_lane_vector(static_cast<index_t>(lane),
                                                          static_cast<index_t>(vecIdx));
    }

    CK_TILE_HOST_DEVICE static auto Register2BMap(const uint32_t lane, const uint32_t vecIdx)
    {
        return BMap::calc_matrix_indices_from_lane_vector(static_cast<index_t>(lane),
                                                          static_cast<index_t>(vecIdx));
    }

    CK_TILE_HOST_DEVICE static auto Register2CMap(const uint32_t lane, const uint32_t vecIdx)
    {
        return CMap::calc_matrix_indices_from_lane_vector(static_cast<index_t>(lane),
                                                          static_cast<index_t>(vecIdx));
    }
};

// ====================== Specializations per target =====================

/**
 * @brief RegisterMapTraits for GFX12 WMMA 16x16x16_F16_F16_F32_GFX12
 */
template <typename CtrlFlags,
          typename CompilerTarget,
          ck_tile::core::arch::mma::MmaOpFamily OpFamily_>
struct RegisterMapTraits<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    OpFamily_,
    ck_tile::core::arch::enable_if_target_family_gfx12_t<CompilerTarget>>>
{
    using MmaOp = ck_tile::core::arch::mma::amdgcn_mma<ck_tile::fp16_t,
                                                       ck_tile::fp16_t,
                                                       ck_tile::fp32_t,
                                                       16u,
                                                       16u,
                                                       16u,
                                                       CtrlFlags,
                                                       CompilerTarget,
                                                       OpFamily_>;

    using MmaTraits = ck_tile::core::arch::mma::MmaOpTraits<MmaOp>;
    static constexpr index_t WaveSize =
        static_cast<index_t>(MmaTraits::CompilerTarget::WAVE_SIZE_ID);
    static constexpr index_t AVecSize = vector_traits<typename MmaTraits::AVecType>::vector_size;
    static constexpr index_t BVecSize = vector_traits<typename MmaTraits::BVecType>::vector_size;
    static constexpr index_t CVecSize = vector_traits<typename MmaTraits::CVecType>::vector_size;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;
    using kCPs2RHssMajor  = sequence<1, 2>;
    using kCPs2RHssMinor  = sequence<1, 0>;
    using kCYs2RHsMajor   = sequence<1, 1>;
    using kCYs2RHsMinor   = sequence<0, 2>;

    // TODO:  remove these and fix constants in amdgcn_mma
    static constexpr index_t kAMBlock     = 1;
    static constexpr index_t kBNBlock     = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 8;
    static constexpr index_t kCMLane      = 2;
    static constexpr index_t kCNLane      = 16;
    static constexpr index_t kCM0PerLane  = 1;
    static constexpr index_t kCM1PerLane  = 8;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<kAMLane>, sequence<kABK0PerLane, kABKLane, kABK1PerLane>>, // <16>, <1, 2, 8>
        tuple<kABPs2RHssMajor>,
        tuple<kABPs2RHssMinor>,
        kABYs2RHsMajor,
        kABYs2RHsMinor>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<kBNLane>, sequence<kABK0PerLane, kABKLane, kABK1PerLane>>, // <16>, <1, 2, 8>
        tuple<kABPs2RHssMajor>,
        tuple<kABPs2RHssMinor>,
        kABYs2RHsMajor,
        kABYs2RHsMinor>;

    using CWarpDstrEncoding =
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<kCM0PerLane, kCMLane, kCM1PerLane>,
                                         sequence<kCNLane>>, // <1, 2, 8>, <16>
                                   tuple<kCPs2RHssMajor>,
                                   tuple<kCPs2RHssMinor>,
                                   kCYs2RHsMajor,
                                   kCYs2RHsMinor>;
};

/**
 * @brief RegisterMapTraits for GFX9 MFMA 16x16x16_F16_F16_F32_GFX9
 */
template <typename CtrlFlags,
          typename CompilerTarget,
          ck_tile::core::arch::mma::MmaOpFamily OpFamily_>
struct RegisterMapTraits<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    OpFamily_,
    ck_tile::core::arch::enable_if_target_family_gfx9_t<CompilerTarget>>>
{
    using MmaOp = ck_tile::core::arch::mma::amdgcn_mma<ck_tile::fp16_t,
                                                       ck_tile::fp16_t,
                                                       ck_tile::fp32_t,
                                                       16u,
                                                       16u,
                                                       16u,
                                                       CtrlFlags,
                                                       CompilerTarget,
                                                       OpFamily_>;

    using MmaTraits = ck_tile::core::arch::mma::MmaOpTraits<MmaOp>;
    static constexpr index_t WaveSize =
        static_cast<index_t>(MmaTraits::CompilerTarget::WAVE_SIZE_ID);
    static constexpr index_t AVecSize = vector_traits<typename MmaTraits::AVecType>::vector_size;
    static constexpr index_t BVecSize = vector_traits<typename MmaTraits::BVecType>::vector_size;
    static constexpr index_t CVecSize = vector_traits<typename MmaTraits::CVecType>::vector_size;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<0, 0>;
    using kABYs2RHsMajor  = sequence<2>;
    using kABYs2RHsMinor  = sequence<1>;
    using kCPs2RHssMajor  = sequence<1, 2>;
    using kCPs2RHssMinor  = sequence<0, 0>;
    using kCYs2RHsMajor   = sequence<1>;
    using kCYs2RHsMinor   = sequence<1>;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<MmaOp::kAMLane>, sequence<MmaOp::kABKLane, MmaOp::kABKPerLane>>,
        tuple<kABPs2RHssMajor>,
        tuple<kABPs2RHssMinor>,
        kABYs2RHsMajor,
        kABYs2RHsMinor>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<MmaOp::kBNLane>, sequence<MmaOp::kABKLane, MmaOp::kABKPerLane>>,
        tuple<kABPs2RHssMajor>,
        tuple<kABPs2RHssMinor>,
        kABYs2RHsMajor,
        kABYs2RHsMinor>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<MmaOp::kCMLane, MmaOp::kCM1PerLane>, sequence<MmaOp::kCNLane>>,
        tuple<kCPs2RHssMajor>,
        tuple<kCPs2RHssMinor>,
        kCYs2RHsMajor,
        kCYs2RHsMinor>;
};

/**
 * @brief RegisterMapTraits for GFX11 WMMA 16x16x16_F16_F16_F32_GFX11
 */
template <typename CtrlFlags,
          typename CompilerTarget,
          ck_tile::core::arch::mma::MmaOpFamily OpFamily_>
struct RegisterMapTraits<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    OpFamily_,
    ck_tile::core::arch::enable_if_target_family_gfx11_t<CompilerTarget>>>
{
    using MmaOp = ck_tile::core::arch::mma::amdgcn_mma<ck_tile::fp16_t,
                                                       ck_tile::fp16_t,
                                                       ck_tile::fp32_t,
                                                       16u,
                                                       16u,
                                                       16u,
                                                       CtrlFlags,
                                                       CompilerTarget,
                                                       OpFamily_>;

    using MmaTraits = ck_tile::core::arch::mma::MmaOpTraits<MmaOp>;
    static constexpr index_t WaveSize =
        static_cast<index_t>(MmaTraits::CompilerTarget::WAVE_SIZE_ID);
    static constexpr index_t AVecSize = vector_traits<typename MmaTraits::AVecType>::vector_size;
    static constexpr index_t BVecSize = vector_traits<typename MmaTraits::BVecType>::vector_size;
    static constexpr index_t CVecSize = vector_traits<typename MmaTraits::CVecType>::vector_size;

    using kABPs2RHssMajor = sequence<0, 1>;
    using kABPs2RHssMinor = sequence<0, 0>;
    using kABYs2RHsMajor  = sequence<2>;
    using kABYs2RHsMinor  = sequence<0>;
    using kCPs2RHssMajor  = sequence<1, 2>;
    using kCPs2RHssMinor  = sequence<1, 0>;
    using kCYs2RHsMajor   = sequence<1>;
    using kCYs2RHsMinor   = sequence<0>;

    // TODO:  remove these and fix constants in amdgcn_mma
    static constexpr index_t kAMBlock     = 1;
    static constexpr index_t kBNBlock     = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 1;
    static constexpr index_t kABK1PerLane = 16;
    static constexpr index_t kCMLane      = 2;
    static constexpr index_t kCNLane      = 16;
    static constexpr index_t kCM0PerLane  = 8;
    static constexpr index_t kCM1PerLane  = 1;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<2>, // kRepeat
                                   tuple<sequence<kAMLane>, sequence<kABK1PerLane>>,
                                   tuple<kABPs2RHssMajor>,
                                   tuple<kABPs2RHssMinor>,
                                   kABYs2RHsMajor,
                                   kABYs2RHsMinor>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<2>, // kRepeat
                                   tuple<sequence<kBNLane>, sequence<kABK1PerLane>>,
                                   tuple<kABPs2RHssMajor>,
                                   tuple<kABPs2RHssMinor>,
                                   kABYs2RHsMajor,
                                   kABYs2RHsMinor>;

    using CWarpDstrEncoding =
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<kCM0PerLane, kCMLane>, sequence<kCNLane>>,
                                   tuple<kCPs2RHssMajor>,
                                   tuple<kCPs2RHssMinor>,
                                   kCYs2RHsMajor,
                                   kCYs2RHsMinor>;
};

// ========================================================================

} // namespace
