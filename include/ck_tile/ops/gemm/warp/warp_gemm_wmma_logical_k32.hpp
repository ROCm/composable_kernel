// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Logical K32 arithmetic on gfx1250: keep the requested fragment geometry
// while zero-padding both operands for a physical K64 WMMA instruction.
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp"

namespace ck_tile {

template <typename F8>
struct WarpGemmWmmaLogicalK32Traits : WmmaTraitsBase<gfx12_t, F8, F8, float, 32>
{
    static_assert(std::is_same_v<F8, fp8_t> || std::is_same_v<F8, bf8_t>);
    using ArchType = gfx125_t;
    using Logical  = WmmaTraitsBase<gfx12_t, F8, F8, float, 32>;
    using Physical = WmmaTraits<gfx125_t, F8, F8, float, 16, 16, 64>;
    using AVecType = typename Logical::AVecType;
    using BVecType = typename Logical::BVecType;
    using CVecType = typename Logical::CVecType;

    static constexpr index_t logical_k  = 32;
    static constexpr index_t physical_k = 64;
    static_assert(Logical::kK == logical_k && Physical::kK == physical_k);
    static_assert(sizeof(AVecType) == 16 && sizeof(BVecType) == 16);
    static_assert(sizeof(typename Physical::AVecType) == 32);
    static_assert(std::is_same_v<CVecType, typename Physical::CVecType>);

    // Under the gfx12 default operand encoding, lane = (k_lane, m_or_n_lane),
    // k_lane in [0,2), m_or_n_lane in [0,16), and local byte i represents
    // k = 16 * (i / 8) + 8 * k_lane + (i % 8).
    // Copying i in [0,16) therefore fills precisely physical K in [0,32).
    // Both physical operands have exact +0 in K in [32,64).
    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a, const BVecType& b, const CVecType& c)
    {
        // FP8/BF8 vector aliases may use _BitInt storage, which is not
        // subscriptable on all supported Clang versions. Move their raw bits
        // through native integer vectors without converting any FP8 values.
        const auto a_bits = bit_cast<int32x4_t>(a);
        const auto b_bits = bit_cast<int32x4_t>(b);
        const int32x8_t padded_a{a_bits[0], a_bits[1], a_bits[2], a_bits[3], 0, 0, 0, 0};
        const int32x8_t padded_b{b_bits[0], b_bits[1], b_bits[2], b_bits[3], 0, 0, 0, 0};
        return Physical::template wmma_intrinsic<Params...>(
            bit_cast<typename Physical::AVecType>(padded_a),
            bit_cast<typename Physical::BVecType>(padded_b),
            c);
    }
};

template <typename F8>
using WarpGemmWmmaLogicalK32Impl = WarpGemmAttributeWmmaImpl<WarpGemmWmmaLogicalK32Traits<F8>>;

// This adapter exposes only the default encoding and non-transposed C.
// The attribute's public kK remains 32 and kKPerThread remains 16; the block
// pipeline therefore advances by the requested logical 32 elements.
template <typename F8>
using WarpGemmWmmaLogicalK32 = WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmWmmaLogicalK32Impl<F8>,
                                                                  false,
                                                                  WGAttrNumAccessEnum::Default,
                                                                  WGAttrNumAccessEnum::Default>>;

} // namespace ck_tile
