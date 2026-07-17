// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

static CK_TILE_HOST_DEVICE constexpr index_t ceil_to_qualified_tile_length(index_t len)
{
    if(len == 96)
        return 128;
    if(len == 160)
        return 256;
    if(len == 192)
        return 192;

    // only length of 96, 160, 192 and power-of-two is supported
    if(!(len & (len - 1)))
        return len;

    return 0;
};

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_>
struct HstuAttentionFwdTileSettingClass
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static_assert(BlockTile::size() == 6, "Check failed!");
    static_assert(Gemm0BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm0WarpTile::size() == 3, "Check failed!");
    static_assert(Gemm1BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm1WarpTile::size() == 3, "Check failed!");

    static constexpr index_t NumGemm0Warps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies<>{}, number<1>{});
    static constexpr index_t NumGemm1Warps =
        reduce_on_sequence(Gemm1BlockWarps{}, multiplies<>{}, number<1>{});
    static_assert(NumGemm1Warps % NumGemm0Warps == 0);

    static constexpr index_t NumWarps = max(NumGemm0Warps, NumGemm1Warps);

    static constexpr index_t kM0    = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0    = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kN0Sub = BlockTile::at(number<2>{}); // tile size for dividing kN0
    static constexpr index_t kN1    = BlockTile::at(number<3>{}); // tile size along v head_dim
    static constexpr index_t kK1    = BlockTile::at(number<4>{}); // tile size along kv gemm unroll
    static constexpr index_t kQKHeaddim =
        BlockTile::at(number<5>{}); // total length of K0, used for pipeline that need load Q at
                                    // once (or repeately load Q as a whole tile)
    static_assert((kN0 % kN0Sub == 0) && (kN0 % kK1 == 0), "Check failed!");
};

template <index_t kM_,       // tile size in seqlen_q dimension
          index_t NumWarps_, // assume all warps are assigned to seqlen_q dimension
          index_t kOHeaddim_>
struct HstuAttentionFwdSplitKVCombineTileSettingClass
{
    static constexpr index_t kM       = kM_;
    static constexpr index_t NumWarps = NumWarps_;

    static_assert(kM % NumWarps == 0, "Check failed!");

    static constexpr index_t kOHeaddim = kOHeaddim_;

    static_assert((kM * kOHeaddim) % (NumWarps * get_warp_size()) == 0, "Check failed!");

    static constexpr index_t kSubOHeaddim = ceil_to_qualified_tile_length(kOHeaddim);
};

// Kernel1: computes S (Gemm0), dP (Gemm2), dQ (Gemm4)
// Gemm0 (S) and Gemm2 (dP) share the same warp configuration.
template <typename BlockTile_,            // sequence<kM0, kN0, kN0Sub, kQKHeaddim>
          typename Gemm0Gemm2BlockWarps_, // shared warp layout for Gemm0 (S) and Gemm2 (dP)
          typename Gemm0Gemm2WarpTile_,   // shared warp tile for Gemm0 (S) and Gemm2 (dP)
          typename Gemm4BlockWarps_,      // Gemm for computing dQ
          typename Gemm4WarpTile_>
struct HstuAttentionBwdTileSettingClassForKernel1
{
    using BlockTile            = remove_cvref_t<BlockTile_>;
    using Gemm0Gemm2BlockWarps = remove_cvref_t<Gemm0Gemm2BlockWarps_>;
    using Gemm0Gemm2WarpTile   = remove_cvref_t<Gemm0Gemm2WarpTile_>;
    using Gemm4BlockWarps      = remove_cvref_t<Gemm4BlockWarps_>;
    using Gemm4WarpTile        = remove_cvref_t<Gemm4WarpTile_>;

    // Aliases for individual Gemm access where needed
    using Gemm0BlockWarps = Gemm0Gemm2BlockWarps;
    using Gemm0WarpTile   = Gemm0Gemm2WarpTile;
    using Gemm2BlockWarps = Gemm0Gemm2BlockWarps;
    using Gemm2WarpTile   = Gemm0Gemm2WarpTile;

    static_assert(BlockTile::size() == 4, "Check failed!");
    static_assert(Gemm0Gemm2BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm0Gemm2WarpTile::size() == 3, "Check failed!");
    static_assert(Gemm4BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm4WarpTile::size() == 3, "Check failed!");

    static constexpr bool IsWarpGemm32 = (Gemm0Gemm2WarpTile::at(number<0>{}) == 32);

    static constexpr index_t NumGemm0Gemm2Warps =
        reduce_on_sequence(Gemm0Gemm2BlockWarps{}, multiplies<>{}, number<1>{});
    static constexpr index_t NumGemm0Warps = NumGemm0Gemm2Warps;
    static constexpr index_t NumGemm2Warps = NumGemm0Gemm2Warps;
    static constexpr index_t NumGemm4Warps =
        reduce_on_sequence(Gemm4BlockWarps{}, multiplies<>{}, number<1>{});

    static_assert(NumGemm4Warps == NumGemm0Gemm2Warps, "Check failed!");

    static constexpr index_t NumWarps = NumGemm0Gemm2Warps;

    static constexpr index_t kM0        = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0        = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kN0Sub     = BlockTile::at(number<2>{}); // tile size for dividing kN0
    static constexpr index_t kQKHeaddim = BlockTile::at(number<3>{}); // total length of QK head_dim
    static constexpr index_t kVHeaddim  = kQKHeaddim; // V shares head dim with K in HSTU
};

// Kernel2: computes dV (Gemm1), dK (Gemm3), S (Gemm0), dP (Gemm2)
// Gemm0 (S) and Gemm2 (dP) share the same warp configuration.
template <typename BlockTile_,            // sequence<kN0, kM0, kM0Sub, kM1, kK1, kQKHeaddim>
          typename Gemm0Gemm2BlockWarps_, // shared warp layout for Gemm0 (S) and Gemm2 (dP)
          typename Gemm0Gemm2WarpTile_,   // shared warp tile for Gemm0 (S) and Gemm2 (dP)
          typename Gemm1BlockWarps_,      // Gemm for computing dV
          typename Gemm1WarpTile_,
          typename Gemm3BlockWarps_, // Gemm for computing dK
          typename Gemm3WarpTile_>
struct HstuAttentionBwdTileSettingClassForKernel2
{
    using BlockTile            = remove_cvref_t<BlockTile_>;
    using Gemm0Gemm2BlockWarps = remove_cvref_t<Gemm0Gemm2BlockWarps_>;
    using Gemm0Gemm2WarpTile   = remove_cvref_t<Gemm0Gemm2WarpTile_>;
    using Gemm1BlockWarps      = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile        = remove_cvref_t<Gemm1WarpTile_>;
    using Gemm3BlockWarps      = remove_cvref_t<Gemm3BlockWarps_>;
    using Gemm3WarpTile        = remove_cvref_t<Gemm3WarpTile_>;

    // Aliases for individual Gemm access where needed
    using Gemm0BlockWarps = Gemm0Gemm2BlockWarps;
    using Gemm0WarpTile   = Gemm0Gemm2WarpTile;
    using Gemm2BlockWarps = Gemm0Gemm2BlockWarps;
    using Gemm2WarpTile   = Gemm0Gemm2WarpTile;

    static_assert(BlockTile::size() == 5, "Check failed!");
    static_assert(Gemm0Gemm2BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm0Gemm2WarpTile::size() == 3, "Check failed!");
    static_assert(Gemm1BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm1WarpTile::size() == 3, "Check failed!");
    static_assert(Gemm3BlockWarps::size() == 3, "Check failed!");
    static_assert(Gemm3WarpTile::size() == 3, "Check failed!");

    static constexpr bool IsWarpGemm32 = (Gemm0Gemm2WarpTile::at(number<0>{}) == 32);

    static constexpr index_t NumGemm0Gemm2Warps =
        reduce_on_sequence(Gemm0Gemm2BlockWarps{}, multiplies<>{}, number<1>{});
    static constexpr index_t NumGemm0Warps = NumGemm0Gemm2Warps;
    static constexpr index_t NumGemm2Warps = NumGemm0Gemm2Warps;
    static constexpr index_t NumGemm1Warps =
        reduce_on_sequence(Gemm1BlockWarps{}, multiplies<>{}, number<1>{});
    static constexpr index_t NumGemm3Warps =
        reduce_on_sequence(Gemm3BlockWarps{}, multiplies<>{}, number<1>{});

    static constexpr index_t NumWarps = max(NumGemm0Gemm2Warps, max(NumGemm1Warps, NumGemm3Warps));

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 =
        BlockTile::at(number<2>{}); // tile size for dividing kQKHeaddim in Gemm0 and Gemm2
    static constexpr index_t kK1 =
        BlockTile::at(number<3>{}); // tile size for dividing kM0 in Gemm1 and Gemm3
    static constexpr index_t kQKHeaddim = BlockTile::at(number<4>{}); // total length of QK head_dim
    static constexpr index_t kVHeaddim  = kQKHeaddim; // V shares head dim with K in HSTU
};

} // namespace ck_tile
