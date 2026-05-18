// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename BlockTile_,
          typename BlockWarps_,
          typename WarpTile_,
          bool PermuteA_ = false,
          bool PermuteB_ = false>
struct TileGemmShape
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;

    static constexpr index_t NumWarps =
        reduce_on_sequence(BlockWarps{}, multiplies<>{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(number<0>{});
    static constexpr index_t kN = BlockTile::at(number<1>{});
    static constexpr index_t kK = BlockTile::at(number<2>{});

    // default set cluster size = 1x1x1
    static constexpr index_t kclusterM = 1;
    static constexpr index_t kclusterN = 1;
    static constexpr index_t kclusterK = 1;

    static constexpr bool PermuteA = PermuteA_;
    static constexpr bool PermuteB = PermuteB_;
#if defined(__gfx125__)
    static constexpr index_t flatNIterPerWarp = WarpTile::at(number<1>{}) / 16;

    static constexpr index_t flatNPerWarp = BlockWarps::at(number<1>{}) * flatNIterPerWarp;
    static constexpr index_t flatKPerWarp =
        WarpTile::at(number<2>{}) * WarpTile::at(number<1>{}) / flatNIterPerWarp;
#else
    static constexpr index_t flatNPerWarp = BlockWarps::at(number<1>{});
    static constexpr index_t flatKPerWarp = WarpTile::at(number<2>{}) * WarpTile::at(number<1>{});
#endif
    static constexpr index_t flatKPerBlock = flatKPerWarp * kK / WarpTile::at(number<2>{});

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "tile_gemm_shape",
                      concat('x', kM, kN, kK, NumWarps),
                      concat('x', BlockWarps::at(number<0>{}), BlockWarps::at(number<1>{}), BlockWarps::at(number<2>{})),
                      concat('x', (WarpTile::at(number<0>{})), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})));
        // clang-format on
    }
};

template <typename ClusterTile_,
          typename BlockTile_,
          typename BlockWarps_,
          typename WarpTile_,
          bool PermuteA_ = false,
          bool PermuteB_ = false>
struct ClusterTileGemmShape
    : public TileGemmShape<BlockTile_, BlockWarps_, WarpTile_, PermuteA_, PermuteB_>
{
    using Base        = TileGemmShape<BlockTile_, BlockWarps_, WarpTile_, PermuteA_, PermuteB_>;
    using ClusterTile = remove_cvref_t<ClusterTile_>;

    static constexpr index_t kclusterM = ClusterTile::at(number<0>{});
    static constexpr index_t kclusterN = ClusterTile::at(number<1>{});
    static constexpr index_t kclusterK = ClusterTile::at(number<2>{});

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "cluster_tile_gemm_shape", 
                      concat('x', kclusterM, 'x', kclusterN, 'x', kclusterK),
                      concat('x', Base::kM, Base::kN, Base::kK, Base::NumWarps),
                      concat('x', Base::BlockWarps::at(number<0>{}), Base::BlockWarps::at(number<1>{}), Base::BlockWarps::at(number<2>{})),
                      concat('x', (Base::WarpTile::at(number<0>{})), Base::WarpTile::at(number<1>{}), Base::WarpTile::at(number<2>{})));
        // clang-format on
    }
};

template <typename T>
struct is_cluster_tile_gemm_shape : std::false_type
{
};

template <typename ClusterTile_,
          typename BlockTile_,
          typename BlockWarps_,
          typename WarpTile_,
          bool PermuteA_,
          bool PermuteB_>
struct is_cluster_tile_gemm_shape<
    ClusterTileGemmShape<ClusterTile_, BlockTile_, BlockWarps_, WarpTile_, PermuteA_, PermuteB_>>
    : std::true_type
{
};

template <typename PrecType, index_t M_Warp_Tile, bool IsFlatMM = false>
constexpr index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
    if constexpr(M_Warp_Tile == 32)
    {
        return 128;
    }
    else
    {
        constexpr bool is_8bit = sizeof(PrecType) == 1;
        return is_8bit ? 64 : 32;
    }
#else
    return 16;
#endif
#else
#if defined(CK_GFX950_SUPPORT)
    constexpr bool is_8bit_float =
        std::is_same_v<PrecType, fp8_t> || std::is_same_v<PrecType, bf8_t>;
    if constexpr(M_Warp_Tile == 32)
        return is_8bit_float ? 64 : 16;
    else
        return is_8bit_float ? 128 : 32;
#else
    if constexpr(M_Warp_Tile == 32)
        return (sizeof(PrecType) == 2 || IsFlatMM == false) ? 16 : 32;
    else
        return (sizeof(PrecType) == 2 || IsFlatMM == false) ? 32 : 64;
#endif
#endif
}

} // namespace ck_tile
