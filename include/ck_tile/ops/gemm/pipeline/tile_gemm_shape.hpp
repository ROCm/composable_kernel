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

    static constexpr bool PermuteA = PermuteA_;
    static constexpr bool PermuteB = PermuteB_;

    static constexpr index_t flatNPerWarp  = BlockWarps::at(number<1>{});
    static constexpr index_t flatKPerWarp  = WarpTile::at(number<2>{}) * WarpTile::at(number<1>{});
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

template <typename PrecType, index_t M_Warp_Tile, bool IsFlatMM = false>
constexpr index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
    return 16;
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

template <typename PrecType, index_t N_Warp_Tile>
constexpr index_t get_k_warp_tile_for_preshuffle_b()
{
#if CK_TILE_USE_WMMA
    return 16;
#else
    // When preshuffle B is enabled, the K_Warp_Tile must be sized appropriately
    // to support both dwordx4 loading instructions and MFMA instruction requirements.
    // A single dwordx4 load may feed one or more MFMA instructions, or conversely,
    // multiple loads may be required for a single MFMA instruction with a larger K dimension
    // (e.g., 16x16x128 on gfx950).

    // To achieve optimal memory bandwidth, each thread loads a minimum of 16 bytes (dwordx4)
    // from global memory.
    const index_t kMaxBytesPerLoad    = 16; // buffer load max 16 bytes
    const index_t kMaxElementsPerLoad = kMaxBytesPerLoad / sizeof(PrecType);
    const index_t kKLanePerWarp       = ck_tile::get_warp_size() / N_Warp_Tile;
    const index_t kKPerWarp           = kMaxElementsPerLoad * kKLanePerWarp;

    // Minimum K_Warp_Tile required by MFMA instructions
    const index_t kMfmaN16Index = 0;
    const index_t kMfmaN32Index = 1;
#if defined(CK_GFX950_SUPPORT)
    const index_t kF8MfmaMaxK[2]  = {128, 64};
    const index_t kF16MfmaMaxK[2] = {32, 16};
#else
    const index_t kF8MfmaMaxK[2]  = {32, 16};
    const index_t kF16MfmaMaxK[2] = {16, 8};
#endif
    const bool kIsF8         = std::is_same_v<PrecType, fp8_t> || std::is_same_v<PrecType, bf8_t>;
    const index_t kMfmaIndex = N_Warp_Tile == 16 ? kMfmaN16Index : kMfmaN32Index;
    const index_t kMfmaMaxK  = kIsF8 ? kF8MfmaMaxK[kMfmaIndex] : kF16MfmaMaxK[kMfmaIndex];

    return max(kKPerWarp, kMfmaMaxK);
#endif
}

} // namespace ck_tile
