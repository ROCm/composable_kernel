// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename AType_,
          typename BType_,
          typename CType_,
          typename BlockWarps_,
          typename WarpGemm_>
struct BlockGemmARegBRegCRegV3CustomPolicy
{
    using AType = ck_tile::remove_cvref_t<AType_>;
    using BType = ck_tile::remove_cvref_t<BType_>;
    using CType = ck_tile::remove_cvref_t<CType_>;

    using BlockWarps = ck_tile::remove_cvref_t<BlockWarps_>;

    static constexpr ck_tile::index_t kMWarps = BlockWarps::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kNWarps = BlockWarps::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t kKWarps = BlockWarps::at(ck_tile::number<2>{});

    using WarpGemm = ck_tile::remove_cvref_t<WarpGemm_>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(WarpGemm{}, kMWarps, kNWarps);
    }
};

} // namespace ck_tile
