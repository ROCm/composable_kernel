// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename AType_,
          typename BType_,
          typename CType_,
          typename BlockWarps_,
          typename WarpGemm_,
          index_t KSubTileNum_ = 1> // this variable is used for split K into multiple subtiles in
                                    // order to reduce register usage per wave>
struct BlockGemmARegBRegCRegV1CustomPolicy
{
    using AType = remove_cvref_t<AType_>;
    using BType = remove_cvref_t<BType_>;
    using CType = remove_cvref_t<CType_>;

    using BlockWarps = remove_cvref_t<BlockWarps_>;

    static constexpr index_t kMWarps = BlockWarps::at(number<0>{});
    static constexpr index_t kNWarps = BlockWarps::at(number<1>{});
    static constexpr index_t kKWarps = BlockWarps::at(number<2>{});

    using WarpGemm = remove_cvref_t<WarpGemm_>;

    static constexpr index_t KSubTileNum = KSubTileNum_;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{}, kMWarps, kNWarps);
    }
};

} // namespace ck_tile
