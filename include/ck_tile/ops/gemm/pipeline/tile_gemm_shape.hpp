// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename BlockTile_, typename BlockWarps_, typename WarpTile_>
struct TileGemmShape
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;

    static constexpr index_t NumWarps = reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(number<0>{});
    static constexpr index_t kN = BlockTile::at(number<1>{});
    static constexpr index_t kK = BlockTile::at(number<2>{});

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat("tile_gemm_shape_", kM, "x", kN, "x", kK, "x", NumWarps, "_",
                      BlockWarps::at(number<0>{}), "x", BlockWarps::at(number<1>{}), "x", BlockWarps::at(number<2>{}), "_",
                      (WarpTile::at(number<0>{})), "x", WarpTile::at(number<1>{}), "x", WarpTile::at(number<2>{}));
        // clang-format on
    }
};

} // namespace ck_tile
