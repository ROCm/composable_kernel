// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

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
#define _TS_ std::to_string
        // clang-format off
        using _SS_ = std::string;

        return _SS_("tile_gemm_shape_") +
                _TS_(kM) + "x" + _TS_(kN) + "x" + _TS_(kK) + "x" + _TS_(NumWarps) + "_" +
                _TS_(BlockWarps::at(number<0>{})) + "x" + _TS_(BlockWarps::at(number<1>{})) + "x" + _TS_(BlockWarps::at(number<2>{})) + "_" +
                _TS_(WarpTile::at(number<0>{})) + "x" + _TS_(WarpTile::at(number<1>{})) + "x" + _TS_(WarpTile::at(number<2>{}));
#undef _TS_
        // clang-format on
    }
};

} // namespace ck_tile
